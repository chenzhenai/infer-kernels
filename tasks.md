教程一共 **15 个阶段 + 1 个结业项目**（约 11–13 周，A100 可完成 0–7 和 9–14；阶段 8 需要 H100 / B200 / B300）。每个阶段的任务是往 `infer-kernels` 里加 kernel、测试和基准，版本都保留以便对比。

## 第一部分 · CUDA 基础与访存

**阶段 0 · 环境准备（1 天，tag `v0.0`）**
- [ ] `scripts/env_check.sh`：打印驱动、Toolkit、GPU、Compute Capability、PyTorch CUDA 版本
- [ ] `tools/device_info.cu`：SM 数、Shared Memory、L2、寄存器，并计算理论带宽
- [ ] `CMakeLists.txt`：按 GPU 选 `CMAKE_CUDA_ARCHITECTURES`（80 / 90a / 100a / 103a）
- [ ] `bench/util.py`：warmup + CUDA Event 计时、带宽 / TFLOPS、结果写 CSV
- 验收：理论带宽与 datasheet 差不超过 5%；`compute-sanitizer` 能报出越界

**阶段 1 · 向量加法（1 天）**
- [ ] `csrc/common.cuh`：`CUDA_CHECK`、`cdiv`
- [ ] `vec_add` 三版：一线程一元素 → grid-stride → `float4`（含尾部）
- [ ] CUDA Event 计时，有效带宽 = `3 × N × 4 B / 时间`
- [ ] 扫描 block 大小 64 / 128 / 256 / 512 / 1024
- 验收：与 CPU 逐元素一致；N = 2^26 时 `float4` 带宽 ≥ 理论值 80%；memcheck 通过

**阶段 2 · 矩阵转置与 Roofline（2 天）**
- [ ] `copy` kernel，作为同读写量的带宽上限
- [ ] `transpose` 三版：朴素 → Shared Memory 32×32 → padding（`tile[32][33]`）或 XOR swizzle
- [ ] `bench/roofline.py`：把各 kernel 标到本机 Roofline 上
- [ ] `__launch_bounds__` 与 occupancy 实验
- 验收：与 `torch.t().contiguous()` 一致（含 1000×3000）；padding/swizzle 带宽 ≥ copy 的 85%；bank conflict 接近 0

**阶段 3 · 归约与 Softmax（3 天）**
- [ ] 求和归约三版：Shared Memory 树形 → warp shuffle → 向量化 + grid-stride + 两趟
- [ ] 行 Softmax：一行一个 warp（行长 ≤ 1024）和一行一个 block（词表 32K–256K）
- [ ] Online softmax：读次数从 3 次减到 2 次
- [ ] 按行长自动选择调度的 dispatch
- 验收：相对 FP64 误差 < 1e-5，带宽 ≥ copy 的 85%；Softmax 对 `torch.softmax`，FP32 < 1e-6、BF16 < 1e-2；极大值 / 全 `-inf` / 长度 1 无 NaN

**阶段 4 · RMSNorm、融合与 PyTorch 绑定（3 天，tag `v0.1`）**
- [ ] `rmsnorm`：一行一个 block，BF16 输入、FP32 累加、16 字节向量化
- [ ] `fused_add_rmsnorm`：`residual += x; x = rmsnorm(residual)`，原地，语义与 vLLM 相同
- [ ] `silu_and_mul`：`[T, 2H]` → `silu(gate) * up`
- [ ] 可选 `layernorm`（Welford 一趟）
- [ ] `csrc/torch_bind.cpp` + `python/infer_kernels/ops.py`，pytest 多形状，对比 eager 与 `torch.compile`
- 验收：hidden ∈ {768, 2048, 4096, 5120, 8192} × tokens ∈ {1, 7, 128, 4096}，BF16 容差 2e-2；tokens=4096 时带宽 ≥ 峰值 70%，快于 eager 2 倍以上

## 第二部分 · 矩阵乘与 Tensor Core

**阶段 5 · SGEMM（4 天）**
- [ ] 按序保留：朴素 → 合并访存 → Shared Memory 分块 → 1D 寄存器分块 → 2D 寄存器分块（BM=BN=128, BK=8, TM=TN=8）→ 向量化 + A 转置存储 → warp 分块
- [ ] 与 `cublasSgemm`（或关闭 TF32 的 `torch.matmul`）对比 GFLOPS
- [ ] 形状扫描：方阵 512–8192，以及 M/N/K=4096、K=14336
- 验收：相对 FP64 < 1e-4（含不能整除 tile 的形状）；4096³ 达到 cuBLAS 的 70% 以上，且无 spill

**阶段 6 · Tensor Core：WMMA 与 mma.sync（3 天，A100 sm_80+）**
- [ ] `hgemm_wmma`：每 warp 16×16，BF16 输入、FP32 累加
- [ ] `hgemm_mma`：block tile 128×128×32，8 warp，warp tile 64×32；`ldmatrix` + `mma.sync.m16n8k16`
- [ ] Shared Memory XOR swizzle；epilogue 把 FP32 累加器转 BF16 并向量化写回
- [ ] 与 `torch.matmul` 对比 LLM 线性层形状
- 验收：相对误差 < 1e-2；4096³ 时 WMMA ≥ cuBLAS 30%，mma.sync 版 ≥ 50%

**阶段 7 · 异步拷贝流水线（4 天，tag `v0.2`）**
- [ ] `hgemm_pipeline`：`cp.async` + 3/4 stage + 寄存器级双缓冲
- [ ] 用 `cp.async` 的 src-size 给越界填 0
- [ ] Threadblock swizzle；M ≤ 64 时 Split-K
- [ ] autotune：在 (BM, BN, BK, STAGES, warp 数) 里按形状选最快并缓存
- 验收：覆盖 Llama-3-8B 形状和非对齐 M；A100 上 4096³ BF16 ≥ cuBLAS 80%（约 230 TFLOPS）

**阶段 8 · Hopper / Blackwell（6–7 天，需 H100 或 B200 / B300）**
- [ ] `tma_copy`：TMA + `cuda::barrier` 搬 2D tile，对比阶段 7 的 `cp.async`
- [ ] `cluster_reduce`：2–4 block 组成 cluster，分布式 Shared Memory 跨 block 归约
- [ ] 基于 CUTLASS 的 sm90 GEMM（TMA + WGMMA + warp specialization）和 sm100 GEMM（TMA + tcgen05 + TMEM）：TN 布局、BF16 输出、bias epilogue
- [ ] 阅读 `examples/cute/tutorial/`，为每一步写数据流注释
- [ ] 可选：用 CuTe DSL 重写同一 GEMM
- 验收：`tma_copy` 单线程发起、带宽 ≥ DRAM 峰值 85%；8192³ BF16 ≥ cuBLAS 90%

## 第三部分 · 推理核心算子

**阶段 9 · Decode GEMV 与量化（4 天，tag `v0.3`）**
- [ ] `gemv_bf16`：每 warp 一行，16 字节读 + warp 归约
- [ ] `gemv_w4a16`：INT4 + group size 128 的 FP16 scale；Python 离线量化打包
- [ ] 小 batch（M ≤ 16）W4A16 GEMM：寄存器内反量化后送 `mma.sync`（Marlin）
- [ ] 动态 per-token FP8 激活量化（可融合进 RMSNorm）+ cuBLASLt / CUTLASS FP8 GEMM（H100+）
- [ ] B200 / B300 上跑通 NVFP4 GEMM，对比 FP8、BF16
- 验收：4096×14336 时 `gemv_bf16` 带宽 ≥ 峰值 80%；W4A16 相对反量化参考 < 1e-3，且快于 BF16 GEMV 2.5 倍以上

**阶段 10 · FlashAttention Prefill（6–7 天，tag `v0.4`）**
- [ ] `attn_naive`：QKᵀ → softmax → PV，作为正确性和性能基线
- [ ] `flash_attn_fwd`：BF16，head_dim 64/128，Br=Bc=64，4 warp（每 warp 16 行 Q）
- [ ] K/V 用 `cp.async` 双缓冲；P 不经 Shared Memory，直接从累加器转成第二个 MMA 的 A
- [ ] 支持 causal / non-causal、GQA、任意序列长度；输出 LSE
- [ ] 可选：varlen（`cu_seqlens`）
- 验收：对 SDPA math 后端，BF16 最大绝对误差 < 2e-2；A100、seqlen 4096、head_dim 128 达到 FA2 官方 TFLOPS 的 50% 以上；64K 显存随长度线性增长

**阶段 11 · Decode Attention 与 Paged KV（4 天，tag `v0.5`）**
- [ ] `rope_and_cache`：Q/K 做 RoPE（neox 与 GPT-J），按 `slot_mapping` 把 K、V 写入分页 Cache
- [ ] `paged_decode_v1`：每个 block 一个 (序列, Q 头)
- [ ] `paged_decode_v2`：每个 block 一个 (序列, KV 头)，组内所有 Q 头共享一次 KV 读取，16 字节向量化
- [ ] `paged_decode_split` + `merge_attn_states`：Split-KV，用 LSE 合并
- [ ] 可选：FP8 E4M3 KV Cache
- 验收：GQA 比例 {1, 4, 8}、打乱的 block table，误差 < 1e-2；batch 64、上下文 4K 时 v2 读带宽 ≥ 峰值 70%；batch 1、上下文 32K 时 Split-KV 快 3 倍以上

**阶段 12 · MoE（4 天，tag `v0.6`）**
- [ ] `topk_softmax`：每 warp 一个 token，softmax + Top-K + 可选重新归一化
- [ ] `moe_align`：统计专家 token 数、前缀和，生成按专家排序并填充到 BLOCK_M 倍数的索引
- [ ] `moe_grouped_gemm`：v1 逐专家调阶段 7 GEMM；v2 单 kernel，tile 查表找专家
- [ ] gate/up GEMM 的 epilogue 融合 SwiGLU；`moe_unpermute` 按 Top-K 权重还原
- 验收：索引与 `torch.topk(softmax)` 完全一致；整层对 PyTorch 参考，BF16 误差 < 2e-2；decode batch 16–64 时 v2 比 v1 快 1.5 倍以上；时间线里没有 D2H 同步

## 第四部分 · 系统级与结业

**阶段 13 · CUDA Graph、Stream 与 PDL（3 天，tag `v0.7`）**
- [ ] `graphs/toy_decoder.cu`：N 层 decoder 一步 decode，eager 与 Graph 各跑一遍
- [ ] batch 分桶 {1, 2, 4, 8, 16, 32, 64, 128}，每桶一个 Graph
- [ ] PyTorch `torch.cuda.CUDAGraph` 捕获同一流程，与 C++ 对比
- [ ] H100 / B200 / B300 上给 RMSNorm → GEMM 启用 PDL
- [ ] 可选：两个 stream 并行（MoE 共享专家与路由专家）
- 验收：Graph 与 eager 逐位一致；batch 1 时 kernel 间隙基本消失；CPU 提交时间变为常数

**阶段 14 · 接入 PyTorch 与端到端（5 天，tag `v0.9`）**
- [ ] 全部算子注册到 `ik::`，每个都有 fake 实现，通过 `torch.library.opcheck`
- [ ] 在 Qwen2.5-0.5B / Llama-3.2-1B（或小 Qwen3-MoE）上替换 RMSNorm、SwiGLU、RoPE+KV 写入、Prefill/Decode Attention、线性层（BF16 或 W4A16）
- [ ] 逐层对比工具：forward hook 记录每层最大误差
- [ ] 生成循环：Prefill 用阶段 10，Decode 用阶段 11 + 阶段 13 的 CUDA Graph
- 验收：`torch.compile` 可跑；最后一层 logits top-1 一致率 ≥ 99%；WikiText-2 困惑度与 HF 差不超过 1%；batch 1 的 TPOT 快于 HF eager 2 倍以上

**结业 · mini-infer（1–2 周，tag `v1.0`）**
- [ ] Paged KV 的块分配与回收 + 简单 continuous batching
- [ ] Prefill：FlashAttention + 融合算子；Decode：Paged Decode + 按 batch 分桶的 CUDA Graph
- [ ] 至少一条量化路径：W4A16（A100），或 FP8 / NVFP4（H100 / B200 / B300）
- [ ] 基准：给定请求长度分布，输出 TTFT、TPOT、吞吐量的 P50 / P99
- [ ] 选做：MoE、FP8 KV Cache、PDL、与 vLLM / SGLang 对比
- 验收：困惑度与 HF 差不超过 1%；batch 1 的 TPOT 达到「权重字节数 / 显存带宽」理论下限的 70% 以上；高并发吞吐比 HF Transformers 高一个数量级；报告含每个算子的 Roofline 和至少一张 Nsight Systems 时间线

版本对应关系：`v0.0` 环境 → `v0.1` 访存算子 → `v0.2` A100 BF16 GEMM → `v0.3` 量化 → `v0.4–v0.6` Attention / MoE → `v0.7` CUDA Graph → `v0.9` 模型替换 → `v1.0` 单卡引擎。仓库里已有的参考实现覆盖阶段 4、11 和一条最小推理链路，后面各阶段可以在那上面替换。