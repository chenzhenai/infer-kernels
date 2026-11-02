// nvcc -O2 -lineinfo -o device_info device_info.cu
// 理论带宽 = 2 × 显存时钟 × 位宽 / 8。时钟来自 cudaDevAttrMemoryClockRate（kHz）。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

struct Spec {
  const char* key;
  double gbps;
};

// 产品规格书上的峰值 DRAM 带宽（GB/s）。匹配 cudaDeviceProp::name 的子串。
static const Spec kSpecs[] = {
    {"A100-SXM4-40GB", 1555}, {"A100-PCIE-40GB", 1555}, {"A100 40GB PCIe", 1555},
    {"A100-SXM4-80GB", 2039}, {"A100-PCIE-80GB", 1935}, {"A100 80GB PCIe", 1935},
    {"H100 80GB HBM3", 3350}, {"H100 NVL", 3900},       {"H100 PCIe", 2000},
    {"H200", 4800},           {"GB200", 8000},          {"B300", 8000},
    {"B200", 8000},
};

static void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

static const Spec* find_spec(const char* name) {
  for (const Spec& spec : kSpecs) {
    if (std::strstr(name, spec.key) != nullptr) {
      return &spec;
    }
  }
  return nullptr;
}

static int report_device(int dev) {
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

  int mem_clk_khz = 0;
  int bus_bits = 0;
  int smem_optin = 0;
  check(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev), "memory clock");
  check(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev), "bus width");
  check(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev),
        "shared memory opt-in");

  const double gbps = 2.0 * static_cast<double>(mem_clk_khz) * 1e3 * (bus_bits / 8.0) / 1e9;

  std::printf("GPU %s | CC %d.%d | SMs %d\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::printf("L2 %.1f MB | smem/block %zu KB | smem/SM %zu KB | smem/block(opt-in) %d KB\n",
              prop.l2CacheSize / 1048576.0, prop.sharedMemPerBlock / 1024,
              prop.sharedMemPerMultiprocessor / 1024, smem_optin / 1024);
  std::printf("regs/block %d | regs/SM %d | mem clock %d kHz | bus %d bit\n", prop.regsPerBlock,
              prop.regsPerMultiprocessor, mem_clk_khz, bus_bits);
  std::printf("Theoretical DRAM BW: %.1f GB/s\n", gbps);

  if (mem_clk_khz <= 0 || bus_bits <= 0) {
    std::fprintf(stderr, "device %d: memory clock or bus width is 0; cannot check datasheet\n",
                 dev);
    return 1;
  }

  const Spec* spec = find_spec(prop.name);
  if (spec == nullptr) {
    std::printf("datasheet: no entry for this device; 5%% check skipped\n");
    return 0;
  }

  const double rel = std::fabs(gbps - spec->gbps) / spec->gbps;
  std::printf("datasheet %.0f GB/s | error %.2f%%\n", spec->gbps, rel * 100.0);
  if (rel > 0.05) {
    std::fprintf(stderr, "device %d: theoretical bandwidth differs from datasheet by more than 5%%\n",
                 dev);
    return 1;
  }
  return 0;
}

int main() {
  int count = 0;
  check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (count <= 0) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }

  int rc = 0;
  for (int dev = 0; dev < count; ++dev) {
    rc |= report_device(dev);
  }
  return rc;
}
