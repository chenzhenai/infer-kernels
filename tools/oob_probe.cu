// 故意越界一格，供 compute-sanitizer --tool memcheck 报错。不要把它当正常程序跑。
#include <cstdio>

#include <cuda_runtime.h>

__global__ void oob_write(int* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) {
    data[i] = i;
  }
}

int main() {
  constexpr int n = 128;
  int* data = nullptr;
  cudaError_t err = cudaMalloc(&data, n * sizeof(int));
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err));
    return 1;
  }

  std::printf("intentional OOB write at index %d (allocation length %d)\n", n, n);
  oob_write<<<1, n>>>(data, n);
  cudaDeviceSynchronize();
  cudaFree(data);
  return 0;
}
