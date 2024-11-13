#pragma once
#include <cstdio>
#include <cstdlib>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file,
                                          int line, bool abort = true) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) assert(0);
  }
}