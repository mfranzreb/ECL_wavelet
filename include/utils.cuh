#pragma once
#include <cstdio>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
__host__ __device__ void gpuAssert(cudaError_t code, const char *file, int line,
                                   bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}