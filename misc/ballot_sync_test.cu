#include <cstdint>
#include <cstdio>

__device__ int global_var = 0;

template <typename T>
__device__ void utils::shareVar(bool condition, T &var, uint32_t const mask) {
  static_assert(std::is_integral<T>::value or std::is_floating_point<T>::value,
                "T must be an integral or floating-point type.");
  uint32_t src_thread = __ballot_sync(mask, condition);
  auto pos = atomicAdd(&global_var, 1);
  printf("Thread %d ends ballot in position %d\n", threadIdx.x, pos);
}

__global__ void test_kernel() {
  int tid = threadIdx.x;
  int num_ballots = 0;
  if (tid == 0) {
    // sleep for a bit
    for (int i = 0; i < 1000000; i++) {
      // Do nothing
    }
    utils::shareVar<int>(true, num_ballots, 1);
  }
  utils::shareVar<int>(tid == 0, num_ballots, ~0);
}

int main() {
  test_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}