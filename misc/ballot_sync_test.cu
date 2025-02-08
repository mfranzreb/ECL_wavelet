#include <cstdint>
#include <cstdio>

template <typename T>
__device__ void shareVar(bool condition, T &var, uint32_t const mask) {
  static_assert(std::is_integral<T>::value or std::is_floating_point<T>::value,
                "T must be an integral or floating-point type.");
  printf("Thread %d starts ballot %d\n", threadIdx.x, var);
  uint32_t src_thread = __ballot_sync(mask, condition);
  printf("Thread %d ends ballot %d\n", threadIdx.x, var);
  // Get the value from the first thread that fulfills the condition
  src_thread = __ffs(src_thread) - 1;
  var = __shfl_sync(mask, var, src_thread);
}

__global__ void test_kernel() {
  int tid = threadIdx.x;
  int num_ballots = 0;
  if (tid == 0) {
    printf("First thread\n");
    // sleep for a bit
    for (int i = 0; i < 1000000; i++) {
      // Do nothing
    }
    shareVar<int>(true, num_ballots, 1);
    num_ballots++;
  }
  shareVar<int>(tid == 0, num_ballots, ~0);
}

int main() {
  test_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();
  return 0;
}