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

__host__ std::pair<int, int> inline getLaunchConfig(size_t num_warps) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;
  int warps_per_block = prop.maxThreadsPerBlock / 32;
  // find max block size that can still fully load an SM
  int max_block_size = warps_per_block;
  while (warps_per_sm % max_block_size != 0) {
    max_block_size -= 1;
  }
  if (num_warps <= max_block_size) {
    return {1, num_warps * 32};
  }
  //? Overthinking it?
  return {1, max_block_size * 32};
}

/*!
 * \brief Find the previous power of two that is smaller or equal to n.
 * \tparam T Type of the number to find the previous power of two for. Must be
 * an unsigned integer.
 * \param n Number to find the previous power of two for.
 * \return Previous power of two that is smaller or equal to n.
 */
template <typename T>
__host__ __device__ inline T getPrevPowTwo(T n) {
  static_assert(std::is_integral<T>::value or std::is_signed<T>::value,
                "T must be an unsigned integral type.");
  if (n == 0) {
    return 0;
  }
  if constexpr (sizeof(T) == 8) {
    return (1ULL << (sizeof(T) - __clzll(n) - 1));
  } else {
    return (1UL << (sizeof(T) - __clz(n) - 1));
  }
}

/*!
 * \brief Check if a number is a power of two.
 * \tparam T Type of the number to check. Must be an
 * unsigned integer.
 * \param n Number to check.
 * \return True if n is a power of two, false otherwise.
 */
template <typename T>
__host__ __device__ inline bool isPowTwo(T const n) {
  static_assert(std::is_integral<T>::value or std::is_signed<T>::value,
                "T must be an unsigned integral type.");
  return (n & (n - 1)) == 0;
}

/*!
 * \brief Calculate 2^n.
 * \tparam T Type of the number to calculate the power of two for. Must be an
 * unsigned integer.
 * \param n Exponent of the power of two.
 * \return 2^n.
 */
template <typename T>
__host__ __device__ inline T powTwo(T n) {
  static_assert(std::is_integral<T>::value or std::is_signed<T>::value,
                "T must be an unsigned integral type.");
  return 2 << n;
}