#pragma once
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file,
                                          int line, bool abort = true) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) assert(0);
  }
}

/*!
 * \brief Get a launch configuration for a kernel given a number of warps, so
 * that threds per block are maximised.
 * \details If no combination is found that
 * perfectly matches the number of warps, the function will return the
 * combination that minimises the difference while not wasting any warps.
 * \param num_warps Number of warps to launch.
 * \param min_block_size Minimum number of threads per block.
 * \param max_block_size Maximum number of threads per block.
 * \return A pair of integers, the first
 * one being the number of blocks and the second one being the number of threads
 * per block.
 */

__host__ std::pair<int, int> inline getLaunchConfig(size_t num_warps,
                                                    int min_block_size,
                                                    int max_block_size) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int const warpSize = prop.warpSize;
  int min_block_size_warps = min_block_size / warpSize;
  int warps_per_sm = prop.maxThreadsPerMultiProcessor / warpSize;
  int warps_per_block = prop.maxThreadsPerBlock / warpSize;
  // find max block size that can still fully load an SM
  max_block_size = std::min(warps_per_block, max_block_size / 32);
  while (warps_per_sm % max_block_size != 0) {
    max_block_size -= 1;
  }
  if (num_warps <= max_block_size) {
    return {1, num_warps * warpSize};
  }
  std::pair<int, int> best_match = {-1, -1};
  int best_difference =
      std::numeric_limits<int>::max();  // Initialize with maximum value

  for (int k = 1; k <= max_block_size; ++k) {
    // Check if max_block_size is divisible by k
    if (max_block_size % k != 0) continue;

    // Calculate block_size and num_blocks
    int block_size = max_block_size / k;

    if (block_size < min_block_size_warps) {
      break;
    }
    int num_blocks = num_warps / block_size;

    // Check if this is a perfect match
    if (num_warps % block_size == 0) {
      return {num_blocks, block_size * warpSize};
    }

    // Otherwise, calculate the difference and update best match if needed
    int difference = num_warps - block_size * num_blocks;
    if (difference > 0 and difference < best_difference) {
      best_difference = difference;
      best_match = {num_blocks, block_size * warpSize};
    }
  }

  return best_match;  // Return the best match found
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

__host__ inline int getWarpSize() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.warpSize;
}

#define kernelCheck() kernelCheckFunc(__FILE__, __LINE__)
__host__ inline void kernelCheckFunc(const char *file, int line) {
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());
}