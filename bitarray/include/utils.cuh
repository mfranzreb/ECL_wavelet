#pragma once
#include <bit>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define WS 32

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

//? If no such thing as wasted warps, then better to overshoot than under?
__host__ std::pair<int, int> inline getLaunchConfig(size_t num_warps,
                                                    int min_block_size,
                                                    int max_block_size) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int min_block_size_warps = min_block_size / WS;
  int warps_per_sm = prop.maxThreadsPerMultiProcessor / WS;
  int warps_per_block = prop.maxThreadsPerBlock / WS;
  // find max block size that can still fully load an SM
  max_block_size = std::min(warps_per_block, max_block_size / 32);
  while (warps_per_sm % max_block_size != 0) {
    max_block_size -= 1;
  }
  if (num_warps <= max_block_size) {
    return {1, num_warps * WS};
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
      return {num_blocks, block_size * WS};
    }

    // Otherwise, calculate the difference and update best match if needed
    int difference = num_warps - block_size * num_blocks;
    if (difference > 0 and difference < best_difference) {
      best_difference = difference;
      best_match = {num_blocks, block_size * WS};
    }
  }

  return best_match;  // Return the best match found
}

__host__ inline int getMaxBlockSize() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.maxThreadsPerBlock;
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
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  return (n & (n - 1)) == 0;
}

// TODO test
/*!
 * \brief Find the previous power of two that is smaller or equal to n.
 * \tparam T Type of the number to find the previous power of two for. Must be
 * an unsigned integer.
 * \param n Number to find the previous power of two for.
 * \return Previous power of two that is smaller than n.
 */
template <typename T>
__device__ inline T getPrevPowTwo(T n) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  if (n == 0) {
    return 0;
  }
  if (isPowTwo(n)) {
    return n >> 1;
  }
  if constexpr (sizeof(T) == 8) {
    return (1ULL << (sizeof(T) * 8 - __clzll(n) - 1));
  } else if constexpr (sizeof(T) == 4) {
    return (1UL << (sizeof(T) * 8 - __clz(n) - 1));
  } else if constexpr (sizeof(T) == 2) {
    return (1U << (sizeof(T) * 8 - (__clz(n) - 16) - 1));
  } else if constexpr (sizeof(T) == 1) {
    return (1U << (sizeof(T) * 8 - (__clz(n) - 24) - 1));
  }
  return 0;
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
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  return 1 << n;
}

// TODO: Add at entrypoints of library
__host__ inline void checkWarpSize() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if (prop.warpSize != WS) {
    fprintf(stderr, "Warp size must be 32, but is %d\n", prop.warpSize);
    exit(EXIT_FAILURE);
  }
}

#define gpuErrchkInternal(ans, file, line) \
  { gpuAssert((ans), file, line); }

#define kernelCheck() kernelCheckFunc(__FILE__, __LINE__)
__host__ inline void kernelCheckFunc(const char *file, int line) {
  gpuErrchkInternal(cudaDeviceSynchronize(), file, line);
  gpuErrchkInternal(cudaPeekAtLastError(), file, line);
}

/*!
 * \brief Get i-th least significant bit of a character. Starting from 0.
 * \param i Index of the bit. LSB is 0.
 * \param c Character to get the bit from.
 * \return Value of the bit.
 */
template <typename T>
__host__ __device__ inline bool getBit(uint8_t const i, T const c) {
  assert(i < sizeof(T) * 8);
  return (c >> i) & 1;
}

template <typename T>
__device__ inline T ceilLog2(T n) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  if constexpr (sizeof(T) == 8) {
    auto highest_set_bit_pos = sizeof(T) * 8 - __clzll(n) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  } else if constexpr (sizeof(T) == 4) {
    auto highest_set_bit_pos = sizeof(T) * 8 - __clz(n) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  } else if constexpr (sizeof(T) == 2) {
    auto highest_set_bit_pos = sizeof(T) * 8 - (__clz(n) - 16) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  } else if constexpr (sizeof(T) == 1) {
    auto highest_set_bit_pos = sizeof(T) * 8 - (__clz(n) - 24) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  }
}

template <typename T>
__host__ inline T ceilLog2Host(T n) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  if constexpr (sizeof(T) == 8) {
    auto highest_set_bit_pos = sizeof(T) * 8 - std::__countl_zero(n) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  } else if constexpr (sizeof(T) == 4) {
    auto highest_set_bit_pos = sizeof(T) * 8 - std::__countl_zero(n) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  } else if constexpr (sizeof(T) == 2) {
    auto highest_set_bit_pos = sizeof(T) * 8 - std::__countl_zero(n) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  } else if constexpr (sizeof(T) == 1) {
    auto highest_set_bit_pos = sizeof(T) * 8 - std::__countl_zero(n) - 1;
    return isPowTwo<T>(n) ? highest_set_bit_pos : highest_set_bit_pos + 1;
  }
}