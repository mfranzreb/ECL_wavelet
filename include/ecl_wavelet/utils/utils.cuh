/*
BSD 3-Clause License

Copyright (c) 2025, Marco Franzreb, Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <bits/stdc++.h>

#include <bit>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

#define WS 32

// Define launch bounds based on compute capability
// __CUDA_ARCH__ is not defined in host code
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ > 800 && __CUDA_ARCH__ < 900
#define MAX_TPB 768
#define MIN_BPM 2
#elif __CUDA_ARCH__ == 750
#define MAX_TPB 1024
#define MIN_BPM 1
#else
#define MAX_TPB 1024
#define MIN_BPM 2
#endif

// Minimal block size that still fully loads an SM
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870
#define MIN_TPB 96
#else
#define MIN_TPB 64
#endif

#define LB(x, y) __launch_bounds__(x, y)
#else
#define LB(x, y)
#endif

namespace ecl {
template <typename T>
struct RankSelectQuery {
  size_t index_;
  T symbol_;
};
namespace utils {
inline uint32_t kMaxTPB = 0;
inline uint32_t kMinTPB = 0;
inline uint32_t kMinBPM = 0;
constexpr uint8_t kBankSizeBytes = 4;
constexpr uint8_t kBanksPerLine = 32;
static constexpr uint8_t kMinAlphabetSize = 2;

__host__ __device__ inline void gpuAssert(cudaError_t code, const char *file,
                                          int line, bool abort = true) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) {
#ifdef __CUDA_ARCH__
      asm("trap;");
#else
      exit(EXIT_FAILURE);
#endif
    }
  }
}
/*!
 * \brief Get a launch configuration for a kernel given a number of warps that
 * matches the given number of warps as closely as possible while ensuring that
 * the block size will fully occupy an SM.
 * \details If no appropriate match is found, it returns -1 for both.
 * \param num_warps Number of warps to launch.
 * \param min_block_size Minimum number of threads per block.
 * \param max_block_size Maximum number of threads per block.
 * \return A pair of integers, the first one being the number of blocks and the
 * second one being the number of threads per block.
 */
__host__ std::pair<int, int> getLaunchConfig(size_t const num_warps,
                                             int const min_block_size,
                                             int max_block_size);

/*!
 * \brief Get a reference to the device properties of the current GPU.
 */
__host__ cudaDeviceProp &getDeviceProperties();

/*!
 * \brief Check if a number is a power of two.
 * \tparam T Type of the number to check. Must be an
 * unsigned integer.
 * \param n Number to check.
 * \return True if n is a power of two, false otherwise.
 */
template <typename T>
__host__ __device__ bool isPowTwo(T const n) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  return (n & (n - 1)) == 0;
}

/*!
 * \brief Find the previous power of two that is smaller or equal to n.
 * \tparam T Type of the number to find the previous power of two for. Must be
 * an unsigned integer.
 * \param n Number to find the previous power of two for.
 * \return Previous power of two that is smaller than n.
 */
template <typename T>
__device__ T getPrevPowTwo(T n) {
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
__host__ __device__ T powTwo(T n) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  return 1 << n;
}

/*!
 * \brief Check if the warp size is 32, and initialize the device dependent
 * variables.
 * \param GPU_index Index of the GPU to check.
 */
__host__ void checkWarpSize(uint32_t const GPU_index);

#define gpuErrchkInternal(ans, file, line) \
  {                                        \
    gpuAssert((ans), file, line);          \
  }
__host__ inline void kernelCheckFunc(const char *file, int line) {
  gpuErrchkInternal(cudaGetLastError(), file, line);
  gpuErrchkInternal(cudaDeviceSynchronize(), file, line);
}

__host__ inline void kernelStreamCheckFunc(cudaStream_t stream,
                                           const char *file, int line) {
  gpuErrchkInternal(cudaGetLastError(), file, line);
  gpuErrchkInternal(cudaStreamSynchronize(stream), file, line);
}

/*!
 * \brief Get i-th least significant bit of a character. Starting from 0.
 * \param i Index of the bit. LSB is 0.
 * \param c Character to get the bit from.
 * \return Value of the bit.
 */
template <typename T>
__host__ __device__ bool getBit(uint8_t const i, T const c) {
  assert(i < sizeof(T) * 8);
  return (c >> i) & 1;
}

template <typename T>
__host__ __device__ uint8_t ceilLog2(T n) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
#ifdef __CUDA_ARCH__
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
#else
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
#endif
}

/*!
 * \brief Find the largest divisor of \c n that is smaller than \c divisor.
 * \tparam T Type of the number to find the divisor for.
 * \param n Number to find the divisor for.
 * \param divisor Divisor to start from.
 * \return Largest divisor of \c n that is smaller than \c divisor.
 */
template <typename T>
__host__ T findLargestDivisor(T const n, T const divisor) {
  if (divisor == 0) return 1;
  return divisor - n % divisor;
}

/*!
 * \brief Helper function to share a variable between certain threads in a warp.
 * \tparam T Type of the variable to be shared. Must be an integral or
 * floating point type.
 * \param condition Condition to be met for being the source of the share. Only
 * one thread should fulfill it.
 * \param var Variable to be shared.
 * \param mask Mask representing the threads that should participate.
 */
template <typename T>
__device__ void shareVar(bool condition, T &var, uint32_t const mask) {
  static_assert(std::is_integral<T>::value or std::is_floating_point<T>::value,
                "T must be an integral or floating-point type.");
  __syncwarp(mask);  // Crazy behaviour on Volta if removed
  uint32_t src_thread = __ballot_sync(mask, condition);
  if (src_thread != 0) {
    // Get the value from the first thread that fulfills the condition
    src_thread = __ffs(src_thread) - 1;
    var = __shfl_sync(mask, var, src_thread);
  }
}

/*!
 * \brief Get the available RAM memory on a Linux system.
 */
int64_t getAvailableMemoryLinux();

}  // namespace utils
}  // namespace ecl

#define gpuErrchk(ans)                                \
  {                                                   \
    ecl::utils::gpuAssert((ans), __FILE__, __LINE__); \
  }

#define kernelStreamCheck(stream) \
  ecl::utils::kernelStreamCheckFunc(stream, __FILE__, __LINE__)

#define kernelCheck() ecl::utils::kernelCheckFunc(__FILE__, __LINE__)
