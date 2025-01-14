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
#define MAX_TPB 512
#define MIN_BPM 3
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
inline uint32_t kMaxTPB = 0;
inline uint32_t kMinTPB = 0;
inline uint32_t kMinBPM = 0;
constexpr uint8_t kBankSizeBytes = 4;
constexpr uint8_t kBanksPerLine = 32;

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
__host__ inline void gpuAssert(cudaError_t code, const char *file, int line,
                               bool abort = true) {
  if (code != cudaSuccess) {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(EXIT_FAILURE);
  }
}

/*!
 * \brief Get a launch configuration for a kernel given a number of warps, so
 * that threads per block are maximised.
 * \details If no combination is found that
 * perfectly matches the number of warps, the function will return the
 * combination that minimises the difference.
 * \param num_warps Number of warps to launch.
 * \param min_block_size Minimum number of threads per block. If num_warps is
 * smaller, the function will return a block size smaller than this.
 * \param max_block_size Maximum number of threads per block.
 * \return A pair of integers, the first one being the number of blocks and the
 * second one being the number of threads per block.
 */
// TODO maybe measure effect of wasted warps
__host__ std::pair<int, int> getLaunchConfig(size_t const num_warps,
                                             int const min_block_size,
                                             int max_block_size);

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

/*! @copydoc getPrevPowTwo */
template <typename T>
__host__ T getPrevPowTwoHost(T n) {
  static_assert(std::is_integral<T>::value and std::is_unsigned<T>::value,
                "T must be an unsigned integral type.");
  if (n == 0) {
    return 0;
  }
  if (isPowTwo(n)) {
    return n >> 1;
  }
  if constexpr (sizeof(T) == 8) {
    return (1ULL << (sizeof(T) * 8 - __builtin_clzll(n) - 1));
  } else if constexpr (sizeof(T) == 4) {
    return (1UL << (sizeof(T) * 8 - __builtin_clz(n) - 1));
  } else if constexpr (sizeof(T) == 2) {
    return (1U << (sizeof(T) * 8 - (__builtin_clz(n) - 16) - 1));
  } else if constexpr (sizeof(T) == 1) {
    return (1U << (sizeof(T) * 8 - (__builtin_clz(n) - 24) - 1));
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

__host__ void checkWarpSize(uint8_t const GPU_index);

#define gpuErrchkInternal(ans, file, line) \
  { gpuAssert((ans), file, line); }

#define kernelCheck() kernelCheckFunc(__FILE__, __LINE__)
__host__ inline void kernelCheckFunc(const char *file, int line) {
  gpuErrchkInternal(cudaDeviceSynchronize(), file, line);
  gpuErrchkInternal(cudaPeekAtLastError(), file, line);
}

#define kernelStreamCheck(stream) \
  kernelStreamCheckFunc(stream, __FILE__, __LINE__)
__host__ inline void kernelStreamCheckFunc(cudaStream_t stream,
                                           const char *file, int line) {
  gpuErrchkInternal(cudaStreamSynchronize(stream), file, line);
  gpuErrchkInternal(cudaPeekAtLastError(), file, line);
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
__device__ T ceilLog2(T n) {
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
__host__ T ceilLog2Host(T n) {
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

struct IdealConfigs {
  uint32_t ideal_TPB_accessKernel = 0;
  size_t ideal_tot_threads_accessKernel = 0;

  uint32_t ideal_TPB_calculateL2EntriesKernel = 0;

  uint32_t ideal_TPB_computeGlobalHistogramKernel = 0;
  size_t ideal_tot_threads_computeGlobalHistogramKernel = 0;

  uint32_t ideal_TPB_fillLevelKernel = 0;
  size_t ideal_tot_threads_fillLevelKernel = 0;

  uint32_t ideal_TPB_rankKernel = 0;
  size_t ideal_tot_threads_rankKernel = 0;
};

__host__ IdealConfigs &getIdealConfigs(const std::string &GPU_name);
}  // namespace ecl