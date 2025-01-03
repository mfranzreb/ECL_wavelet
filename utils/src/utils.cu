#include <GPU_tunes.hpp>
#include <bit>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "utils.cuh"

namespace ecl {
namespace internal {
static cudaDeviceProp prop;
}  // namespace internal

__host__ std::pair<int, int> getLaunchConfig(size_t const num_warps,
                                             int const min_block_size,
                                             int max_block_size) {
  assert(internal::prop.totalGlobalMem != 0);
  int const min_block_size_warps = min_block_size / WS;
  int const warps_per_sm = internal::prop.maxThreadsPerMultiProcessor / WS;
  int const warps_per_block = internal::prop.maxThreadsPerBlock / WS;
  // find max block size that can still fully load an SM
  max_block_size = std::min(warps_per_block, max_block_size / WS);
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
    int num_blocks_high = (num_warps + block_size - 1) / block_size;
    int num_blocks_low = num_warps / block_size;

    // Check if this is a perfect match
    if (num_warps % block_size == 0) {
      return {num_blocks_low, block_size * WS};
    }

    // Otherwise, calculate the difference and update best match if needed
    int difference = block_size * num_blocks_high - num_warps;
    if (difference < best_difference) {
      best_difference = difference;
      best_match = {num_blocks_high, block_size * WS};
    }
    difference = num_warps - block_size * num_blocks_low;
    if (difference < best_difference) {
      best_difference = difference;
      best_match = {num_blocks_low, block_size * WS};
    }
  }

  return best_match;  // Return the best match found
}

__host__ cudaDeviceProp &getDeviceProperties() {
  assert(internal::prop.totalGlobalMem != 0);
  return internal::prop;
}

__host__ void checkWarpSize(uint8_t const GPU_index) {
  if (internal::prop.totalGlobalMem == 0) {
    cudaGetDeviceProperties(&internal::prop, GPU_index);
    auto const threads_per_sm = internal::prop.maxThreadsPerMultiProcessor;
    kMaxTPB = internal::prop.maxThreadsPerBlock;
    // find max block size that can still fully load an SM
    while (threads_per_sm % kMaxTPB != 0) {
      kMaxTPB -= WS;
    }
    assert(kMaxTPB > WS);
    kMinBPM = threads_per_sm / kMaxTPB;
    auto const max_blocks_per_sm = internal::prop.maxBlocksPerMultiProcessor;
    kMinTPB = threads_per_sm / max_blocks_per_sm;
  }
  if (internal::prop.warpSize != WS) {
    fprintf(stderr, "Warp size must be 32, but is %d\n",
            internal::prop.warpSize);
    exit(EXIT_FAILURE);
  }
}

__host__ IdealConfigs &getIdealConfigs(const std::string &GPU_name) {
  auto get_configs = [](std::string GPU_name) {
    if (configs.find(GPU_name) != configs.end()) {
      return configs[GPU_name];
    } else {
      return IdealConfigs();
    }
  };
  static IdealConfigs ideal_configs = get_configs(GPU_name);
  return ideal_configs;
}
}  // namespace ecl