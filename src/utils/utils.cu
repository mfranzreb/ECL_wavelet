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
#include <GPU_tunes.hpp>
#include <bit>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "ecl_wavelet/utils/utils.cuh"

namespace ecl::utils {
static cudaDeviceProp prop;
static IdealConfigs ideal_configs;

__host__ std::pair<int, int> getLaunchConfig(size_t const num_warps,
                                             int const min_block_size,
                                             int max_block_size) {
  assert(prop.totalGlobalMem != 0);
  assert(max_block_size >= min_block_size);
  int const min_block_size_warps = min_block_size / WS;
  int const warps_per_sm = prop.maxThreadsPerMultiProcessor / WS;
  int const warps_per_block = prop.maxThreadsPerBlock / WS;
  // find max block size that can still fully load an SM
  max_block_size = std::min(warps_per_block, max_block_size / WS);
  while (warps_per_sm % max_block_size != 0) {
    max_block_size -= 1;
  }
  if (num_warps <= static_cast<size_t>(max_block_size)) {
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
  assert(prop.totalGlobalMem != 0);
  return prop;
}

__host__ void checkWarpSize(uint32_t const GPU_index) {
  if (prop.totalGlobalMem == 0) {
    gpuErrchk(cudaSetDevice(GPU_index));
    cudaGetDeviceProperties(&prop, GPU_index);
    auto const threads_per_sm = prop.maxThreadsPerMultiProcessor;
    kMaxTPB = prop.maxThreadsPerBlock;
    // find max block size that can still fully load an SM
    while (threads_per_sm % kMaxTPB != 0) {
      kMaxTPB -= WS;
    }
    assert(kMaxTPB > WS);
    kMinBPM = threads_per_sm / kMaxTPB;
    auto const max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    kMinTPB = threads_per_sm / max_blocks_per_sm;
  }
  if (prop.warpSize != WS) {
    fprintf(stderr, "Warp size must be 32, but is %d\n", prop.warpSize);
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
  static bool first_call = true;
  if (first_call) {
    first_call = false;
    ideal_configs = get_configs(GPU_name);
  }
  return ideal_configs;
}

int64_t getAvailableMemoryLinux() {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;

  while (std::getline(meminfo, line)) {
    if (line.find("MemAvailable:") != std::string::npos) {
      std::istringstream iss(line);
      std::string label;
      int64_t value;
      std::string unit;

      iss >> label >> value >> unit;
      return value * 1024;  // Convert from KB to bytes
    }
  }

  return -1;  // Could not determine available memory
}
}  // namespace ecl::utils