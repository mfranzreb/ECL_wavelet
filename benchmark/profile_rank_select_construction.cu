#include <cuda_profiler_api.h>

#include <algorithm>
#include <random>

#include "ecl_wavelet/bitarray/bit_array.cuh"
#include "ecl_wavelet/bitarray/rank_select.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

int main([[maybe_unused]] int argc, char** argv) {
  auto const size = std::stoul(argv[1]);
  auto const num_levels = std::stoul(argv[2]);
  auto const GPU_index = std::stoi(argv[3]);
  ecl::checkWarpSize(GPU_index);
  auto bit_array = ecl::createRandomBitArray(size, num_levels);

  cudaProfilerStart();
  ecl::RankSelect rs(std::move(bit_array), GPU_index);
  cudaProfilerStop();
  return 0;
}