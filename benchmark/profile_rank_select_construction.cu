#include <cuda_profiler_api.h>

#include <algorithm>
#include <bit_array.cuh>
#include <random>
#include <rank_select.cuh>
#include <utils.cuh>

#include "test_benchmark_utils.cuh"

int main(int argc, char** argv) {
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