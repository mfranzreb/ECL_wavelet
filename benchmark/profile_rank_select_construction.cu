#include <cuda_profiler_api.h>

#include <algorithm>
#include <bit_array.cuh>
#include <random>
#include <rank_select.cuh>

#include "test_benchmark_utils.cuh"

int main(int argc, char** argv) {
  // size is first command line argument
  auto const size = std::stoul(argv[1]);
  auto const num_levels = std::stoul(argv[2]);
  auto bit_array = ecl::createRandomBitArray(size, num_levels);

  cudaProfilerStart();
  ecl::RankSelect rs(std::move(bit_array));
  cudaProfilerStop();
  return 0;
}