#include <cuda_profiler_api.h>

#include <random>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"
#include "utils.cuh"

template <typename T>
void profileAccess(size_t const data_size, size_t const alphabet_size,
                   size_t const num_queries, bool const use_profiler_api) {
  std::vector<T> alphabet;
  std::vector<T> data;
  alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  data = ecl::generateRandomData<T>(alphabet, data_size);
  auto queries = ecl::generateRandomAccessQueries(data_size, num_queries);
  ecl::WaveletTree<T> wt(data.data(), data_size, std::move(alphabet), 0);
  auto results = wt.access(queries.data(), num_queries);
  if (use_profiler_api) {
    cudaProfilerStart();
    results = wt.access(queries.data(), num_queries);
    cudaProfilerStop();
  }
}

int main(int argc, char** argv) {
  // size is first command line argument
  auto const data_size = std::stoul(argv[1]);
  auto const alphabet_size = std::stoul(argv[2]);
  auto const num_queries = std::stoul(argv[3]);
  bool const use_profiler_api = argc > 4 ? std::stoi(argv[4]) : false;

  auto queries = ecl::generateRandomAccessQueries(data_size, num_queries);

  if (alphabet_size < std::numeric_limits<uint8_t>::max()) {
    profileAccess<uint8_t>(data_size, alphabet_size, num_queries,
                           use_profiler_api);
  } else if (alphabet_size < std::numeric_limits<uint16_t>::max()) {
    profileAccess<uint16_t>(data_size, alphabet_size, num_queries,
                            use_profiler_api);
  }
  return 0;
}