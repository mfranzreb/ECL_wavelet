#include <cuda_profiler_api.h>

#include <random>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"

template <typename T>
void profileSelect(size_t const data_size, size_t const alphabet_size,
                   size_t const num_queries) {
  std::vector<T> alphabet;
  alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(),
            0);  // Min alphabet so that queries dont change

  auto [data, hist] = ecl::generateRandomDataAndHist<T>(alphabet, data_size);
  auto queries =
      ecl::generateRandomSelectQueries<T>(hist, num_queries, alphabet);
  ecl::WaveletTree<T> wt(data.data(), data_size, std::move(alphabet), 0);

  cudaProfilerStart();
  auto results = wt.select(queries);
  cudaProfilerStop();
}

int main(int argc, char** argv) {
  // size is first command line argument
  auto const data_size = std::stoul(argv[1]);
  auto const alphabet_size = std::stoul(argv[2]);
  auto const num_queries = std::stoul(argv[3]);

  if (alphabet_size < std::numeric_limits<uint8_t>::max()) {
    profileSelect<uint8_t>(data_size, alphabet_size, num_queries);
  } else if (alphabet_size < std::numeric_limits<uint16_t>::max()) {
    profileSelect<uint16_t>(data_size, alphabet_size, num_queries);
  }
}