#include <cuda_profiler_api.h>

#include <random>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"

int main(int argc, char** argv) {
  // size is first command line argument
  auto const data_size = std::stoul(argv[1]);
  auto const alphabet_size = std::stoul(argv[2]);
  auto const num_queries = std::stoul(argv[3]);

  auto queries = ecl::generateRandomQueries(data_size, num_queries);

  if (alphabet_size < std::numeric_limits<uint8_t>::max()) {
    std::vector<uint8_t> alphabet;
    std::vector<uint8_t> data;
    alphabet = std::vector<uint8_t>(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    data = ecl::generateRandomData<uint8_t>(alphabet, data_size);
    ecl::WaveletTree<uint8_t> wt(data.data(), data_size, std::move(alphabet),
                                 0);
    cudaProfilerStart();
    auto results = wt.template access<1>(queries.data(), num_queries);
    cudaProfilerStop();
    results = wt.template access<2>(queries.data(), num_queries);
    results = wt.template access<4>(queries.data(), num_queries);
    results = wt.template access<8>(queries.data(), num_queries);
    results = wt.template access<16>(queries.data(), num_queries);
    results = wt.template access<32>(queries.data(), num_queries);
  } else if (alphabet_size < std::numeric_limits<uint16_t>::max()) {
    std::vector<uint16_t> alphabet;
    std::vector<uint16_t> data;
    alphabet = std::vector<uint16_t>(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    data = ecl::generateRandomData<uint16_t>(alphabet, data_size);
    ecl::WaveletTree<uint16_t> wt(data.data(), data_size, std::move(alphabet),
                                  0);
    cudaProfilerStart();
    auto results = wt.template access<1>(queries.data(), num_queries);
    cudaProfilerStop();
    results = wt.template access<2>(queries.data(), num_queries);
    results = wt.template access<4>(queries.data(), num_queries);
    results = wt.template access<8>(queries.data(), num_queries);
    results = wt.template access<16>(queries.data(), num_queries);
    results = wt.template access<32>(queries.data(), num_queries);
  } else {
    std::vector<uint32_t> alphabet;
    std::vector<uint32_t> data;
    alphabet = std::vector<uint32_t>(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    data = ecl::generateRandomData<uint32_t>(alphabet, data_size);

    ecl::WaveletTree<uint32_t> wt(data.data(), data_size, std::move(alphabet),
                                  0);
    cudaProfilerStart();
    auto results = wt.template access<1>(queries.data(), num_queries);
    cudaProfilerStop();
    results = wt.template access<2>(queries.data(), num_queries);
    results = wt.template access<4>(queries.data(), num_queries);
    results = wt.template access<8>(queries.data(), num_queries);
    results = wt.template access<16>(queries.data(), num_queries);
    results = wt.template access<32>(queries.data(), num_queries);
  }
}