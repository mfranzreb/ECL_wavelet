#include <benchmark/benchmark.h>

#include <random>

#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

template <typename T>
static void BM_Rank(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  auto const num_queries = state.range(2);
  bool const pin_memory = state.range(3);
  bool const sort_queries = state.range(4);

  auto alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0ULL);
  auto data = utils::generateRandomData<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.num_queries"] = num_queries;
  state.counters["param.pin_memory"] = pin_memory;
  state.counters["param.sort_queries"] = sort_queries;

  auto queries =
      utils::generateRandomRankQueries<T>(data_size, num_queries, alphabet);
  if (pin_memory) {
    gpuErrchk(cudaHostRegister(queries.data(), num_queries * sizeof(size_t),
                               cudaHostRegisterPortable));
  }

  auto wt = WaveletTree<T>(data.data(), data_size, std::move(alphabet), 0);

  if (sort_queries) {
    std::sort(queries.begin(), queries.end(), [](auto const& a, auto const& b) {
      return a.symbol_ < b.symbol_;
    });
  }

  for (auto _ : state) {
    auto results = wt.rank(queries.data(), num_queries);
  }
  if (pin_memory) {
    gpuErrchk(cudaHostUnregister(queries.data()));
  }
}

// For initializing CUDA
BENCHMARK(BM_Rank<uint8_t>)
    ->Args({10'000'000'000, 4, 10'000'000, 0, 0})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Rank<uint16_t>)
    ->Args({5'000'000'000, 65'000, 10'000'000, 0, 0})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet, third is the number of queries, last is whether to pin memory
// before benchmark.
BENCHMARK(BM_Rank<uint8_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 5'000'000'000,
                    10'000'000'000},
                   {4, 5, 24, 64, 100, 128, 155, 250},
                   {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {0, 1},
                   {0, 1}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Rank<uint16_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 5'000'000'000},
                   {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000, 64'000},
                   {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {0, 1},
                   {0, 1}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

}  // namespace ecl