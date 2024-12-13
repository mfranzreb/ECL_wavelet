#include <benchmark/benchmark.h>

#include <random>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"

//? How to get max memory usage?
namespace ecl {

template <typename T>
static void BM_WaveletTreeConstruction(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);

  auto [alphabet, data] =
      generateRandomAlphabetAndData<T>(alphabet_size, data_size, true);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;

  for (auto _ : state) {
    state.PauseTiming();
    auto alphabet_copy = alphabet;
    state.ResumeTiming();
    WaveletTree<T> wt(data.data(), data_size, std::move(alphabet_copy), 0);
  }
}

// First argument is the size of the data, second argument is the size of the
// alphabet.
BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000},
                   {4, 26, 45, 64, 155, 256}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_WaveletTreeConstruction<uint16_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000},
                   {1'000, 8'000, 24'000}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl