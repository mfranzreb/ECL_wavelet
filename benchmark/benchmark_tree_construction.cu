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
  bool const is_min_alphabet = state.range(2);

  std::vector<T> alphabet;
  std::vector<T> data;
  if (is_min_alphabet) {
    alphabet = std::vector<T>(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    data = generateRandomData<T>(alphabet, data_size);
  } else {
    std::tie(alphabet, data) =
        generateRandomAlphabetAndData<T>(alphabet_size, data_size, true);
  }

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.is_min_alphabet"] = is_min_alphabet;

  for (auto _ : state) {
    state.PauseTiming();
    auto alphabet_copy = alphabet;
    state.ResumeTiming();
    WaveletTree<T> wt(data.data(), data_size, std::move(alphabet_copy), 0);
  }
}

// For initializing CUDA
BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->Args({100, 4, 1})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet, and the third argument is a boolean that indicates if the alphabet
// is minimal.
BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'500'000'000, 2'000'000'000},
                   {4, 26, 45, 64, 155, 250},
                   {0, 1}})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_WaveletTreeConstruction<uint16_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'200'000'000},
                   {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000, 64'000},
                   {0, 1}})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);
}  // namespace ecl