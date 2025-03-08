#include <benchmark/benchmark.h>

#include <random>
#include <utils.cuh>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

template <typename T>
static void BM_WaveletTreeConstruction(benchmark::State& state) {
  checkWarpSize(0);
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  bool const is_min_alphabet = state.range(2);
  bool const use_alphabet = state.range(3);

  std::vector<T> alphabet;
  std::vector<T> data;
  if (is_min_alphabet) {
    alphabet = std::vector<T>(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    data = generateRandomData<T>(alphabet, data_size);
  } else {
    std::tie(alphabet, data) =
        generateRandomAlphabetAndData<T>(alphabet_size, data_size, true);
  }

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.is_min_alphabet"] = is_min_alphabet;

  // Memory usage
  {
    size_t max_memory_usage = 0;
    std::atomic_bool done{false};
    std::atomic_bool can_start{false};
    std::thread t(measureMemoryUsage, std::ref(done), std::ref(can_start),
                  std::ref(max_memory_usage), 0);
    while (not can_start) {
      std::this_thread::yield();
    }
    auto alphabet_copy = alphabet;
    auto wt = WaveletTree<T>(
        data.data(), data_size,
        use_alphabet ? std::move(alphabet_copy) : std::vector<T>(), 0);
    done = true;
    t.join();
    state.counters["memory_usage"] = max_memory_usage;
  }

  try {
    for (auto _ : state) {
      state.PauseTiming();
      auto alphabet_copy = alphabet;
      state.ResumeTiming();
      WaveletTree<T> wt(
          data.data(), data_size,
          use_alphabet ? std::move(alphabet_copy) : std::vector<T>(), 0);
    }
  } catch (std::runtime_error& e) {
    state.SkipWithError(e.what());
  }
}

// For initializing CUDA
BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->Args({100, 4, 1})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet, and the third argument is a boolean that indicates if the
// alphabet is minimal.
BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'500'000'000, 2'000'000'000},
                   {4, 5, 24, 64, 100, 155, 256},
                   {0, 1}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_WaveletTreeConstruction<uint16_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'200'000'000},
                   {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000, 64'000},
                   {0, 1}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

}  // namespace ecl