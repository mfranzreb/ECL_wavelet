#include <benchmark/benchmark.h>

#include <random>
#include <wavelet_tree.cuh>

namespace ecl {

template <typename T>
std::pair<std::vector<T>, std::vector<T>> generateRandomAlphabetAndData(
    size_t alphabet_size, size_t const data_size) {
  std::vector<T> alphabet(alphabet_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  std::generate(alphabet.begin(), alphabet.end(), [&]() { return dis(gen); });
  // Check that all elements are unique
  std::sort(alphabet.begin(), alphabet.end());
  // remove duplicates
  auto it = std::unique(alphabet.begin(), alphabet.end());
  alphabet_size = std::distance(alphabet.begin(), it);
  alphabet.resize(alphabet_size);

  std::vector<T> data(data_size);
  std::uniform_int_distribution<size_t> dis2(0, alphabet_size - 1);
  std::generate(data.begin(), data.end(),
                [&]() { return alphabet[dis2(gen)]; });

  return std::make_pair(alphabet, data);
}

template <typename T>
static void BM_WaveletTreeConstruction(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);

  auto [alphabet, data] =
      generateRandomAlphabetAndData<T>(alphabet_size, data_size);

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