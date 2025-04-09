#include <benchmark/benchmark.h>

#include <random>
#include <sdsl/int_vector.hpp>
#include <sdsl/wavelet_trees.hpp>

#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

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
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    data = utils::generateRandomData<T>(alphabet, data_size);
  } else {
    std::tie(alphabet, data) =
        utils::generateRandomAlphabetAndData<T>(alphabet_size, data_size, true);
  }

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.is_min_alphabet"] = is_min_alphabet;

  if constexpr (sizeof(T) == 1) {
    sdsl::int_vector<8> signed_data(data.size());
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); i++) {
      signed_data.set_int(8 * i, data[i], 8);
    }
    for (auto _ : state) {
      sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector,
                  sdsl::rank_support_v5<>>
          wt;
      sdsl::construct_im(wt, signed_data);
    }
  } else {
    // Change vec to uint64_t
    sdsl::int_vector<> data64(data.size());
#pragma omp parallel for
    for (size_t i = 0; i < data.size(); i++) {
      data64.set_int(64 * i, static_cast<uint64_t>(data[i]), 64);
    }
    for (auto _ : state) {
      sdsl::wt_pc<
          sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>,
          sdsl::wt_pc<sdsl::balanced_shape>::select_1_type,
          sdsl::wt_pc<sdsl::balanced_shape>::select_0_type, sdsl::int_tree<>>
          wt;
      sdsl::construct_im(wt, data64);
    }
  }
}

BENCHMARK(BM_WaveletTreeConstruction<uint8_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'200'000'000},
                   {4, 5, 16, 17, 32, 33, 64, 65, 128, 129, 255, 256},
                   {0, 1}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_WaveletTreeConstruction<uint16_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'200'000'000},
                   {300, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000, 64'000},
                   {0, 1}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

}  // namespace ecl