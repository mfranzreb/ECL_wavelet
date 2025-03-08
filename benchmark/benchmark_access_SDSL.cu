#include <benchmark/benchmark.h>

#include <random>
#include <utils.cuh>

#include "sdsl/wavelet_trees.hpp"
#include "test_benchmark_utils.cuh"

namespace ecl {

sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>>
    sdsl_byte_wt;

sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>,
            sdsl::wt_pc<sdsl::balanced_shape>::select_1_type,
            sdsl::wt_pc<sdsl::balanced_shape>::select_0_type, sdsl::int_tree<>>
    sdsl_short_wt;

size_t curr_alphabet_size = 0;
size_t curr_data_size = 0;

// SDSL sometimes returns wrong result, "construct_im" ffunction does not work
// properly.
template <typename T>
static void BM_SDSL(benchmark::State& state) {
  auto const num_queries = state.range(0);
  auto const alphabet_size = state.range(1);
  auto const data_size = state.range(2);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.num_queries"] = num_queries;

  bool const rebuild_wt =
      curr_alphabet_size != alphabet_size or curr_data_size != data_size;

  static std::vector<T> alphabet;
  static std::vector<T> data;

  std::vector<size_t> queries;
  if (rebuild_wt) {
    alphabet = std::vector<T>(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    auto data = generateRandomData<T>(alphabet, data_size);
    queries = generateRandomAccessQueries<T>(data_size, num_queries);
    curr_alphabet_size = alphabet_size;
    curr_data_size = data_size;
    // write data to file
    std::ofstream data_file("data_file");
    data_file.write(reinterpret_cast<const char*>(data.data()),
                    data.size() * sizeof(T));
    data_file.close();
  } else {
    queries = generateRandomAccessQueries<T>(data_size, num_queries);
  }

  std::vector<T> results_sdsl(num_queries);
  if constexpr (sizeof(T) == 1) {
    if (rebuild_wt) {
      sdsl::construct(sdsl_byte_wt, "data_file", sizeof(T));
      // delete file
      std::remove("data_file");
    }

    for (auto _ : state) {
#pragma omp parallel for
      for (size_t i = 0; i < num_queries; ++i) {
        results_sdsl[i] = sdsl_byte_wt[queries[i]];
      }
    }
  } else {
    if (rebuild_wt) {
      sdsl::construct(sdsl_short_wt, "data_file", sizeof(T));
      // delete file
      std::remove("data_file");
    }

    for (auto _ : state) {
#pragma omp parallel for
      for (size_t i = 0; i < num_queries; ++i) {
        results_sdsl[i] = sdsl_short_wt[queries[i]];
      }
    }
  }
}

BENCHMARK(BM_SDSL<uint8_t>)
    ->ArgsProduct({{100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {4, 5, 24, 64, 100, 128, 155, 250},
                   {500'000'000, 800'000'000, 1'000'000'000, 1'500'000'000,
                    2'000'000'000}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SDSL<uint16_t>)
    ->ArgsProduct({{100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000, 64'000},
                   {500'000'000, 800'000'000, 1'000'000'000, 1'200'000'000}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

}  // namespace ecl