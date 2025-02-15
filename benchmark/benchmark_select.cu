#include <benchmark/benchmark.h>
#include <nvbio/basic/packed_vector.h>
#include <nvbio/strings/alphabet.h>
#include <nvbio/strings/wavelet_tree.h>

#include <random>
#include <utils.cuh>
#include <wavelet_tree.cuh>

#include "sdsl/wavelet_trees.hpp"
#include "test_benchmark_utils.cuh"

namespace ecl {

template <typename T>
static void BM_Select(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  auto const num_queries = state.range(2);
  bool const pin_memory = state.range(3);
  bool const sort_queries = state.range(4);

  auto alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto [data, hist] = generateRandomDataAndHist<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.num_queries"] = num_queries;
  state.counters["param.pin_memory"] = pin_memory;
  state.counters["param.sort_queries"] = sort_queries;

  auto queries = generateRandomSelectQueries<T>(hist, num_queries, alphabet);
  if (pin_memory) {
    gpuErrchk(cudaHostRegister(queries.data(), num_queries * sizeof(size_t),
                               cudaHostAllocPortable));
  }
  if (sort_queries) {
    std::sort(queries.begin(), queries.end(), [](auto const& a, auto const& b) {
      return a.symbol_ < b.symbol_;
    });
  }

  auto wt = WaveletTree<T>(data.data(), data_size, std::move(alphabet), 0);

  for (auto _ : state) {
    auto results = wt.template select<1>(queries.data(), num_queries);
  }
  if (pin_memory) {
    gpuErrchk(cudaHostUnregister(queries.data()));
  }
}

template <typename T>
static void BM_SDSL(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  auto const num_queries = state.range(2);
  bool const sort_queries = state.range(3);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.num_queries"] = num_queries;
  state.counters["param.sort_queries"] = sort_queries;

  std::vector<T> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);

  auto [data, hist] = generateRandomDataAndHist<T>(alphabet, data_size);

  auto queries = generateRandomSelectQueries<T>(hist, num_queries, alphabet);
  // write data to file
  std::ofstream data_file("data_file");
  data_file.write(reinterpret_cast<const char*>(data.data()),
                  data.size() * sizeof(T));
  data_file.close();

  std::vector<size_t> results_sdsl(num_queries);
  if constexpr (sizeof(T) == 1) {
    sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>>
        wt;
    sdsl::construct(wt, "data_file", sizeof(T));

    // delete file
    std::remove("data_file");

    for (auto _ : state) {
#pragma omp parallel for
      for (size_t i = 0; i < num_queries; ++i) {
        results_sdsl[i] = wt.select(queries[i].index_, queries[i].symbol_);
      }
    }
  } else {
    sdsl::wt_pc<sdsl::balanced_shape, sdsl::bit_vector, sdsl::rank_support_v5<>,
                sdsl::wt_pc<sdsl::balanced_shape>::select_1_type,
                sdsl::wt_pc<sdsl::balanced_shape>::select_0_type,
                sdsl::int_tree<>>
        wt;
    sdsl::construct(wt, "data_file", sizeof(T));

    // delete file
    std::remove("data_file");

    for (auto _ : state) {
#pragma omp parallel for
      for (size_t i = 0; i < num_queries; ++i) {
        results_sdsl[i] = wt.select(queries[i].index_, queries[i].symbol_);
      }
    }
  }
}

// For initializing CUDA
BENCHMARK(BM_Select<uint8_t>)
    ->Args({10'000'000'000, 4, 10'000'000, 0, 0})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Select<uint16_t>)
    ->Args({5'000'000'000, 65'000, 10'000'000, 0, 0})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet, third is the number of queries, last is whether to pin memory
// before benchmark.
BENCHMARK(BM_Select<uint8_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 5'000'000'000,
                    10'000'000'000},
                   {4, 5, 24, 64, 100, 128, 155, 250},
                   {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {0, 1},
                   {0, 1}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Select<uint16_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 5'000'000'000},
                   {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000, 64'000},
                   {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {0, 1},
                   {0, 1}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SDSL<uint8_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 5'000'000'000,
                    10'000'000'000},
                   {4, 5, 24, 64, 100, 128, 155, 250},
                   {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {0, 1}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SDSL<uint16_t>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 5'000'000'000},
                   {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000, 64'000},
                   {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000},
                   {0, 1}})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

}  // namespace ecl