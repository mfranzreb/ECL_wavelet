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
static void BM_Rank(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  auto const num_queries = state.range(2);
  bool const pin_memory = state.range(3);
  bool const sort_queries = state.range(4);

  auto alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.num_queries"] = num_queries;
  state.counters["param.pin_memory"] = pin_memory;
  state.counters["param.sort_queries"] = sort_queries;

  auto queries = generateRandomRankQueries<T>(data_size, num_queries, alphabet);
  if (pin_memory) {
    gpuErrchk(cudaHostRegister(queries.data(), num_queries * sizeof(size_t),
                               cudaHostAllocPortable));
  }

  auto wt = WaveletTree<T>(data.data(), data_size, std::move(alphabet), 0);

  if (sort_queries) {
    std::sort(queries.begin(), queries.end(), [](auto const& a, auto const& b) {
      return a.symbol_ < b.symbol_;
    });
  }

  for (auto _ : state) {
    auto results = wt.template rank<1>(queries.data(), num_queries);
  }
  if (pin_memory) {
    gpuErrchk(cudaHostUnregister(queries.data()));
  }
}

template <nvbio::Alphabet AlphabetType>
std::pair<
    nvbio::PackedVector<nvbio::device_tag,
                        nvbio::AlphabetTraits<AlphabetType>::SYMBOL_SIZE, true>,
    std::vector<RankSelectQuery<uint8_t>>>
getNVbioArgs(size_t const data_size, size_t const num_queries,
             uint8_t const alphabet_size = 0) {
  std::vector<uint8_t> alphabet(alphabet_size);
  std::vector<uint8_t> data;
  if constexpr (AlphabetType == nvbio::DNA) {
    alphabet = std::vector<uint8_t>{'A', 'C', 'G', 'T'};
    data = generateRandomData<uint8_t>(alphabet, data_size);
  } else if constexpr (AlphabetType == nvbio::DNA_N) {
    alphabet = std::vector<uint8_t>{'A', 'C', 'G', 'T', 'N'};
    data = generateRandomData<uint8_t>(alphabet, data_size);
  } else if constexpr (AlphabetType == nvbio::PROTEIN) {
    alphabet = std::vector<uint8_t>{'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                    'S', 'T', 'V', 'W', 'Y', 'B', 'Z', 'X'};
    data = generateRandomData<uint8_t>(alphabet, data_size);
  } else if constexpr (AlphabetType == nvbio::ASCII) {
    std::iota(alphabet.begin(), alphabet.end(), 0);
    std::tie(alphabet, data) =
        generateRandomAlphabetAndData<uint8_t>(alphabet_size, data_size, true);
  }
  uint32_t const alphabet_bits =
      nvbio::AlphabetTraits<AlphabetType>::SYMBOL_SIZE;

  // allocate a host packed vector
  nvbio::PackedVector<nvbio::host_tag, alphabet_bits, true> h_data(data_size);

  // pack the string
  nvbio::from_string<AlphabetType>(
      reinterpret_cast<const char*>(data.data()),
      reinterpret_cast<const char*>(data.data() + data.size()), h_data.begin());

  // copy it to the device
  nvbio::PackedVector<nvbio::device_tag, alphabet_bits, true> d_data(h_data);

  auto queries =
      generateRandomRankQueries<uint8_t>(data_size, num_queries, alphabet);

  return std::make_pair(d_data, queries);
}

static void BM_NVBIO(benchmark::State& state) {
  if (state.range(0) > std::numeric_limits<uint32_t>::max()) {
    state.SkipWithError("Data size is too large for NVBIO.");
    return;
  }
  uint32_t const data_size = static_cast<uint32_t>(state.range(0));
  uint8_t const alphabet_size = static_cast<uint8_t>(state.range(1));
  auto const num_queries = state.range(2);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;

  if (alphabet_size == 4) {
    auto [d_data, queries] = getNVbioArgs<nvbio::DNA>(data_size, num_queries);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);

    for (auto _ : state) {
#pragma omp parallel for
      for (auto const& query : queries) {
        nvbio::rank(wt_view, query.index_, query.symbol_);
      }
    }
  } else if (alphabet_size == 5) {
    auto [d_data, queries] = getNVbioArgs<nvbio::DNA_N>(data_size, num_queries);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);

    for (auto _ : state) {
#pragma omp parallel for
      for (auto const& query : queries) {
        nvbio::rank(wt_view, query.index_, query.symbol_);
      }
    }
  } else if (alphabet_size == 24) {
    auto [d_data, queries] =
        getNVbioArgs<nvbio::PROTEIN>(data_size, num_queries);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);

    for (auto _ : state) {
#pragma omp parallel for
      for (auto const& query : queries) {
        nvbio::rank(wt_view, query.index_, query.symbol_);
      }
    }
  } else {
    auto [d_data, queries] =
        getNVbioArgs<nvbio::ASCII>(data_size, num_queries, alphabet_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);

    for (auto _ : state) {
#pragma omp parallel for
      for (auto const& query : queries) {
        nvbio::rank(wt_view, query.index_, query.symbol_);
      }
    }
  }
}

template <typename T>
static void BM_SDSL(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  auto const num_queries = state.range(2);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.num_queries"] = num_queries;

  auto [alphabet, data] =
      generateRandomAlphabetAndData<T>(alphabet_size, data_size, true);

  auto queries = generateRandomRSQueries<T>(data_size, num_queries, alphabet);
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
        results_sdsl[i] = wt.rank(queries[i].index_, queries[i].symbol_);
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
        results_sdsl[i] = wt.rank(queries[i].index_, queries[i].symbol_);
      }
    }
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

BENCHMARK(BM_NVBIO)
    ->ArgsProduct({{500'000'000, 800'000'000, 1'000'000'000, 1'500'000'000,
                    2'000'000'000},
                   {4, 5, 24, 64, 100, 128, 155, 250},
                   {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000}})
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