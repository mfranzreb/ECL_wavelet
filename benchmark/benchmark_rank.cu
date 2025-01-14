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
std::vector<RankSelectQuery<T>> generateRandomQueries(
    size_t const data_size, size_t const num_queries,
    std::vector<T> const& alphabet) {
  std::vector<RankSelectQuery<T>> queries(num_queries);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, data_size - 1);
  std::uniform_int_distribution<T> dis2(0, alphabet.size() - 1);
  std::generate(queries.begin(), queries.end(), [&]() {
    return RankSelectQuery<T>(dis(gen), alphabet[dis2(gen)]);
  });
  return queries;
}

template <typename T>
static void BM_Rank(benchmark::State& state) {
  auto const data_size = state.range(0);
  auto const alphabet_size = state.range(1);
  auto const num_queries = state.range(2);

  auto alphabet = std::vector<T>(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto data = generateRandomData<T>(alphabet, data_size);

  state.counters["param.data_size"] = data_size;
  state.counters["param.alphabet_size"] = alphabet_size;
  state.counters["param.num_queries"] = num_queries;

  auto queries = generateRandomQueries<T>(data_size, num_queries, alphabet);

  auto wt = WaveletTree<T>(data.data(), data_size, std::move(alphabet), 0);

  for (auto _ : state) {
    auto results = wt.rank(queries);
  }
}

template <nvbio::Alphabet AlphabetType>
std::pair<
    nvbio::PackedVector<nvbio::device_tag,
                        nvbio::AlphabetTraits<AlphabetType>::SYMBOL_SIZE, true>,
    std::vector<uint8_t>>
getNVbioArgs(size_t const data_size, uint8_t const alphabet_size = 0) {
  std::vector<uint8_t> alphabet;
  std::vector<uint8_t> data(data_size);
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

  return std::make_pair(d_data, alphabet);
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
    auto [d_data, alphabet] = getNVbioArgs<nvbio::DNA>(data_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);
    auto queries =
        generateRandomQueries<uint8_t>(data_size, num_queries, alphabet);
    for (auto _ : state) {
#pragma omp parallel for
      for (auto const& query : queries) {
        nvbio::rank(wt_view, query.index_, query.symbol_);
      }
    }
  } else if (alphabet_size == 5) {
    auto [d_data, alphabet] = getNVbioArgs<nvbio::DNA_N>(data_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);
    auto queries =
        generateRandomQueries<uint8_t>(data_size, num_queries, alphabet);
    for (auto _ : state) {
#pragma omp parallel for
      for (auto const& query : queries) {
        nvbio::rank(wt_view, query.index_, query.symbol_);
      }
    }
  } else if (alphabet_size == 24) {
    auto [d_data, alphabet] = getNVbioArgs<nvbio::PROTEIN>(data_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);
    auto queries =
        generateRandomQueries<uint8_t>(data_size, num_queries, alphabet);
    for (auto _ : state) {
#pragma omp parallel for
      for (auto const& query : queries) {
        nvbio::rank(wt_view, query.index_, query.symbol_);
      }
    }
  } else {
    auto [d_data, alphabet] =
        getNVbioArgs<nvbio::ASCII>(data_size, alphabet_size);
    nvbio::WaveletTreeStorage<nvbio::device_tag> wt;
    nvbio::setup(data_size, d_data.begin(), wt);
    auto const wt_view = nvbio::plain_view(
        (const nvbio::WaveletTreeStorage<nvbio::device_tag>&)wt);
    auto queries =
        generateRandomQueries<uint8_t>(data_size, num_queries, alphabet);
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

  auto queries = generateRandomQueries<T>(data_size, num_queries, alphabet);
  // write data to file
  std::ofstream data_file("data_file");
  for (auto const& d : data) {
    data_file << d;
  }
  data_file.close();

  sdsl::wt_blcd<> wt;
  sdsl::construct(wt, "data_file", sizeof(T));

  std::vector<size_t> results_sdsl(num_queries);
  for (auto _ : state) {
#pragma omp parallel for
    for (size_t i = 0; i < num_queries; ++i) {
      results_sdsl[i] = wt.rank(queries[i].index_, queries[i].symbol_);
    }
  }
}

// For initializing CUDA
BENCHMARK(BM_Rank<uint8_t>)
    ->Args({100, 4, 1})
    ->Iterations(10)
    ->Unit(benchmark::kMillisecond);

// First argument is the size of the data, second argument is the size of the
// alphabet.
BENCHMARK(BM_Rank<uint8_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'500'000'000, 2'000'000'000},
                   {4, 5, 24, 64, 100, 155, 250},
                   {100, 1'000, 5'000, 10'000, 50'000, 100'000}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_Rank<uint16_t>)
//     ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
//                     1'200'000'000},
//                    {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000,
//                    64'000},
//{100, 1'000, 5'000, 10'000, 50'000, 100'000}})
//     ->Iterations(5)
//     ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_NVBIO)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'500'000'000, 2'000'000'000},
                   {4, 5, 24, 64, 100, 155, 250},
                   {100, 1'000, 5'000, 10'000, 50'000, 100'000}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SDSL<uint8_t>)
    ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
                    1'500'000'000, 2'000'000'000},
                   {4, 5, 24, 64, 100, 155, 250},
                   {100, 1'000, 5'000, 10'000, 50'000, 100'000}})
    ->Iterations(5)
    ->Unit(benchmark::kMillisecond);

// BENCHMARK(BM_SDSL<uint16_t>)
//     ->ArgsProduct({{200'000'000, 500'000'000, 800'000'000, 1'000'000'000,
//                     1'200'000'000},
//                    {500, 1'000, 2'000, 4'000, 8'000, 16'000, 32'000,
//                    64'000},
//{100, 1'000, 5'000, 10'000, 50'000, 100'000}})
//     ->Iterations(5)
//     ->Unit(benchmark::kMillisecond);

}  // namespace ecl