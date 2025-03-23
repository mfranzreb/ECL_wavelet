#include <benchmark/benchmark.h>

#include <random>
#include <pasta/bit_vector/bit_vector.hpp>
#include <pasta/bit_vector/support/wide_rank_select.hpp>
#include <omp.h>


pasta::BitVector generateRandomBitVector(size_t size,
                                       bool const is_adversarial,
                                       uint8_t const fill_rate,
                                       size_t* one_bits_out=nullptr){
  uint32_t constexpr kWordSize = 64;

  pasta::BitVector bv(size, 0);
  auto bv_data = bv.data();
  auto const num_words = bv_data.size();

  size_t one_bits = 0;
  if (!is_adversarial) {
#pragma omp parallel reduction(+ : one_bits)
    {
      std::random_device rd;
      std::mt19937 gen(rd() ^ omp_get_thread_num());  // Thread-local generator
      std::uniform_int_distribution<size_t> bit_dist(0, 99);

#pragma omp for
      for (size_t i = 0; i < num_words; i++) {
        uint64_t word = 0;
        for (size_t j = 0; j < kWordSize; ++j) {
          bool const flip_bit =
              (static_cast<uint64_t>(bit_dist(gen)) < fill_rate);
          one_bits += flip_bit ? 1 : 0;
          word |= flip_bit << j;
        }
        bv_data[i] = word;
      }
    }
  } else {
    size_t const split_index = (num_words / 100) * (100 - fill_rate) / kWordSize;

#pragma omp parallel reduction(+ : one_bits)
    {
      std::random_device rd;
      std::mt19937 gen(rd() ^ omp_get_thread_num());
      std::uniform_int_distribution<size_t> bit_dist(0, 99);

#pragma omp for
      for (size_t i = 0; i < split_index; ++i) {
        uint64_t word = 0;
        for (size_t j = 0; j < kWordSize; ++j) {
          bool const flip_bit = (static_cast<uint64_t>(bit_dist(gen)) < 1);
          one_bits += flip_bit ? 1 : 0;
          word |= flip_bit << j;
        }
        bv_data[i] = word;
      }

#pragma omp for
      for (size_t i = split_index; i < num_words; ++i) {
        uint64_t word = 0;
        for (size_t j = 0; j < kWordSize; ++j) {
          bool const flip_bit = (static_cast<uint64_t>(bit_dist(gen)) < 99);
          one_bits += flip_bit ? 1 : 0;
          word |= flip_bit << j;
        }
        bv_data[i] = word;
      }
    }
  }
  if (one_bits_out != nullptr) {
    *one_bits_out = one_bits;
  }
  return bv;
}

static void BM_RankSelectConstruction(benchmark::State& state) {
  size_t const size = state.range(0);
  bool const is_adversarial = state.range(1);
  auto const fill_rate = state.range(2);

  state.counters["param.size"] = size;
  state.counters["param.is_adversarial"] = is_adversarial;
  state.counters["param.fill_rate"] = fill_rate;
  
  pasta::BitVector bv = generateRandomBitVector(size, is_adversarial, fill_rate);

  for (auto _ : state) {
    pasta::WideRankSelect<> rs(bv);
  }
}

template <int Value>
static void BM_RankSelectBinaryRank(benchmark::State& state) {
  size_t const size = state.range(0);
  bool const is_adversarial = state.range(1);
  auto const fill_rate = state.range(2);
  size_t const num_queries = state.range(3);

  state.counters["param.size"] = size;
  state.counters["param.is_adversarial"] = is_adversarial;
  state.counters["param.fill_rate"] = fill_rate;
  state.counters["param.num_queries"] = num_queries;
  
  pasta::BitVector bv = generateRandomBitVector(size, is_adversarial, fill_rate);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dist(0, size - 1);
  std::vector<size_t> queries(num_queries);
  std::generate(queries.begin(), queries.end(), [&]() { return dist(gen); });
  std::vector<size_t> results(num_queries);

  if constexpr (Value == 1) {
    pasta::WideRankSelect<pasta::OptimizedFor::ONE_QUERIES, pasta::FindL2WideWith::BINARY_SEARCH> rs(bv);
    for (auto _ : state) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_queries; ++i) {
      results[i] = rs.rank1(queries[i]);
    }
  }
  } else {
    pasta::WideRankSelect<pasta::OptimizedFor::ZERO_QUERIES, pasta::FindL2WideWith::BINARY_SEARCH> rs(bv);
    for (auto _ : state) {
      #pragma omp parallel for
      for (size_t i = 0; i < num_queries; ++i) {
        results[i] = rs.rank0(queries[i]);
      }
    }
  }
  // Check that all results are valid.
  for (size_t i = 0; i < num_queries; ++i) {
    if (results[i] >= size) {
      std::cerr << "Invalid result: " << results[i] << std::endl;
      std::exit(1);
    }
  }
}

template <int Value>
static void BM_RankSelectBinarySelect(benchmark::State& state) {
  size_t const size = state.range(0);
  bool const is_adversarial = state.range(1);
  auto const fill_rate = state.range(2);
  size_t const num_queries = state.range(3);

  state.counters["param.size"] = size;
  state.counters["param.is_adversarial"] = is_adversarial;
  state.counters["param.fill_rate"] = fill_rate;
  state.counters["param.num_queries"] = num_queries;
  
  size_t one_bits = 0;
  pasta::BitVector bv = generateRandomBitVector(size, is_adversarial, fill_rate, &one_bits);

  std::vector<size_t> queries(num_queries);
  std::vector<size_t> results(num_queries);
  std::random_device rd;
  std::mt19937 gen(rd());
  if constexpr (Value == 1) {
    std::uniform_int_distribution<size_t> dist(1, one_bits);
    std::generate(queries.begin(), queries.end(), [&]() { return dist(gen); });

    pasta::WideRankSelect<pasta::OptimizedFor::ONE_QUERIES, pasta::FindL2WideWith::BINARY_SEARCH> rs(bv);
    for (auto _ : state) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_queries; ++i) {
      results[i] = rs.select1(queries[i]);
    }
  }
  } else {
    std::uniform_int_distribution<size_t> dist(1, size - one_bits);
    std::generate(queries.begin(), queries.end(), [&]() { return dist(gen); });

    pasta::WideRankSelect<pasta::OptimizedFor::ZERO_QUERIES, pasta::FindL2WideWith::BINARY_SEARCH> rs(bv);
    for (auto _ : state) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_queries; ++i) {
      results[i] = rs.select0(queries[i]);
    }
  }
}

// Check that all results are valid.
for (size_t i = 0; i < num_queries; ++i) {
  if (results[i] >= size) {
    std::cerr << "Invalid result: " << results[i] << std::endl;
    std::exit(1);
  }
}
}

BENCHMARK(BM_RankSelectConstruction)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RankSelectBinaryRank<0>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RankSelectBinaryRank<1>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RankSelectBinarySelect<0>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RankSelectBinarySelect<1>)
    ->ArgsProduct({{1'000'000'000, 2'000'000'000, 4'000'000'000, 8'000'000'000,
                    16'000'000'000, 32'000'000'000},
                   {0, 1},
                   {10, 50, 90},
                   {100'000'000}})
    ->Unit(benchmark::kMillisecond);