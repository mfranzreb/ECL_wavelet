#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>
#include <wavelet_tree.cuh>

#include "test_benchmark_utils.cuh"

namespace ecl {

static constexpr uint8_t kGPUIndex = 0;

template <typename T>
class WaveletTreeTest : public WaveletTree<T> {
 public:
  using WaveletTree<T>::WaveletTree;
  using WaveletTree<T>::computeGlobalHistogram;
};

template <typename T>
class WaveletTreeTestFixture : public ::testing::Test {
 protected:
  T* result;
  void SetUp() override { gpuErrchk(cudaMallocManaged(&result, sizeof(T))); }
  void TearDown() override { gpuErrchk(cudaFree(result)); }
};

template <typename T>
std::pair<std::vector<T>, std::vector<T>> generateRandomAlphabetDataAndHist(
    size_t const alphabet_size, size_t const data_size,
    std::unordered_map<T, size_t>& hist) {
  if (alphabet_size < 3) {
    throw std::invalid_argument("Alphabet size must be at least 3");
  }
  std::vector<T> alphabet(alphabet_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
  size_t filled = 0;
  do {
    std::generate(alphabet.begin() + filled, alphabet.end(),
                  [&]() { return dis(gen); });
    // Check that all elements are unique
    std::sort(alphabet.begin(), alphabet.end());
    // remove duplicates
    auto it = std::unique(alphabet.begin(), alphabet.end());
    filled = std::distance(alphabet.begin(), it);
    if (filled > 2) {
      if (filled < alphabet_size) {
        alphabet.resize(filled);
      }
      break;
    }
  } while (filled != alphabet_size);

  std::vector<T> data(data_size);
  std::uniform_int_distribution<size_t> dis2(0, filled - 1);
  std::generate(data.begin(), data.end(), [&]() {
    auto const symbol = alphabet[dis2(gen)];
    hist[symbol]++;
    return symbol;
  });

  return std::make_pair(alphabet, data);
}

template <typename T>
std::vector<size_t> calculateHistogram(const std::vector<T>& data,
                                       const std::vector<T>& alphabet) {
  std::vector<size_t> histogram(alphabet.size(), 0);

  // Count occurrences of each value in the data
  for (T const& value : data) {
    auto it = std::find(alphabet.begin(), alphabet.end(), value);
    if (it != alphabet.end()) {
      histogram[std::distance(alphabet.begin(), it)]++;
    }
  }
  return histogram;
}

template <typename T>
__global__ void BAaccessKernel(BitArray bit_array, size_t array_index,
                               size_t index, T* output) {
  *output = static_cast<T>(bit_array.access(array_index, index));
}

template <typename T, int NumThreads>
__host__ void compareAccessResults(WaveletTree<T>& wt,
                                   std::vector<size_t>& indices,
                                   std::vector<T> const& data) {
  auto const results =
      wt.template access<NumThreads>(indices.data(), indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(data[indices[i]], results[i]);
  }
}

template <typename T, int NumThreads>
__host__ void compareRankResults(WaveletTree<T>& wt,
                                 std::vector<RankSelectQuery<T>> const& queries,
                                 std::vector<T> const& data) {
  auto queries_copy = queries;
  auto const results =
      wt.template rank<NumThreads>(queries_copy.data(), queries_copy.size());
  for (size_t i = 0; i < queries.size(); ++i) {
    if (std::count(data.begin(), data.begin() + queries[i].index_,
                   queries[i].symbol_) != results[i]) {
      std::cout << "i: " << i << " index: " << queries[i].index_
                << " symbol: " << queries[i].symbol_ << " count: "
                << std::count(data.begin(), data.begin() + queries[i].index_,
                              queries[i].symbol_)
                << " result: " << results[i] << std::endl;
    }
    queries_copy = queries;
    auto const results2 =
        wt.template rank<NumThreads>(queries_copy.data(), queries_copy.size());
  }
}

using MyTypes = testing::Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(WaveletTreeTestFixture, MyTypes);

TYPED_TEST(WaveletTreeTestFixture, accessRandom) {
  return;
  for (int i = 0; i < 10; i++) {
    // Random data size between 1000 and 1'000'000
    size_t data_size = 1000 + (rand() % 1'000'000);
    size_t alphabet_size =
        3 +
        (rand() %
         (std::min(static_cast<size_t>(std::numeric_limits<TypeParam>::max()),
                   data_size) -
          3));

    auto [alphabet, data] =
        generateRandomAlphabetAndData<TypeParam>(alphabet_size, data_size);
    alphabet_size = alphabet.size();

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    size_t num_indices =
        i % 5 == 0 ? std::min(30 * alphabet_size, 100'000UL) : 100;
    auto indices = generateRandomAccessQueries(data_size, num_indices);

    compareAccessResults<TypeParam, 1>(wt, indices, data);
    compareAccessResults<TypeParam, 2>(wt, indices, data);
    compareAccessResults<TypeParam, 4>(wt, indices, data);
    compareAccessResults<TypeParam, 8>(wt, indices, data);
    compareAccessResults<TypeParam, 16>(wt, indices, data);
    compareAccessResults<TypeParam, 32>(wt, indices, data);
  }
}

TYPED_TEST(WaveletTreeTestFixture, rank) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                            kGPUIndex);

  std::vector<RankSelectQuery<TypeParam>> queries;
  for (size_t i = 0; i < data.size(); ++i) {
    queries.push_back({i, data[i]});
  }
  compareRankResults<TypeParam, 1>(wt, queries, data);
  compareRankResults<TypeParam, 2>(wt, queries, data);
  compareRankResults<TypeParam, 4>(wt, queries, data);
  compareRankResults<TypeParam, 8>(wt, queries, data);
  compareRankResults<TypeParam, 16>(wt, queries, data);
  compareRankResults<TypeParam, 32>(wt, queries, data);
}

TYPED_TEST(WaveletTreeTestFixture, rankRandom) {
  for (int i = 0; i < 10; i++) {
    // Random data size between 1000 and 1'000'000
    size_t data_size = 1000 + (rand() % 1'000'000);
    size_t alphabet_size =
        3 +
        (rand() %
         (std::min(static_cast<size_t>(std::numeric_limits<TypeParam>::max()),
                   data_size) -
          3));

    auto [alphabet, data] =
        generateRandomAlphabetAndData<TypeParam>(alphabet_size, data_size);
    alphabet_size = alphabet.size();
    auto alphabet_copy = alphabet;

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    // Create 100 random rank queries
    size_t num_queries =
        i % 5 == 0 ? std::min(30 * alphabet_size, 100'000UL) : 100;
    auto queries = generateRandomRSQueries<TypeParam>(data_size, num_queries,
                                                      alphabet_copy);

    compareRankResults<TypeParam, 1>(wt, queries, data);
    compareRankResults<TypeParam, 2>(wt, queries, data);
    compareRankResults<TypeParam, 4>(wt, queries, data);
    compareRankResults<TypeParam, 8>(wt, queries, data);
    compareRankResults<TypeParam, 16>(wt, queries, data);
    compareRankResults<TypeParam, 32>(wt, queries, data);
  }
}

TYPED_TEST(WaveletTreeTestFixture, select) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                            kGPUIndex);

  std::vector<RankSelectQuery<TypeParam>> queries;
  for (size_t i = 0; i < data.size(); ++i) {
    queries.push_back({i / 10 + 1, data[i]});
  }
  auto results = wt.select(queries);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(i, results[i]);
  }

  // Check that if there is no n-th occurrence of a symbol, the result is the
  // size of the data
  queries = std::vector<RankSelectQuery<TypeParam>>{{11, 0}};
  results = wt.select(queries);
  EXPECT_EQ(data.size(), results[0]);
}

TYPED_TEST(WaveletTreeTestFixture, selectRandom) {
  int num_iters = 10;
  int queries_per_iter = 100;
  if constexpr (std::is_same<TypeParam, uint64_t>::value or
                std::is_same<TypeParam, uint32_t>::value) {
    num_iters = 5;
    queries_per_iter = 10;
  }
  for (int i = 0; i < num_iters; i++) {
    // Random data size between 1000 and 1'000'000
    size_t data_size = 1000 + (rand() % 1'000'000);
    size_t alphabet_size =
        std::min(3 + (rand() % (data_size - 3)),
                 size_t(std::numeric_limits<TypeParam>::max()));

    std::unordered_map<TypeParam, size_t> hist;
    auto [alphabet, data] = generateRandomAlphabetDataAndHist<TypeParam>(
        alphabet_size, data_size, hist);
    alphabet_size = alphabet.size();
    auto alphabet_copy = alphabet;

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    // Create 100 random select queries
    std::vector<RankSelectQuery<TypeParam>> queries(queries_per_iter);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<TypeParam> dis_alphabet(0, alphabet_size - 1);
    std::generate(queries.begin(), queries.end(), [&]() {
      TypeParam symbol_index;
      TypeParam symbol;
      size_t count;
      do {
        symbol_index = dis_alphabet(gen);
        symbol = alphabet_copy[symbol_index];
        count = hist[symbol];
      } while (count == 0);
      std::uniform_int_distribution<size_t> dis_index(1, count);
      auto index = dis_index(gen);

      return RankSelectQuery<TypeParam>{index, symbol};
    });

    auto queries_copy = queries;
    auto results = wt.select(queries_copy);
    for (size_t j = 0; j < queries.size(); ++j) {
      size_t counts = 0;
      EXPECT_EQ(std::find_if(data.begin(), data.end(),
                             [&](TypeParam c) {
                               return c == queries[j].symbol_ and
                                      ++counts == queries[j].index_;
                             }) -
                    data.begin(),
                results[j]);
    }
  }
}

/*
TYPED_TEST(WaveletTreeTestFixture, fillLevelRandom) {
  for (int i = 0; i < 100; i++) {
    // Random data size between 100 and 1'000'000
    size_t data_size = 100 + (rand() % 1'000'000);

    // Fill a vector with random data
    std::vector<TypeParam> data(data_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<TypeParam> dis(
        0, std::numeric_limits<TypeParam>::max());
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });

    // Copy the data to the device
    TypeParam* d_data;
    gpuErrchk(cudaMalloc(&d_data, sizeof(TypeParam) * data_size));
    gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data_size,
                         cudaMemcpyHostToDevice));

    // CHoose a random alphabet start bit
    uint8_t num_bits = 8 * sizeof(TypeParam);
    uint8_t start_bit = rand() % num_bits;
    // Fill all levels from start_bit to 0
    std::vector<size_t> sizes(start_bit + 1, data_size);
    BitArray ba(sizes, false);
    for (uint8_t level = 0; level <= start_bit; ++level) {
      fillLevelKernel<TypeParam><<<1, 32>>>(ba, d_data, data_size, level);
      kernelCheck();

      // Check that the level is correctly filled
      std::vector<bool> level_should(data_size);
      for (size_t i = 0; i < data_size; ++i) {
        level_should[i] = getBit(start_bit - level, data[i]);
      }
    }
    cudaFree(d_data);
  }
*/
}  // namespace ecl