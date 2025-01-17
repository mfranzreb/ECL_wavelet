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
    size_t alphabet_size, size_t const data_size,
    std::unordered_map<T, size_t>& hist) {
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
  auto const results = wt.template access<NumThreads>(indices);
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(data[indices[i]], results[i]);
  }
}

using MyTypes = testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(WaveletTreeTestFixture, MyTypes);

TYPED_TEST(WaveletTreeTestFixture, WaveletTreeConstructor) {
  std::vector<TypeParam> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  {
    std::vector<TypeParam> alphabet{1, 2, 3, 4, 5, 6, 7, 8, 9};
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);
  }
  {
    data = std::vector<TypeParam>{0, 1, 2, 3};
    std::vector<TypeParam> alphabet{0, 1, 2, 3};

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);
  }
}

TYPED_TEST(WaveletTreeTestFixture, createMinimalCodes) {
  // Check that for powers of two, the vector is empty
  for (size_t i = 4; i < 8 * sizeof(TypeParam); i *= 2) {
    std::vector<TypeParam> alphabet(i);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
    EXPECT_TRUE(codes.empty());
  }

  size_t alphabet_size = 72;
  std::vector<TypeParam> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  auto code_value = 64UL;
  for (int i = 64; i <= 71; ++i) {
    EXPECT_EQ(codes[i - 64].code_, code_value);
    EXPECT_EQ(codes[i - 64].len_, 4);
    code_value += 8;
  }

  alphabet_size = 63;
  alphabet.resize(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  EXPECT_EQ(codes[0].code_, 62);
  EXPECT_EQ(codes[0].len_, 5);

  alphabet_size = 75;
  alphabet.resize(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  code_value = 64UL;
  for (int i = 64; i < 72; ++i) {
    EXPECT_EQ(codes[i - 64].code_, code_value);
    EXPECT_EQ(codes[i - 64].len_, 5);
    code_value += 4;
  }
  EXPECT_EQ(codes[72 - 64].code_, 96);
  EXPECT_EQ(codes[72 - 64].len_, 4);
  EXPECT_EQ(codes[73 - 64].code_, 104);
  EXPECT_EQ(codes[73 - 64].len_, 4);
  EXPECT_EQ(codes[74 - 64].code_, 112);
  EXPECT_EQ(codes[74 - 64].len_, 3);
}

TYPED_TEST(WaveletTreeTestFixture, TestGlobalHistogram) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto const alphabet_size = alphabet.size();

  std::vector<TypeParam> data(1000);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet_size;
  }

  // allocate memory for arguments of kernel
  TypeParam* d_alphabet;
  TypeParam* d_data;
  size_t* d_histogram;
  gpuErrchk(cudaMalloc(&d_alphabet, sizeof(TypeParam) * alphabet_size));
  gpuErrchk(cudaMemcpy(d_alphabet, alphabet.data(),
                       sizeof(TypeParam) * alphabet_size,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_data, sizeof(TypeParam) * data.size()));
  gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data.size(),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_histogram, sizeof(size_t) * alphabet_size));
  gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet_size));

  auto alphabet_copy = alphabet;
  // Create the wavelet tree
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet_copy),
                            kGPUIndex);
  computeGlobalHistogramKernel<TypeParam, true, false, false><<<1, 32>>>(
      wt, d_data, data.size(), d_histogram, d_alphabet, alphabet_size, 0);
  kernelCheck();

  // Pass the histogram to the host
  std::vector<size_t> h_histogram(alphabet_size);
  gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                       sizeof(size_t) * alphabet_size, cudaMemcpyDeviceToHost));

  auto hist_should = calculateHistogram(data, alphabet);
  for (size_t i = 0; i < alphabet_size; ++i) {
    EXPECT_EQ(hist_should[i], h_histogram[i]);
  }

  data = std::vector<TypeParam>(1000, 0);
  gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data.size(),
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet_size));
  computeGlobalHistogramKernel<TypeParam, true, false, false><<<1, 32>>>(
      wt, d_data, data.size(), d_histogram, d_alphabet, alphabet_size, 0);
  kernelCheck();

  gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                       sizeof(size_t) * alphabet_size, cudaMemcpyDeviceToHost));

  hist_should = calculateHistogram(data, alphabet);
  for (size_t i = 0; i < alphabet_size; ++i) {
    EXPECT_EQ(hist_should[i], h_histogram[i]);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    if (i < alphabet_size) {
      data[i] = i;
    } else {
      data[i] = alphabet_size - 1;
    }
  }

  gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data.size(),
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet_size));
  computeGlobalHistogramKernel<TypeParam, true, false, false><<<1, 32>>>(
      wt, d_data, data.size(), d_histogram, d_alphabet, alphabet_size, 0);
  kernelCheck();

  gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                       sizeof(size_t) * alphabet_size, cudaMemcpyDeviceToHost));

  hist_should = calculateHistogram(data, alphabet);
  for (size_t i = 0; i < alphabet_size; ++i) {
    EXPECT_EQ(hist_should[i], h_histogram[i]);
  }

  // Case where last two symbols have different counts
  data = std::vector<TypeParam>(1000, 0);
  for (size_t i = 0; i < 5; ++i) {
    data[i] = 9;
  }
  for (size_t i = 5; i < 15; ++i) {
    data[i] = 8;
  }
  for (size_t i = 15; i < data.size(); ++i) {
    data[i] = i % 8;
  }

  gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data.size(),
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet_size));
  computeGlobalHistogramKernel<TypeParam, true, false, false><<<1, 32>>>(
      wt, d_data, data.size(), d_histogram, d_alphabet, alphabet_size, 0);
  kernelCheck();

  gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                       sizeof(size_t) * alphabet_size, cudaMemcpyDeviceToHost));

  hist_should = calculateHistogram(data, alphabet);
  for (size_t i = 0; i < alphabet_size; ++i) {
    EXPECT_EQ(hist_should[i], h_histogram[i]);
  }

  // Free memory
  gpuErrchk(cudaFree(d_alphabet));
  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_histogram));
}

TYPED_TEST(WaveletTreeTestFixture, TestGlobalHistogramRandom) {
  size_t const data_size = 10000;
  TypeParam* d_data;
  gpuErrchk(cudaMalloc(&d_data, sizeof(TypeParam) * data_size));
  for (int i = 0; i < 100; i++) {
    // Random alphabet size between 3 and data_size
    size_t alphabet_size =
        3 +
        (rand() %
         (std::min(static_cast<size_t>(std::numeric_limits<TypeParam>::max()),
                   data_size) -
          3));

    std::unordered_map<TypeParam, size_t> hist_should;
    auto [alphabet, data] = generateRandomAlphabetDataAndHist<TypeParam>(
        alphabet_size, data_size, hist_should);

    alphabet_size = alphabet.size();

    // allocate memory for arguments of kernel
    TypeParam* d_alphabet;
    size_t* d_histogram;
    gpuErrchk(cudaMalloc(&d_alphabet, sizeof(TypeParam) * alphabet_size));
    gpuErrchk(cudaMemcpy(d_alphabet, alphabet.data(),
                         sizeof(TypeParam) * alphabet_size,
                         cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&d_histogram, sizeof(size_t) * alphabet_size));
    gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet_size));

    gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data_size,
                         cudaMemcpyHostToDevice));

    // Create the wavelet tree
    auto alphabet_copy = alphabet;
    WaveletTreeTest<TypeParam> wt(data.data(), data_size,
                                  std::move(alphabet_copy), kGPUIndex);

    wt.computeGlobalHistogram(isPowTwo<TypeParam>(alphabet_size), data_size,
                              d_data, d_alphabet, d_histogram);

    // Pass the histogram to the host
    std::vector<size_t> h_histogram(alphabet_size);
    gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                         sizeof(size_t) * alphabet_size,
                         cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < alphabet_size; ++i) {
      EXPECT_EQ(hist_should[alphabet[i]], h_histogram[i]);
    }
    gpuErrchk(cudaFree(d_alphabet));
    gpuErrchk(cudaFree(d_histogram));
  }

  // Free memory
  gpuErrchk(cudaFree(d_data));
}

TYPED_TEST(WaveletTreeTestFixture, structure) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                            kGPUIndex);

  // First level
  for (size_t i = 0; i < data.size(); ++i) {
    // 0 to 7 is 0, 8 and 9 is 1
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 0, i, this->result);
    kernelCheck();
    if (i % 10 < 8) {
      EXPECT_EQ(*this->result, 0);
    } else {
      EXPECT_EQ(*this->result, 1);
    }
  }

  // Second level
  for (size_t i = 0; i < 80; ++i) {
    // 4 0s and then 4 1s
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 1, i, this->result);
    kernelCheck();
    if (i % 8 < 4) {
      EXPECT_EQ(*this->result, 0);
    } else {
      EXPECT_EQ(*this->result, 1);
    }
  }
  for (size_t i = 80; i < 100; ++i) {
    // 0101...
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 1, i, this->result);
    kernelCheck();
    if (i % 2 == 0) {
      EXPECT_EQ(*this->result, 0);
    } else {
      EXPECT_EQ(*this->result, 1);
    }
  }

  // Third level
  for (size_t i = 0; i < 80; ++i) {
    // 2 0s and then 2 1s
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 2, i, this->result);
    kernelCheck();
    if (i % 4 < 2) {
      EXPECT_EQ(*this->result, 0);
    } else {
      EXPECT_EQ(*this->result, 1);
    }
  }

  // Fourth level
  for (size_t i = 0; i < 80; ++i) {
    // 0101...
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 3, i, this->result);
    kernelCheck();
    if (i % 2 == 0) {
      EXPECT_EQ(*this->result, 0);
    } else {
      EXPECT_EQ(*this->result, 1);
    }
  }
}

TYPED_TEST(WaveletTreeTestFixture, access) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                            kGPUIndex);

  std::vector<size_t> indices(data.size());
  std::iota(indices.begin(), indices.end(), 0);
  compareAccessResults<TypeParam, 1>(wt, indices, data);
  compareAccessResults<TypeParam, 2>(wt, indices, data);
  compareAccessResults<TypeParam, 4>(wt, indices, data);
  compareAccessResults<TypeParam, 8>(wt, indices, data);
  compareAccessResults<TypeParam, 16>(wt, indices, data);
  compareAccessResults<TypeParam, 32>(wt, indices, data);
}

TYPED_TEST(WaveletTreeTestFixture, accessRandom) {
  for (int i = 0; i < 10; i++) {
    // Random data size between 1000 and 1'000'000
    size_t data_size = 1000 + (rand() % 1'000'000);
    size_t alphabet_size = std::min(
        data_size, size_t(rand() % std::numeric_limits<TypeParam>::max()));

    auto [alphabet, data] =
        generateRandomAlphabetAndData<TypeParam>(alphabet_size, data_size);
    alphabet_size = alphabet.size();

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    // Create 100 random access queries
    std::vector<size_t> indices(100);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, data_size - 1);
    std::generate(indices.begin(), indices.end(), [&]() { return dis(gen); });

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
  auto results = wt.rank(queries);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(i / 10, results[i]);
  }
}

TYPED_TEST(WaveletTreeTestFixture, rankRandom) {
  for (int i = 0; i < 10; i++) {
    // Random data size between 1000 and 1'000'000
    size_t data_size = 1000 + (rand() % 1'000'000);
    size_t alphabet_size =
        std::min(3 + (rand() % (data_size - 3)),
                 size_t(std::numeric_limits<TypeParam>::max()));

    auto [alphabet, data] =
        generateRandomAlphabetAndData<TypeParam>(alphabet_size, data_size);
    alphabet_size = alphabet.size();
    auto alphabet_copy = alphabet;

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    // Create 100 random rank queries
    std::vector<RankSelectQuery<TypeParam>> queries(100);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis_index(0, data_size - 1);
    std::uniform_int_distribution<TypeParam> dis_alphabet(0, alphabet_size - 1);
    std::generate(queries.begin(), queries.end(), [&]() {
      return RankSelectQuery<TypeParam>{dis_index(gen),
                                        alphabet_copy[dis_alphabet(gen)]};
    });

    auto queries_copy = queries;
    auto results = wt.rank(queries_copy);
    for (size_t j = 0; j < queries.size(); ++j) {
      EXPECT_EQ(std::count(data.begin(), data.begin() + queries[j].index_,
                           queries[j].symbol_),
                results[j]);
    }
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