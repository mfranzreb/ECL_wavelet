#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <vector>
#include <wavelet_tree.cuh>

namespace ecl {

template <typename T>
class WaveletTreeTest : public ::testing::Test {
 protected:
  T* result;
  void SetUp() override { gpuErrchk(cudaMallocManaged(&result, sizeof(T))); }
  void TearDown() override { gpuErrchk(cudaFree(result)); }
};

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

using MyTypes = testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(WaveletTreeTest, MyTypes);

/*
TYPED_TEST(WaveletTreeTest, WaveletTreeConstructor) {
  std::vector<TypeParam> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> alphabet{1, 2, 3, 4, 5, 6, 7, 8, 9};
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));
}

TYPED_TEST(WaveletTreeTest, createMinimalCodes) {
  // Check that for powers of two, the minimal code is the same as the value
  for (size_t i = 4; i < 8 * sizeof(TypeParam); i *= 2) {
    std::vector<TypeParam> alphabet(i);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
    for (size_t j = 0; j < alphabet.size(); ++j) {
      EXPECT_EQ(alphabet[j], codes[j].code_);
    }
  }

  size_t alphabet_size = 72;
  std::vector<TypeParam> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  auto code_value = 64UL;
  for (int i = 64; i <= 71; ++i) {
    EXPECT_EQ(codes[i].code_, code_value);
    EXPECT_EQ(codes[i].len_, 4);
    code_value += 8;
  }

  alphabet_size = 63;
  alphabet.resize(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  // All the codes except the last are the same
  for (int i = 0; i < 62; ++i) {
    EXPECT_EQ(codes[i].code_, i);
    EXPECT_EQ(codes[i].len_, 6);
  }
  EXPECT_EQ(codes[62].code_, 62);
  EXPECT_EQ(codes[62].len_, 5);

  alphabet_size = 75;
  alphabet.resize(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0);
  codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  code_value = 64UL;
  for (int i = 64; i < 72; ++i) {
    EXPECT_EQ(codes[i].code_, code_value);
    EXPECT_EQ(codes[i].len_, 5);
    code_value += 4;
  }
  EXPECT_EQ(codes[72].code_, 96);
  EXPECT_EQ(codes[72].len_, 4);
  EXPECT_EQ(codes[73].code_, 104);
  EXPECT_EQ(codes[73].len_, 4);
  EXPECT_EQ(codes[74].code_, 112);
  EXPECT_EQ(codes[74].len_, 3);
}

TYPED_TEST(WaveletTreeTest, TestGlobalHistogram) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  std::vector<TypeParam> data(1000);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }

  // allocate memory for arguments of kernel
  TypeParam* d_alphabet;
  TypeParam* d_data;
  size_t* d_histogram;
  gpuErrchk(cudaMalloc(&d_alphabet, sizeof(TypeParam) * alphabet.size()));
  gpuErrchk(cudaMemcpy(d_alphabet, alphabet.data(),
                       sizeof(TypeParam) * alphabet.size(),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_data, sizeof(TypeParam) * data.size()));
  gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data.size(),
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&d_histogram, sizeof(size_t) * alphabet.size()));
  gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet.size()));

  // Create the wavelet tree
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));
  computeGlobalHistogramKernel<TypeParam><<<1, 32>>>(
      wt, d_data, data.size(), d_histogram, d_alphabet, alphabet.size());
  kernelCheck();

  // Pass the histogram to the host
  std::vector<size_t> h_histogram(alphabet.size());
  gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                       sizeof(size_t) * alphabet.size(),
                       cudaMemcpyDeviceToHost));

  auto hist_should = calculateHistogram(data, alphabet);
  for (size_t i = 0; i < alphabet.size(); ++i) {
    EXPECT_EQ(hist_should[i], h_histogram[i]);
  }

  data = std::vector<TypeParam>(1000, 0);
  gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data.size(),
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet.size()));
  computeGlobalHistogramKernel<TypeParam><<<1, 32>>>(
      wt, d_data, data.size(), d_histogram, d_alphabet, alphabet.size());

  gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                       sizeof(size_t) * alphabet.size(),
                       cudaMemcpyDeviceToHost));

  hist_should = calculateHistogram(data, alphabet);
  for (size_t i = 0; i < alphabet.size(); ++i) {
    EXPECT_EQ(hist_should[i], h_histogram[i]);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    if (i < alphabet.size()) {
      data[i] = i;
    } else {
      data[i] = alphabet.size() - 1;
    }
  }

  gpuErrchk(cudaMemcpy(d_data, data.data(), sizeof(TypeParam) * data.size(),
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_histogram, 0, sizeof(size_t) * alphabet.size()));
  computeGlobalHistogramKernel<TypeParam><<<1, 32>>>(
      wt, d_data, data.size(), d_histogram, d_alphabet, alphabet.size());

  gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                       sizeof(size_t) * alphabet.size(),
                       cudaMemcpyDeviceToHost));

  hist_should = calculateHistogram(data, alphabet);
  for (size_t i = 0; i < alphabet.size(); ++i) {
    EXPECT_EQ(hist_should[i], h_histogram[i]);
  }

  // Free memory
  gpuErrchk(cudaFree(d_alphabet));
  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_histogram));
}

TYPED_TEST(WaveletTreeTest, TestGlobalHistogramRandom) {
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

    auto [alphabet, data] =
        generateRandomAlphabetAndData<TypeParam>(alphabet_size, data_size);

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
    WaveletTree<TypeParam> wt(data.data(), data_size, std::move(alphabet_copy));
    computeGlobalHistogramKernel<TypeParam><<<1, 32>>>(
        wt, d_data, data_size, d_histogram, d_alphabet, alphabet_size);
    kernelCheck();

    // Pass the histogram to the host
    std::vector<size_t> h_histogram(alphabet_size);
    gpuErrchk(cudaMemcpy(h_histogram.data(), d_histogram,
                         sizeof(size_t) * alphabet_size,
                         cudaMemcpyDeviceToHost));

    auto const hist_should = calculateHistogram(data, alphabet);
    for (size_t i = 0; i < alphabet_size; ++i) {
      EXPECT_EQ(hist_should[i], h_histogram[i]);
    }
    gpuErrchk(cudaFree(d_alphabet));
    gpuErrchk(cudaFree(d_histogram));
  }

  // Free memory
  gpuErrchk(cudaFree(d_data));
}

TYPED_TEST(WaveletTreeTest, structure) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));

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

TYPED_TEST(WaveletTreeTest, access) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));

  std::vector<size_t> indices(data.size());
  std::iota(indices.begin(), indices.end(), 0);
  auto results = wt.access(indices);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(data[indices[i]], results[i]);
  }
}

TYPED_TEST(WaveletTreeTest, accessRandom) {
  for (int i = 0; i < 10; i++) {
    // Random data size between 1000 and 1'000'000
    size_t data_size = 1000 + (rand() % 1'000'000);
    size_t alphabet_size =
        std::min(3 + (rand() % (data_size - 3)),
                 size_t(std::numeric_limits<TypeParam>::max()));

    auto [alphabet, data] =
        generateRandomAlphabetAndData<TypeParam>(alphabet_size, data_size);
    alphabet_size = alphabet.size();

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));

    // Create 100 random access queries
    std::vector<size_t> indices(100);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(0, data_size - 1);
    std::generate(indices.begin(), indices.end(), [&]() { return dis(gen); });

    auto results = wt.access(indices);
    for (size_t j = 0; j < indices.size(); ++j) {
      EXPECT_EQ(data[indices[j]], results[j]);
    }
  }
}

TYPED_TEST(WaveletTreeTest, rank) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));

  std::vector<RankSelectQuery<TypeParam>> queries;
  for (size_t i = 0; i < data.size(); ++i) {
    queries.push_back({i, data[i]});
  }
  auto results = wt.rank(queries);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(i / 10, results[i]);
  }
}

TYPED_TEST(WaveletTreeTest, rankRandom) {
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

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));

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
*/
TYPED_TEST(WaveletTreeTest, select) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));

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

TYPED_TEST(WaveletTreeTest, selectRandom) {
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

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet));

    // Create 100 random select queries
    auto const hist = calculateHistogram(data, alphabet_copy);
    std::vector<RankSelectQuery<TypeParam>> queries(100);
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
        count = hist[symbol_index];
        break;
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
TYPED_TEST(WaveletTreeTest, fillLevelRandom) {
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