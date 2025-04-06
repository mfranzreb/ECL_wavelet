#include <gtest/gtest.h>

#include <algorithm>
#include <cub/device/device_select.cuh>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

#include "ecl_wavelet/tree/wavelet_tree.cuh"
#include "ecl_wavelet/utils/test_benchmark_utils.cuh"

namespace ecl {

static constexpr uint8_t kGPUIndex = 0;

template <typename T>
class WaveletTreeTest : public WaveletTree<T> {
 public:
  using WaveletTree<T>::WaveletTree;
  using WaveletTree<T>::computeGlobalHistogram;
  using WaveletTree<T>::getNodeInfos;
  using WaveletTree<T>::alphabet_;
};

template <typename T>
class WaveletTreeTestFixture : public ::testing::Test {
 protected:
  T* result;
  void SetUp() override { gpuErrchk(cudaMallocManaged(&result, sizeof(T))); }
  void TearDown() override { gpuErrchk(cudaFree(result)); }
};

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

template <typename T, bool IsPowTwo>
__global__ void getNodePosKernel(WaveletTree<T> tree, T symbol, uint8_t l,
                                 T* output) {
  *output = tree.getNodePosAtLevel<IsPowTwo>(symbol, l);
}

template <typename T>
__host__ void compareAccessResults(WaveletTree<T>& wt,
                                   std::vector<size_t>& indices,
                                   std::vector<T> const& data) {
  auto const results = wt.access(indices.data(), indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(data[indices[i]], results[i]);
  }
}

template <typename T>
__host__ void compareRankResults(WaveletTree<T>& wt,
                                 std::vector<RankSelectQuery<T>> const& queries,
                                 std::vector<size_t> const& results_should,
                                 size_t const num_queries) {
  auto queries_copy = queries;
  auto const results = wt.rank(queries_copy.data(), num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    EXPECT_EQ(results_should[i], results[i]);
  }
}

template <typename T>
__host__ void compareSelectResults(
    WaveletTree<T>& wt, std::vector<RankSelectQuery<T>> const& queries,
    std::vector<size_t> const& results_should, size_t const num_queries) {
  auto queries_copy = queries;
  auto const results = wt.select(queries_copy.data(), num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    EXPECT_EQ(results_should[i], results[i]);
  }
}

using MyTypes = testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(WaveletTreeTestFixture, MyTypes);

TYPED_TEST(WaveletTreeTestFixture, WaveletTreeConstructor) {
  checkWarpSize(kGPUIndex);
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
  {
    data = std::vector<TypeParam>{0, 1, 2};
    std::vector<TypeParam> alphabet{0, 1, 2};

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);
  }
  if (sizeof(TypeParam) <= 2) {
    size_t free_mem, total_mem;
    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));

    size_t const max_data_size = std::min(
        free_mem / sizeof(TypeParam),
        size_t(10'000'000'000));  // 10 GB should be enough to test all cases
    std::vector<TypeParam> data(max_data_size);
    // Test big data sizes
    {
      std::vector<TypeParam> alphabet{0, 1, 2, 3, 4};
#pragma omp parallel for
      for (size_t i = 0; i < max_data_size; ++i) {
        data[i] = i % alphabet.size();
      }
      std::vector<size_t> data_sizes(50);
      generateRandomNums<size_t, true>(data_sizes, size_t(1000), max_data_size);
      for (size_t data_size : data_sizes) {
        auto alphabet_copy = alphabet;
        try {
          WaveletTree<TypeParam> wt(data.data(), data_size,
                                    std::move(alphabet_copy), kGPUIndex);
        } catch (std::runtime_error& e) {
          assert(std::string(e.what()) ==
                 "Not enough memory available for the wavelet tree.");
          continue;
        }
      }
    }
    {
      size_t const alphabet_size = std::numeric_limits<TypeParam>::max();
      std::vector<TypeParam> alphabet(alphabet_size);
      std::iota(alphabet.begin(), alphabet.end(), 0ULL);
#pragma omp parallel for
      for (size_t i = 0; i < max_data_size; ++i) {
        data[i] = i % alphabet.size();
      }
      std::vector<size_t> data_sizes(50);
      generateRandomNums(data_sizes, size_t(1000), max_data_size);
      for (size_t data_size : data_sizes) {
        auto alphabet_copy = alphabet;
        try {
          WaveletTree<TypeParam> wt(data.data(), data_size,
                                    std::move(alphabet_copy), kGPUIndex);
        } catch (std::runtime_error& e) {
          assert(std::string(e.what()) ==
                 "Not enough memory available for the wavelet tree.");
          continue;
        }
      }
    }
  }
}

TYPED_TEST(WaveletTreeTestFixture, createMinimalCodes) {
  // Check that for powers of two, the vector is empty
  for (size_t i = 4; i < 8 * sizeof(TypeParam); i *= 2) {
    std::vector<TypeParam> alphabet(i);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
    EXPECT_TRUE(codes.empty());
  }

  size_t alphabet_size = 72;
  std::vector<TypeParam> alphabet(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0ULL);
  auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  auto code_value = 64UL;
  for (int i = 64; i <= 71; ++i) {
    EXPECT_EQ(codes[i - 64].code_, code_value);
    EXPECT_EQ(codes[i - 64].len_, 4);
    code_value += 8;
  }

  alphabet_size = 63;
  alphabet.resize(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0ULL);
  codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  EXPECT_EQ(codes[0].code_, 62);
  EXPECT_EQ(codes[0].len_, 5);

  alphabet_size = 75;
  alphabet.resize(alphabet_size);
  std::iota(alphabet.begin(), alphabet.end(), 0ULL);
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

TYPED_TEST(WaveletTreeTestFixture, getNodePosRandom) {
  if (sizeof(TypeParam) >= 4) {
    return;
  }
#pragma nv_diag_suppress 128
  for (int i = 0; i < 10; i++) {
#pragma nv_diag_default 128
    size_t alphabet_size =

        kMinAlphabetSize +
        (rand() % (std::numeric_limits<TypeParam>::max() - kMinAlphabetSize));

    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);

    auto alphabet_copy = alphabet;
    WaveletTree<TypeParam> wt(alphabet.data(), alphabet.size(),
                              std::move(alphabet_copy), kGPUIndex);

    auto const codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
    auto const node_starts =
        WaveletTreeTest<TypeParam>::getNodeInfos(alphabet, codes);
    TypeParam node_counter = 0;
    bool const is_pow_two = isPowTwo<TypeParam>(alphabet_size);
    for (size_t j = 0; j < node_starts.size(); ++j) {
      if (j > 0 and node_starts[j].level_ != node_starts[j - 1].level_) {
        node_counter = 0;
      }
      if (is_pow_two) {
        getNodePosKernel<TypeParam, true><<<1, 1>>>(
            wt, node_starts[j].start_, node_starts[j].level_, this->result);
      } else {
        getNodePosKernel<TypeParam, false><<<1, 1>>>(
            wt, node_starts[j].start_, node_starts[j].level_, this->result);
      }
      kernelCheck();
      EXPECT_EQ(node_counter++, *this->result);
    }
  }
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
  detail::computeGlobalHistogramKernel<TypeParam, true, false, false>
      <<<1, 32>>>(wt, d_data, data.size(), d_histogram, d_alphabet,
                  alphabet_size, 0);
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
  detail::computeGlobalHistogramKernel<TypeParam, true, false, false>
      <<<1, 32>>>(wt, d_data, data.size(), d_histogram, d_alphabet,
                  alphabet_size, 0);
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
  detail::computeGlobalHistogramKernel<TypeParam, true, false, false>
      <<<1, 32>>>(wt, d_data, data.size(), d_histogram, d_alphabet,
                  alphabet_size, 0);
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
  detail::computeGlobalHistogramKernel<TypeParam, true, false, false>
      <<<1, 32>>>(wt, d_data, data.size(), d_histogram, d_alphabet,
                  alphabet_size, 0);
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
  size_t constexpr kNumIters = sizeof(TypeParam) <= 2 ? 50 : 20;
  size_t free_mem, total_mem;
  gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
  auto [alphabet_sizes, data_sizes] =
      generateRandomAlphabetAndDataSizes<TypeParam, true>(
          1000, free_mem / (3 * sizeof(TypeParam)), kNumIters);
  for (size_t i = 0; i < kNumIters; i++) {
    size_t const data_size = data_sizes[i];
    size_t alphabet_size = alphabet_sizes[i];

    std::unordered_map<TypeParam, size_t> hist_should;
    auto [alphabet, data] = generateRandomAlphabetDataAndHist<TypeParam>(
        alphabet_size, data_size, hist_should);

    alphabet_size = alphabet.size();

    // allocate memory for arguments of kernel
    TypeParam* d_data;
    gpuErrchk(cudaMalloc(&d_data, sizeof(TypeParam) * data_size));
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
    try {
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
    } catch (std::runtime_error const& e) {
      assert(std::string(e.what()) ==
             "Not enough memory available for the wavelet tree.");
      gpuErrchk(cudaFree(d_data));
      gpuErrchk(cudaFree(d_alphabet));
      gpuErrchk(cudaFree(d_histogram));
      continue;
    }
    gpuErrchk(cudaFree(d_alphabet));
    gpuErrchk(cudaFree(d_histogram));
    gpuErrchk(cudaFree(d_data));
  }
}

TYPED_TEST(WaveletTreeTestFixture, structure) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<TypeParam> data(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i % alphabet.size();
  }
  {
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

  std::vector<TypeParam> alphabet2{0, 1, 2, 3, 4};
  data = {4, 1, 1, 2, 1, 1, 3, 4, 4, 0, 0, 2, 3, 2, 4, 2, 4, 1, 3, 2};
  WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet2),
                            kGPUIndex);

  // First level
  // 10000001100000101000
  for (size_t i = 0; i < data.size(); ++i) {
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 0, i, this->result);
    kernelCheck();
    if (i == 0 or i == 7 or i == 8 or i == 14 or i == 16) {
      EXPECT_EQ(*this->result, 1);
    } else {
      EXPECT_EQ(*this->result, 0);
    }
  }

  // Second level
  // 001001001111011
  for (size_t i = 0; i < data.size() - 5; ++i) {
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 1, i, this->result);
    kernelCheck();
    if (i == 2 or i == 5 or i == 8 or i == 9 or i == 10 or i == 11 or i == 13 or
        i == 14) {
      EXPECT_EQ(*this->result, 1);
    } else {
      EXPECT_EQ(*this->result, 0);
    }
  }

  // Third level
  // 111100101010010
  for (size_t i = 0; i < data.size() - 5; ++i) {
    BAaccessKernel<TypeParam>
        <<<1, 1>>>(wt.rank_select_.bit_array_, 2, i, this->result);
    kernelCheck();
    if (i == 0 or i == 1 or i == 2 or i == 3 or i == 6 or i == 8 or i == 10 or
        i == 13) {
      EXPECT_EQ(*this->result, 1);
    } else {
      EXPECT_EQ(*this->result, 0);
    }
  }
}

TYPED_TEST(WaveletTreeTestFixture, access) {
  {
    std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<TypeParam> data(100);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet.size();
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    for (uint32_t i = 1; i < data.size(); i++) {
      std::vector<size_t> indices(i);
      std::iota(indices.begin(), indices.end(), 0ULL);
      compareAccessResults<TypeParam>(wt, indices, data);
    }
  }
  if constexpr (sizeof(TypeParam) == 1) {
    size_t alphabet_size = 256;
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    std::vector<TypeParam> data(alphabet_size * 5);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet_size;
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);
    for (uint32_t i = 1; i < data.size(); i++) {
      std::vector<size_t> indices(i);
      std::iota(indices.begin(), indices.end(), 0ULL);
      compareAccessResults<TypeParam>(wt, indices, data);
    }
  }
}

TYPED_TEST(WaveletTreeTestFixture, rank) {
  {
    std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<TypeParam> data(100);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet.size();
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    for (uint32_t i = 1; i < data.size(); i++) {
      std::vector<RankSelectQuery<TypeParam>> queries;
      for (size_t j = 0; j < i; ++j) {
        queries.push_back({j, data[j]});
      }
      std::vector<size_t> results_should(queries.size());
      for (size_t i = 0; i < queries.size(); ++i) {
        results_should[i] = std::count(
            data.begin(), data.begin() + queries[i].index_, queries[i].symbol_);
      }
      compareRankResults<TypeParam>(wt, queries, results_should,
                                    queries.size());
    }
  }

  if constexpr (sizeof(TypeParam) == 1) {
    size_t alphabet_size = 256;
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    std::vector<TypeParam> data(alphabet_size * 5);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet_size;
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    for (uint32_t i = 1; i < data.size(); i++) {
      std::vector<RankSelectQuery<TypeParam>> queries;
      for (size_t j = 0; j < i; ++j) {
        queries.push_back({j, data[j]});
      }
      std::vector<size_t> results_should(queries.size());
      for (size_t i = 0; i < queries.size(); ++i) {
        results_should[i] = std::count(
            data.begin(), data.begin() + queries[i].index_, queries[i].symbol_);
      }
      compareRankResults<TypeParam>(wt, queries, results_should,
                                    queries.size());
    }
  }
}

TYPED_TEST(WaveletTreeTestFixture, select) {
  {
    std::vector<TypeParam> alphabet{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<TypeParam> data(100);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet.size();
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    for (uint32_t i = 1; i < data.size(); i++) {
      std::vector<RankSelectQuery<TypeParam>> queries;
      for (size_t j = 0; j < data.size(); ++j) {
        queries.push_back({j / 10 + 1, data[j]});
      }
      std::vector<size_t> results_should(queries.size());
      std::iota(results_should.begin(), results_should.end(), 0ULL);
      compareSelectResults<TypeParam>(wt, queries, results_should,
                                      queries.size());
    }

    // Check that if there is no n-th occurrence of a symbol, the result is
    // the size of the data
    auto queries = std::vector<RankSelectQuery<TypeParam>>{{11, 0}};
    auto results = wt.select(queries.data(), queries.size());
    EXPECT_EQ(data.size(), results[0]);
  }

  if constexpr (sizeof(TypeParam) == 1) {
    size_t alphabet_size = 256;
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    std::vector<TypeParam> data(alphabet_size * 5);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet_size;
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    std::vector<RankSelectQuery<TypeParam>> queries;
    for (size_t i = 0; i < data.size(); ++i) {
      queries.push_back({i / alphabet_size + 1, data[i]});
    }
    std::vector<size_t> results_should(queries.size());
    std::iota(results_should.begin(), results_should.end(), 0ULL);
    compareSelectResults<TypeParam>(wt, queries, results_should,
                                    queries.size());

    // Check that if there is no n-th occurrence of a symbol, the result is
    // the size of the data
    queries = std::vector<RankSelectQuery<TypeParam>>{{11, 0}};
    auto results = wt.select(queries.data(), queries.size());
    EXPECT_EQ(data.size(), results[0]);
  }
}

TYPED_TEST(WaveletTreeTestFixture, queriesRandom) {
  size_t constexpr kNumIters = sizeof(TypeParam) <= 2 ? 50 : 20;
  size_t constexpr kNumQueries = 100'000;
  size_t free_mem, total_mem;
  gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));

  auto [alphabet_sizes, data_sizes] =
      generateRandomAlphabetAndDataSizes<TypeParam, true>(
          1000, free_mem / sizeof(TypeParam), kNumIters);

  std::vector<RankSelectQuery<TypeParam>> rank_queries(kNumQueries);
  std::vector<RankSelectQuery<TypeParam>> select_queries(kNumQueries);
  std::vector<size_t> rank_results(kNumQueries);
  std::vector<size_t> select_results(kNumQueries);
  for (size_t i = 0; i < kNumIters; i++) {
    size_t const data_size = data_sizes[i];
    size_t alphabet_size = alphabet_sizes[i];
    size_t const num_queries = std::min(data_size / 2, kNumQueries);

    auto alphabet = generateRandomAlphabet<TypeParam>(alphabet_size);
    auto data = generateRandomDataAndRSQueries<TypeParam>(
        alphabet, data_size, num_queries, rank_queries, select_queries,
        rank_results, select_results);

    try {
      WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                                kGPUIndex);
      auto indices = generateRandomAccessQueries(data_size, num_queries);
      compareAccessResults<TypeParam>(wt, indices, data);

      compareRankResults<TypeParam>(wt, rank_queries, rank_results,
                                    num_queries);

      compareSelectResults<TypeParam>(wt, select_queries, select_results,
                                      num_queries);
    } catch (std::runtime_error const& e) {
      assert(std::string(e.what()) ==
             "Not enough memory available for the wavelet tree.");
      continue;
    }
  }
  if constexpr (sizeof(TypeParam) == 1) {
    // test alphabet sizes of 3, 4, 32, 128
    size_t data_size = free_mem / (2 * sizeof(TypeParam));
    size_t const num_queries = std::min(data_size / 2, kNumQueries);
    for (size_t const alphabet_size :
         std::vector<size_t>{kMinAlphabetSize, 4, 32, 128, 256}) {
      std::vector<TypeParam> alphabet(alphabet_size);
      std::iota(alphabet.begin(), alphabet.end(), 0ULL);
      auto data = generateRandomDataAndRSQueries<TypeParam>(
          alphabet, data_size, num_queries, rank_queries, select_queries,
          rank_results, select_results);

      WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                                kGPUIndex);

      auto indices = generateRandomAccessQueries(data_size, num_queries);
      compareAccessResults<TypeParam>(wt, indices, data);

      compareRankResults<TypeParam>(wt, rank_queries, rank_results,
                                    num_queries);

      compareSelectResults<TypeParam>(wt, select_queries, select_results,
                                      num_queries);
    }
  } else if (sizeof(TypeParam) == 2) {
    size_t data_size = free_mem / (2 * sizeof(TypeParam));
    size_t const num_queries = std::min(data_size / 2, kNumQueries);
    size_t alphabet_size = 1 << 16;
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    auto data = generateRandomDataAndRSQueries<TypeParam>(
        alphabet, data_size, num_queries, rank_queries, select_queries,
        rank_results, select_results);

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);
    auto indices = generateRandomAccessQueries(data_size, num_queries);
    compareAccessResults<TypeParam>(wt, indices, data);

    compareRankResults<TypeParam>(wt, rank_queries, rank_results, num_queries);

    compareSelectResults<TypeParam>(wt, select_queries, select_results,
                                    num_queries);
  }
}

TYPED_TEST(WaveletTreeTestFixture, genAlphabetRandom) {
  size_t constexpr kNumIters = 20;
  std::vector<size_t> alphabet_sizes(kNumIters);
  generateRandomNums<size_t>(
      alphabet_sizes, kMinAlphabetSize,
      std::min(static_cast<size_t>(std::numeric_limits<TypeParam>::max()),
               size_t(100'000)));

  for (uint8_t i = 0; i < kNumIters; ++i) {
    size_t const alphabet_size = alphabet_sizes[i];
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0ULL);
    auto data = generateRandomData<TypeParam>(alphabet, 10'000'000);
    // Add alphabet at the end to make sure all symbols are present
    data.insert(data.end(), alphabet.begin(), alphabet.end());
    WaveletTreeTest<TypeParam> wt(data.data(), data.size(),
                                  std::vector<TypeParam>{}, kGPUIndex);
#pragma omp parallel for
    for (size_t j = 0; j < alphabet_size; ++j) {
      EXPECT_EQ(alphabet[j], wt.alphabet_[j]);
    }
  }
}
}  // namespace ecl