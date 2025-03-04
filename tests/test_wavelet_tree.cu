#include <gtest/gtest.h>

#include <algorithm>
#include <cub/device/device_select.cuh>
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
  using WaveletTree<T>::getNodeInfos;
  using WaveletTree<T>::runDeviceSelectIf;
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
                                 std::vector<size_t> const& results_should,
                                 size_t const num_queries) {
  auto queries_copy = queries;
  auto const results =
      wt.template rank<NumThreads>(queries_copy.data(), num_queries);
  for (size_t i = 0; i < num_queries; ++i) {
    EXPECT_EQ(results_should[i], results[i]);
  }
}

template <typename T, int NumThreads>
__host__ void compareSelectResults(
    WaveletTree<T>& wt, std::vector<RankSelectQuery<T>> const& queries,
    std::vector<size_t> const& results_should, size_t const num_queries) {
  auto queries_copy = queries;
  auto const results =
      wt.template select<NumThreads>(queries_copy.data(), num_queries);
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

    size_t const max_data_size = free_mem / sizeof(TypeParam);
    std::vector<TypeParam> data(max_data_size);
    // Test big data sizes
    {
      std::vector<TypeParam> alphabet{0, 1, 2, 3, 4};
#pragma omp parallel for
      for (size_t i = 0; i < max_data_size; ++i) {
        data[i] = i % alphabet.size();
      }
      std::vector<size_t> data_sizes(20);
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
    {
      size_t const alphabet_size = std::numeric_limits<TypeParam>::max();
      std::vector<TypeParam> alphabet(alphabet_size);
      std::iota(alphabet.begin(), alphabet.end(), 0);
#pragma omp parallel for
      for (size_t i = 0; i < max_data_size; ++i) {
        data[i] = i % alphabet.size();
      }
      std::vector<size_t> data_sizes(20);
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
    std::iota(alphabet.begin(), alphabet.end(), 0);

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
  uint8_t constexpr kNumIters = 20;
  size_t free_mem, total_mem;
  gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
  auto [alphabet_sizes, data_sizes] =
      generateRandomAlphabetAndDataSizes<TypeParam>(
          1000, free_mem / sizeof(TypeParam), kNumIters);
  for (int i = 0; i < kNumIters; i++) {
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

    for (int i = 1; i < data.size(); i++) {
      std::vector<size_t> indices(i);
      std::iota(indices.begin(), indices.end(), 0);
      compareAccessResults<TypeParam, 1>(wt, indices, data);
      compareAccessResults<TypeParam, 2>(wt, indices, data);
      compareAccessResults<TypeParam, 4>(wt, indices, data);
      compareAccessResults<TypeParam, 8>(wt, indices, data);
      compareAccessResults<TypeParam, 16>(wt, indices, data);
      compareAccessResults<TypeParam, 32>(wt, indices, data);
    }
  }
  if constexpr (sizeof(TypeParam) == 1) {
    size_t alphabet_size = 256;
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    std::vector<TypeParam> data(alphabet_size * 5);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet_size;
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);
    for (int i = 1; i < data.size(); i++) {
      std::vector<size_t> indices(i);
      std::iota(indices.begin(), indices.end(), 0);
      compareAccessResults<TypeParam, 1>(wt, indices, data);
      compareAccessResults<TypeParam, 2>(wt, indices, data);
      compareAccessResults<TypeParam, 4>(wt, indices, data);
      compareAccessResults<TypeParam, 8>(wt, indices, data);
      compareAccessResults<TypeParam, 16>(wt, indices, data);
      compareAccessResults<TypeParam, 32>(wt, indices, data);
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

    for (int i = 1; i < data.size(); i++) {
      std::vector<RankSelectQuery<TypeParam>> queries;
      for (size_t j = 0; j < i; ++j) {
        queries.push_back({j, data[j]});
      }
      std::vector<size_t> results_should(queries.size());
      for (size_t i = 0; i < queries.size(); ++i) {
        results_should[i] = std::count(
            data.begin(), data.begin() + queries[i].index_, queries[i].symbol_);
      }
      compareRankResults<TypeParam, 1>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 2>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 4>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 8>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 16>(wt, queries, results_should,
                                        queries.size());
      compareRankResults<TypeParam, 32>(wt, queries, results_should,
                                        queries.size());
    }
  }

  if constexpr (sizeof(TypeParam) == 1) {
    size_t alphabet_size = 256;
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    std::vector<TypeParam> data(alphabet_size * 5);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = i % alphabet_size;
    }
    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);

    for (int i = 1; i < data.size(); i++) {
      std::vector<RankSelectQuery<TypeParam>> queries;
      for (size_t j = 0; j < i; ++j) {
        queries.push_back({j, data[j]});
      }
      std::vector<size_t> results_should(queries.size());
      for (size_t i = 0; i < queries.size(); ++i) {
        results_should[i] = std::count(
            data.begin(), data.begin() + queries[i].index_, queries[i].symbol_);
      }
      compareRankResults<TypeParam, 1>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 2>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 4>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 8>(wt, queries, results_should,
                                       queries.size());
      compareRankResults<TypeParam, 16>(wt, queries, results_should,
                                        queries.size());
      compareRankResults<TypeParam, 32>(wt, queries, results_should,
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

    for (int i = 1; i < data.size(); i++) {
      std::vector<RankSelectQuery<TypeParam>> queries;
      for (size_t j = 0; j < data.size(); ++j) {
        queries.push_back({j / 10 + 1, data[j]});
      }
      std::vector<size_t> results_should(queries.size());
      std::iota(results_should.begin(), results_should.end(), 0);
      compareSelectResults<TypeParam, 1>(wt, queries, results_should,
                                         queries.size());
      compareSelectResults<TypeParam, 2>(wt, queries, results_should,
                                         queries.size());
      compareSelectResults<TypeParam, 4>(wt, queries, results_should,
                                         queries.size());
      compareSelectResults<TypeParam, 8>(wt, queries, results_should,
                                         queries.size());
      compareSelectResults<TypeParam, 16>(wt, queries, results_should,
                                          queries.size());
      compareSelectResults<TypeParam, 32>(wt, queries, results_should,
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
    std::iota(alphabet.begin(), alphabet.end(), 0);
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
    std::iota(results_should.begin(), results_should.end(), 0);
    compareSelectResults<TypeParam, 1>(wt, queries, results_should,
                                       queries.size());
    compareSelectResults<TypeParam, 2>(wt, queries, results_should,
                                       queries.size());
    compareSelectResults<TypeParam, 4>(wt, queries, results_should,
                                       queries.size());
    compareSelectResults<TypeParam, 8>(wt, queries, results_should,
                                       queries.size());
    compareSelectResults<TypeParam, 16>(wt, queries, results_should,
                                        queries.size());
    compareSelectResults<TypeParam, 32>(wt, queries, results_should,
                                        queries.size());

    // Check that if there is no n-th occurrence of a symbol, the result is
    // the size of the data
    queries = std::vector<RankSelectQuery<TypeParam>>{{11, 0}};
    auto results = wt.select(queries.data(), queries.size());
    EXPECT_EQ(data.size(), results[0]);
  }
}

TYPED_TEST(WaveletTreeTestFixture, queriesRandom) {
  size_t constexpr kNumIters = 20;
  size_t constexpr kNumQueries = 100'000;
  size_t free_mem, total_mem;
  gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));

  auto [alphabet_sizes, data_sizes] =
      generateRandomAlphabetAndDataSizes<TypeParam>(
          1000, free_mem / sizeof(TypeParam), kNumIters);

  std::vector<RankSelectQuery<TypeParam>> rank_queries(kNumQueries);
  std::vector<RankSelectQuery<TypeParam>> select_queries(kNumQueries);
  std::vector<size_t> rank_results(kNumQueries);
  std::vector<size_t> select_results(kNumQueries);
  for (int i = 0; i < kNumIters; i++) {
    size_t const data_size = data_sizes[i];
    size_t alphabet_size = alphabet_sizes[i];
    size_t const num_queries = std::min(data_size / 2, kNumQueries);

    std::vector<TypeParam> alphabet(alphabet_size);
    generateRandomNums<TypeParam>(alphabet, 0,
                                  std::numeric_limits<TypeParam>::max());
    // remove duplicates
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()),
                   alphabet.end());
    alphabet_size = alphabet.size();
    auto data = generateRandomDataAndRSQueries<TypeParam>(
        alphabet, data_size, num_queries, rank_queries, select_queries,
        rank_results, select_results);
    auto alphabet_copy = alphabet;

    try {
      WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                                kGPUIndex);
      auto indices = generateRandomAccessQueries(data_size, num_queries);
      compareAccessResults<TypeParam, 1>(wt, indices, data);
      compareAccessResults<TypeParam, 2>(wt, indices, data);
      compareAccessResults<TypeParam, 4>(wt, indices, data);
      compareAccessResults<TypeParam, 8>(wt, indices, data);
      compareAccessResults<TypeParam, 16>(wt, indices, data);
      compareAccessResults<TypeParam, 32>(wt, indices, data);

      compareRankResults<TypeParam, 1>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 2>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 4>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 8>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 16>(wt, rank_queries, rank_results,
                                        num_queries);
      compareRankResults<TypeParam, 32>(wt, rank_queries, rank_results,
                                        num_queries);

      compareSelectResults<TypeParam, 1>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 2>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 4>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 8>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 16>(wt, select_queries, select_results,
                                          num_queries);
      compareSelectResults<TypeParam, 32>(wt, select_queries, select_results,
                                          num_queries);
    } catch (std::runtime_error const& e) {
      assert(std::string(e.what()) ==
             "Not enough memory available for the wavelet tree.");
      continue;
    }
  }
  if constexpr (sizeof(TypeParam) == 1) {
    // test alphabet sizes of 3, 4, 32, 128
    size_t data_size = data_sizes[0];
    size_t const num_queries = std::min(data_size / 2, kNumQueries);
    for (size_t const alphabet_size :
         std::vector<size_t>{kMinAlphabetSize, 4, 32, 128, 256}) {
      std::vector<TypeParam> alphabet(alphabet_size);
      std::iota(alphabet.begin(), alphabet.end(), 0);
      auto data = generateRandomDataAndRSQueries<TypeParam>(
          alphabet, data_size, num_queries, rank_queries, select_queries,
          rank_results, select_results);
      auto alphabet_copy = alphabet;

      WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                                kGPUIndex);

      auto indices = generateRandomAccessQueries(data_size, num_queries);
      compareAccessResults<TypeParam, 1>(wt, indices, data);
      compareAccessResults<TypeParam, 2>(wt, indices, data);
      compareAccessResults<TypeParam, 4>(wt, indices, data);
      compareAccessResults<TypeParam, 8>(wt, indices, data);
      compareAccessResults<TypeParam, 16>(wt, indices, data);
      compareAccessResults<TypeParam, 32>(wt, indices, data);

      compareRankResults<TypeParam, 1>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 2>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 4>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 8>(wt, rank_queries, rank_results,
                                       num_queries);
      compareRankResults<TypeParam, 16>(wt, rank_queries, rank_results,
                                        num_queries);
      compareRankResults<TypeParam, 32>(wt, rank_queries, rank_results,
                                        num_queries);

      compareSelectResults<TypeParam, 1>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 2>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 4>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 8>(wt, select_queries, select_results,
                                         num_queries);
      compareSelectResults<TypeParam, 16>(wt, select_queries, select_results,
                                          num_queries);
      compareSelectResults<TypeParam, 32>(wt, select_queries, select_results,
                                          num_queries);
    }
  } else if (sizeof(TypeParam) == 2) {
    size_t data_size = data_sizes[0];
    size_t const num_queries = std::min(data_size / 2, kNumQueries);
    size_t alphabet_size = 1 << 16;
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    auto data = generateRandomDataAndRSQueries<TypeParam>(
        alphabet, data_size, num_queries, rank_queries, select_queries,
        rank_results, select_results);
    auto alphabet_copy = alphabet;

    WaveletTree<TypeParam> wt(data.data(), data.size(), std::move(alphabet),
                              kGPUIndex);
    auto indices = generateRandomAccessQueries(data_size, num_queries);
    compareAccessResults<TypeParam, 1>(wt, indices, data);
    compareAccessResults<TypeParam, 2>(wt, indices, data);
    compareAccessResults<TypeParam, 4>(wt, indices, data);
    compareAccessResults<TypeParam, 8>(wt, indices, data);
    compareAccessResults<TypeParam, 16>(wt, indices, data);
    compareAccessResults<TypeParam, 32>(wt, indices, data);

    compareRankResults<TypeParam, 1>(wt, rank_queries, rank_results,
                                     num_queries);
    compareRankResults<TypeParam, 2>(wt, rank_queries, rank_results,
                                     num_queries);
    compareRankResults<TypeParam, 4>(wt, rank_queries, rank_results,
                                     num_queries);
    compareRankResults<TypeParam, 8>(wt, rank_queries, rank_results,
                                     num_queries);
    compareRankResults<TypeParam, 16>(wt, rank_queries, rank_results,
                                      num_queries);
    compareRankResults<TypeParam, 32>(wt, rank_queries, rank_results,
                                      num_queries);

    compareSelectResults<TypeParam, 1>(wt, select_queries, select_results,
                                       num_queries);
    compareSelectResults<TypeParam, 2>(wt, select_queries, select_results,
                                       num_queries);
    compareSelectResults<TypeParam, 4>(wt, select_queries, select_results,
                                       num_queries);
    compareSelectResults<TypeParam, 8>(wt, select_queries, select_results,
                                       num_queries);
    compareSelectResults<TypeParam, 16>(wt, select_queries, select_results,
                                        num_queries);
    compareSelectResults<TypeParam, 32>(wt, select_queries, select_results,
                                        num_queries);
  }
}

TYPED_TEST(WaveletTreeTestFixture, genAlphabetRandom) {
  uint8_t constexpr kNumIters = 20;
  std::vector<size_t> alphabet_sizes;
  generateRandomNums<size_t>(
      alphabet_sizes, kMinAlphabetSize,
      std::min(static_cast<size_t>(std::numeric_limits<TypeParam>::max()),
               size_t(100'000)));

  for (uint8_t i = 0; i < kNumIters; ++i) {
    size_t const alphabet_size = alphabet_sizes[i];
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
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

TYPED_TEST(WaveletTreeTestFixture, runDeviceSelectIf) {
  std::vector<TypeParam> alphabet{0, 1, 2, 3, 4};
  size_t const alphabet_size = alphabet.size();
  std::vector<size_t> data_sizes;
  size_t free_mem, total_mem;
  gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
  size_t max_data_size = free_mem / (2 * sizeof(TypeParam));
  for (size_t i = 1'000'000; i < max_data_size; i *= 2) {
    data_sizes.push_back(i);
  }
  // CHeck that all data sizes are a multiple of the alphabet size
  assert(std::all_of(data_sizes.begin(), data_sizes.end(),
                     [&](size_t size) { return size % alphabet_size == 0; }));
  std::vector<TypeParam> data(data_sizes.back());
  std::vector<TypeParam> reduced_data(data.size());
  auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
  TypeParam const codes_start = alphabet_size - codes.size();
#pragma omp parallel for
  for (size_t i = 0; i < data.size(); ++i) {
    auto symbol = i % alphabet.size();
    if (symbol >= codes_start) {
      data[i] = codes[symbol - codes_start].code_;
    } else {
      data[i] = symbol;
    }
  }
  TypeParam* d_data = nullptr;
  gpuErrchk(cudaMalloc(&d_data, data.size() * sizeof(TypeParam)));
  gpuErrchk(cudaMemcpy(d_data, data.data(), data.size() * sizeof(TypeParam),
                       cudaMemcpyHostToDevice));

  std::vector<uint8_t> code_lens(codes.back().code_ + 1 - codes_start);
  for (size_t j = 0; j < alphabet_size - codes_start; ++j) {
    code_lens[codes[j].code_ - codes_start] = codes[j].len_;
  }
  uint8_t* d_code_lens = nullptr;
  gpuErrchk(cudaMalloc(&d_code_lens, code_lens.size() * sizeof(uint8_t)));
  gpuErrchk(cudaMemcpy(d_code_lens, code_lens.data(),
                       code_lens.size() * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  auto pred = isLongEnough<TypeParam>(d_code_lens, 0, codes_start);
  gpuErrchk(cub::DeviceSelect::If(nullptr, temp_storage_bytes, data.data(),
                                  &data_sizes.back(),
                                  std::numeric_limits<int>::max(), pred));

  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  for (size_t data_size : data_sizes) {
    auto new_size = WaveletTreeTest<TypeParam>::runDeviceSelectIf(
        d_temp_storage, temp_storage_bytes, d_data, data_size, pred);
    EXPECT_EQ(new_size, 4 * data_size / 5);
    gpuErrchk(cudaMemcpy(reduced_data.data(), d_data,
                         new_size * sizeof(TypeParam), cudaMemcpyDeviceToHost));
#pragma omp parallel for
    for (size_t j = 0; j < new_size; ++j) {
      EXPECT_EQ(reduced_data[j], j % (alphabet.size() - 1));
    }
    gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(TypeParam),
                         cudaMemcpyHostToDevice));
  }

  gpuErrchk(cudaFree(d_data));
  gpuErrchk(cudaFree(d_temp_storage));
  gpuErrchk(cudaFree(d_code_lens));
}

TYPED_TEST(WaveletTreeTestFixture, runDeviceSelectIfRandom) {
  uint8_t constexpr kNumIters = 20;
  size_t free_mem, total_mem;
  gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
  auto [alphabet_sizes, data_sizes] =
      generateRandomAlphabetAndDataSizes<TypeParam>(
          1000, free_mem / sizeof(TypeParam), kNumIters);

  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  std::vector<TypeParam> data(100);
  gpuErrchk(cub::DeviceSelect::If(
      nullptr, temp_storage_bytes, data.data(), &temp_storage_bytes,
      std::numeric_limits<int>::max(), isLongEnough<TypeParam>(nullptr, 0, 0)));
  for (uint8_t i = 0; i < kNumIters; ++i) {
    size_t data_size = data_sizes[i];
    size_t alphabet_size = alphabet_sizes[i];

    // 2*temp_storage_bytes to have a buffer
    bool const use_unified_memory =
        free_mem < 2 * temp_storage_bytes + data_size * sizeof(TypeParam);
    // Make sure that alphabet_size is not a power of 2
    if (isPowTwo(alphabet_size)) {
      alphabet_size--;
    }
    std::vector<TypeParam> alphabet(alphabet_size);
    std::iota(alphabet.begin(), alphabet.end(), 0);
    data = generateRandomData<TypeParam>(alphabet, data_size);
    // Add alphabet at the end to make sure all symbols are present
    for (size_t j = 0; j < alphabet_size; ++j) {
      data[data_size - alphabet_size + j] = alphabet[j];
    }

    auto codes = WaveletTree<TypeParam>::createMinimalCodes(alphabet);
    TypeParam const codes_start = alphabet_size - codes.size();
// transform data to codes
#pragma omp parallel for
    for (size_t j = 0; j < data.size(); ++j) {
      auto symbol = data[j];
      if (symbol >= codes_start) {
        data[j] = codes[symbol - codes_start].code_;
      }
    }

    std::vector<uint8_t> code_lens(codes.back().code_ + 1 - codes_start);
    for (size_t j = 0; j < alphabet_size - codes_start; ++j) {
      code_lens[codes[j].code_ - codes_start] = codes[j].len_;
    }
    uint8_t* d_code_lens = nullptr;
    gpuErrchk(cudaMalloc(&d_code_lens, code_lens.size() * sizeof(uint8_t)));
    gpuErrchk(cudaMemcpy(d_code_lens, code_lens.data(),
                         code_lens.size() * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    if (use_unified_memory) {
      gpuErrchk(cudaMallocManaged(&d_temp_storage, temp_storage_bytes));
    } else {
      gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }
    TypeParam* d_data = nullptr;
    if (use_unified_memory) {
      gpuErrchk(cudaMallocManaged(&d_data, data_size * sizeof(TypeParam)));
    } else {
      gpuErrchk(cudaMalloc(&d_data, data_size * sizeof(TypeParam)));
    }
    gpuErrchk(cudaMemcpy(d_data, data.data(), data_size * sizeof(TypeParam),
                         cudaMemcpyHostToDevice));
    auto data_should = data;

    uint8_t const num_levels = ceilLog2Host(alphabet_size);
    for (uint8_t j = codes.back().len_ - 1; j < num_levels; j++) {
      auto pred = isLongEnough<TypeParam>(d_code_lens, j, codes_start);
      auto const new_size = WaveletTreeTest<TypeParam>::runDeviceSelectIf(
          d_temp_storage, temp_storage_bytes, d_data, data_size, pred);
      gpuErrchk(cudaMemcpy(data.data(), d_data, new_size * sizeof(TypeParam),
                           cudaMemcpyDeviceToHost));
      auto new_end =
          std::remove_if(std::execution::par, data_should.begin(),
                         data_should.begin() + data_size, [&](TypeParam c) {
                           if (c < codes_start) {
                             return false;
                           }
                           assert(code_lens[c - codes_start] > 0);
                           return code_lens[c - codes_start] <= j + 1;
                         });
      EXPECT_EQ(
          static_cast<size_t>(std::distance(data_should.begin(), new_end)),
          new_size);
#pragma omp parallel for
      for (size_t k = 0; k < new_size; ++k) {
        EXPECT_EQ(data[k], data_should[k]);
      }
      data_size = new_size;
    }
    gpuErrchk(cudaFree(d_data));
    gpuErrchk(cudaFree(d_temp_storage));
    gpuErrchk(cudaFree(d_code_lens));
  }
}
}  // namespace ecl