#include <gtest/gtest.h>
#include <omp.h>

#include "ecl_wavelet/utils/test_benchmark_utils.cuh"
#include "ecl_wavelet/utils/utils.cuh"

namespace ecl {

template <typename T>
class UtilsTest : public ::testing::Test {
 protected:
  T *result;
  void SetUp() override { gpuErrchk(cudaMallocManaged(&result, sizeof(T))); }
  void TearDown() override { gpuErrchk(cudaFree(result)); }
};

template <typename T>
__global__ void getPrevPowTwoKernel(T n, T *result) {
  *result = utils::getPrevPowTwo(n);
}

template <typename T>
__global__ void ceilLog2Kernel(T n, T *result) {
  *result = utils::ceilLog2<T>(n);
}

using MyTypes = testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(UtilsTest, MyTypes);

TYPED_TEST(UtilsTest, getPrevPowTwo) {
  TypeParam limit = std::min(
      size_t(1024), static_cast<size_t>(std::numeric_limits<TypeParam>::max()));
  TypeParam expected = 0;
  for (TypeParam i = 0; i < limit; ++i) {
    getPrevPowTwoKernel<TypeParam><<<1, 1>>>(i, this->result);
    kernelCheck();
    EXPECT_EQ(*this->result, expected);
    if (utils::isPowTwo(i)) {
      expected = i;
    }
  }
}

TYPED_TEST(UtilsTest, ceilLog2) {
  TypeParam limit = std::min(
      size_t(1024), static_cast<size_t>(std::numeric_limits<TypeParam>::max()));
  uint32_t expected = 0;
  for (TypeParam i = 1; i < limit; ++i) {
    EXPECT_EQ(utils::ceilLog2<TypeParam>(i), expected);
    ceilLog2Kernel<TypeParam><<<1, 1>>>(i, this->result);
    kernelCheck();
    EXPECT_EQ(*this->result, expected);
    if (utils::isPowTwo(i)) {
      expected++;
    }
  }
}

TYPED_TEST(UtilsTest, RSQueriesGenerator) {
  if constexpr (sizeof(TypeParam) <= 2) {
    size_t constexpr kNumIters = 100;
    size_t const num_queries = omp_get_num_procs();
    auto [alphabet_sizes, data_sizes] =
        utils::generateRandomAlphabetAndDataSizes<TypeParam, true>(
            1000, static_cast<size_t>(2e9) / sizeof(TypeParam), kNumIters);
    std::vector<RankSelectQuery<TypeParam>> rank_queries(num_queries);
    std::vector<RankSelectQuery<TypeParam>> select_queries(num_queries);
    std::vector<size_t> rank_results(num_queries);
    std::vector<size_t> select_results(num_queries);
    for (size_t i = 0; i < kNumIters; ++i) {
      size_t const data_size = data_sizes[i];
      size_t const alphabet_size = alphabet_sizes[i];
      auto alphabet = utils::generateRandomAlphabet<TypeParam>(alphabet_size);
      auto data = utils::generateRandomDataAndRSQueries<TypeParam>(
          alphabet, data_size, num_queries, rank_queries, select_queries,
          rank_results, select_results);
#pragma omp parallel for
      for (size_t j = 0; j < num_queries; ++j) {
        auto rank_result = rank_results[j];
        auto select_result = select_results[j];
        auto rank_should =
            std::count(data.begin(), data.begin() + rank_queries[j].index_,
                       rank_queries[j].symbol_);
        size_t counts = 0;
        auto select_should =
            std::find_if(data.begin(), data.end(),
                         [&](TypeParam c) {
                           return c == select_queries[j].symbol_ and
                                  ++counts == select_queries[j].index_;
                         }) -
            data.begin();
        EXPECT_EQ(rank_result, rank_should);
        EXPECT_EQ(select_result, select_should);
      }
    }
  }
}
}  // namespace ecl