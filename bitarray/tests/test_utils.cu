#include <gtest/gtest.h>

#include "utils.cuh"

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
  *result = getPrevPowTwo(n);
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
    if (isPowTwo(i)) {
      expected = i;
    }
  }
}
}  // namespace ecl