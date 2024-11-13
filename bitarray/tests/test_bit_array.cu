#include <gtest/gtest.h>

#include "bit_array.cuh"

namespace ecl {

template <typename T>
class BitArrayTest : public ::testing::Test {
 protected:
  T* result;
  void SetUp() override { gpuErrchk(cudaMallocManaged(&result, sizeof(T))); }
  void TearDown() override { gpuErrchk(cudaFree(result)); }
};

__global__ void accessKernel(BitArray bit_array, size_t index, bool* output) {
  *output = bit_array.access(index);
}

__global__ void writeWordKernel(BitArray bit_array, size_t index,
                                uint32_t value) {
  bit_array.write_word(index, value);
}

__global__ void wordKernel(BitArray bit_array, size_t index, uint32_t* output) {
  *output = bit_array.word(index);
}

using BitArrayBoolTest = BitArrayTest<bool>;
// Test the constructor that initializes with a specific size
TEST_F(BitArrayBoolTest, ConstructorWithSize) {
  BitArray bit_array(64);
  EXPECT_EQ(bit_array.size(), 64);
  // Additional checks could go here if necessary
}

// Test the constructor that initializes with a specific size and initial value
TEST_F(BitArrayBoolTest, ConstructorWithSizeAndInitValue) {
  BitArray bit_array(64, true);
  for (size_t i = 0; i < 64; ++i) {
    accessKernel<<<1, 1>>>(bit_array, i, result);
    cudaDeviceSynchronize();
    EXPECT_TRUE(*result);
  }
  BitArray bit_array_false(64, false);
  for (size_t i = 0; i < 64; ++i) {
    accessKernel<<<1, 1>>>(bit_array_false, i, result);
    cudaDeviceSynchronize();
    EXPECT_FALSE(*result);
  }
}
// Test the access method to read specific bits
TEST_F(BitArrayBoolTest, AccessAndWriteBits) {
  BitArray bit_array(64, false);
  accessKernel<<<1, 1>>>(bit_array, 10, result);
  cudaDeviceSynchronize();
  EXPECT_FALSE(*result);

  writeWordKernel<<<1, 1>>>(bit_array, 0, 1UL << (31 - 10));  // Set 10-th bit
  accessKernel<<<1, 1>>>(bit_array, 10, result);
  cudaDeviceSynchronize();
  EXPECT_TRUE(*result);

  accessKernel<<<1, 1>>>(bit_array, 9,
                         result);  // Ensure other bits are unchanged
  cudaDeviceSynchronize();
  EXPECT_FALSE(*result);

  writeWordKernel<<<1, 1>>>(bit_array, 35, 1UL << (31 - 10));  // Set bit 42
  accessKernel<<<1, 1>>>(bit_array, 42, result);
  cudaDeviceSynchronize();
  EXPECT_TRUE(*result);

  writeWordKernel<<<1, 1>>>(bit_array, 35, 0);
  accessKernel<<<1, 1>>>(bit_array, 42, result);
  cudaDeviceSynchronize();
  EXPECT_FALSE(*result);

  writeWordKernel<<<1, 1>>>(bit_array, 0, 0b01000000000000000000000000000000);
  accessKernel<<<1, 1>>>(bit_array, 0, result);
  cudaDeviceSynchronize();
  EXPECT_FALSE(*result);

  accessKernel<<<1, 1>>>(bit_array, 1, result);
  cudaDeviceSynchronize();
  EXPECT_TRUE(*result);

  accessKernel<<<1, 1>>>(bit_array, 2, result);
  cudaDeviceSynchronize();
  EXPECT_FALSE(*result);
}

using BitArrayWordTest = BitArrayTest<uint32_t>;
// Test the word method
TEST_F(BitArrayWordTest, Word) {
  BitArray bit_array(64, false);
  writeWordKernel<<<1, 1>>>(bit_array, 0, (1UL << 10) - 9);
  cudaDeviceSynchronize();
  wordKernel<<<1, 1>>>(bit_array, 0, result);
  cudaDeviceSynchronize();
  EXPECT_EQ(*result, (1UL << 10) - 9);
}
}  // namespace ecl
