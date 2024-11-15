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

__global__ void accessKernel(BitArray bit_array, size_t array_index,
                             size_t index, bool* output) {
  *output = bit_array.access(array_index, index);
}

__global__ void writeWordKernel(BitArray bit_array, size_t array_index,
                                size_t index, uint32_t value) {
  bit_array.write_word(array_index, index, value);
}

__global__ void wordKernel(BitArray bit_array, size_t array_index, size_t index,
                           uint32_t* output) {
  *output = bit_array.word(array_index, index);
}

using BitArrayBoolTest = BitArrayTest<bool>;
// Test the constructor that initializes with a specific size
TEST_F(BitArrayBoolTest, ConstructorWithSize) {
  BitArray bit_array(std::vector<size_t>{64});
  EXPECT_EQ(bit_array.size(0), 64);
  // Additional checks could go here if necessary
  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes);
  for (size_t i = 0; i < sizes.size(); ++i) {
    EXPECT_EQ(new_bit_array.size(i), sizes[i]);
  }
}

// Test the constructor that initializes with a specific size and initial value
TEST_F(BitArrayBoolTest, ConstructorWithSizeAndInitValue) {
  BitArray bit_array(std::vector<size_t>{64}, true);
  for (size_t i = 0; i < 64; ++i) {
    accessKernel<<<1, 1>>>(bit_array, 0, i, result);
    cudaDeviceSynchronize();
    EXPECT_TRUE(*result);
  }
  BitArray bit_array_false(std::vector<size_t>{64}, false);
  for (size_t i = 0; i < 64; ++i) {
    accessKernel<<<1, 1>>>(bit_array_false, 0, i, result);
    cudaDeviceSynchronize();
    EXPECT_FALSE(*result);
  }

  // test for multiple bit arrays
  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes, true);
  for (size_t i = 0; i < sizes.size(); ++i) {
    for (size_t j = 0; j < sizes[i]; ++j) {
      accessKernel<<<1, 1>>>(new_bit_array, i, j, result);
      cudaDeviceSynchronize();
      EXPECT_TRUE(*result);
    }
  }
  BitArray new_bit_array_false(sizes, false);
  for (size_t i = 0; i < sizes.size(); ++i) {
    for (size_t j = 0; j < sizes[i]; ++j) {
      accessKernel<<<1, 1>>>(new_bit_array_false, i, j, result);
      cudaDeviceSynchronize();
      EXPECT_FALSE(*result);
    }
  }
}

// Test the access method to read specific bits
TEST_F(BitArrayBoolTest, AccessAndWriteBits) {
  size_t num_words = 11;
  uint8_t bits_in_last_word = 5;
  BitArray bit_array(
      std::vector<size_t>{32 * (num_words - 1) + bits_in_last_word}, false);
  // for each word in the bit array, set a random bit, and check that accessing
  // it works.
  for (int i = 0; i < num_words; ++i) {
    uint8_t bit;
    if (i == num_words - 1) {
      bit = rand() % bits_in_last_word;
    } else {
      bit = rand() % 32;
    }
    uint32_t word = 1UL << (31 - bit);
    writeWordKernel<<<1, 1>>>(bit_array, 0, 32 * i, word);
    cudaDeviceSynchronize();
    accessKernel<<<1, 1>>>(bit_array, 0, 32 * i + bit, result);
    cudaDeviceSynchronize();
    EXPECT_TRUE(*result);
    // check that all other bits are unchanged
    for (int j = 32 * i; j < 32 * (i + 1); ++j) {
      if (j != 32 * i + bit && j < bit_array.size(0)) {
        accessKernel<<<1, 1>>>(bit_array, 0, j, result);
        cudaDeviceSynchronize();
        EXPECT_FALSE(*result);
      }
    }
  }

  // Test for multiple bit arrays
  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes, false);

  writeWordKernel<<<1, 1>>>(new_bit_array, 2, 0, 1UL << 30);
  accessKernel<<<1, 1>>>(new_bit_array, 2, 0, result);
  cudaDeviceSynchronize();
  EXPECT_FALSE(*result);
  accessKernel<<<1, 1>>>(new_bit_array, 2, 1, result);
  cudaDeviceSynchronize();
  EXPECT_TRUE(*result);
  // check that all other arrays are unchanged
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i != 2) {
      for (size_t j = 0; j < sizes[i]; ++j) {
        accessKernel<<<1, 1>>>(new_bit_array, i, j, result);
        cudaDeviceSynchronize();
        EXPECT_FALSE(*result);
      }
    }
  }
}

using BitArrayWordTest = BitArrayTest<uint32_t>;
// Test the word method
TEST_F(BitArrayWordTest, Word) {
  BitArray bit_array(std::vector<size_t>{64}, false);
  uint32_t word = (1UL << 10) - 9;

  writeWordKernel<<<1, 1>>>(bit_array, 0, 0, word);
  cudaDeviceSynchronize();
  wordKernel<<<1, 1>>>(bit_array, 0, 0, result);
  cudaDeviceSynchronize();
  EXPECT_EQ(*result, word);

  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes, false);
  writeWordKernel<<<1, 1>>>(new_bit_array, 2, 0, word);
  cudaDeviceSynchronize();
  wordKernel<<<1, 1>>>(new_bit_array, 2, 0, result);
  cudaDeviceSynchronize();
  EXPECT_EQ(*result, word);

  // check that all other arrays are unchanged
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i != 2) {
      for (size_t j = 0; j < sizes[i]; j += 32) {
        wordKernel<<<1, 1>>>(new_bit_array, i, j, result);
        cudaDeviceSynchronize();
        EXPECT_EQ(*result, 0);
      }
    }
  }
}
}  // namespace ecl
