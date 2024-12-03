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

__global__ void writeWordAtBitKernel(BitArray bit_array, size_t array_index,
                                     size_t index, uint32_t value) {
  bit_array.writeWordAtBit(array_index, index, value);
}

__global__ void wordAtBitKernel(BitArray bit_array, size_t array_index,
                                size_t index, uint32_t* output) {
  *output = bit_array.wordAtBit(array_index, index);
}

__global__ void partialWordKernel(BitArray bit_array, size_t array_index,
                                  size_t index, uint8_t bit_index,
                                  uint32_t* output) {
  *output = bit_array.partialWord(array_index, index, bit_index);
}

__global__ void twoWordsKernel(BitArray bit_array, size_t array_index,
                               size_t index, uint64_t* output) {
  *output = bit_array.twoWords(array_index, index);
}

using BitArrayBoolTest = BitArrayTest<bool>;
// Test the constructor that initializes with a specific size
TEST_F(BitArrayBoolTest, ConstructorWithSize) {
  BitArray bit_array(std::vector<size_t>{64});
  EXPECT_EQ(bit_array.sizeHost(0), 64);
  // Additional checks could go here if necessary
  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes);
  for (size_t i = 0; i < sizes.size(); ++i) {
    EXPECT_EQ(new_bit_array.sizeHost(i), sizes[i]);
  }
}

// Test the constructor that initializes with a specific size and initial
// value
TEST_F(BitArrayBoolTest, ConstructorWithSizeAndInitValue) {
  BitArray bit_array(std::vector<size_t>{64}, true);
  for (size_t i = 0; i < 64; ++i) {
    accessKernel<<<1, 1>>>(bit_array, 0, i, result);
    kernelCheck();
    EXPECT_TRUE(*result);
  }
  BitArray bit_array_false(std::vector<size_t>{64}, false);
  for (size_t i = 0; i < 64; ++i) {
    accessKernel<<<1, 1>>>(bit_array_false, 0, i, result);
    kernelCheck();
    EXPECT_FALSE(*result);
  }

  // test for multiple bit arrays
  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes, true);
  for (size_t i = 0; i < sizes.size(); ++i) {
    for (size_t j = 0; j < sizes[i]; ++j) {
      accessKernel<<<1, 1>>>(new_bit_array, i, j, result);
      kernelCheck();
      EXPECT_TRUE(*result);
    }
  }
  BitArray new_bit_array_false(sizes, false);
  for (size_t i = 0; i < sizes.size(); ++i) {
    for (size_t j = 0; j < sizes[i]; ++j) {
      accessKernel<<<1, 1>>>(new_bit_array_false, i, j, result);
      kernelCheck();
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
  // for each word in the bit array, set a random bit, and check that
  // accessing it works.
  for (int i = 0; i < num_words; ++i) {
    uint8_t bit;
    if (i == num_words - 1) {
      bit = rand() % bits_in_last_word;
    } else {
      bit = rand() % 32;
    }
    uint32_t word = 1UL << bit;
    writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 32 * i, word);
    kernelCheck();
    accessKernel<<<1, 1>>>(bit_array, 0, 32 * i + bit, result);
    kernelCheck();
    EXPECT_TRUE(*result);
    // check that all other bits are unchanged
    for (int j = 32 * i; j < 32 * (i + 1); ++j) {
      if (j != 32 * i + bit and j < bit_array.sizeHost(0)) {
        accessKernel<<<1, 1>>>(bit_array, 0, j, result);
        kernelCheck();
        EXPECT_FALSE(*result);
      }
    }
  }

  // Test for multiple bit arrays
  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes, false);

  writeWordAtBitKernel<<<1, 1>>>(new_bit_array, 2, 0, 1UL << 1);
  kernelCheck();
  accessKernel<<<1, 1>>>(new_bit_array, 2, 0, result);
  kernelCheck();
  EXPECT_FALSE(*result);
  accessKernel<<<1, 1>>>(new_bit_array, 2, 1, result);
  kernelCheck();
  EXPECT_TRUE(*result);
  // check that all other arrays are unchanged
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i != 2) {
      for (size_t j = 0; j < sizes[i]; ++j) {
        accessKernel<<<1, 1>>>(new_bit_array, i, j, result);
        kernelCheck();
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

  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 0, word);
  kernelCheck();
  wordAtBitKernel<<<1, 1>>>(bit_array, 0, 0, result);
  kernelCheck();
  EXPECT_EQ(*result, word);

  std::vector<size_t> sizes{2, 4, 8, 64, 128, 1024};
  BitArray new_bit_array(sizes, false);
  writeWordAtBitKernel<<<1, 1>>>(new_bit_array, 2, 0, word);
  kernelCheck();
  wordAtBitKernel<<<1, 1>>>(new_bit_array, 2, 0, result);
  kernelCheck();
  EXPECT_EQ(*result, word);

  // check that all other arrays are unchanged
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i != 2) {
      for (size_t j = 0; j < sizes[i]; j += 32) {
        wordAtBitKernel<<<1, 1>>>(new_bit_array, i, j, result);
        kernelCheck();
        EXPECT_EQ(*result, 0);
      }
    }
  }
}

TEST_F(BitArrayWordTest, PartialWord) {
  BitArray bit_array(std::vector<size_t>{64}, false);
  uint32_t word = ~0;

  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 0, word);
  kernelCheck();
  partialWordKernel<<<1, 1>>>(bit_array, 0, 0, 4, result);
  kernelCheck();
  EXPECT_EQ(*result, 0b1111);

  word = 0b1010'1010'1010'1010'1010'1010'1010'1010;
  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 0, word);
  kernelCheck();
  partialWordKernel<<<1, 1>>>(bit_array, 0, 0, 4, result);
  kernelCheck();
  EXPECT_EQ(*result, 0b1010);
  partialWordKernel<<<1, 1>>>(bit_array, 0, 0, 9, result);
  kernelCheck();
  EXPECT_EQ(*result, 0b0'1010'1010);
}

using BitArrayTwoWordsTest = BitArrayTest<uint64_t>;
TEST_F(BitArrayTwoWordsTest, TwoWords) {
  BitArray bit_array(std::vector<size_t>{66'000}, false);
  uint32_t word = ~0;

  for (size_t i = 0; i < 2048; i += 2) {
    twoWordsKernel<<<1, 1>>>(bit_array, 0, i, result);
    kernelCheck();
    EXPECT_EQ(*result, 0);
  }

  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 0, word);
  kernelCheck();
  twoWordsKernel<<<1, 1>>>(bit_array, 0, 0, result);
  kernelCheck();
  EXPECT_EQ(*result, 0xFFFFFFFF);

  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 0, 0);
  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 32, word);
  kernelCheck();
  twoWordsKernel<<<1, 1>>>(bit_array, 0, 0, result);
  kernelCheck();
  EXPECT_EQ(*result, 0xFFFFFFFF'00000000);

  word = 0b1010'1010'1010'1010'1010'1010'1010'1010;
  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 0, word);
  writeWordAtBitKernel<<<1, 1>>>(bit_array, 0, 32, 0);
  kernelCheck();
  twoWordsKernel<<<1, 1>>>(bit_array, 0, 0, result);
  kernelCheck();
  EXPECT_EQ(*result, 0b1010'1010'1010'1010'1010'1010'1010'1010);
}
}  // namespace ecl
