#include <gtest/gtest.h>

#include "rank_select.cuh"

namespace ecl {

template <typename T>
class RankSelectTest : public ::testing::Test {
 protected:
  T *result;
  void SetUp() override { gpuErrchk(cudaMallocManaged(&result, sizeof(T))); }
  void TearDown() override { gpuErrchk(cudaFree(result)); }
};

__global__ void writeL1IndexKernel(RankSelect rank_select, uint32_t array_index,
                                   size_t index,
                                   RankSelectConfig::L1_TYPE value) {
  rank_select.writeL1Index(array_index, index, value);
}

__global__ void writeL2IndexKernel(RankSelect rank_select, uint32_t array_index,
                                   size_t index,
                                   RankSelectConfig::L2_TYPE value) {
  rank_select.writeL2Index(array_index, index, value);
}

__global__ void getL1EntryKernel(RankSelect rank_select, uint32_t array_index,
                                 size_t index, size_t *output) {
  *output = rank_select.getL1Entry(array_index, index);
}

__global__ void getL2EntryKernel(RankSelect rank_select, uint32_t array_index,
                                 size_t index, size_t *output) {
  *output = rank_select.getL2Entry(array_index, index);
}

using RankSelectBoolTest = RankSelectTest<bool>;
TEST_F(RankSelectBoolTest, RankSelectConstructor) {
  std::vector<size_t> sizes;
  uint32_t num_arrays = 10;
  for (int i = 0; i < num_arrays; ++i) {
    size_t rand_1 = rand() % 1000;
    size_t rand_2 = rand() % 32;
    sizes.push_back(8 * sizeof(uint32_t) * rand_1 + rand_2);
  }
  BitArray bit_array(sizes);
  RankSelect rank_select(std::move(bit_array));
}

using RankSelectBlocksTest = RankSelectTest<size_t>;
TEST_F(RankSelectBlocksTest, RankSelectIndexSizes) {
  BitArray bit_array(std::vector<size_t>{
      1, RankSelectConfig::L2_BIT_SIZE - 1, RankSelectConfig::L2_BIT_SIZE + 1,
      2 * RankSelectConfig::L2_BIT_SIZE - 1, RankSelectConfig::L1_BIT_SIZE - 1,
      RankSelectConfig::L1_BIT_SIZE + 1,
      2 * RankSelectConfig::L1_BIT_SIZE - 1});
  RankSelect rank_select(std::move(bit_array));

  for (uint32_t i = 0; i < 5; ++i) {
    size_t out = rank_select.getNumL1BlocksHost(i);
    EXPECT_EQ(out, 1);
  }
  for (uint32_t i = 5; i < 7; ++i) {
    size_t out = rank_select.getNumL1BlocksHost(i);
    EXPECT_EQ(out, 2);
  }
  for (uint32_t i = 0; i < 2; ++i) {
    size_t num_l2 = rank_select.getNumL2BlocksHost(i);
    EXPECT_EQ(num_l2, 1);
  }
  for (uint32_t i = 2; i < 4; ++i) {
    size_t num_l2 = rank_select.getNumL2BlocksHost(i);
    EXPECT_EQ(num_l2, 2);
  }
  size_t num_l2 = rank_select.getNumL2BlocksHost(4);
  EXPECT_EQ(num_l2, RankSelectConfig::NUM_L2_PER_L1);

  num_l2 = rank_select.getNumL2BlocksHost(5);
  EXPECT_EQ(num_l2, RankSelectConfig::NUM_L2_PER_L1 + 1);

  num_l2 = rank_select.getNumL2BlocksHost(6);
  EXPECT_EQ(num_l2, 2 * RankSelectConfig::NUM_L2_PER_L1);
}

TEST_F(RankSelectBlocksTest, RankSelectIndexWriting) {
  BitArray bit_array(std::vector<size_t>{
      1, RankSelectConfig::L2_BIT_SIZE - 1, RankSelectConfig::L2_BIT_SIZE + 1,
      2 * RankSelectConfig::L2_BIT_SIZE - 1, RankSelectConfig::L1_BIT_SIZE - 1,
      RankSelectConfig::L1_BIT_SIZE + 1,
      2 * RankSelectConfig::L1_BIT_SIZE - 1});
  RankSelect rank_select(std::move(bit_array));

  // Check that all entries of all arrays are initialized to 0
  for (uint32_t i = 0; i < 7; ++i) {
    size_t num_l1_blocks = rank_select.getNumL1BlocksHost(i);
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
    size_t num_l2 = rank_select.getNumL2BlocksHost(i);
    for (uint32_t j = 0; j < num_l2; ++j) {
      getL2EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
  }

  // set each entry to 1 and check that it is set
  for (uint32_t i = 0; i < 7; ++i) {
    size_t num_l1_blocks = rank_select.getNumL1BlocksHost(i);
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      writeL1IndexKernel<<<1, 1>>>(rank_select, i, j, 1);
      kernelCheck();
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 1);
    }
    size_t num_l2 = rank_select.getNumL2BlocksHost(i);
    for (uint32_t j = 0; j < num_l2; ++j) {
      writeL2IndexKernel<<<1, 1>>>(rank_select, i, j, 1);
      kernelCheck();
      getL2EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 1);
    }
  }
}

TEST_F(RankSelectBlocksTest, RankSelectIndicesContent) {
  {
    BitArray bit_array(std::vector<size_t>{RankSelectConfig::L1_BIT_SIZE + 1},
                       false);

    RankSelect rank_select = createRankSelectStructures(std::move(bit_array));

    // check that all indices are 0
    // for (uint32_t i = 0; i < rank_select.bit_array_.numArrays(); ++i) {
    size_t num_l1_blocks = rank_select.getNumL1BlocksHost(0);
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, 0, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
    size_t num_l2 = rank_select.getNumL2BlocksHost(0);
    for (uint32_t j = 0; j < num_l2; ++j) {
      getL2EntryKernel<<<1, 1>>>(rank_select, 0, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
    //}
  }
  BitArray bit_array(std::vector<size_t>{1, RankSelectConfig::L2_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L2_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L1_BIT_SIZE - 1,
                                         2 * RankSelectConfig::L1_BIT_SIZE + 1},
                     true);

  RankSelect rank_select(createRankSelectStructures(std::move(bit_array)));

  // check that all indices are correct
  for (uint32_t i = 0; i < rank_select.bit_array_.numArrays(); ++i) {
    size_t num_l1_blocks = rank_select.getNumL1BlocksHost(i);
    size_t num_ones = 0;
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, num_ones);
      num_ones += RankSelectConfig::L1_BIT_SIZE;
    }
    size_t num_l2 = rank_select.getNumL2BlocksHost(i);
    num_ones = 0;
    for (uint32_t j = 0; j < num_l2; ++j) {
      if (j % RankSelectConfig::NUM_L2_PER_L1 == 0) {
        num_ones = 0;
      }
      getL2EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, num_ones);
      num_ones += RankSelectConfig::L2_BIT_SIZE;
    }
  }
}
}  // namespace ecl