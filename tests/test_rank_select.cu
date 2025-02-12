#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <vector>

#include "rank_select.cuh"
#include "test_benchmark_utils.cuh"

namespace ecl {

template <typename T>
class RankSelectTest : public ::testing::Test {
 protected:
  T *result;
  void SetUp() override { gpuErrchk(cudaMallocManaged(&result, sizeof(T))); }
  void TearDown() override { gpuErrchk(cudaFree(result)); }
};

__global__ void writeL1IndexKernel(RankSelect rank_select, uint32_t array_index,
                                   size_t index, RSConfig::L1_TYPE value) {
  rank_select.writeL1Index(array_index, index, value);
}

__global__ void writeL2IndexKernel(RankSelect rank_select, uint32_t array_index,
                                   size_t index, RSConfig::L2_TYPE value) {
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

__global__ void getNumLastL2BlocksKernel(RankSelect rank_select,
                                         uint32_t array_index, size_t *output) {
  *output = rank_select.getNumLastL2Blocks(array_index);
}

__global__ void writeWordKernel(BitArray bit_array, size_t array_index,
                                size_t index, uint32_t value) {
  bit_array.writeWord(array_index, index, value);
}

template <int NumThreads>
__global__ void rank0Kernel(RankSelect rank_select, uint32_t array_index,
                            size_t index, size_t *output) {
  assert(blockDim.x == NumThreads);
  RankResult result = rank_select.rank<NumThreads, true, 0>(
      array_index, index, rank_select.bit_array_.getOffset(array_index));
  *output = result.rank;
}

template <int NumThreads>
__global__ void rank1Kernel(RankSelect rank_select, uint32_t array_index,
                            size_t index, size_t *output) {
  assert(blockDim.x == NumThreads);
  RankResult result = rank_select.rank<NumThreads, true, 1>(
      array_index, index, rank_select.bit_array_.getOffset(array_index));
  *output = result.rank;
}

template <int NumThreads>
__global__ void select0Kernel(RankSelect rank_select, uint32_t array_index,
                              size_t index, size_t *output) {
  *output = rank_select.select<0, NumThreads>(
      array_index, index, rank_select.bit_array_.getOffset(array_index));
}

template <int NumThreads>
__global__ void select1Kernel(RankSelect rank_select, uint32_t array_index,
                              size_t index, size_t *output) {
  *output = rank_select.select<1, NumThreads>(
      array_index, index, rank_select.bit_array_.getOffset(array_index));
}

__global__ void getNumL1BlocksKernel(RankSelect rank_select,
                                     uint32_t array_index, size_t *output) {
  *output = rank_select.getNumL1Blocks(array_index);
}

__global__ void getNumL2BlocksKernel(RankSelect rank_select,
                                     uint32_t array_index, size_t *output) {
  *output = rank_select.getNumL2Blocks(array_index);
}

__global__ void getSelectSampleKernel(RankSelect rank_select,
                                      uint32_t array_index, size_t index,
                                      bool val, size_t *output) {
  if (val) {
    *output = rank_select.getSelectSample<1>(array_index, index);
  } else {
    *output = rank_select.getSelectSample<0>(array_index, index);
  }
}

__global__ void getTotalNumValsKernel(RankSelect rank_select,
                                      uint32_t array_index, size_t *output,
                                      bool const val) {
  if (val) {
    *output = rank_select.getTotalNumVals<1>(array_index);
  } else {
    *output = rank_select.getTotalNumVals<0>(array_index);
  }
}

class RankSelectHelper {
 private:
  std::vector<bool> bit_vector_;
  std::vector<size_t> rank_index1_;  // Precomputed rank data for efficiency
  std::vector<size_t> rank_index0_;  // Precomputed rank data for efficiency

 public:
  // Constructor
  RankSelectHelper(size_t const size) : bit_vector_(size) {}

  size_t rank0(size_t index) const {
    assert(index < bit_vector_.size());
    return rank_index0_[index];
  }
  // Rank function: returns the number of 1s up to the given index (exclusive)
  size_t rank1(size_t index) const {
    assert(index < bit_vector_.size());
    return rank_index1_[index];
  }

  // Helper function to build rank prefix sums
  void buildRankIndex() {
    rank_index0_.resize(bit_vector_.size() + 1);
    rank_index0_[0] = 0;
    for (size_t i = 0; i < bit_vector_.size(); ++i) {
      rank_index0_[i + 1] = rank_index0_[i] + (bit_vector_[i] ? 0 : 1);
    }
    rank_index1_.resize(bit_vector_.size() + 1);
    rank_index1_[0] = 0;
    for (size_t i = 0; i < bit_vector_.size(); ++i) {
      rank_index1_[i + 1] = rank_index1_[i] + (bit_vector_[i] ? 1 : 0);
    }
  }

  size_t select0(size_t i) const {
    // Search for first element x such that i == x
    auto it = std::find(rank_index0_.begin(), rank_index0_.end(), i);
    if (it == rank_index0_.end()) {
      return bit_vector_.size();
    }
    auto index = it - rank_index0_.begin() - 1;
    return index;
  }

  // Select function: returns the index of the i-th 1, starting from 1.
  size_t select1(size_t i) const {
    // Search for first element x such that i == x
    auto it = std::find(rank_index1_.begin(), rank_index1_.end(), i);
    if (it == rank_index1_.end()) {
      return bit_vector_.size();
    }
    auto index = it - rank_index1_.begin() - 1;
    return index;
  }

  void writeWord(size_t index, uint32_t value) {
    size_t start_bit = index * (sizeof(uint32_t) * 8);
    for (size_t i = 0; i < (sizeof(uint32_t) * 8); ++i) {
      bit_vector_[start_bit + i] = (value >> i) & 1UL;
    }
  }
  bool operator[](size_t index) const { return bit_vector_[index]; }
};

using RankSelectBoolTest = RankSelectTest<bool>;
TEST_F(RankSelectBoolTest, RankSelectConstructor) {
  std::vector<size_t> sizes;
  uint32_t num_arrays = 10;
  for (int i = 0; i < num_arrays; ++i) {
    size_t rand_1 = rand() % 1000;
    size_t rand_2 = rand() % (sizeof(uint32_t) * 8);
    sizes.push_back(8 * sizeof(uint32_t) * rand_1 + rand_2);
  }
  BitArray bit_array(sizes, false);
  RankSelect rank_select(std::move(bit_array), 0);
}

using RankSelectBlocksTest = RankSelectTest<size_t>;
TEST_F(RankSelectBlocksTest, RankSelectIndexSizes) {
  BitArray bit_array(
      std::vector<size_t>{
          1, RSConfig::L2_BIT_SIZE - 1, RSConfig::L2_BIT_SIZE + 1,
          2 * RSConfig::L2_BIT_SIZE - 1, RSConfig::L1_BIT_SIZE - 1,
          RSConfig::L1_BIT_SIZE + 1, 2 * RSConfig::L1_BIT_SIZE - 1},
      false);
  RankSelect rank_select(std::move(bit_array), 0);

  for (uint32_t i = 0; i < 5; ++i) {
    getNumL1BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    EXPECT_EQ(*result, 1);
  }
  for (uint32_t i = 5; i < 7; ++i) {
    getNumL1BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    EXPECT_EQ(*result, 2);
  }
  for (uint32_t i = 0; i < 2; ++i) {
    getNumL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    EXPECT_EQ(*result, 1);
  }
  for (uint32_t i = 2; i < 4; ++i) {
    getNumL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    EXPECT_EQ(*result, 2);
  }
  for (uint32_t i = 0; i < 2; ++i) {
    getNumLastL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    EXPECT_EQ(*result, 1);
  }
  for (uint32_t i = 2; i < 4; ++i) {
    getNumLastL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    EXPECT_EQ(*result, 2);
  }
  getNumLastL2BlocksKernel<<<1, 1>>>(rank_select, 4, result);
  kernelCheck();
  EXPECT_EQ(*result, RSConfig::NUM_L2_PER_L1);

  getNumLastL2BlocksKernel<<<1, 1>>>(rank_select, 5, result);
  kernelCheck();
  EXPECT_EQ(*result, 1);

  getNumLastL2BlocksKernel<<<1, 1>>>(rank_select, 6, result);
  kernelCheck();
  EXPECT_EQ(*result, RSConfig::NUM_L2_PER_L1);

  getNumL2BlocksKernel<<<1, 1>>>(rank_select, 4, result);
  kernelCheck();
  EXPECT_EQ(*result, RSConfig::NUM_L2_PER_L1);

  getNumL2BlocksKernel<<<1, 1>>>(rank_select, 5, result);
  kernelCheck();
  EXPECT_EQ(*result, RSConfig::NUM_L2_PER_L1 + 1);

  getNumL2BlocksKernel<<<1, 1>>>(rank_select, 6, result);
  kernelCheck();
  EXPECT_EQ(*result, 2 * RSConfig::NUM_L2_PER_L1);
}

TEST_F(RankSelectBlocksTest, RankSelectIndexWriting) {
  BitArray bit_array(
      std::vector<size_t>{
          1, RSConfig::L2_BIT_SIZE - 1, RSConfig::L2_BIT_SIZE + 1,
          2 * RSConfig::L2_BIT_SIZE - 1, RSConfig::L1_BIT_SIZE - 1,
          RSConfig::L1_BIT_SIZE + 1, 2 * RSConfig::L1_BIT_SIZE - 1},
      false);
  RankSelect rank_select(std::move(bit_array), 0);

  // Check that all entries of all arrays are initialized to 0
  for (uint32_t i = 0; i < 7; ++i) {
    getNumL1BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l1_blocks = *result;
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
    getNumL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l2 = *result;
    for (uint32_t j = 0; j < num_l2; ++j) {
      getL2EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
  }

  // set each entry to 1 and check that it is set
  for (uint32_t i = 0; i < 7; ++i) {
    getNumL1BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l1_blocks = *result;
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      writeL1IndexKernel<<<1, 1>>>(rank_select, i, j, 1);
      kernelCheck();
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 1);
    }
    getNumL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l2 = *result;
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
    BitArray bit_array(std::vector<size_t>{RSConfig::L1_BIT_SIZE + 1}, false);

    RankSelect rank_select(std::move(bit_array), 0);

    // check that all indices are 0
    getNumL1BlocksKernel<<<1, 1>>>(rank_select, 0, result);
    kernelCheck();
    uint32_t const num_l1_blocks = *result;
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, 0, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
    getNumL2BlocksKernel<<<1, 1>>>(rank_select, 0, result);
    kernelCheck();
    uint32_t const num_l2 = *result;
    for (uint32_t j = 0; j < num_l2; ++j) {
      getL2EntryKernel<<<1, 1>>>(rank_select, 0, j, result);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }
  }
  BitArray bit_array(
      std::vector<size_t>{
          1, RSConfig::L2_BIT_SIZE + 1, 2 * RSConfig::L2_BIT_SIZE - 1,
          RSConfig::L1_BIT_SIZE - 1, RSConfig::L1_BIT_SIZE + 1,
          2 * RSConfig::L1_BIT_SIZE - 1, 2 * RSConfig::L1_BIT_SIZE + 1},
      true);

  RankSelect rank_select(std::move(bit_array), 0);

  // check that all indices are correct
  for (uint32_t i = 0; i < rank_select.bit_array_.numArrays(); ++i) {
    getNumL1BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l1_blocks = *result;
    size_t num_ones = 0;
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, num_ones);
      num_ones += RSConfig::L1_BIT_SIZE;
    }
    getNumL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l2 = *result;
    num_ones = 0;
    for (uint32_t j = 0; j < num_l2; ++j) {
      if (j % RSConfig::NUM_L2_PER_L1 == 0) {
        num_ones = 0;
      }
      getL2EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, num_ones);
      num_ones += RSConfig::L2_BIT_SIZE;
    }
  }
}

TEST_F(RankSelectBlocksTest, RankSelectSamplesContent) {
  std::vector<size_t> BA_sizes{
      1,
      RSConfig::SELECT_SAMPLE_RATE - 1,
      RSConfig::SELECT_SAMPLE_RATE + 1,
      2 * RSConfig::SELECT_SAMPLE_RATE - 1,
      2 * RSConfig::SELECT_SAMPLE_RATE + 1,
      6 * RSConfig::L2_BIT_SIZE + RSConfig::L2_BIT_SIZE / 5};
  for (bool const val : {true, false}) {
    BitArray bit_array(BA_sizes, val);
    RankSelect rank_select(std::move(bit_array), 0);

    for (uint32_t i = 0; i < BA_sizes.size(); ++i) {
      getTotalNumValsKernel<<<1, 1>>>(rank_select, i, result, val);
      kernelCheck();
      EXPECT_EQ(*result, rank_select.bit_array_.sizeHost(i));
      getTotalNumValsKernel<<<1, 1>>>(rank_select, i, result, !val);
      kernelCheck();
      EXPECT_EQ(*result, 0);
    }

    for (uint32_t i = 0; i < BA_sizes.size(); ++i) {
      size_t num_samples = BA_sizes[i] / RSConfig::SELECT_SAMPLE_RATE;
      for (size_t j = 1; j <= num_samples; ++j) {
        getSelectSampleKernel<<<1, 1>>>(rank_select, i, j - 1, val, result);
        kernelCheck();
        EXPECT_EQ(*result, j * RSConfig::SELECT_SAMPLE_RATE - 1);
      }
    }
  }
}

TEST_F(RankSelectBlocksTest, RankSelectIndicesRandom) {
  int num_arrays = 5;
  std::vector<size_t> sizes(num_arrays);
  std::vector<RankSelectHelper> helpers;
  // Sizes are random between 2000 and 10^6
  std::vector<uint32_t> random_nums(num_arrays);
  generateRandomNums<uint32_t>(random_nums, 2000, 1e6);

  size_t max_size = 0;
  for (int i = 0; i < num_arrays; ++i) {
    sizes[i] = random_nums[i];
    max_size = std::max(max_size, sizes[i]);
    helpers.emplace_back(random_nums[i]);
  }
  BitArray bit_array(sizes, false);
  // For each array, go through each word and set it to a random number
  random_nums.resize((max_size + (sizeof(uint32_t) * 8 - 1)) /
                     (sizeof(uint32_t) * 8));
  generateRandomNums(random_nums, 0U, std::numeric_limits<uint32_t>::max());

  uint32_t *d_words_arr;
  gpuErrchk(cudaMalloc(&d_words_arr, ((max_size + (sizeof(uint32_t) * 8 - 1)) /
                                      (sizeof(uint32_t) * 8)) *
                                         sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(
      d_words_arr, random_nums.data(),
      ((max_size + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8)) *
          sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  for (uint32_t i = 0; i < num_arrays; ++i) {
    size_t num_words =
        (sizes[i] + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8);
    auto [blocks, threads] =
        getLaunchConfig((num_words + WS - 1) / WS, 256, kMaxTPB);
    writeWordsParallelKernel<<<blocks, threads>>>(bit_array, i, d_words_arr,
                                                  num_words);
    for (uint32_t j = 0; j < num_words; ++j) {
      helpers[i].writeWord(j, random_nums[j]);
    }
    kernelCheck();
  }
  for (auto &helper : helpers) {
    helper.buildRankIndex();
  }
  gpuErrchk(cudaFree(d_words_arr));
  RankSelect rank_select(std::move(bit_array), 0);
  for (uint32_t i = 0; i < num_arrays; ++i) {
    // Check that indices are correct
    getNumL1BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l1_blocks = *result;
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, helpers[i].rank1(j * RSConfig::L1_BIT_SIZE));
    }
    getNumL2BlocksKernel<<<1, 1>>>(rank_select, i, result);
    kernelCheck();
    uint32_t const num_l2 = *result;
    size_t local_l1_block = 0;
    for (uint32_t j = 0; j < num_l2; ++j) {
      if (j % RSConfig::NUM_L2_PER_L1 == 0 and j != 0) {
        local_l1_block++;
      }
      getL2EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      auto result_should =
          helpers[i].rank1(j * RSConfig::L2_BIT_SIZE) -
          helpers[i].rank1(local_l1_block * RSConfig::L1_BIT_SIZE);
      kernelCheck();
      EXPECT_EQ(*result, result_should);
    }
  }
}

TEST_F(RankSelectBlocksTest, RankSelectSamplesRandom) {
  int num_arrays = 10;
  for (int _ = 0; _ < 10; _++) {
    std::vector<size_t> sizes(num_arrays);
    std::vector<RankSelectHelper> helpers;
    // Sizes are random between 200 and 10^6
    std::vector<uint32_t> random_nums(num_arrays);
    generateRandomNums<uint32_t>(random_nums, 200, 1e6);

    size_t max_size = 0;
    for (int i = 0; i < num_arrays; ++i) {
      sizes[i] = random_nums[i];
      max_size = std::max(max_size, sizes[i]);
      helpers.emplace_back(random_nums[i]);
    }
    BitArray bit_array(sizes, false);
    // For each array, go through each word and set it to a random number
    random_nums.resize((max_size + (sizeof(uint32_t) * 8 - 1)) /
                       (sizeof(uint32_t) * 8));
    generateRandomNums(random_nums, 0U, std::numeric_limits<uint32_t>::max());

    uint32_t *d_words_arr;
    gpuErrchk(cudaMalloc(
        &d_words_arr,
        ((max_size + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8)) *
            sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(
        d_words_arr, random_nums.data(),
        ((max_size + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8)) *
            sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    for (uint32_t i = 0; i < num_arrays; ++i) {
      size_t num_words =
          (sizes[i] + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8);
      auto [blocks, threads] =
          getLaunchConfig((num_words + WS - 1) / WS, 256, kMaxTPB);
      writeWordsParallelKernel<<<blocks, threads>>>(bit_array, i, d_words_arr,
                                                    num_words);
      for (uint32_t j = 0; j < num_words; ++j) {
        helpers[i].writeWord(j, random_nums[j]);
      }
      kernelCheck();
    }
    for (auto &helper : helpers) {
      helper.buildRankIndex();
    }
    gpuErrchk(cudaFree(d_words_arr));
    RankSelect rank_select(std::move(bit_array), 0);
    for (uint32_t i = 0; i < num_arrays; ++i) {
      auto const array_size = sizes[i];
      size_t current_sample = 0;
      size_t select0 = 0;
      size_t select1 = 0;
      while (select0 != array_size or select1 != array_size) {
        select0 = helpers[i].select0((current_sample + 1) *
                                     RSConfig::SELECT_SAMPLE_RATE);
        if (select0 != array_size) {
          getSelectSampleKernel<<<1, 1>>>(rank_select, i, current_sample, false,
                                          result);
          kernelCheck();
          EXPECT_EQ(*result, select0);
        }
        select1 = helpers[i].select1((current_sample + 1) *
                                     RSConfig::SELECT_SAMPLE_RATE);
        if (select1 != array_size) {
          getSelectSampleKernel<<<1, 1>>>(rank_select, i, current_sample, true,
                                          result);
          kernelCheck();
          EXPECT_EQ(*result, select1);
        }
        current_sample++;
      }
    }
  }
}

TEST_F(RankSelectBlocksTest, RankSelectOperations) {
  for (auto const &val : {true, false}) {
    BitArray bit_array(
        std::vector<size_t>{
            RSConfig::L2_BIT_SIZE + 1, 2 * RSConfig::L2_BIT_SIZE - 1,
            RSConfig::L1_BIT_SIZE - 1, RSConfig::L1_BIT_SIZE + 1,
            2 * RSConfig::L1_BIT_SIZE - 1, 2 * RSConfig::L1_BIT_SIZE + 1},
        val);
    auto num_arrays = bit_array.numArrays();

    RankSelect rank_select(std::move(bit_array), 0);

    std::vector<size_t> random_positions(100);
    std::default_random_engine generator;
    // for each array, test the rank of 100 random positions
    for (uint32_t i = 0; i < num_arrays; ++i) {
      size_t bit_size = rank_select.bit_array_.sizeHost(i);
      std::random_device rd;
      std::mt19937 gen(rd());  // Random number generator
      std::uniform_int_distribution<> dis(1, bit_size - 1);

      std::generate(random_positions.begin(), random_positions.end(),
                    [&]() { return dis(gen); });

      auto rank_tests = [&]<int NumThreads>() {
        for (auto const &pos : random_positions) {
          if (val == true) {
            rank1Kernel<NumThreads>
                <<<1, NumThreads>>>(rank_select, i, pos, result);
            kernelCheck();
            EXPECT_EQ(*result, pos);
            select1Kernel<NumThreads>
                <<<1, NumThreads>>>(rank_select, i, pos, result);
            kernelCheck();
            EXPECT_EQ(*result, pos - 1);
          } else {
            rank0Kernel<NumThreads>
                <<<1, NumThreads>>>(rank_select, i, pos, result);
            kernelCheck();
            EXPECT_EQ(*result, pos);
            select0Kernel<NumThreads>
                <<<1, NumThreads>>>(rank_select, i, pos, result);
            kernelCheck();
            EXPECT_EQ(*result, pos - 1);
          }
        }
        if (val == true) {
          rank1Kernel<NumThreads><<<1, NumThreads>>>(rank_select, i, 0, result);
          kernelCheck();
          EXPECT_EQ(*result, 0);
          rank1Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, bit_size - 1, result);
          kernelCheck();
          EXPECT_EQ(*result, bit_size - 1);
          select1Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, 1, result);
          kernelCheck();
          EXPECT_EQ(*result, 0);
          select1Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, bit_size, result);
          kernelCheck();
          EXPECT_EQ(*result, bit_size - 1);

          rank0Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, bit_size - 1, result);
          kernelCheck();
          EXPECT_EQ(*result, 0);
          select0Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, 1, result);
          kernelCheck();
          EXPECT_EQ(*result, bit_size);
        } else {
          rank0Kernel<NumThreads><<<1, NumThreads>>>(rank_select, i, 0, result);
          kernelCheck();
          EXPECT_EQ(*result, 0);
          rank0Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, bit_size - 1, result);
          kernelCheck();
          EXPECT_EQ(*result, bit_size - 1);
          select0Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, 1, result);
          kernelCheck();
          EXPECT_EQ(*result, 0);
          select0Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, bit_size, result);
          kernelCheck();
          EXPECT_EQ(*result, bit_size - 1);

          rank1Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, bit_size - 1, result);
          kernelCheck();
          EXPECT_EQ(*result, 0);
          select1Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, 1, result);
          kernelCheck();
          EXPECT_EQ(*result, bit_size);
        }
      };
      rank_tests.operator()<32>();
      rank_tests.operator()<16>();
      rank_tests.operator()<8>();
      rank_tests.operator()<4>();
      rank_tests.operator()<2>();
      rank_tests.operator()<1>();
    }
  }

  // test with the top half of the bits in each word set to 1
  uint32_t word = 0xFFFF0000;

  auto sizes = std::vector<size_t>{
      RSConfig::L2_BIT_SIZE + 1,     2 * RSConfig::L2_BIT_SIZE - 1,
      RSConfig::L1_BIT_SIZE - 1,     RSConfig::L1_BIT_SIZE + 1,
      2 * RSConfig::L1_BIT_SIZE - 1, 2 * RSConfig::L1_BIT_SIZE + 1};
  BitArray bit_array(sizes, false);
  auto num_arrays = bit_array.numArrays();

  std::vector<RankSelectHelper> helpers;

  for (uint32_t i = 0; i < num_arrays; ++i) {
    helpers.emplace_back(sizes[i]);
    size_t num_words = (bit_array.sizeHost(i) + WS - 1) / WS;
    std::vector<uint32_t> words(num_words, word);
    uint32_t *d_words_arr;
    gpuErrchk(cudaMalloc(&d_words_arr, num_words * sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(d_words_arr, words.data(),
                         num_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    auto [blocks, threads] =
        getLaunchConfig((num_words + WS - 1) / WS, 256, kMaxTPB);
    writeWordsParallelKernel<<<blocks, threads>>>(bit_array, i, d_words_arr,
                                                  num_words);
    for (uint32_t j = 0; j < num_words; ++j) {
      helpers[i].writeWord(j, word);
    }
    kernelCheck();
    gpuErrchk(cudaFree(d_words_arr));
  }

  RankSelect rank_select(std::move(bit_array), 0);
  for (auto &helper : helpers) {
    helper.buildRankIndex();
  }

  std::vector<size_t> random_positions(100);
  std::default_random_engine generator;
  // for each array, test the rank of 100 random positions
  for (uint32_t i = 0; i < num_arrays; ++i) {
    size_t bit_size = rank_select.bit_array_.sizeHost(i);
    std::random_device rd;
    std::mt19937 gen(rd());  // Random number generator
    std::uniform_int_distribution<> dis(1, bit_size - 1);

    std::generate(random_positions.begin(), random_positions.end(),
                  [&]() { return dis(gen); });

    auto rank_tests = [&]<int NumThreads>() {
      for (auto const &pos : random_positions) {
        rank1Kernel<NumThreads><<<1, NumThreads>>>(rank_select, i, pos, result);
        int8_t set_bits_in_last_word = pos % (sizeof(uint32_t) * 8) - 16;
        size_t num_words = pos / (sizeof(uint32_t) * 8);
        size_t rank1 = (sizeof(uint32_t) * 8) * num_words / 2 +
                       (set_bits_in_last_word > 0 ? set_bits_in_last_word : 0);
        assert(rank1 == helpers[i].rank1(pos));
        kernelCheck();
        EXPECT_EQ(*result, rank1);
        if (rank1 == 0) continue;
        select1Kernel<NumThreads>
            <<<1, NumThreads>>>(rank_select, i, rank1, result);
        size_t select1 = (sizeof(uint32_t) * 8) * (rank1 / 16) - 1;
        if (rank1 % 16 != 0) {
          select1 += 16 + rank1 % 16;
        }
        assert(select1 == helpers[i].select1(rank1));
        kernelCheck();
        EXPECT_EQ(*result, select1);

        rank0Kernel<NumThreads><<<1, NumThreads>>>(rank_select, i, pos, result);
        size_t rank0 = (sizeof(uint32_t) * 8) * num_words / 2 +
                       std::min(16UL, pos % (sizeof(uint32_t) * 8));
        assert(rank0 == helpers[i].rank0(pos));
        kernelCheck();
        EXPECT_EQ(*result, rank0);
        if (rank0 == 0) continue;
        select0Kernel<NumThreads>
            <<<1, NumThreads>>>(rank_select, i, rank0, result);
        size_t select0 = (sizeof(uint32_t) * 8) * (rank0 / 16) - 1;
        if (rank0 % 16 != 0) {
          select0 += std::min(16UL, rank0 % 16);
        } else {
          select0 -= 16;
        }
        assert(select0 == helpers[i].select0(rank0));
        kernelCheck();
        EXPECT_EQ(*result, select0);
      }
    };
    rank_tests.operator()<32>();
    rank_tests.operator()<16>();
    rank_tests.operator()<8>();
    rank_tests.operator()<4>();
    rank_tests.operator()<2>();
    rank_tests.operator()<1>();
  }
}

TEST_F(RankSelectBlocksTest, RankSelectOperationsRandom) {
  int num_arrays = 10;
  for (int _ = 0; _ < 5; _++) {
    std::vector<size_t> sizes(num_arrays);
    std::vector<RankSelectHelper> helpers;
    // Sizes are random between 2000 and 10^6
    std::vector<uint32_t> random_nums(num_arrays);
    generateRandomNums<uint32_t>(random_nums, 2000, 1e6);

    size_t max_size = 0;
    for (int i = 0; i < num_arrays; ++i) {
      sizes[i] = random_nums[i];
      max_size = std::max(max_size, sizes[i]);
      helpers.emplace_back(random_nums[i]);
    }
    BitArray bit_array(sizes, false);
    // For each array, go through each word and set it to a random number
    random_nums.resize((max_size + (sizeof(uint32_t) * 8 - 1)) /
                       (sizeof(uint32_t) * 8));
    generateRandomNums(random_nums, 0U, std::numeric_limits<uint32_t>::max());

    uint32_t *d_words_arr;
    gpuErrchk(cudaMalloc(
        &d_words_arr,
        ((max_size + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8)) *
            sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(
        d_words_arr, random_nums.data(),
        ((max_size + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8)) *
            sizeof(uint32_t),
        cudaMemcpyHostToDevice));
    for (uint32_t i = 0; i < num_arrays; ++i) {
      size_t num_words =
          (sizes[i] + (sizeof(uint32_t) * 8 - 1)) / (sizeof(uint32_t) * 8);
      auto [blocks, threads] =
          getLaunchConfig((num_words + WS - 1) / WS, 256, kMaxTPB);
      writeWordsParallelKernel<<<blocks, threads>>>(bit_array, i, d_words_arr,
                                                    num_words);
      for (uint32_t j = 0; j < num_words; ++j) {
        helpers[i].writeWord(j, random_nums[j]);
      }
      kernelCheck();
    }
    for (auto &helper : helpers) {
      helper.buildRankIndex();
    }
    gpuErrchk(cudaFree(d_words_arr));

    // perform rank and select queries on 100 random places in each array
    uint32_t num_queries = 100;
    RankSelect rank_select(std::move(bit_array), 0);
    for (uint32_t i = 0; i < num_arrays; ++i) {
      auto tests = [&]<int NumThreads>() {
        for (uint32_t j = 0; j < num_queries; ++j) {
          uint32_t index = random_nums[j] % sizes[i];
          rank0Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, index, result);
          auto rank0 = helpers[i].rank0(index);
          kernelCheck();
          EXPECT_EQ(*result, rank0);
          rank1Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, index, result);
          auto rank1 = helpers[i].rank1(index);
          kernelCheck();
          EXPECT_EQ(*result, rank1);
          if (rank0 == 0 or rank1 == 0) continue;
          if (*result != rank1) {
            rank1Kernel<NumThreads>
                <<<1, NumThreads>>>(rank_select, i, index, result);
            kernelCheck();
          }
          select1Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, rank1, result);
          auto select1 = helpers[i].select1(rank1);
          kernelCheck();
          EXPECT_EQ(*result, select1);
          select0Kernel<NumThreads>
              <<<1, NumThreads>>>(rank_select, i, rank0, result);
          auto select0 = helpers[i].select0(rank0);
          kernelCheck();
          EXPECT_EQ(*result, select0);
        }
      };
      tests.operator()<32>();
      tests.operator()<16>();
      tests.operator()<8>();
      tests.operator()<4>();
      tests.operator()<2>();
      tests.operator()<1>();
    }
  }
}

}  // namespace ecl