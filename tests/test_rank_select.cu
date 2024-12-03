#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <vector>

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

__global__ void getNumLastL2BlocksKernel(RankSelect rank_select,
                                         uint32_t array_index, size_t *output) {
  *output = rank_select.getNumLastL2Blocks(array_index);
}

__global__ void writeWordKernel(BitArray bit_array, size_t array_index,
                                size_t index, uint32_t value) {
  bit_array.writeWord(array_index, index, value);
}

__global__ void writeWordsParallelKernel(BitArray bit_array, size_t array_index,
                                         uint32_t *words, size_t num_words) {
  size_t num_threads = blockDim.x * gridDim.x;
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint32_t i = thread_id; i < num_words; i += num_threads) {
    bit_array.writeWord(array_index, i, words[i]);
  }
}

__global__ void rank0Kernel(RankSelect rank_select, uint32_t array_index,
                            size_t index, size_t *output) {
  assert(blockDim.x == WS);
  *output = rank_select.rank0(array_index, index, threadIdx.x, blockDim.x);
}

__global__ void rank1Kernel(RankSelect rank_select, uint32_t array_index,
                            size_t index, size_t *output) {
  assert(blockDim.x == WS);
  *output = rank_select.rank1(array_index, index, threadIdx.x, blockDim.x);
}

__global__ void select0Kernel(RankSelect rank_select, uint32_t array_index,
                              size_t index, size_t *output) {
  assert(blockDim.x == WS);
  *output = rank_select.select<0>(array_index, index, threadIdx.x, blockDim.x);
}

__global__ void select1Kernel(RankSelect rank_select, uint32_t array_index,
                              size_t index, size_t *output) {
  assert(blockDim.x == WS);
  *output = rank_select.select<1>(array_index, index, threadIdx.x, blockDim.x);
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
    size_t start_bit = index * 32;
    for (size_t i = 0; i < 32; ++i) {
      bit_vector_[start_bit + i] = (value >> i) & 1UL;
    }
  }
  bool operator[](size_t index) const { return bit_vector_[index]; }
};

template <typename T>
void generateRandomNums(std::vector<T> &nums_vec, T const min, T const max) {
  std::random_device rd;
  std::mt19937 gen(rd());  // Random number generator
  std::uniform_int_distribution<T> dis(min, max);

  std::generate(nums_vec.begin(), nums_vec.end(), [&]() { return dis(gen); });
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
  BitArray bit_array(sizes, false);
  RankSelect rank_select(std::move(bit_array));
}

using RankSelectBlocksTest = RankSelectTest<size_t>;
TEST_F(RankSelectBlocksTest, RankSelectIndexSizes) {
  BitArray bit_array(std::vector<size_t>{1, RankSelectConfig::L2_BIT_SIZE - 1,
                                         RankSelectConfig::L2_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L2_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L1_BIT_SIZE - 1},
                     false);
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
  EXPECT_EQ(*result, RankSelectConfig::NUM_L2_PER_L1);

  getNumLastL2BlocksKernel<<<1, 1>>>(rank_select, 5, result);
  kernelCheck();
  EXPECT_EQ(*result, 1);

  getNumLastL2BlocksKernel<<<1, 1>>>(rank_select, 6, result);
  kernelCheck();
  EXPECT_EQ(*result, RankSelectConfig::NUM_L2_PER_L1);

  size_t num_l2 = rank_select.getNumL2BlocksHost(4);
  EXPECT_EQ(num_l2, RankSelectConfig::NUM_L2_PER_L1);

  num_l2 = rank_select.getNumL2BlocksHost(5);
  EXPECT_EQ(num_l2, RankSelectConfig::NUM_L2_PER_L1 + 1);

  num_l2 = rank_select.getNumL2BlocksHost(6);
  EXPECT_EQ(num_l2, 2 * RankSelectConfig::NUM_L2_PER_L1);
}

TEST_F(RankSelectBlocksTest, RankSelectIndexWriting) {
  BitArray bit_array(std::vector<size_t>{1, RankSelectConfig::L2_BIT_SIZE - 1,
                                         RankSelectConfig::L2_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L2_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L1_BIT_SIZE - 1},
                     false);
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

    RankSelect rank_select(std::move(bit_array));

    // check that all indices are 0
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
  }
  BitArray bit_array(std::vector<size_t>{1, RankSelectConfig::L2_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L2_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE - 1,
                                         RankSelectConfig::L1_BIT_SIZE + 1,
                                         2 * RankSelectConfig::L1_BIT_SIZE - 1,
                                         2 * RankSelectConfig::L1_BIT_SIZE + 1},
                     true);

  RankSelect rank_select(std::move(bit_array));

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
  random_nums.resize((max_size + 31) / 32);
  generateRandomNums(random_nums, 0U, std::numeric_limits<uint32_t>::max());

  uint32_t *d_words_arr;
  gpuErrchk(
      cudaMalloc(&d_words_arr, ((max_size + 31) / 32) * sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(d_words_arr, random_nums.data(),
                       ((max_size + 31) / 32) * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
  for (uint32_t i = 0; i < num_arrays; ++i) {
    size_t num_words = (sizes[i] + 31) / 32;
    auto [blocks, threads] = getLaunchConfig(num_words / 32, 32, 1024);
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
  RankSelect rank_select(std::move(bit_array));
  for (uint32_t i = 0; i < num_arrays; ++i) {
    // Check that indices are correct
    size_t num_l1_blocks = rank_select.getNumL1BlocksHost(i);
    for (uint32_t j = 0; j < num_l1_blocks; ++j) {
      getL1EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      kernelCheck();
      EXPECT_EQ(*result, helpers[i].rank1(j * RankSelectConfig::L1_BIT_SIZE));
    }
    size_t num_l2 = rank_select.getNumL2BlocksHost(i);
    size_t local_l1_block = 0;
    for (uint32_t j = 0; j < num_l2; ++j) {
      if (j % RankSelectConfig::NUM_L2_PER_L1 == 0 and j != 0) {
        local_l1_block++;
      }
      getL2EntryKernel<<<1, 1>>>(rank_select, i, j, result);
      auto result_should =
          helpers[i].rank1(j * RankSelectConfig::L2_BIT_SIZE) -
          helpers[i].rank1(local_l1_block * RankSelectConfig::L1_BIT_SIZE);
      kernelCheck();
      EXPECT_EQ(*result, result_should);
    }
  }
}

TEST_F(RankSelectBlocksTest, RankSelectOperations) {
  for (auto const &val : {true, false}) {
    BitArray bit_array(
        std::vector<size_t>{RankSelectConfig::L2_BIT_SIZE + 1,
                            2 * RankSelectConfig::L2_BIT_SIZE - 1,
                            RankSelectConfig::L1_BIT_SIZE - 1,
                            RankSelectConfig::L1_BIT_SIZE + 1,
                            2 * RankSelectConfig::L1_BIT_SIZE - 1,
                            2 * RankSelectConfig::L1_BIT_SIZE + 1},
        val);
    auto num_arrays = bit_array.numArrays();

    RankSelect rank_select(std::move(bit_array));

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

      for (auto const &pos : random_positions) {
        if (val == true) {
          rank1Kernel<<<1, 32>>>(rank_select, i, pos, result);
          kernelCheck();
          EXPECT_EQ(*result, pos);
          select1Kernel<<<1, 32>>>(rank_select, i, pos, result);
          kernelCheck();
          EXPECT_EQ(*result, pos - 1);
        } else {
          rank0Kernel<<<1, 32>>>(rank_select, i, pos, result);
          kernelCheck();
          EXPECT_EQ(*result, pos);
          select0Kernel<<<1, 32>>>(rank_select, i, pos, result);
          kernelCheck();
          EXPECT_EQ(*result, pos - 1);
        }
      }
      if (val == true) {
        rank1Kernel<<<1, 32>>>(rank_select, i, 0, result);
        kernelCheck();
        EXPECT_EQ(*result, 0);
        rank1Kernel<<<1, 32>>>(rank_select, i, bit_size - 1, result);
        kernelCheck();
        EXPECT_EQ(*result, bit_size - 1);
        select1Kernel<<<1, 32>>>(rank_select, i, 1, result);
        kernelCheck();
        EXPECT_EQ(*result, 0);
        select1Kernel<<<1, 32>>>(rank_select, i, bit_size, result);
        kernelCheck();
        EXPECT_EQ(*result, bit_size - 1);

        rank0Kernel<<<1, 32>>>(rank_select, i, bit_size - 1, result);
        kernelCheck();
        EXPECT_EQ(*result, 0);
        select0Kernel<<<1, 32>>>(rank_select, i, 1, result);
        kernelCheck();
        EXPECT_EQ(*result, bit_size);
      } else {
        rank0Kernel<<<1, 32>>>(rank_select, i, 0, result);
        kernelCheck();
        EXPECT_EQ(*result, 0);
        rank0Kernel<<<1, 32>>>(rank_select, i, bit_size - 1, result);
        kernelCheck();
        EXPECT_EQ(*result, bit_size - 1);
        select0Kernel<<<1, 32>>>(rank_select, i, 1, result);
        kernelCheck();
        EXPECT_EQ(*result, 0);
        select0Kernel<<<1, 32>>>(rank_select, i, bit_size, result);
        kernelCheck();
        EXPECT_EQ(*result, bit_size - 1);

        rank1Kernel<<<1, 32>>>(rank_select, i, bit_size - 1, result);
        kernelCheck();
        EXPECT_EQ(*result, 0);
        select1Kernel<<<1, 32>>>(rank_select, i, 1, result);
        kernelCheck();
        EXPECT_EQ(*result, bit_size);
      }
    }
  }

  // test with the top half of the bits in each word set to 1
  uint32_t word = 0xFFFF0000;

  auto sizes = std::vector<size_t>{RankSelectConfig::L2_BIT_SIZE + 1,
                                   2 * RankSelectConfig::L2_BIT_SIZE - 1,
                                   RankSelectConfig::L1_BIT_SIZE - 1,
                                   RankSelectConfig::L1_BIT_SIZE + 1,
                                   2 * RankSelectConfig::L1_BIT_SIZE - 1,
                                   2 * RankSelectConfig::L1_BIT_SIZE + 1};
  BitArray bit_array(sizes, false);
  auto num_arrays = bit_array.numArrays();

  std::vector<RankSelectHelper> helpers;

  for (uint32_t i = 0; i < num_arrays; ++i) {
    helpers.emplace_back(sizes[i]);
    size_t num_words = (bit_array.sizeHost(i) + 31) / 32;
    std::vector<uint32_t> words(num_words, word);
    uint32_t *d_words_arr;
    gpuErrchk(cudaMalloc(&d_words_arr, num_words * sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(d_words_arr, words.data(),
                         num_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
    auto [blocks, threads] = getLaunchConfig(num_words / 32, 32, 1024);
    writeWordsParallelKernel<<<blocks, threads>>>(bit_array, i, d_words_arr,
                                                  num_words);
    for (uint32_t j = 0; j < num_words; ++j) {
      helpers[i].writeWord(j, word);
    }
    kernelCheck();
    gpuErrchk(cudaFree(d_words_arr));
  }

  RankSelect rank_select(std::move(bit_array));
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

    for (auto const &pos : random_positions) {
      rank1Kernel<<<1, 32>>>(rank_select, i, pos, result);
      int8_t set_bits_in_last_word = pos % 32 - 16;
      size_t num_words = pos / 32;
      size_t rank1 = 32 * num_words / 2 +
                     (set_bits_in_last_word > 0 ? set_bits_in_last_word : 0);
      assert(rank1 == helpers[i].rank1(pos));
      kernelCheck();
      EXPECT_EQ(*result, rank1);
      if (rank1 == 0) continue;
      select1Kernel<<<1, 32>>>(rank_select, i, rank1, result);
      size_t select1 = 32 * (rank1 / 16) - 1;
      if (rank1 % 16 != 0) {
        select1 += 16 + rank1 % 16;
      }
      assert(select1 == helpers[i].select1(rank1));
      kernelCheck();
      EXPECT_EQ(*result, select1);

      rank0Kernel<<<1, 32>>>(rank_select, i, pos, result);
      size_t rank0 = 32 * num_words / 2 + std::min(16UL, pos % 32);
      assert(rank0 == helpers[i].rank0(pos));
      kernelCheck();
      EXPECT_EQ(*result, rank0);
      if (rank0 == 0) continue;
      select0Kernel<<<1, 32>>>(rank_select, i, rank0, result);
      size_t select0 = 32 * (rank0 / 16) - 1;
      if (rank0 % 16 != 0) {
        select0 += std::min(16UL, rank0 % 16);
      } else {
        select0 -= 16;
      }
      assert(select0 == helpers[i].select0(rank0));
      kernelCheck();
      EXPECT_EQ(*result, select0);
    }
  }
}

TEST_F(RankSelectBlocksTest, RankSelectOperationsRandom) {
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
  random_nums.resize((max_size + 31) / 32);
  generateRandomNums(random_nums, 0U, std::numeric_limits<uint32_t>::max());

  uint32_t *d_words_arr;
  gpuErrchk(
      cudaMalloc(&d_words_arr, ((max_size + 31) / 32) * sizeof(uint32_t)));
  gpuErrchk(cudaMemcpy(d_words_arr, random_nums.data(),
                       ((max_size + 31) / 32) * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
  for (uint32_t i = 0; i < num_arrays; ++i) {
    size_t num_words = (sizes[i] + 31) / 32;
    auto [blocks, threads] = getLaunchConfig(num_words / 32, 32, 1024);
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
  uint32_t num_queries = 500;
  RankSelect rank_select(std::move(bit_array));
  for (uint32_t i = 0; i < num_arrays; ++i) {
    for (uint32_t j = 0; j < num_queries; ++j) {
      uint32_t index = random_nums[j] % sizes[i];
      rank0Kernel<<<1, 32>>>(rank_select, i, index, result);
      auto rank0 = helpers[i].rank0(index);
      kernelCheck();
      EXPECT_EQ(*result, rank0);
      rank1Kernel<<<1, 32>>>(rank_select, i, index, result);
      auto rank1 = helpers[i].rank1(index);
      kernelCheck();
      EXPECT_EQ(*result, rank1);
      if (rank0 == 0 or rank1 == 0) continue;
      if (*result != rank1) {
        rank1Kernel<<<1, 32>>>(rank_select, i, index, result);
        kernelCheck();
      }
      select1Kernel<<<1, 32>>>(rank_select, i, rank1, result);
      auto select1 = helpers[i].select1(rank1);
      kernelCheck();
      EXPECT_EQ(*result, select1);
      select0Kernel<<<1, 32>>>(rank_select, i, rank0, result);
      auto select0 = helpers[i].select0(rank0);
      kernelCheck();
      EXPECT_EQ(*result, select0);
    }
  }
}

}  // namespace ecl