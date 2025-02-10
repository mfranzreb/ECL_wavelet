#include <omp.h>

#include <bit_array.cuh>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <mutex>
#include <numeric>
#include <queue>
#include <rank_select.cuh>
#include <utils.cuh>
#include <vector>

namespace ecl {

__host__ RankSelect::RankSelect(BitArray&& bit_array,
                                uint8_t const GPU_index) noexcept
    : bit_array_(std::move(bit_array)), is_copy_(false) {
  checkWarpSize(GPU_index);

  auto const num_arrays = bit_array_.numArrays();

  // Compute the number of L1 blocks.
  num_l1_blocks_.resize(num_arrays);
  for (size_t i = 0; i < num_arrays; ++i) {
    num_l1_blocks_[i] = (bit_array_.sizeHost(i) + RSConfig::L1_BIT_SIZE - 1) /
                        RSConfig::L1_BIT_SIZE;
  }
  // transfer to device
  gpuErrchk(
      cudaMalloc(&d_num_l1_blocks_, num_l1_blocks_.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_num_l1_blocks_, num_l1_blocks_.data(),
                       num_l1_blocks_.size() * sizeof(size_t),
                       cudaMemcpyHostToDevice));
  size_t total_l1_blocks = 0;
  for (auto const& num_blocks : num_l1_blocks_) {
    total_l1_blocks += num_blocks;
  }
  // Allocate memory for the L1 index.
  // For convenience, right now first entry is for the first block, which is
  // always 0.
  gpuErrchk(
      cudaMalloc(&d_l1_indices_, total_l1_blocks * sizeof(RSConfig::L1_TYPE)));
  gpuErrchk(cudaMemset(d_l1_indices_, 0,
                       total_l1_blocks * sizeof(RSConfig::L1_TYPE)));

  std::vector<size_t> l1_offsets(num_l1_blocks_.size());

  std::exclusive_scan(num_l1_blocks_.begin(), num_l1_blocks_.end(),
                      l1_offsets.begin(), 0);
  gpuErrchk(cudaMalloc(&d_l1_offsets_, l1_offsets.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_l1_offsets_, l1_offsets.data(),
                       l1_offsets.size() * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  // Get how many l2 blocks each last L1 block has
  std::vector<uint16_t> num_last_l2_blocks(num_arrays);
  for (size_t i = 0; i < num_arrays; ++i) {
    num_last_l2_blocks[i] = (bit_array_.sizeHost(i) % RSConfig::L1_BIT_SIZE +
                             RSConfig::L2_BIT_SIZE - 1) /
                            RSConfig::L2_BIT_SIZE;
  }
  // transfer to device
  gpuErrchk(cudaMalloc(&d_num_last_l2_blocks_,
                       num_last_l2_blocks.size() * sizeof(uint16_t)));
  gpuErrchk(cudaMemcpy(d_num_last_l2_blocks_, num_last_l2_blocks.data(),
                       num_last_l2_blocks.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice));

  std::vector<size_t> num_l2_blocks_per_arr(num_arrays);
  for (size_t i = 0; i < num_arrays; ++i) {
    num_l2_blocks_per_arr[i] =
        (num_l1_blocks_[i] - 1) * RSConfig::NUM_L2_PER_L1 +
        num_last_l2_blocks[i];
  }

  total_num_l2_blocks_ = 0;
  for (auto const& num_blocks : num_l2_blocks_per_arr) {
    total_num_l2_blocks_ += num_blocks;
  }

  // Allocate memory for the L2 index.
  // For convenience, right now first entry is for the first block, which is
  // always 0.
  gpuErrchk(cudaMalloc(&d_l2_indices_,
                       total_num_l2_blocks_ * sizeof(RSConfig::L2_TYPE)));
  gpuErrchk(cudaMemset(d_l2_indices_, 0,
                       total_num_l2_blocks_ * sizeof(RSConfig::L2_TYPE)));

  std::exclusive_scan(num_l2_blocks_per_arr.begin(),
                      num_l2_blocks_per_arr.end(),
                      num_l2_blocks_per_arr.begin(), 0);

  gpuErrchk(cudaMalloc(&d_l2_offsets_,
                       num_l2_blocks_per_arr.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_l2_offsets_, num_l2_blocks_per_arr.data(),
                       num_l2_blocks_per_arr.size() * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  // TODO loop unnecessary for wavelet tree
  // Get maximum storage needed for device sums
  size_t temp_storage_bytes = 0;
  for (uint32_t i = 0; i < num_arrays; i++) {
    auto const num_l1_blocks = num_l1_blocks_[i];
    RSConfig::L1_TYPE* d_data = getL1EntryPointer(i, 0);
    size_t prev_storage_bytes = temp_storage_bytes;
    gpuErrchk(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_data,
                                            num_l1_blocks));
    temp_storage_bytes = std::max(temp_storage_bytes, prev_storage_bytes);
  }
  void* d_temp_storage = nullptr;
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // TODO: How to go about this? -> Use Warpsum
  uint32_t constexpr kBlockSize = 256;
  uint32_t constexpr kItemsPerThread =
      (RSConfig::NUM_L2_PER_L1 + kBlockSize - 1) / kBlockSize;
  // Choose maximum possible items per thread
  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib,
                                  calculateL2EntriesKernel<kItemsPerThread>));

  auto max_block_size =
      std::min(kMaxTPB, static_cast<uint32_t>(funcAttrib.maxThreadsPerBlock));

  max_block_size = findLargestDivisor(kMaxTPB, max_block_size);

  auto const prop = getDeviceProperties();

  auto const ideal_TPB =
      getIdealConfigs(prop.name).ideal_TPB_calculateL2EntriesKernel;

#pragma omp parallel for num_threads(num_arrays)
  for (uint32_t i = 0; i < num_arrays; i++) {
    auto const num_l1_blocks = num_l1_blocks_[i];
    auto const num_l2_blocks =
        i == (num_arrays - 1)
            ? total_num_l2_blocks_ - num_l2_blocks_per_arr[i]
            : num_l2_blocks_per_arr[i + 1] - num_l2_blocks_per_arr[i];
    if (num_l2_blocks == 1) {
      continue;
    }

    if (num_l1_blocks == 1) {
      uint32_t block_size =
          std::min(max_block_size, static_cast<uint32_t>(num_l2_blocks * WS));

      calculateL2EntriesKernel<kItemsPerThread>
          <<<1, kBlockSize>>>(*this, i, num_l2_blocks);
      kernelStreamCheck(cudaStreamPerThread);
    } else {
      uint16_t const num_last_l2_blocks =
          (bit_array_.sizeHost(i) % RSConfig::L1_BIT_SIZE +
           RSConfig::L2_BIT_SIZE - 1) /
          RSConfig::L2_BIT_SIZE;

      // Get minimal block size that still fully loads GPU
      auto const min_threads = static_cast<size_t>(
          prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);

      auto block_size = std::min(
          static_cast<uint32_t>(min_threads / num_l1_blocks), max_block_size);

      block_size = (block_size / WS) * WS;

      if (ideal_TPB == 0) {
        // If no ideal block size is given, use the minimum block size
        // possible
        block_size = std::max(block_size, kMinTPB);
      } else if (ideal_TPB > block_size) {
        block_size = ideal_TPB;
      }

      // calculate L2 entries for all L1 blocks
      calculateL2EntriesKernel<kItemsPerThread>
          <<<num_l1_blocks, kBlockSize>>>(*this, i, num_last_l2_blocks);
      kernelStreamCheck(cudaStreamPerThread);

      RSConfig::L1_TYPE* const d_data = getL1EntryPointer(i, 0);

#pragma omp critical
      {  // Run inclusive prefix sum
        gpuErrchk(cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes, d_data, num_l1_blocks));
        kernelCheck();
      }
    }
  }
  // Get the number of ones per bit array
  std::vector<size_t> num_ones_samples_per_array(num_arrays);
  std::vector<size_t> num_zeros_samples_per_array(num_arrays);
  for (uint8_t i = 0; i < num_arrays; i++) {
    size_t last_l1_index;
    RSConfig::L2_TYPE last_l2_index;
    size_t const remaining_bits =
        bit_array_.sizeHost(i) % RSConfig::L2_BIT_SIZE;
    gpuErrchk(cudaMemcpy(&last_l1_index,
                         d_l1_indices_ + l1_offsets[i] + num_l1_blocks_[i] - 1,
                         sizeof(size_t), cudaMemcpyDeviceToHost));
    gpuErrchk(
        cudaMemcpy(&last_l2_index,
                   d_l2_indices_ + num_l2_blocks_per_arr[i] +
                       (bit_array_.sizeHost(i) + RSConfig::L2_BIT_SIZE - 1) /
                           RSConfig::L2_BIT_SIZE -
                       1,
                   sizeof(RSConfig::L2_TYPE), cudaMemcpyDeviceToHost));
    size_t const num_ones = last_l1_index + last_l2_index + remaining_bits;
    size_t const num_zeros =
        bit_array_.sizeHost(i) - last_l1_index - last_l2_index;
    num_ones_samples_per_array[i] = num_ones / RSConfig::SELECT_SAMPLE_RATE;
    num_zeros_samples_per_array[i] = num_zeros / RSConfig::SELECT_SAMPLE_RATE;
  }
  size_t const total_ones_samples = std::accumulate(
      num_ones_samples_per_array.begin(), num_ones_samples_per_array.end(), 0);
  std::exclusive_scan(num_ones_samples_per_array.begin(),
                      num_ones_samples_per_array.end(),
                      num_ones_samples_per_array.begin(), 0);
  gpuErrchk(
      cudaMalloc(&d_select_samples_1_, total_ones_samples * sizeof(size_t)));
  gpuErrchk(
      cudaMalloc(&d_select_samples_1_offsets_, num_arrays * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_select_samples_1_offsets_,
                       num_ones_samples_per_array.data(),
                       num_arrays * sizeof(size_t), cudaMemcpyHostToDevice));

  size_t const total_zeros_samples =
      std::accumulate(num_zeros_samples_per_array.begin(),
                      num_zeros_samples_per_array.end(), 0);
  std::exclusive_scan(num_zeros_samples_per_array.begin(),
                      num_zeros_samples_per_array.end(),
                      num_zeros_samples_per_array.begin(), 0);
  gpuErrchk(
      cudaMalloc(&d_select_samples_0_, total_zeros_samples * sizeof(size_t)));
  gpuErrchk(
      cudaMalloc(&d_select_samples_0_offsets_, num_arrays * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_select_samples_0_offsets_,
                       num_zeros_samples_per_array.data(),
                       num_arrays * sizeof(size_t), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc(&d_total_num_ones_, num_arrays * sizeof(size_t)));
  // #pragma omp parallel for num_threads(num_arrays)
  for (uint8_t i = 0; i < num_arrays; i++) {
    size_t const total_l2_blocks =
        (bit_array_.sizeHost(i) + RSConfig::L2_BIT_SIZE - 1) /
        RSConfig::L2_BIT_SIZE;
    auto const num_blocks = (total_l2_blocks + WS - 1) / WS;
    calculateSelectSamplesKernel<WS>
        <<<num_blocks, kMaxTPB, sizeof(size_t) * kMaxTPB / WS>>>(
            *this, i, num_blocks * (kMaxTPB / WS), total_l2_blocks,
            (bit_array_.sizeHost(i) + sizeof(uint32_t) * 8 - 1) /
                (sizeof(uint32_t) * 8),
            &d_total_num_ones_[i]);
    kernelStreamCheck(cudaStreamPerThread);
  }
  gpuErrchk(cudaFree(d_temp_storage));
  kernelCheck();
}

__host__ RankSelect::RankSelect(RankSelect const& other) noexcept
    : bit_array_(other.bit_array_),
      d_l1_indices_(other.d_l1_indices_),
      d_l2_indices_(other.d_l2_indices_),
      d_l1_offsets_(other.d_l1_offsets_),
      d_l2_offsets_(other.d_l2_offsets_),
      d_num_last_l2_blocks_(other.d_num_last_l2_blocks_),
      d_num_l1_blocks_(other.d_num_l1_blocks_),
      num_l1_blocks_(other.num_l1_blocks_),
      total_num_l2_blocks_(other.total_num_l2_blocks_),
      d_select_samples_0_(other.d_select_samples_0_),
      d_select_samples_1_(other.d_select_samples_1_),
      d_select_samples_0_offsets_(other.d_select_samples_0_offsets_),
      d_select_samples_1_offsets_(other.d_select_samples_1_offsets_),
      d_total_num_ones_(other.d_total_num_ones_),
      is_copy_(true) {}

__host__ RankSelect& RankSelect::operator=(RankSelect&& other) noexcept {
  bit_array_ = std::move(other.bit_array_);
  d_l1_indices_ = other.d_l1_indices_;
  other.d_l1_indices_ = nullptr;
  d_l2_indices_ = other.d_l2_indices_;
  other.d_l2_indices_ = nullptr;
  d_l1_offsets_ = other.d_l1_offsets_;
  other.d_l1_offsets_ = nullptr;
  d_l2_offsets_ = other.d_l2_offsets_;
  other.d_l2_offsets_ = nullptr;
  d_num_last_l2_blocks_ = other.d_num_last_l2_blocks_;
  other.d_num_last_l2_blocks_ = nullptr;
  d_num_l1_blocks_ = other.d_num_l1_blocks_;
  other.d_num_l1_blocks_ = nullptr;
  num_l1_blocks_ = std::move(other.num_l1_blocks_);
  total_num_l2_blocks_ = other.total_num_l2_blocks_;
  d_select_samples_0_ = other.d_select_samples_0_;
  other.d_select_samples_0_ = nullptr;
  d_select_samples_1_ = other.d_select_samples_1_;
  other.d_select_samples_1_ = nullptr;
  d_select_samples_0_offsets_ = other.d_select_samples_0_offsets_;
  other.d_select_samples_0_offsets_ = nullptr;
  d_select_samples_1_offsets_ = other.d_select_samples_1_offsets_;
  other.d_select_samples_1_offsets_ = nullptr;
  d_total_num_ones_ = other.d_total_num_ones_;
  other.d_total_num_ones_ = nullptr;
  is_copy_ = other.is_copy_;
  other.is_copy_ = true;
  return *this;
}

__host__ RankSelect::~RankSelect() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_l1_indices_));
    gpuErrchk(cudaFree(d_l2_indices_));
    gpuErrchk(cudaFree(d_l1_offsets_));
    gpuErrchk(cudaFree(d_l2_offsets_));
    gpuErrchk(cudaFree(d_num_last_l2_blocks_));
    gpuErrchk(cudaFree(d_num_l1_blocks_));
    gpuErrchk(cudaFree(d_select_samples_0_));
    gpuErrchk(cudaFree(d_select_samples_1_));
    gpuErrchk(cudaFree(d_select_samples_0_offsets_));
    gpuErrchk(cudaFree(d_select_samples_1_offsets_));
    gpuErrchk(cudaFree(d_total_num_ones_));
  }
}

__device__ [[nodiscard]] size_t RankSelect::getNumL1Blocks(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  return d_num_l1_blocks_[array_index];
}

__device__ [[nodiscard]] size_t RankSelect::getNumL2Blocks(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  if (array_index == bit_array_.numArrays() - 1) {
    return total_num_l2_blocks_ - d_l2_offsets_[array_index];
  }
  return d_l2_offsets_[array_index + 1] - d_l2_offsets_[array_index];
}

__device__ [[nodiscard]] size_t RankSelect::getNumLastL2Blocks(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  return d_num_last_l2_blocks_[array_index];
}

__device__ void RankSelect::writeL2Index(
    uint32_t const array_index, size_t const index,
    RSConfig::L2_TYPE const value) noexcept {
  assert(array_index < bit_array_.numArrays());
  assert(index < getNumL2Blocks(array_index));
  d_l2_indices_[d_l2_offsets_[array_index] + index] = value;
}

__device__ void RankSelect::writeL1Index(
    uint32_t const array_index, size_t const index,
    RSConfig::L1_TYPE const value) noexcept {
  assert(array_index < bit_array_.numArrays());
  assert(index < d_num_l1_blocks_[array_index]);
  d_l1_indices_[d_l1_offsets_[array_index] + index] = value;
}

__device__ [[nodiscard]] RSConfig::L1_TYPE RankSelect::getL1Entry(
    uint32_t const array_index, size_t const index) const {
  assert(array_index < bit_array_.numArrays());
  assert(index < d_num_l1_blocks_[array_index]);
  return d_l1_indices_[d_l1_offsets_[array_index] + index];
}

__host__ [[nodiscard]] RSConfig::L1_TYPE* RankSelect::getL1EntryPointer(
    uint32_t const array_index, size_t const index) const {
  assert(array_index < bit_array_.numArrays());
  assert(index < num_l1_blocks_[array_index]);
  // Pointer arithmetic
  size_t offset = 0;
  for (size_t i = 0; i < array_index; ++i) {
    offset += num_l1_blocks_[i];
  }
  return d_l1_indices_ + offset + index;
}

__device__ [[nodiscard]] size_t RankSelect::getL2Entry(
    uint32_t const array_index, size_t const index) const {
  assert(array_index < bit_array_.numArrays());
  assert(index < getNumL2Blocks(array_index));
  return d_l2_indices_[d_l2_offsets_[array_index] + index];
}

__device__ void RankSelect::writeNumLastL2Blocks(
    uint32_t const array_index, uint16_t const value) noexcept {
  assert(array_index < bit_array_.numArrays());
  d_num_last_l2_blocks_[array_index] = value;
}

}  // namespace ecl
