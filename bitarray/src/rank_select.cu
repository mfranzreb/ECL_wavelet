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

__host__ [[nodiscard]] size_t RankSelect::getNeededGPUMemory(
    size_t const size, uint8_t const num_arrays) noexcept {
  size_t total_size = 0;
  total_size += num_arrays * sizeof(size_t);  // d_num_l1_blocks_
  size_t const num_l1_blocks_per_arr =
      (size + RSConfig::L1_BIT_SIZE - 1) / RSConfig::L1_BIT_SIZE;
  total_size +=
      num_l1_blocks_per_arr * num_arrays *
      sizeof(RSConfig::L1_TYPE);  // Upper bound of memory for d_l1_indices_
  total_size += num_arrays * sizeof(size_t);    // d_l1_offsets_
  total_size += num_arrays * sizeof(uint16_t);  // d_num_last_l2_blocks_
  total_size +=
      ((size + RSConfig::L2_BIT_SIZE - 1) / RSConfig::L2_BIT_SIZE) *
      num_arrays *
      sizeof(RSConfig::L2_TYPE);  // Upper bound of memory for d_l2_indices_
  total_size += num_arrays * sizeof(size_t);  // d_l2_offsets_
  size_t temp_storage_bytes = 0;
  RSConfig::L1_TYPE* d_data = nullptr;
  gpuErrchk(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_data,
                                          num_l1_blocks_per_arr));
  total_size += temp_storage_bytes;
  total_size += num_arrays * sizeof(size_t);  // d_total_num_ones_
  total_size += num_arrays * (size / RSConfig::SELECT_SAMPLE_RATE) *
                sizeof(size_t);  // d_select_samples_0_ and d_select_samples_1_
  total_size += num_arrays * sizeof(size_t);  // d_select_samples_0_offsets_
  total_size += num_arrays * sizeof(size_t);  // d_select_samples_1_offsets_
  return total_size;
}

}  // namespace ecl
