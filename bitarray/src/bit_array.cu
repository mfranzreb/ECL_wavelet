#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

#include "bit_array.cuh"
#include "utils.cuh"

namespace ecl {
__host__ BitArray::BitArray(std::vector<size_t> const& sizes)
    : num_arrays_(sizes.size()), is_copy_(false) {
  // if any of the sizes is 0, abort
  for (auto const& size : sizes) {
    assert(size > 0);
  }
  bit_sizes_ = sizes;

  // Allocate memory for sizes on the device
  gpuErrchk(cudaMalloc(&d_bit_sizes_, sizes.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_bit_sizes_, sizes.data(),
                       sizes.size() * sizeof(size_t), cudaMemcpyHostToDevice));

  std::vector<size_t> array_sizes(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    array_sizes[i] = (sizes[i] >> 5) + 1;
  }
  gpuErrchk(cudaMalloc(&d_sizes_, array_sizes.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_sizes_, array_sizes.data(),
                       array_sizes.size() * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  size_t total_size = 0;
  for (auto const& size : array_sizes) {
    total_size += size;
  }
  gpuErrchk(cudaMalloc(&d_data_, total_size * sizeof(uint32_t)));
  total_size_ = total_size;

  // perform exclusive sum to get the offsets
  std::exclusive_scan(array_sizes.begin(), array_sizes.end(),
                      array_sizes.begin(), 0);

  gpuErrchk(cudaMalloc(&d_offsets_, array_sizes.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_offsets_, array_sizes.data(),
                       array_sizes.size() * sizeof(size_t),
                       cudaMemcpyHostToDevice));
}

__host__ BitArray::BitArray(std::vector<size_t> const& sizes,
                            bool const init_value)
    : BitArray(sizes) {
  gpuErrchk(cudaMemset(d_data_, init_value ? ~(0UL) : 0UL,
                       total_size_ * sizeof(uint32_t)));
}

__host__ BitArray::BitArray(BitArray const& other)
    : num_arrays_(other.num_arrays_),
      total_size_(other.total_size_),
      d_bit_sizes_(other.d_bit_sizes_),
      bit_sizes_(other.bit_sizes_),
      d_sizes_(other.d_sizes_),
      d_data_(other.d_data_),
      d_offsets_(other.d_offsets_),
      is_copy_(true) {}

__host__ BitArray::BitArray(BitArray&& other) noexcept {
  num_arrays_ = other.num_arrays_;
  total_size_ = other.total_size_;
  d_bit_sizes_ = other.d_bit_sizes_;
  bit_sizes_ = other.bit_sizes_;
  d_sizes_ = other.d_sizes_;
  d_data_ = other.d_data_;
  d_offsets_ = other.d_offsets_;
  is_copy_ = other.is_copy_;
  other.d_data_ = nullptr;
}

__host__ BitArray::~BitArray() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_data_));
    gpuErrchk(cudaFree(d_bit_sizes_));
    gpuErrchk(cudaFree(d_offsets_));
    gpuErrchk(cudaFree(d_sizes_));
  }
}

__device__ [[nodiscard]] bool BitArray::access(
    size_t const array_index, size_t const index) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  // Get position in 32-bit word
  uint8_t const offset = 31 - (index & uint32_t(0b11111));
  // Get relevant word, shift and return bit
  return (d_data_[d_offsets_[array_index] + (index >> 5)] >> offset) & 1UL;
}

__device__ void BitArray::writeWord(size_t const array_index,
                                    size_t const index,
                                    uint32_t const value) noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_sizes_[array_index]);
  d_data_[d_offsets_[array_index] + index] = value;
}

__device__ void BitArray::writeWordAtBit(size_t const array_index,
                                         size_t const index,
                                         uint32_t const value) noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  d_data_[d_offsets_[array_index] + (index / (sizeof(uint32_t) * 8))] = value;
}

__device__ uint32_t BitArray::word(size_t const array_index,
                                   size_t const index) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_sizes_[array_index]);
  return d_data_[d_offsets_[array_index] + index];
}

__device__ uint32_t BitArray::wordAtBit(size_t const array_index,
                                        size_t const index) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  return d_data_[d_offsets_[array_index] + (index / (sizeof(uint32_t) * 8))];
}

__device__ [[nodiscard]] size_t BitArray::size(
    size_t const array_index) const noexcept {
  assert(array_index < num_arrays_);
  return d_bit_sizes_[array_index];
}

__host__ [[nodiscard]] size_t BitArray::sizeHost(
    size_t const array_index) const noexcept {
  assert(array_index < num_arrays_);
  return bit_sizes_[array_index];
}

__device__ [[nodiscard]] size_t BitArray::sizeInWords(
    size_t const array_index) const noexcept {
  assert(array_index < num_arrays_);
  return d_sizes_[array_index];
}

__host__ __device__ [[nodiscard]] size_t BitArray::numArrays() const noexcept {
  return num_arrays_;
}
}  // namespace ecl

/******************************************************************************/