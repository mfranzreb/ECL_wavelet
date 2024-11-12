#include <cassert>
#include <cstdint>

#include "bit_array.cuh"
#include "utils.cuh"

namespace ecl {
__host__ BitArray::BitArray(size_t const size)
    : bit_size_(size), size_((bit_size_ >> 5) + 1), is_copy_(false) {
  gpuErrchk(cudaMalloc(&d_data_, size_ * sizeof(uint32_t)));
}

__host__ BitArray::BitArray(size_t const size, bool const init_value)
    : bit_size_(size), size_((bit_size_ >> 5) + 1), is_copy_(false) {
  gpuErrchk(cudaMalloc(&d_data_, size_ * sizeof(uint32_t)));
  gpuErrchk(
      cudaMemset(d_data_, init_value ? ~(0UL) : 0UL, size_ * sizeof(uint32_t)));
}

__host__ BitArray::BitArray(BitArray const& other)
    : bit_size_(other.bit_size_),
      size_(other.size_),
      is_copy_(true),
      d_data_(other.d_data_) {}

__host__ BitArray::~BitArray() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_data_));
  }
}

__device__ [[nodiscard]] bool BitArray::access(
    size_t const index) const noexcept {
  assert(index < bit_size_);
  // Get position in 32-bit word
  uint8_t const offset = index & uint32_t(0b11111);
  // Get relevant word, shift and return bit
  return (d_data_[index >> 5] >> offset) & 1UL;
}

__device__ void BitArray::write_word(size_t const index,
                                     uint32_t const value) noexcept {
  assert(index < bit_size_);
  d_data_[index / (sizeof(uint32_t) * 8)] = ~value;
}

__device__ uint32_t BitArray::word(size_t const index) const noexcept {
  assert(index < bit_size_);
  return ~d_data_[index / (sizeof(uint32_t) * 8)];
}

__host__ __device__ [[nodiscard]] size_t BitArray::size() const noexcept {
  return bit_size_;
}

}  // namespace ecl

/******************************************************************************/