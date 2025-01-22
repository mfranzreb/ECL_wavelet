#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

#include "bit_array.cuh"
#include "utils.cuh"

// TODO: For funcs that use offsets, give the opt to pass the offset, and avoid
// a mem access

namespace ecl {
__host__ BitArray::BitArray(std::vector<size_t> const& sizes)
    : num_arrays_(sizes.size()), is_copy_(false) {
  assert(sizes.size() < std::numeric_limits<uint8_t>::max());
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
  std::vector<size_t> array_offsets(sizes.size());

  //  Pad arrays to 128 bytes (32 words)
  for (size_t i = 0; i < sizes.size(); ++i) {
    auto size_in_words =
        (sizes[i] + sizeof(uint32_t) * 8 - 1) / (sizeof(uint32_t) * 8);
    array_sizes[i] = size_in_words;
    array_offsets[i] = (size_in_words + sizeof(uint32_t) * 8 - 1) &
                       ~(sizeof(uint32_t) * 8 - 1);
  }

  size_t total_size = 0;
  for (auto const& size : array_offsets) {
    total_size += size;
  }
  // Make total size a multiple of 2 so that "twoWords" function does not go out
  // of bounds
  total_size += total_size % 2;

  gpuErrchk(cudaMalloc(&d_data_, total_size * sizeof(uint32_t)));
  total_size_ = total_size;

  // perform exclusive sum to get the offsets
  std::exclusive_scan(array_offsets.begin(), array_offsets.end(),
                      array_offsets.begin(), 0);

  gpuErrchk(cudaMalloc(&d_offsets_, array_offsets.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_offsets_, array_offsets.data(),
                       array_offsets.size() * sizeof(size_t),
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
      d_data_(other.d_data_),
      d_offsets_(other.d_offsets_),
      is_copy_(true) {}

__host__ BitArray::BitArray(BitArray&& other) noexcept {
  num_arrays_ = other.num_arrays_;
  total_size_ = other.total_size_;
  d_bit_sizes_ = other.d_bit_sizes_;
  other.d_bit_sizes_ = nullptr;
  bit_sizes_ = other.bit_sizes_;
  d_data_ = other.d_data_;
  other.d_data_ = nullptr;
  d_offsets_ = other.d_offsets_;
  other.d_offsets_ = nullptr;
  is_copy_ = other.is_copy_;
  other.is_copy_ = true;
}

__host__ BitArray& BitArray::operator=(BitArray&& other) noexcept {
  num_arrays_ = other.num_arrays_;
  total_size_ = other.total_size_;
  d_bit_sizes_ = other.d_bit_sizes_;
  other.d_bit_sizes_ = nullptr;
  bit_sizes_ = other.bit_sizes_;
  d_data_ = other.d_data_;
  other.d_data_ = nullptr;
  d_offsets_ = other.d_offsets_;
  other.d_offsets_ = nullptr;
  is_copy_ = other.is_copy_;
  other.is_copy_ = true;
  return *this;
}

__host__ BitArray::~BitArray() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_data_));
    gpuErrchk(cudaFree(d_bit_sizes_));
    gpuErrchk(cudaFree(d_offsets_));
  }
}

__device__ [[nodiscard]] bool BitArray::access(
    size_t const array_index, size_t const index) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  // Get position in 32-bit word
  uint8_t const offset = index & uint8_t(0b11111);
  // Get relevant word, shift and return bit
  return (d_data_[d_offsets_[array_index] + (index >> 5)] >> offset) & 1UL;
}

__device__ [[nodiscard]] bool BitArray::access(
    size_t const array_index, size_t const index,
    size_t const offset) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  // Get position in 32-bit word
  uint8_t const bit_offset = index & uint8_t(0b11111);
  // Get relevant word, shift and return bit
  return (d_data_[offset + (index >> 5)] >> bit_offset) & 1U;
}

__device__ void BitArray::writeWord(size_t const array_index,
                                    size_t const index,
                                    uint32_t const value) noexcept {
  assert(array_index < num_arrays_);
  assert(index < sizeInWords(array_index));
  d_data_[d_offsets_[array_index] + index] = value;
}

__device__ void BitArray::writeWordAtBit(size_t const array_index,
                                         size_t const index,
                                         uint32_t const value) noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  d_data_[d_offsets_[array_index] + (index / (sizeof(uint32_t) * 8))] = value;
}

__device__ void BitArray::writeWordAtBit(size_t const array_index,
                                         size_t const index,
                                         uint32_t const value,
                                         size_t const offset) noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  d_data_[offset + (index / (sizeof(uint32_t) * 8))] = value;
}

__device__ size_t BitArray::getOffset(size_t const array_index) const noexcept {
  assert(array_index < num_arrays_);
  return d_offsets_[array_index];
}

__device__ uint32_t BitArray::word(size_t const array_index,
                                   size_t const index) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < sizeInWords(array_index));
  return d_data_[d_offsets_[array_index] + index];
}

__device__ uint32_t BitArray::word(size_t const array_index, size_t const index,
                                   size_t const offset) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < sizeInWords(array_index));
  return d_data_[offset + index];
}

__device__ uint64_t BitArray::twoWords(size_t const array_index,
                                       size_t const index) const noexcept {
  assert(array_index < num_arrays_);
  assert(index + 1 < sizeInWords(array_index));
  assert(index % 2 == 0 or index == 0);
  return reinterpret_cast<const uint64_t*>(
      d_data_)[(d_offsets_[array_index] + index) / 2];
}

__device__ uint64_t BitArray::twoWords(size_t const array_index,
                                       size_t const index,
                                       size_t const offset) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < sizeInWords(array_index));
  assert(index % 2 == 0 or index == 0);
  return reinterpret_cast<const uint64_t*>(d_data_)[(offset + index) / 2];
}

__device__ uint32_t BitArray::wordAtBit(size_t const array_index,
                                        size_t const index) const noexcept {
  assert(array_index < num_arrays_);
  assert(index < d_bit_sizes_[array_index]);
  return d_data_[d_offsets_[array_index] + (index / (sizeof(uint32_t) * 8))];
}

__device__ [[nodiscard]] uint32_t BitArray::partialWord(
    uint32_t word, uint8_t const bit_index) noexcept {
  assert(bit_index <= sizeof(uint32_t) * 8);
  return word & ((1UL << bit_index) - 1);
}

__device__ [[nodiscard]] uint64_t BitArray::partialTwoWords(
    uint64_t word, uint8_t const bit_index) noexcept {
  assert(bit_index <= sizeof(uint64_t) * 8);
  return word & ((1ULL << bit_index) - 1);
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
  return (d_bit_sizes_[array_index] + sizeof(uint32_t) * 8 - 1) /
         (sizeof(uint32_t) * 8);
}

__host__ __device__ [[nodiscard]] uint8_t BitArray::numArrays() const noexcept {
  return num_arrays_;
}
}  // namespace ecl

/******************************************************************************/