/*
BSD 3-Clause License

Copyright (c) 2025, Marco Franzreb, Martin Burtscher

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
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

__host__ [[nodiscard]] size_t BitArray::sizeHost(
    size_t const array_index) const noexcept {
  assert(array_index < num_arrays_);
  return bit_sizes_[array_index];
}

__host__ [[nodiscard]] size_t BitArray::getNeededGPUMemory(
    size_t const size, uint8_t const num_arrays) noexcept {
  size_t total_size = 0;
  total_size += num_arrays * sizeof(size_t);  // Memory for d_bit_sizes_
  total_size +=
      num_arrays *
      (((size + 7) / 8) +
       kBankSizeBytes * kBanksPerLine);  // Upper bound of memory for d_data_
  total_size += num_arrays * sizeof(size_t);  // Memory for d_offsets_
  return total_size;
}

}  // namespace ecl

/******************************************************************************/