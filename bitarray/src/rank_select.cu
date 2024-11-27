#include <bit_array.cuh>
#include <cassert>
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <numeric>
#include <rank_select.cuh>
#include <vector>

namespace ecl {

__host__ RankSelect::RankSelect(BitArray const&& bit_array) noexcept
    : bit_array_(bit_array), is_copy_(false) {
  // Compute the number of L1 blocks.
  num_l1_blocks_.resize(bit_array_.numArrays());
  for (size_t i = 0; i < bit_array_.numArrays(); ++i) {
    num_l1_blocks_[i] =
        (bit_array_.sizeHost(i) + RankSelectConfig::L1_BIT_SIZE - 1) /
        RankSelectConfig::L1_BIT_SIZE;
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
  gpuErrchk(cudaMalloc(&d_l1_indices_,
                       total_l1_blocks * sizeof(RankSelectConfig::L1_TYPE)));
  gpuErrchk(cudaMemset(d_l1_indices_, 0,
                       total_l1_blocks * sizeof(RankSelectConfig::L1_TYPE)));

  std::vector<size_t> l1_offsets(num_l1_blocks_.size());

  std::exclusive_scan(num_l1_blocks_.begin(), num_l1_blocks_.end(),
                      l1_offsets.begin(), 0);
  gpuErrchk(cudaMalloc(&d_l1_offsets_, l1_offsets.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_l1_offsets_, l1_offsets.data(),
                       l1_offsets.size() * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  // Get how many l2 blocks each last L1 block has
  std::vector<uint8_t> num_last_l2_blocks(bit_array_.numArrays());
  for (size_t i = 0; i < bit_array_.numArrays(); ++i) {
    num_last_l2_blocks[i] =
        (bit_array_.sizeHost(i) % RankSelectConfig::L1_BIT_SIZE +
         RankSelectConfig::L2_BIT_SIZE - 1) /
        RankSelectConfig::L2_BIT_SIZE;
  }
  // transfer to device
  gpuErrchk(cudaMalloc(&d_num_last_l2_blocks_,
                       num_last_l2_blocks.size() * sizeof(uint8_t)));
  gpuErrchk(cudaMemcpy(d_num_last_l2_blocks_, num_last_l2_blocks.data(),
                       num_last_l2_blocks.size() * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));

  std::vector<size_t> num_l2_blocks(bit_array_.numArrays());
  for (size_t i = 0; i < bit_array_.numArrays(); ++i) {
    num_l2_blocks[i] =
        (num_l1_blocks_[i] - 1) * RankSelectConfig::NUM_L2_PER_L1 +
        num_last_l2_blocks[i];
  }

  total_num_l2_blocks_ = 0;
  for (auto const& num_blocks : num_l2_blocks) {
    total_num_l2_blocks_ += num_blocks;
  }

  // Allocate memory for the L2 index.
  // For convenience, right now first entry is for the first block, which is
  // always 0.
  gpuErrchk(cudaMalloc(&d_l2_indices_, total_num_l2_blocks_ *
                                           sizeof(RankSelectConfig::L2_TYPE)));
  gpuErrchk(
      cudaMemset(d_l2_indices_, 0,
                 total_num_l2_blocks_ * sizeof(RankSelectConfig::L2_TYPE)));

  std::exclusive_scan(num_l2_blocks.begin(), num_l2_blocks.end(),
                      num_l2_blocks.begin(), 0);

  gpuErrchk(cudaMalloc(&d_l2_offsets_, num_l2_blocks.size() * sizeof(size_t)));
  gpuErrchk(cudaMemcpy(d_l2_offsets_, num_l2_blocks.data(),
                       num_l2_blocks.size() * sizeof(size_t),
                       cudaMemcpyHostToDevice));

  l2_offsets_ = std::move(num_l2_blocks);
}

__host__ RankSelect::RankSelect(RankSelect const& other) noexcept
    : bit_array_(other.bit_array_),
      d_l1_indices_(other.d_l1_indices_),
      d_l2_indices_(other.d_l2_indices_),
      d_l1_offsets_(other.d_l1_offsets_),
      d_l2_offsets_(other.d_l2_offsets_),
      d_num_last_l2_blocks_(other.d_num_last_l2_blocks_),
      d_num_l1_blocks_(other.d_num_l1_blocks_),
      total_num_l2_blocks_(other.total_num_l2_blocks_),
      is_copy_(true) {}

__host__ RankSelect::~RankSelect() {
  if (not is_copy_) {
    gpuErrchk(cudaFree(d_l1_indices_));
    gpuErrchk(cudaFree(d_l2_indices_));
    gpuErrchk(cudaFree(d_l1_offsets_));
    gpuErrchk(cudaFree(d_l2_offsets_));
    gpuErrchk(cudaFree(d_num_last_l2_blocks_));
    gpuErrchk(cudaFree(d_num_l1_blocks_));
  }
}

__device__ [[nodiscard]] size_t RankSelect::rank0(uint32_t const array_index,
                                                  size_t const index,
                                                  uint32_t const t_id,
                                                  uint32_t const num_threads) {
  return index - rank1(array_index, index, t_id, num_threads);
}

__device__ [[nodiscard]] size_t RankSelect::rank1(uint32_t const array_index,
                                                  size_t const index,
                                                  uint32_t const t_id,
                                                  uint32_t const num_threads) {
  assert(array_index < bit_array_.numArrays());
  assert(index < bit_array_.size(array_index));
  size_t const l1_pos = index / RankSelectConfig::L1_BIT_SIZE;
  size_t const l2_pos =
      (index % RankSelectConfig::L1_BIT_SIZE) / RankSelectConfig::L2_BIT_SIZE;
  // Only first thread in the local block stores the result
  size_t result =
      (d_l1_indices_[d_l1_offsets_[array_index] + l1_pos] +
       d_l2_indices_[d_l2_offsets_[array_index] +
                     l1_pos * RankSelectConfig::NUM_L2_PER_L1 + l2_pos]) *
      (t_id == 0);
  size_t const start_word = l1_pos * RankSelectConfig::L1_WORD_SIZE +
                            l2_pos * RankSelectConfig::L2_WORD_SIZE;
  size_t const end_word = index / 32;

  for (size_t i = start_word + t_id; i <= end_word; i += num_threads) {
    if (i == end_word) {
      // Only consider bits up to the index.
      result += __popc(bit_array_.partialWord(array_index, i, index % 32));
    } else {
      result += __popc(bit_array_.word(array_index, i));
    }
  }

  __shared__ typename cub::WarpReduce<size_t>::TempStorage temp_storage;

  result = cub::WarpReduce<size_t>(temp_storage).Sum(result);

  // communicate the result to all threads
  shareVar<size_t>(t_id == 0, result, ~0);

  return result;
}

template <uint32_t Value>
__device__ [[nodiscard]] size_t RankSelect::select(uint32_t const array_index,
                                                   size_t i,
                                                   uint32_t const local_t_id,
                                                   uint32_t const num_threads) {
  assert(array_index < bit_array_.numArrays());
  assert(i <= bit_array_.size(array_index));
  assert(i > 0);
  size_t const l1_offset = d_l1_offsets_[array_index];
  size_t const l2_offset = d_l2_offsets_[array_index];
  uint8_t const num_last_l2_blocks = d_num_last_l2_blocks_[array_index];
  size_t const num_l1_blocks = d_num_l1_blocks_[array_index];
  size_t result = 0;
  // Starting from 1 since 0-th entry always 0.
  size_t j = local_t_id + 1;
  size_t current_val = 0;
  uint32_t active_threads = ~0;
  active_threads = __ballot_sync(active_threads, j < num_l1_blocks);
  while (j < num_l1_blocks) {
    if constexpr (Value == 0) {
      current_val =
          j * RankSelectConfig::L1_BIT_SIZE - d_l1_indices_[l1_offset + j];
    } else {
      current_val = d_l1_indices_[l1_offset + j];
    }
    if (current_val >= i) {
      result = j;
    }
    shareVar<size_t>(result != 0, result, active_threads);
    if (result != 0) {
      j = result - 1;
      break;
    }
    j += num_threads;
    active_threads = __ballot_sync(active_threads, j < num_l1_blocks);
  }
  shareVar<size_t>(result != 0, result, ~0);
  if (result != 0) {
    j = result - 1;
  }

  if constexpr (Value == 0) {
    i -= result == 0
             ? (num_l1_blocks - 1) * RankSelectConfig::L1_BIT_SIZE -
                   d_l1_indices_[l1_offset + num_l1_blocks - 1]
             : j * RankSelectConfig::L1_BIT_SIZE - d_l1_indices_[l1_offset + j];
  } else {
    i -= result == 0 ? d_l1_indices_[l1_offset + num_l1_blocks - 1]
                     : d_l1_indices_[l1_offset + j];
  }
  size_t current_bit = result == 0
                           ? (num_l1_blocks - 1) * RankSelectConfig::L1_BIT_SIZE
                           : j * RankSelectConfig::L1_BIT_SIZE;

  uint32_t const l1_block_length =
      result == 0 ? num_last_l2_blocks : RankSelectConfig::NUM_L2_PER_L1;

  result = 0;
  active_threads = ~0;
  j = local_t_id + 1;
  active_threads = __ballot_sync(active_threads, j < l1_block_length);
  while (j < l1_block_length) {
    if constexpr (Value == 0) {
      current_val =
          j * RankSelectConfig::L2_BIT_SIZE -
          d_l2_indices_[l2_offset +
                        current_bit / RankSelectConfig::L2_BIT_SIZE + j];
    } else {
      current_val =
          d_l2_indices_[l2_offset +
                        current_bit / RankSelectConfig::L2_BIT_SIZE + j];
    }
    if (current_val >= i) {
      result = j;
    }
    shareVar<size_t>(result != 0, result, active_threads);
    if (result != 0) {
      j = result - 1;
      break;
    }
    j += num_threads;
    __syncwarp(active_threads);
    active_threads = __ballot_sync(active_threads, j < l1_block_length);
  }
  shareVar<size_t>(result != 0, result, ~0);
  if (result != 0) {
    j = result - 1;
  }

  if constexpr (Value == 0) {
    i -= result == 0
             ? (l1_block_length - 1) * RankSelectConfig::L2_BIT_SIZE -
                   d_l2_indices_[l2_offset +
                                 current_bit / RankSelectConfig::L2_BIT_SIZE +
                                 l1_block_length - 1]
             : j * RankSelectConfig::L2_BIT_SIZE -
                   d_l2_indices_[l2_offset +
                                 current_bit / RankSelectConfig::L2_BIT_SIZE +
                                 j];
  } else {
    i -= result == 0
             ? d_l2_indices_[l2_offset +
                             current_bit / RankSelectConfig::L2_BIT_SIZE +
                             l1_block_length - 1]
             : d_l2_indices_[l2_offset +
                             current_bit / RankSelectConfig::L2_BIT_SIZE + j];
  }
  current_bit += result == 0
                     ? (l1_block_length - 1) * RankSelectConfig::L2_BIT_SIZE
                     : j * RankSelectConfig::L2_BIT_SIZE;

  size_t const l2_block_length =
      result == 0 ? bit_array_.size(array_index) - current_bit
                  : RankSelectConfig::L2_BIT_SIZE;
  result = 0;

  __shared__ typename cub::WarpScan<uint32_t>::TempStorage temp_storage;
  cub::WarpScan<uint32_t> warp_scan(temp_storage);
  for (j = 0; j < l2_block_length; j += num_threads * sizeof(uint32_t) * 8) {
    auto const bit_index = current_bit + j + sizeof(uint32_t) * 8 * local_t_id;
    uint32_t word;
    if (bit_index - current_bit >= l2_block_length) {
      if constexpr (Value == 0) {
        word = ~0;
      } else {
        word = 0;
      }
    } else {
      word = bit_array_.wordAtBit(array_index, bit_index);
    }
    uint32_t num_vals = 0;
    if constexpr (Value == 0) {
      num_vals = 32 - __popc(word);
    } else {
      num_vals = __popc(word);
    }
    uint32_t cum_vals = 0;

    // inclusive prefix sum
    warp_scan.InclusiveSum(num_vals, cum_vals);
    __syncwarp();

    // Check if the i-th zero/one is in the current word
    uint32_t const vals_at_start = cum_vals - num_vals;
    if (cum_vals >= i and vals_at_start < i) {
      // Find the position of the i-th zero/one in the word
      // 1-indexed to distinguish from having found nothing, which is 0.
      // TODO: faster implementations possible
      i -= vals_at_start;
      result = getNBitPos<Value>(i, word) + bit_index + 1;
    }
    // communicate the result to all threads
    shareVar<size_t>(result != 0, result, ~0);
    if (result != 0) {
      break;
    }
    // if no result found, update i
    i -= cum_vals;
    shareVar<size_t>(local_t_id == (WS - 1), i, ~0);
  }
  shareVar<size_t>(result != 0, result, ~0);

  // If nothing found, return the size of the array
  if (result == 0 or result > bit_array_.size(array_index)) {
    result = bit_array_.size(array_index);
  } else {
    result -= 1;
  }
  return result;
}

template __device__ size_t RankSelect::select<0>(uint32_t const array_index,
                                                 size_t i,
                                                 uint32_t const local_t_id,
                                                 uint32_t const num_threads);

template __device__ size_t RankSelect::select<1>(uint32_t const array_index,
                                                 size_t i,
                                                 uint32_t const local_t_id,
                                                 uint32_t const num_threads);

__device__ [[nodiscard]] size_t RankSelect::getNumL1Blocks(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  return d_num_l1_blocks_[array_index];
}

__host__ [[nodiscard]] size_t RankSelect::getNumL1BlocksHost(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  return num_l1_blocks_[array_index];
}

__device__ [[nodiscard]] size_t RankSelect::getNumL2Blocks(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  if (array_index == bit_array_.numArrays() - 1) {
    return total_num_l2_blocks_ - d_l2_offsets_[array_index];
  }
  return d_l2_offsets_[array_index + 1] - d_l2_offsets_[array_index];
}

__host__ [[nodiscard]] size_t RankSelect::getNumL2BlocksHost(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  if (array_index == bit_array_.numArrays() - 1) {
    return total_num_l2_blocks_ - l2_offsets_[array_index];
  }
  return l2_offsets_[array_index + 1] - l2_offsets_[array_index];
}

__device__ [[nodiscard]] size_t RankSelect::getNumLastL2Blocks(
    uint32_t const array_index) const {
  assert(array_index < bit_array_.numArrays());
  return d_num_last_l2_blocks_[array_index];
}

__device__ void RankSelect::writeL2Index(
    uint32_t const array_index, size_t const index,
    RankSelectConfig::L2_TYPE const value) noexcept {
  assert(array_index < bit_array_.numArrays());
  assert(index < getNumL2Blocks(array_index));
  d_l2_indices_[d_l2_offsets_[array_index] + index] = value;
}

__device__ void RankSelect::writeL1Index(
    uint32_t const array_index, size_t const index,
    RankSelectConfig::L1_TYPE const value) noexcept {
  assert(array_index < bit_array_.numArrays());
  assert(index < d_num_l1_blocks_[array_index]);
  d_l1_indices_[d_l1_offsets_[array_index] + index] = value;
}

__device__ [[nodiscard]] RankSelectConfig::L1_TYPE RankSelect::getL1Entry(
    uint32_t const array_index, size_t const index) const {
  assert(array_index < bit_array_.numArrays());
  assert(index < d_num_l1_blocks_[array_index]);
  return d_l1_indices_[d_l1_offsets_[array_index] + index];
}

__host__ [[nodiscard]] RankSelectConfig::L1_TYPE* RankSelect::getL1EntryPointer(
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

__device__ void RankSelect::writeNumLastL2Blocks(uint32_t const array_index,
                                                 uint8_t const value) noexcept {
  assert(array_index < bit_array_.numArrays());
  d_num_last_l2_blocks_[array_index] = value;
}

__host__ __device__ [[nodiscard]] size_t RankSelect::getTotalNumL2Blocks()
    const {
  return total_num_l2_blocks_;
}

template <typename T>
__device__ void RankSelect::shareVar(bool condition, T& var,
                                     uint32_t const mask) {
  static_assert(std::is_integral<T>::value or std::is_floating_point<T>::value,
                "T must be an integral or floating-point type.");
  __syncwarp(mask);
  uint32_t src_thread = __ballot_sync(mask, condition);
  // Get the value from the first thread that fulfills the condition
  src_thread = __ffs(src_thread) - 1;
  __syncwarp(mask);
  var = __shfl_sync(mask, var, src_thread);
}

template <uint32_t Value>
__device__ [[nodiscard]] uint8_t RankSelect::getNBitPos(uint8_t const n,
                                                        uint32_t word) {
  static_assert(Value == 0 || Value == 1, "Template parameter must be 0 or 1");
  assert(n > 0);
  assert(n <= 32);
  if constexpr (Value == 0) {
    // Find the position of the n-th zero in the word
    for (uint8_t i = 1; i < n; i++) {
      word = word | (word + 1);  // set least significant 0-bit
    }
    return __ffs(~word) - 1;

  } else {
    // Find the position of the n-th one in the word
    for (uint8_t i = 1; i < n; i++) {
      word = word & (word - 1);  // clear least significant 1-bit
    }
    return __ffs(word) - 1;
  }
}

__host__ RankSelect createRankSelectStructures(BitArray&& bit_array) {
  // build structure
  RankSelect rank_select(std::move(bit_array));

  struct cudaFuncAttributes funcAttrib;
  gpuErrchk(cudaFuncGetAttributes(&funcAttrib, calculateL2EntriesKernel));
  int maxThreadsPerBlockL2Kernel = funcAttrib.maxThreadsPerBlock;

  // TODO loop unnecessary for wavelet tree
  // Get maximum storage needed for device sums
  size_t temp_storage_bytes = 0;
  for (uint32_t i = 0; i < rank_select.bit_array_.numArrays(); i++) {
    auto const num_l1_blocks = rank_select.getNumL1BlocksHost(i);
    RankSelectConfig::L1_TYPE* d_data = rank_select.getL1EntryPointer(i, 0);
    size_t prev_storage_bytes = temp_storage_bytes;
    gpuErrchk(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_data,
                                            num_l1_blocks));
    temp_storage_bytes = std::max(temp_storage_bytes, prev_storage_bytes);
  }
  void* d_temp_storage = nullptr;
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  for (uint32_t i = 0; i < rank_select.bit_array_.numArrays(); i++) {
    auto const num_l1_blocks = rank_select.getNumL1BlocksHost(i);
    auto const num_l2_blocks = rank_select.getNumL2BlocksHost(i);
    if (num_l2_blocks == 1) {
      continue;
    }
    //?launch in separate streams?: yes, one per level

    if (num_l1_blocks == 1) {
      size_t block_size =
          std::min(maxThreadsPerBlockL2Kernel, getMaxBlockSize());
      block_size = std::min(block_size, num_l2_blocks * 32);

      calculateL2EntriesKernel<<<1, block_size>>>(rank_select, i,
                                                  num_l2_blocks);
      kernelCheck();
    } else {
      uint8_t const num_last_l2_blocks =
          (rank_select.bit_array_.sizeHost(i) % RankSelectConfig::L1_BIT_SIZE +
           RankSelectConfig::L2_BIT_SIZE - 1) /
          RankSelectConfig::L2_BIT_SIZE;
      auto [_, block_size] = getLaunchConfig(num_l2_blocks - num_last_l2_blocks,
                                             128, maxThreadsPerBlockL2Kernel);

      // calculate L2 entries for all L1 blocks
      calculateL2EntriesKernel<<<num_l1_blocks, block_size>>>(
          rank_select, i, num_last_l2_blocks);
      kernelCheck();

      RankSelectConfig::L1_TYPE* const d_data =
          rank_select.getL1EntryPointer(i, 0);

      // Run inclusive prefix sum
      gpuErrchk(cub::DeviceScan::InclusiveSum(
          d_temp_storage, temp_storage_bytes, d_data, num_l1_blocks));
    }
  }
  gpuErrchk(cudaFree(d_temp_storage));

  return rank_select;
}

__global__ void calculateL2EntriesKernel(RankSelect rank_select,
                                         uint32_t const array_index,
                                         uint8_t const num_last_l2_blocks) {
  assert(blockDim.x % WS == 0);
  __shared__ RankSelectConfig::L2_TYPE
      l2_entries[RankSelectConfig::NUM_L2_PER_L1];
  __shared__ RankSelectConfig::L1_TYPE next_l1_entry;
  __shared__ union {
    typename cub::WarpReduce<RankSelectConfig::L2_TYPE>::TempStorage reduce;
    typename cub::WarpScan<RankSelectConfig::L2_TYPE>::TempStorage scan;
  } cub_storage;

  uint32_t const warp_id = threadIdx.x / WS;
  uint32_t const local_t_id = threadIdx.x % WS;
  uint32_t const num_warps = blockDim.x / WS;
  if (blockIdx.x < gridDim.x - 1) {
    // find L1 block index
    uint32_t const l1_index = blockIdx.x;

    cub::WarpReduce<RankSelectConfig::L2_TYPE> warp_reduce(cub_storage.reduce);
    for (uint32_t i = warp_id; i < RankSelectConfig::NUM_L2_PER_L1;
         i += num_warps) {
      RankSelectConfig::L2_TYPE num_ones = 0;
      size_t const start_word = l1_index * RankSelectConfig::L1_WORD_SIZE +
                                i * RankSelectConfig::L2_WORD_SIZE;

      size_t const end_word = start_word + RankSelectConfig::L2_WORD_SIZE;
      for (size_t j = start_word + local_t_id; j < end_word; j += WS) {
        // Global memory load
        //? Benefits from shmem? no
        // load as 64 bits.
        uint32_t const word = rank_select.bit_array_.word(array_index, j);
        num_ones += __popc(word);
      }

      // Warp reduction
      RankSelectConfig::L2_TYPE const total_ones = warp_reduce.Sum(num_ones);
      __syncwarp();

      if (local_t_id == 0) {
        l2_entries[i] = total_ones;
      }
    }

    __syncthreads();
    // perform warp exclusive sum of l2 entries
    // Make sure that a whole warp executes the scan
    // TODO: right now unnecessary, since NUM_L2_PER_L1 is 32
    uint32_t needed_threads = (RankSelectConfig::NUM_L2_PER_L1 + 31) & ~31;
    if (threadIdx.x < needed_threads) {
      RankSelectConfig::L2_TYPE l2_entry;
      if (threadIdx.x < RankSelectConfig::NUM_L2_PER_L1) {
        l2_entry = l2_entries[threadIdx.x];
      } else {
        l2_entry = 0;
      }
      // last thread writes it's entry to the following L1 block
      if (threadIdx.x == RankSelectConfig::NUM_L2_PER_L1 - 1) {
        next_l1_entry = l2_entry;
      }
      cub::WarpScan<RankSelectConfig::L2_TYPE>(cub_storage.scan)
          .ExclusiveSum(l2_entry, l2_entry);

      // last thread adds it's entry to the following L1 block
      if (threadIdx.x == RankSelectConfig::NUM_L2_PER_L1 - 1) {
        next_l1_entry += l2_entry;
        rank_select.writeL1Index(array_index, l1_index + 1, next_l1_entry);
      }
      // All threads write their result to global memory
      // global memory store
      rank_select.writeL2Index(
          array_index, l1_index * RankSelectConfig::NUM_L2_PER_L1 + threadIdx.x,
          l2_entry);
    }
  }

  else {  //?benign data race?-> use libcu++ atomic store
    rank_select.writeNumLastL2Blocks(array_index, num_last_l2_blocks);
    if (num_last_l2_blocks == 1) {
      return;
    }

    auto const l1_start_word = (rank_select.getNumL1Blocks(array_index) - 1) *
                               RankSelectConfig::L1_WORD_SIZE;

    cub::WarpReduce<RankSelectConfig::L2_TYPE> warp_reduce(cub_storage.reduce);
    for (uint32_t i = warp_id; i < num_last_l2_blocks; i += num_warps) {
      RankSelectConfig::L2_TYPE num_ones = 0;
      size_t const start_word =
          l1_start_word + i * RankSelectConfig::L2_WORD_SIZE;

      size_t const end_word =
          min(rank_select.bit_array_.sizeInWords(array_index),
              start_word + RankSelectConfig::L2_WORD_SIZE);
      for (size_t j = start_word + local_t_id; j < end_word; j += WS) {
        uint32_t const word = rank_select.bit_array_.word(array_index, j);

        // Compute num ones even of last word, since it will not be saved.
        num_ones += __popc(word);
      }

      // Warp reduction
      RankSelectConfig::L2_TYPE const total_ones = warp_reduce.Sum(num_ones);
      __syncwarp();
      if (local_t_id == 0) {
        l2_entries[i] = total_ones;
      }
    }

    __syncthreads();
    // perform warp exclusive sum of l2 entries
    uint32_t needed_threads = (num_last_l2_blocks + 31) & ~31;
    if (threadIdx.x < needed_threads) {
      RankSelectConfig::L2_TYPE l2_entry;
      if (threadIdx.x < num_last_l2_blocks) {
        l2_entry = l2_entries[threadIdx.x];
      } else {
        l2_entry = 0;
      }

      cub::WarpScan<RankSelectConfig::L2_TYPE>(cub_storage.scan)
          .ExclusiveSum(l2_entry, l2_entry);

      // All threads write their result to global memory
      if (threadIdx.x < num_last_l2_blocks) {
        rank_select.writeL2Index(array_index,
                                 (rank_select.getNumL1Blocks(array_index) - 1) *
                                         RankSelectConfig::NUM_L2_PER_L1 +
                                     threadIdx.x,
                                 l2_entry);
      }
    }
  }
  return;
}

}  // namespace ecl
