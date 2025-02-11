#include <iostream>
#include <cstdint>
#include <cstring>
#include <bitset>
#include <bit>
#include <cassert>
#include <random>

#include <cmath>

 template <uint32_t Value, typename T>
  uint8_t getNBitPos(uint8_t const n, T word) {
    static_assert(Value == 0 or Value == 1,
                  "Template parameter must be 0 or 1");
    static_assert(std::is_integral<T>::value, "T must be an integral type.");
    static_assert(sizeof(T) == 4 or sizeof(T) == 8, "T must be 4 or 8 bytes.");
    assert(n > 0);
    if constexpr (sizeof(T) == 4) {
      assert(n <= (sizeof(uint32_t) * 8));
    if constexpr (Value == 0) {
      // Find the position of the n-th zero in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word | (word + 1);  // set least significant 0-bit
      }
      return ffs(~word) - 1;

    } else {
      // Find the position of the n-th one in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word & (word - 1);  // clear least significant 1-bit
      }
      return ffs(word) - 1;
    }
    } else {
      assert(n <= (sizeof(uint64_t) * 8));
      if constexpr (Value == 0) {
      // Find the position of the n-th zero in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word | (word + 1);  // set least significant 0-bit
      }
      return ffsll(~word) - 1;

    } else {
      // Find the position of the n-th one in the word
      for (uint8_t i = 1; i < n; i++) {
        word = word & (word - 1);  // clear least significant 1-bit
      }
      return ffsll(word) - 1;
    }
    }
  }

template <uint32_t Value, typename T>
  uint8_t nth_bit_set(T value, uint8_t n)
{
    if constexpr (Value == 0){
      value = ~value;
    }
    T mask;
    uint8_t shift;
    uint8_t base = 0u;
    uint8_t count;
    if constexpr (sizeof(T) == 4){
      mask = 0x0000FFFFu;
      shift = 16u;
    } else {
      mask = 0x00000000FFFFFFFFu;
      shift = 32u;
    }
uint8_t num_iters = 5;
    if constexpr (sizeof(T) == 8) {
      num_iters = 6;
    }
      for (uint8_t i = 0; i < num_iters; i++){
        if constexpr (sizeof(T) == 4){
          count = __builtin_popcount(value & mask);
        } else {
          count = __builtin_popcountll(value & mask);
        }
        if (n > count) {
            base += shift;
            shift >>= 1;
            mask |= mask << shift;
        } else {
            shift >>= 1;
            mask >>= shift;
        }
      }
      return base;
}

  int main(int argc, char* argv[]) {
    // test random words for 32 and 64 bit
    uint32_t word32;
    uint64_t word64;

    uint8_t n;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis32(0, std::numeric_limits<uint32_t>::max());
    std::uniform_int_distribution<uint64_t> dis64(0, std::numeric_limits<uint64_t>::max());
    std::vector<uint32_t> words32(10000);
    std::vector<uint64_t> words64(10000);
    for (int i = 0; i < 10000; i++) {
      words32[i] = dis32(gen);
      words64[i] = dis64(gen);
    }
    //for each word, test 5 random positions
    for (int i = 0; i < 10000; i++) {
      word32 = words32[i];
      word64 = words64[i];
        auto ones_in_word = __builtin_popcount(word32);
        auto ones_in_word64 = __builtin_popcountll(word64);
      for (int j = 0; j < 5; j++) {
        n = std::uniform_int_distribution<uint8_t>(1, ones_in_word)(gen);
        auto result_should = getNBitPos<1>(n, word32);
        auto result = nth_bit_set<1>(word32, n);
        assert(result_should == result);
        n = std::uniform_int_distribution<uint8_t>(1, 32 - ones_in_word)(gen);
        result_should = getNBitPos<0>(n, word32);
        result = nth_bit_set<0>(word32, n);
        assert(result_should == result);
        n = std::uniform_int_distribution<uint8_t>(1, ones_in_word64)(gen);
        result_should = getNBitPos<1>(n, word64);
        result = nth_bit_set<1>(word64, n);
        assert(result_should == result);
        n = std::uniform_int_distribution<uint8_t>(1, 64 - ones_in_word64)(gen);
        result_should = getNBitPos<0>(n, word64);
        result = nth_bit_set<0>(word64, n);
        assert(result_should == result);
      }
    }
  }