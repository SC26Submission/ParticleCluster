#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

constexpr size_t m = 16; // number of bits to encode quantization codes
constexpr size_t reserve_factor = 1;

template <size_t M>
using UIntForSize = std::conditional_t<
    M == 8, uint16_t,
    std::conditional_t<M == 16, uint32_t,
                       std::conditional_t<M == 32, uint64_t, void>>>;

using UInt = UIntForSize<m>;

template <size_t M>
using UInt2ForSize = std::conditional_t<
    M == 8, uint8_t,
    std::conditional_t<M == 16, uint16_t,
                       std::conditional_t<M == 32, uint32_t, void>>>;

using UInt2 = UInt2ForSize<m>;

enum class OrderMode { KD_TREE, MORTON_CODE };

template <typename T> struct CompressedData {
  size_t size_flag{0};
  size_t size_quant{0};
  size_t size_edit{0};
  std::unordered_map<uint8_t, std::pair<uint32_t, int>> code_table_flag;
  std::unordered_map<UInt, std::pair<uint32_t, int>> code_table_quant;
  std::unordered_map<UInt2, std::pair<uint32_t, int>> code_table_edit;
  size_t bit_stream_size_flag{0};
  size_t bit_stream_size_quant{0};
  size_t bit_stream_size_edit{0};
  std::vector<uint8_t> compressed_lossless_flag;
  std::vector<uint8_t> compressed_quant_codes;
  std::vector<uint8_t> compressed_quant_edits;
  std::vector<T> lossless_values;
  T grid_min_x{}, grid_min_y{}, grid_min_z{};
  size_t grid_dim_x{}, grid_dim_y{}, grid_dim_z{};
  T xi{}, b{};
  // Lossless edit mode (isPGD=false): bitmask (N bits, packed) + Huffman+ZSTD
  std::vector<uint8_t> compressed_lossless_edit_flag;
  std::unordered_map<uint8_t, std::pair<uint32_t, int>> code_table_edit_flag;
  size_t size_edit_flag{0};
  size_t bit_stream_size_edit_flag{0};
  std::vector<T> lossless_edit_values; // D interleaved values per marked position
  // Sparse edit mapping: visit-order positions of editable particles
  std::vector<int> editable_visit_positions;
  size_t N_local{0}; // number of local (non-ghost) particles; 0 = all local
};
