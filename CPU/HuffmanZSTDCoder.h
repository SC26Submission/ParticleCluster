#pragma once

#include "config.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <zstd.h>

template <typename T> struct HuffmanNode {
  T value;
  uint32_t freq;
  int left, right;

  HuffmanNode() : value(T()), freq(0), left(-1), right(-1) {}
  HuffmanNode(T v, uint32_t f) : value(v), freq(f), left(-1), right(-1) {}
  bool is_leaf() const { return left == -1 && right == -1; }
};

template <typename T> class HuffmanZstdCompressor {
public:
  struct CompressionResult {
    uint8_t *compressed_data = nullptr;
    size_t compressed_size = 0;

    T *symbol_table = nullptr;
    uint32_t *code_table = nullptr;
    uint8_t *length_table = nullptr;
    int num_symbols = 0;

    size_t original_bitstream_size = 0;
    size_t num_elements = 0;

    void freeSymbolTables() {
      if (symbol_table)
        delete[] symbol_table;
      if (code_table)
        delete[] code_table;
      if (length_table)
        delete[] length_table;
      symbol_table = nullptr;
      code_table = nullptr;
      length_table = nullptr;
    }

    void freeCompressedData() {
      if (compressed_data)
        delete[] compressed_data;
      compressed_data = nullptr;
    }
  };

private:
  void countFrequencies(const T *data, size_t N, std::vector<T> &unique,
                        std::vector<uint32_t> &counts) {
    // Sort and run-length encode
    std::vector<T> sorted(data, data + N);
    std::sort(sorted.begin(), sorted.end());

    unique.clear();
    counts.clear();

    if (N == 0)
      return;

    T current = sorted[0];
    uint32_t count = 1;

    for (size_t i = 1; i < N; ++i) {
      if (sorted[i] == current) {
        ++count;
      } else {
        unique.push_back(current);
        counts.push_back(count);
        current = sorted[i];
        count = 1;
      }
    }
    unique.push_back(current);
    counts.push_back(count);
  }

  void buildHuffmanTree(const T *symbols, const uint32_t *freqs, int n,
                        HuffmanNode<T> *&nodes, int &root_idx) {
    if (n == 0) {
      root_idx = -1;
      nodes = nullptr;
      return;
    }

    if (n == 1) {
      nodes = new HuffmanNode<T>[1];
      nodes[0] = HuffmanNode<T>(symbols[0], freqs[0]);
      root_idx = 0;
      return;
    }

    using PQNode = std::pair<uint32_t, int>;
    std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> pq;

    nodes = new HuffmanNode<T>[2 * n - 1];

    for (int i = 0; i < n; ++i) {
      nodes[i] = HuffmanNode<T>(symbols[i], freqs[i]);
      pq.push({freqs[i], i});
    }

    int next_node = n;

    while (pq.size() > 1) {
      auto [freq1, idx1] = pq.top();
      pq.pop();
      auto [freq2, idx2] = pq.top();
      pq.pop();

      nodes[next_node].freq = freq1 + freq2;
      nodes[next_node].left = idx1;
      nodes[next_node].right = idx2;

      pq.push({nodes[next_node].freq, next_node});
      ++next_node;
    }

    root_idx = pq.top().second;
  }

  void generateCodes(const HuffmanNode<T> *nodes, int node_idx,
                     std::unordered_map<T, std::pair<uint32_t, int>> &code_map,
                     uint32_t code, int length) {
    if (node_idx == -1)
      return;

    const auto &node = nodes[node_idx];

    if (node.is_leaf()) {
      code_map[node.value] = {code, length == 0 ? 1 : length};
    } else {
      generateCodes(nodes, node.left, code_map, code << 1, length + 1);
      generateCodes(nodes, node.right, code_map, (code << 1) | 1, length + 1);
    }
  }

  // Maps data values to Huffman codes
  void buildLookup(const T *data, const T *symbols, const uint32_t *codes,
                   const uint8_t *lengths, int num_symbols, uint32_t *out_codes,
                   uint8_t *out_lengths, size_t N) {
    for (size_t idx = 0; idx < N; ++idx) {
      T value = data[idx];

      // Binary search
      int left = 0, right = num_symbols - 1;
      bool found = false;
      while (left <= right) {
        int mid = (left + right) >> 1;
        if (symbols[mid] == value) {
          out_codes[idx] = codes[mid];
          out_lengths[idx] = lengths[mid];
          found = true;
          break;
        } else if (symbols[mid] < value) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }

      if (!found) {
        out_codes[idx] = 0;
        out_lengths[idx] = 1;
      }
    }
  }

  void packBytesAligned(const uint32_t *codes, const uint8_t *lengths,
                        const size_t *bit_positions, uint8_t *output,
                        size_t N) {
    for (size_t idx = 0; idx < N; ++idx) {
      uint32_t code = codes[idx];
      uint8_t len = lengths[idx];
      size_t bit_start = bit_positions[idx];

      for (int i = 0; i < len; ++i) {
        size_t bit_pos = bit_start + i;
        size_t byte_idx = bit_pos >> 3;
        int bit_idx = 7 - (bit_pos & 7);

        if ((code >> (len - 1 - i)) & 1) {
          output[byte_idx] |= (1 << bit_idx);
        }
      }
    }
  }

public:
  CompressionResult compress(const T *data, size_t N, int zstd_level = 3,
                             bool verbose = false) {
    if (N == 0) {
      throw std::runtime_error("Empty input data");
    }

    // Count frequencies
    std::vector<T> unique;
    std::vector<uint32_t> counts;
    countFrequencies(data, N, unique, counts);
    int num_unique = static_cast<int>(unique.size());

    if (verbose) {
      printf("  Frequency count: %d unique\n", num_unique);
    }

    // Build Huffman tree
    HuffmanNode<T> *nodes;
    int root_idx;
    buildHuffmanTree(unique.data(), counts.data(), num_unique, nodes, root_idx);

    std::unordered_map<T, std::pair<uint32_t, int>> code_map;
    generateCodes(nodes, root_idx, code_map, 0, 0);

    // Sort symbols for binary search
    T *h_symbols = new T[num_unique];
    std::copy(unique.begin(), unique.end(), h_symbols);
    std::sort(h_symbols, h_symbols + num_unique);

    uint32_t *h_codes = new uint32_t[num_unique];
    uint8_t *h_lengths = new uint8_t[num_unique];

    for (int i = 0; i < num_unique; ++i) {
      auto it = code_map.find(h_symbols[i]);
      h_codes[i] = it->second.first;
      h_lengths[i] = it->second.second;
    }

    delete[] nodes;

    // Build per-element lookup
    std::vector<uint32_t> out_codes(N);
    std::vector<uint8_t> out_lengths(N);

    buildLookup(data, h_symbols, h_codes, h_lengths, num_unique,
                out_codes.data(), out_lengths.data(), N);

    // Compute bit positions (exclusive prefix sum)
    std::vector<size_t> bit_positions(N);
    bit_positions[0] = 0;
    for (size_t i = 1; i < N; ++i) {
      bit_positions[i] = bit_positions[i - 1] + out_lengths[i - 1];
    }

    size_t total_bits = bit_positions[N - 1] + out_lengths[N - 1];
    size_t total_bytes = (total_bits + 7) >> 3;

    // Pack bits
    std::vector<uint8_t> packed(total_bytes, 0);
    packBytesAligned(out_codes.data(), out_lengths.data(), bit_positions.data(),
                     packed.data(), N);

    // ZSTD compression
    size_t zstd_bound = ZSTD_compressBound(total_bytes);
    uint8_t *h_compressed = new uint8_t[zstd_bound];

    size_t compressed_size = ZSTD_compress(
        h_compressed, zstd_bound, packed.data(), total_bytes, zstd_level);

    if (ZSTD_isError(compressed_size)) {
      delete[] h_compressed;
      delete[] h_symbols;
      delete[] h_codes;
      delete[] h_lengths;
      throw std::runtime_error("ZSTD compression failed");
    }

    // Resize compressed data
    uint8_t *h_compressed_final = new uint8_t[compressed_size];
    std::copy(h_compressed, h_compressed + compressed_size, h_compressed_final);
    delete[] h_compressed;

    return {h_compressed_final, compressed_size, h_symbols,   h_codes,
            h_lengths,          num_unique,      total_bytes, N};
  }

  void decompress(const CompressionResult &result, T *output,
                  bool verbose = false) {
    // ZSTD decompression
    std::vector<uint8_t> bitstream(result.original_bitstream_size);

    size_t decompressed_size =
        ZSTD_decompress(bitstream.data(), result.original_bitstream_size,
                        result.compressed_data, result.compressed_size);

    if (ZSTD_isError(decompressed_size)) {
      throw std::runtime_error("ZSTD decompression failed");
    }

    // Build reverse lookup table
    struct CodeEntry {
      uint32_t code;
      T symbol;
    };

    std::vector<std::vector<CodeEntry>> reverse_lookup(33);

    for (int i = 0; i < result.num_symbols; ++i) {
      uint8_t len = result.length_table[i];
      uint32_t code = result.code_table[i];
      reverse_lookup[len].push_back({code, result.symbol_table[i]});
    }

    // Sort each length's entries by code
    for (int len = 1; len <= 32; ++len) {
      std::sort(reverse_lookup[len].begin(), reverse_lookup[len].end(),
                [](const CodeEntry &a, const CodeEntry &b) {
                  return a.code < b.code;
                });
    }

    // Decode
    size_t output_idx = 0;
    size_t bit_pos = 0;
    size_t total_bits = result.original_bitstream_size * 8;

    int min_len = 32, max_len = 1;
    for (int len = 1; len <= 32; ++len) {
      if (!reverse_lookup[len].empty()) {
        min_len = std::min(min_len, len);
        max_len = std::max(max_len, len);
      }
    }

    while (output_idx < result.num_elements && bit_pos < total_bits) {
      bool found = false;

      for (int len = min_len; len <= max_len && bit_pos + len <= total_bits;
           ++len) {
        if (reverse_lookup[len].empty())
          continue;

        uint32_t code = 0;
        size_t byte_idx = bit_pos >> 3;
        int bit_offset = bit_pos & 7;

        if (bit_offset + len <= 8) {
          code = (bitstream[byte_idx] >> (8 - bit_offset - len)) &
                 ((1u << len) - 1);
        } else if (bit_offset + len <= 16) {
          uint16_t two_bytes =
              ((uint16_t)bitstream[byte_idx] << 8) | bitstream[byte_idx + 1];
          code = (two_bytes >> (16 - bit_offset - len)) & ((1u << len) - 1);
        } else if (bit_offset + len <= 24) {
          uint32_t three_bytes = ((uint32_t)bitstream[byte_idx] << 16) |
                                 ((uint32_t)bitstream[byte_idx + 1] << 8) |
                                 bitstream[byte_idx + 2];
          code = (three_bytes >> (24 - bit_offset - len)) & ((1u << len) - 1);
        } else if (bit_offset + len <= 32) {
          uint32_t four_bytes = ((uint32_t)bitstream[byte_idx] << 24) |
                                ((uint32_t)bitstream[byte_idx + 1] << 16) |
                                ((uint32_t)bitstream[byte_idx + 2] << 8) |
                                bitstream[byte_idx + 3];
          code = (four_bytes >> (32 - bit_offset - len)) & ((1u << len) - 1);
        } else {
          for (int i = 0; i < len; ++i) {
            size_t b_idx = (bit_pos + i) >> 3;
            int b_bit = 7 - ((bit_pos + i) & 7);
            int bit = (bitstream[b_idx] >> b_bit) & 1;
            code = (code << 1) | bit;
          }
        }

        // Binary search
        auto &entries = reverse_lookup[len];
        int left = 0, right = entries.size() - 1;
        while (left <= right) {
          int mid = (left + right) >> 1;
          if (entries[mid].code == code) {
            output[output_idx++] = entries[mid].symbol;
            bit_pos += len;
            found = true;
            break;
          } else if (entries[mid].code < code) {
            left = mid + 1;
          } else {
            right = mid - 1;
          }
        }

        if (found)
          break;
      }

      if (!found) {
        throw std::runtime_error(
            "Decompression failed: invalid bitstream at bit " +
            std::to_string(bit_pos));
      }
    }
  }

  void writeCompressionResult(const CompressionResult &result,
                              const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    file.write(reinterpret_cast<const char *>(&result.compressed_size),
               sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&result.num_symbols),
               sizeof(int));
    file.write(reinterpret_cast<const char *>(&result.original_bitstream_size),
               sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&result.num_elements),
               sizeof(size_t));

    if (result.compressed_data != nullptr && result.compressed_size > 0) {
      file.write(reinterpret_cast<const char *>(result.compressed_data),
                 result.compressed_size);
    }
    if (result.symbol_table != nullptr && result.num_symbols > 0) {
      file.write(reinterpret_cast<const char *>(result.symbol_table),
                 result.num_symbols * sizeof(T));
    }
    if (result.code_table != nullptr && result.num_symbols > 0) {
      file.write(reinterpret_cast<const char *>(result.code_table),
                 result.num_symbols * sizeof(uint32_t));
    }
    if (result.length_table != nullptr && result.num_symbols > 0) {
      file.write(reinterpret_cast<const char *>(result.length_table),
                 result.num_symbols * sizeof(uint8_t));
    }

    if (!file) {
      throw std::runtime_error("Failed to write to file: " + filename);
    }
  }

  CompressionResult readCompressionResult(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    CompressionResult result;

    file.read(reinterpret_cast<char *>(&result.compressed_size),
              sizeof(size_t));
    file.read(reinterpret_cast<char *>(&result.num_symbols), sizeof(int));
    file.read(reinterpret_cast<char *>(&result.original_bitstream_size),
              sizeof(size_t));
    file.read(reinterpret_cast<char *>(&result.num_elements), sizeof(size_t));

    if (result.compressed_size > 0) {
      result.compressed_data = new uint8_t[result.compressed_size];
      file.read(reinterpret_cast<char *>(result.compressed_data),
                result.compressed_size);
    }

    if (result.num_symbols > 0) {
      result.symbol_table = new T[result.num_symbols];
      file.read(reinterpret_cast<char *>(result.symbol_table),
                result.num_symbols * sizeof(T));

      result.code_table = new uint32_t[result.num_symbols];
      file.read(reinterpret_cast<char *>(result.code_table),
                result.num_symbols * sizeof(uint32_t));

      result.length_table = new uint8_t[result.num_symbols];
      file.read(reinterpret_cast<char *>(result.length_table),
                result.num_symbols * sizeof(uint8_t));
    }

    if (!file) {
      if (result.compressed_data != nullptr)
        delete[] result.compressed_data;
      if (result.symbol_table != nullptr)
        delete[] result.symbol_table;
      if (result.code_table != nullptr)
        delete[] result.code_table;
      if (result.length_table != nullptr)
        delete[] result.length_table;
      throw std::runtime_error("Failed to read complete data from file");
    }

    return result;
  }
};

// Convenience functions
template <typename T>
typename HuffmanZstdCompressor<T>::CompressionResult
compressHuffmanZstd(const T *data, size_t N, int zstd_level = 3,
                    bool verbose = false) {
  HuffmanZstdCompressor<T> compressor;
  return compressor.compress(data, N, zstd_level, verbose);
}

template <typename T>
void decompressHuffmanZstd(
    const typename HuffmanZstdCompressor<T>::CompressionResult &result,
    T *output, bool verbose = false) {
  HuffmanZstdCompressor<T> compressor;
  compressor.decompress(result, output, verbose);
}

template <typename T>
void writeCompressedData(
    const typename HuffmanZstdCompressor<T>::CompressionResult &result,
    const std::string &filename) {
  HuffmanZstdCompressor<T> compressor;
  compressor.writeCompressionResult(result, filename);
}

template <typename T>
typename HuffmanZstdCompressor<T>::CompressionResult
readCompressedData(const std::string &filename) {
  HuffmanZstdCompressor<T> compressor;
  return compressor.readCompressionResult(filename);
}

// API-compatible wrapper functions that bridge HuffmanZstdCompressor
// class to the vector/unordered_map interface used by particle_compression.h.

template <typename T>
std::vector<uint8_t> huffmanZstdCompress(
    const std::vector<T> &data,
    std::unordered_map<T, std::pair<uint32_t, int>> &codeTableOut,
    size_t &bitstreamSizeOut, int zstdLevel = 3) {
  if (data.empty())
    return {};

  HuffmanZstdCompressor<T> compressor;
  auto result = compressor.compress(data.data(), data.size(), zstdLevel);

  // Convert parallel arrays back to unordered_map for caller
  codeTableOut.clear();
  codeTableOut.reserve(result.num_symbols);
  for (int i = 0; i < result.num_symbols; ++i) {
    codeTableOut[result.symbol_table[i]] = {result.code_table[i],
                                            result.length_table[i]};
  }

  bitstreamSizeOut = result.original_bitstream_size;

  // Copy compressed data to vector
  std::vector<uint8_t> compressed(
      result.compressed_data, result.compressed_data + result.compressed_size);

  result.freeSymbolTables();
  result.freeCompressedData();

  return compressed;
}

template <typename T>
std::vector<T> huffmanZstdDecompress(
    const std::vector<uint8_t> &compressed,
    const std::unordered_map<T, std::pair<uint32_t, int>> &codeTable,
    size_t numElements, size_t originalBitstreamSize) {
  if (compressed.empty() || numElements == 0)
    return {};

  // Build a CompressionResult from the old-style parameters
  typename HuffmanZstdCompressor<T>::CompressionResult result;

  int num_symbols = static_cast<int>(codeTable.size());
  result.compressed_data = new uint8_t[compressed.size()];
  std::copy(compressed.begin(), compressed.end(), result.compressed_data);
  result.compressed_size = compressed.size();

  result.symbol_table = new T[num_symbols];
  result.code_table = new uint32_t[num_symbols];
  result.length_table = new uint8_t[num_symbols];
  result.num_symbols = num_symbols;
  result.original_bitstream_size = originalBitstreamSize;
  result.num_elements = numElements;

  int i = 0;
  for (const auto &[sym, code_len] : codeTable) {
    result.symbol_table[i] = sym;
    result.code_table[i] = code_len.first;
    result.length_table[i] = static_cast<uint8_t>(code_len.second);
    ++i;
  }

  std::vector<T> output(numElements);
  HuffmanZstdCompressor<T> compressor;
  compressor.decompress(result, output.data());

  result.freeSymbolTables();
  result.freeCompressedData();

  return output;
}
