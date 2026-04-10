#pragma once

#include "config.cuh"
#include <algorithm>
#include <cub/cub.cuh>
#include <fstream>
#include <queue>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <unordered_map>
#include <vector>
#include <zstd.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(_err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

template <typename T> struct HuffmanNode {
  T value;
  uint32_t freq;
  int left, right;

  HuffmanNode() : value(T()), freq(0), left(-1), right(-1) {}
  HuffmanNode(T v, uint32_t f) : value(v), freq(f), left(-1), right(-1) {}
  bool is_leaf() const { return left == -1 && right == -1; }
};

template <typename T>
__global__ void
buildLookupKernel(const T *__restrict__ data, const T *__restrict__ symbols,
                  const uint32_t *__restrict__ codes,
                  const uint8_t *__restrict__ lengths, int num_symbols,
                  uint32_t *__restrict__ out_codes,
                  uint8_t *__restrict__ out_lengths, size_t N) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  T value = data[idx];

  // Binary search
  int left = 0, right = num_symbols - 1;
  while (left <= right) {
    int mid = (left + right) >> 1;
    if (symbols[mid] == value) {
      out_codes[idx] = codes[mid];
      out_lengths[idx] = lengths[mid];
      return;
    } else if (symbols[mid] < value) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  out_codes[idx] = 0;
  out_lengths[idx] = 1;
}

__global__ void packBytesAligned(const uint32_t *__restrict__ codes,
                                 const uint8_t *__restrict__ lengths,
                                 const size_t *__restrict__ bit_positions,
                                 uint8_t *__restrict__ output, size_t N);

template <typename T>
__global__ void histogramKernel(const T *__restrict__ data, size_t N,
                                uint32_t *__restrict__ histogram) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < N) {
    atomicAdd(&histogram[static_cast<uint32_t>(data[idx])], 1u);
  }
}

struct Uint8ToSizeT {
  __device__ __forceinline__ size_t operator()(uint8_t val) const {
    return static_cast<size_t>(val);
  }
};

template <typename T> class HuffmanZstdCompressor {
public:
  struct CompressionResult {
    uint8_t *compressed_data = nullptr; // Device or host pointer
    size_t compressed_size = 0;

    T *symbol_table = nullptr;       // Host pointer
    uint32_t *code_table = nullptr;  // Host pointer
    uint8_t *length_table = nullptr; // Host pointer
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
  void countFrequencies(const T *d_data, size_t N, T *&h_symbols_out,
                        uint32_t *&h_freqs_out, int &num_unique) {
    // Find max value to determine if histogram approach is feasible
    T *d_max;
    CHECK_CUDA(cudaMalloc(&d_max, sizeof(T)));
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Max(d_temp, temp_bytes, d_data, d_max, N);
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceReduce::Max(d_temp, temp_bytes, d_data, d_max, N);
    CHECK_CUDA(cudaFree(d_temp));

    T h_max;
    CHECK_CUDA(
        cudaMemcpy(&h_max, d_max, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_max));

    size_t num_bins = static_cast<size_t>(h_max) + 1;

    if (num_bins <= (1u << 20)) {
      // Histogram approach: O(num_bins) GPU memory instead of O(N)
      uint32_t *d_histogram;
      CHECK_CUDA(cudaMalloc(&d_histogram, num_bins * sizeof(uint32_t)));
      CHECK_CUDA(cudaMemset(d_histogram, 0, num_bins * sizeof(uint32_t)));

      dim3 block(256);
      dim3 grid((N + block.x - 1) / block.x);
      histogramKernel<<<grid, block>>>(d_data, N, d_histogram);
      CHECK_CUDA(cudaDeviceSynchronize());

      std::vector<uint32_t> h_histogram(num_bins);
      CHECK_CUDA(cudaMemcpy(h_histogram.data(), d_histogram,
                             num_bins * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(d_histogram));

      // Extract non-zero entries
      num_unique = 0;
      for (size_t i = 0; i < num_bins; ++i) {
        if (h_histogram[i] > 0)
          num_unique++;
      }

      h_symbols_out = new T[num_unique];
      h_freqs_out = new uint32_t[num_unique];
      int idx = 0;
      for (size_t i = 0; i < num_bins; ++i) {
        if (h_histogram[i] > 0) {
          h_symbols_out[idx] = static_cast<T>(i);
          h_freqs_out[idx] = h_histogram[i];
          idx++;
        }
      }
    } else {
      // Fallback: sort + run-length encode for large value ranges
      T *d_sorted;
      CHECK_CUDA(cudaMalloc(&d_sorted, N * sizeof(T)));

      d_temp = nullptr;
      temp_bytes = 0;
      cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_data, d_sorted, N);

      T *d_unique;
      uint32_t *d_counts;
      size_t temp_bytes_rle = 0;
      CHECK_CUDA(cudaMalloc(&d_unique, N * sizeof(T)));
      CHECK_CUDA(cudaMalloc(&d_counts, N * sizeof(uint32_t)));
      int *d_num_runs;
      CHECK_CUDA(cudaMalloc(&d_num_runs, sizeof(int)));

      cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes_rle, d_sorted,
                                         d_unique, d_counts, d_num_runs, N);

      size_t max_temp = std::max(temp_bytes, temp_bytes_rle);
      CHECK_CUDA(cudaMalloc(&d_temp, max_temp));

      cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_data, d_sorted, N);
      cub::DeviceRunLengthEncode::Encode(d_temp, temp_bytes_rle, d_sorted,
                                         d_unique, d_counts, d_num_runs, N);

      CHECK_CUDA(cudaMemcpy(&num_unique, d_num_runs, sizeof(int),
                             cudaMemcpyDeviceToHost));

      h_symbols_out = new T[num_unique];
      h_freqs_out = new uint32_t[num_unique];
      CHECK_CUDA(cudaMemcpy(h_symbols_out, d_unique, num_unique * sizeof(T),
                             cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(h_freqs_out, d_counts,
                             num_unique * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));

      CHECK_CUDA(cudaFree(d_temp));
      CHECK_CUDA(cudaFree(d_sorted));
      CHECK_CUDA(cudaFree(d_num_runs));
      CHECK_CUDA(cudaFree(d_unique));
      CHECK_CUDA(cudaFree(d_counts));
    }
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

public:
  CompressionResult compress(const T *d_data, size_t N, int zstd_level = 3,
                             bool verbose = false) {
    if (N == 0) {
      throw std::runtime_error("Empty input data");
    }

    cudaEvent_t start, stop;
    float ms;

    if (verbose) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
    }

    // Count frequencies (returns host arrays directly)
    T *h_symbols;
    uint32_t *h_freqs;
    int num_unique;

    countFrequencies(d_data, N, h_symbols, h_freqs, num_unique);

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Frequency count: %.2f ms (%d unique)\n", ms, num_unique);
      cudaEventRecord(start);
    }

    // Build Huffman tree
    HuffmanNode<T> *nodes;
    int root_idx;
    buildHuffmanTree(h_symbols, h_freqs, num_unique, nodes, root_idx);

    std::unordered_map<T, std::pair<uint32_t, int>> code_map;
    generateCodes(nodes, root_idx, code_map, 0, 0);

    // Sort symbols for binary search
    std::sort(h_symbols, h_symbols + num_unique);

    uint32_t *h_codes = new uint32_t[num_unique];
    uint8_t *h_lengths = new uint8_t[num_unique];

    for (int i = 0; i < num_unique; ++i) {
      auto it = code_map.find(h_symbols[i]);
      h_codes[i] = it->second.first;
      h_lengths[i] = it->second.second;
    }

    delete[] nodes;
    delete[] h_freqs;

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Build tree: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // Copy lookup tables to GPU
    T *d_symbols;
    uint32_t *d_codes_lut;
    uint8_t *d_lengths_lut;

    CHECK_CUDA(cudaMalloc(&d_symbols, num_unique * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_codes_lut, num_unique * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_lengths_lut, num_unique * sizeof(uint8_t)));

    CHECK_CUDA(cudaMemcpy(d_symbols, h_symbols, num_unique * sizeof(T),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_codes_lut, h_codes, num_unique * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_lengths_lut, h_lengths,
                          num_unique * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    // Build per-element lookup
    uint32_t *d_codes;
    uint8_t *d_lengths;
    CHECK_CUDA(cudaMalloc(&d_codes, N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_lengths, N * sizeof(uint8_t)));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    buildLookupKernel<<<grid, block>>>(d_data, d_symbols, d_codes_lut,
                                       d_lengths_lut, num_unique, d_codes,
                                       d_lengths, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Build lookup: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // Compute bit positions using transform iterator (avoids 8N-byte d_bit_lens)
    size_t *d_bit_positions;
    CHECK_CUDA(cudaMalloc(&d_bit_positions, N * sizeof(size_t)));

    auto d_len_as_size = thrust::make_transform_iterator(
        thrust::device_pointer_cast(d_lengths), Uint8ToSizeT{});

    void *d_temp = nullptr;
    size_t temp_bytes = 0;

    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_len_as_size,
                                  d_bit_positions, N);
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_len_as_size,
                                  d_bit_positions, N);
    CHECK_CUDA(cudaFree(d_temp));

    size_t last_pos;
    uint8_t last_len_u8;
    CHECK_CUDA(cudaMemcpy(&last_pos, &d_bit_positions[N - 1], sizeof(size_t),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&last_len_u8, &d_lengths[N - 1], sizeof(uint8_t),
                          cudaMemcpyDeviceToHost));
    size_t total_bits = last_pos + last_len_u8;
    size_t total_bytes = (total_bits + 7) >> 3;

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Compute positions: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // Pack bits
    uint8_t *d_packed;
    CHECK_CUDA(cudaMalloc(&d_packed, total_bytes));
    CHECK_CUDA(cudaMemset(d_packed, 0, total_bytes));

    packBytesAligned<<<grid, block>>>(d_codes, d_lengths, d_bit_positions,
                                      d_packed, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Pack bits: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // Copy to host
    uint8_t *h_bitstream = new uint8_t[total_bytes];
    CHECK_CUDA(
        cudaMemcpy(h_bitstream, d_packed, total_bytes, cudaMemcpyDeviceToHost));

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Copy to host: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // ZSTD compression
    size_t zstd_bound = ZSTD_compressBound(total_bytes);
    uint8_t *h_compressed = new uint8_t[zstd_bound];

    size_t compressed_size = ZSTD_compress(
        h_compressed, zstd_bound, h_bitstream, total_bytes, zstd_level);

    if (ZSTD_isError(compressed_size)) {
      delete[] h_bitstream;
      delete[] h_compressed;
      delete[] h_symbols;
      delete[] h_codes;
      delete[] h_lengths;
      throw std::runtime_error("ZSTD compression failed");
    }

    // Resize compressed data (allocate exact size)
    uint8_t *h_compressed_final = new uint8_t[compressed_size];
    std::copy(h_compressed, h_compressed + compressed_size, h_compressed_final);
    delete[] h_compressed;
    delete[] h_bitstream;

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  ZSTD compress: %.2f ms\n", ms);
    }

    // Cleanup GPU memory
    CHECK_CUDA(cudaFree(d_symbols));
    CHECK_CUDA(cudaFree(d_codes_lut));
    CHECK_CUDA(cudaFree(d_lengths_lut));
    CHECK_CUDA(cudaFree(d_codes));
    CHECK_CUDA(cudaFree(d_lengths));
    CHECK_CUDA(cudaFree(d_bit_positions));
    CHECK_CUDA(cudaFree(d_packed));

    if (verbose) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

    return {h_compressed_final, compressed_size, h_symbols,   h_codes,
            h_lengths,          num_unique,      total_bytes, N};
  }

  void decompress(const CompressionResult &result, T *d_output,
                  bool verbose = false) {
    cudaEvent_t start, stop;
    float ms;

    if (verbose) {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
    }

    // ZSTD decompression
    uint8_t *h_bitstream = new uint8_t[result.original_bitstream_size];

    size_t decompressed_size =
        ZSTD_decompress(h_bitstream, result.original_bitstream_size,
                        result.compressed_data, result.compressed_size);

    if (ZSTD_isError(decompressed_size)) {
      delete[] h_bitstream;
      throw std::runtime_error("ZSTD decompression failed");
    }

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  ZSTD decompress: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // Build optimized reverse lookup table with better data structure
    // Use array-based lookup for faster access
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

    // Sort each length's entries by code for potential binary search
    for (int len = 1; len <= 32; ++len) {
      std::sort(reverse_lookup[len].begin(), reverse_lookup[len].end(),
                [](const CodeEntry &a, const CodeEntry &b) {
                  return a.code < b.code;
                });
    }

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Build reverse lookup: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // Decode to host with highly optimized lookup
    T *h_output = new T[result.num_elements];
    size_t output_idx = 0;
    size_t bit_pos = 0;
    size_t total_bits = result.original_bitstream_size * 8;

    const uint8_t *bitstream = h_bitstream;

    // Pre-compute which lengths actually exist
    int min_len = 32, max_len = 1;
    for (int len = 1; len <= 32; ++len) {
      if (!reverse_lookup[len].empty()) {
        min_len = std::min(min_len, len);
        max_len = std::max(max_len, len);
      }
    }

    while (output_idx < result.num_elements && bit_pos < total_bits) {
      bool found = false;

      // Only check lengths that exist, starting from min to max
      for (int len = min_len; len <= max_len && bit_pos + len <= total_bits;
           ++len) {
        if (reverse_lookup[len].empty())
          continue;

        uint32_t code = 0;

        // Optimized bit extraction
        size_t byte_idx = bit_pos >> 3;
        int bit_offset = bit_pos & 7;

        // Fast path: extract from pre-loaded bytes
        if (bit_offset + len <= 8) {
          // Fits in single byte
          code = (bitstream[byte_idx] >> (8 - bit_offset - len)) &
                 ((1u << len) - 1);
        } else if (bit_offset + len <= 16) {
          // Fits in two bytes
          uint16_t two_bytes =
              ((uint16_t)bitstream[byte_idx] << 8) | bitstream[byte_idx + 1];
          code = (two_bytes >> (16 - bit_offset - len)) & ((1u << len) - 1);
        } else if (bit_offset + len <= 24) {
          // Fits in three bytes
          uint32_t three_bytes = ((uint32_t)bitstream[byte_idx] << 16) |
                                 ((uint32_t)bitstream[byte_idx + 1] << 8) |
                                 bitstream[byte_idx + 2];
          code = (three_bytes >> (24 - bit_offset - len)) & ((1u << len) - 1);
        } else if (bit_offset + len <= 32) {
          // Fits in four bytes
          uint32_t four_bytes = ((uint32_t)bitstream[byte_idx] << 24) |
                                ((uint32_t)bitstream[byte_idx + 1] << 16) |
                                ((uint32_t)bitstream[byte_idx + 2] << 8) |
                                bitstream[byte_idx + 3];
          code = (four_bytes >> (32 - bit_offset - len)) & ((1u << len) - 1);
        } else {
          // Rare case: extract bit by bit
          for (int i = 0; i < len; ++i) {
            size_t b_idx = (bit_pos + i) >> 3;
            int b_bit = 7 - ((bit_pos + i) & 7);
            int bit = (bitstream[b_idx] >> b_bit) & 1;
            code = (code << 1) | bit;
          }
        }

        // Binary search in sorted array
        auto &entries = reverse_lookup[len];
        int left = 0, right = entries.size() - 1;
        while (left <= right) {
          int mid = (left + right) >> 1;
          if (entries[mid].code == code) {
            h_output[output_idx++] = entries[mid].symbol;
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
        delete[] h_bitstream;
        delete[] h_output;
        throw std::runtime_error(
            "Decompression failed: invalid bitstream at bit " +
            std::to_string(bit_pos));
      }
    }

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  CPU decode: %.2f ms\n", ms);
      cudaEventRecord(start);
    }

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_output, h_output, result.num_elements * sizeof(T),
                          cudaMemcpyHostToDevice));

    if (verbose) {
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&ms, start, stop);
      printf("  Copy to device: %.2f ms\n", ms);
    }

    delete[] h_bitstream;
    delete[] h_output;

    if (verbose) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
  }

  void writeCompressionResult(const CompressionResult &result,
                              const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write all scalar fields needed for decompression
    file.write(reinterpret_cast<const char *>(&result.compressed_size),
               sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&result.num_symbols),
               sizeof(int));
    file.write(reinterpret_cast<const char *>(&result.original_bitstream_size),
               sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&result.num_elements),
               sizeof(size_t));

    // Write all array data needed for decompression
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

    // Read all scalar fields
    file.read(reinterpret_cast<char *>(&result.compressed_size),
              sizeof(size_t));
    file.read(reinterpret_cast<char *>(&result.num_symbols), sizeof(int));
    file.read(reinterpret_cast<char *>(&result.original_bitstream_size),
              sizeof(size_t));
    file.read(reinterpret_cast<char *>(&result.num_elements), sizeof(size_t));

    // Allocate and read all arrays
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
      // Clean up on failure
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

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

template <typename T>
typename HuffmanZstdCompressor<T>::CompressionResult
compressHuffmanZstd(const T *d_data, size_t N, int zstd_level = 3,
                    bool verbose = false) {
  HuffmanZstdCompressor<T> compressor;
  return compressor.compress(d_data, N, zstd_level, verbose);
}

template <typename T>
void decompressHuffmanZstd(
    const typename HuffmanZstdCompressor<T>::CompressionResult &result,
    T *d_output, bool verbose = false) {
  HuffmanZstdCompressor<T> compressor;
  compressor.decompress(result, d_output, verbose);
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

// ============================================================================
// API-compatible wrappers: std::vector/std::unordered_map interface
// (matches the CPU HuffmanZSTDCoder.h API used by particle_compression.cu)
// ============================================================================

// Device-pointer version: data already on GPU, no extra copy
template <typename T>
std::vector<uint8_t>
huffmanZstdCompressDevice(const T *d_data, size_t N,
                          std::unordered_map<T, std::pair<uint32_t, int>> &codeTableOut,
                          size_t &bitstreamSizeOut, int zstdLevel = 3) {
  if (N == 0)
    return {};

  HuffmanZstdCompressor<T> compressor;
  auto result = compressor.compress(d_data, N, zstdLevel, false);

  codeTableOut.clear();
  codeTableOut.reserve(result.num_symbols);
  for (int i = 0; i < result.num_symbols; ++i) {
    codeTableOut[result.symbol_table[i]] = {result.code_table[i],
                                            (int)result.length_table[i]};
  }
  bitstreamSizeOut = result.original_bitstream_size;

  std::vector<uint8_t> compressed(result.compressed_data,
                                   result.compressed_data + result.compressed_size);
  result.freeSymbolTables();
  result.freeCompressedData();
  return compressed;
}

// Host-vector version: copies to device internally
template <typename T>
std::vector<uint8_t>
huffmanZstdCompress(const std::vector<T> &data,
                    std::unordered_map<T, std::pair<uint32_t, int>> &codeTableOut,
                    size_t &bitstreamSizeOut, int zstdLevel = 3) {
  if (data.empty())
    return {};

  T *d_data;
  CHECK_CUDA(cudaMalloc(&d_data, data.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_data, data.data(), data.size() * sizeof(T),
                        cudaMemcpyHostToDevice));

  auto result = huffmanZstdCompressDevice(d_data, data.size(), codeTableOut,
                                          bitstreamSizeOut, zstdLevel);
  CHECK_CUDA(cudaFree(d_data));
  return result;
}

template <typename T>
std::vector<T>
huffmanZstdDecompress(const std::vector<uint8_t> &compressed,
                      const std::unordered_map<T, std::pair<uint32_t, int>> &codeTable,
                      size_t numElements, size_t originalBitstreamSize) {
  if (compressed.empty() || numElements == 0)
    return {};

  int num_symbols = static_cast<int>(codeTable.size());
  typename HuffmanZstdCompressor<T>::CompressionResult result;
  result.compressed_data = new uint8_t[compressed.size()];
  std::copy(compressed.begin(), compressed.end(), result.compressed_data);
  result.compressed_size = compressed.size();
  result.num_symbols = num_symbols;
  result.original_bitstream_size = originalBitstreamSize;
  result.num_elements = numElements;

  result.symbol_table = new T[num_symbols];
  result.code_table = new uint32_t[num_symbols];
  result.length_table = new uint8_t[num_symbols];
  int i = 0;
  for (const auto &[sym, code_len] : codeTable) {
    result.symbol_table[i] = sym;
    result.code_table[i] = code_len.first;
    result.length_table[i] = static_cast<uint8_t>(code_len.second);
    ++i;
  }

  T *d_output;
  CHECK_CUDA(cudaMalloc(&d_output, numElements * sizeof(T)));
  HuffmanZstdCompressor<T> compressor;
  compressor.decompress(result, d_output, false);

  std::vector<T> output(numElements);
  CHECK_CUDA(cudaMemcpy(output.data(), d_output, numElements * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_output));

  result.freeSymbolTables();
  result.freeCompressedData();
  return output;
}