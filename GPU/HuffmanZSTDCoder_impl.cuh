#pragma once

#include <algorithm>
#include <cub/cub.cuh>
#include <queue>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <unordered_map>
#include <vector>
#include <zstd.h>

// ============================================================================
// CUDA Kernels
// ============================================================================

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

__device__ void atomicOrByte(uint8_t *address, uint8_t val) {
  unsigned int *base_addr = (unsigned int *)((size_t)address & ~3);
  unsigned int byte_offset = (size_t)address & 3;
  unsigned int shift = byte_offset * 8;
  unsigned int mask = (unsigned int)val << shift;
  atomicOr(base_addr, mask);
}

__global__ void packBytesAligned(const uint32_t *__restrict__ codes,
                                 const uint8_t *__restrict__ lengths,
                                 const size_t *__restrict__ bit_positions,
                                 uint8_t *__restrict__ output, size_t N) {

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  uint32_t code = codes[idx];
  uint8_t len = lengths[idx];
  size_t bit_start = bit_positions[idx];

  for (int i = 0; i < len; ++i) {
    size_t bit_pos = bit_start + i;
    size_t byte_idx = bit_pos >> 3;
    int bit_idx = 7 - (bit_pos & 7);

    if ((code >> (len - 1 - i)) & 1) {
      atomicOrByte(&output[byte_idx], (uint8_t)(1 << bit_idx));
    }
  }
}

// ============================================================================
// HuffmanNode Implementation
// ============================================================================

template <typename T>
HuffmanNode<T>::HuffmanNode() : value(T()), freq(0), left(-1), right(-1) {}

template <typename T>
HuffmanNode<T>::HuffmanNode(T v, uint32_t f)
    : value(v), freq(f), left(-1), right(-1) {}

template <typename T> bool HuffmanNode<T>::is_leaf() const {
  return left == -1 && right == -1;
}

// ============================================================================
// CompressionResult Implementation
// ============================================================================

template <typename T>
void HuffmanZstdCompressor<T>::CompressionResult::freeSymbolTables() {
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

template <typename T>
void HuffmanZstdCompressor<T>::CompressionResult::freeCompressedData() {
  if (compressed_data)
    delete[] compressed_data;
  compressed_data = nullptr;
}

// ============================================================================
// Frequency Counting - Histogram-based (uint8_t)
// ============================================================================

template <typename T>
template <typename U>
typename std::enable_if<sizeof(U) == 1, void>::type
HuffmanZstdCompressor<T>::countFrequencies(const T *d_data, size_t N,
                                           T *&d_unique, uint32_t *&d_counts,
                                           int &num_unique) {

  const int MAX_BINS = 256;
  uint32_t *d_histogram;
  CHECK_CUDA(cudaMalloc(&d_histogram, MAX_BINS * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemset(d_histogram, 0, MAX_BINS * sizeof(uint32_t)));

  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  int num_levels = MAX_BINS + 1;
  T lower_level = 0;
  T upper_level = MAX_BINS;

  cub::DeviceHistogram::HistogramEven(d_temp, temp_bytes, d_data, d_histogram,
                                      num_levels, lower_level, upper_level, N);
  CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
  cub::DeviceHistogram::HistogramEven(d_temp, temp_bytes, d_data, d_histogram,
                                      num_levels, lower_level, upper_level, N);
  CHECK_CUDA(cudaFree(d_temp));

  uint32_t *h_histogram = new uint32_t[MAX_BINS];
  CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, MAX_BINS * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  num_unique = 0;
  for (int i = 0; i < MAX_BINS; ++i) {
    if (h_histogram[i] > 0)
      num_unique++;
  }

  T *h_symbols = new T[num_unique];
  uint32_t *h_counts = new uint32_t[num_unique];

  int idx = 0;
  for (int i = 0; i < MAX_BINS; ++i) {
    if (h_histogram[i] > 0) {
      h_symbols[idx] = static_cast<T>(i);
      h_counts[idx] = h_histogram[i];
      idx++;
    }
  }

  CHECK_CUDA(cudaMalloc(&d_unique, num_unique * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_counts, num_unique * sizeof(uint32_t)));
  CHECK_CUDA(cudaMemcpy(d_unique, h_symbols, num_unique * sizeof(T),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_counts, h_counts, num_unique * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  delete[] h_histogram;
  delete[] h_symbols;
  delete[] h_counts;
  CHECK_CUDA(cudaFree(d_histogram));
}

// ============================================================================
// Frequency Counting - Sort-based (larger types)
// ============================================================================

template <typename T>
template <typename U>
typename std::enable_if<(sizeof(U) > 1), void>::type
HuffmanZstdCompressor<T>::countFrequencies(const T *d_data, size_t N,
                                           T *&d_unique, uint32_t *&d_counts,
                                           int &num_unique) {

  T *d_sorted;
  CHECK_CUDA(cudaMalloc(&d_sorted, N * sizeof(T)));

  void *d_temp = nullptr;
  size_t temp_bytes = 0;

  cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_data, d_sorted, N);

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
  cub::DeviceRunLengthEncode::Encode(d_temp, temp_bytes_rle, d_sorted, d_unique,
                                     d_counts, d_num_runs, N);

  CHECK_CUDA(
      cudaMemcpy(&num_unique, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_temp));
  CHECK_CUDA(cudaFree(d_sorted));
  CHECK_CUDA(cudaFree(d_num_runs));
}

// ============================================================================
// Huffman Tree Building
// ============================================================================

template <typename T>
void HuffmanZstdCompressor<T>::buildHuffmanTree(const T *symbols,
                                                const uint32_t *freqs, int n,
                                                HuffmanNode<T> *&nodes,
                                                int &root_idx) {

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

// ============================================================================
// Code Generation
// ============================================================================

template <typename T>
void HuffmanZstdCompressor<T>::generateCodes(
    const HuffmanNode<T> *nodes, int node_idx,
    std::unordered_map<T, std::pair<uint32_t, int>> &code_map, uint32_t code,
    int length) {

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

// ============================================================================
// Compression
// ============================================================================

template <typename T>
typename HuffmanZstdCompressor<T>::CompressionResult
HuffmanZstdCompressor<T>::compress(const T *d_data, size_t N, int zstd_level,
                                   bool verbose) {

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

  // Count frequencies
  T *d_unique;
  uint32_t *d_counts;
  int num_unique;

  countFrequencies(d_data, N, d_unique, d_counts, num_unique);

  // Copy to host for tree building
  T *h_symbols = new T[num_unique];
  uint32_t *h_freqs = new uint32_t[num_unique];

  CHECK_CUDA(cudaMemcpy(h_symbols, d_unique, num_unique * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_freqs, d_counts, num_unique * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(d_unique));
  CHECK_CUDA(cudaFree(d_counts));

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
  CHECK_CUDA(cudaMemcpy(d_lengths_lut, h_lengths, num_unique * sizeof(uint8_t),
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

  // Compute bit positions
  size_t *d_bit_lens, *d_bit_positions;
  CHECK_CUDA(cudaMalloc(&d_bit_lens, N * sizeof(size_t)));
  CHECK_CUDA(cudaMalloc(&d_bit_positions, N * sizeof(size_t)));

  thrust::device_ptr<uint8_t> d_len_ptr(d_lengths);
  thrust::device_ptr<size_t> d_bitlen_ptr(d_bit_lens);
  thrust::copy(d_len_ptr, d_len_ptr + N, d_bitlen_ptr);

  void *d_temp = nullptr;
  size_t temp_bytes = 0;

  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_bit_lens, d_bit_positions,
                                N);
  CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_bit_lens, d_bit_positions,
                                N);
  CHECK_CUDA(cudaFree(d_temp));

  size_t last_pos, last_len;
  CHECK_CUDA(cudaMemcpy(&last_pos, &d_bit_positions[N - 1], sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&last_len, &d_bit_lens[N - 1], sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  size_t total_bits = last_pos + last_len;
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

  // Copy to host with pinned memory
  uint8_t *h_bitstream;
  CHECK_CUDA(cudaMallocHost(&h_bitstream, total_bytes));
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

  size_t compressed_size = ZSTD_compress(h_compressed, zstd_bound, h_bitstream,
                                         total_bytes, zstd_level);

  if (ZSTD_isError(compressed_size)) {
    CHECK_CUDA(cudaFreeHost(h_bitstream));
    delete[] h_compressed;
    delete[] h_symbols;
    delete[] h_codes;
    delete[] h_lengths;
    throw std::runtime_error("ZSTD compression failed");
  }

  uint8_t *h_compressed_final = new uint8_t[compressed_size];
  std::copy(h_compressed, h_compressed + compressed_size, h_compressed_final);
  delete[] h_compressed;
  CHECK_CUDA(cudaFreeHost(h_bitstream));

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
  CHECK_CUDA(cudaFree(d_bit_lens));
  CHECK_CUDA(cudaFree(d_bit_positions));
  CHECK_CUDA(cudaFree(d_packed));

  if (verbose) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return {h_compressed_final, compressed_size, h_symbols,   h_codes,
          h_lengths,          num_unique,      total_bytes, N};
}

// ============================================================================
// Decompression
// ============================================================================

template <typename T>
void HuffmanZstdCompressor<T>::decompress(const CompressionResult &result,
                                          T *d_output, bool verbose) {

  cudaEvent_t start, stop;
  float ms;

  if (verbose) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  // ZSTD decompression with pinned memory
  uint8_t *h_bitstream;
  CHECK_CUDA(cudaMallocHost(&h_bitstream, result.original_bitstream_size));

  size_t decompressed_size =
      ZSTD_decompress(h_bitstream, result.original_bitstream_size,
                      result.compressed_data, result.compressed_size);

  if (ZSTD_isError(decompressed_size)) {
    CHECK_CUDA(cudaFreeHost(h_bitstream));
    throw std::runtime_error("ZSTD decompression failed");
  }

  if (verbose) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  ZSTD decompress: %.2f ms\n", ms);
    cudaEventRecord(start);
  }

  // Build reverse lookup
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

  for (int len = 1; len <= 32; ++len) {
    std::sort(
        reverse_lookup[len].begin(), reverse_lookup[len].end(),
        [](const CodeEntry &a, const CodeEntry &b) { return a.code < b.code; });
  }

  if (verbose) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("  Build reverse lookup: %.2f ms\n", ms);
    cudaEventRecord(start);
  }

  // Decode with pinned memory
  T *h_output;
  CHECK_CUDA(cudaMallocHost(&h_output, result.num_elements * sizeof(T)));
  size_t output_idx = 0;
  size_t bit_pos = 0;
  size_t total_bits = result.original_bitstream_size * 8;

  const uint8_t *bitstream = h_bitstream;

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
        code =
            (bitstream[byte_idx] >> (8 - bit_offset - len)) & ((1u << len) - 1);
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
      CHECK_CUDA(cudaFreeHost(h_bitstream));
      CHECK_CUDA(cudaFreeHost(h_output));
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

  CHECK_CUDA(cudaFreeHost(h_bitstream));
  CHECK_CUDA(cudaFreeHost(h_output));

  if (verbose) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
}

// ============================================================================
// Convenience Functions
// ============================================================================

template <typename T>
typename HuffmanZstdCompressor<T>::CompressionResult
compressHuffmanZstd(const T *d_data, size_t N, int zstd_level, bool verbose) {
  HuffmanZstdCompressor<T> compressor;
  return compressor.compress(d_data, N, zstd_level, verbose);
}

template <typename T>
void decompressHuffmanZstd(
    const typename HuffmanZstdCompressor<T>::CompressionResult &result,
    T *d_output, bool verbose) {
  HuffmanZstdCompressor<T> compressor;
  compressor.decompress(result, d_output, verbose);
}