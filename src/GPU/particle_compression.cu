#include "FOFHaloFinder.cuh"
#include "particle_compression.cuh"
#include <chrono>
#include <climits>
#include <cstring>

// Globals defined in main.cu/main_mpi.cu
extern double lr;
extern int max_iter;
extern FoFConstraintStrategy fof_constraint_strategy;
extern int target_cell_occupancy;

static size_t editCodeTableBytes(size_t entries) {
  return entries * (sizeof(UInt2) + sizeof(uint32_t) + sizeof(int));
}

static std::vector<uint8_t> copyQuantizedEditsDevice(const UInt2 *d_data,
                                                     size_t count) {
  std::vector<uint8_t> bytes(count * sizeof(UInt2));
  if (!bytes.empty()) {
    CUDA_CHECK(cudaMemcpy(bytes.data(), d_data, bytes.size(),
                          cudaMemcpyDeviceToHost));
  }
  return bytes;
}

template <typename T>
static void compressQuantizedEditsDevice(const UInt2 *d_quant_edits,
                                         size_t num_edit_values,
                                         CompressedData<T> &compressed) {
  compressed.compressed_quant_edits.clear();
  compressed.code_table_edit.clear();
  compressed.bit_stream_size_edit = 0;
  if (num_edit_values == 0)
    return;

  std::unordered_map<UInt2, std::pair<uint32_t, int>> huffman_table;
  size_t huffman_bitstream_bytes = 0;
  std::vector<uint8_t> huffman_payload =
      huffmanCompressDevice(d_quant_edits, num_edit_values, huffman_table,
                            huffman_bitstream_bytes);

  size_t raw_bytes = num_edit_values * sizeof(UInt2);
  size_t huffman_total =
      huffman_payload.size() + editCodeTableBytes(huffman_table.size());
  if (huffman_total < raw_bytes) {
    compressed.compressed_quant_edits = std::move(huffman_payload);
    compressed.code_table_edit = std::move(huffman_table);
    compressed.bit_stream_size_edit = huffman_bitstream_bytes;
  } else {
    compressed.compressed_quant_edits =
        copyQuantizedEditsDevice(d_quant_edits, num_edit_values);
  }
}

template <typename T>
static std::vector<UInt2>
decompressQuantizedEdits(const CompressedData<T> &compressed) {
  if (compressed.size_edit == 0)
    return {};

  size_t raw_bytes = compressed.size_edit * sizeof(UInt2);
  if (compressed.code_table_edit.empty()) {
    if (compressed.compressed_quant_edits.size() != raw_bytes)
      throw std::runtime_error("Raw PGD edit stream has invalid size");
    std::vector<UInt2> quant_edits(compressed.size_edit);
    std::memcpy(quant_edits.data(), compressed.compressed_quant_edits.data(),
                raw_bytes);
    return quant_edits;
  }

  if (compressed.compressed_quant_edits.size() ==
      compressed.bit_stream_size_edit) {
    return huffmanDecompress(compressed.compressed_quant_edits,
                             compressed.code_table_edit, compressed.size_edit,
                             compressed.bit_stream_size_edit);
  }

  return huffmanZstdDecompress(compressed.compressed_quant_edits,
                               compressed.code_table_edit, compressed.size_edit,
                               compressed.bit_stream_size_edit);
}

static inline int maxCellsForTargetOccupancy(int N, int default_occupancy) {
  int occupancy =
      target_cell_occupancy > 0 ? target_cell_occupancy : default_occupancy;
  long long max_cells = (static_cast<long long>(N) + occupancy - 1) / occupancy;
  if (max_cells > INT_MAX)
    return INT_MAX;
  return std::max(1, static_cast<int>(max_cells));
}

template <typename T> static inline int gridDimForRange(T range, T grid_len) {
  if (range <= T(0))
    return 1;
  int dim = static_cast<int>(std::ceil(range / grid_len));
  return std::max(1, dim);
}

template <typename T>
static inline long long cellCount2D(T range_x, T range_y, T grid_len,
                                    int &dim_x, int &dim_y) {
  dim_x = gridDimForRange(range_x, grid_len);
  dim_y = gridDimForRange(range_y, grid_len);
  return (long long)dim_x * dim_y;
}

template <typename T>
static inline long long cellCount3D(T range_x, T range_y, T range_z, T grid_len,
                                    int &dim_x, int &dim_y, int &dim_z) {
  dim_x = gridDimForRange(range_x, grid_len);
  dim_y = gridDimForRange(range_y, grid_len);
  dim_z = gridDimForRange(range_z, grid_len);
  return (long long)dim_x * dim_y * dim_z;
}

// Coarsen grid if num_cells exceeds max_cells budget
template <typename T>
static void coarsenGrid2D(T &grid_len, int &grid_dim_x, int &grid_dim_y,
                          T range_x, T range_y, int max_cells) {
  long long nc = (long long)grid_dim_x * grid_dim_y;
  if (nc <= max_cells)
    return;

  T original_grid_len = grid_len;
  T scale = std::sqrt(static_cast<T>(nc) / max_cells);
  T low = original_grid_len;
  T high = original_grid_len * scale;
  int dim_x = grid_dim_x;
  int dim_y = grid_dim_y;
  long long nc_new = cellCount2D(range_x, range_y, high, dim_x, dim_y);

  while (nc_new > max_cells) {
    low = high;
    high *= static_cast<T>(2);
    nc_new = cellCount2D(range_x, range_y, high, dim_x, dim_y);
  }

  for (int iter = 0; iter < 32; ++iter) {
    T mid = (low + high) * static_cast<T>(0.5);
    int mid_x, mid_y;
    long long mid_cells = cellCount2D(range_x, range_y, mid, mid_x, mid_y);
    if (mid_cells <= max_cells) {
      high = mid;
      dim_x = mid_x;
      dim_y = mid_y;
      nc_new = mid_cells;
    } else {
      low = mid;
    }
  }

  grid_len = high;
  grid_dim_x = dim_x;
  grid_dim_y = dim_y;
  printf("Grid coarsened: %lld -> %lld cells (grid_len %e, factor %.2fx)\n", nc,
         nc_new, (double)grid_len, (double)(grid_len / original_grid_len));
}

template <typename T>
static void coarsenGrid3D(T &grid_len, int &grid_dim_x, int &grid_dim_y,
                          int &grid_dim_z, T range_x, T range_y, T range_z,
                          int max_cells) {
  long long nc = (long long)grid_dim_x * grid_dim_y * grid_dim_z;
  if (nc <= max_cells)
    return;

  T original_grid_len = grid_len;
  T scale = std::cbrt(static_cast<T>(nc) / max_cells);
  T low = original_grid_len;
  T high = original_grid_len * scale;
  int dim_x = grid_dim_x;
  int dim_y = grid_dim_y;
  int dim_z = grid_dim_z;
  long long nc_new =
      cellCount3D(range_x, range_y, range_z, high, dim_x, dim_y, dim_z);

  while (nc_new > max_cells) {
    low = high;
    high *= static_cast<T>(2);
    nc_new = cellCount3D(range_x, range_y, range_z, high, dim_x, dim_y, dim_z);
  }

  for (int iter = 0; iter < 32; ++iter) {
    T mid = (low + high) * static_cast<T>(0.5);
    int mid_x, mid_y, mid_z;
    long long mid_cells =
        cellCount3D(range_x, range_y, range_z, mid, mid_x, mid_y, mid_z);
    if (mid_cells <= max_cells) {
      high = mid;
      dim_x = mid_x;
      dim_y = mid_y;
      dim_z = mid_z;
      nc_new = mid_cells;
    } else {
      low = mid;
    }
  }

  grid_len = high;
  grid_dim_x = dim_x;
  grid_dim_y = dim_y;
  grid_dim_z = dim_z;
  printf("Grid coarsened: %lld -> %lld cells (grid_len %e, factor %.2fx)\n", nc,
         nc_new, (double)grid_len, (double)(grid_len / original_grid_len));
}

// Helpers
std::vector<uint8_t> packBits(const std::vector<bool> &bits) {
  std::vector<uint8_t> packed;
  packed.reserve((bits.size() + 7) / 8);
  uint8_t curr_byte = 0;
  int bit_pos = 0;
  for (bool bit : bits) {
    if (bit)
      curr_byte |= (1 << (7 - bit_pos));
    if (++bit_pos == 8) {
      packed.push_back(curr_byte);
      curr_byte = 0;
      bit_pos = 0;
    }
  }
  if (bit_pos > 0)
    packed.push_back(curr_byte);
  return packed;
}

std::vector<bool> unpackBits(const std::vector<uint8_t> packed, int num_bits) {
  if (num_bits == 0)
    num_bits = static_cast<int>(packed.size() * 8);
  std::vector<bool> unpacked;
  unpacked.reserve(num_bits);
  int bit_pos = 0;
  for (uint8_t byte : packed) {
    for (int i = 7; i >= 0 && bit_pos < num_bits; --i) {
      unpacked.push_back((byte >> i) & 1);
      ++bit_pos;
    }
    if (bit_pos >= num_bits)
      break;
  }
  return unpacked;
}

// Helper: Compact pass
template <typename T>
void compactPass(const bool *d_lossless_flag, const UInt *d_temp_qcode,
                 const T *d_temp_lval, const int *d_cell_start,
                 const int *d_cell_quant_count,
                 const int *d_cell_lossless_count, int num_cells, int N, int D,
                 UInt **d_quant_codes_out, T **d_lossless_values_out,
                 int &num_quant_codes, int &num_lossless_values) {
  // Prefix sum to get offsets
  int *d_quant_offsets, *d_lossless_offsets;
  CUDA_CHECK(cudaMalloc(&d_quant_offsets, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_lossless_offsets, num_cells * sizeof(int)));

  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_cell_quant_count,
                                d_quant_offsets, num_cells);
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_cell_quant_count,
                                d_quant_offsets, num_cells);
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_cell_lossless_count,
                                d_lossless_offsets, num_cells);
  CUDA_CHECK(cudaFree(d_temp));

  // Get total sizes
  int last_quant_offset, last_quant_count;
  int last_lossless_offset, last_lossless_count;
  CUDA_CHECK(cudaMemcpy(&last_quant_offset, &d_quant_offsets[num_cells - 1],
                        sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last_quant_count, &d_cell_quant_count[num_cells - 1],
                        sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last_lossless_offset,
                        &d_lossless_offsets[num_cells - 1], sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last_lossless_count,
                        &d_cell_lossless_count[num_cells - 1], sizeof(int),
                        cudaMemcpyDeviceToHost));

  num_quant_codes = last_quant_offset + last_quant_count;
  num_lossless_values = last_lossless_offset + last_lossless_count;

  // Allocate output arrays
  CUDA_CHECK(cudaMalloc(d_quant_codes_out, num_quant_codes * sizeof(UInt)));
  if (num_lossless_values > 0) {
    CUDA_CHECK(
        cudaMalloc(d_lossless_values_out, num_lossless_values * sizeof(T)));
  } else {
    *d_lossless_values_out = nullptr;
  }

  // Compact
  int num_blocks = (num_cells + num_threads - 1) / num_threads;
  compactCellCompressionOutputs_kernel<T><<<num_blocks, num_threads>>>(
      d_lossless_flag, d_temp_qcode, d_temp_lval, d_cell_start, d_quant_offsets,
      d_lossless_offsets, num_cells, N, D, *d_quant_codes_out,
      *d_lossless_values_out);

  CUDA_CHECK(cudaFree(d_quant_offsets));
  CUDA_CHECK(cudaFree(d_lossless_offsets));
}

static const char *fofConstraintStrategyName(FoFConstraintStrategy strategy) {
  switch (strategy) {
  case FoFConstraintStrategy::PAIRWISE_VULNERABILITY:
    return "1/pairwise-vulnerability";
  case FoFConstraintStrategy::SAFE_COMPONENT_FILTERING:
    return "2/safe-component-filtering";
  case FoFConstraintStrategy::CONTRACTED_HALO_FOREST:
    return "3/contracted-halo-forest";
  }
  return "unknown";
}

static inline bool needsSafeComponentRoots(FoFConstraintStrategy strategy) {
  return strategy != FoFConstraintStrategy::PAIRWISE_VULNERABILITY;
}

static inline bool needsOriginalHaloRoots(FoFConstraintStrategy strategy) {
  return strategy == FoFConstraintStrategy::CONTRACTED_HALO_FOREST;
}

__device__ __forceinline__ int find_root_readonly(int *parent, int x) {
  int p = parent[x];
  while (p != x) {
    x = p;
    p = parent[x];
  }
  return x;
}

__device__ __forceinline__ bool atomicUnionMin(int *parent, int x, int y) {
  while (true) {
    x = find_root_readonly(parent, x);
    y = find_root_readonly(parent, y);
    if (x == y)
      return false;
    int hi = max(x, y);
    int lo = min(x, y);
    int old = atomicCAS(&parent[hi], hi, lo);
    if (old == hi)
      return true;
  }
}

__global__ void markSafeComponentFilteringPairs_kernel(const int2 *pairs,
                                                       int num_pairs,
                                                       const int *safe_roots,
                                                       bool *keep) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pairs)
    return;
  int2 pair = pairs[idx];
  int p = pair.x;
  int q = pair.y;
  keep[idx] = safe_roots[p] != safe_roots[q];
}

__global__ void markContractedHaloForestPairs_kernel(
    const int2 *pairs, const bool *signs, int num_pairs, const int *safe_roots,
    const int *halo_roots, int *supernode_parent, bool *keep) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pairs)
    return;

  int2 pair = pairs[idx];
  int p = pair.x;
  int q = pair.y;
  int safe_p = safe_roots[p];
  int safe_q = safe_roots[q];

  // Potentially-created link: preserve only if it bridges halos
  if (signs[idx]) {
    keep[idx] = halo_roots[p] != halo_roots[q];
    return;
  }

  // Potentially-broken link within the same supernode: remove
  if (safe_p == safe_q) {
    keep[idx] = false;
    return;
  }

  // Potentially-broken links between two supernodes: keep a
  // spanning forest. Links are chosen by parallel edge order & atomic race;
  // same number of preserved links with MST
  keep[idx] = atomicUnionMin(supernode_parent, safe_p, safe_q);
}

static void makeIdentityRoots(int N, int **d_roots_out) {
  *d_roots_out = nullptr;
  if (N == 0)
    return;
  CUDA_CHECK(cudaMalloc(d_roots_out, (size_t)N * sizeof(int)));
  int blocks = (N + num_threads - 1) / num_threads;
  iota_kernel<<<blocks, num_threads>>>(*d_roots_out, N);
  CUDA_CHECK(cudaPeekAtLastError());
}

static void flattenRootsInPlace(int *d_roots, int N) {
  if (!d_roots || N == 0)
    return;
  int blocks = (N + num_threads - 1) / num_threads;
  flattenRoots_kernel<<<blocks, num_threads>>>(d_roots, d_roots, N);
  CUDA_CHECK(cudaPeekAtLastError());
}

struct BoolToIntOp {
  __host__ __device__ int operator()(bool v) const { return v ? 1 : 0; }
};

static void exclusiveScanBoolMask(const bool *d_mask, int N,
                                  int **d_offsets_out) {
  *d_offsets_out = nullptr;
  if (N == 0)
    return;

  CUDA_CHECK(cudaMalloc(d_offsets_out, (size_t)N * sizeof(int)));

  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  auto mask_as_int = thrust::make_transform_iterator(d_mask, BoolToIntOp{});
  cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, mask_as_int,
                                *d_offsets_out, N);
  if (temp_bytes > 0)
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, mask_as_int, *d_offsets_out,
                                N);
  if (d_temp)
    CUDA_CHECK(cudaFree(d_temp));
}

static int countTrueFlags(const bool *d_flags, int num_flags) {
  if (num_flags == 0)
    return 0;

  int *d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));

  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  auto flags_as_int = thrust::make_transform_iterator(d_flags, BoolToIntOp{});
  cub::DeviceReduce::Sum(nullptr, temp_bytes, flags_as_int, d_count, num_flags);
  if (temp_bytes > 0)
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  cub::DeviceReduce::Sum(d_temp, temp_bytes, flags_as_int, d_count, num_flags);

  int h_count = 0;
  CUDA_CHECK(
      cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (d_temp)
    CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_count));
  return h_count;
}

template <typename T>
static void compressEditMaskDevice(bool *d_edit_mask, int N,
                                   CompressedData<T> &compressed) {
  compressed.compressed_lossless_edit_flag.clear();
  compressed.code_table_edit_flag.clear();
  compressed.size_edit_flag = 0;
  compressed.bit_stream_size_edit_flag = 0;

  if (N == 0)
    return;

  int num_mask_bytes = (N + 7) / 8;
  uint8_t *d_packed_mask = nullptr;
  CUDA_CHECK(cudaMalloc(&d_packed_mask, (size_t)num_mask_bytes));
  int blocks = (num_mask_bytes + num_threads - 1) / num_threads;
  packLosslessFlags_kernel<<<blocks, num_threads>>>(d_edit_mask, d_packed_mask,
                                                    N, num_mask_bytes);
  CUDA_CHECK(cudaPeekAtLastError());

  compressed.size_edit_flag = num_mask_bytes;
  compressed.compressed_lossless_edit_flag = huffmanZstdCompressDevice(
      d_packed_mask, num_mask_bytes, compressed.code_table_edit_flag,
      compressed.bit_stream_size_edit_flag);
  CUDA_CHECK(cudaFree(d_packed_mask));
}

template <typename T>
static void clearCompressedEdits(CompressedData<T> &compressed) {
  compressed.size_edit = 0;
  compressed.bit_stream_size_edit = 0;
  compressed.compressed_quant_edits.clear();
  compressed.code_table_edit.clear();
  compressEditMaskDevice<T>(nullptr, 0, compressed);
}

static void compactActivePairs(int2 **d_pairs, bool **d_signs, int &num_pairs,
                               bool *d_keep) {
  if (num_pairs == 0)
    return;

  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  int h_count = countTrueFlags(d_keep, num_pairs);

  int *d_num_selected = nullptr;
  int2 *d_new_pairs = nullptr;
  bool *d_new_signs = nullptr;
  if (h_count > 0) {
    CUDA_CHECK(cudaMalloc(&d_num_selected, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_pairs, (size_t)h_count * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_new_signs, (size_t)h_count * sizeof(bool)));

    size_t pair_temp_bytes = 0;
    size_t sign_temp_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, pair_temp_bytes, *d_pairs, d_keep,
                               d_new_pairs, d_num_selected, num_pairs);
    cub::DeviceSelect::Flagged(nullptr, sign_temp_bytes, *d_signs, d_keep,
                               d_new_signs, d_num_selected, num_pairs);

    temp_bytes =
        pair_temp_bytes > sign_temp_bytes ? pair_temp_bytes : sign_temp_bytes;
    if (temp_bytes > 0)
      CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    cub::DeviceSelect::Flagged(d_temp, temp_bytes, *d_pairs, d_keep,
                               d_new_pairs, d_num_selected, num_pairs);
#ifdef DEBUG_CHECKS
    int h_pair_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_pair_count, d_num_selected, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (h_pair_count != h_count) {
      fprintf(stderr, "WARNING: compacted pair count %d != reduced count %d\n",
              h_pair_count, h_count);
    }
#endif
    cub::DeviceSelect::Flagged(d_temp, temp_bytes, *d_signs, d_keep,
                               d_new_signs, d_num_selected, num_pairs);

#ifdef DEBUG_CHECKS
    int h_sign_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_sign_count, d_num_selected, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (h_sign_count != h_count) {
      fprintf(stderr, "WARNING: compacted sign count %d != pair count %d\n",
              h_sign_count, h_count);
    }
#endif
  }

  CUDA_CHECK(cudaFree(*d_pairs));
  CUDA_CHECK(cudaFree(*d_signs));
  if (d_temp)
    CUDA_CHECK(cudaFree(d_temp));
  if (d_num_selected)
    CUDA_CHECK(cudaFree(d_num_selected));
  *d_pairs = d_new_pairs;
  *d_signs = d_new_signs;
  num_pairs = h_count;
}

template <typename T>
static void applyFoFConstraintStrategy2D(CompressionState2D<T> &state,
                                         const int *d_safe_roots,
                                         const int *d_halo_roots) {
  FoFConstraintStrategy strategy = fof_constraint_strategy;
  if (strategy == FoFConstraintStrategy::PAIRWISE_VULNERABILITY)
    return;

  int num_vul_pairs = state.num_vulnerable_pairs;
  if (num_vul_pairs == 0) {
    printf("Number of vulnerable pairs: 0\n");
    return;
  }

  if (!d_safe_roots) {
    fprintf(stderr, "ERROR: FoF strategy %s requires safe-component roots.\n",
            fofConstraintStrategyName(strategy));
    return;
  }
  if (strategy == FoFConstraintStrategy::CONTRACTED_HALO_FOREST &&
      !d_halo_roots) {
    fprintf(stderr, "ERROR: FoF strategy %s requires original-halo roots.\n",
            fofConstraintStrategyName(strategy));
    return;
  }

  bool *d_keep = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keep, (size_t)num_vul_pairs * sizeof(bool)));
  int blocks = (num_vul_pairs + num_threads - 1) / num_threads;

  if (strategy == FoFConstraintStrategy::SAFE_COMPONENT_FILTERING) {
    markSafeComponentFilteringPairs_kernel<<<blocks, num_threads>>>(
        state.d_vulnerable_pairs, num_vul_pairs, d_safe_roots, d_keep);
  } else {
    int *d_supernode_parent = nullptr;
    makeIdentityRoots(state.N, &d_supernode_parent);
    markContractedHaloForestPairs_kernel<<<blocks, num_threads>>>(
        state.d_vulnerable_pairs, state.d_signs, num_vul_pairs, d_safe_roots,
        d_halo_roots, d_supernode_parent, d_keep);
    CUDA_CHECK(cudaFree(d_supernode_parent));
  }
  CUDA_CHECK(cudaPeekAtLastError());

  compactActivePairs(&state.d_vulnerable_pairs, &state.d_signs,
                     state.num_vulnerable_pairs, d_keep);
  printf("Number of vulnerable pairs: %d\n", num_vul_pairs);
  printf("Number of active pairs: %d\n", state.num_vulnerable_pairs);

  CUDA_CHECK(cudaFree(d_keep));
}

template <typename T>
static void applyFoFConstraintStrategy3D(CompressionState3D<T> &state,
                                         const int *d_safe_roots,
                                         const int *d_halo_roots) {
  FoFConstraintStrategy strategy = fof_constraint_strategy;
  if (strategy == FoFConstraintStrategy::PAIRWISE_VULNERABILITY) {
    printf("FoF constraint strategy: %s\n",
           fofConstraintStrategyName(strategy));
    return;
  }

  int num_vul_pairs = state.num_vulnerable_pairs;
  if (num_vul_pairs == 0) {
    printf("Number of vulnerable pairs: 0\n");
    printf("Number of active pairs: 0\n");
    return;
  }

  if (!d_safe_roots) {
    fprintf(stderr, "ERROR: FoF strategy %s requires safe-component roots.\n",
            fofConstraintStrategyName(strategy));
    return;
  }
  if (strategy == FoFConstraintStrategy::CONTRACTED_HALO_FOREST &&
      !d_halo_roots) {
    fprintf(stderr, "ERROR: FoF strategy %s requires original-halo roots.\n",
            fofConstraintStrategyName(strategy));
    return;
  }

  bool *d_keep = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keep, (size_t)num_vul_pairs * sizeof(bool)));
  int blocks = (num_vul_pairs + num_threads - 1) / num_threads;

  if (strategy == FoFConstraintStrategy::SAFE_COMPONENT_FILTERING) {
    markSafeComponentFilteringPairs_kernel<<<blocks, num_threads>>>(
        state.d_vulnerable_pairs, num_vul_pairs, d_safe_roots, d_keep);
  } else {
    int *d_supernode_parent = nullptr;
    makeIdentityRoots(state.N, &d_supernode_parent);
    markContractedHaloForestPairs_kernel<<<blocks, num_threads>>>(
        state.d_vulnerable_pairs, state.d_signs, num_vul_pairs, d_safe_roots,
        d_halo_roots, d_supernode_parent, d_keep);
    CUDA_CHECK(cudaFree(d_supernode_parent));
  }
  CUDA_CHECK(cudaPeekAtLastError());

  compactActivePairs(&state.d_vulnerable_pairs, &state.d_signs,
                     state.num_vulnerable_pairs, d_keep);
  printf("Number of vulnerable pairs: %d\n", num_vul_pairs);
  printf("Number of active pairs: %d\n", state.num_vulnerable_pairs);

  CUDA_CHECK(cudaFree(d_keep));
}

static void buildCellPairList2D(const int *d_cell_start, int num_cells, int N,
                                int grid_dim_x, int grid_dim_y,
                                int2 **d_cell_pairs_out,
                                int *num_cell_pairs_out) {
  *d_cell_pairs_out = nullptr;
  *num_cell_pairs_out = 0;

  int *d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  int blocks = (num_cells + num_threads - 1) / num_threads;
  countCellPairs2D_kernel<<<blocks, num_threads>>>(
      d_cell_start, num_cells, N, grid_dim_x, grid_dim_y, d_count);

  int h_count = 0;
  CUDA_CHECK(
      cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_count == 0) {
    CUDA_CHECK(cudaFree(d_count));
    return;
  }

  CUDA_CHECK(cudaMalloc(d_cell_pairs_out, (size_t)h_count * sizeof(int2)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  fillCellPairs2D_kernel<<<blocks, num_threads>>>(d_cell_start, num_cells, N,
                                                  grid_dim_x, grid_dim_y,
                                                  *d_cell_pairs_out, d_count);
  CUDA_CHECK(cudaPeekAtLastError());

#ifdef DEBUG_CHECKS
  int h_fill_count = 0;
  CUDA_CHECK(
      cudaMemcpy(&h_fill_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_fill_count != h_count) {
    fprintf(stderr, "WARNING: cell-pair fill count %d != count pass %d\n",
            h_fill_count, h_count);
  }
#endif

  *num_cell_pairs_out = h_count;
  CUDA_CHECK(cudaFree(d_count));
}

static void buildCellPairList3D(const int *d_cell_start, int num_cells, int N,
                                int grid_dim_x, int grid_dim_y, int grid_dim_z,
                                int2 **d_cell_pairs_out,
                                int *num_cell_pairs_out) {
  *d_cell_pairs_out = nullptr;
  *num_cell_pairs_out = 0;

  int *d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

  int blocks = (num_cells + num_threads - 1) / num_threads;
  countCellPairs3D_kernel<<<blocks, num_threads>>>(
      d_cell_start, num_cells, N, grid_dim_x, grid_dim_y, grid_dim_z, d_count);

  int h_count = 0;
  CUDA_CHECK(
      cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_count == 0) {
    CUDA_CHECK(cudaFree(d_count));
    return;
  }

  CUDA_CHECK(cudaMalloc(d_cell_pairs_out, (size_t)h_count * sizeof(int2)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  fillCellPairs3D_kernel<<<blocks, num_threads>>>(
      d_cell_start, num_cells, N, grid_dim_x, grid_dim_y, grid_dim_z,
      *d_cell_pairs_out, d_count);
  CUDA_CHECK(cudaPeekAtLastError());

#ifdef DEBUG_CHECKS
  int h_fill_count = 0;
  CUDA_CHECK(
      cudaMemcpy(&h_fill_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_fill_count != h_count) {
    fprintf(stderr, "WARNING: cell-pair fill count %d != count pass %d\n",
            h_fill_count, h_count);
  }
#endif

  *num_cell_pairs_out = h_count;
  CUDA_CHECK(cudaFree(d_count));
}

template <typename T>
void findVulnerablePairs2D(const T *d_org_xx, const T *d_org_yy,
                           const int *d_cell_start,
                           const int *d_cell_pts_sorted,
                           int2 **d_vulnerable_pairs_out, bool **d_signs_out,
                           int *num_vulnerable_pairs_out, int grid_dim_x,
                           int grid_dim_y, int N, T lower_bound_sq,
                           T upper_bound_sq, T sign_bound_sq,
                           int **d_safe_roots_out = nullptr,
                           int **d_halo_roots_out = nullptr, int N_local = 0) {
  *d_vulnerable_pairs_out = nullptr;
  *d_signs_out = nullptr;
  *num_vulnerable_pairs_out = 0;
  if (d_safe_roots_out)
    *d_safe_roots_out = nullptr;
  if (d_halo_roots_out)
    *d_halo_roots_out = nullptr;

  int num_cells = grid_dim_x * grid_dim_y;
  int2 *d_cell_pairs = nullptr;
  int num_cell_pairs = 0;
  buildCellPairList2D(d_cell_start, num_cells, N, grid_dim_x, grid_dim_y,
                      &d_cell_pairs, &num_cell_pairs);
  if (num_cell_pairs == 0)
    return;

  int *d_safe_roots = nullptr;
  int *d_halo_roots = nullptr;
  if (d_safe_roots_out)
    makeIdentityRoots(N, &d_safe_roots);
  if (d_halo_roots_out)
    makeIdentityRoots(N, &d_halo_roots);
  int *d_safe_union_roots =
      (d_safe_roots && lower_bound_sq > T(0)) ? d_safe_roots : nullptr;

  // Pass 1: count vulnerable pairs & build safe components and halos by union
  int *d_count;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairsCellPairs2D_kernel<true><<<num_cell_pairs, num_threads>>>(
      d_org_xx, d_org_yy, d_cell_start, d_cell_pts_sorted, d_cell_pairs,
      num_cell_pairs, num_cells, nullptr, nullptr, d_count, N, lower_bound_sq,
      upper_bound_sq, sign_bound_sq, d_safe_union_roots, d_halo_roots,
      lower_bound_sq, sign_bound_sq, 0, N_local);

  int h_num_vulnerable_pairs;
  CUDA_CHECK(cudaMemcpy(&h_num_vulnerable_pairs, d_count, sizeof(int),
                        cudaMemcpyDeviceToHost));
  *num_vulnerable_pairs_out = h_num_vulnerable_pairs;
  if (h_num_vulnerable_pairs == 0) {
    if (d_safe_roots)
      CUDA_CHECK(cudaFree(d_safe_roots));
    if (d_halo_roots)
      CUDA_CHECK(cudaFree(d_halo_roots));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_cell_pairs));
    return;
  }

  // Pass 2: allocate with margin to absorb any count mismatch between passes
  int alloc_pairs =
      h_num_vulnerable_pairs + h_num_vulnerable_pairs / 100 + 1024;
  CUDA_CHECK(
      cudaMalloc(d_vulnerable_pairs_out, (size_t)alloc_pairs * sizeof(int2)));
  CUDA_CHECK(cudaMalloc(d_signs_out, (size_t)alloc_pairs * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairsCellPairs2D_kernel<false><<<num_cell_pairs, num_threads>>>(
      d_org_xx, d_org_yy, d_cell_start, d_cell_pts_sorted, d_cell_pairs,
      num_cell_pairs, num_cells, *d_vulnerable_pairs_out, *d_signs_out, d_count,
      N, lower_bound_sq, upper_bound_sq, sign_bound_sq, nullptr, nullptr, T(0),
      T(0), alloc_pairs, N_local);
  CUDA_CHECK(cudaPeekAtLastError());

  int h_count2;
  CUDA_CHECK(
      cudaMemcpy(&h_count2, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_count2 != h_num_vulnerable_pairs) {
    printf("WARNING: pass2 count %d != pass1 count %d (delta %d)\n", h_count2,
           h_num_vulnerable_pairs, h_count2 - h_num_vulnerable_pairs);
  }
  *num_vulnerable_pairs_out = h_count2; // use actual pass2 count

  if (d_safe_roots) {
    flattenRootsInPlace(d_safe_roots, N);
    *d_safe_roots_out = d_safe_roots;
  }
  if (d_halo_roots) {
    flattenRootsInPlace(d_halo_roots, N);
    *d_halo_roots_out = d_halo_roots;
  }

  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_cell_pairs));
}

template <typename T>
void findVulnerablePairs3D(const T *d_org_xx, const T *d_org_yy,
                           const T *d_org_zz, const int *d_cell_start,
                           const int *d_cell_pts_sorted,
                           int2 **d_vulnerable_pairs_out, bool **d_signs_out,
                           int *num_vulnerable_pairs_out, int grid_dim_x,
                           int grid_dim_y, int grid_dim_z, int N,
                           T lower_bound_sq, T upper_bound_sq, T sign_bound_sq,
                           int **d_safe_roots_out = nullptr,
                           int **d_halo_roots_out = nullptr, int N_local = 0) {
  *d_vulnerable_pairs_out = nullptr;
  *d_signs_out = nullptr;
  *num_vulnerable_pairs_out = 0;
  if (d_safe_roots_out)
    *d_safe_roots_out = nullptr;
  if (d_halo_roots_out)
    *d_halo_roots_out = nullptr;

  int num_cells = grid_dim_x * grid_dim_y * grid_dim_z;
  int2 *d_cell_pairs = nullptr;
  int num_cell_pairs = 0;
  buildCellPairList3D(d_cell_start, num_cells, N, grid_dim_x, grid_dim_y,
                      grid_dim_z, &d_cell_pairs, &num_cell_pairs);
  if (num_cell_pairs == 0)
    return;

  int *d_safe_roots = nullptr;
  int *d_halo_roots = nullptr;
  if (d_safe_roots_out)
    makeIdentityRoots(N, &d_safe_roots);
  if (d_halo_roots_out)
    makeIdentityRoots(N, &d_halo_roots);
  int *d_safe_union_roots =
      (d_safe_roots && lower_bound_sq > T(0)) ? d_safe_roots : nullptr;

  // Pass 1: count & union
  int *d_count;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairsCellPairs3D_kernel<true><<<num_cell_pairs, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, d_cell_start, d_cell_pts_sorted,
      d_cell_pairs, num_cell_pairs, num_cells, nullptr, nullptr, d_count, N,
      lower_bound_sq, upper_bound_sq, sign_bound_sq, d_safe_union_roots,
      d_halo_roots, lower_bound_sq, sign_bound_sq, 0, N_local);

  int h_num_vulnerable_pairs;
  CUDA_CHECK(cudaMemcpy(&h_num_vulnerable_pairs, d_count, sizeof(int),
                        cudaMemcpyDeviceToHost));
  *num_vulnerable_pairs_out = h_num_vulnerable_pairs;
  if (h_num_vulnerable_pairs == 0) {
    if (d_safe_roots)
      CUDA_CHECK(cudaFree(d_safe_roots));
    if (d_halo_roots)
      CUDA_CHECK(cudaFree(d_halo_roots));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_cell_pairs));
    return;
  }

  // Pass 2: allocate with margin and populate
  int alloc_pairs =
      h_num_vulnerable_pairs + h_num_vulnerable_pairs / 100 + 1024;
  CUDA_CHECK(
      cudaMalloc(d_vulnerable_pairs_out, (size_t)alloc_pairs * sizeof(int2)));
  CUDA_CHECK(cudaMalloc(d_signs_out, (size_t)alloc_pairs * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairsCellPairs3D_kernel<false><<<num_cell_pairs, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, d_cell_start, d_cell_pts_sorted,
      d_cell_pairs, num_cell_pairs, num_cells, *d_vulnerable_pairs_out,
      *d_signs_out, d_count, N, lower_bound_sq, upper_bound_sq, sign_bound_sq,
      nullptr, nullptr, T(0), T(0), alloc_pairs, N_local);
  CUDA_CHECK(cudaPeekAtLastError());

  int h_count2;
  CUDA_CHECK(
      cudaMemcpy(&h_count2, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_count2 != h_num_vulnerable_pairs) {
    printf("WARNING: pass2 count %d != pass1 count %d (delta %d)\n", h_count2,
           h_num_vulnerable_pairs, h_count2 - h_num_vulnerable_pairs);
  }
  *num_vulnerable_pairs_out = h_count2;
  if (d_safe_roots) {
    flattenRootsInPlace(d_safe_roots, N);
    *d_safe_roots_out = d_safe_roots;
  }
  if (d_halo_roots) {
    flattenRootsInPlace(d_halo_roots, N);
    *d_halo_roots_out = d_halo_roots;
  }

  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_cell_pairs));
}

// ============================================================================
// Decompression functions
// (CPU based. GPU version needs storing per-cell offsets -> storage overhead)
// ============================================================================

template <typename T, OrderMode Mode>
void decompressWithEditParticles2D(const CompressedData<T> &compressed,
                                   T *decomp_xx, T *decomp_yy, int N, T xi,
                                   T b) {
  if (compressed.size_flag == 0)
    return;
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<uint8_t> packed_lossless_flag = huffmanZstdDecompress(
      compressed.compressed_lossless_flag, compressed.code_table_flag,
      compressed.size_flag, compressed.bit_stream_size_flag);
  std::vector<bool> lossless_flag = unpackBits(packed_lossless_flag, 2 * N);
  std::vector<UInt> quant_codes = huffmanZstdDecompress(
      compressed.compressed_quant_codes, compressed.code_table_quant,
      compressed.size_quant, compressed.bit_stream_size_quant);
  T grid_len = b + 2 * std::sqrt(2) * xi;
  T min_x = compressed.grid_min_x;
  T min_y = compressed.grid_min_y;

  int i_f{0}, i_q{0}, i_l{0}, i_out{0};
  T norm = (4 * xi) / ((1 << m) - 1);

  // Step 1: Decompress positions
  for (int id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
    for (int id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {
      T prev_x, prev_y;
      if constexpr (Mode == OrderMode::KD_TREE) {
        prev_x = min_x + (id_x + T(0.5)) * grid_len;
        prev_y = min_y + (id_y + T(0.5)) * grid_len;
      } else {
        prev_x = min_x + id_x * grid_len;
        prev_y = min_y + id_y * grid_len;
      }

      while (quant_codes[i_q] > 0) {
        if (lossless_flag[i_f++]) {
          prev_x = compressed.lossless_values[i_l++];
        } else {
          prev_x += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
        }
        decomp_xx[i_out] = prev_x;
        if (lossless_flag[i_f++]) {
          prev_y = compressed.lossless_values[i_l++];
        } else {
          prev_y += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
        }
        decomp_yy[i_out++] = prev_y;
        if (quant_codes[i_q] == static_cast<UInt>(1 << m))
          break;
      }
      i_q++;
    }
  }

  // Step 2: Apply sparse edits
  if (compressed.size_edit > 0) {
    if (compressed.size_edit_flag == 0)
      throw std::runtime_error("PGD edit stream is missing edit support mask");
    std::vector<UInt2> quant_edits = decompressQuantizedEdits(compressed);
    std::vector<uint8_t> packed_edit_mask = huffmanZstdDecompress(
        compressed.compressed_lossless_edit_flag,
        compressed.code_table_edit_flag, compressed.size_edit_flag,
        compressed.bit_stream_size_edit_flag);
    std::vector<bool> edit_mask = unpackBits(packed_edit_mask, N);
    int j = 0;
    for (int i = 0; i < N; ++i) {
      if (!edit_mask[i])
        continue;
      decomp_xx[i] += static_cast<T>(quant_edits[2 * j]) * norm - 2 * xi;
      decomp_yy[i] += static_cast<T>(quant_edits[2 * j + 1]) * norm - 2 * xi;
      ++j;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  printf("Decompression time: %f seconds\n", time.count());
}

template <typename T, OrderMode Mode>
void decompressWithEditParticles3D(const CompressedData<T> &compressed,
                                   T *decomp_xx, T *decomp_yy, T *decomp_zz,
                                   int N, T xi, T b) {
  if (compressed.size_flag == 0)
    return;
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<uint8_t> packed_lossless_flag = huffmanZstdDecompress(
      compressed.compressed_lossless_flag, compressed.code_table_flag,
      compressed.size_flag, compressed.bit_stream_size_flag);
  std::vector<bool> lossless_flag = unpackBits(packed_lossless_flag, 3 * N);
  std::vector<UInt> quant_codes = huffmanZstdDecompress(
      compressed.compressed_quant_codes, compressed.code_table_quant,
      compressed.size_quant, compressed.bit_stream_size_quant);
  T grid_len = b + 2 * std::sqrt(3) * xi;
  T min_x = compressed.grid_min_x;
  T min_y = compressed.grid_min_y;
  T min_z = compressed.grid_min_z;

  int i_f{0}, i_q{0}, i_l{0}, i_out{0};
  T norm = (4 * xi) / ((1 << m) - 1);

  // Step 1: Decompress positions
  for (int id_z = 0; id_z < compressed.grid_dim_z; ++id_z) {
    for (int id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
      for (int id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {
        T prev_x, prev_y, prev_z;
        if constexpr (Mode == OrderMode::KD_TREE) {
          prev_x = min_x + (id_x + T(0.5)) * grid_len;
          prev_y = min_y + (id_y + T(0.5)) * grid_len;
          prev_z = min_z + (id_z + T(0.5)) * grid_len;
        } else {
          prev_x = min_x + id_x * grid_len;
          prev_y = min_y + id_y * grid_len;
          prev_z = min_z + id_z * grid_len;
        }

        while (quant_codes[i_q] > 0) {
          if (lossless_flag[i_f++]) {
            prev_x = compressed.lossless_values[i_l++];
          } else {
            prev_x += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_xx[i_out] = prev_x;
          if (lossless_flag[i_f++]) {
            prev_y = compressed.lossless_values[i_l++];
          } else {
            prev_y += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_yy[i_out] = prev_y;
          if (lossless_flag[i_f++]) {
            prev_z = compressed.lossless_values[i_l++];
          } else {
            prev_z += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_zz[i_out++] = prev_z;
          if (quant_codes[i_q] == static_cast<UInt>(1 << m))
            break;
        }
        i_q++;
      }
    }
  }

  // Step 2: Apply sparse edits
  if (compressed.size_edit > 0) {
    if (compressed.size_edit_flag == 0)
      throw std::runtime_error("PGD edit stream is missing edit support mask");
    std::vector<UInt2> quant_edits = decompressQuantizedEdits(compressed);
    std::vector<uint8_t> packed_edit_mask = huffmanZstdDecompress(
        compressed.compressed_lossless_edit_flag,
        compressed.code_table_edit_flag, compressed.size_edit_flag,
        compressed.bit_stream_size_edit_flag);
    std::vector<bool> edit_mask = unpackBits(packed_edit_mask, N);
    int j = 0;
    for (int i = 0; i < N; ++i) {
      if (!edit_mask[i])
        continue;
      decomp_xx[i] += static_cast<T>(quant_edits[3 * j]) * norm - 2 * xi;
      decomp_yy[i] += static_cast<T>(quant_edits[3 * j + 1]) * norm - 2 * xi;
      decomp_zz[i] += static_cast<T>(quant_edits[3 * j + 2]) * norm - 2 * xi;
      ++j;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  printf("Decompression time: %f seconds\n", time.count());
}

template <typename T, OrderMode Mode>
void decompressParticles2D(const CompressedData<T> &compressed, T *decomp_xx,
                           T *decomp_yy, int N, T xi, T b) {
  auto decomp_start = std::chrono::high_resolution_clock::now();

  if (compressed.size_flag == 0)
    return;

  std::vector<uint8_t> packed_lossless_flag = huffmanZstdDecompress(
      compressed.compressed_lossless_flag, compressed.code_table_flag,
      compressed.size_flag, compressed.bit_stream_size_flag);
  std::vector<bool> lossless_flag = unpackBits(packed_lossless_flag, 2 * N);
  std::vector<UInt> quant_codes = huffmanZstdDecompress(
      compressed.compressed_quant_codes, compressed.code_table_quant,
      compressed.size_quant, compressed.bit_stream_size_quant);

  T grid_len = b + 2 * std::sqrt(2) * xi;
  T min_x = compressed.grid_min_x;
  T min_y = compressed.grid_min_y;

  int i_f{0}, i_q{0}, i_l{0}, i_out{0};

  for (int id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
    for (int id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {
      T prev_x, prev_y;
      if constexpr (Mode == OrderMode::KD_TREE) {
        prev_x = min_x + (id_x + T(0.5)) * grid_len;
        prev_y = min_y + (id_y + T(0.5)) * grid_len;
      } else {
        prev_x = min_x + id_x * grid_len;
        prev_y = min_y + id_y * grid_len;
      }

      while (quant_codes[i_q] > 0) {
        if (lossless_flag[i_f++]) {
          prev_x = compressed.lossless_values[i_l++];
        } else {
          prev_x += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
        }
        decomp_xx[i_out] = prev_x;
        if (lossless_flag[i_f++]) {
          prev_y = compressed.lossless_values[i_l++];
        } else {
          prev_y += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
        }
        decomp_yy[i_out++] = prev_y;
        if (quant_codes[i_q] == static_cast<UInt>(1 << m))
          break;
      }
      i_q++;
    }
  }

  auto decomp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = decomp_end - decomp_start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

template <typename T, OrderMode Mode>
void decompressParticles3D(const CompressedData<T> &compressed, T *decomp_xx,
                           T *decomp_yy, T *decomp_zz, int N, T xi, T b) {
  auto decomp_start = std::chrono::high_resolution_clock::now();
  if (compressed.size_flag == 0)
    return;

  std::vector<uint8_t> packed_lossless_flag = huffmanZstdDecompress(
      compressed.compressed_lossless_flag, compressed.code_table_flag,
      compressed.size_flag, compressed.bit_stream_size_flag);
  std::vector<bool> lossless_flag = unpackBits(packed_lossless_flag, 3 * N);
  std::vector<UInt> quant_codes = huffmanZstdDecompress(
      compressed.compressed_quant_codes, compressed.code_table_quant,
      compressed.size_quant, compressed.bit_stream_size_quant);

  T grid_len = b + 2 * std::sqrt(3) * xi;
  T min_x = compressed.grid_min_x;
  T min_y = compressed.grid_min_y;
  T min_z = compressed.grid_min_z;

  int i_f{0}, i_q{0}, i_l{0}, i_out{0};

  for (int id_z = 0; id_z < compressed.grid_dim_z; ++id_z) {
    for (int id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
      for (int id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {
        T prev_x, prev_y, prev_z;
        if constexpr (Mode == OrderMode::KD_TREE) {
          prev_x = min_x + (id_x + T(0.5)) * grid_len;
          prev_y = min_y + (id_y + T(0.5)) * grid_len;
          prev_z = min_z + (id_z + T(0.5)) * grid_len;
        } else {
          prev_x = min_x + id_x * grid_len;
          prev_y = min_y + id_y * grid_len;
          prev_z = min_z + id_z * grid_len;
        }

        while (quant_codes[i_q] > 0) {
          if (lossless_flag[i_f++]) {
            prev_x = compressed.lossless_values[i_l++];
          } else {
            prev_x += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_xx[i_out] = prev_x;
          if (lossless_flag[i_f++]) {
            prev_y = compressed.lossless_values[i_l++];
          } else {
            prev_y += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_yy[i_out] = prev_y;
          if (lossless_flag[i_f++]) {
            prev_z = compressed.lossless_values[i_l++];
          } else {
            prev_z += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_zz[i_out++] = prev_z;
          if (quant_codes[i_q] == static_cast<UInt>(1 << m))
            break;
        }
        i_q++;
      }
    }
  }

  auto decomp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = decomp_end - decomp_start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

template <typename T>
void reconstructEditParticles2D(const CompressedData<T> &compressed,
                                T *decomp_xx, T *decomp_yy, int N, T xi) {
  auto start = std::chrono::high_resolution_clock::now();
  if (compressed.size_edit == 0)
    return;
  if (compressed.size_edit_flag == 0)
    throw std::runtime_error("PGD edit stream is missing edit support mask");

  std::vector<UInt2> quant_edits = decompressQuantizedEdits(compressed);

  T norm = (4 * xi) / ((1 << m) - 1);
  std::vector<uint8_t> packed_edit_mask = huffmanZstdDecompress(
      compressed.compressed_lossless_edit_flag, compressed.code_table_edit_flag,
      compressed.size_edit_flag, compressed.bit_stream_size_edit_flag);
  std::vector<bool> edit_mask = unpackBits(packed_edit_mask, N);
  int j = 0;
  for (int i = 0; i < N; ++i) {
    if (!edit_mask[i])
      continue;
    T edit = static_cast<T>(quant_edits[2 * j]) * norm - 2 * xi;
    decomp_xx[i] += edit;
    edit = static_cast<T>(quant_edits[2 * j + 1]) * norm - 2 * xi;
    decomp_yy[i] += edit;
    ++j;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  printf("Decompression time: %f seconds\n", time.count());
}

template <typename T>
void reconstructEditParticles3D(const CompressedData<T> &compressed,
                                T *decomp_xx, T *decomp_yy, T *decomp_zz, int N,
                                T xi) {
  auto start = std::chrono::high_resolution_clock::now();
  if (compressed.size_edit == 0)
    return;
  if (compressed.size_edit_flag == 0)
    throw std::runtime_error("PGD edit stream is missing edit support mask");

  std::vector<UInt2> quant_edits = decompressQuantizedEdits(compressed);

  T norm = (4 * xi) / ((1 << m) - 1);
  std::vector<uint8_t> packed_edit_mask = huffmanZstdDecompress(
      compressed.compressed_lossless_edit_flag, compressed.code_table_edit_flag,
      compressed.size_edit_flag, compressed.bit_stream_size_edit_flag);
  std::vector<bool> edit_mask = unpackBits(packed_edit_mask, N);
  int j = 0;
  for (int i = 0; i < N; ++i) {
    if (!edit_mask[i])
      continue;
    T edit = static_cast<T>(quant_edits[3 * j]) * norm - 2 * xi;
    decomp_xx[i] += edit;
    edit = static_cast<T>(quant_edits[3 * j + 1]) * norm - 2 * xi;
    decomp_yy[i] += edit;
    edit = static_cast<T>(quant_edits[3 * j + 2]) * norm - 2 * xi;
    decomp_zz[i] += edit;
    ++j;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  printf("Decompression time: %f seconds\n", time.count());
}

// ============================================================================
// Compression / Correction functions
// ============================================================================

// Compression with edit
template <typename T, OrderMode Mode>
void compressWithEditParticles2D(const T *d_org_xx, const T *d_org_yy, T min_x,
                                 T range_x, T min_y, T range_y, int N, T xi,
                                 T b, CompressionState2D<T> &state,
                                 CompressedData<T> &compressed, int N_local) {
  if (N == 0)
    return;

  auto start = std::chrono::high_resolution_clock::now();

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.grid_len = b + 2 * std::sqrt(2) * xi;
  state.grid_dim_x = gridDimForRange(range_x, state.grid_len);
  state.grid_dim_y = gridDimForRange(range_y, state.grid_len);
  int max_cells = maxCellsForTargetOccupancy(N, 8);
  coarsenGrid2D(state.grid_len, state.grid_dim_x, state.grid_dim_y, range_x,
                range_y, max_cells);
  state.num_cells = state.grid_dim_x * state.grid_dim_y;

  compressed.grid_dim_x = state.grid_dim_x;
  compressed.grid_dim_y = state.grid_dim_y;
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.xi = xi;
  compressed.b = b;

  int num_values = 2 * N;

  // Particle partitioning (d_visit_order initially holds cell-sorted indices)
  particlePartition2D(d_org_xx, d_org_yy, min_x, min_y, state.grid_len,
                      state.grid_dim_x, state.grid_dim_y, N,
                      &state.d_cell_start, &state.d_visit_order);

  // Find vulnerable pairs
  T lower_bound = b - 2 * std::sqrt(2) * xi;
  T upper_bound = b + 2 * std::sqrt(2) * xi;
  T lower_bound_sq = (lower_bound < 0) ? 0 : lower_bound * lower_bound;
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  int *d_safe_roots = nullptr;
  int *d_halo_roots = nullptr;
  FoFConstraintStrategy strategy = fof_constraint_strategy;
  bool need_safe_roots = needsSafeComponentRoots(strategy);
  bool need_halo_roots = needsOriginalHaloRoots(strategy);
  int **safe_roots_out = need_safe_roots ? &d_safe_roots : nullptr;
  int **halo_roots_out = need_halo_roots ? &d_halo_roots : nullptr;
  findVulnerablePairs2D(
      d_org_xx, d_org_yy, state.d_cell_start, state.d_visit_order,
      &state.d_vulnerable_pairs, &state.d_signs, &state.num_vulnerable_pairs,
      state.grid_dim_x, state.grid_dim_y, N, lower_bound_sq, upper_bound_sq,
      sign_bound_sq, safe_roots_out, halo_roots_out, N_local);
  compressed.N_local = static_cast<size_t>(N_local);
  applyFoFConstraintStrategy2D(state, d_safe_roots, d_halo_roots);
  if (d_safe_roots)
    CUDA_CHECK(cudaFree(d_safe_roots));
  if (d_halo_roots)
    CUDA_CHECK(cudaFree(d_halo_roots));
  int editable_capacity =
      editableParticleCapacity(N, state.num_vulnerable_pairs, N_local);
  state.d_editable_pts_ht = createEmptyHashTable(N, editable_capacity);
  buildHashTableFromPairs(state.d_vulnerable_pairs, state.num_vulnerable_pairs,
                          state.d_editable_pts_ht, N_local);

  // Get editable particle count and free HT until PGD needs it
  CUDA_CHECK(cudaMemcpy(&state.num_editable_pts,
                        state.d_editable_pts_ht.counter, sizeof(int),
                        cudaMemcpyDeviceToHost));
  destroyHashTable(state.d_editable_pts_ht);

  // Compression (one thread per cell, d_visit_order reordered in-place)
  UInt *d_temp_qcode;
  T *d_temp_lval;
  int *d_cell_quant_count, *d_cell_lossless_count;
  CUDA_CHECK(cudaMalloc(&state.d_decomp_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_decomp_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_lossless_flag, num_values * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_temp_qcode, num_values * sizeof(UInt)));
  CUDA_CHECK(cudaMalloc(&d_temp_lval, num_values * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_cell_quant_count, state.num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_lossless_count, state.num_cells * sizeof(int)));

  int num_blocks = (state.num_cells + num_threads - 1) / num_threads;
  compressParticles2D_kernel<T, Mode><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, state.d_cell_start, state.num_cells, N, min_x, min_y,
      state.grid_len, xi, state.grid_dim_x, state.d_visit_order,
      state.d_lossless_flag, state.d_decomp_xx, state.d_decomp_yy, d_temp_qcode,
      d_temp_lval, d_cell_quant_count, d_cell_lossless_count);

  // Compact quant code and lossless values
  compactPass<T>(state.d_lossless_flag, d_temp_qcode, d_temp_lval,
                 state.d_cell_start, d_cell_quant_count, d_cell_lossless_count,
                 state.num_cells, N, 2, &state.d_quant_codes,
                 &state.d_lossless_values, state.num_quant_codes,
                 state.num_lossless_values);

  CUDA_CHECK(cudaFree(d_temp_qcode));
  CUDA_CHECK(cudaFree(d_temp_lval));
  CUDA_CHECK(cudaFree(d_cell_quant_count));
  CUDA_CHECK(cudaFree(d_cell_lossless_count));
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;

  // Pack lossless flags
  int num_flag_bytes = (num_values + 7) / 8;
  uint8_t *d_packed_flags;
  CUDA_CHECK(cudaMalloc(&d_packed_flags, num_flag_bytes));
  int flag_blocks = (num_flag_bytes + num_threads - 1) / num_threads;
  packLosslessFlags_kernel<<<flag_blocks, num_threads>>>(
      state.d_lossless_flag, d_packed_flags, num_values, num_flag_bytes);

  // Free lossless_flag now that it's packed (before Huffman to reduce peak mem)
  CUDA_CHECK(cudaFree(state.d_lossless_flag));
  state.d_lossless_flag = nullptr;

  // Copy lossless values to host before freeing
  if (state.num_lossless_values > 0) {
    compressed.lossless_values.resize(state.num_lossless_values);
    CUDA_CHECK(cudaMemcpy(
        compressed.lossless_values.data(), state.d_lossless_values,
        state.num_lossless_values * sizeof(T), cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(state.d_lossless_values));
  state.d_lossless_values = nullptr;

  // Huffman compress packed flags, then free before compressing quant codes
  compressed.size_flag = num_flag_bytes;
  compressed.size_quant = state.num_quant_codes;
  compressed.compressed_lossless_flag = huffmanZstdCompressDevice(
      d_packed_flags, num_flag_bytes, compressed.code_table_flag,
      compressed.bit_stream_size_flag);
  CUDA_CHECK(cudaFree(d_packed_flags));

  compressed.compressed_quant_codes = huffmanZstdCompressDevice(
      state.d_quant_codes, state.num_quant_codes, compressed.code_table_quant,
      compressed.bit_stream_size_quant);
  CUDA_CHECK(cudaFree(state.d_quant_codes));
  state.d_quant_codes = nullptr;

  printf("Number of editable particles: %d\n", state.num_editable_pts);

  // PGD with Adam optimizer
  if (state.num_editable_pts > 0) {
    // Recreate HT (was freed before compression to reduce peak memory)
    state.d_editable_pts_ht = createEmptyHashTable(N, state.num_editable_pts);
    buildHashTableFromPairs(state.d_vulnerable_pairs,
                            state.num_vulnerable_pairs, state.d_editable_pts_ht,
                            N_local);
    int E = state.num_editable_pts;
    T *d_grad_x, *d_grad_y;
    T *d_loss;

    CUDA_CHECK(cudaMalloc(&d_grad_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_y, 0, E * sizeof(T)));

    // Adam moment buffers
    T *d_m_x, *d_m_y, *d_v_x, *d_v_y;
    CUDA_CHECK(cudaMalloc(&d_m_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_m_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_y, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_y, 0, E * sizeof(T)));

    T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt(2);
    T convergence_tol = 1e-10;
    T decomp_tol = convergence_tol * convergence_tol;

    // Adam hyperparameters
    T adam_alpha = static_cast<T>(lr);
    T adam_beta1 = static_cast<T>(0.9);
    T adam_beta2 = static_cast<T>(0.999);
    T adam_eps = static_cast<T>(1e-8);

    T final_loss = 0;
    int iter = 0;
    T beta1_t = 1, beta2_t = 1;

    int lossBlocks =
        (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
    int sharedMem = num_threads * sizeof(T);
    int updateBlocks = (E + num_threads - 1) / num_threads;

    for (iter = 0; iter < max_iter; iter++) {
      CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
      computePGDLoss2D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, b, xi, state.num_vulnerable_pairs, d_loss,
          decomp_tol);
      CUDA_CHECK(
          cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
      if (final_loss < convergence_tol) {
        break;
      }

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));

      computePGDGradients2D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_editable_pts_ht, b, xi,
          state.num_vulnerable_pairs, decomp_tol, d_grad_x, d_grad_y);

      // Adam bias-corrected learning rate
      beta1_t *= adam_beta1;
      beta2_t *= adam_beta2;
      T lr_t = adam_alpha * sqrt(1 - beta2_t) / (1 - beta1_t);

      updatePGDPositionsAdam2D_kernel<<<updateBlocks, num_threads>>>(
          d_org_xx, d_org_yy, d_grad_x, d_grad_y, state.d_editable_pts_ht,
          state.d_decomp_xx, state.d_decomp_yy, state.d_edit_x, state.d_edit_y,
          d_m_x, d_m_y, d_v_x, d_v_y, adam_beta1, adam_beta2, adam_eps, lr_t,
          xi);
    }
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
    computePGDLoss2D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
        state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
        state.d_decomp_yy, b, xi, state.num_vulnerable_pairs, d_loss,
        decomp_tol);
    CUDA_CHECK(
        cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_m_x));
    CUDA_CHECK(cudaFree(d_m_y));
    CUDA_CHECK(cudaFree(d_v_x));
    CUDA_CHECK(cudaFree(d_v_y));
    CUDA_CHECK(cudaFree(d_loss));

    // Quantize edits (sparse: only particles with nonzero quantized edits)
    bool *d_edit_mask = nullptr;
    int *d_edit_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edit_mask, (size_t)N * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_edit_mask, 0, (size_t)N * sizeof(bool)));
    T norm = ((1 << m) - 1) / (4 * xi);
    T edit_tol = T(1) / norm;

    int quantBlocks = (N + num_threads - 1) / num_threads;
    buildEditMaskFromVisitOrder2D_kernel<<<quantBlocks, num_threads>>>(
        state.d_visit_order, state.d_editable_pts_ht, state.d_edit_x,
        state.d_edit_y, d_edit_mask, N, edit_tol);
    int num_nonzero_edits = countTrueFlags(d_edit_mask, N);
    int num_edit_values = 2 * num_nonzero_edits;
    compressed.size_edit = num_edit_values;
    printf("Number of nonzero edited particles: %d\n", num_nonzero_edits);

    if (num_nonzero_edits > 0) {
      CUDA_CHECK(
          cudaMalloc(&state.d_quant_edits, num_edit_values * sizeof(UInt2)));
      exclusiveScanBoolMask(d_edit_mask, N, &d_edit_offsets);
      quantizeMaskedEdits2D_kernel<<<quantBlocks, num_threads>>>(
          state.d_edit_x, state.d_edit_y, state.d_visit_order,
          state.d_editable_pts_ht, d_edit_mask, d_edit_offsets,
          state.d_quant_edits, xi, norm, N);
      CUDA_CHECK(cudaPeekAtLastError());
      compressEditMaskDevice(d_edit_mask, N, compressed);

      CUDA_CHECK(cudaFree(d_edit_offsets));

      compressQuantizedEditsDevice(state.d_quant_edits, num_edit_values,
                                   compressed);
    } else {
      clearCompressedEdits(compressed);
    }
    CUDA_CHECK(cudaFree(d_edit_mask));

    printf("Number of iterations: %d\n", iter);
    printf("PGD final loss: %e\n", final_loss);
  } else {
    clearCompressedEdits(compressed);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.compressed_quant_edits.size() +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      compressed.code_table_edit_flag.size() * 9 +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 2 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n", time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
}

template <typename T, OrderMode Mode>
void compressWithEditParticles3D(const T *d_org_xx, const T *d_org_yy,
                                 const T *d_org_zz, T min_x, T range_x, T min_y,
                                 T range_y, T min_z, T range_z, int N, T xi,
                                 T b, CompressionState3D<T> &state,
                                 CompressedData<T> &compressed, int N_local) {
  if (N == 0)
    return;

  auto start = std::chrono::high_resolution_clock::now();

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.min_z = min_z;
  state.grid_len = b + 2 * std::sqrt(3) * xi;
  state.grid_dim_x = gridDimForRange(range_x, state.grid_len);
  state.grid_dim_y = gridDimForRange(range_y, state.grid_len);
  state.grid_dim_z = gridDimForRange(range_z, state.grid_len);
  int max_cells = maxCellsForTargetOccupancy(N, 4);
  coarsenGrid3D(state.grid_len, state.grid_dim_x, state.grid_dim_y,
                state.grid_dim_z, range_x, range_y, range_z, max_cells);
  state.num_cells = state.grid_dim_x * state.grid_dim_y * state.grid_dim_z;
  printf("Grid: %d x %d x %d = %d cells, grid_len = %e\n", state.grid_dim_x,
         state.grid_dim_y, state.grid_dim_z, state.num_cells,
         (double)state.grid_len);

  compressed.grid_dim_x = state.grid_dim_x;
  compressed.grid_dim_y = state.grid_dim_y;
  compressed.grid_dim_z = state.grid_dim_z;
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.grid_min_z = min_z;
  compressed.xi = xi;
  compressed.b = b;

  int num_values = 3 * N;

  // Particle partitioning (d_visit_order initially holds cell-sorted indices)
  particlePartition3D(d_org_xx, d_org_yy, d_org_zz, min_x, min_y, min_z,
                      state.grid_len, state.grid_dim_x, state.grid_dim_y,
                      state.grid_dim_z, N, &state.d_cell_start,
                      &state.d_visit_order);

  // Find vulnerable pairs
  T lower_bound = b - 2 * std::sqrt(3) * xi;
  T upper_bound = b + 2 * std::sqrt(3) * xi;
  T lower_bound_sq = (lower_bound < 0) ? 0 : lower_bound * lower_bound;
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  int *d_safe_roots = nullptr;
  int *d_halo_roots = nullptr;
  FoFConstraintStrategy strategy = fof_constraint_strategy;
  bool need_safe_roots = needsSafeComponentRoots(strategy);
  bool need_halo_roots = needsOriginalHaloRoots(strategy);
  int **safe_roots_out = need_safe_roots ? &d_safe_roots : nullptr;
  int **halo_roots_out = need_halo_roots ? &d_halo_roots : nullptr;
  findVulnerablePairs3D(
      d_org_xx, d_org_yy, d_org_zz, state.d_cell_start, state.d_visit_order,
      &state.d_vulnerable_pairs, &state.d_signs, &state.num_vulnerable_pairs,
      state.grid_dim_x, state.grid_dim_y, state.grid_dim_z, N, lower_bound_sq,
      upper_bound_sq, sign_bound_sq, safe_roots_out, halo_roots_out, N_local);
  compressed.N_local = static_cast<size_t>(N_local);
  applyFoFConstraintStrategy3D(state, d_safe_roots, d_halo_roots);
  if (d_safe_roots)
    CUDA_CHECK(cudaFree(d_safe_roots));
  if (d_halo_roots)
    CUDA_CHECK(cudaFree(d_halo_roots));
  int editable_capacity =
      editableParticleCapacity(N, state.num_vulnerable_pairs, N_local);
  state.d_editable_pts_ht = createEmptyHashTable(N, editable_capacity);
  buildHashTableFromPairs(state.d_vulnerable_pairs, state.num_vulnerable_pairs,
                          state.d_editable_pts_ht, N_local);

  // Get editable particle count and free HT until PGD needs it
  CUDA_CHECK(cudaMemcpy(&state.num_editable_pts,
                        state.d_editable_pts_ht.counter, sizeof(int),
                        cudaMemcpyDeviceToHost));
  destroyHashTable(state.d_editable_pts_ht);

  // Compression (one thread per cell, d_visit_order reordered in-place)
  UInt *d_temp_qcode; // quantization codes in full size (compact later)
  T *d_temp_lval;     // lossless values in full size (compact later)
  int *d_cell_quant_count, *d_cell_lossless_count;
  CUDA_CHECK(cudaMalloc(&state.d_decomp_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_decomp_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_decomp_zz, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_lossless_flag, num_values * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_temp_qcode, num_values * sizeof(UInt)));
  CUDA_CHECK(cudaMalloc(&d_temp_lval, num_values * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_cell_quant_count, state.num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_lossless_count, state.num_cells * sizeof(int)));

  int num_blocks = (state.num_cells + num_threads - 1) / num_threads;
  compressParticles3D_kernel<T, Mode><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, state.d_cell_start, state.num_cells, N,
      min_x, min_y, min_z, state.grid_len, xi, state.grid_dim_x,
      state.grid_dim_y, state.d_visit_order, state.d_lossless_flag,
      state.d_decomp_xx, state.d_decomp_yy, state.d_decomp_zz, d_temp_qcode,
      d_temp_lval, d_cell_quant_count, d_cell_lossless_count);

  // Compact quant code and lossless values
  compactPass<T>(state.d_lossless_flag, d_temp_qcode, d_temp_lval,
                 state.d_cell_start, d_cell_quant_count, d_cell_lossless_count,
                 state.num_cells, N, 3, &state.d_quant_codes,
                 &state.d_lossless_values, state.num_quant_codes,
                 state.num_lossless_values);

  CUDA_CHECK(cudaFree(d_temp_qcode));
  CUDA_CHECK(cudaFree(d_temp_lval));
  CUDA_CHECK(cudaFree(d_cell_quant_count));
  CUDA_CHECK(cudaFree(d_cell_lossless_count));
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;

  // Pack lossless flags
  int num_flag_bytes = (num_values + 7) / 8;
  uint8_t *d_packed_flags;
  CUDA_CHECK(cudaMalloc(&d_packed_flags, num_flag_bytes));
  int flag_blocks = (num_flag_bytes + num_threads - 1) / num_threads;
  packLosslessFlags_kernel<<<flag_blocks, num_threads>>>(
      state.d_lossless_flag, d_packed_flags, num_values, num_flag_bytes);

  // Free lossless_flag now that it's packed (before Huffman to reduce peak mem)
  CUDA_CHECK(cudaFree(state.d_lossless_flag));
  state.d_lossless_flag = nullptr;

  // Copy lossless values to host before freeing
  if (state.num_lossless_values > 0) {
    compressed.lossless_values.resize(state.num_lossless_values);
    CUDA_CHECK(cudaMemcpy(
        compressed.lossless_values.data(), state.d_lossless_values,
        state.num_lossless_values * sizeof(T), cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(state.d_lossless_values));
  state.d_lossless_values = nullptr;

  // Huffman compress packed flags, then free before compressing quant codes
  compressed.size_flag = num_flag_bytes;
  compressed.size_quant = state.num_quant_codes;
  compressed.compressed_lossless_flag = huffmanZstdCompressDevice(
      d_packed_flags, num_flag_bytes, compressed.code_table_flag,
      compressed.bit_stream_size_flag);
  CUDA_CHECK(cudaFree(d_packed_flags));

  compressed.compressed_quant_codes = huffmanZstdCompressDevice(
      state.d_quant_codes, state.num_quant_codes, compressed.code_table_quant,
      compressed.bit_stream_size_quant);
  CUDA_CHECK(cudaFree(state.d_quant_codes));
  state.d_quant_codes = nullptr;

  printf("Number of editable particles: %d\n", state.num_editable_pts);

  // PGD with Adam optimizer
  if (state.num_editable_pts > 0) {
    // Recreate HT (was freed before compression to reduce peak memory)
    state.d_editable_pts_ht = createEmptyHashTable(N, state.num_editable_pts);
    buildHashTableFromPairs(state.d_vulnerable_pairs,
                            state.num_vulnerable_pairs, state.d_editable_pts_ht,
                            N_local);

    int E = state.num_editable_pts;
    T *d_grad_x, *d_grad_y, *d_grad_z;
    T *d_loss;

    CUDA_CHECK(cudaMalloc(&d_grad_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_grad_z, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_z, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_z, 0, E * sizeof(T)));

    // Adam moment buffers
    T *d_m_x, *d_m_y, *d_m_z, *d_v_x, *d_v_y, *d_v_z;
    CUDA_CHECK(cudaMalloc(&d_m_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_m_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_m_z, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_z, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_z, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_z, 0, E * sizeof(T)));

    T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt(3);
    T convergence_tol = 1e-10;
    T decomp_tol = convergence_tol * convergence_tol;

    // Adam hyperparameters
    T adam_alpha = static_cast<T>(lr);
    T adam_beta1 = static_cast<T>(0.9);
    T adam_beta2 = static_cast<T>(0.999);
    T adam_eps = static_cast<T>(1e-8);

    T final_loss = 0;
    int iter = 0;
    T beta1_t = 1, beta2_t = 1;
    int lossBlocks =
        (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
    int sharedMem = num_threads * sizeof(T);
    int updateBlocks = (E + num_threads - 1) / num_threads;

    for (iter = 0; iter < max_iter; iter++) {
      CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
      computePGDLoss3D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_decomp_zz, b, xi,
          state.num_vulnerable_pairs, d_loss, decomp_tol);
      CUDA_CHECK(
          cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
      if (final_loss < convergence_tol) {
        break;
      }

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_z, 0, E * sizeof(T)));

      computePGDGradients3D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_decomp_zz, state.d_editable_pts_ht, b, xi,
          state.num_vulnerable_pairs, decomp_tol, d_grad_x, d_grad_y, d_grad_z);

      // Adam bias-corrected learning rate
      beta1_t *= adam_beta1;
      beta2_t *= adam_beta2;
      T lr_t = adam_alpha * sqrt(1 - beta2_t) / (1 - beta1_t);

      updatePGDPositionsAdam3D_kernel<<<updateBlocks, num_threads>>>(
          d_org_xx, d_org_yy, d_org_zz, d_grad_x, d_grad_y, d_grad_z,
          state.d_editable_pts_ht, state.d_decomp_xx, state.d_decomp_yy,
          state.d_decomp_zz, state.d_edit_x, state.d_edit_y, state.d_edit_z,
          d_m_x, d_m_y, d_m_z, d_v_x, d_v_y, d_v_z, adam_beta1, adam_beta2,
          adam_eps, lr_t, xi);
    }
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
    computePGDLoss3D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
        state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
        state.d_decomp_yy, state.d_decomp_zz, b, xi, state.num_vulnerable_pairs,
        d_loss, decomp_tol);
    CUDA_CHECK(
        cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_grad_z));
    CUDA_CHECK(cudaFree(d_m_x));
    CUDA_CHECK(cudaFree(d_m_y));
    CUDA_CHECK(cudaFree(d_m_z));
    CUDA_CHECK(cudaFree(d_v_x));
    CUDA_CHECK(cudaFree(d_v_y));
    CUDA_CHECK(cudaFree(d_v_z));
    CUDA_CHECK(cudaFree(d_loss));

    // Quantize edits (sparse: only particles with nonzero quantized edits)
    bool *d_edit_mask = nullptr;
    int *d_edit_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edit_mask, (size_t)N * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_edit_mask, 0, (size_t)N * sizeof(bool)));
    T norm = ((1 << m) - 1) / (4 * xi);
    T edit_tol = T(1) / norm;

    int quantBlocks = (N + num_threads - 1) / num_threads;
    buildEditMaskFromVisitOrder3D_kernel<<<quantBlocks, num_threads>>>(
        state.d_visit_order, state.d_editable_pts_ht, state.d_edit_x,
        state.d_edit_y, state.d_edit_z, d_edit_mask, N, edit_tol);
    int num_nonzero_edits = countTrueFlags(d_edit_mask, N);
    int num_edit_values = 3 * num_nonzero_edits;
    compressed.size_edit = num_edit_values;
    printf("Number of nonzero edited particles: %d\n", num_nonzero_edits);

    if (num_nonzero_edits > 0) {
      CUDA_CHECK(
          cudaMalloc(&state.d_quant_edits, num_edit_values * sizeof(UInt2)));
      exclusiveScanBoolMask(d_edit_mask, N, &d_edit_offsets);
      quantizeMaskedEdits3D_kernel<<<quantBlocks, num_threads>>>(
          state.d_edit_x, state.d_edit_y, state.d_edit_z, state.d_visit_order,
          state.d_editable_pts_ht, d_edit_mask, d_edit_offsets,
          state.d_quant_edits, xi, norm, N);
      CUDA_CHECK(cudaPeekAtLastError());
      compressEditMaskDevice(d_edit_mask, N, compressed);

      CUDA_CHECK(cudaFree(d_edit_offsets));

      compressQuantizedEditsDevice(state.d_quant_edits, num_edit_values,
                                   compressed);
    } else {
      clearCompressedEdits(compressed);
    }
    CUDA_CHECK(cudaFree(d_edit_mask));

    printf("Number of iterations: %d\n", iter);
    printf("PGD final loss: %e\n", final_loss);
  } else {
    clearCompressedEdits(compressed);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.compressed_quant_edits.size() +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      compressed.code_table_edit_flag.size() * 9 +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  printf("sizes before lossless: %zu, after lossless: %zu\n",
         state.num_quant_codes * sizeof(UInt2),
         compressed.compressed_quant_edits.size() +
             compressed.code_table_edit.size() * (sizeof(UInt2) + 8));

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 3 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n", time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
}

// Compression only
template <typename T, OrderMode Mode>
void compressParticles2D(const T *d_org_xx, const T *d_org_yy, T min_x,
                         T range_x, T min_y, T range_y, int N, T xi, T b,
                         CompressionState2D<T> &state,
                         CompressedData<T> &compressed) {
  if (N == 0)
    return;

  auto start = std::chrono::high_resolution_clock::now();

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.grid_len = b + 2 * std::sqrt(2) * xi;
  state.grid_dim_x = gridDimForRange(range_x, state.grid_len);
  state.grid_dim_y = gridDimForRange(range_y, state.grid_len);
  int max_cells = maxCellsForTargetOccupancy(N, 8);
  coarsenGrid2D(state.grid_len, state.grid_dim_x, state.grid_dim_y, range_x,
                range_y, max_cells);
  state.num_cells = state.grid_dim_x * state.grid_dim_y;

  compressed.grid_dim_x = state.grid_dim_x;
  compressed.grid_dim_y = state.grid_dim_y;
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.xi = xi;
  compressed.b = b;

  int num_values = 2 * N;

  // Particle partitioning
  // Particle partitioning (d_visit_order initially holds cell-sorted indices)
  particlePartition2D(d_org_xx, d_org_yy, min_x, min_y, state.grid_len,
                      state.grid_dim_x, state.grid_dim_y, N,
                      &state.d_cell_start, &state.d_visit_order);

  // Compression (one thread per cell, d_visit_order reordered in-place)
  UInt *d_temp_qcode;
  T *d_temp_lval;
  int *d_cell_quant_count, *d_cell_lossless_count;
  CUDA_CHECK(cudaMalloc(&state.d_decomp_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_decomp_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_lossless_flag, num_values * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_temp_qcode, num_values * sizeof(UInt)));
  CUDA_CHECK(cudaMalloc(&d_temp_lval, num_values * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_cell_quant_count, state.num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_lossless_count, state.num_cells * sizeof(int)));

  int num_blocks = (state.num_cells + num_threads - 1) / num_threads;
  compressParticles2D_kernel<T, Mode><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, state.d_cell_start, state.num_cells, N, min_x, min_y,
      state.grid_len, xi, state.grid_dim_x, state.d_visit_order,
      state.d_lossless_flag, state.d_decomp_xx, state.d_decomp_yy, d_temp_qcode,
      d_temp_lval, d_cell_quant_count, d_cell_lossless_count);

  // Compact pass
  compactPass<T>(state.d_lossless_flag, d_temp_qcode, d_temp_lval,
                 state.d_cell_start, d_cell_quant_count, d_cell_lossless_count,
                 state.num_cells, N, 2, &state.d_quant_codes,
                 &state.d_lossless_values, state.num_quant_codes,
                 state.num_lossless_values);

  // Pack lossless flags
  int num_flag_bytes = (num_values + 7) / 8;
  uint8_t *d_packed_flags;
  CUDA_CHECK(cudaMalloc(&d_packed_flags, num_flag_bytes));
  int flag_blocks = (num_flag_bytes + num_threads - 1) / num_threads;
  packLosslessFlags_kernel<<<flag_blocks, num_threads>>>(
      state.d_lossless_flag, d_packed_flags, num_values, num_flag_bytes);

  // Free buffers no longer needed before Huffman to reduce peak memory
  CUDA_CHECK(cudaFree(state.d_lossless_flag));
  state.d_lossless_flag = nullptr;
  CUDA_CHECK(cudaFree(d_temp_qcode));
  CUDA_CHECK(cudaFree(d_temp_lval));
  CUDA_CHECK(cudaFree(d_cell_quant_count));
  CUDA_CHECK(cudaFree(d_cell_lossless_count));
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;

  // Copy lossless values to host before freeing
  if (state.num_lossless_values > 0) {
    compressed.lossless_values.resize(state.num_lossless_values);
    CUDA_CHECK(cudaMemcpy(
        compressed.lossless_values.data(), state.d_lossless_values,
        state.num_lossless_values * sizeof(T), cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(state.d_lossless_values));
  state.d_lossless_values = nullptr;

  // Huffman compress packed flags, then free before compressing quant codes
  compressed.size_flag = num_flag_bytes;
  compressed.size_quant = state.num_quant_codes;
  compressed.compressed_lossless_flag = huffmanZstdCompressDevice(
      d_packed_flags, num_flag_bytes, compressed.code_table_flag,
      compressed.bit_stream_size_flag);
  CUDA_CHECK(cudaFree(d_packed_flags));

  compressed.compressed_quant_codes = huffmanZstdCompressDevice(
      state.d_quant_codes, state.num_quant_codes, compressed.code_table_quant,
      compressed.bit_stream_size_quant);
  CUDA_CHECK(cudaFree(state.d_quant_codes));
  state.d_quant_codes = nullptr;

  // No edits in compress-only mode
  compressed.size_edit = 0;

  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 2 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n", time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
}

template <typename T, OrderMode Mode>
void compressParticles3D(const T *d_org_xx, const T *d_org_yy,
                         const T *d_org_zz, T min_x, T range_x, T min_y,
                         T range_y, T min_z, T range_z, int N, T xi, T b,
                         CompressionState3D<T> &state,
                         CompressedData<T> &compressed) {
  if (N == 0)
    return;

  auto start = std::chrono::high_resolution_clock::now();

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.min_z = min_z;
  state.grid_len = b + 2 * std::sqrt(3) * xi;
  state.grid_dim_x = gridDimForRange(range_x, state.grid_len);
  state.grid_dim_y = gridDimForRange(range_y, state.grid_len);
  state.grid_dim_z = gridDimForRange(range_z, state.grid_len);
  int max_cells = maxCellsForTargetOccupancy(N, 4);
  coarsenGrid3D(state.grid_len, state.grid_dim_x, state.grid_dim_y,
                state.grid_dim_z, range_x, range_y, range_z, max_cells);
  state.num_cells = state.grid_dim_x * state.grid_dim_y * state.grid_dim_z;

  compressed.grid_dim_x = state.grid_dim_x;
  compressed.grid_dim_y = state.grid_dim_y;
  compressed.grid_dim_z = state.grid_dim_z;
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.grid_min_z = min_z;
  compressed.xi = xi;
  compressed.b = b;

  int num_values = 3 * N;

  // Particle partitioning (d_visit_order initially holds cell-sorted indices)
  particlePartition3D(d_org_xx, d_org_yy, d_org_zz, min_x, min_y, min_z,
                      state.grid_len, state.grid_dim_x, state.grid_dim_y,
                      state.grid_dim_z, N, &state.d_cell_start,
                      &state.d_visit_order);

  // Compression (one thread per cell, d_visit_order reordered in-place)
  UInt *d_temp_qcode;
  T *d_temp_lval;
  int *d_cell_quant_count, *d_cell_lossless_count;
  CUDA_CHECK(cudaMalloc(&state.d_decomp_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_decomp_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_decomp_zz, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&state.d_lossless_flag, num_values * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_temp_qcode, num_values * sizeof(UInt)));
  CUDA_CHECK(cudaMalloc(&d_temp_lval, num_values * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_cell_quant_count, state.num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_lossless_count, state.num_cells * sizeof(int)));

  int num_blocks = (state.num_cells + num_threads - 1) / num_threads;
  compressParticles3D_kernel<T, Mode><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, state.d_cell_start, state.num_cells, N,
      min_x, min_y, min_z, state.grid_len, xi, state.grid_dim_x,
      state.grid_dim_y, state.d_visit_order, state.d_lossless_flag,
      state.d_decomp_xx, state.d_decomp_yy, state.d_decomp_zz, d_temp_qcode,
      d_temp_lval, d_cell_quant_count, d_cell_lossless_count);

  // Compact pass
  compactPass<T>(state.d_lossless_flag, d_temp_qcode, d_temp_lval,
                 state.d_cell_start, d_cell_quant_count, d_cell_lossless_count,
                 state.num_cells, N, 3, &state.d_quant_codes,
                 &state.d_lossless_values, state.num_quant_codes,
                 state.num_lossless_values);

  // Pack lossless flags
  int num_flag_bytes = (num_values + 7) / 8;
  uint8_t *d_packed_flags;
  CUDA_CHECK(cudaMalloc(&d_packed_flags, num_flag_bytes));
  int flag_blocks = (num_flag_bytes + num_threads - 1) / num_threads;
  packLosslessFlags_kernel<<<flag_blocks, num_threads>>>(
      state.d_lossless_flag, d_packed_flags, num_values, num_flag_bytes);

  // Free buffers no longer needed before Huffman to reduce peak memory
  CUDA_CHECK(cudaFree(state.d_lossless_flag));
  state.d_lossless_flag = nullptr;
  CUDA_CHECK(cudaFree(d_temp_qcode));
  CUDA_CHECK(cudaFree(d_temp_lval));
  CUDA_CHECK(cudaFree(d_cell_quant_count));
  CUDA_CHECK(cudaFree(d_cell_lossless_count));
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;

  // Copy lossless values to host before freeing
  if (state.num_lossless_values > 0) {
    compressed.lossless_values.resize(state.num_lossless_values);
    CUDA_CHECK(cudaMemcpy(
        compressed.lossless_values.data(), state.d_lossless_values,
        state.num_lossless_values * sizeof(T), cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(state.d_lossless_values));
  state.d_lossless_values = nullptr;

  // Huffman compress packed flags, then free before compressing quant codes
  compressed.size_flag = num_flag_bytes;
  compressed.size_quant = state.num_quant_codes;
  compressed.compressed_lossless_flag = huffmanZstdCompressDevice(
      d_packed_flags, num_flag_bytes, compressed.code_table_flag,
      compressed.bit_stream_size_flag);
  CUDA_CHECK(cudaFree(d_packed_flags));

  compressed.compressed_quant_codes = huffmanZstdCompressDevice(
      state.d_quant_codes, state.num_quant_codes, compressed.code_table_quant,
      compressed.bit_stream_size_quant);
  CUDA_CHECK(cudaFree(state.d_quant_codes));
  state.d_quant_codes = nullptr;

  // No edits in compress-only mode
  compressed.size_edit = 0;

  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 3 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n", time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
}

// Edit only
template <typename T, OrderMode Mode>
void editParticles2D(const T *d_org_xx, const T *d_org_yy, T *d_base_decomp_xx,
                     T *d_base_decomp_yy, T min_x, T range_x, T min_y,
                     T range_y, int N, T xi, T b, CompressionState2D<T> &state,
                     CompressedData<T> &compressed, int N_local,
                     MPI_Comm comm) {
  if (N == 0)
    return;

  auto start = std::chrono::high_resolution_clock::now();

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.grid_len = b + 2 * std::sqrt(2) * xi;
  state.grid_dim_x = gridDimForRange(range_x, state.grid_len);
  state.grid_dim_y = gridDimForRange(range_y, state.grid_len);
  int max_cells = maxCellsForTargetOccupancy(N, 8);
  coarsenGrid2D(state.grid_len, state.grid_dim_x, state.grid_dim_y, range_x,
                range_y, max_cells);
  state.num_cells = state.grid_dim_x * state.grid_dim_y;

  compressed.grid_dim_x = state.grid_dim_x;
  compressed.grid_dim_y = state.grid_dim_y;
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.xi = xi;
  compressed.b = b;

  // Particle partitioning (d_visit_order initially holds cell-sorted indices)
  particlePartition2D(d_org_xx, d_org_yy, min_x, min_y, state.grid_len,
                      state.grid_dim_x, state.grid_dim_y, N,
                      &state.d_cell_start, &state.d_visit_order);

  // Find vulnerable pairs
  T lower_bound = b - 2 * std::sqrt(2) * xi;
  T upper_bound = b + 2 * std::sqrt(2) * xi;
  T lower_bound_sq = (lower_bound < 0) ? 0 : lower_bound * lower_bound;
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  int *d_safe_roots = nullptr;
  int *d_halo_roots = nullptr;
  FoFConstraintStrategy strategy = fof_constraint_strategy;
  bool need_safe_roots = needsSafeComponentRoots(strategy);
  bool need_halo_roots = needsOriginalHaloRoots(strategy);
  int **safe_roots_out = need_safe_roots ? &d_safe_roots : nullptr;
  int **halo_roots_out = need_halo_roots ? &d_halo_roots : nullptr;
  findVulnerablePairs2D(
      d_org_xx, d_org_yy, state.d_cell_start, state.d_visit_order,
      &state.d_vulnerable_pairs, &state.d_signs, &state.num_vulnerable_pairs,
      state.grid_dim_x, state.grid_dim_y, N, lower_bound_sq, upper_bound_sq,
      sign_bound_sq, safe_roots_out, halo_roots_out, N_local);
  compressed.N_local = static_cast<size_t>(N_local);
  applyFoFConstraintStrategy2D(state, d_safe_roots, d_halo_roots);
  if (d_safe_roots)
    CUDA_CHECK(cudaFree(d_safe_roots));
  if (d_halo_roots)
    CUDA_CHECK(cudaFree(d_halo_roots));
  int editable_capacity =
      editableParticleCapacity(N, state.num_vulnerable_pairs, N_local);
  state.d_editable_pts_ht = createEmptyHashTable(N, editable_capacity);
  buildHashTableFromPairs(state.d_vulnerable_pairs, state.num_vulnerable_pairs,
                          state.d_editable_pts_ht, N_local);
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;
  CUDA_CHECK(cudaFree(state.d_visit_order));
  state.d_visit_order = nullptr;

  // Use provided base decompressed coordinates as state decomp arrays
  state.d_decomp_xx = d_base_decomp_xx;
  state.d_decomp_yy = d_base_decomp_yy;

  // Get editable particle count
  CUDA_CHECK(cudaMemcpy(&state.num_editable_pts,
                        state.d_editable_pts_ht.counter, sizeof(int),
                        cudaMemcpyDeviceToHost));

  printf("Number of editable particles: %d\n", state.num_editable_pts);

  // PGD with Adam optimizer
  T final_loss = 0;
  int iter = 0;
  int E = state.num_editable_pts;
  if (E > 0) {
    T *d_grad_x, *d_grad_y;
    T *d_loss;

    CUDA_CHECK(cudaMalloc(&d_grad_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_y, 0, E * sizeof(T)));

    // Adam moment buffers
    T *d_m_x, *d_m_y, *d_v_x, *d_v_y;
    CUDA_CHECK(cudaMalloc(&d_m_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_m_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_y, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_y, 0, E * sizeof(T)));

    T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt(2);
    T convergence_tol = 1e-10;
    T decomp_tol = convergence_tol * convergence_tol;

    T adam_alpha = static_cast<T>(lr);
    T adam_beta1 = static_cast<T>(0.9);
    T adam_beta2 = static_cast<T>(0.999);
    T adam_eps = static_cast<T>(1e-8);
    T beta1_t = 1, beta2_t = 1;
    int lossBlocks =
        (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
    int sharedMem = num_threads * sizeof(T);
    int updateBlocks = (E + num_threads - 1) / num_threads;

    for (iter = 0; iter < max_iter; iter++) {
      CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
      computePGDLoss2D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, b, xi, state.num_vulnerable_pairs, d_loss,
          decomp_tol);
      CUDA_CHECK(
          cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
      T convergence_loss = final_loss;
#ifdef USE_MPI
      if (comm != MPI_COMM_NULL) {
        MPI_Allreduce(&final_loss, &convergence_loss, 1,
                      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                      MPI_SUM, comm);
      }
#endif
      if (convergence_loss < convergence_tol) {
        final_loss = convergence_loss;
        break;
      }

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));

      computePGDGradients2D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_editable_pts_ht, b, xi,
          state.num_vulnerable_pairs, decomp_tol, d_grad_x, d_grad_y);

      beta1_t *= adam_beta1;
      beta2_t *= adam_beta2;
      T lr_t = adam_alpha * sqrt(1 - beta2_t) / (1 - beta1_t);

      updatePGDPositionsAdam2D_kernel<<<updateBlocks, num_threads>>>(
          d_org_xx, d_org_yy, d_grad_x, d_grad_y, state.d_editable_pts_ht,
          state.d_decomp_xx, state.d_decomp_yy, state.d_edit_x, state.d_edit_y,
          d_m_x, d_m_y, d_v_x, d_v_y, adam_beta1, adam_beta2, adam_eps, lr_t,
          xi);
    }
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
    computePGDLoss2D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
        state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
        state.d_decomp_yy, b, xi, state.num_vulnerable_pairs, d_loss,
        decomp_tol);
    CUDA_CHECK(
        cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
#ifdef USE_MPI
    if (comm != MPI_COMM_NULL) {
      T global_loss;
      MPI_Allreduce(&final_loss, &global_loss, 1,
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, comm);
      final_loss = global_loss;
    }
#endif

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_m_x));
    CUDA_CHECK(cudaFree(d_m_y));
    CUDA_CHECK(cudaFree(d_v_x));
    CUDA_CHECK(cudaFree(d_v_y));
    CUDA_CHECK(cudaFree(d_loss));

    // Quantize edits on GPU (sparse: only particles with nonzero edits)
    bool *d_edit_mask = nullptr;
    int *d_edit_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edit_mask, (size_t)N * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_edit_mask, 0, (size_t)N * sizeof(bool)));
    T norm = ((1 << m) - 1) / (4 * xi);
    T edit_tol = T(1) / norm;

    int editMaskBlocks = (E + num_threads - 1) / num_threads;
    buildEditMaskFromParticles2D_kernel<<<editMaskBlocks, num_threads>>>(
        state.d_editable_pts_ht, state.d_edit_x, state.d_edit_y, d_edit_mask, E,
        edit_tol);
    int num_nonzero_edits = countTrueFlags(d_edit_mask, N);
    int num_edit_values = 2 * num_nonzero_edits;
    compressed.size_edit = num_edit_values;
    printf("Number of nonzero edited particles: %d\n", num_nonzero_edits);

    if (num_nonzero_edits > 0) {
      CUDA_CHECK(
          cudaMalloc(&state.d_quant_edits, num_edit_values * sizeof(UInt2)));
      exclusiveScanBoolMask(d_edit_mask, N, &d_edit_offsets);
      int quantBlocks = (N + num_threads - 1) / num_threads;
      quantizeMaskedEdits2D_kernel<<<quantBlocks, num_threads>>>(
          state.d_edit_x, state.d_edit_y, nullptr, state.d_editable_pts_ht,
          d_edit_mask, d_edit_offsets, state.d_quant_edits, xi, norm, N);
      CUDA_CHECK(cudaPeekAtLastError());

      compressEditMaskDevice(d_edit_mask, N, compressed);

      CUDA_CHECK(cudaFree(d_edit_offsets));

      compressQuantizedEditsDevice(state.d_quant_edits, num_edit_values,
                                   compressed);
    } else {
      clearCompressedEdits(compressed);
    }
    CUDA_CHECK(cudaFree(d_edit_mask));

    printf("Number of iterations: %d\n", iter);
    printf("PGD final loss: %e\n", final_loss);
  } else {
    clearCompressedEdits(compressed);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  printf("Compression time: %f seconds\n", time.count());

  size_t additional_size =
      compressed.compressed_quant_edits.size() +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  printf("Additional storage: %zu bytes\n", additional_size);
}

template <typename T, OrderMode Mode>
void editParticles3D(const T *d_org_xx, const T *d_org_yy, const T *d_org_zz,
                     T *d_base_decomp_xx, T *d_base_decomp_yy,
                     T *d_base_decomp_zz, T min_x, T range_x, T min_y,
                     T range_y, T min_z, T range_z, int N, T xi, T b,
                     CompressionState3D<T> &state,
                     CompressedData<T> &compressed, int N_local,
                     MPI_Comm comm) {
  if (N == 0)
    return;

#ifdef USE_MPI
  double t_fof_start = MPI_Wtime();
#else
  double t_fof_start =
      std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
#endif
  auto start = std::chrono::high_resolution_clock::now();

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.min_z = min_z;
  state.grid_len = b + 2 * std::sqrt(3) * xi;
  state.grid_dim_x = gridDimForRange(range_x, state.grid_len);
  state.grid_dim_y = gridDimForRange(range_y, state.grid_len);
  state.grid_dim_z = gridDimForRange(range_z, state.grid_len);
  int max_cells = maxCellsForTargetOccupancy(N, 4);
  coarsenGrid3D(state.grid_len, state.grid_dim_x, state.grid_dim_y,
                state.grid_dim_z, range_x, range_y, range_z, max_cells);
  state.num_cells = state.grid_dim_x * state.grid_dim_y * state.grid_dim_z;

  compressed.grid_dim_x = state.grid_dim_x;
  compressed.grid_dim_y = state.grid_dim_y;
  compressed.grid_dim_z = state.grid_dim_z;
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.grid_min_z = min_z;
  compressed.xi = xi;
  compressed.b = b;

  // Particle partitioning (d_visit_order initially holds cell-sorted indices)
  particlePartition3D(d_org_xx, d_org_yy, d_org_zz, min_x, min_y, min_z,
                      state.grid_len, state.grid_dim_x, state.grid_dim_y,
                      state.grid_dim_z, N, &state.d_cell_start,
                      &state.d_visit_order);

  T lower_bound = b - 2 * std::sqrt(3) * xi;
  T upper_bound = b + 2 * std::sqrt(3) * xi;
  T lower_bound_sq = (lower_bound < 0) ? 0 : lower_bound * lower_bound;
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  int *d_safe_roots = nullptr;
  int *d_halo_roots = nullptr;
  FoFConstraintStrategy strategy = fof_constraint_strategy;
  bool need_safe_roots = needsSafeComponentRoots(strategy);
  bool need_halo_roots = needsOriginalHaloRoots(strategy);
  int **safe_roots_out = need_safe_roots ? &d_safe_roots : nullptr;
  int **halo_roots_out = need_halo_roots ? &d_halo_roots : nullptr;
  findVulnerablePairs3D(
      d_org_xx, d_org_yy, d_org_zz, state.d_cell_start, state.d_visit_order,
      &state.d_vulnerable_pairs, &state.d_signs, &state.num_vulnerable_pairs,
      state.grid_dim_x, state.grid_dim_y, state.grid_dim_z, N, lower_bound_sq,
      upper_bound_sq, sign_bound_sq, safe_roots_out, halo_roots_out, N_local);
  compressed.N_local = static_cast<size_t>(N_local);
  applyFoFConstraintStrategy3D(state, d_safe_roots, d_halo_roots);
  if (d_safe_roots)
    CUDA_CHECK(cudaFree(d_safe_roots));
  if (d_halo_roots)
    CUDA_CHECK(cudaFree(d_halo_roots));
  int editable_capacity =
      editableParticleCapacity(N, state.num_vulnerable_pairs, N_local);
  state.d_editable_pts_ht = createEmptyHashTable(N, editable_capacity);
  buildHashTableFromPairs(state.d_vulnerable_pairs, state.num_vulnerable_pairs,
                          state.d_editable_pts_ht, N_local);
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;
  CUDA_CHECK(cudaFree(state.d_visit_order));
  state.d_visit_order = nullptr;

  // Use provided base decompressed coordinates as state decomp arrays
  state.d_decomp_xx = d_base_decomp_xx;
  state.d_decomp_yy = d_base_decomp_yy;
  state.d_decomp_zz = d_base_decomp_zz;

  // Get editable particle count
  CUDA_CHECK(cudaMemcpy(&state.num_editable_pts,
                        state.d_editable_pts_ht.counter, sizeof(int),
                        cudaMemcpyDeviceToHost));

  printf("Number of editable particles: %d\n", state.num_editable_pts);

#ifdef USE_MPI
  printf("[Timer] FOF setup: %f seconds\n", MPI_Wtime() - t_fof_start);
#else
  printf("[Timer] FOF setup: %f seconds\n",
         std::chrono::duration<double>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
                 .count() -
             t_fof_start);
#endif
  fflush(stdout);

  // PGD with Adam optimizer
#ifdef USE_MPI
  double t_pgd_start = MPI_Wtime();
#else
  double t_pgd_start =
      std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
#endif
  double t_allreduce_total = 0.0;
  int allreduce_loss_calls = 0;
  if (state.num_editable_pts > 0) {
    int E = state.num_editable_pts;
    T *d_grad_x, *d_grad_y, *d_grad_z;
    T *d_loss;

    CUDA_CHECK(cudaMalloc(&d_grad_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_grad_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_grad_z, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&state.d_edit_z, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(state.d_edit_z, 0, E * sizeof(T)));

    // Adam moment buffers
    T *d_m_x, *d_m_y, *d_m_z, *d_v_x, *d_v_y, *d_v_z;
    CUDA_CHECK(cudaMalloc(&d_m_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_m_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_m_z, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_x, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_y, E * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_v_z, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_m_z, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_x, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_y, 0, E * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_v_z, 0, E * sizeof(T)));

    T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt(3);
    T convergence_tol = 1e-10;
    T decomp_tol = convergence_tol * convergence_tol;

    T adam_alpha = static_cast<T>(lr);
    T adam_beta1 = static_cast<T>(0.9);
    T adam_beta2 = static_cast<T>(0.999);
    T adam_eps = static_cast<T>(1e-8);

    T final_loss = 0;
    int iter = 0;
    T beta1_t = 1, beta2_t = 1;
    int lossBlocks =
        (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
    int sharedMem = num_threads * sizeof(T);
    int updateBlocks = (E + num_threads - 1) / num_threads;

    for (iter = 0; iter < max_iter; iter++) {
      // Check convergence periodically to reduce D2H sync overhead
      CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
      computePGDLoss3D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_decomp_zz, b, xi,
          state.num_vulnerable_pairs, d_loss, decomp_tol);
      CUDA_CHECK(
          cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
      // printf("iter %d, loss: %f\n", iter, final_loss);
      T convergence_loss = final_loss;
#ifdef USE_MPI
      if (comm != MPI_COMM_NULL) {
        double t_ar0 = MPI_Wtime();
        MPI_Allreduce(&final_loss, &convergence_loss, 1,
                      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                      MPI_SUM, comm);
        ++allreduce_loss_calls;
        t_allreduce_total += MPI_Wtime() - t_ar0;
      }
#endif
      if (convergence_loss < convergence_tol) {
        final_loss = convergence_loss;
        break;
      }

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_z, 0, E * sizeof(T)));

      computePGDGradients3D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_decomp_zz, state.d_editable_pts_ht, b, xi,
          state.num_vulnerable_pairs, decomp_tol, d_grad_x, d_grad_y, d_grad_z);

      beta1_t *= adam_beta1;
      beta2_t *= adam_beta2;
      T lr_t = adam_alpha * sqrt(1 - beta2_t) / (1 - beta1_t);

      updatePGDPositionsAdam3D_kernel<<<updateBlocks, num_threads>>>(
          d_org_xx, d_org_yy, d_org_zz, d_grad_x, d_grad_y, d_grad_z,
          state.d_editable_pts_ht, state.d_decomp_xx, state.d_decomp_yy,
          state.d_decomp_zz, state.d_edit_x, state.d_edit_y, state.d_edit_z,
          d_m_x, d_m_y, d_m_z, d_v_x, d_v_y, d_v_z, adam_beta1, adam_beta2,
          adam_eps, lr_t, xi);
    }
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
    computePGDLoss3D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
        state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
        state.d_decomp_yy, state.d_decomp_zz, b, xi, state.num_vulnerable_pairs,
        d_loss, decomp_tol);
    CUDA_CHECK(
        cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
#ifdef USE_MPI
    if (comm != MPI_COMM_NULL) {
      T global_loss;
      double t_ar0 = MPI_Wtime();
      MPI_Allreduce(&final_loss, &global_loss, 1,
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, comm);
      ++allreduce_loss_calls;
      t_allreduce_total += MPI_Wtime() - t_ar0;
      final_loss = global_loss;
    }
#endif
#ifdef USE_MPI
    double t_pgd_elapsed = MPI_Wtime() - t_pgd_start;
#else
    double t_pgd_elapsed =
        std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count() -
        t_pgd_start;
#endif
    printf("[Timer] PGD iterations (%d iters): %f seconds\n", iter,
           t_pgd_elapsed);
    printf("[Timer] MPI_Allreduce (loss, %d calls): %f seconds\n",
           allreduce_loss_calls, t_allreduce_total);
    fflush(stdout);

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_grad_z));
    CUDA_CHECK(cudaFree(d_m_x));
    CUDA_CHECK(cudaFree(d_m_y));
    CUDA_CHECK(cudaFree(d_m_z));
    CUDA_CHECK(cudaFree(d_v_x));
    CUDA_CHECK(cudaFree(d_v_y));
    CUDA_CHECK(cudaFree(d_v_z));
    CUDA_CHECK(cudaFree(d_loss));

    // Quantize edits on GPU (sparse: only particles with nonzero edits)
    bool *d_edit_mask = nullptr;
    int *d_edit_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_edit_mask, (size_t)N * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_edit_mask, 0, (size_t)N * sizeof(bool)));
    T norm = ((1 << m) - 1) / (4 * xi);
    T edit_tol = T(1) / norm;

    int editMaskBlocks = (E + num_threads - 1) / num_threads;
    buildEditMaskFromParticles3D_kernel<<<editMaskBlocks, num_threads>>>(
        state.d_editable_pts_ht, state.d_edit_x, state.d_edit_y, state.d_edit_z,
        d_edit_mask, E, edit_tol);
    int num_nonzero_edits = countTrueFlags(d_edit_mask, N);
    int num_edit_values = 3 * num_nonzero_edits;
    compressed.size_edit = num_edit_values;
    printf("Number of nonzero edited particles: %d\n", num_nonzero_edits);

    if (num_nonzero_edits > 0) {
      CUDA_CHECK(
          cudaMalloc(&state.d_quant_edits, num_edit_values * sizeof(UInt2)));
      exclusiveScanBoolMask(d_edit_mask, N, &d_edit_offsets);
      int quantBlocks = (N + num_threads - 1) / num_threads;
      quantizeMaskedEdits3D_kernel<<<quantBlocks, num_threads>>>(
          state.d_edit_x, state.d_edit_y, state.d_edit_z, nullptr,
          state.d_editable_pts_ht, d_edit_mask, d_edit_offsets,
          state.d_quant_edits, xi, norm, N);
      CUDA_CHECK(cudaPeekAtLastError());
      compressEditMaskDevice(d_edit_mask, N, compressed);

      CUDA_CHECK(cudaFree(d_edit_offsets));

      compressQuantizedEditsDevice(state.d_quant_edits, num_edit_values,
                                   compressed);
    } else {
      clearCompressedEdits(compressed);
    }
    CUDA_CHECK(cudaFree(d_edit_mask));

    printf("Number of iterations: %d\n", iter);
    printf("PGD final loss: %e\n", final_loss);
  } else {
    clearCompressedEdits(compressed);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end - start;
  printf("Compression time: %f seconds\n", time.count());

  size_t additional_size =
      compressed.compressed_quant_edits.size() +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  printf("Additional storage: %zu bytes\n", additional_size);
}

// Getter functions
template <typename T>
void getDecompressedCoords2D(const CompressionState2D<T> &state, T *h_decomp_xx,
                             T *h_decomp_yy) {
  CUDA_CHECK(cudaMemcpy(h_decomp_xx, state.d_decomp_xx, state.N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_decomp_yy, state.d_decomp_yy, state.N * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

template <typename T>
void getDecompressedCoords3D(const CompressionState3D<T> &state, T *h_decomp_xx,
                             T *h_decomp_yy, T *h_decomp_zz) {
  CUDA_CHECK(cudaMemcpy(h_decomp_xx, state.d_decomp_xx, state.N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_decomp_yy, state.d_decomp_yy, state.N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_decomp_zz, state.d_decomp_zz, state.N * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

template <typename T>
void getVisitOrder(const CompressionState2D<T> &state, int *h_visit_order) {
  CUDA_CHECK(cudaMemcpy(h_visit_order, state.d_visit_order,
                        state.N * sizeof(int), cudaMemcpyDeviceToHost));
}

template <typename T>
void getVisitOrder(const CompressionState3D<T> &state, int *h_visit_order) {
  CUDA_CHECK(cudaMemcpy(h_visit_order, state.d_visit_order,
                        state.N * sizeof(int), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Explicit template instantiations
// ============================================================================
// Decompression functions
template void decompressWithEditParticles2D<float, OrderMode::KD_TREE>(
    const CompressedData<float> &, float *, float *, int, float, float);
template void decompressWithEditParticles2D<float, OrderMode::MORTON_CODE>(
    const CompressedData<float> &, float *, float *, int, float, float);
template void decompressWithEditParticles2D<double, OrderMode::KD_TREE>(
    const CompressedData<double> &, double *, double *, int, double, double);
template void decompressWithEditParticles2D<double, OrderMode::MORTON_CODE>(
    const CompressedData<double> &, double *, double *, int, double, double);

template void decompressWithEditParticles3D<float, OrderMode::KD_TREE>(
    const CompressedData<float> &, float *, float *, float *, int, float,
    float);
template void decompressWithEditParticles3D<float, OrderMode::MORTON_CODE>(
    const CompressedData<float> &, float *, float *, float *, int, float,
    float);
template void decompressWithEditParticles3D<double, OrderMode::KD_TREE>(
    const CompressedData<double> &, double *, double *, double *, int, double,
    double);
template void decompressWithEditParticles3D<double, OrderMode::MORTON_CODE>(
    const CompressedData<double> &, double *, double *, double *, int, double,
    double);

template void decompressParticles2D<float, OrderMode::KD_TREE>(
    const CompressedData<float> &, float *, float *, int, float, float);
template void decompressParticles2D<float, OrderMode::MORTON_CODE>(
    const CompressedData<float> &, float *, float *, int, float, float);
template void decompressParticles2D<double, OrderMode::KD_TREE>(
    const CompressedData<double> &, double *, double *, int, double, double);
template void decompressParticles2D<double, OrderMode::MORTON_CODE>(
    const CompressedData<double> &, double *, double *, int, double, double);

template void
decompressParticles3D<float, OrderMode::KD_TREE>(const CompressedData<float> &,
                                                 float *, float *, float *, int,
                                                 float, float);
template void decompressParticles3D<float, OrderMode::MORTON_CODE>(
    const CompressedData<float> &, float *, float *, float *, int, float,
    float);
template void decompressParticles3D<double, OrderMode::KD_TREE>(
    const CompressedData<double> &, double *, double *, double *, int, double,
    double);
template void decompressParticles3D<double, OrderMode::MORTON_CODE>(
    const CompressedData<double> &, double *, double *, double *, int, double,
    double);

template void reconstructEditParticles2D<float>(const CompressedData<float> &,
                                                float *, float *, int, float);
template void reconstructEditParticles2D<double>(const CompressedData<double> &,
                                                 double *, double *, int,
                                                 double);

template void reconstructEditParticles3D<float>(const CompressedData<float> &,
                                                float *, float *, float *, int,
                                                float);
template void reconstructEditParticles3D<double>(const CompressedData<double> &,
                                                 double *, double *, double *,
                                                 int, double);

// Compression with edit API instantiations
template void compressWithEditParticles2D<float, OrderMode::KD_TREE>(
    const float *, const float *, float, float, float, float, int, float, float,
    CompressionState2D<float> &, CompressedData<float> &, int);
template void compressWithEditParticles2D<float, OrderMode::MORTON_CODE>(
    const float *, const float *, float, float, float, float, int, float, float,
    CompressionState2D<float> &, CompressedData<float> &, int);
template void compressWithEditParticles2D<double, OrderMode::KD_TREE>(
    const double *, const double *, double, double, double, double, int, double,
    double, CompressionState2D<double> &, CompressedData<double> &, int);
template void compressWithEditParticles2D<double, OrderMode::MORTON_CODE>(
    const double *, const double *, double, double, double, double, int, double,
    double, CompressionState2D<double> &, CompressedData<double> &, int);

template void compressWithEditParticles3D<float, OrderMode::KD_TREE>(
    const float *, const float *, const float *, float, float, float, float,
    float, float, int, float, float, CompressionState3D<float> &,
    CompressedData<float> &, int);
template void compressWithEditParticles3D<float, OrderMode::MORTON_CODE>(
    const float *, const float *, const float *, float, float, float, float,
    float, float, int, float, float, CompressionState3D<float> &,
    CompressedData<float> &, int);
template void compressWithEditParticles3D<double, OrderMode::KD_TREE>(
    const double *, const double *, const double *, double, double, double,
    double, double, double, int, double, double, CompressionState3D<double> &,
    CompressedData<double> &, int);
template void compressWithEditParticles3D<double, OrderMode::MORTON_CODE>(
    const double *, const double *, const double *, double, double, double,
    double, double, double, int, double, double, CompressionState3D<double> &,
    CompressedData<double> &, int);

template void compressParticles2D<float, OrderMode::KD_TREE>(
    const float *, const float *, float, float, float, float, int, float, float,
    CompressionState2D<float> &, CompressedData<float> &);
template void compressParticles2D<float, OrderMode::MORTON_CODE>(
    const float *, const float *, float, float, float, float, int, float, float,
    CompressionState2D<float> &, CompressedData<float> &);
template void compressParticles2D<double, OrderMode::KD_TREE>(
    const double *, const double *, double, double, double, double, int, double,
    double, CompressionState2D<double> &, CompressedData<double> &);
template void compressParticles2D<double, OrderMode::MORTON_CODE>(
    const double *, const double *, double, double, double, double, int, double,
    double, CompressionState2D<double> &, CompressedData<double> &);

template void compressParticles3D<float, OrderMode::KD_TREE>(
    const float *, const float *, const float *, float, float, float, float,
    float, float, int, float, float, CompressionState3D<float> &,
    CompressedData<float> &);
template void compressParticles3D<float, OrderMode::MORTON_CODE>(
    const float *, const float *, const float *, float, float, float, float,
    float, float, int, float, float, CompressionState3D<float> &,
    CompressedData<float> &);
template void compressParticles3D<double, OrderMode::KD_TREE>(
    const double *, const double *, const double *, double, double, double,
    double, double, double, int, double, double, CompressionState3D<double> &,
    CompressedData<double> &);
template void compressParticles3D<double, OrderMode::MORTON_CODE>(
    const double *, const double *, const double *, double, double, double,
    double, double, double, int, double, double, CompressionState3D<double> &,
    CompressedData<double> &);

template void editParticles2D<float, OrderMode::KD_TREE>(
    const float *, const float *, float *, float *, float, float, float, float,
    int, float, float, CompressionState2D<float> &, CompressedData<float> &,
    int, MPI_Comm);
template void editParticles2D<float, OrderMode::MORTON_CODE>(
    const float *, const float *, float *, float *, float, float, float, float,
    int, float, float, CompressionState2D<float> &, CompressedData<float> &,
    int, MPI_Comm);
template void editParticles2D<double, OrderMode::KD_TREE>(
    const double *, const double *, double *, double *, double, double, double,
    double, int, double, double, CompressionState2D<double> &,
    CompressedData<double> &, int, MPI_Comm);
template void editParticles2D<double, OrderMode::MORTON_CODE>(
    const double *, const double *, double *, double *, double, double, double,
    double, int, double, double, CompressionState2D<double> &,
    CompressedData<double> &, int, MPI_Comm);

template void editParticles3D<float, OrderMode::KD_TREE>(
    const float *, const float *, const float *, float *, float *, float *,
    float, float, float, float, float, float, int, float, float,
    CompressionState3D<float> &, CompressedData<float> &, int, MPI_Comm);
template void editParticles3D<float, OrderMode::MORTON_CODE>(
    const float *, const float *, const float *, float *, float *, float *,
    float, float, float, float, float, float, int, float, float,
    CompressionState3D<float> &, CompressedData<float> &, int, MPI_Comm);
template void editParticles3D<double, OrderMode::KD_TREE>(
    const double *, const double *, const double *, double *, double *,
    double *, double, double, double, double, double, double, int, double,
    double, CompressionState3D<double> &, CompressedData<double> &, int,
    MPI_Comm);
template void editParticles3D<double, OrderMode::MORTON_CODE>(
    const double *, const double *, const double *, double *, double *,
    double *, double, double, double, double, double, double, int, double,
    double, CompressionState3D<double> &, CompressedData<double> &, int,
    MPI_Comm);

template void getDecompressedCoords2D<float>(const CompressionState2D<float> &,
                                             float *, float *);
template void
getDecompressedCoords2D<double>(const CompressionState2D<double> &, double *,
                                double *);

template void getDecompressedCoords3D<float>(const CompressionState3D<float> &,
                                             float *, float *, float *);
template void
getDecompressedCoords3D<double>(const CompressionState3D<double> &, double *,
                                double *, double *);

template void getVisitOrder<float>(const CompressionState2D<float> &, int *);
template void getVisitOrder<double>(const CompressionState2D<double> &, int *);
template void getVisitOrder<float>(const CompressionState3D<float> &, int *);
template void getVisitOrder<double>(const CompressionState3D<double> &, int *);
