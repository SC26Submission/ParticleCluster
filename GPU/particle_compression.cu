#include "FOFHaloFinder.cuh"
#include "particle_compression.cuh"
#include <chrono>

// Globals defined in main.cu
extern double lr;
extern int max_iter;

// Coarsen grid if num_cells exceeds max_cells budget
template <typename T>
static void coarsenGrid2D(T &grid_len, int &grid_dim_x, int &grid_dim_y,
                          T range_x, T range_y, int max_cells) {
  long long nc = (long long)grid_dim_x * grid_dim_y;
  if (nc <= max_cells)
    return;
  T scale = std::sqrt(static_cast<T>(nc) / max_cells);
  grid_len *= scale;
  grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));
  long long nc_new = (long long)grid_dim_x * grid_dim_y;
  while (nc_new > max_cells) {
    grid_len *= static_cast<T>(1.01);
    grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
    grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));
    nc_new = (long long)grid_dim_x * grid_dim_y;
  }
  printf("Grid coarsened: %lld -> %lld cells (grid_len %e, factor %.2fx)\n", nc,
         nc_new, (double)grid_len, (double)scale);
}

template <typename T>
static void coarsenGrid3D(T &grid_len, int &grid_dim_x, int &grid_dim_y,
                          int &grid_dim_z, T range_x, T range_y, T range_z,
                          int max_cells) {
  long long nc = (long long)grid_dim_x * grid_dim_y * grid_dim_z;
  if (nc <= max_cells)
    return;
  T scale = std::cbrt(static_cast<T>(nc) / max_cells);
  grid_len *= scale;
  grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));
  grid_dim_z = std::max(1, static_cast<int>(std::ceil(range_z / grid_len)));
  long long nc_new = (long long)grid_dim_x * grid_dim_y * grid_dim_z;
  // Rounding may overshoot; iterate if needed
  while (nc_new > max_cells) {
    grid_len *= static_cast<T>(1.01);
    grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
    grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));
    grid_dim_z = std::max(1, static_cast<int>(std::ceil(range_z / grid_len)));
    nc_new = (long long)grid_dim_x * grid_dim_y * grid_dim_z;
  }
  printf("Grid coarsened: %lld -> %lld cells (grid_len %e, factor %.2fx)\n", nc,
         nc_new, (double)grid_len, (double)scale);
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
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_quant_offsets));
  CUDA_CHECK(cudaFree(d_lossless_offsets));
}

// Helper: MCC and ARI calculation
template <typename T> T calculateMCC(int tp_l, int tn_l, int fp_l, int fn_l) {
  T tp = static_cast<T>(tp_l);
  T tn = static_cast<T>(tn_l);
  T fp = static_cast<T>(fp_l);
  T fn = static_cast<T>(fn_l);

  T mcc = (tp * tn - fp * fn) / (std::sqrt(tp + fp) * std::sqrt(tp + fn) *
                                 std::sqrt(tn + fp) * std::sqrt(tn + fn));
  return mcc;
}

template <typename T>
T calculateARI(long long tp_h, long long tn_h, long long fp_h, long long fn_h) {
  T tp = static_cast<T>(tp_h);
  T tn = static_cast<T>(tn_h);
  T fp = static_cast<T>(fp_h);
  T fn = static_cast<T>(fn_h);

  T ari =
      2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
  return ari;
}

template <typename T>
void findVulnerablePairs2D(const T *d_org_xx, const T *d_org_yy,
                           const int *d_cell_start,
                           const int *d_cell_pts_sorted,
                           HashTable &d_editable_pts_ht,
                           int **d_vulnerable_pairs_out, bool **d_signs_out,
                           int *num_vulnerable_pairs_out, T min_x, T min_y,
                           T grid_len, int grid_dim_x, int grid_dim_y, int N,
                           T lower_bound_sq, T upper_bound_sq, T sign_bound_sq,
                           int N_local = 0) {
  int num_blocks = (N + num_threads - 1) / num_threads;

  // Pass 1: count-only to get exact pair count
  int *d_count;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairs2D_kernel<true><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_cell_start, d_cell_pts_sorted, d_editable_pts_ht,
      nullptr, nullptr, d_count, min_x, min_y, grid_len, grid_dim_x, grid_dim_y,
      N, lower_bound_sq, upper_bound_sq, sign_bound_sq, 0, N_local);

  int h_num_vulnerable_pairs;
  CUDA_CHECK(cudaMemcpy(&h_num_vulnerable_pairs, d_count, sizeof(int),
                        cudaMemcpyDeviceToHost));
  *num_vulnerable_pairs_out = h_num_vulnerable_pairs;

  // Pass 2: allocate with margin to absorb any count mismatch between passes
  int alloc_pairs =
      h_num_vulnerable_pairs + h_num_vulnerable_pairs / 100 + 1024;
  CUDA_CHECK(
      cudaMalloc(d_vulnerable_pairs_out, 2LL * alloc_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(d_signs_out, (size_t)alloc_pairs * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairs2D_kernel<false><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_cell_start, d_cell_pts_sorted, d_editable_pts_ht,
      *d_vulnerable_pairs_out, *d_signs_out, d_count, min_x, min_y, grid_len,
      grid_dim_x, grid_dim_y, N, lower_bound_sq, upper_bound_sq, sign_bound_sq,
      alloc_pairs, N_local);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_count2;
  CUDA_CHECK(
      cudaMemcpy(&h_count2, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_count2 != h_num_vulnerable_pairs) {
    printf("WARNING: pass2 count %d != pass1 count %d (delta %d)\n", h_count2,
           h_num_vulnerable_pairs, h_count2 - h_num_vulnerable_pairs);
  }
  *num_vulnerable_pairs_out = h_count2; // use actual pass2 count

  CUDA_CHECK(cudaFree(d_count));
}

template <typename T>
void findVulnerablePairs3D(const T *d_org_xx, const T *d_org_yy,
                           const T *d_org_zz, const int *d_cell_start,
                           const int *d_cell_pts_sorted,
                           HashTable &d_editable_pts_ht,
                           int **d_vulnerable_pairs_out, bool **d_signs_out,
                           int *num_vulnerable_pairs_out, T min_x, T min_y,
                           T min_z, T grid_len, int grid_dim_x, int grid_dim_y,
                           int grid_dim_z, int N, T lower_bound_sq,
                           T upper_bound_sq, T sign_bound_sq, int N_local = 0) {
  int num_blocks = (N + num_threads - 1) / num_threads;

  // Pass 1: count
  int *d_count;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairs3D_kernel<true><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, d_cell_start, d_cell_pts_sorted,
      d_editable_pts_ht, nullptr, nullptr, d_count, min_x, min_y, min_z,
      grid_len, grid_dim_x, grid_dim_y, grid_dim_z, N, lower_bound_sq,
      upper_bound_sq, sign_bound_sq, 0, N_local);
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_num_vulnerable_pairs;
  CUDA_CHECK(cudaMemcpy(&h_num_vulnerable_pairs, d_count, sizeof(int),
                        cudaMemcpyDeviceToHost));
  *num_vulnerable_pairs_out = h_num_vulnerable_pairs;
  printf("Vulnerable pairs counted: %d\n", h_num_vulnerable_pairs);

  // Pass 2: allocate with margin and populate
  int alloc_pairs =
      h_num_vulnerable_pairs + h_num_vulnerable_pairs / 100 + 1024;
  CUDA_CHECK(
      cudaMalloc(d_vulnerable_pairs_out, 2LL * alloc_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(d_signs_out, (size_t)alloc_pairs * sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  findVulnerablePairs3D_kernel<false><<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, d_cell_start, d_cell_pts_sorted,
      d_editable_pts_ht, *d_vulnerable_pairs_out, *d_signs_out, d_count, min_x,
      min_y, min_z, grid_len, grid_dim_x, grid_dim_y, grid_dim_z, N,
      lower_bound_sq, upper_bound_sq, sign_bound_sq, alloc_pairs, N_local);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_count2;
  CUDA_CHECK(
      cudaMemcpy(&h_count2, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (h_count2 != h_num_vulnerable_pairs) {
    printf("WARNING: pass2 count %d != pass1 count %d (delta %d)\n", h_count2,
           h_num_vulnerable_pairs, h_count2 - h_num_vulnerable_pairs);
  }
  *num_vulnerable_pairs_out = h_count2;

  CUDA_CHECK(cudaFree(d_count));
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
    std::vector<UInt2> quant_edits = huffmanZstdDecompress(
        compressed.compressed_quant_edits, compressed.code_table_edit,
        compressed.size_edit, compressed.bit_stream_size_edit);
    int E = static_cast<int>(compressed.editable_visit_positions.size());
    for (int j = 0; j < E; ++j) {
      int i = compressed.editable_visit_positions[j];
      decomp_xx[i] += static_cast<T>(quant_edits[2 * j]) * norm - 2 * xi;
      decomp_yy[i] += static_cast<T>(quant_edits[2 * j + 1]) * norm - 2 * xi;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
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
    std::vector<UInt2> quant_edits = huffmanZstdDecompress(
        compressed.compressed_quant_edits, compressed.code_table_edit,
        compressed.size_edit, compressed.bit_stream_size_edit);
    int E = static_cast<int>(compressed.editable_visit_positions.size());
    for (int j = 0; j < E; ++j) {
      int i = compressed.editable_visit_positions[j];
      decomp_xx[i] += static_cast<T>(quant_edits[3 * j]) * norm - 2 * xi;
      decomp_yy[i] += static_cast<T>(quant_edits[3 * j + 1]) * norm - 2 * xi;
      decomp_zz[i] += static_cast<T>(quant_edits[3 * j + 2]) * norm - 2 * xi;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
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

  std::vector<UInt2> quant_edits = huffmanZstdDecompress(
      compressed.compressed_quant_edits, compressed.code_table_edit,
      compressed.size_edit, compressed.bit_stream_size_edit);

  T norm = (4 * xi) / ((1 << m) - 1);
  int E = static_cast<int>(compressed.editable_visit_positions.size());
  for (int j = 0; j < E; ++j) {
    int i = compressed.editable_visit_positions[j];
    T edit = static_cast<T>(quant_edits[2 * j]) * norm - 2 * xi;
    decomp_xx[i] += edit;
    edit = static_cast<T>(quant_edits[2 * j + 1]) * norm - 2 * xi;
    decomp_yy[i] += edit;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

template <typename T>
void reconstructEditParticles3D(const CompressedData<T> &compressed,
                                T *decomp_xx, T *decomp_yy, T *decomp_zz, int N,
                                T xi) {
  auto start = std::chrono::high_resolution_clock::now();
  if (compressed.size_edit == 0)
    return;

  std::vector<UInt2> quant_edits = huffmanZstdDecompress(
      compressed.compressed_quant_edits, compressed.code_table_edit,
      compressed.size_edit, compressed.bit_stream_size_edit);

  T norm = (4 * xi) / ((1 << m) - 1);
  int E = static_cast<int>(compressed.editable_visit_positions.size());
  for (int j = 0; j < E; ++j) {
    int i = compressed.editable_visit_positions[j];
    T edit = static_cast<T>(quant_edits[3 * j]) * norm - 2 * xi;
    decomp_xx[i] += edit;
    edit = static_cast<T>(quant_edits[3 * j + 1]) * norm - 2 * xi;
    decomp_yy[i] += edit;
    edit = static_cast<T>(quant_edits[3 * j + 2]) * norm - 2 * xi;
    decomp_zz[i] += edit;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

// Compression with edit
template <typename T, OrderMode Mode>
void compressWithEditParticles2D(const T *d_org_xx, const T *d_org_yy, T min_x,
                                 T range_x, T min_y, T range_y, int N, T xi,
                                 T b, CompressionState2D<T> &state,
                                 CompressedData<T> &compressed, int N_local) {
  cudaEvent_t ev_start, ev_end, ev_s1a, ev_s1b, ev_s2a, ev_s2b;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_end));
  CUDA_CHECK(cudaEventCreate(&ev_s1a));
  CUDA_CHECK(cudaEventCreate(&ev_s1b));
  CUDA_CHECK(cudaEventCreate(&ev_s2a));
  CUDA_CHECK(cudaEventCreate(&ev_s2b));
  CUDA_CHECK(cudaEventRecord(ev_start));
  if (N == 0) {
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaEventDestroy(ev_s1a));
    CUDA_CHECK(cudaEventDestroy(ev_s1b));
    CUDA_CHECK(cudaEventDestroy(ev_s2a));
    CUDA_CHECK(cudaEventDestroy(ev_s2b));
    return;
  }

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.grid_len = b + 2 * std::sqrt(2) * xi;
  state.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / state.grid_len)));
  state.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / state.grid_len)));
  coarsenGrid2D(state.grid_len, state.grid_dim_x, state.grid_dim_y, range_x,
                range_y, std::max(1, N / 4));
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
  state.d_editable_pts_ht = createEmptyHashTable(N);
  findVulnerablePairs2D(d_org_xx, d_org_yy, state.d_cell_start,
                        state.d_visit_order, state.d_editable_pts_ht,
                        &state.d_vulnerable_pairs, &state.d_signs,
                        &state.num_vulnerable_pairs, min_x, min_y,
                        state.grid_len, state.grid_dim_x, state.grid_dim_y, N,
                        lower_bound_sq, upper_bound_sq, sign_bound_sq, N_local);
  compressed.N_local = static_cast<size_t>(N_local);

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
  CUDA_CHECK(cudaDeviceSynchronize());

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
  CUDA_CHECK(cudaDeviceSynchronize());

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

  CUDA_CHECK(cudaEventRecord(ev_s1a));
  printf("Number of vulnerable pairs: %d\n", state.num_vulnerable_pairs);
  printf("Number of editable particles: %d\n", state.num_editable_pts);

  // Calculate MCC and ARI metrics before PGD
  int tp_l, tn_l, fp_l, fn_l;
  getViolatedPairCount2D(state, b, tp_l, tn_l, fp_l, fn_l);
  T mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
  printf(
      "Before editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
      tp_l, tn_l, fp_l, fn_l, mcc);

  // long long tp_h, tn_h, fp_h, fn_h;
  // calculateARI2D(d_org_xx, d_org_yy, state.d_decomp_xx, state.d_decomp_yy,
  //                min_x, range_x, min_y, range_y, N, b, tp_h, tn_h, fp_h,
  //                fn_h);
  // T ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
  // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
  // tp_h,
  //        tn_h, fp_h, fn_h, ari);
  CUDA_CHECK(cudaEventRecord(ev_s1b));

  // PGD with Adam optimizer
  if (state.num_editable_pts > 0) {
    // Recreate HT (was freed before compression to reduce peak memory)
    state.d_editable_pts_ht = createEmptyHashTable(N);
    rebuildHashTableFromPairs(state.d_vulnerable_pairs,
                              state.num_vulnerable_pairs,
                              state.d_editable_pts_ht);
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
    T convergence_tol = max_quant_dist_err * max_quant_dist_err;
    T decomp_tol = convergence_tol * convergence_tol;

    // Adam hyperparameters
    T adam_alpha = static_cast<T>(lr);
    T adam_beta1 = static_cast<T>(0.9);
    T adam_beta2 = static_cast<T>(0.999);
    T adam_eps = static_cast<T>(1e-8);

    T final_loss = 0;
    int iter_used = 0;
    T beta1_t = 1, beta2_t = 1;

    int lossBlocks =
        (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
    int sharedMem = num_threads * sizeof(T);
    int updateBlocks = (E + num_threads - 1) / num_threads;
    constexpr int loss_check_interval = 10;

    for (int iter = 0; iter < max_iter; iter++) {
      // Check convergence periodically to reduce D2H sync overhead
      if (iter % loss_check_interval == 0) {
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
        computePGDLoss2D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
            state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
            state.d_decomp_yy, b, xi, state.num_vulnerable_pairs, d_loss,
            decomp_tol);
        CUDA_CHECK(
            cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
        if (final_loss < convergence_tol) {
          iter_used = iter + 1;
          break;
        }
      }

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));

      computePGDGradients2D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_editable_pts_ht, b,
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
      iter_used = iter + 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_m_x));
    CUDA_CHECK(cudaFree(d_m_y));
    CUDA_CHECK(cudaFree(d_v_x));
    CUDA_CHECK(cudaFree(d_v_y));
    CUDA_CHECK(cudaFree(d_loss));

    // Quantize edits (sparse: only E editable particles)
    int edit_values = 2 * E;
    CUDA_CHECK(cudaMalloc(&state.d_quant_edits, edit_values * sizeof(UInt2)));
    int *d_editable_visit_positions;
    CUDA_CHECK(cudaMalloc(&d_editable_visit_positions, E * sizeof(int)));
    T norm = ((1 << m) - 1) / (4 * xi);

    int quantBlocks = (N + num_threads - 1) / num_threads;
    quantizeEdits2D_kernel<<<quantBlocks, num_threads>>>(
        state.d_edit_x, state.d_edit_y, state.d_visit_order,
        state.d_editable_pts_ht, state.d_quant_edits,
        d_editable_visit_positions, xi, norm, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy editable visit positions to host
    compressed.editable_visit_positions.resize(E);
    CUDA_CHECK(cudaMemcpy(compressed.editable_visit_positions.data(),
                          d_editable_visit_positions, E * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_editable_visit_positions));

    // Apply quantized edits
    T dequant_norm = (4 * xi) / ((1 << m) - 1);
    int editBlocks = (E + num_threads - 1) / num_threads;
    applyQuantizedEdits2D_kernel<T><<<editBlocks, num_threads>>>(
        state.d_quant_edits, state.d_editable_pts_ht, state.d_decomp_xx,
        state.d_decomp_yy, state.d_edit_x, state.d_edit_y, xi, dequant_norm, E);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate MCC and ARI metrics after PGD
    CUDA_CHECK(cudaEventRecord(ev_s2a));
    printf("Number of iterations: %d\n", iter_used);
    printf("PGD final loss: %e\n", final_loss);
    getViolatedPairCount2D(state, b, tp_l, tn_l, fp_l, fn_l);
    mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
    printf(
        "After editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
        tp_l, tn_l, fp_l, fn_l, mcc);

    // calculateARI2D(d_org_xx, d_org_yy, state.d_decomp_xx, state.d_decomp_yy,
    //                min_x, range_x, min_y, range_y, N, b, tp_h, tn_h, fp_h,
    //                fn_h);
    // ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
    // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
    //        tp_h, tn_h, fp_h, fn_h, ari);
    CUDA_CHECK(cudaEventRecord(ev_s2b));

    // Huffman compress edits
    compressed.size_edit = edit_values;
    compressed.compressed_quant_edits = huffmanZstdCompressDevice(
        state.d_quant_edits, edit_values, compressed.code_table_edit,
        compressed.bit_stream_size_edit);
  } else {
    compressed.size_edit = 0;
  }

  CUDA_CHECK(cudaEventRecord(ev_end));
  CUDA_CHECK(cudaEventSynchronize(ev_end));
  float total_ms, s1_ms, s2_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
  CUDA_CHECK(cudaEventElapsedTime(&s1_ms, ev_s1a, ev_s1b));
  if (state.num_editable_pts > 0)
    CUDA_CHECK(cudaEventElapsedTime(&s2_ms, ev_s2a, ev_s2b));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_end));
  CUDA_CHECK(cudaEventDestroy(ev_s1a));
  CUDA_CHECK(cudaEventDestroy(ev_s1b));
  CUDA_CHECK(cudaEventDestroy(ev_s2a));
  CUDA_CHECK(cudaEventDestroy(ev_s2b));

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.compressed_quant_edits.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 2 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n",
         (total_ms - s1_ms - s2_ms) / 1000.0f);
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  calculateStatistics2D(d_org_xx, d_org_yy, state.d_decomp_xx,
                        state.d_decomp_yy, range_x, range_y, N);
}

template <typename T, OrderMode Mode>
void compressWithEditParticles3D(const T *d_org_xx, const T *d_org_yy,
                                 const T *d_org_zz, T min_x, T range_x, T min_y,
                                 T range_y, T min_z, T range_z, int N, T xi,
                                 T b, CompressionState3D<T> &state,
                                 CompressedData<T> &compressed, int N_local) {
  cudaEvent_t ev_start, ev_end, ev_s1a, ev_s1b, ev_s2a, ev_s2b;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_end));
  CUDA_CHECK(cudaEventCreate(&ev_s1a));
  CUDA_CHECK(cudaEventCreate(&ev_s1b));
  CUDA_CHECK(cudaEventCreate(&ev_s2a));
  CUDA_CHECK(cudaEventCreate(&ev_s2b));
  CUDA_CHECK(cudaEventRecord(ev_start));
  if (N == 0) {
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaEventDestroy(ev_s1a));
    CUDA_CHECK(cudaEventDestroy(ev_s1b));
    CUDA_CHECK(cudaEventDestroy(ev_s2a));
    CUDA_CHECK(cudaEventDestroy(ev_s2b));
    return;
  }

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.min_z = min_z;
  state.grid_len = b + 2 * std::sqrt(3) * xi;
  state.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / state.grid_len)));
  state.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / state.grid_len)));
  state.grid_dim_z =
      std::max(1, static_cast<int>(std::ceil(range_z / state.grid_len)));
  coarsenGrid3D(state.grid_len, state.grid_dim_x, state.grid_dim_y,
                state.grid_dim_z, range_x, range_y, range_z,
                std::max(1, N / 4));
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
  state.d_editable_pts_ht = createEmptyHashTable(N);
  findVulnerablePairs3D(
      d_org_xx, d_org_yy, d_org_zz, state.d_cell_start, state.d_visit_order,
      state.d_editable_pts_ht, &state.d_vulnerable_pairs, &state.d_signs,
      &state.num_vulnerable_pairs, min_x, min_y, min_z, state.grid_len,
      state.grid_dim_x, state.grid_dim_y, state.grid_dim_z, N, lower_bound_sq,
      upper_bound_sq, sign_bound_sq, N_local);
  compressed.N_local = static_cast<size_t>(N_local);

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
  CUDA_CHECK(cudaDeviceSynchronize());

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
  CUDA_CHECK(cudaDeviceSynchronize());

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

  CUDA_CHECK(cudaEventRecord(ev_s1a));
  printf("Number of vulnerable pairs: %d\n", state.num_vulnerable_pairs);
  printf("Number of editable particles: %d\n", state.num_editable_pts);

  // Calculate MCC and ARI metrics before PGD
  int tp_l, tn_l, fp_l, fn_l;
  getViolatedPairCount3D(state, b, tp_l, tn_l, fp_l, fn_l);
  T mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
  printf(
      "Before editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
      tp_l, tn_l, fp_l, fn_l, mcc);

  // long long tp_h, tn_h, fp_h, fn_h;
  // calculateARI3D(d_org_xx, d_org_yy, d_org_zz, state.d_decomp_xx,
  //                state.d_decomp_yy, state.d_decomp_zz, min_x, range_x, min_y,
  //                range_y, min_z, range_z, N, b, tp_h, tn_h, fp_h, fn_h);
  // T ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
  // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
  // tp_h,
  //        tn_h, fp_h, fn_h, ari);
  CUDA_CHECK(cudaEventRecord(ev_s1b));

  // PGD with Adam optimizer
  if (state.num_editable_pts > 0) {
    // Recreate HT (was freed before compression to reduce peak memory)
    state.d_editable_pts_ht = createEmptyHashTable(N);
    rebuildHashTableFromPairs(state.d_vulnerable_pairs,
                              state.num_vulnerable_pairs,
                              state.d_editable_pts_ht);

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
    T convergence_tol = max_quant_dist_err * max_quant_dist_err;
    T decomp_tol = convergence_tol * convergence_tol;

    // Adam hyperparameters
    T adam_alpha = static_cast<T>(lr);
    T adam_beta1 = static_cast<T>(0.9);
    T adam_beta2 = static_cast<T>(0.999);
    T adam_eps = static_cast<T>(1e-8);

    T final_loss = 0;
    int iter_used = 0;
    T beta1_t = 1, beta2_t = 1;
    int lossBlocks =
        (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
    int sharedMem = num_threads * sizeof(T);
    int updateBlocks = (E + num_threads - 1) / num_threads;
    constexpr int loss_check_interval = 10;

    for (int iter = 0; iter < max_iter; iter++) {
      // Check convergence periodically to reduce D2H sync overhead
      if (iter % loss_check_interval == 0) {
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
        computePGDLoss3D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
            state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
            state.d_decomp_yy, state.d_decomp_zz, b, xi,
            state.num_vulnerable_pairs, d_loss, decomp_tol);
        CUDA_CHECK(
            cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
        if (final_loss < convergence_tol) {
          iter_used = iter + 1;
          break;
        }
      }

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_z, 0, E * sizeof(T)));

      computePGDGradients3D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_decomp_zz, state.d_editable_pts_ht, b,
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
      iter_used = iter + 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

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

    // Quantize edits (sparse: only E editable particles)
    int edit_values = 3 * E;
    CUDA_CHECK(cudaMalloc(&state.d_quant_edits, edit_values * sizeof(UInt2)));
    int *d_editable_visit_positions;
    CUDA_CHECK(cudaMalloc(&d_editable_visit_positions, E * sizeof(int)));
    T norm = ((1 << m) - 1) / (4 * xi);

    int quantBlocks = (N + num_threads - 1) / num_threads;
    quantizeEdits3D_kernel<<<quantBlocks, num_threads>>>(
        state.d_edit_x, state.d_edit_y, state.d_edit_z, state.d_visit_order,
        state.d_editable_pts_ht, state.d_quant_edits,
        d_editable_visit_positions, xi, norm, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy editable visit positions to host
    compressed.editable_visit_positions.resize(E);
    CUDA_CHECK(cudaMemcpy(compressed.editable_visit_positions.data(),
                          d_editable_visit_positions, E * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_editable_visit_positions));

    // Apply quantized edits
    T dequant_norm = (4 * xi) / ((1 << m) - 1);
    int editBlocks = (E + num_threads - 1) / num_threads;
    applyQuantizedEdits3D_kernel<T><<<editBlocks, num_threads>>>(
        state.d_quant_edits, state.d_editable_pts_ht, state.d_decomp_xx,
        state.d_decomp_yy, state.d_decomp_zz, state.d_edit_x, state.d_edit_y,
        state.d_edit_z, xi, dequant_norm, E);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate MCC and ARI metrics after PGD
    CUDA_CHECK(cudaEventRecord(ev_s2a));
    printf("Number of iterations: %d\n", iter_used);
    printf("PGD final loss: %e\n", final_loss);
    getViolatedPairCount3D(state, b, tp_l, tn_l, fp_l, fn_l);
    mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
    printf(
        "After editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
        tp_l, tn_l, fp_l, fn_l, mcc);

    // calculateARI3D(d_org_xx, d_org_yy, d_org_zz, state.d_decomp_xx,
    //                state.d_decomp_yy, state.d_decomp_zz, min_x, range_x,
    //                min_y, range_y, min_z, range_z, N, b, tp_h, tn_h, fp_h,
    //                fn_h);
    // ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
    // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
    //        tp_h, tn_h, fp_h, fn_h, ari);
    CUDA_CHECK(cudaEventRecord(ev_s2b));

    // Huffman compress edits
    compressed.size_edit = edit_values;
    compressed.compressed_quant_edits = huffmanZstdCompressDevice(
        state.d_quant_edits, edit_values, compressed.code_table_edit,
        compressed.bit_stream_size_edit);
  } else {
    compressed.size_edit = 0;
  }

  CUDA_CHECK(cudaEventRecord(ev_end));
  CUDA_CHECK(cudaEventSynchronize(ev_end));
  float total_ms, s1_ms, s2_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
  CUDA_CHECK(cudaEventElapsedTime(&s1_ms, ev_s1a, ev_s1b));
  if (state.num_editable_pts > 0)
    CUDA_CHECK(cudaEventElapsedTime(&s2_ms, ev_s2a, ev_s2b));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_end));
  CUDA_CHECK(cudaEventDestroy(ev_s1a));
  CUDA_CHECK(cudaEventDestroy(ev_s1b));
  CUDA_CHECK(cudaEventDestroy(ev_s2a));
  CUDA_CHECK(cudaEventDestroy(ev_s2b));

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.compressed_quant_edits.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 3 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n",
         (total_ms - s1_ms - s2_ms) / 1000.0f);
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  calculateStatistics3D(d_org_xx, d_org_yy, d_org_zz, state.d_decomp_xx,
                        state.d_decomp_yy, state.d_decomp_zz, range_x, range_y,
                        range_z, N);
}

// Compression only
template <typename T, OrderMode Mode>
void compressParticles2D(const T *d_org_xx, const T *d_org_yy, T min_x,
                         T range_x, T min_y, T range_y, int N, T xi, T b,
                         CompressionState2D<T> &state,
                         CompressedData<T> &compressed) {
  cudaEvent_t ev_start, ev_end;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_end));
  CUDA_CHECK(cudaEventRecord(ev_start));
  if (N == 0) {
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    return;
  }

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.grid_len = b + 2 * std::sqrt(2) * xi;
  state.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / state.grid_len)));
  state.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / state.grid_len)));
  coarsenGrid2D(state.grid_len, state.grid_dim_x, state.grid_dim_y, range_x,
                range_y, std::max(1, N / 4));
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
  CUDA_CHECK(cudaDeviceSynchronize());

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
  CUDA_CHECK(cudaDeviceSynchronize());

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

  CUDA_CHECK(cudaEventRecord(ev_end));
  CUDA_CHECK(cudaEventSynchronize(ev_end));
  float comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, ev_start, ev_end));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_end));

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 2 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n", comp_time_ms / 1000.0f);
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  calculateStatistics2D(d_org_xx, d_org_yy, state.d_decomp_xx,
                        state.d_decomp_yy, range_x, range_y, N);
}

template <typename T, OrderMode Mode>
void compressParticles3D(const T *d_org_xx, const T *d_org_yy,
                         const T *d_org_zz, T min_x, T range_x, T min_y,
                         T range_y, T min_z, T range_z, int N, T xi, T b,
                         CompressionState3D<T> &state,
                         CompressedData<T> &compressed) {
  cudaEvent_t ev_start, ev_end;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_end));
  CUDA_CHECK(cudaEventRecord(ev_start));
  if (N == 0) {
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    return;
  }

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.min_z = min_z;
  state.grid_len = b + 2 * std::sqrt(3) * xi;
  state.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / state.grid_len)));
  state.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / state.grid_len)));
  state.grid_dim_z =
      std::max(1, static_cast<int>(std::ceil(range_z / state.grid_len)));
  coarsenGrid3D(state.grid_len, state.grid_dim_x, state.grid_dim_y,
                state.grid_dim_z, range_x, range_y, range_z,
                std::max(1, N / 4));
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
  CUDA_CHECK(cudaDeviceSynchronize());

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
  CUDA_CHECK(cudaDeviceSynchronize());

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

  CUDA_CHECK(cudaEventRecord(ev_end));
  CUDA_CHECK(cudaEventSynchronize(ev_end));
  float comp_time_ms;
  CUDA_CHECK(cudaEventElapsedTime(&comp_time_ms, ev_start, ev_end));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_end));

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      sizeof(T) * compressed.lossless_values.size();

  T compression_ratio = (sizeof(T) * num_values) / (T)compressed_size;
  T bpp = 3 * 8 * sizeof(T) / compression_ratio;

  printf("Compression time: %f seconds\n", comp_time_ms / 1000.0f);
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  calculateStatistics3D(d_org_xx, d_org_yy, d_org_zz, state.d_decomp_xx,
                        state.d_decomp_yy, state.d_decomp_zz, range_x, range_y,
                        range_z, N);
}

// Edit only
template <typename T, OrderMode Mode>
void editParticles2D(const T *d_org_xx, const T *d_org_yy, T *d_base_decomp_xx,
                     T *d_base_decomp_yy, T min_x, T range_x, T min_y,
                     T range_y, int N, T xi, T b, CompressionState2D<T> &state,
                     CompressedData<T> &compressed, int N_local,
                     MPI_Comm comm) {
  cudaEvent_t ev_start, ev_end, ev_s1a, ev_s1b, ev_s2a, ev_s2b;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_end));
  CUDA_CHECK(cudaEventCreate(&ev_s1a));
  CUDA_CHECK(cudaEventCreate(&ev_s1b));
  CUDA_CHECK(cudaEventCreate(&ev_s2a));
  CUDA_CHECK(cudaEventCreate(&ev_s2b));
  CUDA_CHECK(cudaEventRecord(ev_start));
  if (N == 0) {
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaEventDestroy(ev_s1a));
    CUDA_CHECK(cudaEventDestroy(ev_s1b));
    CUDA_CHECK(cudaEventDestroy(ev_s2a));
    CUDA_CHECK(cudaEventDestroy(ev_s2b));
    return;
  }

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.grid_len = b + 2 * std::sqrt(2) * xi;
  state.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / state.grid_len)));
  state.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / state.grid_len)));
  coarsenGrid2D(state.grid_len, state.grid_dim_x, state.grid_dim_y, range_x,
                range_y, std::max(1, N / 4));
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
  state.d_editable_pts_ht = createEmptyHashTable(N);
  findVulnerablePairs2D(d_org_xx, d_org_yy, state.d_cell_start,
                        state.d_visit_order, state.d_editable_pts_ht,
                        &state.d_vulnerable_pairs, &state.d_signs,
                        &state.num_vulnerable_pairs, min_x, min_y,
                        state.grid_len, state.grid_dim_x, state.grid_dim_y, N,
                        lower_bound_sq, upper_bound_sq, sign_bound_sq, N_local);
  compressed.N_local = static_cast<size_t>(N_local);
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;

  // Use provided base decompressed coordinates as state decomp arrays
  state.d_decomp_xx = d_base_decomp_xx;
  state.d_decomp_yy = d_base_decomp_yy;

  // Edit-only mode: overwrite d_visit_order with iota (particle index order)
  {
    int iota_blocks = (N + num_threads - 1) / num_threads;
    iota_kernel<<<iota_blocks, num_threads>>>(state.d_visit_order, N);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Get editable particle count
  CUDA_CHECK(cudaMemcpy(&state.num_editable_pts,
                        state.d_editable_pts_ht.counter, sizeof(int),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(ev_s1a));
  printf("Number of vulnerable pairs: %d\n", state.num_vulnerable_pairs);
  printf("Number of editable particles: %d\n", state.num_editable_pts);

  // Calculate MCC and ARI metrics before PGD
  int tp_l, tn_l, fp_l, fn_l;
  getViolatedPairCount2D(state, b, tp_l, tn_l, fp_l, fn_l);
  T mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
  printf(
      "Before editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
      tp_l, tn_l, fp_l, fn_l, mcc);

  // long long tp_h, tn_h, fp_h, fn_h;
  // calculateARI2D(d_org_xx, d_org_yy, state.d_decomp_xx, state.d_decomp_yy,
  //                min_x, range_x, min_y, range_y, N, b, tp_h, tn_h, fp_h,
  //                fn_h);
  // T ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
  // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
  // tp_h,
  //        tn_h, fp_h, fn_h, ari);
  CUDA_CHECK(cudaEventRecord(ev_s1b));

  // PGD with Adam optimizer
  T final_loss = 0;
  int iter_used = 0;
  if (state.num_editable_pts > 0) {
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
    T convergence_tol = max_quant_dist_err * max_quant_dist_err;
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
    constexpr int loss_check_interval = 10;

    for (int iter = 0; iter < max_iter; iter++) {
      // Check convergence periodically to reduce D2H sync overhead
      if (iter % loss_check_interval == 0) {
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
          if (global_loss < convergence_tol)
            break;
        } else
#endif
            if (final_loss < convergence_tol)
          break;
      }

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));

      computePGDGradients2D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_editable_pts_ht, b,
          state.num_vulnerable_pairs, decomp_tol, d_grad_x, d_grad_y);

      beta1_t *= adam_beta1;
      beta2_t *= adam_beta2;
      T lr_t = adam_alpha * sqrt(1 - beta2_t) / (1 - beta1_t);

      updatePGDPositionsAdam2D_kernel<<<updateBlocks, num_threads>>>(
          d_org_xx, d_org_yy, d_grad_x, d_grad_y, state.d_editable_pts_ht,
          state.d_decomp_xx, state.d_decomp_yy, state.d_edit_x, state.d_edit_y,
          d_m_x, d_m_y, d_v_x, d_v_y, adam_beta1, adam_beta2, adam_eps, lr_t,
          xi);
      iter_used = iter + 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_grad_x));
    CUDA_CHECK(cudaFree(d_grad_y));
    CUDA_CHECK(cudaFree(d_m_x));
    CUDA_CHECK(cudaFree(d_m_y));
    CUDA_CHECK(cudaFree(d_v_x));
    CUDA_CHECK(cudaFree(d_v_y));
    CUDA_CHECK(cudaFree(d_loss));

    // Quantize edits on GPU (sparse: only E editable particles)
    int edit_values = 2 * E;
    CUDA_CHECK(cudaMalloc(&state.d_quant_edits, edit_values * sizeof(UInt2)));
    int *d_editable_visit_positions;
    CUDA_CHECK(cudaMalloc(&d_editable_visit_positions, E * sizeof(int)));
    T norm = ((1 << m) - 1) / (4 * xi);

    int quantBlocks = (N + num_threads - 1) / num_threads;
    quantizeEdits2D_kernel<<<quantBlocks, num_threads>>>(
        state.d_edit_x, state.d_edit_y, state.d_visit_order,
        state.d_editable_pts_ht, state.d_quant_edits,
        d_editable_visit_positions, xi, norm, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy editable visit positions to host
    compressed.editable_visit_positions.resize(E);
    CUDA_CHECK(cudaMemcpy(compressed.editable_visit_positions.data(),
                          d_editable_visit_positions, E * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_editable_visit_positions));

    // Apply quantized edits on GPU
    T dequant_norm = (4 * xi) / ((1 << m) - 1);
    int editBlocks = (E + num_threads - 1) / num_threads;
    applyQuantizedEdits2D_kernel<T><<<editBlocks, num_threads>>>(
        state.d_quant_edits, state.d_editable_pts_ht, state.d_decomp_xx,
        state.d_decomp_yy, state.d_edit_x, state.d_edit_y, xi, dequant_norm, E);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate MCC and ARI metrics after PGD
    CUDA_CHECK(cudaEventRecord(ev_s2a));
    printf("Number of iterations: %d\n", iter_used);
    printf("PGD final loss: %e\n", final_loss);
    getViolatedPairCount2D(state, b, tp_l, tn_l, fp_l, fn_l);
    mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
    printf(
        "After editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
        tp_l, tn_l, fp_l, fn_l, mcc);

    // calculateARI2D(d_org_xx, d_org_yy, state.d_decomp_xx, state.d_decomp_yy,
    //                min_x, range_x, min_y, range_y, N, b, tp_h, tn_h, fp_h,
    //                fn_h);
    // ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
    // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
    //        tp_h, tn_h, fp_h, fn_h, ari);
    CUDA_CHECK(cudaEventRecord(ev_s2b));

    // Huffman compress edits directly from device
    compressed.size_edit = edit_values;
    compressed.compressed_quant_edits = huffmanZstdCompressDevice(
        state.d_quant_edits, edit_values, compressed.code_table_edit,
        compressed.bit_stream_size_edit);
  } else {
    compressed.size_edit = 0;
  }

  CUDA_CHECK(cudaEventRecord(ev_end));
  CUDA_CHECK(cudaEventSynchronize(ev_end));
  float total_ms, s1_ms, s2_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
  CUDA_CHECK(cudaEventElapsedTime(&s1_ms, ev_s1a, ev_s1b));
  if (state.num_editable_pts > 0)
    CUDA_CHECK(cudaEventElapsedTime(&s2_ms, ev_s2a, ev_s2b));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_end));
  CUDA_CHECK(cudaEventDestroy(ev_s1a));
  CUDA_CHECK(cudaEventDestroy(ev_s1b));
  CUDA_CHECK(cudaEventDestroy(ev_s2a));
  CUDA_CHECK(cudaEventDestroy(ev_s2b));

  size_t additional_size =
      compressed.compressed_quant_edits.size() +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  printf("Edit time: %f seconds\n", (total_ms - s1_ms - s2_ms) / 1000.0f);
  printf("Additional storage: %zu bytes\n", additional_size);
  calculateStatistics2D(d_org_xx, d_org_yy, state.d_decomp_xx,
                        state.d_decomp_yy, range_x, range_y, N);
}

template <typename T, OrderMode Mode>
void editParticles3D(const T *d_org_xx, const T *d_org_yy, const T *d_org_zz,
                     T *d_base_decomp_xx, T *d_base_decomp_yy,
                     T *d_base_decomp_zz, T min_x, T range_x, T min_y,
                     T range_y, T min_z, T range_z, int N, T xi, T b,
                     CompressionState3D<T> &state,
                     CompressedData<T> &compressed, int N_local,
                     MPI_Comm comm) {
#ifdef USE_MPI
  double t_fof_start = MPI_Wtime();
#else
  double t_fof_start = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
  cudaEvent_t ev_start, ev_end, ev_s1a, ev_s1b, ev_s2a, ev_s2b;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_end));
  CUDA_CHECK(cudaEventCreate(&ev_s1a));
  CUDA_CHECK(cudaEventCreate(&ev_s1b));
  CUDA_CHECK(cudaEventCreate(&ev_s2a));
  CUDA_CHECK(cudaEventCreate(&ev_s2b));
  CUDA_CHECK(cudaEventRecord(ev_start));
  if (N == 0) {
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaEventDestroy(ev_s1a));
    CUDA_CHECK(cudaEventDestroy(ev_s1b));
    CUDA_CHECK(cudaEventDestroy(ev_s2a));
    CUDA_CHECK(cudaEventDestroy(ev_s2b));
    return;
  }

  state.xi = xi;
  state.b = b;
  state.N = N;
  state.min_x = min_x;
  state.min_y = min_y;
  state.min_z = min_z;
  state.grid_len = b + 2 * std::sqrt(3) * xi;
  state.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / state.grid_len)));
  state.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / state.grid_len)));
  state.grid_dim_z =
      std::max(1, static_cast<int>(std::ceil(range_z / state.grid_len)));
  coarsenGrid3D(state.grid_len, state.grid_dim_x, state.grid_dim_y,
                state.grid_dim_z, range_x, range_y, range_z,
                std::max(1, N / 4));
  state.num_cells = state.grid_dim_x * state.grid_dim_y * state.grid_dim_z;

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
  state.d_editable_pts_ht = createEmptyHashTable(N);
  findVulnerablePairs3D(
      d_org_xx, d_org_yy, d_org_zz, state.d_cell_start, state.d_visit_order,
      state.d_editable_pts_ht, &state.d_vulnerable_pairs, &state.d_signs,
      &state.num_vulnerable_pairs, min_x, min_y, min_z, state.grid_len,
      state.grid_dim_x, state.grid_dim_y, state.grid_dim_z, N, lower_bound_sq,
      upper_bound_sq, sign_bound_sq, N_local);
  compressed.N_local = static_cast<size_t>(N_local);
  CUDA_CHECK(cudaFree(state.d_cell_start));
  state.d_cell_start = nullptr;

  // Use provided base decompressed coordinates as state decomp arrays
  state.d_decomp_xx = d_base_decomp_xx;
  state.d_decomp_yy = d_base_decomp_yy;
  state.d_decomp_zz = d_base_decomp_zz;

  // Edit-only mode: overwrite d_visit_order with iota (particle index order)
  {
    int iota_blocks = (N + num_threads - 1) / num_threads;
    iota_kernel<<<iota_blocks, num_threads>>>(state.d_visit_order, N);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Get editable particle count
  CUDA_CHECK(cudaMemcpy(&state.num_editable_pts,
                        state.d_editable_pts_ht.counter, sizeof(int),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventRecord(ev_s1a));
  printf("Number of vulnerable pairs: %d\n", state.num_vulnerable_pairs);
  printf("Number of editable particles: %d\n", state.num_editable_pts);

  // Calculate MCC and ARI metrics before PGD
  int tp_l, tn_l, fp_l, fn_l;
  getViolatedPairCount3D(state, b, tp_l, tn_l, fp_l, fn_l);
  T mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
  printf(
      "Before editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
      tp_l, tn_l, fp_l, fn_l, mcc);

  // long long tp_h, tn_h, fp_h, fn_h;
  // calculateARI3D(d_org_xx, d_org_yy, d_org_zz, state.d_decomp_xx,
  //                state.d_decomp_yy, state.d_decomp_zz, min_x, range_x, min_y,
  //                range_y, min_z, range_z, N, b, tp_h, tn_h, fp_h, fn_h);
  // T ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
  // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
  // tp_h,
  //        tn_h, fp_h, fn_h, ari);
  CUDA_CHECK(cudaEventRecord(ev_s1b));

#ifdef USE_MPI
  printf("[Timer] FOF setup: %f seconds\n", MPI_Wtime() - t_fof_start);
#else
  printf("[Timer] FOF setup: %f seconds\n",
         std::chrono::duration<double>(
             std::chrono::high_resolution_clock::now().time_since_epoch()).count()
         - t_fof_start);
#endif
  fflush(stdout);

  // PGD with Adam optimizer
#ifdef USE_MPI
  double t_pgd_start = MPI_Wtime();
#else
  double t_pgd_start = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
  double t_allreduce_total = 0.0;
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
    T convergence_tol = max_quant_dist_err * max_quant_dist_err;
    T decomp_tol = convergence_tol * convergence_tol;

    T adam_alpha = static_cast<T>(lr);
    T adam_beta1 = static_cast<T>(0.9);
    T adam_beta2 = static_cast<T>(0.999);
    T adam_eps = static_cast<T>(1e-8);

    T final_loss = 0;
    int iter_used = 0;
    T beta1_t = 1, beta2_t = 1;
    int lossBlocks =
        (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
    int sharedMem = num_threads * sizeof(T);
    int updateBlocks = (E + num_threads - 1) / num_threads;

    for (int iter = 0; iter < max_iter; iter++) {
      // Check convergence periodically to reduce D2H sync overhead
      CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(T)));
      computePGDLoss3D_kernel<<<lossBlocks, num_threads, sharedMem>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_decomp_zz, b, xi,
          state.num_vulnerable_pairs, d_loss, decomp_tol);
      CUDA_CHECK(
          cudaMemcpy(&final_loss, d_loss, sizeof(T), cudaMemcpyDeviceToHost));
      // printf("iter %d, loss: %f\n", iter, final_loss);
#ifdef USE_MPI
      if (comm != MPI_COMM_NULL) {
        T global_loss;
        double t_ar0 = MPI_Wtime();
        MPI_Allreduce(&final_loss, &global_loss, 1,
                      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                      MPI_SUM, comm);
        t_allreduce_total += MPI_Wtime() - t_ar0;
        if (global_loss < convergence_tol)
          break;
      } else
#endif
          if (final_loss < convergence_tol)
        break;

      CUDA_CHECK(cudaMemset(d_grad_x, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_y, 0, E * sizeof(T)));
      CUDA_CHECK(cudaMemset(d_grad_z, 0, E * sizeof(T)));

      computePGDGradients3D_kernel<<<lossBlocks, num_threads>>>(
          state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
          state.d_decomp_yy, state.d_decomp_zz, state.d_editable_pts_ht, b,
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
      iter_used = iter + 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
#ifdef USE_MPI
    double t_pgd_elapsed = MPI_Wtime() - t_pgd_start;
#else
    double t_pgd_elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()
        - t_pgd_start;
#endif
    printf("[Timer] PGD iterations (%d iters): %f seconds\n", iter_used,
           t_pgd_elapsed);
    printf("[Timer] MPI_Allreduce (loss, %d calls): %f seconds\n", iter_used,
           t_allreduce_total);
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

    // Quantize edits on GPU (sparse: only E editable particles)
    int edit_values = 3 * E;
    CUDA_CHECK(cudaMalloc(&state.d_quant_edits, edit_values * sizeof(UInt2)));
    int *d_editable_visit_positions;
    CUDA_CHECK(cudaMalloc(&d_editable_visit_positions, E * sizeof(int)));
    T norm = ((1 << m) - 1) / (4 * xi);

    int quantBlocks = (N + num_threads - 1) / num_threads;
    quantizeEdits3D_kernel<<<quantBlocks, num_threads>>>(
        state.d_edit_x, state.d_edit_y, state.d_edit_z, state.d_visit_order,
        state.d_editable_pts_ht, state.d_quant_edits,
        d_editable_visit_positions, xi, norm, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy editable visit positions to host
    compressed.editable_visit_positions.resize(E);
    CUDA_CHECK(cudaMemcpy(compressed.editable_visit_positions.data(),
                          d_editable_visit_positions, E * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_editable_visit_positions));

    // Apply quantized edits on GPU
    T dequant_norm = (4 * xi) / ((1 << m) - 1);
    int editBlocks = (E + num_threads - 1) / num_threads;
    applyQuantizedEdits3D_kernel<T><<<editBlocks, num_threads>>>(
        state.d_quant_edits, state.d_editable_pts_ht, state.d_decomp_xx,
        state.d_decomp_yy, state.d_decomp_zz, state.d_edit_x, state.d_edit_y,
        state.d_edit_z, xi, dequant_norm, E);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate MCC and ARI metrics after PGD
    CUDA_CHECK(cudaEventRecord(ev_s2a));
    printf("Number of iterations: %d\n", iter_used);
    printf("PGD final loss: %e\n", final_loss);
    getViolatedPairCount3D(state, b, tp_l, tn_l, fp_l, fn_l);
    mcc = calculateMCC<T>(tp_l, tn_l, fp_l, fn_l);
    printf(
        "After editing: TP_l = %d, TN_l = %d, FP_l = %d, FN_l = %d, MCC = %f\n",
        tp_l, tn_l, fp_l, fn_l, mcc);

    // calculateARI3D(d_org_xx, d_org_yy, d_org_zz, state.d_decomp_xx,
    //                state.d_decomp_yy, state.d_decomp_zz, min_x, range_x,
    //                min_y, range_y, min_z, range_z, N, b, tp_h, tn_h, fp_h,
    //                fn_h);
    // ari = calculateARI<T>(tp_h, tn_h, fp_h, fn_h);
    // printf("TP_h = %lld, TN_h = %lld, FP_h = %lld, FN_h = %lld, ARI = %f\n",
    //        tp_h, tn_h, fp_h, fn_h, ari);
    CUDA_CHECK(cudaEventRecord(ev_s2b));

    // Huffman compress edits
    compressed.size_edit = edit_values;
    compressed.compressed_quant_edits = huffmanZstdCompressDevice(
        state.d_quant_edits, edit_values, compressed.code_table_edit,
        compressed.bit_stream_size_edit);
  } else {
    compressed.size_edit = 0;
  }

  CUDA_CHECK(cudaEventRecord(ev_end));
  CUDA_CHECK(cudaEventSynchronize(ev_end));
  float total_ms, s1_ms, s2_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
  CUDA_CHECK(cudaEventElapsedTime(&s1_ms, ev_s1a, ev_s1b));
  if (state.num_editable_pts > 0)
    CUDA_CHECK(cudaEventElapsedTime(&s2_ms, ev_s2a, ev_s2b));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_end));
  CUDA_CHECK(cudaEventDestroy(ev_s1a));
  CUDA_CHECK(cudaEventDestroy(ev_s1b));
  CUDA_CHECK(cudaEventDestroy(ev_s2a));
  CUDA_CHECK(cudaEventDestroy(ev_s2b));

  size_t additional_size =
      compressed.compressed_quant_edits.size() +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  printf("Edit time: %f seconds\n", (total_ms - s1_ms - s2_ms) / 1000.0f);
  printf("Additional storage: %zu bytes\n", additional_size);
  calculateStatistics3D(d_org_xx, d_org_yy, d_org_zz, state.d_decomp_xx,
                        state.d_decomp_yy, state.d_decomp_zz, range_x, range_y,
                        range_z, N);
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

template <typename T>
void getViolatedPairCount2D(const CompressionState2D<T> &state, T b, int &h_tp,
                            int &h_tn, int &h_fp, int &h_fn) {
  if (state.num_vulnerable_pairs == 0) {
    h_tp = 0;
    h_tn = 0;
    h_fp = 0;
    h_fn = 0;
    return;
  }

  int *d_tp, *d_tn, *d_fp, *d_fn;
  CUDA_CHECK(cudaMalloc(&d_tp, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tn, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fp, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fn, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_tp, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_tn, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_fp, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_fn, 0, sizeof(int)));

  int vblocks = (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
  countViolations2D_kernel<T><<<vblocks, num_threads>>>(
      state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
      state.d_decomp_yy, b * b, state.num_vulnerable_pairs, d_tp, d_tn, d_fp,
      d_fn);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_tp, d_tp, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_tn, d_tn, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_fp, d_fp, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_fn, d_fn, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_tp));
  CUDA_CHECK(cudaFree(d_tn));
  CUDA_CHECK(cudaFree(d_fp));
  CUDA_CHECK(cudaFree(d_fn));
}

template <typename T>
void getViolatedPairCount3D(const CompressionState3D<T> &state, T b, int &h_tp,
                            int &h_tn, int &h_fp, int &h_fn) {
  if (state.num_vulnerable_pairs == 0) {
    h_tp = 0;
    h_tn = 0;
    h_fp = 0;
    h_fn = 0;
    return;
  }

  int *d_tp, *d_tn, *d_fp, *d_fn;
  CUDA_CHECK(cudaMalloc(&d_tp, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tn, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fp, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fn, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_tp, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_tn, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_fp, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_fn, 0, sizeof(int)));

  int vblocks = (state.num_vulnerable_pairs + num_threads - 1) / num_threads;
  countViolations3D_kernel<T><<<vblocks, num_threads>>>(
      state.d_vulnerable_pairs, state.d_signs, state.d_decomp_xx,
      state.d_decomp_yy, state.d_decomp_zz, b * b, state.num_vulnerable_pairs,
      d_tp, d_tn, d_fp, d_fn);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_tp, d_tp, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_tn, d_tn, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_fp, d_fp, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_fn, d_fn, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_tp));
  CUDA_CHECK(cudaFree(d_tn));
  CUDA_CHECK(cudaFree(d_fp));
  CUDA_CHECK(cudaFree(d_fn));
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

template void getViolatedPairCount2D<float>(const CompressionState2D<float> &,
                                            float, int &, int &, int &, int &);
template void getViolatedPairCount2D<double>(const CompressionState2D<double> &,
                                             double, int &, int &, int &,
                                             int &);
template void getViolatedPairCount3D<float>(const CompressionState3D<float> &,
                                            float, int &, int &, int &, int &);
template void getViolatedPairCount3D<double>(const CompressionState3D<double> &,
                                             double, int &, int &, int &,
                                             int &);