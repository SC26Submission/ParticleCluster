#include "FOFHaloFinder.cuh"
#include "particle_compression.cuh"

// Initialize Union-Find: each particle is its own parent
__global__ void initUnionFind_kernel(int *parent, int *rank, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    parent[idx] = idx;
    rank[idx] = 0;
  }
}

// Flatten union-find: resolve each element to its root
__global__ void flattenRoots_kernel(int *parent, int *roots, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    roots[idx] = find_uf(parent, idx);
  }
}

// Compute C(n,2) = n*(n-1)/2 for each count in an array, store as long long
__global__ void chooseTwo_kernel(const int *counts, long long *results,
                                 int num_entries) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_entries) {
    long long n = counts[idx];
    results[idx] = n * (n - 1) / 2;
  }
}

// Pack two int32 roots into one int64 key for contingency table
__global__ void packPairKeys_kernel(const int *org_roots,
                                    const int *decomp_roots, long long *keys,
                                    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    keys[idx] = ((long long)(unsigned int)org_roots[idx] << 32) |
                (unsigned int)decomp_roots[idx];
  }
}

// Kernel to build halos using union-find (2D)
template <typename T>
__global__ void
buildHalos2D_kernel(const T *d_xx, const T *d_yy, const int *d_cell_start,
                    const int *d_cell_pts_sorted, int num_cells, int N, T min_x,
                    T min_y, T b, int grid_dim_x, int grid_dim_y, T b_sq,
                    int *d_parent, int *d_rank) {
  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  if (cell_start == cell_end)
    return;

  int id_x = cell_id % grid_dim_x;
  int id_y = cell_id / grid_dim_x;

  // Check pairs within the same cell
  for (int i = cell_start; i < cell_end; i++) {
    int pi = d_cell_pts_sorted[i];
    T pi_x = d_xx[pi];
    T pi_y = d_yy[pi];

    for (int j = i + 1; j < cell_end; j++) {
      int pj = d_cell_pts_sorted[j];
      T pj_x = d_xx[pj];
      T pj_y = d_yy[pj];

      T dx = pi_x - pj_x;
      T dy = pi_y - pj_y;
      T dist_sq = dx * dx + dy * dy;

      if (dist_sq <= b_sq) {
        union_uf(d_parent, d_rank, pi, pj);
      }
    }
  }

  // Check pairs with forward neighboring cells
  static constexpr int neighbor_offsets[4][2] = {
      {0, 1}, {1, 0}, {1, 1}, {1, -1}};
  for (int idx = 0; idx < 4; ++idx) {
    int nx = id_x + neighbor_offsets[idx][0];
    int ny = id_y + neighbor_offsets[idx][1];

    if (nx < 0 || nx >= grid_dim_x || ny < 0 || ny >= grid_dim_y)
      continue;

    int neighbor_cell_id = ny * grid_dim_x + nx;
    int neighbor_start = d_cell_start[neighbor_cell_id];
    int neighbor_end = (neighbor_cell_id == num_cells - 1)
                           ? N
                           : d_cell_start[neighbor_cell_id + 1];

    for (int i = cell_start; i < cell_end; i++) {
      int pi = d_cell_pts_sorted[i];
      T pi_x = d_xx[pi];
      T pi_y = d_yy[pi];

      for (int j = neighbor_start; j < neighbor_end; j++) {
        int pj = d_cell_pts_sorted[j];
        T pj_x = d_xx[pj];
        T pj_y = d_yy[pj];

        T dx = pi_x - pj_x;
        T dy = pi_y - pj_y;
        T dist_sq = dx * dx + dy * dy;

        if (dist_sq <= b_sq) {
          union_uf(d_parent, d_rank, pi, pj);
        }
      }
    }
  }
}

// Kernel to build halos using union-find (3D)
template <typename T>
__global__ void
buildHalos3D_kernel(const T *d_xx, const T *d_yy, const T *d_zz,
                    const int *d_cell_start, const int *d_cell_pts_sorted,
                    int num_cells, int N, T min_x, T min_y, T min_z, T b,
                    int grid_dim_x, int grid_dim_y, int grid_dim_z, T b_sq,
                    int *d_parent, int *d_rank) {
  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  if (cell_start == cell_end)
    return;

  int id_x = cell_id % grid_dim_x;
  int id_y = (cell_id / grid_dim_x) % grid_dim_y;
  int id_z = cell_id / (grid_dim_x * grid_dim_y);

  // Check pairs within the same cell first
  for (int i = cell_start; i < cell_end; i++) {
    int pi = d_cell_pts_sorted[i];
    T pi_x = d_xx[pi];
    T pi_y = d_yy[pi];
    T pi_z = d_zz[pi];

    for (int j = i + 1; j < cell_end; j++) {
      int pj = d_cell_pts_sorted[j];
      T pj_x = d_xx[pj];
      T pj_y = d_yy[pj];
      T pj_z = d_zz[pj];

      T dx = pi_x - pj_x;
      T dy = pi_y - pj_y;
      T dz = pi_z - pj_z;
      T dist_sq = dx * dx + dy * dy + dz * dz;

      if (dist_sq <= b_sq) {
        union_uf(d_parent, d_rank, pi, pj);
      }
    }
  }

  // Check pairs with forward neighboring cells (only 13 to avoid duplicates)
  static constexpr int neighbor_offsets[13][3] = {
      {1, 0, 0},  {0, 1, 0},  {0, 0, 1},  {1, 1, 0},  {1, -1, 0},
      {1, 0, 1},  {1, 0, -1}, {0, 1, 1},  {0, 1, -1}, {1, 1, 1},
      {1, 1, -1}, {1, -1, 1}, {1, -1, -1}};
  for (int idx = 0; idx < 13; ++idx) {
    int nx = id_x + neighbor_offsets[idx][0];
    int ny = id_y + neighbor_offsets[idx][1];
    int nz = id_z + neighbor_offsets[idx][2];

    if (nx < 0 || nx >= grid_dim_x || ny < 0 || ny >= grid_dim_y || nz < 0 ||
        nz >= grid_dim_z)
      continue;

    int neighbor_cell_id = nz * grid_dim_x * grid_dim_y + ny * grid_dim_x + nx;
    int neighbor_start = d_cell_start[neighbor_cell_id];
    int neighbor_end = (neighbor_cell_id == num_cells - 1)
                           ? N
                           : d_cell_start[neighbor_cell_id + 1];

    for (int i = cell_start; i < cell_end; i++) {
      int pi = d_cell_pts_sorted[i];
      T pi_x = d_xx[pi];
      T pi_y = d_yy[pi];
      T pi_z = d_zz[pi];

      for (int j = neighbor_start; j < neighbor_end; j++) {
        int pj = d_cell_pts_sorted[j];
        T pj_x = d_xx[pj];
        T pj_y = d_yy[pj];
        T pj_z = d_zz[pj];

        T dx = pi_x - pj_x;
        T dy = pi_y - pj_y;
        T dz = pi_z - pj_z;
        T dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq <= b_sq) {
          union_uf(d_parent, d_rank, pi, pj);
        }
      }
    }
  }
}

// ============================================================================
// CONTINGENCY-TABLE ARI: O(N log N)
// ============================================================================

// After building union-find for both partitions, compute TP/TN/FP/FN via:
//   TP = sum_ij C(n_ij, 2)
//   TP + FN = sum_i C(a_i, 2)   (a_i = row sums = org cluster sizes)
//   TP + FP = sum_j C(b_j, 2)   (b_j = col sums = decomp cluster sizes)
//   TP + FP + FN + TN = C(N, 2)

static void computeTPTNFPFN(UnionFind &org_uf, UnionFind &decomp_uf, int N,
                             long long &h_tp, long long &h_tn, long long &h_fp,
                             long long &h_fn) {
  int blocks = (N + num_threads - 1) / num_threads;

  // Step 1: flatten roots
  int *d_org_roots, *d_decomp_roots;
  CUDA_CHECK(cudaMalloc(&d_org_roots, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_decomp_roots, N * sizeof(int)));
  flattenRoots_kernel<<<blocks, num_threads>>>(org_uf.parent, d_org_roots, N);
  flattenRoots_kernel<<<blocks, num_threads>>>(decomp_uf.parent, d_decomp_roots,
                                               N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 2: compute sum_i C(a_i, 2) — org cluster sizes
  // Sort org roots, run-length encode to get cluster sizes
  int *d_org_sorted;
  CUDA_CHECK(cudaMalloc(&d_org_sorted, N * sizeof(int)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_org_roots, d_org_sorted,
                                   N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_org_roots, d_org_sorted,
                                   N);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  int *d_org_unique, *d_org_counts, *d_num_org_clusters;
  CUDA_CHECK(cudaMalloc(&d_org_unique, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_org_counts, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_num_org_clusters, sizeof(int)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_org_sorted,
                                       d_org_unique, d_org_counts,
                                       d_num_org_clusters, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_org_sorted,
                                       d_org_unique, d_org_counts,
                                       d_num_org_clusters, N);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  int h_num_org_clusters;
  CUDA_CHECK(cudaMemcpy(&h_num_org_clusters, d_num_org_clusters, sizeof(int),
                         cudaMemcpyDeviceToHost));

  // C(a_i, 2) for each cluster, then sum
  long long *d_org_c2;
  CUDA_CHECK(cudaMalloc(&d_org_c2, h_num_org_clusters * sizeof(long long)));
  int c2_blocks = (h_num_org_clusters + num_threads - 1) / num_threads;
  chooseTwo_kernel<<<c2_blocks, num_threads>>>(d_org_counts, d_org_c2,
                                               h_num_org_clusters);

  long long *d_sum_org_c2;
  CUDA_CHECK(cudaMalloc(&d_sum_org_c2, sizeof(long long)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_org_c2, d_sum_org_c2,
                           h_num_org_clusters);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_org_c2, d_sum_org_c2,
                           h_num_org_clusters);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  long long h_sum_org_c2; // = TP + FN
  CUDA_CHECK(cudaMemcpy(&h_sum_org_c2, d_sum_org_c2, sizeof(long long),
                         cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_org_c2));
  CUDA_CHECK(cudaFree(d_sum_org_c2));
  CUDA_CHECK(cudaFree(d_org_unique));
  CUDA_CHECK(cudaFree(d_org_counts));
  CUDA_CHECK(cudaFree(d_num_org_clusters));
  CUDA_CHECK(cudaFree(d_org_sorted));

  // Step 3: compute sum_j C(b_j, 2) — decomp cluster sizes
  int *d_decomp_sorted;
  CUDA_CHECK(cudaMalloc(&d_decomp_sorted, N * sizeof(int)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_decomp_roots,
                                   d_decomp_sorted, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_decomp_roots,
                                   d_decomp_sorted, N);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  int *d_decomp_unique, *d_decomp_counts, *d_num_decomp_clusters;
  CUDA_CHECK(cudaMalloc(&d_decomp_unique, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_decomp_counts, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_num_decomp_clusters, sizeof(int)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_decomp_sorted,
                                       d_decomp_unique, d_decomp_counts,
                                       d_num_decomp_clusters, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_decomp_sorted,
                                       d_decomp_unique, d_decomp_counts,
                                       d_num_decomp_clusters, N);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  int h_num_decomp_clusters;
  CUDA_CHECK(cudaMemcpy(&h_num_decomp_clusters, d_num_decomp_clusters,
                         sizeof(int), cudaMemcpyDeviceToHost));

  long long *d_decomp_c2;
  CUDA_CHECK(
      cudaMalloc(&d_decomp_c2, h_num_decomp_clusters * sizeof(long long)));
  c2_blocks = (h_num_decomp_clusters + num_threads - 1) / num_threads;
  chooseTwo_kernel<<<c2_blocks, num_threads>>>(d_decomp_counts, d_decomp_c2,
                                               h_num_decomp_clusters);

  long long *d_sum_decomp_c2;
  CUDA_CHECK(cudaMalloc(&d_sum_decomp_c2, sizeof(long long)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_decomp_c2, d_sum_decomp_c2,
                           h_num_decomp_clusters);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_decomp_c2, d_sum_decomp_c2,
                           h_num_decomp_clusters);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  long long h_sum_decomp_c2; // = TP + FP
  CUDA_CHECK(cudaMemcpy(&h_sum_decomp_c2, d_sum_decomp_c2, sizeof(long long),
                         cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_decomp_c2));
  CUDA_CHECK(cudaFree(d_sum_decomp_c2));
  CUDA_CHECK(cudaFree(d_decomp_unique));
  CUDA_CHECK(cudaFree(d_decomp_counts));
  CUDA_CHECK(cudaFree(d_num_decomp_clusters));
  CUDA_CHECK(cudaFree(d_decomp_sorted));

  // Step 4: compute sum_ij C(n_ij, 2) — contingency table entries
  // Encode (org_root, decomp_root) as a 64-bit key, sort, run-length encode
  long long *d_pair_keys;
  CUDA_CHECK(cudaMalloc(&d_pair_keys, N * sizeof(long long)));
  packPairKeys_kernel<<<blocks, num_threads>>>(d_org_roots, d_decomp_roots,
                                               d_pair_keys, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_org_roots));
  CUDA_CHECK(cudaFree(d_decomp_roots));

  // Sort pair keys
  long long *d_pair_keys_sorted;
  CUDA_CHECK(cudaMalloc(&d_pair_keys_sorted, N * sizeof(long long)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_pair_keys,
                                   d_pair_keys_sorted, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_pair_keys,
                                   d_pair_keys_sorted, N);
    CUDA_CHECK(cudaFree(d_tmp));
  }
  CUDA_CHECK(cudaFree(d_pair_keys));

  // Run-length encode to get contingency counts
  long long *d_pair_unique;
  int *d_pair_counts, *d_num_pairs;
  CUDA_CHECK(cudaMalloc(&d_pair_unique, N * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&d_pair_counts, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_num_pairs, sizeof(int)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_pair_keys_sorted,
                                       d_pair_unique, d_pair_counts,
                                       d_num_pairs, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_pair_keys_sorted,
                                       d_pair_unique, d_pair_counts,
                                       d_num_pairs, N);
    CUDA_CHECK(cudaFree(d_tmp));
  }
  CUDA_CHECK(cudaFree(d_pair_keys_sorted));
  CUDA_CHECK(cudaFree(d_pair_unique));

  int h_num_pairs;
  CUDA_CHECK(
      cudaMemcpy(&h_num_pairs, d_num_pairs, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_num_pairs));

  // C(n_ij, 2) for each contingency entry, then sum
  long long *d_pair_c2;
  CUDA_CHECK(cudaMalloc(&d_pair_c2, h_num_pairs * sizeof(long long)));
  c2_blocks = (h_num_pairs + num_threads - 1) / num_threads;
  chooseTwo_kernel<<<c2_blocks, num_threads>>>(d_pair_counts, d_pair_c2,
                                               h_num_pairs);
  CUDA_CHECK(cudaFree(d_pair_counts));

  long long *d_sum_pair_c2;
  CUDA_CHECK(cudaMalloc(&d_sum_pair_c2, sizeof(long long)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_pair_c2, d_sum_pair_c2,
                           h_num_pairs);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_pair_c2, d_sum_pair_c2,
                           h_num_pairs);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  long long h_sum_pair_c2; // = TP
  CUDA_CHECK(cudaMemcpy(&h_sum_pair_c2, d_sum_pair_c2, sizeof(long long),
                         cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_pair_c2));
  CUDA_CHECK(cudaFree(d_sum_pair_c2));

  // Step 5: derive TP, FP, FN, TN
  long long total_pairs = (long long)N * (N - 1) / 2;
  h_tp = h_sum_pair_c2;
  h_fn = h_sum_org_c2 - h_tp;
  h_fp = h_sum_decomp_c2 - h_tp;
  h_tn = total_pairs - h_tp - h_fp - h_fn;
}

// ============================================================================
// ARI CALCULATION FUNCTIONS
// ============================================================================

template <typename T>
void calculateARI2D(const T *d_org_xx, const T *d_org_yy, T *d_decomp_xx,
                    T *d_decomp_yy, T min_x, T range_x, T min_y, T range_y,
                    int N, T b, long long &h_tp, long long &h_tn,
                    long long &h_fp, long long &h_fn) {
  if (N == 0) {
    h_tp = 0;
    h_tn = 0;
    h_fp = 0;
    h_fn = 0;
    return;
  }

  T b_sq = b * b;

  // Partition both datasets into cells
  int grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / b)));
  int grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / b)));
  int num_cells = grid_dim_x * grid_dim_y;

  int *d_org_cell_start, *d_org_cell_pts_sorted;
  particlePartition2D(d_org_xx, d_org_yy, min_x, min_y, b, grid_dim_x,
                      grid_dim_y, N, &d_org_cell_start, &d_org_cell_pts_sorted);

  T min_x_decomp, max_x_decomp, range_x_decomp, min_y_decomp, max_y_decomp,
      range_y_decomp;
  getRange(d_decomp_xx, N, min_x_decomp, max_x_decomp, range_x_decomp);
  getRange(d_decomp_yy, N, min_y_decomp, max_y_decomp, range_y_decomp);

  int grid_dim_x_decomp =
      std::max(1, static_cast<int>(std::ceil(range_x_decomp / b)));
  int grid_dim_y_decomp =
      std::max(1, static_cast<int>(std::ceil(range_y_decomp / b)));

  int *d_decomp_cell_start, *d_decomp_cell_pts_sorted;
  particlePartition2D(d_decomp_xx, d_decomp_yy, min_x_decomp, min_y_decomp, b,
                      grid_dim_x_decomp, grid_dim_y_decomp, N,
                      &d_decomp_cell_start, &d_decomp_cell_pts_sorted);

  // Build halos using union-find
  UnionFind org_uf = createUnionFind(N);
  UnionFind decomp_uf = createUnionFind(N);
  int init_blocks = (N + num_threads - 1) / num_threads;
  initUnionFind_kernel<<<init_blocks, num_threads>>>(org_uf.parent, org_uf.rank,
                                                     N);
  initUnionFind_kernel<<<init_blocks, num_threads>>>(decomp_uf.parent,
                                                     decomp_uf.rank, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  int cell_blocks = (num_cells + num_threads - 1) / num_threads;
  buildHalos2D_kernel<T><<<cell_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_org_cell_start, d_org_cell_pts_sorted, num_cells, N,
      min_x, min_y, b, grid_dim_x, grid_dim_y, b_sq, org_uf.parent,
      org_uf.rank);

  int decomp_num_cells = grid_dim_x_decomp * grid_dim_y_decomp;
  int decomp_cell_blocks = (decomp_num_cells + num_threads - 1) / num_threads;
  buildHalos2D_kernel<T><<<decomp_cell_blocks, num_threads>>>(
      d_decomp_xx, d_decomp_yy, d_decomp_cell_start, d_decomp_cell_pts_sorted,
      decomp_num_cells, N, min_x_decomp, min_y_decomp, b, grid_dim_x_decomp,
      grid_dim_y_decomp, b_sq, decomp_uf.parent, decomp_uf.rank);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compute TP/TN/FP/FN via contingency table
  computeTPTNFPFN(org_uf, decomp_uf, N, h_tp, h_tn, h_fp, h_fn);

  // Cleanup
  CUDA_CHECK(cudaFree(d_org_cell_start));
  CUDA_CHECK(cudaFree(d_org_cell_pts_sorted));
  CUDA_CHECK(cudaFree(d_decomp_cell_start));
  CUDA_CHECK(cudaFree(d_decomp_cell_pts_sorted));
  destroyUnionFind(org_uf);
  destroyUnionFind(decomp_uf);
}

template <typename T>
void calculateARI3D(const T *d_org_xx, const T *d_org_yy, const T *d_org_zz,
                    T *d_decomp_xx, T *d_decomp_yy, T *d_decomp_zz, T min_x,
                    T range_x, T min_y, T range_y, T min_z, T range_z, int N,
                    T b, long long &h_tp, long long &h_tn, long long &h_fp,
                    long long &h_fn) {
  if (N == 0) {
    h_tp = 0;
    h_tn = 0;
    h_fp = 0;
    h_fn = 0;
    return;
  }

  T b_sq = b * b;

  // Partition both datasets into cells
  int grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / b)));
  int grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / b)));
  int grid_dim_z = std::max(1, static_cast<int>(std::ceil(range_z / b)));
  int num_cells = grid_dim_x * grid_dim_y * grid_dim_z;

  int *d_org_cell_start, *d_org_cell_pts_sorted;
  particlePartition3D(d_org_xx, d_org_yy, d_org_zz, min_x, min_y, min_z, b,
                      grid_dim_x, grid_dim_y, grid_dim_z, N, &d_org_cell_start,
                      &d_org_cell_pts_sorted);

  T min_x_decomp, min_y_decomp, min_z_decomp, max_x_decomp, max_y_decomp,
      max_z_decomp, range_x_decomp, range_y_decomp, range_z_decomp;
  getRange(d_decomp_xx, N, min_x_decomp, max_x_decomp, range_x_decomp);
  getRange(d_decomp_yy, N, min_y_decomp, max_y_decomp, range_y_decomp);
  getRange(d_decomp_zz, N, min_z_decomp, max_z_decomp, range_z_decomp);

  int grid_dim_x_decomp =
      std::max(1, static_cast<int>(std::ceil(range_x_decomp / b)));
  int grid_dim_y_decomp =
      std::max(1, static_cast<int>(std::ceil(range_y_decomp / b)));
  int grid_dim_z_decomp =
      std::max(1, static_cast<int>(std::ceil(range_z_decomp / b)));

  int *d_decomp_cell_start, *d_decomp_cell_pts_sorted;
  particlePartition3D(d_decomp_xx, d_decomp_yy, d_decomp_zz, min_x_decomp,
                      min_y_decomp, min_z_decomp, b, grid_dim_x_decomp,
                      grid_dim_y_decomp, grid_dim_z_decomp, N,
                      &d_decomp_cell_start, &d_decomp_cell_pts_sorted);

  // Build halos using union-find
  UnionFind org_uf = createUnionFind(N);
  UnionFind decomp_uf = createUnionFind(N);
  int init_blocks = (N + num_threads - 1) / num_threads;
  initUnionFind_kernel<<<init_blocks, num_threads>>>(org_uf.parent, org_uf.rank,
                                                     N);
  initUnionFind_kernel<<<init_blocks, num_threads>>>(decomp_uf.parent,
                                                     decomp_uf.rank, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  int cell_blocks = (num_cells + num_threads - 1) / num_threads;
  buildHalos3D_kernel<T><<<cell_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, d_org_cell_start, d_org_cell_pts_sorted,
      num_cells, N, min_x, min_y, min_z, b, grid_dim_x, grid_dim_y, grid_dim_z,
      b_sq, org_uf.parent, org_uf.rank);

  int decomp_num_cells =
      grid_dim_x_decomp * grid_dim_y_decomp * grid_dim_z_decomp;
  int decomp_cell_blocks = (decomp_num_cells + num_threads - 1) / num_threads;
  buildHalos3D_kernel<T><<<decomp_cell_blocks, num_threads>>>(
      d_decomp_xx, d_decomp_yy, d_decomp_zz, d_decomp_cell_start,
      d_decomp_cell_pts_sorted, decomp_num_cells, N, min_x_decomp, min_y_decomp,
      min_z_decomp, b, grid_dim_x_decomp, grid_dim_y_decomp, grid_dim_z_decomp,
      b_sq, decomp_uf.parent, decomp_uf.rank);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Compute TP/TN/FP/FN via contingency table
  computeTPTNFPFN(org_uf, decomp_uf, N, h_tp, h_tn, h_fp, h_fn);

  // Cleanup
  CUDA_CHECK(cudaFree(d_org_cell_start));
  CUDA_CHECK(cudaFree(d_org_cell_pts_sorted));
  CUDA_CHECK(cudaFree(d_decomp_cell_start));
  CUDA_CHECK(cudaFree(d_decomp_cell_pts_sorted));
  destroyUnionFind(org_uf);
  destroyUnionFind(decomp_uf);
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ============================================================================

template void calculateARI2D<float>(const float *, const float *, float *,
                                    float *, float, float, float, float, int,
                                    float, long long &, long long &,
                                    long long &, long long &);
template void calculateARI2D<double>(const double *, const double *, double *,
                                     double *, double, double, double, double,
                                     int, double, long long &, long long &,
                                     long long &, long long &);

template void calculateARI3D<float>(const float *, const float *, const float *,
                                    float *, float *, float *, float, float,
                                    float, float, float, float, int, float,
                                    long long &, long long &, long long &,
                                    long long &);
template void calculateARI3D<double>(const double *, const double *,
                                     const double *, double *, double *,
                                     double *, double, double, double, double,
                                     double, double, int, double, long long &,
                                     long long &, long long &, long long &);

template __global__ void buildHalos2D_kernel<float>(const float *,
                                                    const float *, const int *,
                                                    const int *, int, int,
                                                    float, float, float, int,
                                                    int, float, int *, int *);
template __global__ void
buildHalos2D_kernel<double>(const double *, const double *, const int *,
                            const int *, int, int, double, double, double, int,
                            int, double, int *, int *);

template __global__ void
buildHalos3D_kernel<float>(const float *, const float *, const float *,
                           const int *, const int *, int, int, float, float,
                           float, float, int, int, int, float, int *, int *);
template __global__ void
buildHalos3D_kernel<double>(const double *, const double *, const double *,
                            const int *, const int *, int, int, double, double,
                            double, double, int, int, int, double, int *,
                            int *);
