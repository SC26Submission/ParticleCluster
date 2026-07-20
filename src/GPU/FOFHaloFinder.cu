#include "FOFHaloFinder.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

extern double target_cell_occupancy;

namespace {

static int maxCellsForTargetOccupancy(int N) {
  if (target_cell_occupancy <= 0)
    return std::numeric_limits<int>::max();

  double occupancy = target_cell_occupancy;
  double max_cells_f = std::ceil(static_cast<double>(N) / occupancy);
  if (max_cells_f > static_cast<double>(std::numeric_limits<int>::max()))
    return std::numeric_limits<int>::max();
  long long max_cells = static_cast<long long>(max_cells_f);
  if (max_cells > static_cast<long long>(std::numeric_limits<int>::max()))
    return std::numeric_limits<int>::max();
  return std::max(1, static_cast<int>(max_cells));
}

template <typename T> static int gridDimForRange(T range, T grid_len) {
  if (range <= T(0))
    return 1;
  return std::max(1, static_cast<int>(std::ceil(range / grid_len)));
}

template <typename T>
static long long cellCount2D(T range_x, T range_y, T grid_len, int &dim_x,
                             int &dim_y) {
  dim_x = gridDimForRange(range_x, grid_len);
  dim_y = gridDimForRange(range_y, grid_len);
  return static_cast<long long>(dim_x) * dim_y;
}

template <typename T>
static long long cellCount3D(T range_x, T range_y, T range_z, T grid_len,
                             int &dim_x, int &dim_y, int &dim_z) {
  dim_x = gridDimForRange(range_x, grid_len);
  dim_y = gridDimForRange(range_y, grid_len);
  dim_z = gridDimForRange(range_z, grid_len);
  return static_cast<long long>(dim_x) * dim_y * dim_z;
}

template <typename T>
static void coarsenFoFGrid2D(T &grid_len, int &grid_dim_x, int &grid_dim_y,
                             T range_x, T range_y, int max_cells) {
  long long nc = static_cast<long long>(grid_dim_x) * grid_dim_y;
  if (nc <= max_cells)
    return;

  T original_grid_len = grid_len;
  T scale = std::sqrt(static_cast<T>(nc) / static_cast<T>(max_cells));
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
  std::printf("FoF grid coarsened: %lld -> %lld cells (grid_len %e, factor "
              "%.2fx)\n",
              nc, nc_new, static_cast<double>(grid_len),
              static_cast<double>(grid_len / original_grid_len));
}

template <typename T>
static void coarsenFoFGrid3D(T &grid_len, int &grid_dim_x, int &grid_dim_y,
                             int &grid_dim_z, T range_x, T range_y, T range_z,
                             int max_cells) {
  long long nc = static_cast<long long>(grid_dim_x) * grid_dim_y * grid_dim_z;
  if (nc <= max_cells)
    return;

  T original_grid_len = grid_len;
  T scale = std::cbrt(static_cast<T>(nc) / static_cast<T>(max_cells));
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
  std::printf("FoF grid coarsened: %lld -> %lld cells (grid_len %e, factor "
              "%.2fx)\n",
              nc, nc_new, static_cast<double>(grid_len),
              static_cast<double>(grid_len / original_grid_len));
}

static int checkedCellCount(long long cell_count, const char *context) {
  if (cell_count <= 0 ||
      cell_count > static_cast<long long>(std::numeric_limits<int>::max())) {
    std::fprintf(stderr, "%s cell count %lld exceeds int range\n", context,
                 cell_count);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(cell_count);
}

} // namespace

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

template <typename T>
void computeFoFRoots2D(const T *d_xx, const T *d_yy, T min_x, T range_x,
                       T min_y, T range_y, int N, T b, int **d_roots_out) {
  *d_roots_out = nullptr;
  if (N == 0)
    return;

  T b_sq = b * b;
  if (b <= T(0)) {
    std::fprintf(stderr, "FoF linking length must be positive\n");
    std::exit(EXIT_FAILURE);
  }
  T grid_len = b;
  int grid_dim_x, grid_dim_y;
  long long num_cells_ll =
      cellCount2D(range_x, range_y, grid_len, grid_dim_x, grid_dim_y);
  int max_cells = maxCellsForTargetOccupancy(N);
  coarsenFoFGrid2D(grid_len, grid_dim_x, grid_dim_y, range_x, range_y,
                   max_cells);
  num_cells_ll = static_cast<long long>(grid_dim_x) * grid_dim_y;
  int num_cells = checkedCellCount(num_cells_ll, "computeFoFRoots2D");

  int *d_cell_start = nullptr;
  int *d_cell_pts_sorted = nullptr;
  particlePartition2D(d_xx, d_yy, min_x, min_y, grid_len, grid_dim_x,
                      grid_dim_y, N, &d_cell_start, &d_cell_pts_sorted);

  UnionFind uf = createUnionFind(N);
  int init_blocks = (N + num_threads - 1) / num_threads;
  initUnionFind_kernel<<<init_blocks, num_threads>>>(uf.parent, uf.rank, N);

  int cell_blocks = (num_cells + num_threads - 1) / num_threads;
  buildHalos2D_kernel<T><<<cell_blocks, num_threads>>>(
      d_xx, d_yy, d_cell_start, d_cell_pts_sorted, num_cells, N, min_x, min_y,
      b, grid_dim_x, grid_dim_y, b_sq, uf.parent, uf.rank);

  CUDA_CHECK(cudaMalloc(d_roots_out, N * sizeof(int)));
  flattenRoots_kernel<<<init_blocks, num_threads>>>(uf.parent, *d_roots_out, N);

  CUDA_CHECK(cudaFree(d_cell_start));
  CUDA_CHECK(cudaFree(d_cell_pts_sorted));
  destroyUnionFind(uf);
}

template <typename T>
void computeFoFRoots3D(const T *d_xx, const T *d_yy, const T *d_zz, T min_x,
                       T range_x, T min_y, T range_y, T min_z, T range_z, int N,
                       T b, int **d_roots_out) {
  *d_roots_out = nullptr;
  if (N == 0)
    return;

  T b_sq = b * b;
  if (b <= T(0)) {
    std::fprintf(stderr, "FoF linking length must be positive\n");
    std::exit(EXIT_FAILURE);
  }
  T grid_len = b;
  int grid_dim_x, grid_dim_y, grid_dim_z;
  long long num_cells_ll = cellCount3D(range_x, range_y, range_z, grid_len,
                                       grid_dim_x, grid_dim_y, grid_dim_z);
  int max_cells = maxCellsForTargetOccupancy(N);
  coarsenFoFGrid3D(grid_len, grid_dim_x, grid_dim_y, grid_dim_z, range_x,
                   range_y, range_z, max_cells);
  num_cells_ll = static_cast<long long>(grid_dim_x) * grid_dim_y * grid_dim_z;
  int num_cells = checkedCellCount(num_cells_ll, "computeFoFRoots3D");

  int *d_cell_start = nullptr;
  int *d_cell_pts_sorted = nullptr;
  particlePartition3D(d_xx, d_yy, d_zz, min_x, min_y, min_z, grid_len,
                      grid_dim_x, grid_dim_y, grid_dim_z, N, &d_cell_start,
                      &d_cell_pts_sorted);

  UnionFind uf = createUnionFind(N);
  int init_blocks = (N + num_threads - 1) / num_threads;
  initUnionFind_kernel<<<init_blocks, num_threads>>>(uf.parent, uf.rank, N);

  int cell_blocks = (num_cells + num_threads - 1) / num_threads;
  buildHalos3D_kernel<T><<<cell_blocks, num_threads>>>(
      d_xx, d_yy, d_zz, d_cell_start, d_cell_pts_sorted, num_cells, N, min_x,
      min_y, min_z, b, grid_dim_x, grid_dim_y, grid_dim_z, b_sq, uf.parent,
      uf.rank);

  CUDA_CHECK(cudaMalloc(d_roots_out, N * sizeof(int)));
  flattenRoots_kernel<<<init_blocks, num_threads>>>(uf.parent, *d_roots_out, N);

  CUDA_CHECK(cudaFree(d_cell_start));
  CUDA_CHECK(cudaFree(d_cell_pts_sorted));
  destroyUnionFind(uf);
}

// Explicit template instantiations
template void computeFoFRoots2D<float>(const float *, const float *, float,
                                       float, float, float, int, float, int **);
template void computeFoFRoots2D<double>(const double *, const double *, double,
                                        double, double, double, int, double,
                                        int **);

template void computeFoFRoots3D<float>(const float *, const float *,
                                       const float *, float, float, float,
                                       float, float, float, int, float, int **);
template void computeFoFRoots3D<double>(const double *, const double *,
                                        const double *, double, double, double,
                                        double, double, double, int, double,
                                        int **);

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
