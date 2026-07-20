#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

// =============================================================================
// CPU Union-Find with path compression and union by rank
// =============================================================================

struct CPUUnionFind {
  std::vector<int> parent;
  std::vector<int> rnk;

  CPUUnionFind() = default;
  explicit CPUUnionFind(size_t N) : parent(N), rnk(N, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]]; // path compression
      x = parent[x];
    }
    return x;
  }

  void unite(int x, int y) {
    x = find(x);
    y = find(y);
    if (x == y)
      return;
    if (rnk[x] < rnk[y])
      std::swap(x, y);
    parent[y] = x;
    if (rnk[x] == rnk[y])
      rnk[x]++;
  }
};

// =============================================================================
// CPU FOF 3D — sparse cell map, 13 forward neighbors
// =============================================================================

template <typename T>
CPUUnionFind cpuFOF3D(const T *xx, const T *yy, const T *zz, size_t N, T b) {
  CPUUnionFind uf(N);
  if (N == 0)
    return uf;

  T b_sq = b * b;

  // Compute ranges
  T min_x = xx[0], max_x = xx[0];
  T min_y = yy[0], max_y = yy[0];
  T min_z = zz[0], max_z = zz[0];
  for (size_t i = 1; i < N; i++) {
    if (xx[i] < min_x) min_x = xx[i];
    if (xx[i] > max_x) max_x = xx[i];
    if (yy[i] < min_y) min_y = yy[i];
    if (yy[i] > max_y) max_y = yy[i];
    if (zz[i] < min_z) min_z = zz[i];
    if (zz[i] > max_z) max_z = zz[i];
  }
  T range_x = max_x - min_x;
  T range_y = max_y - min_y;
  T range_z = max_z - min_z;

  int grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / b)));
  int grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / b)));
  int grid_dim_z = std::max(1, static_cast<int>(std::ceil(range_z / b)));
  size_t dim_xy = static_cast<size_t>(grid_dim_x) * grid_dim_y;

  // Sparse cell map
  std::unordered_map<size_t, std::vector<size_t>> cell_map;
  for (size_t i = 0; i < N; i++) {
    size_t cx = static_cast<size_t>(std::floor((xx[i] - min_x) / b));
    size_t cy = static_cast<size_t>(std::floor((yy[i] - min_y) / b));
    size_t cz = static_cast<size_t>(std::floor((zz[i] - min_z) / b));
    cx = std::min(cx, static_cast<size_t>(grid_dim_x - 1));
    cy = std::min(cy, static_cast<size_t>(grid_dim_y - 1));
    cz = std::min(cz, static_cast<size_t>(grid_dim_z - 1));
    cell_map[cz * dim_xy + cy * grid_dim_x + cx].push_back(i);
  }

  // 13 forward neighbors to avoid duplicate pair checks
  static const int offsets[13][3] = {
      {1, 0, 0},  {0, 1, 0},  {0, 0, 1},  {1, 1, 0},  {1, -1, 0},
      {1, 0, 1},  {1, 0, -1}, {0, 1, 1},  {0, 1, -1}, {1, 1, 1},
      {1, 1, -1}, {1, -1, 1}, {1, -1, -1}};

  for (auto &kv : cell_map) {
    size_t cid = kv.first;
    auto &indices = kv.second;
    int id_x = cid % grid_dim_x;
    int id_y = (cid / grid_dim_x) % grid_dim_y;
    int id_z = cid / dim_xy;

    // Within-cell pairs
    for (size_t a = 0; a < indices.size(); a++) {
      size_t pi = indices[a];
      for (size_t bb = a + 1; bb < indices.size(); bb++) {
        size_t pj = indices[bb];
        T dx = xx[pi] - xx[pj];
        T dy = yy[pi] - yy[pj];
        T dz = zz[pi] - zz[pj];
        if (dx * dx + dy * dy + dz * dz <= b_sq)
          uf.unite(static_cast<int>(pi), static_cast<int>(pj));
      }
    }

    // Forward neighbor pairs
    for (int ni = 0; ni < 13; ni++) {
      int nx = id_x + offsets[ni][0];
      int ny = id_y + offsets[ni][1];
      int nz = id_z + offsets[ni][2];
      if (nx < 0 || nx >= grid_dim_x || ny < 0 || ny >= grid_dim_y ||
          nz < 0 || nz >= grid_dim_z)
        continue;
      size_t ncid =
          static_cast<size_t>(nz) * dim_xy + static_cast<size_t>(ny) * grid_dim_x + nx;
      auto nit = cell_map.find(ncid);
      if (nit == cell_map.end())
        continue;
      const auto &nindices = nit->second;
      for (size_t pi : indices) {
        for (size_t pj : nindices) {
          T dx = xx[pi] - xx[pj];
          T dy = yy[pi] - yy[pj];
          T dz = zz[pi] - zz[pj];
          if (dx * dx + dy * dy + dz * dz <= b_sq)
            uf.unite(static_cast<int>(pi), static_cast<int>(pj));
        }
      }
    }
  }

  return uf;
}

// =============================================================================
// CPU FOF 2D — sparse cell map, 4 forward neighbors
// =============================================================================

template <typename T>
CPUUnionFind cpuFOF2D(const T *xx, const T *yy, size_t N, T b) {
  CPUUnionFind uf(N);
  if (N == 0)
    return uf;

  T b_sq = b * b;

  T min_x = xx[0], max_x = xx[0];
  T min_y = yy[0], max_y = yy[0];
  for (size_t i = 1; i < N; i++) {
    if (xx[i] < min_x) min_x = xx[i];
    if (xx[i] > max_x) max_x = xx[i];
    if (yy[i] < min_y) min_y = yy[i];
    if (yy[i] > max_y) max_y = yy[i];
  }
  T range_x = max_x - min_x;
  T range_y = max_y - min_y;

  int grid_dim_x = std::max(1, static_cast<int>(std::ceil(range_x / b)));
  int grid_dim_y = std::max(1, static_cast<int>(std::ceil(range_y / b)));

  std::unordered_map<size_t, std::vector<size_t>> cell_map;
  for (size_t i = 0; i < N; i++) {
    size_t cx = static_cast<size_t>(std::floor((xx[i] - min_x) / b));
    size_t cy = static_cast<size_t>(std::floor((yy[i] - min_y) / b));
    cx = std::min(cx, static_cast<size_t>(grid_dim_x - 1));
    cy = std::min(cy, static_cast<size_t>(grid_dim_y - 1));
    cell_map[cy * grid_dim_x + cx].push_back(i);
  }

  static const int offsets[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

  for (auto &kv : cell_map) {
    size_t cid = kv.first;
    auto &indices = kv.second;
    int id_x = cid % grid_dim_x;
    int id_y = cid / grid_dim_x;

    for (size_t a = 0; a < indices.size(); a++) {
      size_t pi = indices[a];
      for (size_t bb = a + 1; bb < indices.size(); bb++) {
        size_t pj = indices[bb];
        T dx = xx[pi] - xx[pj];
        T dy = yy[pi] - yy[pj];
        if (dx * dx + dy * dy <= b_sq)
          uf.unite(static_cast<int>(pi), static_cast<int>(pj));
      }
    }

    for (int ni = 0; ni < 4; ni++) {
      int nx = id_x + offsets[ni][0];
      int ny = id_y + offsets[ni][1];
      if (nx < 0 || nx >= grid_dim_x || ny < 0 || ny >= grid_dim_y)
        continue;
      size_t ncid = static_cast<size_t>(ny) * grid_dim_x + nx;
      auto nit = cell_map.find(ncid);
      if (nit == cell_map.end())
        continue;
      const auto &nindices = nit->second;
      for (size_t pi : indices) {
        for (size_t pj : nindices) {
          T dx = xx[pi] - xx[pj];
          T dy = yy[pi] - yy[pj];
          if (dx * dx + dy * dy <= b_sq)
            uf.unite(static_cast<int>(pi), static_cast<int>(pj));
        }
      }
    }
  }

  return uf;
}

// =============================================================================
// Compute TP/TN/FP/FN from two union-finds (contingency table method)
// Only counts pairs among the first N_local particles.
// =============================================================================

inline void computeLocalTPTNFPFN(CPUUnionFind &org_uf, CPUUnionFind &decomp_uf,
                                 size_t N_local, long long &tp, long long &tn,
                                 long long &fp, long long &fn) {
  // Flatten roots for first N_local particles
  std::vector<int> org_roots(N_local), decomp_roots(N_local);
  for (size_t i = 0; i < N_local; i++) {
    org_roots[i] = org_uf.find(static_cast<int>(i));
    decomp_roots[i] = decomp_uf.find(static_cast<int>(i));
  }

  // Count local cluster sizes and contingency table entries
  std::unordered_map<int, long long> org_counts, decomp_counts;
  std::unordered_map<long long, long long> pair_counts;

  for (size_t i = 0; i < N_local; i++) {
    org_counts[org_roots[i]]++;
    decomp_counts[decomp_roots[i]]++;
    // Pack (org_root, decomp_root) into 64-bit key
    long long key = (static_cast<long long>(static_cast<unsigned int>(org_roots[i])) << 32) |
                    static_cast<unsigned int>(decomp_roots[i]);
    pair_counts[key]++;
  }

  // sum_i C(a_i, 2) for org clusters
  long long sum_org_c2 = 0;
  for (auto &kv : org_counts)
    sum_org_c2 += kv.second * (kv.second - 1) / 2;

  // sum_j C(b_j, 2) for decomp clusters
  long long sum_decomp_c2 = 0;
  for (auto &kv : decomp_counts)
    sum_decomp_c2 += kv.second * (kv.second - 1) / 2;

  // sum_ij C(n_ij, 2) for contingency entries
  long long sum_pair_c2 = 0;
  for (auto &kv : pair_counts)
    sum_pair_c2 += kv.second * (kv.second - 1) / 2;

  long long total_pairs = static_cast<long long>(N_local) * (N_local - 1) / 2;
  tp = sum_pair_c2;
  fn = sum_org_c2 - tp;
  fp = sum_decomp_c2 - tp;
  tn = total_pairs - tp - fp - fn;
}

// =============================================================================
// Distributed ARI 3D
//
// #4: Optionally uses subsampling when N_ext exceeds a threshold to reduce
// FOF cost from O(N²) to O(sample²). Falls back to exact when N_ext is small.
// =============================================================================

static constexpr size_t ARI_SAMPLE_THRESHOLD = 500000;
static constexpr size_t ARI_SAMPLE_SIZE = 100000;

template <typename T>
void distributedARI3D(const T *ext_org_xx, const T *ext_org_yy,
                      const T *ext_org_zz, const T *ext_decomp_xx,
                      const T *ext_decomp_yy, const T *ext_decomp_zz,
                      size_t N_local, size_t N_ext, T b, int mpi_rank,
                      int mpi_size, MPI_Comm comm) {

  // #4: Subsample when dataset is large
  bool use_sampling = (N_ext > ARI_SAMPLE_THRESHOLD && N_local > ARI_SAMPLE_SIZE);
  size_t eff_N_local = N_local;
  size_t eff_N_ext = N_ext;

  // Temporary arrays for sampled data
  std::vector<T> s_org_xx, s_org_yy, s_org_zz;
  std::vector<T> s_dec_xx, s_dec_yy, s_dec_zz;
  std::vector<size_t> sample_map; // sampled index -> original index

  const T *fof_org_xx = ext_org_xx;
  const T *fof_org_yy = ext_org_yy;
  const T *fof_org_zz = ext_org_zz;
  const T *fof_dec_xx = ext_decomp_xx;
  const T *fof_dec_yy = ext_decomp_yy;
  const T *fof_dec_zz = ext_decomp_zz;

  if (use_sampling) {
    // Subsample local particles + keep all ghost particles
    size_t sample_local = std::min(ARI_SAMPLE_SIZE, N_local);
    size_t ghost_count = N_ext - N_local;
    eff_N_local = sample_local;
    eff_N_ext = sample_local + ghost_count;

    // Random sample of local indices
    sample_map.resize(N_local);
    std::iota(sample_map.begin(), sample_map.end(), 0);
    std::mt19937 rng(42 + mpi_rank);
    std::shuffle(sample_map.begin(), sample_map.end(), rng);
    sample_map.resize(sample_local);
    std::sort(sample_map.begin(), sample_map.end());

    s_org_xx.resize(eff_N_ext); s_org_yy.resize(eff_N_ext); s_org_zz.resize(eff_N_ext);
    s_dec_xx.resize(eff_N_ext); s_dec_yy.resize(eff_N_ext); s_dec_zz.resize(eff_N_ext);

    // Copy sampled local particles
    for (size_t i = 0; i < sample_local; ++i) {
      size_t idx = sample_map[i];
      s_org_xx[i] = ext_org_xx[idx]; s_org_yy[i] = ext_org_yy[idx]; s_org_zz[i] = ext_org_zz[idx];
      s_dec_xx[i] = ext_decomp_xx[idx]; s_dec_yy[i] = ext_decomp_yy[idx]; s_dec_zz[i] = ext_decomp_zz[idx];
    }
    // Copy all ghost particles
    std::memcpy(s_org_xx.data() + sample_local, ext_org_xx + N_local, ghost_count * sizeof(T));
    std::memcpy(s_org_yy.data() + sample_local, ext_org_yy + N_local, ghost_count * sizeof(T));
    std::memcpy(s_org_zz.data() + sample_local, ext_org_zz + N_local, ghost_count * sizeof(T));
    std::memcpy(s_dec_xx.data() + sample_local, ext_decomp_xx + N_local, ghost_count * sizeof(T));
    std::memcpy(s_dec_yy.data() + sample_local, ext_decomp_yy + N_local, ghost_count * sizeof(T));
    std::memcpy(s_dec_zz.data() + sample_local, ext_decomp_zz + N_local, ghost_count * sizeof(T));

    fof_org_xx = s_org_xx.data(); fof_org_yy = s_org_yy.data(); fof_org_zz = s_org_zz.data();
    fof_dec_xx = s_dec_xx.data(); fof_dec_yy = s_dec_yy.data(); fof_dec_zz = s_dec_zz.data();

    printf("Rank %d: ARI sampling %zu/%zu local particles + %zu ghosts\n",
           mpi_rank, sample_local, N_local, ghost_count);
  }

  printf("Rank %d: running FOF on original extended arrays (N_ext=%zu)...\n",
         mpi_rank, eff_N_ext);

  CPUUnionFind org_uf = cpuFOF3D(fof_org_xx, fof_org_yy, fof_org_zz, eff_N_ext, b);

  printf("Rank %d: running FOF on decompressed extended arrays...\n", mpi_rank);

  CPUUnionFind decomp_uf = cpuFOF3D(fof_dec_xx, fof_dec_yy, fof_dec_zz, eff_N_ext, b);

  long long tp, tn, fp, fn;
  computeLocalTPTNFPFN(org_uf, decomp_uf, eff_N_local, tp, tn, fp, fn);

  printf("Rank %d: local TP=%lld TN=%lld FP=%lld FN=%lld%s\n", mpi_rank, tp, tn,
         fp, fn, use_sampling ? " (sampled)" : "");

  long long global_tp, global_tn, global_fp, global_fn;
  MPI_Reduce(&tp, &global_tp, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&tn, &global_tn, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&fp, &global_fp, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&fn, &global_fn, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

  if (mpi_rank == 0) {
    double num = 2.0 * (static_cast<double>(global_tp) * global_tn -
                         static_cast<double>(global_fp) * global_fn);
    double den =
        static_cast<double>(global_tp + global_fp) * (global_fp + global_tn) +
        static_cast<double>(global_tp + global_fn) * (global_fn + global_tn);
    double ari = (den == 0.0) ? 1.0 : num / den;
    printf("ARI%s: %.10f (TP=%lld, TN=%lld, FP=%lld, FN=%lld)\n",
           use_sampling ? " (approx)" : "", ari, global_tp,
           global_tn, global_fp, global_fn);
  }
}

// =============================================================================
// Distributed ARI 2D (#4: with optional subsampling)
// =============================================================================

template <typename T>
void distributedARI2D(const T *ext_org_xx, const T *ext_org_yy,
                      const T *ext_decomp_xx, const T *ext_decomp_yy,
                      size_t N_local, size_t N_ext, T b, int mpi_rank,
                      int mpi_size, MPI_Comm comm) {

  bool use_sampling = (N_ext > ARI_SAMPLE_THRESHOLD && N_local > ARI_SAMPLE_SIZE);
  size_t eff_N_local = N_local;
  size_t eff_N_ext = N_ext;

  std::vector<T> s_org_xx, s_org_yy;
  std::vector<T> s_dec_xx, s_dec_yy;

  const T *fof_org_xx = ext_org_xx;
  const T *fof_org_yy = ext_org_yy;
  const T *fof_dec_xx = ext_decomp_xx;
  const T *fof_dec_yy = ext_decomp_yy;

  if (use_sampling) {
    size_t sample_local = std::min(ARI_SAMPLE_SIZE, N_local);
    size_t ghost_count = N_ext - N_local;
    eff_N_local = sample_local;
    eff_N_ext = sample_local + ghost_count;

    std::vector<size_t> sample_map(N_local);
    std::iota(sample_map.begin(), sample_map.end(), 0);
    std::mt19937 rng(42 + mpi_rank);
    std::shuffle(sample_map.begin(), sample_map.end(), rng);
    sample_map.resize(sample_local);
    std::sort(sample_map.begin(), sample_map.end());

    s_org_xx.resize(eff_N_ext); s_org_yy.resize(eff_N_ext);
    s_dec_xx.resize(eff_N_ext); s_dec_yy.resize(eff_N_ext);

    for (size_t i = 0; i < sample_local; ++i) {
      size_t idx = sample_map[i];
      s_org_xx[i] = ext_org_xx[idx]; s_org_yy[i] = ext_org_yy[idx];
      s_dec_xx[i] = ext_decomp_xx[idx]; s_dec_yy[i] = ext_decomp_yy[idx];
    }
    std::memcpy(s_org_xx.data() + sample_local, ext_org_xx + N_local, ghost_count * sizeof(T));
    std::memcpy(s_org_yy.data() + sample_local, ext_org_yy + N_local, ghost_count * sizeof(T));
    std::memcpy(s_dec_xx.data() + sample_local, ext_decomp_xx + N_local, ghost_count * sizeof(T));
    std::memcpy(s_dec_yy.data() + sample_local, ext_decomp_yy + N_local, ghost_count * sizeof(T));

    fof_org_xx = s_org_xx.data(); fof_org_yy = s_org_yy.data();
    fof_dec_xx = s_dec_xx.data(); fof_dec_yy = s_dec_yy.data();

    printf("Rank %d: ARI sampling %zu/%zu local particles + %zu ghosts\n",
           mpi_rank, sample_local, N_local, ghost_count);
  }

  printf("Rank %d: running FOF on original extended arrays (N_ext=%zu)...\n",
         mpi_rank, eff_N_ext);

  CPUUnionFind org_uf = cpuFOF2D(fof_org_xx, fof_org_yy, eff_N_ext, b);

  printf("Rank %d: running FOF on decompressed extended arrays...\n", mpi_rank);

  CPUUnionFind decomp_uf = cpuFOF2D(fof_dec_xx, fof_dec_yy, eff_N_ext, b);

  long long tp, tn, fp, fn;
  computeLocalTPTNFPFN(org_uf, decomp_uf, eff_N_local, tp, tn, fp, fn);

  printf("Rank %d: local TP=%lld TN=%lld FP=%lld FN=%lld%s\n", mpi_rank, tp, tn,
         fp, fn, use_sampling ? " (sampled)" : "");

  long long global_tp, global_tn, global_fp, global_fn;
  MPI_Reduce(&tp, &global_tp, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&tn, &global_tn, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&fp, &global_fp, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
  MPI_Reduce(&fn, &global_fn, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

  if (mpi_rank == 0) {
    double num = 2.0 * (static_cast<double>(global_tp) * global_tn -
                         static_cast<double>(global_fp) * global_fn);
    double den =
        static_cast<double>(global_tp + global_fp) * (global_fp + global_tn) +
        static_cast<double>(global_tp + global_fn) * (global_fn + global_tn);
    double ari = (den == 0.0) ? 1.0 : num / den;
    printf("ARI%s: %.10f (TP=%lld, TN=%lld, FP=%lld, FN=%lld)\n",
           use_sampling ? " (approx)" : "", ari, global_tp,
           global_tn, global_fp, global_fn);
  }
}
