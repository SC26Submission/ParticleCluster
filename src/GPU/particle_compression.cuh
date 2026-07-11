#pragma once

#include "HuffmanZSTDCoder.cuh"
#include "util.cuh"
#include <climits>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#define FOFPZ_MPI_COMM_DEFAULT = MPI_COMM_NULL
#else
typedef int MPI_Comm;
static const MPI_Comm MPI_COMM_NULL = 0;
#define FOFPZ_MPI_COMM_DEFAULT = 0
#endif

constexpr int max_num_local_pair = 32;
constexpr int MAX_CELL_PTS = 8192; // Max particles per cell for GPU compression

__global__ inline void iota_kernel(int *arr, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    arr[i] = i;
}

// KD-tree node
struct KDNode {
  int local_idx; // index in [0, cell_size)
  int left;      // index in kdnodes[], -1 if none
  int right;     // index in kdnodes[], -1 if none
};

// Dense map for editable particles: map[particle_id] -> seq_idx
struct HashTable {
  int *map;              // map[particle_id] = seq_idx, -1 if not editable
  int *particles;        // particles[seq_idx] = particle_id (reverse map)
  int N;                 // size of map array (total particle count)
  int particle_capacity; // capacity of particles array
  int *counter;          // atomic counter for sequential index
};

inline int editableParticleCapacity(int N, int num_pairs, int N_local = 0) {
  if (N <= 0 || num_pairs <= 0)
    return 0;
  long long particle_limit = (N_local > 0 && N_local < N) ? N_local : N;
  long long endpoint_limit = 2LL * num_pairs;
  long long capacity =
      endpoint_limit < particle_limit ? endpoint_limit : particle_limit;
  return capacity > INT_MAX ? INT_MAX : static_cast<int>(capacity);
}

// Compression state
template <typename T> struct CompressionState2D {
  // Parameters
  T xi, b;
  int N;
  int num_cells;
  T grid_len;
  T min_x, min_y;
  int grid_dim_x, grid_dim_y;

  // Compression outputs
  T *d_decomp_xx = nullptr;
  T *d_decomp_yy = nullptr;
  int *d_visit_order = nullptr;
  bool *d_lossless_flag = nullptr; // 2*N
  UInt *d_quant_codes = nullptr;   // compacted
  T *d_lossless_values = nullptr;  // compacted
  int *d_cell_start = nullptr;

  // Sizes after compaction
  int num_quant_codes = 0;
  int num_lossless_values = 0;

  // PGD state
  T *d_edit_x = nullptr;
  T *d_edit_y = nullptr;
  UInt2 *d_quant_edits = nullptr; // 2*N
  int num_editable_pts = 0;

  // Vulnerable pairs (each entry packs the two particle IDs of a pair)
  int2 *d_vulnerable_pairs = nullptr;
  bool *d_signs = nullptr;
  int num_vulnerable_pairs = 0;

  // Hash table for editable particles
  HashTable d_editable_pts_ht{};

  void free() {
    if (d_decomp_xx)
      CUDA_CHECK(cudaFree(d_decomp_xx));
    if (d_decomp_yy)
      CUDA_CHECK(cudaFree(d_decomp_yy));
    if (d_visit_order)
      CUDA_CHECK(cudaFree(d_visit_order));
    if (d_lossless_flag)
      CUDA_CHECK(cudaFree(d_lossless_flag));
    if (d_quant_codes)
      CUDA_CHECK(cudaFree(d_quant_codes));
    if (d_lossless_values)
      CUDA_CHECK(cudaFree(d_lossless_values));
    if (d_cell_start)
      CUDA_CHECK(cudaFree(d_cell_start));
    if (d_edit_x)
      CUDA_CHECK(cudaFree(d_edit_x));
    if (d_edit_y)
      CUDA_CHECK(cudaFree(d_edit_y));
    if (d_quant_edits)
      CUDA_CHECK(cudaFree(d_quant_edits));
    if (d_vulnerable_pairs)
      CUDA_CHECK(cudaFree(d_vulnerable_pairs));
    if (d_signs)
      CUDA_CHECK(cudaFree(d_signs));
    d_decomp_xx = d_decomp_yy = nullptr;
    d_visit_order = nullptr;
    d_lossless_flag = nullptr;
    d_quant_codes = nullptr;
    d_lossless_values = nullptr;
    d_cell_start = nullptr;
    d_edit_x = d_edit_y = nullptr;
    d_quant_edits = nullptr;
    d_vulnerable_pairs = nullptr;
    d_signs = nullptr;
  }
};

template <typename T> struct CompressionState3D {
  // Parameters
  T xi, b;
  int N;
  int num_cells;
  T grid_len;
  T min_x, min_y, min_z;
  int grid_dim_x, grid_dim_y, grid_dim_z;

  // Compression outputs
  T *d_decomp_xx = nullptr;
  T *d_decomp_yy = nullptr;
  T *d_decomp_zz = nullptr;
  int *d_visit_order = nullptr;
  bool *d_lossless_flag = nullptr; // 3*N
  UInt *d_quant_codes = nullptr;   // compacted
  T *d_lossless_values = nullptr;  // compacted
  int *d_cell_start = nullptr;

  // Sizes after compaction
  int num_quant_codes = 0;
  int num_lossless_values = 0;

  // PGD state
  T *d_edit_x = nullptr;
  T *d_edit_y = nullptr;
  T *d_edit_z = nullptr;
  UInt2 *d_quant_edits = nullptr; // 3*N
  int num_editable_pts = 0;

  // Vulnerable pairs (each entry packs the two particle IDs of a pair)
  int2 *d_vulnerable_pairs = nullptr;
  bool *d_signs = nullptr;
  int num_vulnerable_pairs = 0;

  // Hash table for editable particles
  HashTable d_editable_pts_ht{};

  void free() {
    if (d_decomp_xx)
      CUDA_CHECK(cudaFree(d_decomp_xx));
    if (d_decomp_yy)
      CUDA_CHECK(cudaFree(d_decomp_yy));
    if (d_decomp_zz)
      CUDA_CHECK(cudaFree(d_decomp_zz));
    if (d_visit_order)
      CUDA_CHECK(cudaFree(d_visit_order));
    if (d_lossless_flag)
      CUDA_CHECK(cudaFree(d_lossless_flag));
    if (d_quant_codes)
      CUDA_CHECK(cudaFree(d_quant_codes));
    if (d_lossless_values)
      CUDA_CHECK(cudaFree(d_lossless_values));
    if (d_cell_start)
      CUDA_CHECK(cudaFree(d_cell_start));
    if (d_edit_x)
      CUDA_CHECK(cudaFree(d_edit_x));
    if (d_edit_y)
      CUDA_CHECK(cudaFree(d_edit_y));
    if (d_edit_z)
      CUDA_CHECK(cudaFree(d_edit_z));
    if (d_quant_edits)
      CUDA_CHECK(cudaFree(d_quant_edits));
    if (d_vulnerable_pairs)
      CUDA_CHECK(cudaFree(d_vulnerable_pairs));
    if (d_signs)
      CUDA_CHECK(cudaFree(d_signs));
    d_decomp_xx = d_decomp_yy = d_decomp_zz = nullptr;
    d_visit_order = nullptr;
    d_lossless_flag = nullptr;
    d_quant_codes = nullptr;
    d_lossless_values = nullptr;
    d_cell_start = nullptr;
    d_edit_x = d_edit_y = d_edit_z = nullptr;
    d_quant_edits = nullptr;
    d_vulnerable_pairs = nullptr;
    d_signs = nullptr;
  }
};

// Lookup index in dense map
__device__ __forceinline__ int findIndex(const HashTable &ht, int searchValue) {
  if (searchValue < 0 || searchValue >= ht.N)
    return -1;
  return ht.map[searchValue];
}

__device__ __forceinline__ int findPairScanRootReadonly(int *parent, int x) {
  int p = parent[x];
  while (p != x) {
    x = p;
    p = parent[x];
  }
  return x;
}

__device__ __forceinline__ void unionPairScanMin(int *parent, int x, int y) {
  while (true) {
    x = findPairScanRootReadonly(parent, x);
    y = findPairScanRootReadonly(parent, y);
    if (x == y)
      return;

    int hi = max(x, y);
    int lo = min(x, y);
    int old = atomicCAS(&parent[hi], hi, lo);
    if (old == hi)
      return;
  }
}

// Insert a particle into the dense map
__device__ __forceinline__ void insertIntoHashTable(HashTable &ht, int key) {
  if (key < 0 || key >= ht.N) {
    printf("Failed to insert key %d\n", key);
    return;
  }
  // Atomically claim this slot (only first thread wins per key)
  int old = atomicCAS(&ht.map[key], -1, -2); // -2 = "being inserted"
  if (old == -1) {
    int idx = atomicAdd(ht.counter, 1);
    if (idx < ht.particle_capacity && ht.particles) {
      ht.map[key] = idx;
      ht.particles[idx] = key;
    } else {
      ht.map[key] = -1;
      printf("Failed to insert key %d: capacity %d exceeded\n", key,
             ht.particle_capacity);
    }
  }
}

// Host function to create dense editable particle map
inline HashTable createEmptyHashTable(int N, int particle_capacity) {
  HashTable ht;
  ht.N = N;
  ht.particle_capacity = particle_capacity;

  CUDA_CHECK(cudaMalloc(&ht.map, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&ht.particles, ht.particle_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&ht.counter, sizeof(int)));

  // Initialize map to -1 (not editable)
  CUDA_CHECK(cudaMemset(ht.map, 0xFF, N * sizeof(int)));
  CUDA_CHECK(cudaMemset(ht.counter, 0, sizeof(int)));

  return ht;
}

// Cleanup editable particle map
inline void destroyHashTable(HashTable &ht) {
  if (ht.map)
    CUDA_CHECK(cudaFree(ht.map));
  if (ht.particles)
    CUDA_CHECK(cudaFree(ht.particles));
  if (ht.counter)
    CUDA_CHECK(cudaFree(ht.counter));
  ht.map = nullptr;
  ht.particles = nullptr;
  ht.counter = nullptr;
  ht.N = 0;
  ht.particle_capacity = 0;
}

// Build HT from vulnerable pairs array
static __global__ void buildHashTable_kernel(const int2 *d_vulnerable_pairs,
                                             int num_pairs, HashTable ht,
                                             int N_local) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_pairs)
    return;
  int2 pair = d_vulnerable_pairs[tid];
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    int key = (e == 0) ? pair.x : pair.y;
    if (N_local > 0 && key >= N_local)
      continue;
    insertIntoHashTable(ht, key);
  }
}

inline void buildHashTableFromPairs(const int2 *d_vulnerable_pairs,
                                    int num_pairs, HashTable &ht,
                                    int N_local = 0) {
  if (num_pairs == 0)
    return;
  int blocks = (num_pairs + num_threads - 1) / num_threads;
  buildHashTable_kernel<<<blocks, num_threads>>>(d_vulnerable_pairs, num_pairs,
                                                 ht, N_local);
  CUDA_CHECK(cudaPeekAtLastError());
}

// Helper: get coordinate value by axis
template <typename T>
__device__ __forceinline__ T getCoord2D(int axis, int idx, const T *d_x,
                                        const T *d_y, const int *local_pts) {
  int gpt = local_pts[idx];
  return axis ? d_y[gpt] : d_x[gpt];
}

template <typename T>
__device__ __forceinline__ T getCoord3D(int axis, int idx, const T *d_x,
                                        const T *d_y, const T *d_z,
                                        const int *local_pts) {
  int gpt = local_pts[idx];
  return (axis == 0) ? d_x[gpt] : (axis == 1) ? d_y[gpt] : d_z[gpt];
}

// Helper: partition for quickselect; returns pivot position
template <typename T>
__device__ __forceinline__ int
partition2D(int *arr, int left, int right, int pivot_idx, int axis,
            const T *d_x, const T *d_y, const int *local_pts) {
  T pivot_val = getCoord2D(axis, arr[pivot_idx], d_x, d_y, local_pts);
  // Move pivot to end
  int tmp = arr[pivot_idx];
  arr[pivot_idx] = arr[right];
  arr[right] = tmp;
  int store = left;
  for (int i = left; i < right; i++) {
    if (getCoord2D(axis, arr[i], d_x, d_y, local_pts) < pivot_val) {
      tmp = arr[store];
      arr[store] = arr[i];
      arr[i] = tmp;
      store++;
    }
  }
  tmp = arr[store];
  arr[store] = arr[right];
  arr[right] = tmp;
  return store;
}

template <typename T>
__device__ __forceinline__ int
partition3D(int *arr, int left, int right, int pivot_idx, int axis,
            const T *d_x, const T *d_y, const T *d_z, const int *local_pts) {
  T pivot_val = getCoord3D(axis, arr[pivot_idx], d_x, d_y, d_z, local_pts);
  int tmp = arr[pivot_idx];
  arr[pivot_idx] = arr[right];
  arr[right] = tmp;
  int store = left;
  for (int i = left; i < right; i++) {
    if (getCoord3D(axis, arr[i], d_x, d_y, d_z, local_pts) < pivot_val) {
      tmp = arr[store];
      arr[store] = arr[i];
      arr[i] = tmp;
      store++;
    }
  }
  tmp = arr[store];
  arr[store] = arr[right];
  arr[right] = tmp;
  return store;
}

// Helper: quickselect to find k-th smallest and partition around it
template <typename T>
__device__ __forceinline__ void
quickselect2D(int *arr, int left, int right, int k, int axis, const T *d_x,
              const T *d_y, const int *local_pts) {
  while (left < right) {
    // Choose median-of-3 as pivot for better performance
    int mid = (left + right) >> 1;
    T vl = getCoord2D(axis, arr[left], d_x, d_y, local_pts);
    T vm = getCoord2D(axis, arr[mid], d_x, d_y, local_pts);
    T vr = getCoord2D(axis, arr[right], d_x, d_y, local_pts);
    int pivot_idx;
    if ((vl <= vm && vm <= vr) || (vr <= vm && vm <= vl))
      pivot_idx = mid;
    else if ((vm <= vl && vl <= vr) || (vr <= vl && vl <= vm))
      pivot_idx = left;
    else
      pivot_idx = right;
    int pos =
        partition2D(arr, left, right, pivot_idx, axis, d_x, d_y, local_pts);
    if (pos == k)
      return;
    else if (k < pos)
      right = pos - 1;
    else
      left = pos + 1;
  }
}

template <typename T>
__device__ __forceinline__ void
quickselect3D(int *arr, int left, int right, int k, int axis, const T *d_x,
              const T *d_y, const T *d_z, const int *local_pts) {
  while (left < right) {
    int mid = (left + right) >> 1;
    T vl = getCoord3D(axis, arr[left], d_x, d_y, d_z, local_pts);
    T vm = getCoord3D(axis, arr[mid], d_x, d_y, d_z, local_pts);
    T vr = getCoord3D(axis, arr[right], d_x, d_y, d_z, local_pts);
    int pivot_idx;
    if ((vl <= vm && vm <= vr) || (vr <= vm && vm <= vl))
      pivot_idx = mid;
    else if ((vm <= vl && vl <= vr) || (vr <= vl && vl <= vm))
      pivot_idx = left;
    else
      pivot_idx = right;
    int pos = partition3D(arr, left, right, pivot_idx, axis, d_x, d_y, d_z,
                          local_pts);
    if (pos == k)
      return;
    else if (k < pos)
      right = pos - 1;
    else
      left = pos + 1;
  }
}

// KD-tree build using quickselect
template <typename T>
__device__ int buildKDTree2D(KDNode *nodes, int &node_cnt, int *working, int n,
                             int depth, const T *d_x, const T *d_y,
                             const int *local_pts) {
  struct BuildTask {
    short start, len; // Use short to save memory (max 1024)
    short depth;
    int *result;
  };
  BuildTask stk[15]; // log2(1024) + margin
  int top = 0;
  int root_nid = -1;
  stk[top++] = {0, (short)n, (short)depth, &root_nid};
  while (top > 0) {
    BuildTask t = stk[--top];
    if (t.len <= 0) {
      *t.result = -1;
      continue;
    }
    int nid = node_cnt++;
    *t.result = nid;
    if (t.len == 1) {
      nodes[nid] = {working[t.start], -1, -1};
      continue;
    }
    int axis = t.depth & 1;
    int mid = t.len >> 1;
    // Quickselect partitions array so working[start+mid] is median
    quickselect2D(working + t.start, 0, t.len - 1, mid, axis, d_x, d_y,
                  local_pts);
    nodes[nid] = {working[t.start + mid], -1, -1};
    // Push right then left (LIFO: left processed first)
    stk[top++] = {(short)(t.start + mid + 1), (short)(t.len - mid - 1),
                  (short)(t.depth + 1), &nodes[nid].right};
    stk[top++] = {t.start, (short)mid, (short)(t.depth + 1), &nodes[nid].left};
  }
  return root_nid;
}

template <typename T>
__device__ int buildKDTree3D(KDNode *nodes, int &node_cnt, int *working, int n,
                             int depth, const T *d_x, const T *d_y,
                             const T *d_z, const int *local_pts) {
  struct BuildTask {
    short start, len;
    short depth;
    int *result;
  };
  BuildTask stk[15];
  int top = 0;
  int root_nid = -1;
  stk[top++] = {0, (short)n, (short)depth, &root_nid};
  while (top > 0) {
    BuildTask t = stk[--top];
    if (t.len <= 0) {
      *t.result = -1;
      continue;
    }
    int nid = node_cnt++;
    *t.result = nid;
    if (t.len == 1) {
      nodes[nid] = {working[t.start], -1, -1};
      continue;
    }
    int axis = t.depth % 3;
    int mid = t.len >> 1;
    quickselect3D(working + t.start, 0, t.len - 1, mid, axis, d_x, d_y, d_z,
                  local_pts);
    nodes[nid] = {working[t.start + mid], -1, -1};
    stk[top++] = {(short)(t.start + mid + 1), (short)(t.len - mid - 1),
                  (short)(t.depth + 1), &nodes[nid].right};
    stk[top++] = {t.start, (short)mid, (short)(t.depth + 1), &nodes[nid].left};
  }
  return root_nid;
}

// Nearest neighbor search
template <typename T>
__device__ void
nearestNeighborKD2D(const KDNode *nodes, int nid, const bool *visited, T qx,
                    T qy, const T *d_x, const T *d_y, const int *local_pts,
                    int depth, int &best_li, T &best_dsq) {
  // Compact task: use negative nid to indicate "far" subtree needing prune
  // check Store diff_sq separately indexed by depth to avoid large struct
  struct NNTask {
    int nid; // negative means far subtree
    short depth;
  };
  NNTask stk[30];      // 2x depth for near+far at each level
  T diff_sq_cache[15]; // Cache diff_sq for far nodes by depth
  int top = 0;
  stk[top++] = {nid, (short)depth};
  while (top > 0) {
    NNTask t = stk[--top];
    int cur_nid = t.nid;
    bool is_far = cur_nid < 0;
    if (is_far)
      cur_nid = ~cur_nid; // Decode actual nid
    if (cur_nid < 0)
      continue; // Invalid node (-1 encoded as ~(-1) = 0, but original -1 stays)

    // For far subtrees, check if worth exploring
    if (is_far && diff_sq_cache[t.depth - 1] >= best_dsq)
      continue;

    int li = nodes[cur_nid].local_idx;
    int gpt = local_pts[li];
    if (!visited[li]) {
      T dx = qx - d_x[gpt], dy = qy - d_y[gpt];
      T d = dx * dx + dy * dy;
      if (d < best_dsq) {
        best_dsq = d;
        best_li = li;
      }
    }
    int axis = t.depth & 1;
    T split = axis ? d_y[gpt] : d_x[gpt];
    T diff = (axis ? qy : qx) - split;
    T dsq = diff * diff;
    int left = nodes[cur_nid].left;
    int right = nodes[cur_nid].right;
    int near_nid = (diff < 0) ? left : right;
    int far_nid = (diff < 0) ? right : left;
    // Store diff_sq for this depth's far subtree
    diff_sq_cache[t.depth] = dsq;
    // Push far (encoded negative) first, near second (LIFO)
    if (far_nid >= 0)
      stk[top++] = {~far_nid, (short)(t.depth + 1)};
    if (near_nid >= 0)
      stk[top++] = {near_nid, (short)(t.depth + 1)};
  }
}

template <typename T>
__device__ void nearestNeighborKD3D(const KDNode *nodes, int nid,
                                    const bool *visited, T qx, T qy, T qz,
                                    const T *d_x, const T *d_y, const T *d_z,
                                    const int *local_pts, int depth,
                                    int &best_li, T &best_dsq) {
  struct NNTask {
    int nid;
    short depth;
  };
  NNTask stk[30];
  T diff_sq_cache[15];
  int top = 0;
  stk[top++] = {nid, (short)depth};
  while (top > 0) {
    NNTask t = stk[--top];
    int cur_nid = t.nid;
    bool is_far = cur_nid < 0;
    if (is_far)
      cur_nid = ~cur_nid;
    if (cur_nid < 0)
      continue;

    if (is_far && diff_sq_cache[t.depth - 1] >= best_dsq)
      continue;

    int li = nodes[cur_nid].local_idx;
    int gpt = local_pts[li];
    if (!visited[li]) {
      T dx = qx - d_x[gpt], dy = qy - d_y[gpt], dz = qz - d_z[gpt];
      T d = dx * dx + dy * dy + dz * dz;
      if (d < best_dsq) {
        best_dsq = d;
        best_li = li;
      }
    }
    int axis = t.depth % 3;
    T split = (axis == 0) ? d_x[gpt] : (axis == 1) ? d_y[gpt] : d_z[gpt];
    T diff = (axis == 0) ? qx - split : (axis == 1) ? qy - split : qz - split;
    T dsq = diff * diff;
    int left = nodes[cur_nid].left;
    int right = nodes[cur_nid].right;
    int near_nid = (diff < 0) ? left : right;
    int far_nid = (diff < 0) ? right : left;
    diff_sq_cache[t.depth] = dsq;
    if (far_nid >= 0)
      stk[top++] = {~far_nid, (short)(t.depth + 1)};
    if (near_nid >= 0)
      stk[top++] = {near_nid, (short)(t.depth + 1)};
  }
}

__device__ __forceinline__ void triangularPairFromLinear(long long k, int n,
                                                         int &i, int &j) {
  double dn = static_cast<double>(n);
  double dk = static_cast<double>(k);
  double t = 2.0 * dn - 1.0;
  int ii = static_cast<int>((t - sqrt(t * t - 8.0 * dk)) * 0.5);

  long long before = static_cast<long long>(ii) * (2LL * n - ii - 1) / 2;
  while (ii > 0 && before > k) {
    --ii;
    before = static_cast<long long>(ii) * (2LL * n - ii - 1) / 2;
  }
  long long next = before + (n - ii - 1);
  while (k >= next && ii + 1 < n) {
    ++ii;
    before = next;
    next += (n - ii - 1);
  }

  i = ii;
  j = ii + 1 + static_cast<int>(k - before);
}

template <bool CountOnly>
__device__ __forceinline__ void
appendVulnerablePair(int pt_idx1, int pt_idx2, bool sign,
                     int2 *d_vulnerable_pairs, bool *d_signs,
                     int *d_num_vulnerable_pairs, int local_pairs[][2],
                     bool *local_signs, int &num_local_pair, int max_pairs) {
  if constexpr (!CountOnly) {
    local_pairs[num_local_pair][0] = pt_idx1;
    local_pairs[num_local_pair][1] = pt_idx2;
    local_signs[num_local_pair] = sign;
  }
  ++num_local_pair;

  if (num_local_pair < max_num_local_pair)
    return;

  // buffer-full flush (atomic-contention amortization)
  if constexpr (CountOnly) {
    atomicAdd(d_num_vulnerable_pairs, num_local_pair);
  } else {
    int base_pos = atomicAdd(d_num_vulnerable_pairs, num_local_pair);
    int write_count = min(num_local_pair, max(0, max_pairs - base_pos));
    for (int j = 0; j < write_count; ++j) {
      d_vulnerable_pairs[base_pos + j] =
          make_int2(local_pairs[j][0], local_pairs[j][1]);
      d_signs[base_pos + j] = local_signs[j];
    }
  }
  num_local_pair = 0;
}

template <bool CountOnly>
__device__ __forceinline__ void
flushVulnerablePairs(int2 *d_vulnerable_pairs, bool *d_signs,
                     int *d_num_vulnerable_pairs, int local_pairs[][2],
                     bool *local_signs, int &num_local_pair, int max_pairs) {
  if (num_local_pair == 0)
    return;

  if constexpr (CountOnly) {
    atomicAdd(d_num_vulnerable_pairs, num_local_pair);
  } else {
    int base_pos = atomicAdd(d_num_vulnerable_pairs, num_local_pair);
    int write_count = min(num_local_pair, max(0, max_pairs - base_pos));
    for (int i = 0; i < write_count; ++i) {
      d_vulnerable_pairs[base_pos + i] =
          make_int2(local_pairs[i][0], local_pairs[i][1]);
      d_signs[base_pos + i] = local_signs[i];
    }
  }
  num_local_pair = 0;
}

static __global__ void countCellPairs2D_kernel(const int *d_cell_start,
                                               int num_cells, int N,
                                               int grid_dim_x, int grid_dim_y,
                                               int *d_count) {
  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  int cell_size = cell_end - cell_start;
  if (cell_size == 0)
    return;

  int count = (cell_size >= 2) ? 1 : 0;
  int id_x = cell_id % grid_dim_x;
  int id_y = cell_id / grid_dim_x;
  static constexpr int neighbor_offsets[4][2] = {
      {0, 1}, {1, 0}, {1, 1}, {1, -1}};

  for (int idx = 0; idx < 4; ++idx) {
    int nx = id_x + neighbor_offsets[idx][0];
    int ny = id_y + neighbor_offsets[idx][1];
    if (nx < 0 || nx >= grid_dim_x || ny < 0 || ny >= grid_dim_y)
      continue;

    int neighbor_cell = ny * grid_dim_x + nx;
    int neighbor_start = d_cell_start[neighbor_cell];
    int neighbor_end =
        (neighbor_cell == num_cells - 1) ? N : d_cell_start[neighbor_cell + 1];
    if (neighbor_start < neighbor_end)
      ++count;
  }

  if (count > 0)
    atomicAdd(d_count, count);
}

static __global__ void countCellPairs3D_kernel(const int *d_cell_start,
                                               int num_cells, int N,
                                               int grid_dim_x, int grid_dim_y,
                                               int grid_dim_z, int *d_count) {
  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  int cell_size = cell_end - cell_start;
  if (cell_size == 0)
    return;

  int count = (cell_size >= 2) ? 1 : 0;
  int id_x = cell_id % grid_dim_x;
  int id_y = (cell_id / grid_dim_x) % grid_dim_y;
  int id_z = cell_id / (grid_dim_x * grid_dim_y);
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

    int neighbor_cell = nz * grid_dim_x * grid_dim_y + ny * grid_dim_x + nx;
    int neighbor_start = d_cell_start[neighbor_cell];
    int neighbor_end =
        (neighbor_cell == num_cells - 1) ? N : d_cell_start[neighbor_cell + 1];
    if (neighbor_start < neighbor_end)
      ++count;
  }

  if (count > 0)
    atomicAdd(d_count, count);
}

static __global__ void fillCellPairs2D_kernel(const int *d_cell_start,
                                              int num_cells, int N,
                                              int grid_dim_x, int grid_dim_y,
                                              int2 *d_cell_pairs,
                                              int *d_count) {
  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  int cell_size = cell_end - cell_start;
  if (cell_size == 0)
    return;

  if (cell_size >= 2) {
    int pos = atomicAdd(d_count, 1);
    d_cell_pairs[pos] = make_int2(cell_id, cell_id);
  }

  int id_x = cell_id % grid_dim_x;
  int id_y = cell_id / grid_dim_x;
  static constexpr int neighbor_offsets[4][2] = {
      {0, 1}, {1, 0}, {1, 1}, {1, -1}};

  for (int idx = 0; idx < 4; ++idx) {
    int nx = id_x + neighbor_offsets[idx][0];
    int ny = id_y + neighbor_offsets[idx][1];
    if (nx < 0 || nx >= grid_dim_x || ny < 0 || ny >= grid_dim_y)
      continue;

    int neighbor_cell = ny * grid_dim_x + nx;
    int neighbor_start = d_cell_start[neighbor_cell];
    int neighbor_end =
        (neighbor_cell == num_cells - 1) ? N : d_cell_start[neighbor_cell + 1];
    if (neighbor_start >= neighbor_end)
      continue;

    int pos = atomicAdd(d_count, 1);
    d_cell_pairs[pos] = make_int2(cell_id, neighbor_cell);
  }
}

static __global__ void
fillCellPairs3D_kernel(const int *d_cell_start, int num_cells, int N,
                       int grid_dim_x, int grid_dim_y, int grid_dim_z,
                       int2 *d_cell_pairs, int *d_count) {
  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  int cell_size = cell_end - cell_start;
  if (cell_size == 0)
    return;

  if (cell_size >= 2) {
    int pos = atomicAdd(d_count, 1);
    d_cell_pairs[pos] = make_int2(cell_id, cell_id);
  }

  int id_x = cell_id % grid_dim_x;
  int id_y = (cell_id / grid_dim_x) % grid_dim_y;
  int id_z = cell_id / (grid_dim_x * grid_dim_y);
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

    int neighbor_cell = nz * grid_dim_x * grid_dim_y + ny * grid_dim_x + nx;
    int neighbor_start = d_cell_start[neighbor_cell];
    int neighbor_end =
        (neighbor_cell == num_cells - 1) ? N : d_cell_start[neighbor_cell + 1];
    if (neighbor_start >= neighbor_end)
      continue;

    int pos = atomicAdd(d_count, 1);
    d_cell_pairs[pos] = make_int2(cell_id, neighbor_cell);
  }
}

template <bool CountOnly, typename T>
__global__ void findVulnerablePairsCellPairs2D_kernel(
    const T *__restrict__ d_org_xx, const T *__restrict__ d_org_yy,
    const int *d_cell_start, const int *d_cell_pts_sorted,
    const int2 *d_cell_pairs, int num_cell_pairs, int num_cells,
    int2 *d_vulnerable_pairs, bool *d_signs, int *d_num_vulnerable_pairs, int N,
    T lower_bound_sq, T upper_bound_sq, T sign_bound_sq, int *d_safe_roots,
    int *d_halo_roots, T safe_bound_sq, T halo_bound_sq, int max_pairs,
    int N_local = 0) {
  if (N_local == 0)
    N_local = N;

  int cell_pair_idx = blockIdx.x;
  if (cell_pair_idx >= num_cell_pairs)
    return;

  int2 cell_pair = d_cell_pairs[cell_pair_idx];
  int cell_a = cell_pair.x;
  int cell_b = cell_pair.y;
  int start_a = d_cell_start[cell_a];
  int end_a = (cell_a == num_cells - 1) ? N : d_cell_start[cell_a + 1];
  int start_b = d_cell_start[cell_b];
  int end_b = (cell_b == num_cells - 1) ? N : d_cell_start[cell_b + 1];
  int n_a = end_a - start_a;
  int n_b = end_b - start_b;
  bool same_cell = (cell_a == cell_b);

  long long total_pairs = same_cell
                              ? (static_cast<long long>(n_a) * (n_a - 1)) / 2
                              : static_cast<long long>(n_a) * n_b;

  int local_pairs[max_num_local_pair][2];
  bool local_signs[max_num_local_pair];
  int num_local_pair = 0;

  for (long long k = threadIdx.x; k < total_pairs; k += blockDim.x) {
    int local_i, local_j;
    if (same_cell) {
      triangularPairFromLinear(k, n_a, local_i, local_j);
    } else {
      local_i = static_cast<int>(k / n_b);
      local_j = static_cast<int>(k - static_cast<long long>(local_i) * n_b);
    }

    int pt_idx1 = d_cell_pts_sorted[start_a + local_i];
    int pt_idx2 = d_cell_pts_sorted[(same_cell ? start_a : start_b) + local_j];

    T dx = d_org_xx[pt_idx1] - d_org_xx[pt_idx2];
    T dy = d_org_yy[pt_idx1] - d_org_yy[pt_idx2];
    T dist_sq = dx * dx + dy * dy;
    if constexpr (CountOnly) {
      if (d_safe_roots && dist_sq <= safe_bound_sq)
        unionPairScanMin(d_safe_roots, pt_idx1, pt_idx2);
      if (d_halo_roots && dist_sq <= halo_bound_sq)
        unionPairScanMin(d_halo_roots, pt_idx1, pt_idx2);
    }

    if (pt_idx1 >= N_local && pt_idx2 >= N_local)
      continue;

    if (dist_sq > lower_bound_sq && dist_sq <= upper_bound_sq) {
      appendVulnerablePair<CountOnly>(pt_idx1, pt_idx2, dist_sq > sign_bound_sq,
                                      d_vulnerable_pairs, d_signs,
                                      d_num_vulnerable_pairs, local_pairs,
                                      local_signs, num_local_pair, max_pairs);
    }
  }

  flushVulnerablePairs<CountOnly>(d_vulnerable_pairs, d_signs,
                                  d_num_vulnerable_pairs, local_pairs,
                                  local_signs, num_local_pair, max_pairs);
}

template <bool CountOnly, typename T>
__global__ void findVulnerablePairsCellPairs3D_kernel(
    const T *__restrict__ d_org_xx, const T *__restrict__ d_org_yy,
    const T *__restrict__ d_org_zz, const int *d_cell_start,
    const int *d_cell_pts_sorted, const int2 *d_cell_pairs, int num_cell_pairs,
    int num_cells, int2 *d_vulnerable_pairs, bool *d_signs,
    int *d_num_vulnerable_pairs, int N, T lower_bound_sq, T upper_bound_sq,
    T sign_bound_sq, int *d_safe_roots, int *d_halo_roots, T safe_bound_sq,
    T halo_bound_sq, int max_pairs, int N_local = 0) {
  if (N_local == 0)
    N_local = N;

  int cell_pair_idx = blockIdx.x;
  if (cell_pair_idx >= num_cell_pairs)
    return;

  int2 cell_pair = d_cell_pairs[cell_pair_idx];
  int cell_a = cell_pair.x;
  int cell_b = cell_pair.y;
  int start_a = d_cell_start[cell_a];
  int end_a = (cell_a == num_cells - 1) ? N : d_cell_start[cell_a + 1];
  int start_b = d_cell_start[cell_b];
  int end_b = (cell_b == num_cells - 1) ? N : d_cell_start[cell_b + 1];
  int n_a = end_a - start_a;
  int n_b = end_b - start_b;
  bool same_cell = (cell_a == cell_b);

  long long total_pairs = same_cell
                              ? (static_cast<long long>(n_a) * (n_a - 1)) / 2
                              : static_cast<long long>(n_a) * n_b;

  int local_pairs[max_num_local_pair][2];
  bool local_signs[max_num_local_pair];
  int num_local_pair = 0;

  for (long long k = threadIdx.x; k < total_pairs; k += blockDim.x) {
    int local_i, local_j;
    if (same_cell) {
      triangularPairFromLinear(k, n_a, local_i, local_j);
    } else {
      local_i = static_cast<int>(k / n_b);
      local_j = static_cast<int>(k - static_cast<long long>(local_i) * n_b);
    }

    int pt_idx1 = d_cell_pts_sorted[start_a + local_i];
    int pt_idx2 = d_cell_pts_sorted[(same_cell ? start_a : start_b) + local_j];

    T dx = d_org_xx[pt_idx1] - d_org_xx[pt_idx2];
    T dy = d_org_yy[pt_idx1] - d_org_yy[pt_idx2];
    T dz = d_org_zz[pt_idx1] - d_org_zz[pt_idx2];
    T dist_sq = dx * dx + dy * dy + dz * dz;
    if constexpr (CountOnly) {
      if (d_safe_roots && dist_sq <= safe_bound_sq)
        unionPairScanMin(d_safe_roots, pt_idx1, pt_idx2);
      if (d_halo_roots && dist_sq <= halo_bound_sq)
        unionPairScanMin(d_halo_roots, pt_idx1, pt_idx2);
    }

    if (pt_idx1 >= N_local && pt_idx2 >= N_local)
      continue;

    if (dist_sq > lower_bound_sq && dist_sq <= upper_bound_sq) {
      appendVulnerablePair<CountOnly>(pt_idx1, pt_idx2, dist_sq > sign_bound_sq,
                                      d_vulnerable_pairs, d_signs,
                                      d_num_vulnerable_pairs, local_pairs,
                                      local_signs, num_local_pair, max_pairs);
    }
  }

  flushVulnerablePairs<CountOnly>(d_vulnerable_pairs, d_signs,
                                  d_num_vulnerable_pairs, local_pairs,
                                  local_signs, num_local_pair, max_pairs);
}

// (De)quantization device functions
template <typename T>
__device__ __forceinline__ int quantize_device(T value, T xi) {
  return static_cast<int>(roundf(value / (2 * xi))) + (1 << (m - 1));
}

template <typename T>
__device__ __forceinline__ T dequantize_device(int quantized, T xi) {
  return static_cast<T>(quantized - (1 << (m - 1))) * 2 * xi;
}

// Compress a single coordinate value
template <typename T>
__device__ __forceinline__ T compressCoord(T org_val, T prev_val, T xi,
                                           int out_idx, bool *d_lossless_flag,
                                           UInt *d_temp_qcode, T *d_temp_lval,
                                           int &qcount, int &lcount) {
  int quant = quantize_device(org_val - prev_val, xi);
  if (quant > 0 && quant < (1 << m)) {
    d_lossless_flag[out_idx] = false;
    d_temp_qcode[out_idx] = (UInt)quant;
    d_temp_lval[out_idx] = T(0);
    qcount++;
    return dequantize_device(quant, xi) + prev_val;
  } else {
    d_lossless_flag[out_idx] = true;
    d_temp_qcode[out_idx] = (UInt)0;
    d_temp_lval[out_idx] = org_val;
    lcount++;
    return org_val;
  }
}

// Helper: Morton code
__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ __forceinline__ uint32_t morton2D(uint32_t x, uint32_t y) {
  return (expandBits(y) << 1) | expandBits(x);
}

__device__ __forceinline__ uint32_t morton3D(uint32_t x, uint32_t y,
                                             uint32_t z) {
  return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}

// Radix sort for Morton codes
// Uses 8-bit radix (4 passes for 32-bit keys)
__device__ __forceinline__ void radixSort(int *order, const uint32_t *codes,
                                          int n) {
  if (n <= 1)
    return;

  // For very small arrays, use insertion sort
  if (n <= 8) {
    for (int i = 1; i < n; i++) {
      int key = order[i];
      uint32_t kcode = codes[key];
      int j = i - 1;
      while (j >= 0 && codes[order[j]] > kcode) {
        order[j + 1] = order[j];
        j--;
      }
      order[j + 1] = key;
    }
    return;
  }

  // Radix sort with 8-bit radix (256 buckets, 4 passes)
  int temp[MAX_CELL_PTS];
  int count[256];

  for (int shift = 0; shift < 32; shift += 8) {
    // Count occurrences
#pragma unroll
    for (int i = 0; i < 256; i++)
      count[i] = 0;
    for (int i = 0; i < n; i++) {
      int digit = (codes[order[i]] >> shift) & 0xFF;
      count[digit]++;
    }

    // Prefix sum
    int total = 0;
#pragma unroll
    for (int i = 0; i < 256; i++) {
      int c = count[i];
      count[i] = total;
      total += c;
    }

    // Scatter to temp
    for (int i = 0; i < n; i++) {
      int idx = order[i];
      int digit = (codes[idx] >> shift) & 0xFF;
      temp[count[digit]++] = idx;
    }

    // Copy back
    for (int i = 0; i < n; i++)
      order[i] = temp[i];
  }
}

template <typename T, OrderMode Mode>
__global__ void compressParticles2D_kernel(
    const T *d_org_xx, const T *d_org_yy, const int *d_cell_start,
    int num_cells, int N, T min_x, T min_y, T grid_len, T xi, int grid_dim_x,
    int *d_visit_order, bool *d_lossless_flag, T *d_decomp_xx, T *d_decomp_yy,
    UInt *d_temp_qcode, T *d_temp_lval, int *d_cell_quant_count,
    int *d_cell_lossless_count) {

  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  int n = cell_end - cell_start;
  if (n == 0) {
    d_cell_quant_count[cell_id] = 1; // empty cell marker
    d_cell_lossless_count[cell_id] = 0;
    return;
  }
  if (n > MAX_CELL_PTS) {
    printf("Warning: cell %d: %d pts > MAX_CELL_PTS, skipping\n", cell_id, n);
    d_cell_quant_count[cell_id] = 1;
    d_cell_lossless_count[cell_id] = 0;
    return;
  }

  int qcount = 0, lcount = 0;

  // Copy cell particle indices to local memory before overwriting d_visit_order
  int local_pts_buf[MAX_CELL_PTS];
  for (int i = 0; i < n; i++)
    local_pts_buf[i] = d_visit_order[cell_start + i];
  const int *local_pts = local_pts_buf;

  int id_x = cell_id % grid_dim_x;
  int id_y = cell_id / grid_dim_x;

  T prev_x, prev_y;

  if constexpr (Mode == OrderMode::MORTON_CODE) {
    T cell_min_x = min_x + (T)id_x * grid_len;
    T cell_min_y = min_y + (T)id_y * grid_len;
    prev_x = cell_min_x;
    prev_y = cell_min_y;

    // Compute Morton codes, then sort order[] by codes[]
    uint32_t codes[MAX_CELL_PTS];
    int order[MAX_CELL_PTS];
    for (int i = 0; i < n; i++) {
      order[i] = i;
      int idx = local_pts[i];
      uint32_t ix = (uint32_t)min_templated(
          (T)1023.0, (d_org_xx[idx] - cell_min_x) / grid_len * (T)1024.0);
      uint32_t iy = (uint32_t)min_templated(
          (T)1023.0, (d_org_yy[idx] - cell_min_y) / grid_len * (T)1024.0);
      codes[i] = morton2D(ix, iy);
    }
    radixSort(order, codes, n);

    // Visit by Morton code order and compress
    for (int i = 0; i < n; i++) {
      int org_idx = local_pts[order[i]];
      int pos = cell_start + i;
      d_visit_order[pos] = org_idx;

      prev_x =
          compressCoord(d_org_xx[org_idx], prev_x, xi, 2 * pos, d_lossless_flag,
                        d_temp_qcode, d_temp_lval, qcount, lcount);
      d_decomp_xx[org_idx] = prev_x;

      prev_y = compressCoord(d_org_yy[org_idx], prev_y, xi, 2 * pos + 1,
                             d_lossless_flag, d_temp_qcode, d_temp_lval, qcount,
                             lcount);
      d_decomp_yy[org_idx] = prev_y;
    }

  } else { // KD_TREE
    prev_x = min_x + (id_x + T(0.5)) * grid_len;
    prev_y = min_y + (id_y + T(0.5)) * grid_len;

    KDNode kdnodes[MAX_CELL_PTS];
    int working[MAX_CELL_PTS];
    for (int i = 0; i < n; i++)
      working[i] = i;
    int node_cnt = 0;
    int root = buildKDTree2D(kdnodes, node_cnt, working, n, 0, d_org_xx,
                             d_org_yy, local_pts);

    bool visited[MAX_CELL_PTS];
    for (int i = 0; i < n; i++)
      visited[i] = false;

    // Visit by nearest neighbor and compress
    for (int i = 0; i < n; i++) {
      int next_local_idx = 0;
      T best_dist_sq = (T)1e30;
      nearestNeighborKD2D(kdnodes, root, visited, prev_x, prev_y, d_org_xx,
                          d_org_yy, local_pts, 0, next_local_idx, best_dist_sq);
      visited[next_local_idx] = true;

      int org_idx = local_pts[next_local_idx];
      int pos = cell_start + i;
      d_visit_order[pos] = org_idx;

      prev_x =
          compressCoord(d_org_xx[org_idx], prev_x, xi, 2 * pos, d_lossless_flag,
                        d_temp_qcode, d_temp_lval, qcount, lcount);
      d_decomp_xx[org_idx] = prev_x;

      prev_y = compressCoord(d_org_yy[org_idx], prev_y, xi, 2 * pos + 1,
                             d_lossless_flag, d_temp_qcode, d_temp_lval, qcount,
                             lcount);
      d_decomp_yy[org_idx] = prev_y;
    }
  }

  d_cell_quant_count[cell_id] = qcount + 1; // +1 for end-of-cell marker
  d_cell_lossless_count[cell_id] = lcount;
}

template <typename T, OrderMode Mode>
__global__ void compressParticles3D_kernel(
    const T *d_org_xx, const T *d_org_yy, const T *d_org_zz,
    const int *d_cell_start, int num_cells, int N, T min_x, T min_y, T min_z,
    T grid_len, T xi, int grid_dim_x, int grid_dim_y, int *d_visit_order,
    bool *d_lossless_flag, T *d_decomp_xx, T *d_decomp_yy, T *d_decomp_zz,
    UInt *d_temp_qcode, T *d_temp_lval, int *d_cell_quant_count,
    int *d_cell_lossless_count) {

  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  int n = cell_end - cell_start;
  if (n == 0) {
    d_cell_quant_count[cell_id] = 1; // empty cell marker
    d_cell_lossless_count[cell_id] = 0;
    return;
  }
  if (n > MAX_CELL_PTS) {
    printf("Warning: cell %d: %d pts > MAX_CELL_PTS, skipping\n", cell_id, n);
    d_cell_quant_count[cell_id] = 1;
    d_cell_lossless_count[cell_id] = 0;
    return;
  }

  int qcount = 0, lcount = 0;

  // Copy cell particle indices to local memory before overwriting d_visit_order
  int local_pts_buf[MAX_CELL_PTS];
  for (int i = 0; i < n; i++)
    local_pts_buf[i] = d_visit_order[cell_start + i];
  const int *local_pts = local_pts_buf;

  int id_x = cell_id % grid_dim_x;
  int id_y = (cell_id / grid_dim_x) % grid_dim_y;
  int id_z = cell_id / (grid_dim_x * grid_dim_y);

  T prev_x, prev_y, prev_z;

  if constexpr (Mode == OrderMode::MORTON_CODE) {
    T cell_min_x = min_x + (T)id_x * grid_len;
    T cell_min_y = min_y + (T)id_y * grid_len;
    T cell_min_z = min_z + (T)id_z * grid_len;
    prev_x = cell_min_x;
    prev_y = cell_min_y;
    prev_z = cell_min_z;

    uint32_t codes[MAX_CELL_PTS];
    int order[MAX_CELL_PTS];
    for (int i = 0; i < n; i++) {
      order[i] = i;
      int idx = local_pts[i];
      uint32_t ix = (uint32_t)min_templated(
          (T)1023.0, (d_org_xx[idx] - cell_min_x) / grid_len * (T)1024.0);
      uint32_t iy = (uint32_t)min_templated(
          (T)1023.0, (d_org_yy[idx] - cell_min_y) / grid_len * (T)1024.0);
      uint32_t iz = (uint32_t)min_templated(
          (T)1023.0, (d_org_zz[idx] - cell_min_z) / grid_len * (T)1024.0);
      codes[i] = morton3D(ix, iy, iz);
    }
    radixSort(order, codes, n);

    for (int i = 0; i < n; i++) {
      int org_idx = local_pts[order[i]];
      int pos = cell_start + i;
      d_visit_order[pos] = org_idx;

      prev_x =
          compressCoord(d_org_xx[org_idx], prev_x, xi, 3 * pos, d_lossless_flag,
                        d_temp_qcode, d_temp_lval, qcount, lcount);
      d_decomp_xx[org_idx] = prev_x;

      prev_y = compressCoord(d_org_yy[org_idx], prev_y, xi, 3 * pos + 1,
                             d_lossless_flag, d_temp_qcode, d_temp_lval, qcount,
                             lcount);
      d_decomp_yy[org_idx] = prev_y;

      prev_z = compressCoord(d_org_zz[org_idx], prev_z, xi, 3 * pos + 2,
                             d_lossless_flag, d_temp_qcode, d_temp_lval, qcount,
                             lcount);
      d_decomp_zz[org_idx] = prev_z;
    }

  } else { // KD_TREE
    prev_x = min_x + (id_x + T(0.5)) * grid_len;
    prev_y = min_y + (id_y + T(0.5)) * grid_len;
    prev_z = min_z + (id_z + T(0.5)) * grid_len;

    KDNode kdnodes[MAX_CELL_PTS];
    int working[MAX_CELL_PTS];
    for (int i = 0; i < n; i++)
      working[i] = i;
    int node_cnt = 0;
    int root = buildKDTree3D(kdnodes, node_cnt, working, n, 0, d_org_xx,
                             d_org_yy, d_org_zz, local_pts);

    bool visited[MAX_CELL_PTS];
    for (int i = 0; i < n; i++)
      visited[i] = false;

    for (int i = 0; i < n; i++) {
      int next_local_idx = 0;
      T best_dist_sq = (T)1e30;
      nearestNeighborKD3D(kdnodes, root, visited, prev_x, prev_y, prev_z,
                          d_org_xx, d_org_yy, d_org_zz, local_pts, 0,
                          next_local_idx, best_dist_sq);
      visited[next_local_idx] = true;

      int org_idx = local_pts[next_local_idx];
      int pos = cell_start + i;
      d_visit_order[pos] = org_idx;

      prev_x =
          compressCoord(d_org_xx[org_idx], prev_x, xi, 3 * pos, d_lossless_flag,
                        d_temp_qcode, d_temp_lval, qcount, lcount);
      d_decomp_xx[org_idx] = prev_x;

      prev_y = compressCoord(d_org_yy[org_idx], prev_y, xi, 3 * pos + 1,
                             d_lossless_flag, d_temp_qcode, d_temp_lval, qcount,
                             lcount);
      d_decomp_yy[org_idx] = prev_y;

      prev_z = compressCoord(d_org_zz[org_idx], prev_z, xi, 3 * pos + 2,
                             d_lossless_flag, d_temp_qcode, d_temp_lval, qcount,
                             lcount);
      d_decomp_zz[org_idx] = prev_z;
    }
  }

  d_cell_quant_count[cell_id] = qcount + 1; // +1 for end-of-cell marker
  d_cell_lossless_count[cell_id] = lcount;
}

template <typename T>
__global__ void
computePGDLoss2D_kernel(const int2 *vulnerable_links, const bool *signs,
                        const T *decomp_x, const T *decomp_y, T b, T xi,
                        int num_pairs, T *loss_buffer, T decomp_tol) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt_templated(T(2));
  T upper_bound_dist = b + max_quant_dist_err;
  T lower_bound_dist = max_templated(T(0), b - max_quant_dist_err);
  T local_loss = 0;

  if (idx < num_pairs) {
    int2 link = vulnerable_links[idx];
    int i = link.x;
    int j = link.y;
    T dx = decomp_x[i] - decomp_x[j];
    T dy = decomp_y[i] - decomp_y[j];
    T d_decomp = sqrt_templated(dx * dx + dy * dy);

    if (d_decomp >= decomp_tol) {
      bool sign = signs[idx];
      if (sign && d_decomp <= upper_bound_dist) {
        T violation = d_decomp - upper_bound_dist;
        local_loss = violation * violation;
      } else if (!sign && d_decomp > lower_bound_dist) {
        T violation = d_decomp - lower_bound_dist;
        local_loss = violation * violation;
      }
    }
  }

  // Parallel reduction in shared memory
  extern __shared__ unsigned char sdata_raw_2d[];
  T *sdata = reinterpret_cast<T *>(sdata_raw_2d);
  int tid = threadIdx.x;
  sdata[tid] = local_loss;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(loss_buffer, sdata[0]);
  }
}

template <typename T>
__global__ void computePGDLoss3D_kernel(const int2 *vulnerable_links,
                                        const bool *signs, const T *decomp_x,
                                        const T *decomp_y, const T *decomp_z,
                                        T b, T xi, int num_pairs,
                                        T *loss_buffer, T decomp_tol) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt_templated(T(3));
  T upper_bound_dist = b + max_quant_dist_err;
  T lower_bound_dist = max_templated(T(0), b - max_quant_dist_err);
  T local_loss = 0;

  if (idx < num_pairs) {
    int2 link = vulnerable_links[idx];
    int i = link.x;
    int j = link.y;
    T dx = decomp_x[i] - decomp_x[j];
    T dy = decomp_y[i] - decomp_y[j];
    T dz = decomp_z[i] - decomp_z[j];
    T d_decomp = sqrt_templated(dx * dx + dy * dy + dz * dz);

    if (d_decomp >= decomp_tol) {
      bool sign = signs[idx];
      if (sign && d_decomp <= upper_bound_dist) {
        T violation = d_decomp - upper_bound_dist;
        local_loss = violation * violation;
      } else if (!sign && d_decomp > lower_bound_dist) {
        T violation = d_decomp - lower_bound_dist;
        local_loss = violation * violation;
      }
    }
  }

  // Parallel reduction in shared memory
  extern __shared__ unsigned char sdata_raw_2d[];
  T *sdata = reinterpret_cast<T *>(sdata_raw_2d);
  int tid = threadIdx.x;
  sdata[tid] = local_loss;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(loss_buffer, sdata[0]);
  }
}

template <typename T>
__global__ void computePGDGradients2D_kernel(
    const int2 *vulnerable_links, const bool *signs, const T *decomp_x,
    const T *decomp_y, HashTable editable_pts_ht, T b, T xi, int num_pairs,
    T decomp_tol, T *grad_x, T *grad_y) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt_templated(T(2));
  T upper_bound_dist = b + max_quant_dist_err;
  T lower_bound_dist = max_templated(T(0), b - max_quant_dist_err);

  if (idx < num_pairs) {
    int2 link = vulnerable_links[idx];
    int i = link.x;
    int j = link.y;

    T dx = decomp_x[i] - decomp_x[j];
    T dy = decomp_y[i] - decomp_y[j];
    T d_decomp = sqrt_templated(dx * dx + dy * dy);

    if (d_decomp < decomp_tol)
      return;

    bool sign = signs[idx];
    if ((sign && d_decomp <= upper_bound_dist) ||
        (!sign && d_decomp > lower_bound_dist)) {
      T target = sign ? upper_bound_dist : lower_bound_dist;
      T tmp = 2 * (d_decomp - target) / d_decomp;

      int ii = findIndex(editable_pts_ht, i);
      int jj = findIndex(editable_pts_ht, j);

      T grad_contrib_x = tmp * dx;
      T grad_contrib_y = tmp * dy;

      if (ii >= 0) {
        atomicAdd(&grad_x[ii], grad_contrib_x);
        atomicAdd(&grad_y[ii], grad_contrib_y);
      }
      if (jj >= 0) {
        atomicAdd(&grad_x[jj], -grad_contrib_x);
        atomicAdd(&grad_y[jj], -grad_contrib_y);
      }
    }
  }
}

template <typename T>
__global__ void computePGDGradients3D_kernel(
    const int2 *vulnerable_links, const bool *signs, const T *decomp_x,
    const T *decomp_y, const T *decomp_z, HashTable editable_pts_ht, T b, T xi,
    int num_pairs, T decomp_tol, T *grad_x, T *grad_y, T *grad_z) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt_templated(T(3));
  T upper_bound_dist = b + max_quant_dist_err;
  T lower_bound_dist = max_templated(T(0), b - max_quant_dist_err);

  if (idx < num_pairs) {
    int2 link = vulnerable_links[idx];
    int i = link.x;
    int j = link.y;

    T dx = decomp_x[i] - decomp_x[j];
    T dy = decomp_y[i] - decomp_y[j];
    T dz = decomp_z[i] - decomp_z[j];
    T d_decomp = sqrt_templated(dx * dx + dy * dy + dz * dz);

    if (d_decomp < decomp_tol)
      return;

    bool sign = signs[idx];
    if ((sign && d_decomp <= upper_bound_dist) ||
        (!sign && d_decomp > lower_bound_dist)) {
      T target = sign ? upper_bound_dist : lower_bound_dist;
      T tmp = 2 * (d_decomp - target) / d_decomp;

      int ii = findIndex(editable_pts_ht, i);
      int jj = findIndex(editable_pts_ht, j);

      T grad_contrib_x = tmp * dx;
      T grad_contrib_y = tmp * dy;
      T grad_contrib_z = tmp * dz;

      if (ii >= 0) {
        atomicAdd(&grad_x[ii], grad_contrib_x);
        atomicAdd(&grad_y[ii], grad_contrib_y);
        atomicAdd(&grad_z[ii], grad_contrib_z);
      }
      if (jj >= 0) {
        atomicAdd(&grad_x[jj], -grad_contrib_x);
        atomicAdd(&grad_y[jj], -grad_contrib_y);
        atomicAdd(&grad_z[jj], -grad_contrib_z);
      }
    }
  }
}

template <typename T>
__global__ void
updatePGDPositions2D_kernel(const T *d_org_xx, const T *d_org_yy,
                            const T *grad_x, const T *grad_y,
                            HashTable editable_pts_ht, T *decomp_x, T *decomp_y,
                            T *edit_x, T *edit_y, T lr, T xi) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *editable_pts_ht.counter) {
    int i = editable_pts_ht.particles[idx]; // seq_idx → particle_idx

    // Save previous positions
    T prev_x = decomp_x[i];
    T prev_y = decomp_y[i];

    // Gradient descent step
    decomp_x[i] -= lr * grad_x[idx];
    decomp_y[i] -= lr * grad_y[idx];

    // Project onto box constraints
    T lower_x = d_org_xx[i] - xi;
    T upper_x = d_org_xx[i] + xi;
    T lower_y = d_org_yy[i] - xi;
    T upper_y = d_org_yy[i] + xi;

    decomp_x[i] = max_templated(lower_x, min_templated(upper_x, decomp_x[i]));
    decomp_y[i] = max_templated(lower_y, min_templated(upper_y, decomp_y[i]));

    // Accumulate edits
    edit_x[idx] += decomp_x[i] - prev_x;
    edit_y[idx] += decomp_y[i] - prev_y;
  }
}

template <typename T>
__global__ void updatePGDPositions3D_kernel(
    const T *d_org_xx, const T *d_org_yy, const T *d_org_zz, const T *grad_x,
    const T *grad_y, const T *grad_z, HashTable editable_pts_ht, T *decomp_x,
    T *decomp_y, T *decomp_z, T *edit_x, T *edit_y, T *edit_z, T lr, T xi) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *editable_pts_ht.counter) {
    int i = editable_pts_ht.particles[idx]; // seq_idx → particle_idx

    // Save previous positions
    T prev_x = decomp_x[i];
    T prev_y = decomp_y[i];
    T prev_z = decomp_z[i];

    // Gradient descent step
    decomp_x[i] -= lr * grad_x[idx];
    decomp_y[i] -= lr * grad_y[idx];
    decomp_z[i] -= lr * grad_z[idx];

    // Project onto box constraints
    T lower_x = d_org_xx[i] - xi;
    T upper_x = d_org_xx[i] + xi;
    T lower_y = d_org_yy[i] - xi;
    T upper_y = d_org_yy[i] + xi;
    T lower_z = d_org_zz[i] - xi;
    T upper_z = d_org_zz[i] + xi;

    decomp_x[i] = max_templated(lower_x, min_templated(upper_x, decomp_x[i]));
    decomp_y[i] = max_templated(lower_y, min_templated(upper_y, decomp_y[i]));
    decomp_z[i] = max_templated(lower_z, min_templated(upper_z, decomp_z[i]));

    // Accumulate edits
    edit_x[idx] += decomp_x[i] - prev_x;
    edit_y[idx] += decomp_y[i] - prev_y;
    edit_z[idx] += decomp_z[i] - prev_z;
  }
}

// Adam optimizer update kernels
template <typename T>
__global__ void updatePGDPositionsAdam2D_kernel(
    const T *d_org_xx, const T *d_org_yy, const T *grad_x, const T *grad_y,
    HashTable editable_pts_ht, T *decomp_x, T *decomp_y, T *edit_x, T *edit_y,
    T *m_x, T *m_y, T *v_x, T *v_y, T beta1, T beta2, T eps, T lr_t, T xi) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *editable_pts_ht.counter) {
    int i = editable_pts_ht.particles[idx];

    T prev_x = decomp_x[i];
    T prev_y = decomp_y[i];

    // Adam moment updates
    T gx = grad_x[idx], gy = grad_y[idx];
    m_x[idx] = beta1 * m_x[idx] + (1 - beta1) * gx;
    m_y[idx] = beta1 * m_y[idx] + (1 - beta1) * gy;
    v_x[idx] = beta2 * v_x[idx] + (1 - beta2) * gx * gx;
    v_y[idx] = beta2 * v_y[idx] + (1 - beta2) * gy * gy;

    // lr_t already has bias correction baked in: alpha * sqrt(1-beta2^t) /
    // (1-beta1^t)
    decomp_x[i] -= lr_t * m_x[idx] / (sqrt_templated(v_x[idx]) + eps);
    decomp_y[i] -= lr_t * m_y[idx] / (sqrt_templated(v_y[idx]) + eps);

    // Project onto box constraints
    decomp_x[i] = max_templated(d_org_xx[i] - xi,
                                min_templated(d_org_xx[i] + xi, decomp_x[i]));
    decomp_y[i] = max_templated(d_org_yy[i] - xi,
                                min_templated(d_org_yy[i] + xi, decomp_y[i]));

    edit_x[idx] += decomp_x[i] - prev_x;
    edit_y[idx] += decomp_y[i] - prev_y;
  }
}

template <typename T>
__global__ void updatePGDPositionsAdam3D_kernel(
    const T *d_org_xx, const T *d_org_yy, const T *d_org_zz, const T *grad_x,
    const T *grad_y, const T *grad_z, HashTable editable_pts_ht, T *decomp_x,
    T *decomp_y, T *decomp_z, T *edit_x, T *edit_y, T *edit_z, T *m_x, T *m_y,
    T *m_z, T *v_x, T *v_y, T *v_z, T beta1, T beta2, T eps, T lr_t, T xi) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *editable_pts_ht.counter) {
    int i = editable_pts_ht.particles[idx];

    T prev_x = decomp_x[i];
    T prev_y = decomp_y[i];
    T prev_z = decomp_z[i];

    // Adam moment updates
    T gx = grad_x[idx], gy = grad_y[idx], gz = grad_z[idx];
    m_x[idx] = beta1 * m_x[idx] + (1 - beta1) * gx;
    m_y[idx] = beta1 * m_y[idx] + (1 - beta1) * gy;
    m_z[idx] = beta1 * m_z[idx] + (1 - beta1) * gz;
    v_x[idx] = beta2 * v_x[idx] + (1 - beta2) * gx * gx;
    v_y[idx] = beta2 * v_y[idx] + (1 - beta2) * gy * gy;
    v_z[idx] = beta2 * v_z[idx] + (1 - beta2) * gz * gz;

    // lr_t already has bias correction baked in: alpha * sqrt(1-beta2^t) /
    // (1-beta1^t)
    decomp_x[i] -= lr_t * m_x[idx] / (sqrt_templated(v_x[idx]) + eps);
    decomp_y[i] -= lr_t * m_y[idx] / (sqrt_templated(v_y[idx]) + eps);
    decomp_z[i] -= lr_t * m_z[idx] / (sqrt_templated(v_z[idx]) + eps);

    // Project onto box constraints
    decomp_x[i] = max_templated(d_org_xx[i] - xi,
                                min_templated(d_org_xx[i] + xi, decomp_x[i]));
    decomp_y[i] = max_templated(d_org_yy[i] - xi,
                                min_templated(d_org_yy[i] + xi, decomp_y[i]));
    decomp_z[i] = max_templated(d_org_zz[i] - xi,
                                min_templated(d_org_zz[i] + xi, decomp_z[i]));

    edit_x[idx] += decomp_x[i] - prev_x;
    edit_y[idx] += decomp_y[i] - prev_y;
    edit_z[idx] += decomp_z[i] - prev_z;
  }
}

template <typename T>
__device__ __forceinline__ bool
hasNonzeroEdit2D(const T *edit_x, const T *edit_y, int edit_idx, T edit_tol) {
  return abs_templated(edit_x[edit_idx]) > edit_tol ||
         abs_templated(edit_y[edit_idx]) > edit_tol;
}

template <typename T>
__device__ __forceinline__ bool
hasNonzeroEdit3D(const T *edit_x, const T *edit_y, const T *edit_z,
                 int edit_idx, T edit_tol) {
  return abs_templated(edit_x[edit_idx]) > edit_tol ||
         abs_templated(edit_y[edit_idx]) > edit_tol ||
         abs_templated(edit_z[edit_idx]) > edit_tol;
}

template <typename T>
__global__ void
buildEditMaskFromParticles2D_kernel(HashTable ht, const T *edit_x,
                                    const T *edit_y, bool *edit_mask, int E,
                                    T edit_tol) {
  int edit_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (edit_idx >= E)
    return;
  if (hasNonzeroEdit2D(edit_x, edit_y, edit_idx, edit_tol))
    edit_mask[ht.particles[edit_idx]] = true;
}

template <typename T>
__global__ void
buildEditMaskFromParticles3D_kernel(HashTable ht, const T *edit_x,
                                    const T *edit_y, const T *edit_z,
                                    bool *edit_mask, int E, T edit_tol) {
  int edit_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (edit_idx >= E)
    return;
  if (hasNonzeroEdit3D(edit_x, edit_y, edit_z, edit_idx, edit_tol))
    edit_mask[ht.particles[edit_idx]] = true;
}

template <typename T>
__global__ void
buildEditMaskFromVisitOrder2D_kernel(const int *visit_order, HashTable ht,
                                     const T *edit_x, const T *edit_y,
                                     bool *edit_mask, int N, T edit_tol) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= N)
    return;

  int particle_idx = visit_order[pos];
  int edit_idx = findIndex(ht, particle_idx);
  if (edit_idx >= 0 && hasNonzeroEdit2D(edit_x, edit_y, edit_idx, edit_tol))
    edit_mask[pos] = true;
}

template <typename T>
__global__ void buildEditMaskFromVisitOrder3D_kernel(
    const int *visit_order, HashTable ht, const T *edit_x, const T *edit_y,
    const T *edit_z, bool *edit_mask, int N, T edit_tol) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= N)
    return;

  int particle_idx = visit_order[pos];
  int edit_idx = findIndex(ht, particle_idx);
  if (edit_idx >= 0 &&
      hasNonzeroEdit3D(edit_x, edit_y, edit_z, edit_idx, edit_tol))
    edit_mask[pos] = true;
}

template <typename T>
__global__ void
quantizeMaskedEdits2D_kernel(const T *edit_x, const T *edit_y,
                             const int *visit_order, HashTable editable_pts_ht,
                             const bool *edit_mask, const int *edit_offsets,
                             UInt2 *quant_edits, T xi, T norm, int N) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= N)
    return;
  if (!edit_mask[pos])
    return;

  int particle_idx = visit_order ? visit_order[pos] : pos;
  int edit_idx = findIndex(editable_pts_ht, particle_idx);
  if (edit_idx < 0)
    return;

  int out_idx = edit_offsets[pos];
  quant_edits[2 * out_idx] =
      static_cast<UInt2>((edit_x[edit_idx] + 2 * xi) * norm);
  quant_edits[2 * out_idx + 1] =
      static_cast<UInt2>((edit_y[edit_idx] + 2 * xi) * norm);
}

template <typename T>
__global__ void
quantizeMaskedEdits3D_kernel(const T *edit_x, const T *edit_y, const T *edit_z,
                             const int *visit_order, HashTable editable_pts_ht,
                             const bool *edit_mask, const int *edit_offsets,
                             UInt2 *quant_edits, T xi, T norm, int N) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= N)
    return;
  if (!edit_mask[pos])
    return;

  int particle_idx = visit_order ? visit_order[pos] : pos;
  int edit_idx = findIndex(editable_pts_ht, particle_idx);
  if (edit_idx < 0)
    return;

  int out_idx = edit_offsets[pos];
  quant_edits[3 * out_idx] =
      static_cast<UInt2>((edit_x[edit_idx] + 2 * xi) * norm);
  quant_edits[3 * out_idx + 1] =
      static_cast<UInt2>((edit_y[edit_idx] + 2 * xi) * norm);
  quant_edits[3 * out_idx + 2] =
      static_cast<UInt2>((edit_z[edit_idx] + 2 * xi) * norm);
}

template <typename T>
__global__ void
computeMortonCodes2D_kernel(const T *x, const T *y, uint32_t *morton_codes,
                            int *indices, int n, T min_x, T min_y, T grid_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int i = indices[idx];
    uint32_t ix = static_cast<uint32_t>(
        min_templated(1023.0, (x[i] - min_x) / grid_len * 1024.0));
    uint32_t iy = static_cast<uint32_t>(
        min_templated(1023.0, (y[i] - min_y) / grid_len * 1024.0));

    morton_codes[idx] = morton2D(ix, iy);
  }
}

template <typename T>
__global__ void computeMortonCodes3D_kernel(const T *x, const T *y, const T *z,
                                            uint32_t *morton_codes,
                                            int *indices, int n, T min_x,
                                            T min_y, T min_z, T grid_len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int i = indices[idx];
    uint32_t ix = static_cast<uint32_t>(
        min_templated(1023.0, (x[i] - min_x) / grid_len * 1024.0));
    uint32_t iy = static_cast<uint32_t>(
        min_templated(1023.0, (y[i] - min_y) / grid_len * 1024.0));
    uint32_t iz = static_cast<uint32_t>(
        min_templated(1023.0, (z[i] - min_z) / grid_len * 1024.0));

    morton_codes[idx] = morton3D(ix, iy, iz);
  }
}

template <typename T>
__global__ void computeErrorStatistics2D_kernel(
    const T *d_org_xx, const T *d_org_yy, const T *decomp_x, const T *decomp_y,
    T *mae_buffer, T *mse_buffer, T *max_err_buffer, int N) {

  extern __shared__ unsigned char sdata_raw_err2d[];
  T *sdata = reinterpret_cast<T *>(sdata_raw_err2d);
  T *s_mae = sdata;
  T *s_mse = &sdata[blockDim.x];
  T *s_max = &sdata[2 * blockDim.x];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  T local_mae = 0;
  T local_mse = 0;
  T local_max = 0;

  if (idx < N) {
    T err_x = decomp_x[idx] - d_org_xx[idx];
    T err_y = decomp_y[idx] - d_org_yy[idx];

    T abs_err_x = abs_templated(err_x);
    T abs_err_y = abs_templated(err_y);

    local_max = max_templated(abs_err_x, abs_err_y);
    local_mae = local_max;
    local_mse = err_x * err_x + err_y * err_y;
  }

  s_mae[tid] = local_mae;
  s_mse[tid] = local_mse;
  s_max[tid] = local_max;
  __syncthreads();

  // Reduction for mae and mse (sum)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_mae[tid] += s_mae[tid + s];
      s_mse[tid] += s_mse[tid + s];
      s_max[tid] = max_templated(s_max[tid], s_max[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(mae_buffer, s_mae[0]);
    atomicAdd(mse_buffer, s_mse[0]);

    // Atomic max for double precision
    unsigned long long *addr_as_ull = (unsigned long long *)max_err_buffer;
    unsigned long long old = *addr_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(addr_as_ull, assumed,
                      __double_as_longlong(max_templated(
                          s_max[0], __longlong_as_double(assumed))));
    } while (assumed != old);
  }
}

template <typename T>
__global__ void computeErrorStatistics3D_kernel(
    const T *d_org_xx, const T *d_org_yy, const T *d_org_zz, const T *decomp_x,
    const T *decomp_y, const T *decomp_z, T *mae_buffer, T *mse_buffer,
    T *max_err_buffer, int N) {

  extern __shared__ unsigned char sdata_raw_err3d[];
  T *sdata = reinterpret_cast<T *>(sdata_raw_err3d);
  T *s_mae = sdata;
  T *s_mse = &sdata[blockDim.x];
  T *s_max = &sdata[2 * blockDim.x];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  T local_mae = 0;
  T local_mse = 0;
  T local_max = 0;

  if (idx < N) {
    T err_x = decomp_x[idx] - d_org_xx[idx];
    T err_y = decomp_y[idx] - d_org_yy[idx];
    T err_z = decomp_z[idx] - d_org_zz[idx];

    T abs_err_x = abs_templated(err_x);
    T abs_err_y = abs_templated(err_y);
    T abs_err_z = abs_templated(err_z);

    local_max = max_templated(abs_err_x, max_templated(abs_err_y, abs_err_z));
    local_mae = local_max;
    local_mse = err_x * err_x + err_y * err_y + err_z * err_z;
  }

  s_mae[tid] = local_mae;
  s_mse[tid] = local_mse;
  s_max[tid] = local_max;
  __syncthreads();

  // Reduction for mae and mse (sum)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_mae[tid] += s_mae[tid + s];
      s_mse[tid] += s_mse[tid + s];
      s_max[tid] = max_templated(s_max[tid], s_max[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(mae_buffer, s_mae[0]);
    atomicAdd(mse_buffer, s_mse[0]);

    // Atomic max for double precision
    unsigned long long *addr_as_ull = (unsigned long long *)max_err_buffer;
    unsigned long long old = *addr_as_ull, assumed;
    do {
      assumed = old;
      old = atomicCAS(addr_as_ull, assumed,
                      __double_as_longlong(max_templated(
                          s_max[0], __longlong_as_double(assumed))));
    } while (assumed != old);
  }
}

// CPU versions
template <typename T> inline int quantize(T value, T xi) {
  return static_cast<int>(std::round(value / (2 * xi))) + (1 << (m - 1));
}

template <typename T> inline T dequantize(int quantized, T xi) {
  return static_cast<T>(quantized - (1 << (m - 1))) * 2 * xi;
}

// Kernel to compact full arrays into non-zero arrays
template <typename T>
__global__ void compactCellCompressionOutputs_kernel(
    const bool *d_lossless_flag, const UInt *d_temp_qcode, const T *d_temp_lval,
    const int *d_cell_start, const int *d_quant_offsets,
    const int *d_lossless_offsets, int num_cells, int N, int D,
    UInt *d_quant_codes, T *d_lossless_values) {
  int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell_id >= num_cells)
    return;

  int cell_start = d_cell_start[cell_id];
  int cell_end = (cell_id == num_cells - 1) ? N : d_cell_start[cell_id + 1];
  int qoff = d_quant_offsets[cell_id];
  int loff = d_lossless_offsets[cell_id];

  if (cell_start == cell_end) {
    d_quant_codes[qoff] = static_cast<UInt>(0);
    return;
  }

  for (int pos = cell_start; pos < cell_end; ++pos) {
    for (int coord = 0; coord < D; ++coord) {
      if (d_lossless_flag[D * pos + coord]) {
        d_lossless_values[loff++] = d_temp_lval[D * pos + coord];
      } else {
        d_quant_codes[qoff++] = d_temp_qcode[D * pos + coord];
      }
    }
  }
  d_quant_codes[qoff] = static_cast<UInt>(1 << m); // end-of-cell marker
}

// Kernel to pack lossless flags into bytes (8 flags per byte)
template <int Dummy = 0>
__global__ void packLosslessFlags_kernel(const bool *d_lossless_flag,
                                         uint8_t *d_packed_flags, int num_flags,
                                         int num_bytes) {
  int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (byte_idx >= num_bytes)
    return;

  uint8_t packed = 0;
  int base = byte_idx * 8;
  for (int i = 0; i < 8 && base + i < num_flags; ++i) {
    if (d_lossless_flag[base + i])
      packed |= (1 << (7 - i));
  }
  d_packed_flags[byte_idx] = packed;
}

// Kernel to apply quantized edits to decomp positions (after PGD)
// Iterates over E editable particles only
template <typename T>
__global__ void
applyQuantizedEdits2D_kernel(const UInt2 *d_quant_edits, const HashTable ht,
                             T *d_decomp_xx, T *d_decomp_yy, const T *d_edit_x,
                             const T *d_edit_y, T xi, T dequant_norm, int E) {
  int edit_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (edit_idx >= E)
    return;

  int particle_id = ht.particles[edit_idx];
  T dequant_edit_x =
      static_cast<T>(d_quant_edits[2 * edit_idx]) * dequant_norm - 2 * xi;
  T dequant_edit_y =
      static_cast<T>(d_quant_edits[2 * edit_idx + 1]) * dequant_norm - 2 * xi;
  d_decomp_xx[particle_id] += dequant_edit_x - d_edit_x[edit_idx];
  d_decomp_yy[particle_id] += dequant_edit_y - d_edit_y[edit_idx];
}

template <typename T>
__global__ void
applyQuantizedEdits3D_kernel(const UInt2 *d_quant_edits, const HashTable ht,
                             T *d_decomp_xx, T *d_decomp_yy, T *d_decomp_zz,
                             const T *d_edit_x, const T *d_edit_y,
                             const T *d_edit_z, T xi, T dequant_norm, int E) {
  int edit_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (edit_idx >= E)
    return;

  int particle_id = ht.particles[edit_idx];
  T dequant_edit_x =
      static_cast<T>(d_quant_edits[3 * edit_idx]) * dequant_norm - 2 * xi;
  T dequant_edit_y =
      static_cast<T>(d_quant_edits[3 * edit_idx + 1]) * dequant_norm - 2 * xi;
  T dequant_edit_z =
      static_cast<T>(d_quant_edits[3 * edit_idx + 2]) * dequant_norm - 2 * xi;
  d_decomp_xx[particle_id] += dequant_edit_x - d_edit_x[edit_idx];
  d_decomp_yy[particle_id] += dequant_edit_y - d_edit_y[edit_idx];
  d_decomp_zz[particle_id] += dequant_edit_z - d_edit_z[edit_idx];
}

template <typename T>
__global__ void applyMaskedQuantizedEdits2D_kernel(
    const UInt2 *d_quant_edits, const bool *edit_mask, const int *edit_offsets,
    const int *visit_order, const HashTable ht, T *d_decomp_xx, T *d_decomp_yy,
    const T *d_edit_x, const T *d_edit_y, T xi, T dequant_norm, int N) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= N || !edit_mask[pos])
    return;

  int particle_id = visit_order ? visit_order[pos] : pos;
  int edit_idx = findIndex(ht, particle_id);
  if (edit_idx < 0)
    return;

  int quant_idx = edit_offsets[pos];
  T dequant_edit_x =
      static_cast<T>(d_quant_edits[2 * quant_idx]) * dequant_norm - 2 * xi;
  T dequant_edit_y =
      static_cast<T>(d_quant_edits[2 * quant_idx + 1]) * dequant_norm - 2 * xi;
  d_decomp_xx[particle_id] += dequant_edit_x - d_edit_x[edit_idx];
  d_decomp_yy[particle_id] += dequant_edit_y - d_edit_y[edit_idx];
}

template <typename T>
__global__ void applyMaskedQuantizedEdits3D_kernel(
    const UInt2 *d_quant_edits, const bool *edit_mask, const int *edit_offsets,
    const int *visit_order, const HashTable ht, T *d_decomp_xx, T *d_decomp_yy,
    T *d_decomp_zz, const T *d_edit_x, const T *d_edit_y, const T *d_edit_z,
    T xi, T dequant_norm, int N) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= N || !edit_mask[pos])
    return;

  int particle_id = visit_order ? visit_order[pos] : pos;
  int edit_idx = findIndex(ht, particle_id);
  if (edit_idx < 0)
    return;

  int quant_idx = edit_offsets[pos];
  T dequant_edit_x =
      static_cast<T>(d_quant_edits[3 * quant_idx]) * dequant_norm - 2 * xi;
  T dequant_edit_y =
      static_cast<T>(d_quant_edits[3 * quant_idx + 1]) * dequant_norm - 2 * xi;
  T dequant_edit_z =
      static_cast<T>(d_quant_edits[3 * quant_idx + 2]) * dequant_norm - 2 * xi;
  d_decomp_xx[particle_id] += dequant_edit_x - d_edit_x[edit_idx];
  d_decomp_yy[particle_id] += dequant_edit_y - d_edit_y[edit_idx];
  d_decomp_zz[particle_id] += dequant_edit_z - d_edit_z[edit_idx];
}

// Kernel to count PGD violations on GPU
template <typename T>
__global__ void
countViolations2D_kernel(const int2 *d_vulnerable_pairs, const bool *d_signs,
                         const T *d_decomp_xx, const T *d_decomp_yy, T b_sq,
                         int num_pairs, int *d_tp, int *d_tn, int *d_fp,
                         int *d_fn) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_pairs)
    return;

  int2 pair = d_vulnerable_pairs[k];
  int p = pair.x;
  int q = pair.y;
  T dx = d_decomp_xx[p] - d_decomp_xx[q];
  T dy = d_decomp_yy[p] - d_decomp_yy[q];
  T dist_sq = dx * dx + dy * dy;
  bool sign = d_signs[k];
  bool decomp_sign = dist_sq > b_sq;
  if (sign && decomp_sign) {
    atomicAdd(d_tp, 1);
  } else if (!sign && !decomp_sign) {
    atomicAdd(d_tn, 1);
  } else if (sign && !decomp_sign) {
    atomicAdd(d_fp, 1); // Exist link in decomp data but not in original data
  } else {
    atomicAdd(d_fn, 1);
  }
}

template <typename T>
__global__ void
countViolations3D_kernel(const int2 *d_vulnerable_pairs, const bool *d_signs,
                         const T *d_decomp_xx, const T *d_decomp_yy,
                         const T *d_decomp_zz, T b_sq, int num_pairs, int *d_tp,
                         int *d_tn, int *d_fp, int *d_fn) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= num_pairs)
    return;

  int2 pair = d_vulnerable_pairs[k];
  int p = pair.x;
  int q = pair.y;
  T dx = d_decomp_xx[p] - d_decomp_xx[q];
  T dy = d_decomp_yy[p] - d_decomp_yy[q];
  T dz = d_decomp_zz[p] - d_decomp_zz[q];
  T dist_sq = dx * dx + dy * dy + dz * dz;
  bool sign = d_signs[k];
  bool decomp_sign = dist_sq > b_sq;
  if (sign && decomp_sign) {
    atomicAdd(d_tp, 1);
  } else if (!sign && !decomp_sign) {
    atomicAdd(d_tn, 1);
  } else if (sign && !decomp_sign) {
    atomicAdd(d_fp, 1); // Exist link in decomp data but not in original data
  } else {
    atomicAdd(d_fn, 1);
  }
}

// ============================================================================
// Forward declarations for decompression functions
// (defined in particle_compression.cu with explicit instantiations)
// ============================================================================

template <typename T, OrderMode Mode>
void decompressWithEditParticles2D(const CompressedData<T> &compressed,
                                   T *decomp_xx, T *decomp_yy, int N, T xi,
                                   T b);

template <typename T, OrderMode Mode>
void decompressWithEditParticles3D(const CompressedData<T> &compressed,
                                   T *decomp_xx, T *decomp_yy, T *decomp_zz,
                                   int N, T xi, T b);

template <typename T, OrderMode Mode>
void decompressParticles2D(const CompressedData<T> &compressed, T *decomp_xx,
                           T *decomp_yy, int N, T xi, T b);

template <typename T, OrderMode Mode>
void decompressParticles3D(const CompressedData<T> &compressed, T *decomp_xx,
                           T *decomp_yy, T *decomp_zz, int N, T xi, T b);

template <typename T>
void reconstructEditParticles2D(const CompressedData<T> &compressed,
                                T *decomp_xx, T *decomp_yy, int N, T xi);

template <typename T>
void reconstructEditParticles3D(const CompressedData<T> &compressed,
                                T *decomp_xx, T *decomp_yy, T *decomp_zz, int N,
                                T xi);
// ============================================================================
// Compress & Edit API
// ============================================================================
template <typename T, OrderMode Mode>
void compressWithEditParticles2D(const T *d_org_xx, const T *d_org_yy, T min_x,
                                 T range_x, T min_y, T range_y, int N, T xi,
                                 T b, CompressionState2D<T> &state,
                                 CompressedData<T> &compressed,
                                 int N_local = 0);

template <typename T, OrderMode Mode>
void compressWithEditParticles3D(const T *d_org_xx, const T *d_org_yy,
                                 const T *d_org_zz, T min_x, T range_x, T min_y,
                                 T range_y, T min_z, T range_z, int N, T xi,
                                 T b, CompressionState3D<T> &state,
                                 CompressedData<T> &compressed,
                                 int N_local = 0);

// Compression-only (no PGD edits) - GPU versions
template <typename T, OrderMode Mode>
void compressParticles2D(const T *d_org_xx, const T *d_org_yy, T min_x,
                         T range_x, T min_y, T range_y, int N, T xi, T b,
                         CompressionState2D<T> &state,
                         CompressedData<T> &compressed);

template <typename T, OrderMode Mode>
void compressParticles3D(const T *d_org_xx, const T *d_org_yy,
                         const T *d_org_zz, T min_x, T range_x, T min_y,
                         T range_y, T min_z, T range_z, int N, T xi, T b,
                         CompressionState3D<T> &state,
                         CompressedData<T> &compressed);

// Edit-only (PGD or lossless edit) - GPU versions
template <typename T, OrderMode Mode>
void editParticles2D(const T *d_org_xx, const T *d_org_yy, T *d_base_decomp_xx,
                     T *d_base_decomp_yy, T min_x, T range_x, T min_y,
                     T range_y, int N, T xi, T b, CompressionState2D<T> &state,
                     CompressedData<T> &compressed, int N_local = 0,
                     MPI_Comm comm FOFPZ_MPI_COMM_DEFAULT);

template <typename T, OrderMode Mode>
void editParticles3D(const T *d_org_xx, const T *d_org_yy, const T *d_org_zz,
                     T *d_base_decomp_xx, T *d_base_decomp_yy,
                     T *d_base_decomp_zz, T min_x, T range_x, T min_y,
                     T range_y, T min_z, T range_z, int N, T xi, T b,
                     CompressionState3D<T> &state,
                     CompressedData<T> &compressed, int N_local = 0,
                     MPI_Comm comm FOFPZ_MPI_COMM_DEFAULT);

// ============================================================================
// Getter functions: copy device data to host on demand
// ============================================================================
template <typename T>
void getDecompressedCoords2D(const CompressionState2D<T> &state, T *h_decomp_xx,
                             T *h_decomp_yy);

template <typename T>
void getDecompressedCoords3D(const CompressionState3D<T> &state, T *h_decomp_xx,
                             T *h_decomp_yy, T *h_decomp_zz);

template <typename T>
void getVisitOrder(const CompressionState2D<T> &state, int *h_visit_order);

template <typename T>
void getVisitOrder(const CompressionState3D<T> &state, int *h_visit_order);
