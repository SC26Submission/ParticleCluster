#pragma once

#include "util.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <numeric>
#include <thread>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// =============================================================================
// Data structures
// =============================================================================

struct BoundingBox {
  double lo[3];
  double hi[3];
};

struct DistributedContext {
  int rank;
  int size;
  std::vector<BoundingBox> local_bboxes;  // one per spatial cluster
  std::vector<BoundingBox> padded_bboxes; // each expanded by ghost_width
  std::vector<int> neighbors;
  // Cached per-rank padded bboxes from Allgatherv (used by exchangeGhosts)
  std::vector<std::vector<BoundingBox>> all_padded_per_rank;
};

template <typename T> struct GhostBuffer {
  int source_rank;
  size_t count;
  std::vector<T> xx, yy, zz; // host-side ghost data
};

// =============================================================================
// BoundingBox serialization helpers
// =============================================================================

inline std::vector<double> flattenBBoxes(const std::vector<BoundingBox> &boxes) {
  std::vector<double> flat(boxes.size() * 6);
  for (size_t i = 0; i < boxes.size(); ++i) {
    for (int d = 0; d < 3; ++d) {
      flat[i * 6 + d] = boxes[i].lo[d];
      flat[i * 6 + 3 + d] = boxes[i].hi[d];
    }
  }
  return flat;
}

inline std::vector<BoundingBox> unflattenBBoxes(const double *flat, int count) {
  std::vector<BoundingBox> boxes(count);
  for (int i = 0; i < count; ++i) {
    for (int d = 0; d < 3; ++d) {
      boxes[i].lo[d] = flat[i * 6 + d];
      boxes[i].hi[d] = flat[i * 6 + 3 + d];
    }
  }
  return boxes;
}

// A1: Fused serialization — tight + padded in one buffer (12 doubles per box)
inline std::vector<double>
flattenBBoxesFused(const std::vector<BoundingBox> &tight,
                   const std::vector<BoundingBox> &padded) {
  size_t n = tight.size();
  std::vector<double> flat(n * 12);
  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d < 3; ++d) {
      flat[i * 12 + d] = tight[i].lo[d];
      flat[i * 12 + 3 + d] = tight[i].hi[d];
      flat[i * 12 + 6 + d] = padded[i].lo[d];
      flat[i * 12 + 9 + d] = padded[i].hi[d];
    }
  }
  return flat;
}

inline void unflattenBBoxesFused(const double *flat, int count,
                                 std::vector<BoundingBox> &tight,
                                 std::vector<BoundingBox> &padded) {
  tight.resize(count);
  padded.resize(count);
  for (int i = 0; i < count; ++i) {
    for (int d = 0; d < 3; ++d) {
      tight[i].lo[d] = flat[i * 12 + d];
      tight[i].hi[d] = flat[i * 12 + 3 + d];
      padded[i].lo[d] = flat[i * 12 + 6 + d];
      padded[i].hi[d] = flat[i * 12 + 9 + d];
    }
  }
}

inline bool boxesOverlap(const BoundingBox &a, const BoundingBox &b, int D) {
  for (int d = 0; d < D; ++d) {
    if (a.hi[d] < b.lo[d] || a.lo[d] > b.hi[d])
      return false;
  }
  return true;
}

inline bool anyBoxOverlaps(const std::vector<BoundingBox> &padded_set,
                           const std::vector<BoundingBox> &tight_set, int D) {
  for (const auto &p : padded_set) {
    for (const auto &t : tight_set) {
      if (boxesOverlap(p, t, D))
        return true;
    }
  }
  return false;
}

// =============================================================================
// Spatial cluster detection (GPU: thrust::sort, host gap detection)
// =============================================================================

template <typename T>
std::vector<BoundingBox> detectSpatialClusters_GPU(const T *d_xx, const T *d_yy,
                                                   const T *d_zz, int N,
                                                   double ghost_width, int D) {
  if (N == 0)
    return {};

  const T *d_coords[3] = {d_xx, d_yy, d_zz};
  std::vector<std::vector<std::pair<double, double>>> intervals(D);

  // #2: Run per-dimension sorts concurrently using threads
  auto sortDimGPU = [&](int dim) {
    thrust::device_vector<T> d_sorted(d_coords[dim], d_coords[dim] + N);
    thrust::sort(d_sorted.begin(), d_sorted.end());

    std::vector<T> h_sorted(N);
    thrust::copy(d_sorted.begin(), d_sorted.end(), h_sorted.begin());

    double lo = h_sorted[0];
    for (int i = 1; i < N; ++i) {
      if (h_sorted[i] - h_sorted[i - 1] > ghost_width) {
        intervals[dim].push_back({lo, (double)h_sorted[i - 1]});
        lo = h_sorted[i];
      }
    }
    intervals[dim].push_back({lo, (double)h_sorted[N - 1]});
  };

  // Each dimension writes to its own intervals[dim], so no data race
  std::vector<std::thread> sort_threads;
  for (int dim = 0; dim < D; ++dim)
    sort_threads.emplace_back(sortDimGPU, dim);
  for (auto &t : sort_threads)
    t.join();

  std::vector<BoundingBox> candidates;
  if (D == 2) {
    for (const auto &ix : intervals[0]) {
      for (const auto &iy : intervals[1]) {
        BoundingBox box;
        box.lo[0] = ix.first;
        box.hi[0] = ix.second;
        box.lo[1] = iy.first;
        box.hi[1] = iy.second;
        box.lo[2] = 0.0;
        box.hi[2] = 0.0;
        candidates.push_back(box);
      }
    }
  } else {
    for (const auto &ix : intervals[0]) {
      for (const auto &iy : intervals[1]) {
        for (const auto &iz : intervals[2]) {
          BoundingBox box;
          box.lo[0] = ix.first;
          box.hi[0] = ix.second;
          box.lo[1] = iy.first;
          box.hi[1] = iy.second;
          box.lo[2] = iz.first;
          box.hi[2] = iz.second;
          candidates.push_back(box);
        }
      }
    }
  }

  // #1: Hash-based O(N) occupancy via binary search on interval boundaries
  std::vector<T> h_xx(N), h_yy(N), h_zz;
  CUDA_CHECK(
      cudaMemcpy(h_xx.data(), d_xx, N * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(h_yy.data(), d_yy, N * sizeof(T), cudaMemcpyDeviceToHost));
  if (D >= 3 && d_zz) {
    h_zz.resize(N);
    CUDA_CHECK(
        cudaMemcpy(h_zz.data(), d_zz, N * sizeof(T), cudaMemcpyDeviceToHost));
  }

  std::vector<std::vector<double>> lo_bounds(D);
  for (int dim = 0; dim < D; ++dim) {
    lo_bounds[dim].reserve(intervals[dim].size());
    for (const auto &iv : intervals[dim])
      lo_bounds[dim].push_back(iv.first);
  }

  std::vector<size_t> interval_dims(D);
  for (int dim = 0; dim < D; ++dim)
    interval_dims[dim] = intervals[dim].size();

  std::vector<bool> occupied(candidates.size(), false);
  size_t num_occupied = 0;

  for (int i = 0; i < N && num_occupied < candidates.size(); ++i) {
    double pt[3] = {(double)h_xx[i], (double)h_yy[i],
                     (D >= 3 && !h_zz.empty()) ? (double)h_zz[i] : 0.0};
    size_t cidx = 0;
    size_t stride = 1;
    bool found = true;
    for (int dim = D - 1; dim >= 0; --dim) {
      auto &lb = lo_bounds[dim];
      auto it = std::upper_bound(lb.begin(), lb.end(), pt[dim]);
      if (it == lb.begin()) { found = false; break; }
      size_t idx = static_cast<size_t>(std::distance(lb.begin(), it) - 1);
      if (pt[dim] > intervals[dim][idx].second) { found = false; break; }
      cidx += idx * stride;
      stride *= interval_dims[dim];
    }
    if (found && !occupied[cidx]) {
      occupied[cidx] = true;
      num_occupied++;
    }
  }

  std::vector<BoundingBox> result;
  for (size_t c = 0; c < candidates.size(); ++c) {
    if (occupied[c])
      result.push_back(candidates[c]);
  }

  return result;
}

// =============================================================================
// Bounding box utilities
// =============================================================================

template <typename T>
void computeLocalBBox2D_GPU(const T *d_xx, const T *d_yy, int N,
                            BoundingBox &bbox) {
  T min_x, max_x, range_x, min_y, max_y, range_y;
  getRange(d_xx, N, min_x, max_x, range_x);
  getRange(d_yy, N, min_y, max_y, range_y);
  bbox.lo[0] = min_x;
  bbox.hi[0] = max_x;
  bbox.lo[1] = min_y;
  bbox.hi[1] = max_y;
  bbox.lo[2] = 0.0;
  bbox.hi[2] = 0.0;
}

template <typename T>
void computeLocalBBox3D_GPU(const T *d_xx, const T *d_yy, const T *d_zz,
                            int N, BoundingBox &bbox) {
  T min_x, max_x, range_x, min_y, max_y, range_y, min_z, max_z, range_z;
  getRange(d_xx, N, min_x, max_x, range_x);
  getRange(d_yy, N, min_y, max_y, range_y);
  getRange(d_zz, N, min_z, max_z, range_z);
  bbox.lo[0] = min_x;
  bbox.hi[0] = max_x;
  bbox.lo[1] = min_y;
  bbox.hi[1] = max_y;
  bbox.lo[2] = min_z;
  bbox.hi[2] = max_z;
}

// =============================================================================
// D1: Spatial hash for neighbor discovery
// =============================================================================

struct CellKey {
  int x, y, z;
  bool operator==(const CellKey &o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct CellKeyHash {
  size_t operator()(const CellKey &c) const {
    size_t h = std::hash<int>{}(c.x);
    h ^= std::hash<int>{}(c.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(c.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

// =============================================================================
// A1 + D1: Optimized neighbor discovery
// =============================================================================

inline void discoverNeighbors(DistributedContext &ctx, double ghost_width,
                              int D, MPI_Comm comm) {
  // One cluster per rank: pad the single local bbox
  ctx.padded_bboxes.resize(1);
  for (int d = 0; d < D; ++d) {
    ctx.padded_bboxes[0].lo[d] = ctx.local_bboxes[0].lo[d] - ghost_width;
    ctx.padded_bboxes[0].hi[d] = ctx.local_bboxes[0].hi[d] + ghost_width;
  }
  for (int d = D; d < 3; ++d) {
    ctx.padded_bboxes[0].lo[d] = 0.0;
    ctx.padded_bboxes[0].hi[d] = 0.0;
  }

  // One bbox per rank: use Allgather (12 doubles per rank: tight + padded fused)
  std::vector<double> local_fused =
      flattenBBoxesFused(ctx.local_bboxes, ctx.padded_bboxes);  // 12 doubles
  std::vector<double> all_fused_flat(ctx.size * 12);
  MPI_Allgather(local_fused.data(), 12, MPI_DOUBLE,
                all_fused_flat.data(), 12, MPI_DOUBLE, comm);

  // Reconstruct per-rank tight and padded bboxes
  std::vector<std::vector<BoundingBox>> all_tight(ctx.size);
  ctx.all_padded_per_rank.resize(ctx.size);
  for (int r = 0; r < ctx.size; ++r)
    unflattenBBoxesFused(all_fused_flat.data() + r * 12, 1,
                         all_tight[r], ctx.all_padded_per_rank[r]);

  // O(ranks^2) neighbor check — fast for <= hundreds of ranks
  ctx.neighbors.clear();
  for (int r = 0; r < ctx.size; ++r) {
    if (r == ctx.rank)
      continue;
    if (anyBoxOverlaps(ctx.padded_bboxes, all_tight[r], D))
      ctx.neighbors.push_back(r);
  }
  std::sort(ctx.neighbors.begin(), ctx.neighbors.end());

  if (ctx.rank == 0)
    printf("Rank 0: %zu neighbors\n", ctx.neighbors.size());
}

// =============================================================================
// C2: GPU-side ghost identification kernels
// =============================================================================

// Each thread tests one particle against the neighbor's padded bounding boxes.
// hull = [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z] for fast rejection.
// boxes = num_boxes * 6 doubles, each [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z].
template <typename T>
__global__ void ghostIdentifyKernel(const T *xx, const T *yy, const T *zz,
                                    int N_local, const double *hull,
                                    const double *boxes, int num_boxes, int D,
                                    int *out_indices, int *out_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N_local)
    return;

  double px = (double)xx[i];
  double py = (double)yy[i];
  double pz = (D >= 3 && zz) ? (double)zz[i] : 0.0;

  // Hull rejection
  if (px < hull[0] || px > hull[3])
    return;
  if (py < hull[1] || py > hull[4])
    return;
  if (D >= 3 && (pz < hull[2] || pz > hull[5]))
    return;

  // Fine-grained box test
  for (int b = 0; b < num_boxes; ++b) {
    const double *box = boxes + b * 6;
    bool inside = true;
    if (px < box[0] || px > box[3])
      inside = false;
    else if (py < box[1] || py > box[4])
      inside = false;
    else if (D >= 3 && (pz < box[2] || pz > box[5]))
      inside = false;
    if (inside) {
      int pos = atomicAdd(out_count, 1);
      out_indices[pos] = i;
      return;
    }
  }
}

// A3: Gather selected particles into packed SoA buffer [xx_0..n | yy_0..n | zz_0..n]
template <typename T>
__global__ void gatherPackedKernel(const T *xx, const T *yy, const T *zz,
                                   const int *indices, int count, int D,
                                   T *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int idx = indices[i];
  out[i] = xx[idx];
  out[count + i] = yy[idx];
  if (D >= 3)
    out[2 * count + i] = zz[idx];
}

// =============================================================================
// Ghost exchange structures
// =============================================================================

template <typename T> struct PendingGhostExchange {
  std::vector<MPI_Request> reqs;
  int num_send_reqs = 0; // B2: first N reqs are sends, rest are recvs

  // A3: Packed send buffers (must stay alive until complete)
  std::vector<std::vector<T>> send_packed;
#ifdef USE_CUDA_AWARE_MPI
  std::vector<T *> send_device_bufs; // device buffers for CUDA-aware sends
#endif

  // A3: Packed recv buffers (unpacked to GhostBuffer on completion)
  std::vector<std::vector<T>> recv_packed;

  // Saved host-side send indices per neighbor (for base data re-exchange)
  std::vector<std::vector<int>> send_indices;

  int rank;
  size_t N_local;
  int D;
};

// =============================================================================
// C2 + A3 + B1: Optimized ghost exchange (GPU ghost ID, packed, pipelined)
// =============================================================================

template <typename T>
PendingGhostExchange<T>
beginGhostExchange3D(const T *d_xx, const T *d_yy, const T *d_zz,
                     size_t N_local, const DistributedContext &ctx,
                     std::vector<GhostBuffer<T>> &recv_ghosts,
                     size_t &total_ghost_count, MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  PendingGhostExchange<T> pending;
  pending.rank = ctx.rank;
  pending.N_local = N_local;
  pending.D = 3;
  pending.send_packed.resize(num_neighbors);
  pending.send_indices.resize(num_neighbors);

  MPI_Datatype mpi_T =
      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  // C2: Allocate reusable device buffers for GPU ghost identification
  int *d_indices = nullptr, *d_count = nullptr;
  double *d_hull = nullptr, *d_boxes = nullptr;
  size_t max_boxes = 0;
  for (int n = 0; n < num_neighbors; ++n)
    max_boxes =
        std::max(max_boxes, ctx.all_padded_per_rank[ctx.neighbors[n]].size());

  CUDA_CHECK(cudaMalloc(&d_indices, N_local * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_hull, 6 * sizeof(double)));
  if (max_boxes > 0)
    CUDA_CHECK(cudaMalloc(&d_boxes, max_boxes * 6 * sizeof(double)));

  // B1: Pipeline — identify, pack, and send for each neighbor
  std::vector<size_t> send_counts(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    const auto &neighbor_boxes = ctx.all_padded_per_rank[ctx.neighbors[n]];
    int num_boxes = static_cast<int>(neighbor_boxes.size());

    if (num_boxes == 0) {
      send_counts[n] = 0;
      continue;
    }

    // Compute hull on host
    double hull[6]; // lo_x, lo_y, lo_z, hi_x, hi_y, hi_z
    hull[0] = neighbor_boxes[0].lo[0];
    hull[1] = neighbor_boxes[0].lo[1];
    hull[2] = neighbor_boxes[0].lo[2];
    hull[3] = neighbor_boxes[0].hi[0];
    hull[4] = neighbor_boxes[0].hi[1];
    hull[5] = neighbor_boxes[0].hi[2];
    for (int b = 1; b < num_boxes; ++b) {
      for (int d = 0; d < 3; ++d) {
        hull[d] = std::min(hull[d], neighbor_boxes[b].lo[d]);
        hull[3 + d] = std::max(hull[3 + d], neighbor_boxes[b].hi[d]);
      }
    }

    // Flatten boxes to [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z] per box
    std::vector<double> boxes_flat(num_boxes * 6);
    for (int b = 0; b < num_boxes; ++b) {
      for (int d = 0; d < 3; ++d) {
        boxes_flat[b * 6 + d] = neighbor_boxes[b].lo[d];
        boxes_flat[b * 6 + 3 + d] = neighbor_boxes[b].hi[d];
      }
    }

    // Upload hull and boxes to device
    CUDA_CHECK(
        cudaMemcpy(d_hull, hull, 6 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boxes, boxes_flat.data(),
                          num_boxes * 6 * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    // Launch ghost identification kernel
    int block = 256;
    int grid = (static_cast<int>(N_local) + block - 1) / block;
    ghostIdentifyKernel<T><<<grid, block>>>(d_xx, d_yy, d_zz,
                                            static_cast<int>(N_local), d_hull,
                                            d_boxes, num_boxes, 3, d_indices,
                                            d_count);

    // Read back count
    int count = 0;
    CUDA_CHECK(
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    send_counts[n] = count;

    if (count > 0) {
      // Save host-side send indices for base data re-exchange
      pending.send_indices[n].resize(count);
      CUDA_CHECK(cudaMemcpy(pending.send_indices[n].data(), d_indices,
                            count * sizeof(int), cudaMemcpyDeviceToHost));

      // A3: Pack on device into SoA buffer
      T *d_pack_buf;
      CUDA_CHECK(cudaMalloc(&d_pack_buf, (size_t)count * 3 * sizeof(T)));
      int gblock = 256;
      int ggrid = (count + gblock - 1) / gblock;
      gatherPackedKernel<T>
          <<<ggrid, gblock>>>(d_xx, d_yy, d_zz, d_indices, count, 3, d_pack_buf);

#ifdef USE_CUDA_AWARE_MPI
      // C1: Send directly from device
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Request req;
      MPI_Isend(d_pack_buf, count * 3, mpi_T, ctx.neighbors[n], 1, comm, &req);
      pending.reqs.push_back(req);
      pending.num_send_reqs++;
      pending.send_device_bufs.push_back(d_pack_buf);
#else
      // Copy to host, then send
      pending.send_packed[n].resize((size_t)count * 3);
      CUDA_CHECK(cudaMemcpy(pending.send_packed[n].data(), d_pack_buf,
                            (size_t)count * 3 * sizeof(T),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_pack_buf));

      // B1: Post non-blocking send (returns immediately, next neighbor starts)
      MPI_Request req;
      MPI_Isend(pending.send_packed[n].data(), count * 3, mpi_T,
                ctx.neighbors[n], 1, comm, &req);
      pending.reqs.push_back(req);
      pending.num_send_reqs++;
#endif
    }
  }

  // Free reusable device buffers
  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_hull));
  if (max_boxes > 0)
    CUDA_CHECK(cudaFree(d_boxes));

  // Exchange counts (blocking, small)
  std::vector<MPI_Request> count_reqs;
  std::vector<size_t> recv_counts(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    MPI_Request req;
    MPI_Isend(&send_counts[n], 1, MPI_UNSIGNED_LONG, ctx.neighbors[n], 0, comm,
              &req);
    count_reqs.push_back(req);
    MPI_Irecv(&recv_counts[n], 1, MPI_UNSIGNED_LONG, ctx.neighbors[n], 0, comm,
              &req);
    count_reqs.push_back(req);
  }
  MPI_Waitall(static_cast<int>(count_reqs.size()), count_reqs.data(),
              MPI_STATUSES_IGNORE);

  // Allocate recv GhostBuffers and packed recv buffers
  recv_ghosts.resize(num_neighbors);
  pending.recv_packed.resize(num_neighbors);
  total_ghost_count = 0;
  for (int n = 0; n < num_neighbors; ++n) {
    recv_ghosts[n].source_rank = ctx.neighbors[n];
    recv_ghosts[n].count = recv_counts[n];
    recv_ghosts[n].xx.resize(recv_counts[n]);
    recv_ghosts[n].yy.resize(recv_counts[n]);
    recv_ghosts[n].zz.resize(recv_counts[n]);
    total_ghost_count += recv_counts[n];

    // A3: Single packed recv buffer per neighbor
    if (recv_counts[n] > 0)
      pending.recv_packed[n].resize(recv_counts[n] * 3);
  }

  // A3: Post non-blocking receives (1 Irecv per neighbor instead of 3)
  for (int n = 0; n < num_neighbors; ++n) {
    if (recv_counts[n] > 0) {
      MPI_Request req;
      MPI_Irecv(pending.recv_packed[n].data(),
                static_cast<int>(recv_counts[n] * 3), mpi_T, ctx.neighbors[n],
                1, comm, &req);
      pending.reqs.push_back(req);
    }
  }

  return pending;
}

template <typename T>
PendingGhostExchange<T>
beginGhostExchange2D(const T *d_xx, const T *d_yy, size_t N_local,
                     const DistributedContext &ctx,
                     std::vector<GhostBuffer<T>> &recv_ghosts,
                     size_t &total_ghost_count, MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  PendingGhostExchange<T> pending;
  pending.rank = ctx.rank;
  pending.N_local = N_local;
  pending.D = 2;
  pending.send_packed.resize(num_neighbors);
  pending.send_indices.resize(num_neighbors);

  MPI_Datatype mpi_T =
      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  // C2: Allocate reusable device buffers
  int *d_indices = nullptr, *d_count = nullptr;
  double *d_hull = nullptr, *d_boxes = nullptr;
  size_t max_boxes = 0;
  for (int n = 0; n < num_neighbors; ++n)
    max_boxes =
        std::max(max_boxes, ctx.all_padded_per_rank[ctx.neighbors[n]].size());

  CUDA_CHECK(cudaMalloc(&d_indices, N_local * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_hull, 6 * sizeof(double)));
  if (max_boxes > 0)
    CUDA_CHECK(cudaMalloc(&d_boxes, max_boxes * 6 * sizeof(double)));

  std::vector<size_t> send_counts(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    const auto &neighbor_boxes = ctx.all_padded_per_rank[ctx.neighbors[n]];
    int num_boxes = static_cast<int>(neighbor_boxes.size());

    if (num_boxes == 0) {
      send_counts[n] = 0;
      continue;
    }

    double hull[6];
    hull[0] = neighbor_boxes[0].lo[0];
    hull[1] = neighbor_boxes[0].lo[1];
    hull[2] = 0.0;
    hull[3] = neighbor_boxes[0].hi[0];
    hull[4] = neighbor_boxes[0].hi[1];
    hull[5] = 0.0;
    for (int b = 1; b < num_boxes; ++b) {
      for (int d = 0; d < 2; ++d) {
        hull[d] = std::min(hull[d], neighbor_boxes[b].lo[d]);
        hull[3 + d] = std::max(hull[3 + d], neighbor_boxes[b].hi[d]);
      }
    }

    std::vector<double> boxes_flat(num_boxes * 6);
    for (int b = 0; b < num_boxes; ++b) {
      for (int d = 0; d < 3; ++d) {
        boxes_flat[b * 6 + d] = neighbor_boxes[b].lo[d];
        boxes_flat[b * 6 + 3 + d] = neighbor_boxes[b].hi[d];
      }
    }

    CUDA_CHECK(
        cudaMemcpy(d_hull, hull, 6 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boxes, boxes_flat.data(),
                          num_boxes * 6 * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int block = 256;
    int grid = (static_cast<int>(N_local) + block - 1) / block;
    ghostIdentifyKernel<T><<<grid, block>>>(
        d_xx, d_yy, (const T *)nullptr, static_cast<int>(N_local), d_hull,
        d_boxes, num_boxes, 2, d_indices, d_count);

    int count = 0;
    CUDA_CHECK(
        cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    send_counts[n] = count;

    if (count > 0) {
      // Save host-side send indices for base data re-exchange
      pending.send_indices[n].resize(count);
      CUDA_CHECK(cudaMemcpy(pending.send_indices[n].data(), d_indices,
                            count * sizeof(int), cudaMemcpyDeviceToHost));

      T *d_pack_buf;
      CUDA_CHECK(cudaMalloc(&d_pack_buf, (size_t)count * 2 * sizeof(T)));
      int gblock = 256;
      int ggrid = (count + gblock - 1) / gblock;
      gatherPackedKernel<T><<<ggrid, gblock>>>(
          d_xx, d_yy, (const T *)nullptr, d_indices, count, 2, d_pack_buf);

#ifdef USE_CUDA_AWARE_MPI
      CUDA_CHECK(cudaDeviceSynchronize());
      MPI_Request req;
      MPI_Isend(d_pack_buf, count * 2, mpi_T, ctx.neighbors[n], 1, comm, &req);
      pending.reqs.push_back(req);
      pending.num_send_reqs++;
      pending.send_device_bufs.push_back(d_pack_buf);
#else
      pending.send_packed[n].resize((size_t)count * 2);
      CUDA_CHECK(cudaMemcpy(pending.send_packed[n].data(), d_pack_buf,
                            (size_t)count * 2 * sizeof(T),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_pack_buf));

      MPI_Request req;
      MPI_Isend(pending.send_packed[n].data(), count * 2, mpi_T,
                ctx.neighbors[n], 1, comm, &req);
      pending.reqs.push_back(req);
      pending.num_send_reqs++;
#endif
    }
  }

  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_hull));
  if (max_boxes > 0)
    CUDA_CHECK(cudaFree(d_boxes));

  // Exchange counts
  std::vector<MPI_Request> count_reqs;
  std::vector<size_t> recv_counts(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    MPI_Request req;
    MPI_Isend(&send_counts[n], 1, MPI_UNSIGNED_LONG, ctx.neighbors[n], 0, comm,
              &req);
    count_reqs.push_back(req);
    MPI_Irecv(&recv_counts[n], 1, MPI_UNSIGNED_LONG, ctx.neighbors[n], 0, comm,
              &req);
    count_reqs.push_back(req);
  }
  MPI_Waitall(static_cast<int>(count_reqs.size()), count_reqs.data(),
              MPI_STATUSES_IGNORE);

  recv_ghosts.resize(num_neighbors);
  pending.recv_packed.resize(num_neighbors);
  total_ghost_count = 0;
  for (int n = 0; n < num_neighbors; ++n) {
    recv_ghosts[n].source_rank = ctx.neighbors[n];
    recv_ghosts[n].count = recv_counts[n];
    recv_ghosts[n].xx.resize(recv_counts[n]);
    recv_ghosts[n].yy.resize(recv_counts[n]);
    total_ghost_count += recv_counts[n];

    if (recv_counts[n] > 0)
      pending.recv_packed[n].resize(recv_counts[n] * 2);
  }

  for (int n = 0; n < num_neighbors; ++n) {
    if (recv_counts[n] > 0) {
      MPI_Request req;
      MPI_Irecv(pending.recv_packed[n].data(),
                static_cast<int>(recv_counts[n] * 2), mpi_T, ctx.neighbors[n],
                1, comm, &req);
      pending.reqs.push_back(req);
    }
  }

  return pending;
}

// =============================================================================
// B2: Progressive ghost exchange completion with MPI_Waitsome
// =============================================================================

template <typename T>
void completeGhostExchange(PendingGhostExchange<T> &pending,
                           std::vector<GhostBuffer<T>> &recv_ghosts,
                           size_t total_ghost_count) {
  if (pending.reqs.empty()) {
    printf("Rank %d: %zu local, %zu ghost particles\n", pending.rank,
           pending.N_local, total_ghost_count);
    return;
  }

  int num_reqs = static_cast<int>(pending.reqs.size());

  // B2: Use MPI_Waitsome for progressive completion
  std::vector<int> completed(num_reqs);
  std::vector<MPI_Status> statuses(num_reqs);
  int remaining = num_reqs;

  // Build recv request index -> neighbor index mapping
  // Recv requests start at index num_send_reqs in the reqs vector.
  // But not every neighbor has a recv (only those with recv_count > 0).
  std::vector<int> recv_req_to_neighbor;
  for (size_t n = 0; n < recv_ghosts.size(); ++n) {
    if (recv_ghosts[n].count > 0)
      recv_req_to_neighbor.push_back(static_cast<int>(n));
  }

  while (remaining > 0) {
    int outcount;
    MPI_Waitsome(num_reqs, pending.reqs.data(), &outcount, completed.data(),
                 statuses.data());
    if (outcount == MPI_UNDEFINED)
      break;

    for (int i = 0; i < outcount; ++i) {
      int idx = completed[i];
      // Check if this is a recv request
      if (idx >= pending.num_send_reqs) {
        int recv_idx = idx - pending.num_send_reqs;
        int neighbor_idx = recv_req_to_neighbor[recv_idx];
        auto &g = recv_ghosts[neighbor_idx];
        const T *packed = pending.recv_packed[neighbor_idx].data();
        size_t cnt = g.count;

        // Unpack SoA packed buffer -> separate xx/yy/zz
        std::memcpy(g.xx.data(), packed, cnt * sizeof(T));
        std::memcpy(g.yy.data(), packed + cnt, cnt * sizeof(T));
        if (pending.D >= 3)
          std::memcpy(g.zz.data(), packed + 2 * cnt, cnt * sizeof(T));

        // Free packed recv buffer immediately
        pending.recv_packed[neighbor_idx].clear();
        pending.recv_packed[neighbor_idx].shrink_to_fit();
      }
    }
    remaining -= outcount;
  }

#ifdef USE_CUDA_AWARE_MPI
  // Free device send buffers
  for (auto *p : pending.send_device_bufs)
    cudaFree(p);
  pending.send_device_bufs.clear();
#endif

  printf("Rank %d: %zu local, %zu ghost particles\n", pending.rank,
         pending.N_local, total_ghost_count);

  pending.send_packed.clear();
  pending.recv_packed.clear();
  pending.reqs.clear();
}

// =============================================================================
// Re-exchange ghost data using saved send indices (for base decompressed data)
// Guarantees identical ghost particle ordering/count as the original exchange.
// =============================================================================

template <typename T>
void reexchangeGhostData3D(const std::vector<std::vector<int>> &send_indices,
                            const T *h_xx, const T *h_yy, const T *h_zz,
                            const std::vector<GhostBuffer<T>> &orig_ghosts,
                            const DistributedContext &ctx,
                            std::vector<GhostBuffer<T>> &new_ghosts,
                            MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  MPI_Datatype mpi_T = std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  // Pack send buffers on host using saved indices
  std::vector<std::vector<T>> send_bufs(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = send_indices[n].size();
    send_bufs[n].resize(cnt * 3);
    for (size_t i = 0; i < cnt; ++i) {
      int idx = send_indices[n][i];
      send_bufs[n][i]           = h_xx[idx];
      send_bufs[n][cnt + i]     = h_yy[idx];
      send_bufs[n][2 * cnt + i] = h_zz[idx];
    }
  }

  // Allocate recv buffers matching original ghost counts
  new_ghosts.resize(num_neighbors);
  std::vector<std::vector<T>> recv_bufs(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    new_ghosts[n].source_rank = ctx.neighbors[n];
    new_ghosts[n].count = orig_ghosts[n].count;
    new_ghosts[n].xx.resize(orig_ghosts[n].count);
    new_ghosts[n].yy.resize(orig_ghosts[n].count);
    new_ghosts[n].zz.resize(orig_ghosts[n].count);
    if (orig_ghosts[n].count > 0)
      recv_bufs[n].resize(orig_ghosts[n].count * 3);
  }

  std::vector<MPI_Request> reqs;
  for (int n = 0; n < num_neighbors; ++n) {
    if (!send_bufs[n].empty()) {
      MPI_Request req;
      MPI_Isend(send_bufs[n].data(), static_cast<int>(send_bufs[n].size()),
                mpi_T, ctx.neighbors[n], 2, comm, &req);
      reqs.push_back(req);
    }
    if (orig_ghosts[n].count > 0) {
      MPI_Request req;
      MPI_Irecv(recv_bufs[n].data(),
                static_cast<int>(orig_ghosts[n].count * 3),
                mpi_T, ctx.neighbors[n], 2, comm, &req);
      reqs.push_back(req);
    }
  }
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = orig_ghosts[n].count;
    if (cnt > 0) {
      std::memcpy(new_ghosts[n].xx.data(), recv_bufs[n].data(),           cnt * sizeof(T));
      std::memcpy(new_ghosts[n].yy.data(), recv_bufs[n].data() + cnt,     cnt * sizeof(T));
      std::memcpy(new_ghosts[n].zz.data(), recv_bufs[n].data() + 2 * cnt, cnt * sizeof(T));
    }
  }
}

template <typename T>
void reexchangeGhostData2D(const std::vector<std::vector<int>> &send_indices,
                            const T *h_xx, const T *h_yy,
                            const std::vector<GhostBuffer<T>> &orig_ghosts,
                            const DistributedContext &ctx,
                            std::vector<GhostBuffer<T>> &new_ghosts,
                            MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  MPI_Datatype mpi_T = std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  std::vector<std::vector<T>> send_bufs(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = send_indices[n].size();
    send_bufs[n].resize(cnt * 2);
    for (size_t i = 0; i < cnt; ++i) {
      int idx = send_indices[n][i];
      send_bufs[n][i]       = h_xx[idx];
      send_bufs[n][cnt + i] = h_yy[idx];
    }
  }

  new_ghosts.resize(num_neighbors);
  std::vector<std::vector<T>> recv_bufs(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    new_ghosts[n].source_rank = ctx.neighbors[n];
    new_ghosts[n].count = orig_ghosts[n].count;
    new_ghosts[n].xx.resize(orig_ghosts[n].count);
    new_ghosts[n].yy.resize(orig_ghosts[n].count);
    if (orig_ghosts[n].count > 0)
      recv_bufs[n].resize(orig_ghosts[n].count * 2);
  }

  std::vector<MPI_Request> reqs;
  for (int n = 0; n < num_neighbors; ++n) {
    if (!send_bufs[n].empty()) {
      MPI_Request req;
      MPI_Isend(send_bufs[n].data(), static_cast<int>(send_bufs[n].size()),
                mpi_T, ctx.neighbors[n], 2, comm, &req);
      reqs.push_back(req);
    }
    if (orig_ghosts[n].count > 0) {
      MPI_Request req;
      MPI_Irecv(recv_bufs[n].data(),
                static_cast<int>(orig_ghosts[n].count * 2),
                mpi_T, ctx.neighbors[n], 2, comm, &req);
      reqs.push_back(req);
    }
  }
  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = orig_ghosts[n].count;
    if (cnt > 0) {
      std::memcpy(new_ghosts[n].xx.data(), recv_bufs[n].data(),       cnt * sizeof(T));
      std::memcpy(new_ghosts[n].yy.data(), recv_bufs[n].data() + cnt, cnt * sizeof(T));
    }
  }
}

// =============================================================================
// Build extended device arrays: [local | ghosts]
// =============================================================================

// #7: Fused extended array build — async D2D for local, async H2D per ghost
template <typename T>
void buildExtendedDeviceArrays3D(const T *d_local_xx, const T *d_local_yy,
                                 const T *d_local_zz, int N_local,
                                 const std::vector<GhostBuffer<T>> &ghosts,
                                 size_t total_ghost_count, T *&d_ext_xx,
                                 T *&d_ext_yy, T *&d_ext_zz, int &N_ext) {
  N_ext = N_local + static_cast<int>(total_ghost_count);

  CUDA_CHECK(cudaMalloc(&d_ext_xx, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_ext_yy, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_ext_zz, (size_t)N_ext * sizeof(T)));

  // Async D2D copies for local portion on 3 streams
  cudaStream_t s_local[3];
  for (int i = 0; i < 3; ++i)
    CUDA_CHECK(cudaStreamCreate(&s_local[i]));
  CUDA_CHECK(cudaMemcpyAsync(d_ext_xx, d_local_xx, (size_t)N_local * sizeof(T),
                              cudaMemcpyDeviceToDevice, s_local[0]));
  CUDA_CHECK(cudaMemcpyAsync(d_ext_yy, d_local_yy, (size_t)N_local * sizeof(T),
                              cudaMemcpyDeviceToDevice, s_local[1]));
  CUDA_CHECK(cudaMemcpyAsync(d_ext_zz, d_local_zz, (size_t)N_local * sizeof(T),
                              cudaMemcpyDeviceToDevice, s_local[2]));

  // Async H2D copies for ghosts — one stream per nonempty ghost buffer
  int num_nonempty = 0;
  for (const auto &g : ghosts)
    if (g.count > 0) num_nonempty++;

  std::vector<cudaStream_t> g_streams(num_nonempty);
  for (int i = 0; i < num_nonempty; ++i)
    CUDA_CHECK(cudaStreamCreate(&g_streams[i]));

  size_t offset = N_local;
  int si = 0;
  for (const auto &g : ghosts) {
    if (g.count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_ext_xx + offset, g.xx.data(),
                                  g.count * sizeof(T), cudaMemcpyHostToDevice, g_streams[si]));
      CUDA_CHECK(cudaMemcpyAsync(d_ext_yy + offset, g.yy.data(),
                                  g.count * sizeof(T), cudaMemcpyHostToDevice, g_streams[si]));
      CUDA_CHECK(cudaMemcpyAsync(d_ext_zz + offset, g.zz.data(),
                                  g.count * sizeof(T), cudaMemcpyHostToDevice, g_streams[si]));
      offset += g.count;
      si++;
    }
  }

  for (int i = 0; i < 3; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(s_local[i]));
    CUDA_CHECK(cudaStreamDestroy(s_local[i]));
  }
  for (int i = 0; i < num_nonempty; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(g_streams[i]));
    CUDA_CHECK(cudaStreamDestroy(g_streams[i]));
  }
}

// #7: Fused extended array build 2D — async streams
template <typename T>
void buildExtendedDeviceArrays2D(const T *d_local_xx, const T *d_local_yy,
                                 int N_local,
                                 const std::vector<GhostBuffer<T>> &ghosts,
                                 size_t total_ghost_count, T *&d_ext_xx,
                                 T *&d_ext_yy, int &N_ext) {
  N_ext = N_local + static_cast<int>(total_ghost_count);

  CUDA_CHECK(cudaMalloc(&d_ext_xx, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_ext_yy, (size_t)N_ext * sizeof(T)));

  cudaStream_t s_local[2];
  for (int i = 0; i < 2; ++i)
    CUDA_CHECK(cudaStreamCreate(&s_local[i]));
  CUDA_CHECK(cudaMemcpyAsync(d_ext_xx, d_local_xx, (size_t)N_local * sizeof(T),
                              cudaMemcpyDeviceToDevice, s_local[0]));
  CUDA_CHECK(cudaMemcpyAsync(d_ext_yy, d_local_yy, (size_t)N_local * sizeof(T),
                              cudaMemcpyDeviceToDevice, s_local[1]));

  int num_nonempty = 0;
  for (const auto &g : ghosts)
    if (g.count > 0) num_nonempty++;

  std::vector<cudaStream_t> g_streams(num_nonempty);
  for (int i = 0; i < num_nonempty; ++i)
    CUDA_CHECK(cudaStreamCreate(&g_streams[i]));

  size_t offset = N_local;
  int si = 0;
  for (const auto &g : ghosts) {
    if (g.count > 0) {
      CUDA_CHECK(cudaMemcpyAsync(d_ext_xx + offset, g.xx.data(),
                                  g.count * sizeof(T), cudaMemcpyHostToDevice, g_streams[si]));
      CUDA_CHECK(cudaMemcpyAsync(d_ext_yy + offset, g.yy.data(),
                                  g.count * sizeof(T), cudaMemcpyHostToDevice, g_streams[si]));
      offset += g.count;
      si++;
    }
  }

  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(s_local[i]));
    CUDA_CHECK(cudaStreamDestroy(s_local[i]));
  }
  for (int i = 0; i < num_nonempty; ++i) {
    CUDA_CHECK(cudaStreamSynchronize(g_streams[i]));
    CUDA_CHECK(cudaStreamDestroy(g_streams[i]));
  }
}
