#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <numeric>
#include <omp.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// =============================================================================
// Data structures
// =============================================================================

struct BoundingBox {
  double lo[3]; // min x, y, z (unused dims = 0)
  double hi[3]; // max x, y, z (unused dims = 0)
};

struct DistributedContext {
  int rank;
  int size;
  std::vector<BoundingBox> local_bboxes;  // one per spatial cluster
  std::vector<BoundingBox> padded_bboxes; // each local_bbox expanded by ghost_width
  std::vector<int> neighbors;             // ranks with any overlapping cluster
  // Cached per-rank padded bboxes from Allgatherv (used by exchangeGhosts)
  std::vector<std::vector<BoundingBox>> all_padded_per_rank;
};

template <typename T> struct GhostBuffer {
  int source_rank;
  size_t count;
  std::vector<T> xx, yy, zz; // ghost coordinates (only first D used)
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

/// Check if two boxes overlap in D dimensions
inline bool boxesOverlap(const BoundingBox &a, const BoundingBox &b, int D) {
  for (int d = 0; d < D; ++d) {
    if (a.hi[d] < b.lo[d] || a.lo[d] > b.hi[d])
      return false;
  }
  return true;
}

/// Check if ANY box in padded_set overlaps ANY box in tight_set
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
// Spatial cluster detection
// =============================================================================

template <typename T>
std::vector<BoundingBox> detectSpatialClusters(const T *xx, const T *yy,
                                               const T *zz, size_t N,
                                               double ghost_width, int D) {
  if (N == 0)
    return {};

  const T *coords[3] = {xx, yy, zz};
  std::vector<std::vector<std::pair<double, double>>> intervals(D);

  // #2: Run per-dimension sorts concurrently using threads
  auto sortAndFindGaps = [&](int d) {
    std::vector<T> sorted(coords[d], coords[d] + N);
    std::sort(sorted.begin(), sorted.end());

    double lo = sorted[0];
    for (size_t i = 1; i < N; ++i) {
      if (sorted[i] - sorted[i - 1] > ghost_width) {
        intervals[d].push_back({lo, (double)sorted[i - 1]});
        lo = sorted[i];
      }
    }
    intervals[d].push_back({lo, (double)sorted[N - 1]});
  };

  if (D >= 2) {
    std::vector<std::thread> threads;
    for (int d = 0; d < D; ++d)
      threads.emplace_back(sortAndFindGaps, d);
    for (auto &t : threads)
      t.join();
  } else {
    sortAndFindGaps(0);
  }

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

  // #1: Hash-based O(N) occupancy check instead of O(N×C) brute-force.
  // Map each candidate box to a grid cell key for fast lookup.
  // Since candidates are the Cartesian product of per-axis intervals,
  // we can assign each particle to its interval index per axis directly.

  // Build sorted interval boundaries per axis for binary search
  std::vector<std::vector<double>> lo_bounds(D);
  for (int d = 0; d < D; ++d) {
    lo_bounds[d].reserve(intervals[d].size());
    for (const auto &iv : intervals[d])
      lo_bounds[d].push_back(iv.first);
  }

  // Map from (ix, iy, iz) tuple to candidate index
  std::vector<size_t> interval_dims(D);
  for (int d = 0; d < D; ++d)
    interval_dims[d] = intervals[d].size();

  // #8: OpenMP parallel occupancy check
  std::vector<int> occupied(candidates.size(), 0);

  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < N; ++i) {
    double pt[3] = {(double)xx[i], (double)yy[i], zz ? (double)zz[i] : 0.0};
    size_t cidx = 0;
    size_t stride = 1;
    bool found = true;
    for (int d = D - 1; d >= 0; --d) {
      const auto &lb = lo_bounds[d];
      auto it = std::upper_bound(lb.begin(), lb.end(), pt[d]);
      if (it == lb.begin()) { found = false; break; }
      size_t idx = static_cast<size_t>(std::distance(lb.begin(), it) - 1);
      if (pt[d] > intervals[d][idx].second) { found = false; break; }
      cidx += idx * stride;
      stride *= interval_dims[d];
    }
    if (found && !occupied[cidx]) {
      occupied[cidx] = 1;
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
void computeLocalBBox2D(const T *xx, const T *yy, size_t N,
                        BoundingBox &bbox) {
  if (N == 0)
    return;
  double lo_x = xx[0], hi_x = xx[0];
  double lo_y = yy[0], hi_y = yy[0];
  for (size_t i = 1; i < N; ++i) {
    if (xx[i] < lo_x) lo_x = xx[i];
    if (xx[i] > hi_x) hi_x = xx[i];
    if (yy[i] < lo_y) lo_y = yy[i];
    if (yy[i] > hi_y) hi_y = yy[i];
  }
  bbox.lo[0] = lo_x; bbox.hi[0] = hi_x;
  bbox.lo[1] = lo_y; bbox.hi[1] = hi_y;
  bbox.lo[2] = 0.0;  bbox.hi[2] = 0.0;
}

template <typename T>
void computeLocalBBox3D(const T *xx, const T *yy, const T *zz, size_t N,
                        BoundingBox &bbox) {
  if (N == 0)
    return;
  double lo_x = xx[0], hi_x = xx[0];
  double lo_y = yy[0], hi_y = yy[0];
  double lo_z = zz[0], hi_z = zz[0];
  for (size_t i = 1; i < N; ++i) {
    if (xx[i] < lo_x) lo_x = xx[i];
    if (xx[i] > hi_x) hi_x = xx[i];
    if (yy[i] < lo_y) lo_y = yy[i];
    if (yy[i] > hi_y) hi_y = yy[i];
    if (zz[i] < lo_z) lo_z = zz[i];
    if (zz[i] > hi_z) hi_z = zz[i];
  }
  bbox.lo[0] = lo_x; bbox.hi[0] = hi_x;
  bbox.lo[1] = lo_y; bbox.hi[1] = hi_y;
  bbox.lo[2] = lo_z; bbox.hi[2] = hi_z;
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
// Ghost particle identification (host-side)
// =============================================================================

inline bool pointInAnyBox(double px, double py, double pz,
                          const std::vector<BoundingBox> &boxes, int D) {
  for (const auto &box : boxes) {
    bool inside = true;
    if (px < box.lo[0] || px > box.hi[0]) { inside = false; }
    else if (py < box.lo[1] || py > box.hi[1]) { inside = false; }
    else if (D >= 3 && (pz < box.lo[2] || pz > box.hi[2])) { inside = false; }
    if (inside) return true;
  }
  return false;
}

template <typename T>
void identifyGhostSendList2D(const T *xx, const T *yy, size_t N_local,
                             const std::vector<BoundingBox> &neighbor_padded_bboxes,
                             std::vector<size_t> &send_indices) {
  send_indices.clear();
  if (neighbor_padded_bboxes.empty()) return;
  BoundingBox hull = neighbor_padded_bboxes[0];
  for (size_t b = 1; b < neighbor_padded_bboxes.size(); ++b) {
    for (int d = 0; d < 2; ++d) {
      hull.lo[d] = std::min(hull.lo[d], neighbor_padded_bboxes[b].lo[d]);
      hull.hi[d] = std::max(hull.hi[d], neighbor_padded_bboxes[b].hi[d]);
    }
  }
  for (size_t i = 0; i < N_local; ++i) {
    if (xx[i] < hull.lo[0] || xx[i] > hull.hi[0] ||
        yy[i] < hull.lo[1] || yy[i] > hull.hi[1])
      continue;
    if (pointInAnyBox((double)xx[i], (double)yy[i], 0.0,
                      neighbor_padded_bboxes, 2))
      send_indices.push_back(i);
  }
}

template <typename T>
void identifyGhostSendList3D(const T *xx, const T *yy, const T *zz,
                             size_t N_local,
                             const std::vector<BoundingBox> &neighbor_padded_bboxes,
                             std::vector<size_t> &send_indices) {
  send_indices.clear();
  if (neighbor_padded_bboxes.empty()) return;
  BoundingBox hull = neighbor_padded_bboxes[0];
  for (size_t b = 1; b < neighbor_padded_bboxes.size(); ++b) {
    for (int d = 0; d < 3; ++d) {
      hull.lo[d] = std::min(hull.lo[d], neighbor_padded_bboxes[b].lo[d]);
      hull.hi[d] = std::max(hull.hi[d], neighbor_padded_bboxes[b].hi[d]);
    }
  }
  for (size_t i = 0; i < N_local; ++i) {
    if (xx[i] < hull.lo[0] || xx[i] > hull.hi[0] ||
        yy[i] < hull.lo[1] || yy[i] > hull.hi[1] ||
        zz[i] < hull.lo[2] || zz[i] > hull.hi[2])
      continue;
    if (pointInAnyBox((double)xx[i], (double)yy[i], (double)zz[i],
                      neighbor_padded_bboxes, 3))
      send_indices.push_back(i);
  }
}

// =============================================================================
// Ghost exchange structures
// =============================================================================

template <typename T> struct PendingGhostExchange {
  std::vector<MPI_Request> reqs;
  int num_send_reqs = 0; // B2: first N reqs are sends, rest are recvs

  // A3: Packed send buffers (SoA: [xx|yy|zz])
  std::vector<std::vector<T>> send_packed;

  // A3: Packed recv buffers (unpacked to GhostBuffer on completion)
  std::vector<std::vector<T>> recv_packed;

  // Saved send indices per neighbor (for re-exchange with base data)
  std::vector<std::vector<size_t>> send_indices;

  int rank;
  size_t N_local;
  int D;
};

// =============================================================================
// A3 + B1: Optimized ghost exchange (packed buffers, pipelined identification)
// =============================================================================

template <typename T>
PendingGhostExchange<T>
beginGhostExchange3D(const T *xx, const T *yy, const T *zz, size_t N_local,
                     const DistributedContext &ctx,
                     std::vector<GhostBuffer<T>> &recv_ghosts,
                     size_t &total_ghost_count, MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  PendingGhostExchange<T> pending;
  pending.rank = ctx.rank;
  pending.N_local = N_local;
  pending.D = 3;
  pending.send_packed.resize(num_neighbors);

  MPI_Datatype mpi_T =
      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  // B1: Pipeline — identify, pack, and send for each neighbor
  std::vector<std::vector<size_t>> send_indices(num_neighbors);
  std::vector<size_t> send_counts(num_neighbors);
  pending.send_indices.resize(num_neighbors);

  for (int n = 0; n < num_neighbors; ++n) {
    identifyGhostSendList3D(xx, yy, zz, N_local,
                            ctx.all_padded_per_rank[ctx.neighbors[n]],
                            send_indices[n]);
    pending.send_indices[n] = send_indices[n]; // save for re-exchange
    size_t cnt = send_indices[n].size();
    send_counts[n] = cnt;

    if (cnt > 0) {
      // A3: Pack into single SoA buffer [xx|yy|zz]
      pending.send_packed[n].resize(cnt * 3);
      T *buf = pending.send_packed[n].data();
      for (size_t j = 0; j < cnt; ++j) {
        size_t idx = send_indices[n][j];
        buf[j] = xx[idx];
        buf[cnt + j] = yy[idx];
        buf[2 * cnt + j] = zz[idx];
      }

      // B1: Post non-blocking send immediately (next neighbor ID starts)
      MPI_Request req;
      MPI_Isend(buf, static_cast<int>(cnt * 3), mpi_T, ctx.neighbors[n], 1,
                comm, &req);
      pending.reqs.push_back(req);
      pending.num_send_reqs++;
    }
  }

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
beginGhostExchange2D(const T *xx, const T *yy, size_t N_local,
                     const DistributedContext &ctx,
                     std::vector<GhostBuffer<T>> &recv_ghosts,
                     size_t &total_ghost_count, MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  PendingGhostExchange<T> pending;
  pending.rank = ctx.rank;
  pending.N_local = N_local;
  pending.D = 2;
  pending.send_packed.resize(num_neighbors);

  MPI_Datatype mpi_T =
      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  std::vector<std::vector<size_t>> send_indices(num_neighbors);
  std::vector<size_t> send_counts(num_neighbors);
  pending.send_indices.resize(num_neighbors);

  for (int n = 0; n < num_neighbors; ++n) {
    identifyGhostSendList2D(xx, yy, N_local,
                            ctx.all_padded_per_rank[ctx.neighbors[n]],
                            send_indices[n]);
    pending.send_indices[n] = send_indices[n]; // save for re-exchange
    size_t cnt = send_indices[n].size();
    send_counts[n] = cnt;

    if (cnt > 0) {
      pending.send_packed[n].resize(cnt * 2);
      T *buf = pending.send_packed[n].data();
      for (size_t j = 0; j < cnt; ++j) {
        size_t idx = send_indices[n][j];
        buf[j] = xx[idx];
        buf[cnt + j] = yy[idx];
      }

      MPI_Request req;
      MPI_Isend(buf, static_cast<int>(cnt * 2), mpi_T, ctx.neighbors[n], 1,
                comm, &req);
      pending.reqs.push_back(req);
      pending.num_send_reqs++;
    }
  }

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
    printf("Rank %d: %zu local particles, %zu ghost particles\n", pending.rank,
           pending.N_local, total_ghost_count);
    return;
  }

  int num_reqs = static_cast<int>(pending.reqs.size());

  // Build recv request index -> neighbor index mapping
  std::vector<int> recv_req_to_neighbor;
  for (size_t n = 0; n < recv_ghosts.size(); ++n) {
    if (recv_ghosts[n].count > 0)
      recv_req_to_neighbor.push_back(static_cast<int>(n));
  }

  // B2: Progressive completion
  std::vector<int> completed(num_reqs);
  std::vector<MPI_Status> statuses(num_reqs);
  int remaining = num_reqs;

  while (remaining > 0) {
    int outcount;
    MPI_Waitsome(num_reqs, pending.reqs.data(), &outcount, completed.data(),
                 statuses.data());
    if (outcount == MPI_UNDEFINED)
      break;

    for (int i = 0; i < outcount; ++i) {
      int idx = completed[i];
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

        pending.recv_packed[neighbor_idx].clear();
        pending.recv_packed[neighbor_idx].shrink_to_fit();
      }
    }
    remaining -= outcount;
  }

  printf("Rank %d: %zu local particles, %zu ghost particles\n", pending.rank,
         pending.N_local, total_ghost_count);

  pending.send_packed.clear();
  pending.recv_packed.clear();
  pending.reqs.clear();
}

// =============================================================================
// Build extended arrays: [local | ghost_0 | ghost_1 | ...]
// =============================================================================

template <typename T>
void buildExtendedArrays2D(const T *local_xx, const T *local_yy,
                           size_t N_local,
                           const std::vector<GhostBuffer<T>> &ghosts,
                           size_t total_ghost_count, T *&ext_xx, T *&ext_yy,
                           size_t &N_ext) {
  N_ext = N_local + total_ghost_count;
  ext_xx = new T[N_ext];
  ext_yy = new T[N_ext];

  std::memcpy(ext_xx, local_xx, N_local * sizeof(T));
  std::memcpy(ext_yy, local_yy, N_local * sizeof(T));

  size_t offset = N_local;
  for (const auto &g : ghosts) {
    std::memcpy(ext_xx + offset, g.xx.data(), g.count * sizeof(T));
    std::memcpy(ext_yy + offset, g.yy.data(), g.count * sizeof(T));
    offset += g.count;
  }
}

template <typename T>
void buildExtendedArrays3D(const T *local_xx, const T *local_yy,
                           const T *local_zz, size_t N_local,
                           const std::vector<GhostBuffer<T>> &ghosts,
                           size_t total_ghost_count, T *&ext_xx, T *&ext_yy,
                           T *&ext_zz, size_t &N_ext) {
  N_ext = N_local + total_ghost_count;
  ext_xx = new T[N_ext];
  ext_yy = new T[N_ext];
  ext_zz = new T[N_ext];

  std::memcpy(ext_xx, local_xx, N_local * sizeof(T));
  std::memcpy(ext_yy, local_yy, N_local * sizeof(T));
  std::memcpy(ext_zz, local_zz, N_local * sizeof(T));

  size_t offset = N_local;
  for (const auto &g : ghosts) {
    std::memcpy(ext_xx + offset, g.xx.data(), g.count * sizeof(T));
    std::memcpy(ext_yy + offset, g.yy.data(), g.count * sizeof(T));
    std::memcpy(ext_zz + offset, g.zz.data(), g.count * sizeof(T));
    offset += g.count;
  }
}

/// Free extended arrays allocated by buildExtendedArrays
template <typename T>
void freeExtendedArrays(T *ext_xx, T *ext_yy, T *ext_zz = nullptr) {
  delete[] ext_xx;
  delete[] ext_yy;
  if (ext_zz)
    delete[] ext_zz;
}

// =============================================================================
// Re-exchange ghost data using saved send_indices (for base decompressed coords)
// Uses the same particle indices as the original exchange so ghost ordering
// matches perfectly with the original ghost buffers.
// =============================================================================

template <typename T>
void reexchangeGhostData3D(const std::vector<std::vector<size_t>> &send_indices,
                            const T *h_xx, const T *h_yy, const T *h_zz,
                            const std::vector<GhostBuffer<T>> &orig_ghosts,
                            const DistributedContext &ctx,
                            std::vector<GhostBuffer<T>> &new_ghosts,
                            MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  MPI_Datatype mpi_T = std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  std::vector<std::vector<T>> send_bufs(num_neighbors);
  std::vector<MPI_Request> reqs;

  // Pack and send base coords for the same particles as original exchange
  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = send_indices[n].size();
    if (cnt > 0) {
      send_bufs[n].resize(cnt * 3);
      T *buf = send_bufs[n].data();
      for (size_t j = 0; j < cnt; ++j) {
        size_t idx = send_indices[n][j];
        buf[j]           = h_xx[idx];
        buf[cnt + j]     = h_yy[idx];
        buf[2 * cnt + j] = h_zz[idx];
      }
      MPI_Request req;
      MPI_Isend(buf, static_cast<int>(cnt * 3), mpi_T, ctx.neighbors[n], 2,
                comm, &req);
      reqs.push_back(req);
    }
  }

  // Recv base coords — same counts as original ghost buffers
  new_ghosts.resize(num_neighbors);
  std::vector<std::vector<T>> recv_bufs(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = orig_ghosts[n].count;
    new_ghosts[n].source_rank = ctx.neighbors[n];
    new_ghosts[n].count = cnt;
    new_ghosts[n].xx.resize(cnt);
    new_ghosts[n].yy.resize(cnt);
    new_ghosts[n].zz.resize(cnt);
    if (cnt > 0) {
      recv_bufs[n].resize(cnt * 3);
      MPI_Request req;
      MPI_Irecv(recv_bufs[n].data(), static_cast<int>(cnt * 3), mpi_T,
                ctx.neighbors[n], 2, comm, &req);
      reqs.push_back(req);
    }
  }

  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  // Unpack
  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = orig_ghosts[n].count;
    if (cnt > 0) {
      std::memcpy(new_ghosts[n].xx.data(), recv_bufs[n].data(), cnt * sizeof(T));
      std::memcpy(new_ghosts[n].yy.data(), recv_bufs[n].data() + cnt, cnt * sizeof(T));
      std::memcpy(new_ghosts[n].zz.data(), recv_bufs[n].data() + 2 * cnt, cnt * sizeof(T));
    }
  }
}

template <typename T>
void reexchangeGhostData2D(const std::vector<std::vector<size_t>> &send_indices,
                            const T *h_xx, const T *h_yy,
                            const std::vector<GhostBuffer<T>> &orig_ghosts,
                            const DistributedContext &ctx,
                            std::vector<GhostBuffer<T>> &new_ghosts,
                            MPI_Comm comm) {
  int num_neighbors = static_cast<int>(ctx.neighbors.size());
  MPI_Datatype mpi_T = std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;

  std::vector<std::vector<T>> send_bufs(num_neighbors);
  std::vector<MPI_Request> reqs;

  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = send_indices[n].size();
    if (cnt > 0) {
      send_bufs[n].resize(cnt * 2);
      T *buf = send_bufs[n].data();
      for (size_t j = 0; j < cnt; ++j) {
        size_t idx = send_indices[n][j];
        buf[j]       = h_xx[idx];
        buf[cnt + j] = h_yy[idx];
      }
      MPI_Request req;
      MPI_Isend(buf, static_cast<int>(cnt * 2), mpi_T, ctx.neighbors[n], 2,
                comm, &req);
      reqs.push_back(req);
    }
  }

  new_ghosts.resize(num_neighbors);
  std::vector<std::vector<T>> recv_bufs(num_neighbors);
  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = orig_ghosts[n].count;
    new_ghosts[n].source_rank = ctx.neighbors[n];
    new_ghosts[n].count = cnt;
    new_ghosts[n].xx.resize(cnt);
    new_ghosts[n].yy.resize(cnt);
    if (cnt > 0) {
      recv_bufs[n].resize(cnt * 2);
      MPI_Request req;
      MPI_Irecv(recv_bufs[n].data(), static_cast<int>(cnt * 2), mpi_T,
                ctx.neighbors[n], 2, comm, &req);
      reqs.push_back(req);
    }
  }

  MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

  for (int n = 0; n < num_neighbors; ++n) {
    size_t cnt = orig_ghosts[n].count;
    if (cnt > 0) {
      std::memcpy(new_ghosts[n].xx.data(), recv_bufs[n].data(), cnt * sizeof(T));
      std::memcpy(new_ghosts[n].yy.data(), recv_bufs[n].data() + cnt, cnt * sizeof(T));
    }
  }
}
