#pragma once

#include "HuffmanZSTDCoder.h"
#include "config.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <omp.h>
#include <queue>
#include <unordered_map>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#else
typedef int MPI_Comm;
static const MPI_Comm MPI_COMM_NULL = 0;
#endif

extern size_t max_iter;
extern double lr;

enum class OrderMode { KD_TREE, MORTON_CODE };

// K-D Tree Implementation
template <size_t D, typename T> struct KDNode {
  size_t id; // index in original (not partition of) particle dataset
  T point[D];
  std::unique_ptr<KDNode> left;
  std::unique_ptr<KDNode> right;
};

template <size_t D, typename T> class KDTree {
private:
  std::unique_ptr<KDNode<D, T>> root;
  const T *coords[D];

  std::unique_ptr<KDNode<D, T>> build(std::vector<size_t> &indices,
                                      size_t depth) {
    if (indices.empty())
      return nullptr;

    size_t axis = depth % D;
    size_t median = indices.size() / 2;

    // Sort by current axis
    std::nth_element(
        indices.begin(), indices.begin() + median, indices.end(),
        [&](size_t a, size_t b) { return coords[axis][a] < coords[axis][b]; });

    auto node = std::make_unique<KDNode<D, T>>();
    node->id = indices[median];
    for (size_t d = 0; d < D; ++d) {
      node->point[d] = coords[d][indices[median]];
    }

    std::vector<size_t> left_indices(indices.begin(), indices.begin() + median);
    std::vector<size_t> right_indices(indices.begin() + median + 1,
                                      indices.end());

    node->left = build(left_indices, depth + 1);
    node->right = build(right_indices, depth + 1);

    return node;
  }

  void nearest(const KDNode<D, T> *node, const T query[D], size_t depth,
               size_t &best_id, T &best_dist_sq,
               const std::vector<bool> &visited) const {
    if (!node)
      return;

    if (!visited[node->id]) {
      T dist_sq = 0;
      for (size_t d = 0; d < D; ++d) {
        T diff = node->point[d] - query[d];
        dist_sq += diff * diff;
      }

      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_id = node->id;
      }
    }

    int axis = depth % D;
    T diff = query[axis] - node->point[axis];

    const KDNode<D, T> *near_node =
        (diff < 0) ? node->left.get() : node->right.get();
    const KDNode<D, T> *far_node =
        (diff < 0) ? node->right.get() : node->left.get();

    nearest(near_node, query, depth + 1, best_id, best_dist_sq, visited);

    // Splitting plane is farther than best point, cannot prune far side
    if (diff * diff < best_dist_sq) {
      nearest(far_node, query, depth + 1, best_id, best_dist_sq, visited);
    }
  }

public:
  KDTree(const T *const coords_[D], const std::vector<size_t> &indices) {
    for (size_t d = 0; d < D; ++d) {
      coords[d] = coords_[d];
    }
    std::vector<size_t> indices_copy = indices;
    root = build(indices_copy, 0);
  }

  size_t findNearest(const T query[D], const std::vector<bool> &visited) const {
    size_t best_id = 0;
    T best_dist_sq = std::numeric_limits<T>::max();
    nearest(root.get(), query, 0, best_id, best_dist_sq, visited);
    return best_id;
  }
};

// Morton code / Z-order curve
inline uint32_t expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// 2D Morton code
inline uint32_t morton2D(uint32_t x, uint32_t y) {
  return (expandBits(y) << 1) | expandBits(x);
}

// 3D Morton code
inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
  return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}

template <typename T>
uint32_t computeMortonCode2D(T x, T y, T min_x, T min_y, T cell_size) {
  uint32_t ix =
      static_cast<uint32_t>(std::min(1023.0, (x - min_x) / cell_size * 1024.0));
  uint32_t iy =
      static_cast<uint32_t>(std::min(1023.0, (y - min_y) / cell_size * 1024.0));
  return morton2D(ix, iy);
}

template <typename T>
uint32_t computeMortonCode3D(T x, T y, T z, T min_x, T min_y, T min_z,
                             T cell_size) {
  uint32_t ix =
      static_cast<uint32_t>(std::min(1023.0, (x - min_x) / cell_size * 1024.0));
  uint32_t iy =
      static_cast<uint32_t>(std::min(1023.0, (y - min_y) / cell_size * 1024.0));
  uint32_t iz =
      static_cast<uint32_t>(std::min(1023.0, (z - min_z) / cell_size * 1024.0));
  return morton3D(ix, iy, iz);
}

// Compressed data structure
template <typename T> struct CompressionResults2D {
  T xi, b;
  // Interleaved: x0, y0, x1, y1, ...
  std::vector<bool> lossless_flag;
  std::vector<UInt> quant_codes;
  // Following variables are invisible in decompression stage
  std::vector<size_t> visit_order;
  T *decomp_xx = nullptr, *decomp_yy = nullptr;
  ~CompressionResults2D() {
    delete[] decomp_xx;
    delete[] decomp_yy;
  }
};

template <typename T> struct CompressionResults3D {
  T xi, b;
  // Interleaved: x0, y0, z0, x1, y1, z1, ...
  std::vector<bool> lossless_flag;
  std::vector<UInt> quant_codes;
  // Following variables are invisible in decompression stage
  std::vector<size_t> visit_order;
  T *decomp_xx = nullptr, *decomp_yy = nullptr, *decomp_zz = nullptr;
  ~CompressionResults3D() {
    delete[] decomp_xx;
    delete[] decomp_yy;
    delete[] decomp_zz;
  }
};

// (De)quantization helper functions
template <typename T> inline int quantize(T value, T xi) {
  return static_cast<int>(std::round(value / (2 * xi))) + (1 << (m - 1));
}

template <typename T> inline T dequantize(int quantized, T xi) {
  return static_cast<T>(quantized - (1 << (m - 1))) * 2 * xi;
}

template <typename T>
void predictQuantize(T value, T pred_value, T xi, T &decomp_value,
                     std::vector<bool> &lossless_flag,
                     std::vector<UInt> &quant_codes,
                     std::vector<T> &lossless_values) {
  T pred_err = value - pred_value;
  int quant_code = quantize(pred_err, xi);
  if (quant_code > 0 && quant_code < (1 << m)) {
    // predictable
    quant_codes.push_back(static_cast<UInt>(quant_code));
    lossless_flag.push_back(false);
    decomp_value = dequantize(quant_code, xi) + pred_value;
  } else {
    // unpredictable
    lossless_values.push_back(value);
    lossless_flag.push_back(true);
    decomp_value = value;
  }
}

// Bit-(un)pack helper functions
inline std::vector<uint8_t> packBits(const std::vector<bool> &bits) {
  std::vector<uint8_t> packed;
  packed.reserve((bits.size() + 7) / 8);

  uint8_t curr_byte = 0;
  size_t bit_pos = 0;

  for (bool bit : bits) {
    if (bit)
      curr_byte |= (1 << (7 - bit_pos));
    if (++bit_pos == 8) {
      packed.push_back(curr_byte);
      curr_byte = 0;
      bit_pos = 0;
    }
  }
  // Push final byte if there are remaining bits
  if (bit_pos > 0)
    packed.push_back(curr_byte);

  return packed;
}

inline std::vector<bool> unpackBits(const std::vector<uint8_t> packed,
                                    size_t num_bits = 0) {
  if (num_bits == 0)
    num_bits = packed.size() * 8;

  std::vector<bool> unpacked;
  unpacked.reserve(num_bits);

  size_t bit_pos = 0;
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

// Compression functions for one cell
template <typename T>
void compressParticles2D_KDTree(const T *org_xx, const T *org_yy,
                                const std::vector<size_t> &indices, T start_x,
                                T start_y, CompressionResults2D<T> &result,
                                CompressedData<T> &compressed,
                                std::vector<bool> &visited) {
  const T *coords[2] = {org_xx, org_yy};
  KDTree<2, T> kdtree(coords, indices);

  T query[2] = {start_x, start_y};
  T decomp_x, decomp_y;

  for (size_t i = 0; i < indices.size(); ++i) {
    // Find the next unvisited closet particle
    size_t nearest = kdtree.findNearest(query, visited);

    // Compress the particle
    predictQuantize(org_xx[nearest], query[0], result.xi, decomp_x,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    predictQuantize(org_yy[nearest], query[1], result.xi, decomp_y,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    result.decomp_xx[nearest] = decomp_x;
    result.decomp_yy[nearest] = decomp_y;

    // Mark the particle as visited
    visited[nearest] = true;
    result.visit_order.push_back(nearest);

    // Update current particle
    query[0] = decomp_x;
    query[1] = decomp_y;
  }
  // Insert end mark for each block
  result.quant_codes.push_back(static_cast<UInt>(1 << m));
}

template <typename T>
void compressParticles2D_Morton(const T *org_xx, const T *org_yy,
                                const std::vector<size_t> &indices, T start_x,
                                T start_y, CompressionResults2D<T> &result,
                                CompressedData<T> &compressed) {
  T grid_len = result.b + 2 * std::sqrt(2) * result.xi;
  std::vector<std::pair<uint32_t, size_t>> morton_pairs;
  morton_pairs.reserve(indices.size());

  for (size_t id : indices) {
    uint32_t code =
        computeMortonCode2D(org_xx[id], org_yy[id], start_x, start_y, grid_len);
    morton_pairs.emplace_back(code, id);
  }

  std::sort(morton_pairs.begin(), morton_pairs.end());

  T decomp_x, decomp_y;

  for (const auto &pair : morton_pairs) {
    // Compress the particle
    predictQuantize(org_xx[pair.second], start_x, result.xi, decomp_x,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    predictQuantize(org_yy[pair.second], start_y, result.xi, decomp_y,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    result.decomp_xx[pair.second] = decomp_x;
    result.decomp_yy[pair.second] = decomp_y;

    // Record visit order
    result.visit_order.push_back(pair.second);

    // Update current particle
    start_x = decomp_x;
    start_y = decomp_y;
  }
  // Insert end mark for each block
  result.quant_codes.push_back(static_cast<UInt>(1 << m));
}

template <typename T>
void compressParticles3D_KDTree(const T *org_xx, const T *org_yy,
                                const T *org_zz,
                                const std::vector<size_t> &indices, T start_x,
                                T start_y, T start_z,
                                CompressionResults3D<T> &result,
                                CompressedData<T> &compressed,
                                std::vector<bool> &visited) {
  const T *coords[3] = {org_xx, org_yy, org_zz};
  KDTree<3, T> kdtree(coords, indices);

  T query[3] = {start_x, start_y, start_z};
  T decomp_x, decomp_y, decomp_z;

  for (size_t i = 0; i < indices.size(); ++i) {
    // Find the next unvisited closet particle
    size_t nearest = kdtree.findNearest(query, visited);

    // Compress the particle
    predictQuantize(org_xx[nearest], query[0], result.xi, decomp_x,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    predictQuantize(org_yy[nearest], query[1], result.xi, decomp_y,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    predictQuantize(org_zz[nearest], query[2], result.xi, decomp_z,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    result.decomp_xx[nearest] = decomp_x;
    result.decomp_yy[nearest] = decomp_y;
    result.decomp_zz[nearest] = decomp_z;

    // Mark the particle as visited
    visited[nearest] = true;
    result.visit_order.push_back(nearest);

    // Update current particle
    query[0] = decomp_x;
    query[1] = decomp_y;
    query[2] = decomp_z;
  }
  // Insert end mark for each block
  result.quant_codes.push_back(static_cast<UInt>(1 << m));
}

template <typename T>
void compressParticles3D_Morton(const T *org_xx, const T *org_yy,
                                const T *org_zz,
                                const std::vector<size_t> &indices, T start_x,
                                T start_y, T start_z,
                                CompressionResults3D<T> &result,
                                CompressedData<T> &compressed) {
  T grid_len = result.b + 2 * std::sqrt(3) * result.xi;
  std::vector<std::pair<uint32_t, size_t>> morton_pairs;
  morton_pairs.reserve(indices.size());

  for (size_t id : indices) {
    uint32_t code = computeMortonCode3D(org_xx[id], org_yy[id], org_zz[id],
                                        start_x, start_y, start_z, grid_len);
    morton_pairs.emplace_back(code, id);
  }

  std::sort(morton_pairs.begin(), morton_pairs.end());

  T decomp_x, decomp_y, decomp_z;

  for (const auto &pair : morton_pairs) {
    // Compress the particle
    predictQuantize(org_xx[pair.second], start_x, result.xi, decomp_x,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    predictQuantize(org_yy[pair.second], start_y, result.xi, decomp_y,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    predictQuantize(org_zz[pair.second], start_z, result.xi, decomp_z,
                    result.lossless_flag, result.quant_codes,
                    compressed.lossless_values);
    result.decomp_xx[pair.second] = decomp_x;
    result.decomp_yy[pair.second] = decomp_y;
    result.decomp_zz[pair.second] = decomp_z;

    // Record visit order
    result.visit_order.push_back(pair.second);

    // Update current particle
    start_x = decomp_x;
    start_y = decomp_y;
    start_z = decomp_z;
  }
  // Insert end mark for each block
  result.quant_codes.push_back(static_cast<UInt>(1 << m));
}

template <typename T>
void projectedGradientDescent2D(
    const T *org_xx, const T *org_yy,
    const std::vector<size_t> &vulnerable_pairs, const std::vector<bool> &signs,
    const std::unordered_map<size_t, size_t> &editable_pts_map,
    const std::vector<size_t> &editable_pts, T *decomp_xx, T *decomp_yy,
    std::vector<T> &edit_x, std::vector<T> &edit_y, T b, T xi,
    size_t &iter_used, T &final_loss, MPI_Comm comm = MPI_COMM_NULL) {

  if (editable_pts_map.empty())
    return;

  // max error in distance that quantizing the coordinates can introduce
  T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt(2);
  T upper_bound_dist = b + max_quant_dist_err;
  T lower_bound_dist = b - max_quant_dist_err;
  T convergence_tol = max_quant_dist_err * max_quant_dist_err;
  size_t n = editable_pts.size();
  std::vector<T> grad_x(n, T(0));
  std::vector<T> grad_y(n, T(0));
  edit_x.resize(n, 0);
  edit_y.resize(n, 0);
  size_t num_pairs = vulnerable_pairs.size() / 2;

  // Adam optimizer state
  T adam_alpha = static_cast<T>(lr);
  T adam_beta1 = static_cast<T>(0.9);
  T adam_beta2 = static_cast<T>(0.999);
  T adam_eps = static_cast<T>(1e-8);
  T beta1_t = 1, beta2_t = 1;
  std::vector<T> m_x(n, T(0)), m_y(n, T(0));
  std::vector<T> v_x(n, T(0)), v_y(n, T(0));

  auto distance_decomp = [&](size_t i, size_t j) {
    T dx = decomp_xx[i] - decomp_xx[j];
    T dy = decomp_yy[i] - decomp_yy[j];
    return std::sqrt(dx * dx + dy * dy);
  };

#ifdef USE_MPI
  double t_pgd_start_2d = MPI_Wtime();
#else
  double t_pgd_start_2d = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
  double t_allreduce_total_2d = 0.0;
  for (iter_used = 0; iter_used < max_iter; ++iter_used) {
    // #9: OpenMP parallel loss computation (reduction)
    final_loss = 0;
    #pragma omp parallel for reduction(+:final_loss) schedule(static)
    for (size_t p = 0; p < num_pairs; ++p) {
      size_t i = vulnerable_pairs[2 * p];
      size_t j = vulnerable_pairs[2 * p + 1];
      T d_decomp = distance_decomp(i, j);
      if (signs[p] && d_decomp <= upper_bound_dist) {
        T violation = d_decomp - upper_bound_dist;
        final_loss += violation * violation;
      } else if (!signs[p] && d_decomp > lower_bound_dist) {
        T violation = d_decomp - lower_bound_dist;
        final_loss += violation * violation;
      }
    }
#ifdef USE_MPI
    if (comm != MPI_COMM_NULL) {
      T global_loss;
      double t_ar0 = MPI_Wtime();
      MPI_Allreduce(&final_loss, &global_loss, 1,
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, comm);
      t_allreduce_total_2d += MPI_Wtime() - t_ar0;
      if (global_loss < convergence_tol)
        break;
    } else
#endif
    if (final_loss < convergence_tol)
      break;

    std::fill(grad_x.begin(), grad_x.end(), T(0));
    std::fill(grad_y.begin(), grad_y.end(), T(0));

    // #9: Gradient accumulation (atomic updates for thread safety)
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < num_pairs; ++p) {
      size_t i = vulnerable_pairs[2 * p];
      size_t j = vulnerable_pairs[2 * p + 1];
      auto it_i = editable_pts_map.find(i);
      auto it_j = editable_pts_map.find(j);

      T d_decomp = distance_decomp(i, j);
      if (d_decomp < max_quant_dist_err)
        continue;
      if ((signs[p] && d_decomp <= b) || (!signs[p] && d_decomp > b)) {
        T tmp = 2 * (d_decomp - b) / d_decomp;
        T gc_x = tmp * (decomp_xx[i] - decomp_xx[j]);
        T gc_y = tmp * (decomp_yy[i] - decomp_yy[j]);
        if (it_i != editable_pts_map.end()) {
          #pragma omp atomic
          grad_x[it_i->second] += gc_x;
          #pragma omp atomic
          grad_y[it_i->second] += gc_y;
        }
        if (it_j != editable_pts_map.end()) {
          #pragma omp atomic
          grad_x[it_j->second] -= gc_x;
          #pragma omp atomic
          grad_y[it_j->second] -= gc_y;
        }
      }
    }

    // Adam bias-corrected learning rate
    beta1_t *= adam_beta1;
    beta2_t *= adam_beta2;
    T lr_t = adam_alpha * std::sqrt(1 - beta2_t) / (1 - beta1_t);

    // Adam update & project onto boxes
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; ++idx) {
      size_t i = editable_pts[idx];
      T prev_x = decomp_xx[i];
      T prev_y = decomp_yy[i];

      // Adam moment updates
      m_x[idx] = adam_beta1 * m_x[idx] + (1 - adam_beta1) * grad_x[idx];
      m_y[idx] = adam_beta1 * m_y[idx] + (1 - adam_beta1) * grad_y[idx];
      v_x[idx] = adam_beta2 * v_x[idx] + (1 - adam_beta2) * grad_x[idx] * grad_x[idx];
      v_y[idx] = adam_beta2 * v_y[idx] + (1 - adam_beta2) * grad_y[idx] * grad_y[idx];

      decomp_xx[i] -= lr_t * m_x[idx] / (std::sqrt(v_x[idx]) + adam_eps);
      decomp_yy[i] -= lr_t * m_y[idx] / (std::sqrt(v_y[idx]) + adam_eps);

      // Project onto box constraints
      T lower_x = org_xx[i] - xi;
      T upper_x = org_xx[i] + xi;
      T lower_y = org_yy[i] - xi;
      T upper_y = org_yy[i] + xi;
      if (decomp_xx[i] < lower_x) {
        decomp_xx[i] = lower_x;
      } else if (decomp_xx[i] > upper_x) {
        decomp_xx[i] = upper_x;
      }
      if (decomp_yy[i] < lower_y) {
        decomp_yy[i] = lower_y;
      } else if (decomp_yy[i] > upper_y) {
        decomp_yy[i] = upper_y;
      }
      edit_x[idx] += decomp_xx[i] - prev_x;
      edit_y[idx] += decomp_yy[i] - prev_y;
    }
  }
#ifdef USE_MPI
  double t_pgd_elapsed_2d = MPI_Wtime() - t_pgd_start_2d;
#else
  double t_pgd_elapsed_2d = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count()
      - t_pgd_start_2d;
#endif
  printf("[Timer] PGD iterations (%zu iters): %f seconds\n", iter_used,
         t_pgd_elapsed_2d);
  printf("[Timer] MPI_Allreduce (loss, %zu calls): %f seconds\n", iter_used,
         t_allreduce_total_2d);
  fflush(stdout);
}

template <typename T>
void projectedGradientDescent3D(
    const T *org_xx, const T *org_yy, const T *org_zz,
    const std::vector<size_t> &vulnerable_pairs, const std::vector<bool> &signs,
    const std::unordered_map<size_t, size_t> &editable_pts_map,
    const std::vector<size_t> &editable_pts, T *decomp_xx, T *decomp_yy,
    T *decomp_zz, std::vector<T> &edit_x, std::vector<T> &edit_y,
    std::vector<T> &edit_z, T b, T xi, size_t &iter_used, T &final_loss,
    MPI_Comm comm = MPI_COMM_NULL) {

  if (editable_pts_map.empty())
    return;

  // max error in distance that quantizing the coordinates can introduce
  T max_quant_dist_err = 2 * xi / ((1 << m) - 1) * 2 * sqrt(3);
  T upper_bound_dist = b + max_quant_dist_err;
  T lower_bound_dist = b - max_quant_dist_err;
  T convergence_tol = max_quant_dist_err * max_quant_dist_err;
  size_t n = editable_pts.size();
  std::vector<T> grad_x(n, T(0));
  std::vector<T> grad_y(n, T(0));
  std::vector<T> grad_z(n, T(0));
  edit_x.resize(n, 0);
  edit_y.resize(n, 0);
  edit_z.resize(n, 0);
  size_t num_pairs = vulnerable_pairs.size() / 2;

  // Adam optimizer state
  T adam_alpha = static_cast<T>(lr);
  T adam_beta1 = static_cast<T>(0.9);
  T adam_beta2 = static_cast<T>(0.999);
  T adam_eps = static_cast<T>(1e-8);
  T beta1_t = 1, beta2_t = 1;
  std::vector<T> m_x(n, T(0)), m_y(n, T(0)), m_z(n, T(0));
  std::vector<T> v_x(n, T(0)), v_y(n, T(0)), v_z(n, T(0));

  auto distance_decomp = [&](size_t i, size_t j) {
    T dx = decomp_xx[i] - decomp_xx[j];
    T dy = decomp_yy[i] - decomp_yy[j];
    T dz = decomp_zz[i] - decomp_zz[j];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  };

#ifdef USE_MPI
  double t_pgd_start_3d = MPI_Wtime();
#else
  double t_pgd_start_3d = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
#endif
  double t_allreduce_total_3d = 0.0;
  for (iter_used = 0; iter_used < max_iter; ++iter_used) {

    // #9: OpenMP parallel loss computation 3D (reduction)
    final_loss = 0;
    #pragma omp parallel for reduction(+:final_loss) schedule(static)
    for (size_t p = 0; p < num_pairs; ++p) {
      size_t i = vulnerable_pairs[2 * p];
      size_t j = vulnerable_pairs[2 * p + 1];
      T d_decomp = distance_decomp(i, j);
      if (signs[p] && d_decomp <= upper_bound_dist) {
        T violation = d_decomp - upper_bound_dist;
        final_loss += violation * violation;
      } else if (!signs[p] && d_decomp > lower_bound_dist) {
        T violation = d_decomp - lower_bound_dist;
        final_loss += violation * violation;
      }
    }
#ifdef USE_MPI
    if (comm != MPI_COMM_NULL) {
      T global_loss;
      double t_ar0 = MPI_Wtime();
      MPI_Allreduce(&final_loss, &global_loss, 1,
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, comm);
      t_allreduce_total_3d += MPI_Wtime() - t_ar0;
      if (global_loss < convergence_tol)
        break;
    } else
#endif
    if (final_loss < convergence_tol)
      break;

    std::fill(grad_x.begin(), grad_x.end(), T(0));
    std::fill(grad_y.begin(), grad_y.end(), T(0));
    std::fill(grad_z.begin(), grad_z.end(), T(0));

    // #9: Gradient accumulation 3D (atomic updates)
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < num_pairs; ++p) {
      size_t i = vulnerable_pairs[2 * p];
      size_t j = vulnerable_pairs[2 * p + 1];
      auto it_i = editable_pts_map.find(i);
      auto it_j = editable_pts_map.find(j);

      T d_decomp = distance_decomp(i, j);
      if (d_decomp < max_quant_dist_err)
        continue;
      if ((signs[p] && d_decomp <= b) || (!signs[p] && d_decomp > b)) {
        T tmp = 2 * (d_decomp - b) / d_decomp;
        T gc_x = tmp * (decomp_xx[i] - decomp_xx[j]);
        T gc_y = tmp * (decomp_yy[i] - decomp_yy[j]);
        T gc_z = tmp * (decomp_zz[i] - decomp_zz[j]);
        if (it_i != editable_pts_map.end()) {
          #pragma omp atomic
          grad_x[it_i->second] += gc_x;
          #pragma omp atomic
          grad_y[it_i->second] += gc_y;
          #pragma omp atomic
          grad_z[it_i->second] += gc_z;
        }
        if (it_j != editable_pts_map.end()) {
          #pragma omp atomic
          grad_x[it_j->second] -= gc_x;
          #pragma omp atomic
          grad_y[it_j->second] -= gc_y;
          #pragma omp atomic
          grad_z[it_j->second] -= gc_z;
        }
      }
    }

    // Adam bias-corrected learning rate
    beta1_t *= adam_beta1;
    beta2_t *= adam_beta2;
    T lr_t = adam_alpha * std::sqrt(1 - beta2_t) / (1 - beta1_t);

    // Adam update & project onto boxes
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; ++idx) {
      size_t i = editable_pts[idx];
      T prev_x = decomp_xx[i];
      T prev_y = decomp_yy[i];
      T prev_z = decomp_zz[i];

      // Adam moment updates
      m_x[idx] = adam_beta1 * m_x[idx] + (1 - adam_beta1) * grad_x[idx];
      m_y[idx] = adam_beta1 * m_y[idx] + (1 - adam_beta1) * grad_y[idx];
      m_z[idx] = adam_beta1 * m_z[idx] + (1 - adam_beta1) * grad_z[idx];
      v_x[idx] = adam_beta2 * v_x[idx] + (1 - adam_beta2) * grad_x[idx] * grad_x[idx];
      v_y[idx] = adam_beta2 * v_y[idx] + (1 - adam_beta2) * grad_y[idx] * grad_y[idx];
      v_z[idx] = adam_beta2 * v_z[idx] + (1 - adam_beta2) * grad_z[idx] * grad_z[idx];

      decomp_xx[i] -= lr_t * m_x[idx] / (std::sqrt(v_x[idx]) + adam_eps);
      decomp_yy[i] -= lr_t * m_y[idx] / (std::sqrt(v_y[idx]) + adam_eps);
      decomp_zz[i] -= lr_t * m_z[idx] / (std::sqrt(v_z[idx]) + adam_eps);

      // Project onto box constraints
      T lower_x = org_xx[i] - xi;
      T upper_x = org_xx[i] + xi;
      T lower_y = org_yy[i] - xi;
      T upper_y = org_yy[i] + xi;
      T lower_z = org_zz[i] - xi;
      T upper_z = org_zz[i] + xi;
      if (decomp_xx[i] < lower_x) {
        decomp_xx[i] = lower_x;
      } else if (decomp_xx[i] > upper_x) {
        decomp_xx[i] = upper_x;
      }
      if (decomp_yy[i] < lower_y) {
        decomp_yy[i] = lower_y;
      } else if (decomp_yy[i] > upper_y) {
        decomp_yy[i] = upper_y;
      }
      if (decomp_zz[i] < lower_z) {
        decomp_zz[i] = lower_z;
      } else if (decomp_zz[i] > upper_z) {
        decomp_zz[i] = upper_z;
      }
      edit_x[idx] += decomp_xx[i] - prev_x;
      edit_y[idx] += decomp_yy[i] - prev_y;
      edit_z[idx] += decomp_zz[i] - prev_z;
    }
  }
#ifdef USE_MPI
  double t_pgd_elapsed_3d = MPI_Wtime() - t_pgd_start_3d;
#else
  double t_pgd_elapsed_3d = std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count()
      - t_pgd_start_3d;
#endif
  printf("[Timer] PGD iterations (%zu iters): %f seconds\n", iter_used,
         t_pgd_elapsed_3d);
  printf("[Timer] MPI_Allreduce (loss, %zu calls): %f seconds\n", iter_used,
         t_allreduce_total_3d);
  fflush(stdout);
}

template <typename T, OrderMode Mode>
void compressWithEditParticles2D(const T *org_xx, const T *org_yy, T min_x,
                                 T range_x, T min_y, T range_y, size_t N, T xi,
                                 T b, bool isPGD,
                                 CompressionResults2D<T> &result,
                                 CompressedData<T> &compressed,
                                 size_t N_local = 0) {

  auto start = std::chrono::high_resolution_clock::now();

  if (N == 0)
    return;

  size_t eff_N_local = (N_local == 0) ? N : N_local;

  result.xi = xi;
  result.b = b;

  // Reserve space for compression results
  result.lossless_flag.reserve(2 * N);
  result.quant_codes.reserve(2 * N);
  compressed.lossless_values.reserve(2 * N);
  result.visit_order.reserve(N);
  result.decomp_xx = new T[N];
  result.decomp_yy = new T[N];

  // Compute grid dimensions
  T grid_len = b + 2 * std::sqrt(2) * xi;
  compressed.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  compressed.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));

  // Store grid min
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.xi = xi;
  compressed.b = b;
  compressed.N_local = N_local;

  // Partition particles into cells (sparse)
  std::unordered_map<size_t, std::vector<size_t>> cell_map;

  for (size_t i = 0; i < N; ++i) {
    size_t cid_x =
        static_cast<size_t>(std::floor((org_xx[i] - min_x) / grid_len));
    size_t cid_y =
        static_cast<size_t>(std::floor((org_yy[i] - min_y) / grid_len));
    size_t id = cid_y * compressed.grid_dim_x + cid_x;
    cell_map[id].push_back(i);
  }

  // Collect and sort non-empty cell IDs
  std::vector<size_t> non_empty_ids;
  non_empty_ids.reserve(cell_map.size());
  for (const auto &entry : cell_map)
    non_empty_ids.push_back(entry.first);
  std::sort(non_empty_ids.begin(), non_empty_ids.end());

  // Process each cell (Vulnerable pairs detection & Compression)
  std::vector<size_t> vulnerable_pairs;
  std::vector<bool> signs;
  std::unordered_map<size_t, size_t> editable_pts_map;
  std::vector<size_t> editable_pts;
  vulnerable_pairs.reserve(2 * N / reserve_factor);
  signs.reserve(N / reserve_factor);
  editable_pts_map.reserve(2 * N / reserve_factor);
  editable_pts.reserve(2 * N / reserve_factor);
  T lower_bound = b - 2 * std::sqrt(2) * xi;
  T upper_bound = b + 2 * std::sqrt(2) * xi;
  T lower_bound_sq = (lower_bound < 0) ? 0 : lower_bound * lower_bound;
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  static constexpr int neighbor_offsets[4][2] = {
      {1, 0}, {0, 1}, {1, 1}, {1, -1}};

  auto distanceSquared = [&](size_t i, size_t j) {
    T dx = org_xx[i] - org_xx[j];
    T dy = org_yy[i] - org_yy[j];
    return dx * dx + dy * dy;
  };
  auto addPt = [&](size_t i) {
    if (i >= eff_N_local) return;
    if (editable_pts_map.find(i) == editable_pts_map.end()) {
      editable_pts_map[i] = editable_pts.size();
      editable_pts.push_back(i);
    }
  };
  auto checkAndAddPair = [&](size_t i, size_t j) {
    if (i >= eff_N_local && j >= eff_N_local) return;
    T dist_sq = distanceSquared(i, j);
    if (dist_sq > lower_bound_sq && dist_sq <= upper_bound_sq) {
      vulnerable_pairs.push_back(i);
      vulnerable_pairs.push_back(j);
      signs.push_back(dist_sq > sign_bound_sq);
      addPt(i);
      addPt(j);
    }
  };

  std::vector<bool> visited(N, false);
  size_t prev_cell = 0;
  size_t total_cells_2d = (size_t)compressed.grid_dim_x * compressed.grid_dim_y;

  for (size_t ci = 0; ci < non_empty_ids.size(); ++ci) {
    size_t id = non_empty_ids[ci];

    if (id > prev_cell)
      result.quant_codes.insert(result.quant_codes.end(),
                                 id - prev_cell, static_cast<UInt>(0));

    size_t id_x = id % compressed.grid_dim_x;
    size_t id_y = id / compressed.grid_dim_x;

    auto &indices = cell_map[id];

    // Check pairs within the same cell
    for (size_t i = 0; i < indices.size(); ++i) {
      for (size_t j = i + 1; j < indices.size(); ++j) {
        checkAndAddPair(indices[i], indices[j]);
      }
    }

    // Check pairs with neighboring cells
    for (const auto &offset : neighbor_offsets) {
      int nx = id_x + offset[0];
      int ny = id_y + offset[1];

      if (nx < 0 || nx >= static_cast<int>(compressed.grid_dim_x) || ny < 0 ||
          ny >= static_cast<int>(compressed.grid_dim_y))
        continue;

      size_t neighbor_id = ny * compressed.grid_dim_x + nx;
      auto nit = cell_map.find(neighbor_id);
      if (nit == cell_map.end()) continue;
      const auto &n_indices = nit->second;

      for (size_t i : indices) {
        for (size_t j : n_indices) {
          checkAndAddPair(i, j);
        }
      }
    }

    // Compress particles during reordering by k-d tree or Morton code
    if constexpr (Mode == OrderMode::KD_TREE) {
      T cell_center_x = min_x + (id_x + T(0.5)) * grid_len;
      T cell_center_y = min_y + (id_y + T(0.5)) * grid_len;

      compressParticles2D_KDTree(org_xx, org_yy, indices, cell_center_x,
                                 cell_center_y, result, compressed, visited);
    } else {
      T cell_min_corner_x = min_x + id_x * grid_len;
      T cell_min_corner_y = min_y + id_y * grid_len;

      compressParticles2D_Morton(org_xx, org_yy, indices, cell_min_corner_x,
                                 cell_min_corner_y, result, compressed);
    }
    prev_cell = id + 1;
  }

  if (total_cells_2d > prev_cell)
    result.quant_codes.insert(result.quant_codes.end(),
                               total_cells_2d - prev_cell,
                               static_cast<UInt>(0));

  printf("Number of vulnerable pairs: %zu\n", vulnerable_pairs.size() / 2);
  printf("Number of editable particles: %zu\n", editable_pts.size());

  // Count violated pairs before PGD
  auto countViolations2D = [&]() {
    size_t violations = 0;
    T b_sq = b * b;
    for (size_t k = 0; k < vulnerable_pairs.size(); k += 2) {
      size_t p = vulnerable_pairs[k];
      size_t q = vulnerable_pairs[k + 1];
      T dx = result.decomp_xx[p] - result.decomp_xx[q];
      T dy = result.decomp_yy[p] - result.decomp_yy[q];
      T dist_sq = dx * dx + dy * dy;
      if ((dist_sq > b_sq) != signs[k / 2])
        violations++;
    }
    return violations;
  };
  printf("Violated pairs before editing: %zu\n", countViolations2D());

  if (isPGD) {
    // Projected gradient descent
    std::vector<T> edit_x;
    std::vector<T> edit_y;
    size_t iter_used = 0;
    T final_loss = 0;
    projectedGradientDescent2D(org_xx, org_yy, vulnerable_pairs, signs,
                               editable_pts_map, editable_pts, result.decomp_xx,
                               result.decomp_yy, edit_x, edit_y, b, xi,
                               iter_used, final_loss);

    // Quantize edits (min: -2 * xi, max 2 * xi)
    // Re-apply quantized edits to get the true decompressed values meanwhile
    std::vector<UInt2> quant_edits(2 * N,
                                   static_cast<UInt2>(((1 << m) - 1) / 2));
    T quant_norm = ((1 << m) - 1) / (4 * xi);
    T dequant_norm = (4 * xi) / ((1 << m) - 1);
    for (size_t i = 0; i < N; ++i) {
      size_t idx = result.visit_order[i];

      auto it = editable_pts_map.find(idx);
      if (it != editable_pts_map.end()) {
        size_t edit_idx = it->second;

        // quantize
        UInt2 quant_edit_x = (edit_x[edit_idx] + 2 * xi) * quant_norm;
        UInt2 quant_edit_y = (edit_y[edit_idx] + 2 * xi) * quant_norm;
        quant_edits[2 * i] = quant_edit_x;
        quant_edits[2 * i + 1] = quant_edit_y;

        // dequantize and apply to base values directly
        T dequant_edit_x = static_cast<T>(quant_edit_x) * dequant_norm - 2 * xi;
        T dequant_edit_y = static_cast<T>(quant_edit_y) * dequant_norm - 2 * xi;
        result.decomp_xx[idx] += dequant_edit_x - edit_x[edit_idx];
        result.decomp_yy[idx] += dequant_edit_y - edit_y[edit_idx];
      }
    }

    printf("Violated pairs after PGD: %zu\n", countViolations2D());
    printf("Number of iterations: %zu\n", iter_used);
    printf("PGD final loss: %f\n", final_loss);

    // Pack flag variables
    std::vector<uint8_t> packed_lossless_flag = packBits(result.lossless_flag);
    compressed.size_flag = packed_lossless_flag.size();
    compressed.size_quant = result.quant_codes.size();

    // Lossless compression
    compressed.compressed_lossless_flag =
        huffmanZstdCompress(packed_lossless_flag, compressed.code_table_flag,
                            compressed.bit_stream_size_flag);
    compressed.compressed_quant_codes =
        huffmanZstdCompress(result.quant_codes, compressed.code_table_quant,
                            compressed.bit_stream_size_quant);
    compressed.compressed_quant_edits =
        huffmanZstdCompress(quant_edits, compressed.code_table_edit,
                            compressed.bit_stream_size_edit);
    compressed.size_edit = 2 * N;
  } else {
    // Lossless edit mode: store editable points' original values directly
    std::vector<bool> edit_flag(N, false);
    for (size_t i = 0; i < N; ++i) {
      size_t idx = result.visit_order[i];
      if (editable_pts_map.count(idx)) {
        edit_flag[i] = true;
        compressed.lossless_edit_values.push_back(org_xx[idx]);
        compressed.lossless_edit_values.push_back(org_yy[idx]);
        result.decomp_xx[idx] = org_xx[idx];
        result.decomp_yy[idx] = org_yy[idx];
      }
    }
    std::vector<uint8_t> packed_flag = packBits(edit_flag);
    compressed.size_edit_flag = packed_flag.size();
    compressed.compressed_lossless_edit_flag =
        huffmanZstdCompress(packed_flag, compressed.code_table_edit_flag,
                            compressed.bit_stream_size_edit_flag);
    compressed.size_edit = 0;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> comp_time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.compressed_quant_edits.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      sizeof(T) * compressed.lossless_values.size() +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  T compression_ratio = static_cast<T>(sizeof(T) * 2 * N) / compressed_size;
  T bpp = 2 * 8 * sizeof(T) / compression_ratio;
  // #9: OpenMP parallel error computation
  T mae = 0;
  T mse = 0;
  #pragma omp parallel for reduction(max:mae) reduction(+:mse) schedule(static)
  for (size_t i = 0; i < N; ++i) {
    T err_x = std::abs(result.decomp_xx[i] - org_xx[i]);
    T err_y = std::abs(result.decomp_yy[i] - org_yy[i]);
    if (err_x > mae)
      mae = err_x;
    if (err_y > mae)
      mae = err_y;
    mse += err_x * err_x + err_y * err_y;
  }
  mse /= (2 * N);
  T rmse = std::sqrt(mse);
  T nrmse = rmse / std::max(range_x, range_y);
  T psnr = -20 * std::log10(nrmse);

  printf("Compression time: %f seconds\n", comp_time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

template <typename T, OrderMode Mode>
void compressParticles2D(const T *org_xx, const T *org_yy, T min_x, T range_x,
                         T min_y, T range_y, size_t N, T xi, T b,
                         CompressionResults2D<T> &result,
                         CompressedData<T> &compressed) {

  auto start = std::chrono::high_resolution_clock::now();

  if (N == 0)
    return;

  result.xi = xi;
  result.b = b;

  // Reserve space for compression results
  result.lossless_flag.reserve(2 * N);
  result.quant_codes.reserve(2 * N);
  compressed.lossless_values.reserve(2 * N);
  result.visit_order.reserve(N);
  result.decomp_xx = new T[N];
  result.decomp_yy = new T[N];

  // Compute grid dimensions
  T grid_len = b + 2 * std::sqrt(2) * xi;
  compressed.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  compressed.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));

  // Store grid min
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.xi = xi;
  compressed.b = b;

  // Partition particles into cells (sparse)
  std::unordered_map<size_t, std::vector<size_t>> cell_map;

  for (size_t i = 0; i < N; ++i) {
    size_t cid_x =
        static_cast<size_t>(std::floor((org_xx[i] - min_x) / grid_len));
    size_t cid_y =
        static_cast<size_t>(std::floor((org_yy[i] - min_y) / grid_len));
    size_t id = cid_y * compressed.grid_dim_x + cid_x;
    cell_map[id].push_back(i);
  }

  // Collect and sort non-empty cell IDs
  std::vector<size_t> non_empty_ids;
  non_empty_ids.reserve(cell_map.size());
  for (const auto &entry : cell_map)
    non_empty_ids.push_back(entry.first);
  std::sort(non_empty_ids.begin(), non_empty_ids.end());

  size_t total_cells_cp2 = (size_t)compressed.grid_dim_x * compressed.grid_dim_y;

  // Process each cell (Compression)
  switch (Mode) {
  case OrderMode::KD_TREE: {
    std::vector<bool> visited(N, false);
    size_t prev_cell = 0;

    for (size_t ci = 0; ci < non_empty_ids.size(); ++ci) {
      size_t id = non_empty_ids[ci];
      if (id > prev_cell)
        result.quant_codes.insert(result.quant_codes.end(),
                                   id - prev_cell, static_cast<UInt>(0));

      size_t id_x = id % compressed.grid_dim_x;
      size_t id_y = id / compressed.grid_dim_x;

      auto &indices = cell_map[id];

      T cell_center_x = min_x + (id_x + T(0.5)) * grid_len;
      T cell_center_y = min_y + (id_y + T(0.5)) * grid_len;

      compressParticles2D_KDTree(org_xx, org_yy, indices, cell_center_x,
                                 cell_center_y, result, compressed, visited);
      prev_cell = id + 1;
    }
    if (total_cells_cp2 > prev_cell)
      result.quant_codes.insert(result.quant_codes.end(),
                                 total_cells_cp2 - prev_cell,
                                 static_cast<UInt>(0));
    break;
  }
  case OrderMode::MORTON_CODE: {
    size_t prev_cell = 0;

    for (size_t ci = 0; ci < non_empty_ids.size(); ++ci) {
      size_t id = non_empty_ids[ci];
      if (id > prev_cell)
        result.quant_codes.insert(result.quant_codes.end(),
                                   id - prev_cell, static_cast<UInt>(0));

      size_t id_x = id % compressed.grid_dim_x;
      size_t id_y = id / compressed.grid_dim_x;

      auto &indices = cell_map[id];

      T cell_min_corner_x = min_x + id_x * grid_len;
      T cell_min_corner_y = min_y + id_y * grid_len;

      compressParticles2D_Morton(org_xx, org_yy, indices, cell_min_corner_x,
                                 cell_min_corner_y, result, compressed);
      prev_cell = id + 1;
    }
    if (total_cells_cp2 > prev_cell)
      result.quant_codes.insert(result.quant_codes.end(),
                                 total_cells_cp2 - prev_cell,
                                 static_cast<UInt>(0));
    break;
  }
  }

  // Pack flag variables
  std::vector<uint8_t> packed_lossless_flag = packBits(result.lossless_flag);
  compressed.size_flag = packed_lossless_flag.size();
  compressed.size_quant = result.quant_codes.size();

  // Lossless compression
  compressed.compressed_lossless_flag =
      huffmanZstdCompress(packed_lossless_flag, compressed.code_table_flag,
                          compressed.bit_stream_size_flag);
  compressed.compressed_quant_codes =
      huffmanZstdCompress(result.quant_codes, compressed.code_table_quant,
                          compressed.bit_stream_size_quant);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> comp_time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      sizeof(T) * compressed.lossless_values.size();
  T compression_ratio = static_cast<T>(sizeof(T) * 2 * N) / compressed_size;
  T bpp = 2 * 8 * sizeof(T) / compression_ratio;
  // #9: OpenMP parallel error computation
  T mae = 0;
  T mse = 0;
  #pragma omp parallel for reduction(max:mae) reduction(+:mse) schedule(static)
  for (size_t i = 0; i < N; ++i) {
    T err_x = std::abs(result.decomp_xx[i] - org_xx[i]);
    T err_y = std::abs(result.decomp_yy[i] - org_yy[i]);
    if (err_x > mae)
      mae = err_x;
    if (err_y > mae)
      mae = err_y;
    mse += err_x * err_x + err_y * err_y;
  }
  mse /= (2 * N);
  T rmse = std::sqrt(mse);
  T nrmse = rmse / std::max(range_x, range_y);
  T psnr = -20 * std::log10(nrmse);

  printf("Compression time: %f seconds\n", comp_time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

template <typename T>
void editParticles2D(const T *org_xx, const T *org_yy, T min_x, T range_x,
                     T min_y, T range_y, size_t N, T xi, T b, bool isPGD,
                     CompressionResults2D<T> &result,
                     CompressedData<T> &compressed,
                     size_t N_local = 0, MPI_Comm comm = MPI_COMM_NULL) {

  auto start = std::chrono::high_resolution_clock::now();

  if (N == 0)
    return;

  size_t eff_N_local = (N_local == 0) ? N : N_local;

  result.xi = xi;
  result.b = b;

  // Compute grid dimensions
  T grid_len = b + 2 * std::sqrt(2) * xi;
  compressed.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  compressed.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));

  // Store grid min
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.xi = xi;
  compressed.b = b;
  compressed.N_local = N_local;

  // Partition particles into cells (sparse)
  std::unordered_map<size_t, std::vector<size_t>> cell_map;

  for (size_t i = 0; i < N; ++i) {
    size_t cid_x =
        static_cast<size_t>(std::floor((org_xx[i] - min_x) / grid_len));
    size_t cid_y =
        static_cast<size_t>(std::floor((org_yy[i] - min_y) / grid_len));
    size_t id = cid_y * compressed.grid_dim_x + cid_x;
    cell_map[id].push_back(i);
  }

  // Process each cell (Vulnerable pairs detection)
  std::vector<size_t> vulnerable_pairs;
  std::vector<bool> signs;
  std::unordered_map<size_t, size_t> editable_pts_map;
  std::vector<size_t> editable_pts;
  vulnerable_pairs.reserve(2 * N / reserve_factor);
  signs.reserve(N / reserve_factor);
  editable_pts_map.reserve(2 * N / reserve_factor);
  editable_pts.reserve(2 * N / reserve_factor);
  T lower_bound = b - 2 * std::sqrt(2) * xi;
  T upper_bound = b + 2 * std::sqrt(2) * xi;
  T lower_bound_sq;
  if (lower_bound < 0) {
    lower_bound_sq = 0;
  } else {
    lower_bound_sq = lower_bound * lower_bound;
  }
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  static constexpr int neighbor_offsets[4][2] = {
      {0, 1}, {1, 0}, {1, 1}, {1, -1}};

  auto distanceSquared = [&](size_t i, size_t j) {
    T dx = org_xx[i] - org_xx[j];
    T dy = org_yy[i] - org_yy[j];
    return dx * dx + dy * dy;
  };
  auto addPt = [&](size_t i) {
    if (i >= eff_N_local) return;
    if (editable_pts_map.find(i) == editable_pts_map.end()) {
      editable_pts_map[i] = editable_pts.size();
      editable_pts.push_back(i);
    }
  };
  auto checkAndAddPair = [&](size_t i, size_t j) {
    if (i >= eff_N_local && j >= eff_N_local) return;
    T dist_sq = distanceSquared(i, j);
    if (dist_sq > lower_bound_sq && dist_sq <= upper_bound_sq) {
      vulnerable_pairs.push_back(i);
      vulnerable_pairs.push_back(j);
      signs.push_back(dist_sq > sign_bound_sq);
      addPt(i);
      addPt(j);
    }
  };

  std::vector<bool> visited(N, false);

  for (auto &kv : cell_map) {
    size_t id = kv.first;
    auto &indices = kv.second;

    size_t id_x = id % compressed.grid_dim_x;
    size_t id_y = id / compressed.grid_dim_x;

    // Check pairs within the same cell
    for (size_t i = 0; i < indices.size(); ++i) {
      for (size_t j = i + 1; j < indices.size(); ++j) {
        checkAndAddPair(indices[i], indices[j]);
      }
    }

    // Check pairs with neighboring cells
    for (const auto &offset : neighbor_offsets) {
      int nx = id_x + offset[0];
      int ny = id_y + offset[1];

      if (nx < 0 || nx >= static_cast<int>(compressed.grid_dim_x) || ny < 0 ||
          ny >= static_cast<int>(compressed.grid_dim_y))
        continue;

      size_t neighbor_id = ny * compressed.grid_dim_x + nx;
      auto nit = cell_map.find(neighbor_id);
      if (nit == cell_map.end()) continue;
      const auto &n_indices = nit->second;

      for (size_t i : indices) {
        for (size_t j : n_indices) {
          checkAndAddPair(i, j);
        }
      }
    }
  }

  printf("Number of vulnerable pairs: %zu\n", vulnerable_pairs.size() / 2);
  printf("Number of editable particles: %zu\n", editable_pts.size());

  // Count violated pairs before PGD
  auto countViolations2D = [&]() {
    size_t violations = 0;
    T b_sq = b * b;
    for (size_t k = 0; k < vulnerable_pairs.size(); k += 2) {
      size_t p = vulnerable_pairs[k];
      size_t q = vulnerable_pairs[k + 1];
      T dx = result.decomp_xx[p] - result.decomp_xx[q];
      T dy = result.decomp_yy[p] - result.decomp_yy[q];
      T dist_sq = dx * dx + dy * dy;
      if ((dist_sq > b_sq) != signs[k / 2])
        violations++;
    }
    return violations;
  };
  printf("Violated pairs before editing: %zu\n", countViolations2D());

  if (isPGD) {
    // Projected gradient descent
    std::vector<T> edit_x;
    std::vector<T> edit_y;
    size_t iter_used = 0;
    T final_loss = 0;
    projectedGradientDescent2D(org_xx, org_yy, vulnerable_pairs, signs,
                               editable_pts_map, editable_pts, result.decomp_xx,
                               result.decomp_yy, edit_x, edit_y, b, xi,
                               iter_used, final_loss, comm);

    // Quantize edits (min: -2 * xi, max 2 * xi)
    std::vector<UInt2> quant_edits(2 * N,
                                   static_cast<UInt2>(((1 << m) - 1) / 2));
    T quant_norm = ((1 << m) - 1) / (4 * xi);
    T dequant_norm = (4 * xi) / ((1 << m) - 1);
    for (const auto &[i, ii] : editable_pts_map) {
      UInt2 quant_edit_x = (edit_x[ii] + 2 * xi) * quant_norm;
      UInt2 quant_edit_y = (edit_y[ii] + 2 * xi) * quant_norm;
      quant_edits[2 * i] = quant_edit_x;
      quant_edits[2 * i + 1] = quant_edit_y;

      // dequantize and apply to base values directly
      T dequant_edit_x = static_cast<T>(quant_edit_x) * dequant_norm - 2 * xi;
      T dequant_edit_y = static_cast<T>(quant_edit_y) * dequant_norm - 2 * xi;
      result.decomp_xx[i] += dequant_edit_x - edit_x[ii];
      result.decomp_yy[i] += dequant_edit_y - edit_y[ii];
    }
    printf("Violated pairs after PGD: %zu\n", countViolations2D());
    printf("Number of iterations: %zu\n", iter_used);
    printf("PGD final loss: %f\n", final_loss);

    compressed.compressed_quant_edits =
        huffmanZstdCompress(quant_edits, compressed.code_table_edit,
                            compressed.bit_stream_size_edit);
    compressed.size_edit = 2 * N;

  } else {
    // Lossless edit mode: store original values in original particle order
    std::vector<bool> edit_flag(N, false);
    for (size_t i = 0; i < N; ++i) {
      if (editable_pts_map.count(i)) {
        edit_flag[i] = true;
        compressed.lossless_edit_values.push_back(org_xx[i]);
        compressed.lossless_edit_values.push_back(org_yy[i]);
        result.decomp_xx[i] = org_xx[i];
        result.decomp_yy[i] = org_yy[i];
      }
    }
    std::vector<uint8_t> packed_flag = packBits(edit_flag);
    compressed.size_edit_flag = packed_flag.size();
    compressed.compressed_lossless_edit_flag =
        huffmanZstdCompress(packed_flag, compressed.code_table_edit_flag,
                            compressed.bit_stream_size_edit_flag);
    compressed.size_edit = 0;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> comp_time = end - start;

  size_t additional_size =
      compressed.compressed_quant_edits.size() +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  // #9: OpenMP parallel error computation
  T mae = 0;
  T mse = 0;
  #pragma omp parallel for reduction(max:mae) reduction(+:mse) schedule(static)
  for (size_t i = 0; i < N; ++i) {
    T err_x = std::abs(result.decomp_xx[i] - org_xx[i]);
    T err_y = std::abs(result.decomp_yy[i] - org_yy[i]);
    if (err_x > mae)
      mae = err_x;
    if (err_y > mae)
      mae = err_y;
    mse += err_x * err_x + err_y * err_y;
  }
  mse /= (2 * N);
  T rmse = std::sqrt(mse);
  T nrmse = rmse / std::max(range_x, range_y);
  T psnr = -20 * std::log10(nrmse);

  printf("Compression time: %f seconds\n", comp_time.count());
  printf("Additional storage: %zu bytes\n", additional_size);
  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

template <typename T, OrderMode Mode>
void compressWithEditParticles3D(const T *org_xx, const T *org_yy,
                                 const T *org_zz, T min_x, T range_x, T min_y,
                                 T range_y, T min_z, T range_z, size_t N, T xi,
                                 T b, bool isPGD,
                                 CompressionResults3D<T> &result,
                                 CompressedData<T> &compressed,
                                 size_t N_local = 0) {

  auto start = std::chrono::high_resolution_clock::now();
  auto phase_start = start;
  auto phase_end = start;

  if (N == 0)
    return;

  size_t eff_N_local = (N_local == 0) ? N : N_local;

  result.xi = xi;
  result.b = b;

  // Reserve space for compression results
  result.lossless_flag.reserve(3 * N);
  result.quant_codes.reserve(3 * N);
  compressed.lossless_values.reserve(3 * N);
  result.visit_order.reserve(N);
  result.decomp_xx = new T[N];
  result.decomp_yy = new T[N];
  result.decomp_zz = new T[N];

  phase_start = std::chrono::high_resolution_clock::now();

  // Compute grid dimensions
  T grid_len = b + 2 * std::sqrt(3) * xi;
  compressed.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  compressed.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));
  compressed.grid_dim_z =
      std::max(1, static_cast<int>(std::ceil(range_z / grid_len)));

  // Store grid min
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.grid_min_z = min_z;
  compressed.xi = xi;
  compressed.b = b;
  compressed.N_local = N_local;

  // Partition particles into cells (sparse storage to avoid OOM on
  // non-contiguous layouts where the coordinate range has large gaps)
  std::unordered_map<size_t, std::vector<size_t>> cell_map;
  size_t dim_xy_3 = (size_t)compressed.grid_dim_x * compressed.grid_dim_y;

  for (size_t i = 0; i < N; ++i) {
    size_t cid_x =
        static_cast<size_t>(std::floor((org_xx[i] - min_x) / grid_len));
    size_t cid_y =
        static_cast<size_t>(std::floor((org_yy[i] - min_y) / grid_len));
    size_t cid_z =
        static_cast<size_t>(std::floor((org_zz[i] - min_z) / grid_len));
    size_t id = cid_z * dim_xy_3 + cid_y * compressed.grid_dim_x + cid_x;
    cell_map[id].push_back(i);
  }

  // Collect and sort non-empty cell IDs for raster-order processing
  std::vector<size_t> non_empty_ids;
  non_empty_ids.reserve(cell_map.size());
  for (const auto &entry : cell_map)
    non_empty_ids.push_back(entry.first);
  std::sort(non_empty_ids.begin(), non_empty_ids.end());

  phase_end = std::chrono::high_resolution_clock::now();
  printf("[Timer] Grid partitioning: %f seconds\n",
         std::chrono::duration<double>(phase_end - phase_start).count());
  phase_start = phase_end;

  // Process each cell (Vulnerable pairs detection & Compression)
  std::vector<size_t> vulnerable_pairs;
  std::vector<bool> signs;
  std::unordered_map<size_t, size_t> editable_pts_map;
  std::vector<size_t> editable_pts;
  vulnerable_pairs.reserve(2 * N / reserve_factor);
  signs.reserve(N / reserve_factor);
  editable_pts_map.reserve(2 * N / reserve_factor);
  editable_pts.reserve(2 * N / reserve_factor);
  T lower_bound = b - 2 * std::sqrt(3) * xi;
  T upper_bound = b + 2 * std::sqrt(3) * xi;
  T lower_bound_sq = (lower_bound < 0) ? 0 : lower_bound * lower_bound;
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  static constexpr int neighbor_offsets[13][3] = {
      {1, 0, 0},  {0, 1, 0},  {0, 0, 1},  {1, 1, 0},  {1, -1, 0},
      {1, 0, 1},  {1, 0, -1}, {0, 1, 1},  {0, 1, -1}, {1, 1, 1},
      {1, 1, -1}, {1, -1, 1}, {1, -1, -1}};

  auto distanceSquared = [&](size_t i, size_t j) {
    T dx = org_xx[i] - org_xx[j];
    T dy = org_yy[i] - org_yy[j];
    T dz = org_zz[i] - org_zz[j];
    return dx * dx + dy * dy + dz * dz;
  };
  auto addPt = [&](size_t i) {
    if (i >= eff_N_local) return;
    if (editable_pts_map.find(i) == editable_pts_map.end()) {
      editable_pts_map[i] = editable_pts.size();
      editable_pts.push_back(i);
    }
  };
  auto checkAndAddPair = [&](size_t i, size_t j) {
    if (i >= eff_N_local && j >= eff_N_local) return;
    T dist_sq = distanceSquared(i, j);
    if (dist_sq > lower_bound_sq && dist_sq <= upper_bound_sq) {
      vulnerable_pairs.push_back(i);
      vulnerable_pairs.push_back(j);
      signs.push_back(dist_sq > sign_bound_sq);
      addPt(i);
      addPt(j);
    }
  };

  // --- Pass 1: VP detection ---
  for (size_t ci = 0; ci < non_empty_ids.size(); ++ci) {
    size_t id = non_empty_ids[ci];

    // Recover grid coordinates from linear cell ID
    size_t id_x = id % compressed.grid_dim_x;
    size_t id_y = (id / compressed.grid_dim_x) % compressed.grid_dim_y;
    size_t id_z = id / dim_xy_3;

    auto &indices = cell_map[id];

    // Find vulnerable pairs with distances in (b - 2 * xi, b + 2 * xi]
    // Check pairs within the same cell
    for (size_t i = 0; i < indices.size(); ++i) {
      for (size_t j = i + 1; j < indices.size(); ++j) {
        checkAndAddPair(indices[i], indices[j]);
      }
    }

    // Check pairs with neighboring cells
    for (const auto &offset : neighbor_offsets) {
      int nx = id_x + offset[0];
      int ny = id_y + offset[1];
      int nz = id_z + offset[2];

      if (nx < 0 || nx >= static_cast<int>(compressed.grid_dim_x) ||
          ny < 0 || ny >= static_cast<int>(compressed.grid_dim_y) ||
          nz < 0 || nz >= static_cast<int>(compressed.grid_dim_z))
        continue;

      size_t neighbor_id =
          nz * dim_xy_3 + ny * compressed.grid_dim_x + nx;
      auto nit = cell_map.find(neighbor_id);
      if (nit == cell_map.end()) continue;
      const auto &n_indices = nit->second;

      for (size_t i : indices) {
        for (size_t j : n_indices) {
          checkAndAddPair(i, j);
        }
      }
    }
  }

  phase_end = std::chrono::high_resolution_clock::now();
  printf("[Timer] VP detection: %f seconds\n",
         std::chrono::duration<double>(phase_end - phase_start).count());
  phase_start = phase_end;

  // --- Pass 2: Reorder + compress ---
  std::vector<bool> visited(N, false);
  size_t prev_cell = 0;

  for (size_t ci = 0; ci < non_empty_ids.size(); ++ci) {
    size_t id = non_empty_ids[ci];

    // Insert zeros for empty cells before this one
    if (id > prev_cell)
      result.quant_codes.insert(result.quant_codes.end(),
                                 id - prev_cell, static_cast<UInt>(0));

    size_t id_x = id % compressed.grid_dim_x;
    size_t id_y = (id / compressed.grid_dim_x) % compressed.grid_dim_y;
    size_t id_z = id / dim_xy_3;

    auto &indices = cell_map[id];

    // Compress particles during reordering by k-d tree or Morton code
    if constexpr (Mode == OrderMode::KD_TREE) {
      T cell_center_x = min_x + (id_x + T(0.5)) * grid_len;
      T cell_center_y = min_y + (id_y + T(0.5)) * grid_len;
      T cell_center_z = min_z + (id_z + T(0.5)) * grid_len;

      compressParticles3D_KDTree(
          org_xx, org_yy, org_zz, indices, cell_center_x, cell_center_y,
          cell_center_z, result, compressed, visited);
    } else {
      T cell_min_corner_x = min_x + id_x * grid_len;
      T cell_min_corner_y = min_y + id_y * grid_len;
      T cell_min_corner_z = min_z + id_z * grid_len;
      compressParticles3D_Morton(org_xx, org_yy, org_zz, indices,
                                   cell_min_corner_x, cell_min_corner_y,
                                   cell_min_corner_z, result, compressed);
    }
    prev_cell = id + 1;
  }

  // Trailing empty cells
  {
    size_t total_cells = (size_t)compressed.grid_dim_x *
                         compressed.grid_dim_y * compressed.grid_dim_z;
    if (total_cells > prev_cell)
      result.quant_codes.insert(result.quant_codes.end(),
                                 total_cells - prev_cell,
                                 static_cast<UInt>(0));
  }

  phase_end = std::chrono::high_resolution_clock::now();
  printf("[Timer] Reorder + compress: %f seconds\n",
         std::chrono::duration<double>(phase_end - phase_start).count());
  phase_start = phase_end;

  printf("Number of vulnerable pairs: %zu\n", vulnerable_pairs.size() / 2);
  printf("Number of editable particles: %zu\n", editable_pts.size());

  // Count violated pairs before PGD
  auto countViolations3D = [&]() {
    size_t violations = 0;
    T b_sq = b * b;
    for (size_t k = 0; k < vulnerable_pairs.size(); k += 2) {
      size_t p = vulnerable_pairs[k];
      size_t q = vulnerable_pairs[k + 1];
      T dx = result.decomp_xx[p] - result.decomp_xx[q];
      T dy = result.decomp_yy[p] - result.decomp_yy[q];
      T dz = result.decomp_zz[p] - result.decomp_zz[q];
      T dist_sq = dx * dx + dy * dy + dz * dz;
      if ((dist_sq > b_sq) != signs[k / 2])
        violations++;
    }
    return violations;
  };
  printf("Violated pairs before editing: %zu\n", countViolations3D());

  if (isPGD) {
    // Projected gradient descent
    std::vector<T> edit_x;
    std::vector<T> edit_y;
    std::vector<T> edit_z;
    size_t iter_used = 0;
    T final_loss = 0;
    projectedGradientDescent3D(org_xx, org_yy, org_zz, vulnerable_pairs, signs,
                               editable_pts_map, editable_pts, result.decomp_xx,
                               result.decomp_yy, result.decomp_zz, edit_x,
                               edit_y, edit_z, b, xi, iter_used, final_loss);

    // Quantize edits (min: -2 * xi, max 2 * xi)
    // Re-apply quantized edits to get the true decompressed values meanwhile
    std::vector<UInt2> quant_edits(3 * N,
                                   static_cast<UInt2>(((1 << m) - 1) / 2));
    T quant_norm = ((1 << m) - 1) / (4 * xi);
    T dequant_norm = (4 * xi) / ((1 << m) - 1);
    for (size_t i = 0; i < N; ++i) {
      size_t idx = result.visit_order[i];

      auto it = editable_pts_map.find(idx);
      if (it != editable_pts_map.end()) {
        size_t edit_idx = it->second;

        // quantize
        UInt2 quant_edit_x = (edit_x[edit_idx] + 2 * xi) * quant_norm;
        UInt2 quant_edit_y = (edit_y[edit_idx] + 2 * xi) * quant_norm;
        UInt2 quant_edit_z = (edit_z[edit_idx] + 2 * xi) * quant_norm;
        quant_edits[3 * i] = quant_edit_x;
        quant_edits[3 * i + 1] = quant_edit_y;
        quant_edits[3 * i + 2] = quant_edit_z;

        // dequantize and apply to base values directly
        T dequant_edit_x = static_cast<T>(quant_edit_x) * dequant_norm - 2 * xi;
        T dequant_edit_y = static_cast<T>(quant_edit_y) * dequant_norm - 2 * xi;
        T dequant_edit_z = static_cast<T>(quant_edit_z) * dequant_norm - 2 * xi;
        result.decomp_xx[idx] += dequant_edit_x - edit_x[edit_idx];
        result.decomp_yy[idx] += dequant_edit_y - edit_y[edit_idx];
        result.decomp_zz[idx] += dequant_edit_z - edit_z[edit_idx];
      }
    }

    phase_end = std::chrono::high_resolution_clock::now();
    printf("[Timer] PGD + edit quantization: %f seconds\n",
           std::chrono::duration<double>(phase_end - phase_start).count());
    phase_start = phase_end;

    printf("Violated pairs after PGD: %zu\n", countViolations3D());
    printf("Number of iterations: %zu\n", iter_used);
    printf("PGD final loss: %f\n", final_loss);

    // Pack flag variables
    std::vector<uint8_t> packed_lossless_flag = packBits(result.lossless_flag);
    compressed.size_flag = packed_lossless_flag.size();
    compressed.size_quant = result.quant_codes.size();

    // Lossless compression
    compressed.compressed_lossless_flag =
        huffmanZstdCompress(packed_lossless_flag, compressed.code_table_flag,
                            compressed.bit_stream_size_flag);
    compressed.compressed_quant_codes =
        huffmanZstdCompress(result.quant_codes, compressed.code_table_quant,
                            compressed.bit_stream_size_quant);
    compressed.compressed_quant_edits =
        huffmanZstdCompress(quant_edits, compressed.code_table_edit,
                            compressed.bit_stream_size_edit);
    compressed.size_edit = 3 * N;

    phase_end = std::chrono::high_resolution_clock::now();
    printf("[Timer] Huffman + ZSTD encoding: %f seconds\n",
           std::chrono::duration<double>(phase_end - phase_start).count());
  } else {
    // Lossless edit mode: store editable points' original values directly
    std::vector<bool> edit_flag(N, false);
    for (size_t i = 0; i < N; ++i) {
      size_t idx = result.visit_order[i];
      if (editable_pts_map.count(idx)) {
        edit_flag[i] = true;
        compressed.lossless_edit_values.push_back(org_xx[idx]);
        compressed.lossless_edit_values.push_back(org_yy[idx]);
        compressed.lossless_edit_values.push_back(org_zz[idx]);
        result.decomp_xx[idx] = org_xx[idx];
        result.decomp_yy[idx] = org_yy[idx];
        result.decomp_zz[idx] = org_zz[idx];
      }
    }
    std::vector<uint8_t> packed_flag = packBits(edit_flag);
    compressed.size_edit_flag = packed_flag.size();
    compressed.compressed_lossless_edit_flag =
        huffmanZstdCompress(packed_flag, compressed.code_table_edit_flag,
                            compressed.bit_stream_size_edit_flag);
    compressed.size_edit = 0;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> comp_time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.compressed_quant_edits.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      sizeof(T) * compressed.lossless_values.size() +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  T compression_ratio = static_cast<T>(sizeof(T) * 3 * N) / compressed_size;
  T bpp = 3 * 8 * sizeof(T) / compression_ratio;
  // #9: OpenMP parallel error computation
  T mae = 0;
  T mse = 0;
  #pragma omp parallel for reduction(max:mae) reduction(+:mse) schedule(static)
  for (size_t i = 0; i < N; ++i) {
    T err_x = std::abs(result.decomp_xx[i] - org_xx[i]);
    T err_y = std::abs(result.decomp_yy[i] - org_yy[i]);
    T err_z = std::abs(result.decomp_zz[i] - org_zz[i]);
    if (err_x > mae)
      mae = err_x;
    if (err_y > mae)
      mae = err_y;
    if (err_z > mae)
      mae = err_z;
    mse += err_x * err_x + err_y * err_y + err_z * err_z;
  }
  mse /= (3 * N);
  T rmse = std::sqrt(mse);
  T max_range = std::max(range_x, range_y);
  T nrmse = rmse / std::max(max_range, range_z);
  T psnr = -20 * std::log10(nrmse);

  printf("Compression time: %f seconds\n", comp_time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

template <typename T, OrderMode Mode>
void compressParticles3D(const T *org_xx, const T *org_yy, const T *org_zz,
                         T min_x, T range_x, T min_y, T range_y, T min_z,
                         T range_z, size_t N, T xi, T b,
                         CompressionResults3D<T> &result,
                         CompressedData<T> &compressed) {

  auto start = std::chrono::high_resolution_clock::now();

  if (N == 0)
    return;

  result.xi = xi;
  result.b = b;

  // Reserve space for compression results
  result.lossless_flag.reserve(3 * N);
  result.quant_codes.reserve(3 * N);
  compressed.lossless_values.reserve(3 * N);
  result.visit_order.reserve(N);
  result.decomp_xx = new T[N];
  result.decomp_yy = new T[N];
  result.decomp_zz = new T[N];

  // Compute grid dimensions
  T grid_len = b + 2 * std::sqrt(3) * xi;
  compressed.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  compressed.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));
  compressed.grid_dim_z =
      std::max(1, static_cast<int>(std::ceil(range_z / grid_len)));

  // Store grid min
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.grid_min_z = min_z;
  compressed.xi = xi;
  compressed.b = b;

  // Partition particles into cells (sparse)
  std::unordered_map<size_t, std::vector<size_t>> cell_map;
  size_t dim_xy_cp3 = (size_t)compressed.grid_dim_x * compressed.grid_dim_y;

  for (size_t i = 0; i < N; ++i) {
    size_t cid_x =
        static_cast<size_t>(std::floor((org_xx[i] - min_x) / grid_len));
    size_t cid_y =
        static_cast<size_t>(std::floor((org_yy[i] - min_y) / grid_len));
    size_t cid_z =
        static_cast<size_t>(std::floor((org_zz[i] - min_z) / grid_len));
    size_t id = cid_z * dim_xy_cp3 + cid_y * compressed.grid_dim_x + cid_x;
    cell_map[id].push_back(i);
  }

  // Collect and sort non-empty cell IDs
  std::vector<size_t> non_empty_ids;
  non_empty_ids.reserve(cell_map.size());
  for (const auto &entry : cell_map)
    non_empty_ids.push_back(entry.first);
  std::sort(non_empty_ids.begin(), non_empty_ids.end());

  size_t total_cells_cp3 = dim_xy_cp3 * compressed.grid_dim_z;

  // Process each cell (Compression)
  switch (Mode) {
  case OrderMode::KD_TREE: {
    std::vector<bool> visited(N, false);
    size_t prev_cell = 0;

    for (size_t ci = 0; ci < non_empty_ids.size(); ++ci) {
      size_t id = non_empty_ids[ci];
      if (id > prev_cell)
        result.quant_codes.insert(result.quant_codes.end(),
                                   id - prev_cell, static_cast<UInt>(0));

      size_t id_x = id % compressed.grid_dim_x;
      size_t id_y = (id / compressed.grid_dim_x) % compressed.grid_dim_y;
      size_t id_z = id / dim_xy_cp3;

      auto &indices = cell_map[id];

      T cell_center_x = min_x + (id_x + T(0.5)) * grid_len;
      T cell_center_y = min_y + (id_y + T(0.5)) * grid_len;
      T cell_center_z = min_z + (id_z + T(0.5)) * grid_len;

      compressParticles3D_KDTree(
          org_xx, org_yy, org_zz, indices, cell_center_x, cell_center_y,
          cell_center_z, result, compressed, visited);
      prev_cell = id + 1;
    }
    if (total_cells_cp3 > prev_cell)
      result.quant_codes.insert(result.quant_codes.end(),
                                 total_cells_cp3 - prev_cell,
                                 static_cast<UInt>(0));
    break;
  }

  case OrderMode::MORTON_CODE: {
    size_t prev_cell = 0;

    for (size_t ci = 0; ci < non_empty_ids.size(); ++ci) {
      size_t id = non_empty_ids[ci];
      if (id > prev_cell)
        result.quant_codes.insert(result.quant_codes.end(),
                                   id - prev_cell, static_cast<UInt>(0));

      size_t id_x = id % compressed.grid_dim_x;
      size_t id_y = (id / compressed.grid_dim_x) % compressed.grid_dim_y;
      size_t id_z = id / dim_xy_cp3;

      auto &indices = cell_map[id];

      T cell_min_corner_x = min_x + id_x * grid_len;
      T cell_min_corner_y = min_y + id_y * grid_len;
      T cell_min_corner_z = min_z + id_z * grid_len;
      compressParticles3D_Morton(org_xx, org_yy, org_zz, indices,
                                   cell_min_corner_x, cell_min_corner_y,
                                   cell_min_corner_z, result, compressed);
      prev_cell = id + 1;
    }
    if (total_cells_cp3 > prev_cell)
      result.quant_codes.insert(result.quant_codes.end(),
                                 total_cells_cp3 - prev_cell,
                                 static_cast<UInt>(0));
    break;
  }
  }

  // Pack flag variables
  std::vector<uint8_t> packed_lossless_flag = packBits(result.lossless_flag);
  compressed.size_flag = packed_lossless_flag.size();
  compressed.size_quant = result.quant_codes.size();

  // Lossless compression
  compressed.compressed_lossless_flag =
      huffmanZstdCompress(packed_lossless_flag, compressed.code_table_flag,
                          compressed.bit_stream_size_flag);
  compressed.compressed_quant_codes =
      huffmanZstdCompress(result.quant_codes, compressed.code_table_quant,
                          compressed.bit_stream_size_quant);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> comp_time = end - start;

  size_t compressed_size =
      compressed.compressed_lossless_flag.size() +
      compressed.compressed_quant_codes.size() +
      compressed.code_table_flag.size() * 9 +
      compressed.code_table_quant.size() * (sizeof(UInt) + 8) +
      sizeof(T) * compressed.lossless_values.size();
  T compression_ratio = static_cast<T>(sizeof(T) * 3 * N) / compressed_size;
  T bpp = 3 * 8 * sizeof(T) / compression_ratio;
  // #9: OpenMP parallel error computation
  T mae = 0;
  T mse = 0;
  #pragma omp parallel for reduction(max:mae) reduction(+:mse) schedule(static)
  for (size_t i = 0; i < N; ++i) {
    T err_x = std::abs(result.decomp_xx[i] - org_xx[i]);
    T err_y = std::abs(result.decomp_yy[i] - org_yy[i]);
    T err_z = std::abs(result.decomp_zz[i] - org_zz[i]);
    if (err_x > mae)
      mae = err_x;
    if (err_y > mae)
      mae = err_y;
    if (err_z > mae)
      mae = err_z;
    mse += err_x * err_x + err_y * err_y + err_z * err_z;
  }
  mse /= (3 * N);
  T rmse = std::sqrt(mse);
  T max_range = std::max(range_x, range_y);
  T nrmse = rmse / std::max(max_range, range_z);
  T psnr = -20 * std::log10(nrmse);

  printf("Compression time: %f seconds\n", comp_time.count());
  printf("Compression ratio: %f\n", compression_ratio);
  printf("BPP: %f\n", bpp);
  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

template <typename T>
void editParticles3D(const T *org_xx, const T *org_yy, const T *org_zz, T min_x,
                     T range_x, T min_y, T range_y, T min_z, T range_z,
                     size_t N, T xi, T b, bool isPGD,
                     CompressionResults3D<T> &result,
                     CompressedData<T> &compressed,
                     size_t N_local = 0, MPI_Comm comm = MPI_COMM_NULL) {

  auto start = std::chrono::high_resolution_clock::now();
  auto phase_start = start;
  auto phase_end = start;

  if (N == 0)
    return;

  size_t eff_N_local = (N_local == 0) ? N : N_local;

  result.xi = xi;
  result.b = b;

  // Compute grid dimensions
  T grid_len = b + 2 * std::sqrt(3) * xi;
  compressed.grid_dim_x =
      std::max(1, static_cast<int>(std::ceil(range_x / grid_len)));
  compressed.grid_dim_y =
      std::max(1, static_cast<int>(std::ceil(range_y / grid_len)));
  compressed.grid_dim_z =
      std::max(1, static_cast<int>(std::ceil(range_z / grid_len)));

  // Store grid min
  compressed.grid_min_x = min_x;
  compressed.grid_min_y = min_y;
  compressed.grid_min_z = min_z;
  compressed.xi = xi;
  compressed.b = b;
  compressed.N_local = N_local;

  // Partition particles into cells (sparse)
  std::unordered_map<size_t, std::vector<size_t>> cell_map;
  size_t dim_xy_e3 = (size_t)compressed.grid_dim_x * compressed.grid_dim_y;

  for (size_t i = 0; i < N; ++i) {
    size_t cid_x =
        static_cast<size_t>(std::floor((org_xx[i] - min_x) / grid_len));
    size_t cid_y =
        static_cast<size_t>(std::floor((org_yy[i] - min_y) / grid_len));
    size_t cid_z =
        static_cast<size_t>(std::floor((org_zz[i] - min_z) / grid_len));
    size_t id = cid_z * dim_xy_e3 + cid_y * compressed.grid_dim_x + cid_x;
    cell_map[id].push_back(i);
  }

  phase_end = std::chrono::high_resolution_clock::now();
  printf("[Timer] Grid partitioning (edit): %f seconds\n",
         std::chrono::duration<double>(phase_end - phase_start).count());
  phase_start = phase_end;

  // Process each cell (Vulnerable pairs detection)
  std::vector<size_t> vulnerable_pairs;
  std::vector<bool> signs;
  std::unordered_map<size_t, size_t> editable_pts_map;
  std::vector<size_t> editable_pts;
  vulnerable_pairs.reserve(2 * N / reserve_factor);
  signs.reserve(N / reserve_factor);
  editable_pts_map.reserve(2 * N / reserve_factor);
  editable_pts.reserve(2 * N / reserve_factor);
  T lower_bound = b - 2 * std::sqrt(3) * xi;
  T upper_bound = b + 2 * std::sqrt(3) * xi;
  T lower_bound_sq;
  if (lower_bound < 0) {
    lower_bound_sq = 0;
  } else {
    lower_bound_sq = lower_bound * lower_bound;
  }
  T upper_bound_sq = upper_bound * upper_bound;
  T sign_bound_sq = b * b;
  static constexpr int neighbor_offsets[13][3] = {
      {1, 0, 0},  {0, 1, 0},  {0, 0, 1},  {1, 1, 0},  {1, -1, 0},
      {1, 0, 1},  {1, 0, -1}, {0, 1, 1},  {0, 1, -1}, {1, 1, 1},
      {1, 1, -1}, {1, -1, 1}, {1, -1, -1}};

  auto distanceSquared = [&](size_t i, size_t j) {
    T dx = org_xx[i] - org_xx[j];
    T dy = org_yy[i] - org_yy[j];
    T dz = org_zz[i] - org_zz[j];
    return dx * dx + dy * dy + dz * dz;
  };
  auto addPt = [&](size_t i) {
    if (i >= eff_N_local) return;
    if (editable_pts_map.find(i) == editable_pts_map.end()) {
      editable_pts_map[i] = editable_pts.size();
      editable_pts.push_back(i);
    }
  };
  auto checkAndAddPair = [&](size_t i, size_t j) {
    if (i >= eff_N_local && j >= eff_N_local) return;
    T dist_sq = distanceSquared(i, j);
    if (dist_sq > lower_bound_sq && dist_sq <= upper_bound_sq) {
      vulnerable_pairs.push_back(i);
      vulnerable_pairs.push_back(j);
      signs.push_back(dist_sq > sign_bound_sq);
      addPt(i);
      addPt(j);
    }
  };

  std::vector<bool> visited(N, false);

  for (auto &kv : cell_map) {
    size_t id = kv.first;
    auto &indices = kv.second;

    size_t id_x = id % compressed.grid_dim_x;
    size_t id_y = (id / compressed.grid_dim_x) % compressed.grid_dim_y;
    size_t id_z = id / dim_xy_e3;

    // Find vulnerable links with distances in (b - 2 * xi, b + 2 * xi]
    for (size_t i = 0; i < indices.size(); ++i) {
      for (size_t j = i + 1; j < indices.size(); ++j) {
        checkAndAddPair(indices[i], indices[j]);
      }
    }

    // Check pairs with neighboring cells
    for (const auto &offset : neighbor_offsets) {
      int nx = id_x + offset[0];
      int ny = id_y + offset[1];
      int nz = id_z + offset[2];

      if (nx < 0 || nx >= static_cast<int>(compressed.grid_dim_x) ||
          ny < 0 || ny >= static_cast<int>(compressed.grid_dim_y) ||
          nz < 0 || nz >= static_cast<int>(compressed.grid_dim_z))
        continue;

      size_t neighbor_id =
          nz * dim_xy_e3 + ny * compressed.grid_dim_x + nx;
      auto nit = cell_map.find(neighbor_id);
      if (nit == cell_map.end()) continue;
      const auto &n_indices = nit->second;

      for (size_t i : indices) {
        for (size_t j : n_indices) {
          checkAndAddPair(i, j);
        }
      }
    }
  }

  phase_end = std::chrono::high_resolution_clock::now();
  printf("[Timer] VP detection (edit): %f seconds\n",
         std::chrono::duration<double>(phase_end - phase_start).count());
  phase_start = phase_end;

  printf("Number of vulnerable pairs: %zu\n", vulnerable_pairs.size() / 2);
  printf("Number of editable particles: %zu\n", editable_pts.size());

  // Count violated pairs before PGD
  auto countViolations3D = [&]() {
    size_t violations = 0;
    T b_sq = b * b;
    for (size_t k = 0; k < vulnerable_pairs.size(); k += 2) {
      size_t p = vulnerable_pairs[k];
      size_t q = vulnerable_pairs[k + 1];
      T dx = result.decomp_xx[p] - result.decomp_xx[q];
      T dy = result.decomp_yy[p] - result.decomp_yy[q];
      T dz = result.decomp_zz[p] - result.decomp_zz[q];
      T dist_sq = dx * dx + dy * dy + dz * dz;
      if ((dist_sq > b_sq) != signs[k / 2])
        violations++;
    }
    return violations;
  };
  printf("Violated pairs before editing: %zu\n", countViolations3D());

  if (isPGD) {
    // Projected gradient descent
    std::vector<T> edit_x;
    std::vector<T> edit_y;
    std::vector<T> edit_z;
    size_t iter_used = 0;
    T final_loss = 0;
    projectedGradientDescent3D(org_xx, org_yy, org_zz, vulnerable_pairs, signs,
                               editable_pts_map, editable_pts, result.decomp_xx,
                               result.decomp_yy, result.decomp_zz, edit_x,
                               edit_y, edit_z, b, xi, iter_used, final_loss,
                               comm);

    // Quantize edits (min: -2 * xi, max 2 * xi)
    std::vector<UInt2> quant_edits(3 * N,
                                   static_cast<UInt2>(((1 << m) - 1) / 2));
    T quant_norm = ((1 << m) - 1) / (4 * xi);
    T dequant_norm = (4 * xi) / ((1 << m) - 1);

    for (const auto &[i, ii] : editable_pts_map) {
      UInt2 quant_edit_x = (edit_x[ii] + 2 * xi) * quant_norm;
      UInt2 quant_edit_y = (edit_y[ii] + 2 * xi) * quant_norm;
      UInt2 quant_edit_z = (edit_z[ii] + 2 * xi) * quant_norm;
      quant_edits[3 * i] = quant_edit_x;
      quant_edits[3 * i + 1] = quant_edit_y;
      quant_edits[3 * i + 2] = quant_edit_z;

      // dequantize and apply to base values directly
      T dequant_edit_x = static_cast<T>(quant_edit_x) * dequant_norm - 2 * xi;
      T dequant_edit_y = static_cast<T>(quant_edit_y) * dequant_norm - 2 * xi;
      T dequant_edit_z = static_cast<T>(quant_edit_z) * dequant_norm - 2 * xi;
      result.decomp_xx[i] += dequant_edit_x - edit_x[ii];
      result.decomp_yy[i] += dequant_edit_y - edit_y[ii];
      result.decomp_zz[i] += dequant_edit_z - edit_z[ii];
    }
    phase_end = std::chrono::high_resolution_clock::now();
    printf("[Timer] PGD + edit quantization (edit): %f seconds\n",
           std::chrono::duration<double>(phase_end - phase_start).count());
    phase_start = phase_end;

    printf("Violated pairs after PGD: %zu\n", countViolations3D());
    printf("Number of iterations: %zu\n", iter_used);
    printf("PGD final loss: %f\n", final_loss);

    compressed.compressed_quant_edits =
        huffmanZstdCompress(quant_edits, compressed.code_table_edit,
                            compressed.bit_stream_size_edit);
    compressed.size_edit = 3 * N;

    phase_end = std::chrono::high_resolution_clock::now();
    printf("[Timer] Huffman + ZSTD encoding (edit): %f seconds\n",
           std::chrono::duration<double>(phase_end - phase_start).count());

  } else {
    // Lossless edit mode: store original values in original particle order
    std::vector<bool> edit_flag(N, false);
    for (size_t i = 0; i < N; ++i) {
      if (editable_pts_map.count(i)) {
        edit_flag[i] = true;
        compressed.lossless_edit_values.push_back(org_xx[i]);
        compressed.lossless_edit_values.push_back(org_yy[i]);
        compressed.lossless_edit_values.push_back(org_zz[i]);
        result.decomp_xx[i] = org_xx[i];
        result.decomp_yy[i] = org_yy[i];
        result.decomp_zz[i] = org_zz[i];
      }
    }
    std::vector<uint8_t> packed_flag = packBits(edit_flag);
    compressed.size_edit_flag = packed_flag.size();
    compressed.compressed_lossless_edit_flag =
        huffmanZstdCompress(packed_flag, compressed.code_table_edit_flag,
                            compressed.bit_stream_size_edit_flag);
    compressed.size_edit = 0;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> comp_time = end - start;

  size_t additional_size =
      compressed.compressed_quant_edits.size() +
      compressed.code_table_edit.size() * (sizeof(UInt2) + 8) +
      compressed.compressed_lossless_edit_flag.size() +
      compressed.code_table_edit_flag.size() * 9 +
      sizeof(T) * compressed.lossless_edit_values.size();
  // #9: OpenMP parallel error computation
  T mae = 0;
  T mse = 0;
  #pragma omp parallel for reduction(max:mae) reduction(+:mse) schedule(static)
  for (size_t i = 0; i < N; ++i) {
    T err_x = std::abs(result.decomp_xx[i] - org_xx[i]);
    T err_y = std::abs(result.decomp_yy[i] - org_yy[i]);
    T err_z = std::abs(result.decomp_zz[i] - org_zz[i]);
    if (err_x > mae)
      mae = err_x;
    if (err_y > mae)
      mae = err_y;
    if (err_z > mae)
      mae = err_z;
    mse += err_x * err_x + err_y * err_y + err_z * err_z;
  }
  mse /= (3 * N);
  T rmse = std::sqrt(mse);
  T max_range = std::max(range_x, range_y);
  T nrmse = rmse / std::max(max_range, range_z);
  T psnr = -20 * std::log10(nrmse);

  printf("Compression time: %f seconds\n", comp_time.count());
  printf("Additional storage: %zu bytes\n", additional_size);
  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

// Decompression functions
template <typename T, OrderMode Mode>
void decompressWithEditParticles2D(const CompressedData<T> &compressed,
                                   T *decomp_xx, T *decomp_yy, size_t N, T xi,
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
  std::vector<UInt2> quant_edits = huffmanZstdDecompress(
      compressed.compressed_quant_edits, compressed.code_table_edit,
      compressed.size_edit, compressed.bit_stream_size_edit);

  T grid_len = b + 2 * std::sqrt(2) * xi;
  T min_x = compressed.grid_min_x;
  T min_y = compressed.grid_min_y;

  size_t i_f{0}, i_q{0}, i_l{0},
      i_out{0}; // counters for flags, quants, lossless, output
  T edit;
  T norm = (4 * xi) / ((1 << m) - 1);

  for (size_t id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
    for (size_t id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {

      T prev_x, prev_y;
      if constexpr (Mode == OrderMode::KD_TREE) {
        prev_x = min_x + (id_x + T(0.5)) * grid_len;
        prev_y = min_y + (id_y + T(0.5)) * grid_len;
      } else {
        prev_x = min_x + id_x * grid_len;
        prev_y = min_y + id_y * grid_len;
      }

      while (quant_codes[i_q] > 0) {
        edit = static_cast<T>(quant_edits[i_f]) * norm - 2 * xi;
        if (lossless_flag[i_f++]) {
          prev_x = compressed.lossless_values[i_l++];
        } else {
          prev_x += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
        }
        decomp_xx[i_out] = prev_x + edit;
        edit = static_cast<T>(quant_edits[i_f]) * norm - 2 * xi;
        if (lossless_flag[i_f++]) {
          prev_y = compressed.lossless_values[i_l++];
        } else {
          prev_y += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
        }
        decomp_yy[i_out++] = prev_y + edit;
        if (quant_codes[i_q] == static_cast<UInt>(1 << m))
          break;
      }
      i_q++;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

template <typename T, OrderMode Mode>
void decompressParticles2D(const CompressedData<T> &compressed, T *decomp_xx,
                           T *decomp_yy, size_t N, T xi, T b) {
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

  size_t i_f{0}, i_q{0}, i_l{0},
      i_out{0}; // counters for flags, quants, lossless, output

  for (size_t id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
    for (size_t id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {

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
}

template <typename T>
void reconstructEditParticles2D(const CompressedData<T> &compressed,
                                T *decomp_xx, T *decomp_yy, size_t N, T xi) {
  auto start = std::chrono::high_resolution_clock::now();

  if (compressed.size_edit == 0)
    return;

  std::vector<UInt2> quant_edits = huffmanZstdDecompress(
      compressed.compressed_quant_edits, compressed.code_table_edit,
      compressed.size_edit, compressed.bit_stream_size_edit);

  T norm = (4 * xi) / ((1 << m) - 1);
  T edit;

  for (size_t i = 0; i < N; ++i) {
    edit = static_cast<T>(quant_edits[2 * i]) * norm - 2 * xi;
    decomp_xx[i] += edit;
    edit = static_cast<T>(quant_edits[2 * i + 1]) * norm - 2 * xi;
    decomp_yy[i] += edit;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

template <typename T, OrderMode Mode>
void decompressWithEditParticles3D(const CompressedData<T> &compressed,
                                   T *decomp_xx, T *decomp_yy, T *decomp_zz,
                                   size_t N, T xi, T b) {
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
  std::vector<UInt2> quant_edits = huffmanZstdDecompress(
      compressed.compressed_quant_edits, compressed.code_table_edit,
      compressed.size_edit, compressed.bit_stream_size_edit);

  T grid_len = b + 2 * std::sqrt(3) * xi;
  T min_x = compressed.grid_min_x;
  T min_y = compressed.grid_min_y;
  T min_z = compressed.grid_min_z;

  size_t i_f{0}, i_q{0}, i_l{0},
      i_out{0}; // counters for flags, quants, lossless, output
  T edit;
  T norm = (4 * xi) / ((1 << m) - 1);

  for (size_t id_z = 0; id_z < compressed.grid_dim_z; ++id_z) {
    for (size_t id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
      for (size_t id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {

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
          edit = static_cast<T>(quant_edits[i_f]) * norm - 2 * xi;
          if (lossless_flag[i_f++]) {
            prev_x = compressed.lossless_values[i_l++];
          } else {
            prev_x += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_xx[i_out] = prev_x + edit;
          edit = static_cast<T>(quant_edits[i_f]) * norm - 2 * xi;
          if (lossless_flag[i_f++]) {
            prev_y = compressed.lossless_values[i_l++];
          } else {
            prev_y += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_yy[i_out] = prev_y + edit;
          edit = static_cast<T>(quant_edits[i_f]) * norm - 2 * xi;
          if (lossless_flag[i_f++]) {
            prev_z = compressed.lossless_values[i_l++];
          } else {
            prev_z += dequantize(static_cast<int>(quant_codes[i_q++]), xi);
          }
          decomp_zz[i_out++] = prev_z + edit;
          if (quant_codes[i_q] == static_cast<UInt>(1 << m))
            break;
        }
        i_q++;
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

template <typename T, OrderMode Mode>
void decompressParticles3D(const CompressedData<T> &compressed, T *decomp_xx,
                           T *decomp_yy, T *decomp_zz, size_t N, T xi, T b) {
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

  size_t i_f{0}, i_q{0}, i_l{0},
      i_out{0}; // counters for flags, quants, lossless, output

  for (size_t id_z = 0; id_z < compressed.grid_dim_z; ++id_z) {
    for (size_t id_y = 0; id_y < compressed.grid_dim_y; ++id_y) {
      for (size_t id_x = 0; id_x < compressed.grid_dim_x; ++id_x) {

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
}

template <typename T>
void reconstructEditParticles3D(const CompressedData<T> &compressed,
                                T *decomp_xx, T *decomp_yy, T *decomp_zz,
                                size_t N, T xi) {
  auto start = std::chrono::high_resolution_clock::now();

  if (compressed.size_edit == 0)
    return;

  std::vector<UInt2> quant_edits = huffmanZstdDecompress(
      compressed.compressed_quant_edits, compressed.code_table_edit,
      compressed.size_edit, compressed.bit_stream_size_edit);

  T norm = (4 * xi) / ((1 << m) - 1);
  T edit;

  for (size_t i = 0; i < N; ++i) {
    edit = static_cast<T>(quant_edits[3 * i]) * norm - 2 * xi;
    decomp_xx[i] += edit;
    edit = static_cast<T>(quant_edits[3 * i + 1]) * norm - 2 * xi;
    decomp_yy[i] += edit;
    edit = static_cast<T>(quant_edits[3 * i + 2]) * norm - 2 * xi;
    decomp_zz[i] += edit;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decomp_time = end - start;
  printf("Decompression time: %f seconds\n", decomp_time.count());
}

template <typename T>
void applyLosslessEdits2D(const CompressedData<T> &compressed, T *decomp_xx,
                          T *decomp_yy, size_t N) {
  if (compressed.compressed_lossless_edit_flag.empty())
    return;
  auto packed_mask = huffmanZstdDecompress(
      compressed.compressed_lossless_edit_flag, compressed.code_table_edit_flag,
      compressed.size_edit_flag, compressed.bit_stream_size_edit_flag);
  auto edit_flag = unpackBits(packed_mask, N);
  size_t j = 0;
  for (size_t i = 0; i < N; ++i) {
    if (edit_flag[i]) {
      decomp_xx[i] = compressed.lossless_edit_values[2 * j];
      decomp_yy[i] = compressed.lossless_edit_values[2 * j + 1];
      ++j;
    }
  }
}

template <typename T>
void applyLosslessEdits3D(const CompressedData<T> &compressed, T *decomp_xx,
                          T *decomp_yy, T *decomp_zz, size_t N) {
  if (compressed.compressed_lossless_edit_flag.empty())
    return;
  auto packed_mask = huffmanZstdDecompress(
      compressed.compressed_lossless_edit_flag, compressed.code_table_edit_flag,
      compressed.size_edit_flag, compressed.bit_stream_size_edit_flag);
  auto edit_flag = unpackBits(packed_mask, N);
  size_t j = 0;
  for (size_t i = 0; i < N; ++i) {
    if (edit_flag[i]) {
      decomp_xx[i] = compressed.lossless_edit_values[3 * j];
      decomp_yy[i] = compressed.lossless_edit_values[3 * j + 1];
      decomp_zz[i] = compressed.lossless_edit_values[3 * j + 2];
      ++j;
    }
  }
}
