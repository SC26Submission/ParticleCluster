#pragma once

#include "config.cuh"
#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(error));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int num_threads = 256;

// ==============================================================================
// Math helpers
// ==============================================================================

template <typename T> __device__ __forceinline__ T abs_templated(T x);

template <> __device__ __forceinline__ float abs_templated<float>(float x) {
  return fabsf(x);
}

template <> __device__ __forceinline__ double abs_templated<double>(double x) {
  return fabs(x);
}

template <typename T> __device__ __forceinline__ T sqrt_templated(T x);

template <> __device__ __forceinline__ float sqrt_templated<float>(float x) {
  return __fsqrt_rn(x);
}

template <> __device__ __forceinline__ double sqrt_templated<double>(double x) {
  return __dsqrt_rn(x);
}

template <typename T> __device__ __forceinline__ T min_templated(T a, T b) {
  return a < b ? a : b;
}

template <typename T> __device__ __forceinline__ T max_templated(T a, T b) {
  return a > b ? a : b;
}

// ==============================================================================
// Min/max reduction utilities
// ==============================================================================

template <typename T> struct MinMaxPair {
  T min_val, max_val;
};

template <typename T> struct ToMinMaxOp {
  __device__ __forceinline__ MinMaxPair<T> operator()(const T val) const {
    return {val, val};
  }
};

template <typename T> struct MinMaxReduceOp {
  __device__ __forceinline__ MinMaxPair<T>
  operator()(const MinMaxPair<T> a, const MinMaxPair<T> b) const {
    return {a.min_val < b.min_val ? a.min_val : b.min_val,
            a.max_val > b.max_val ? a.max_val : b.max_val};
  }
};

template <typename T> struct AbsDiffOp {
  __device__ __forceinline__ T operator()(const thrust::tuple<T, T> &t) const {
    T a = thrust::get<0>(t);
    T b = thrust::get<1>(t);
    T diff = a - b;
    return (diff < 0) ? -diff : diff;
  }
};

/**
 * Compute min, max, and range of a data array using CUB reduction
 */
template <typename T>
void getRange(const T *d_data, int N, T &min_val, T &max_val, T &range_val) {
  using Pair = MinMaxPair<T>;
  auto d_in = thrust::make_transform_iterator(d_data, ToMinMaxOp<T>{});

  Pair *d_out;
  cudaMalloc(&d_out, sizeof(Pair));

  Pair init{std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()};
  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out, N,
                            MinMaxReduceOp<T>{}, init);
  cudaMalloc(&d_temp, temp_bytes);
  cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out, N,
                            MinMaxReduceOp<T>{}, init);

  Pair h_result;
  cudaMemcpy(&h_result, d_out, sizeof(Pair), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  cudaFree(d_temp);

  min_val = h_result.min_val;
  max_val = h_result.max_val;
  range_val = max_val - min_val;
}

template <typename T> T getMaxAbsErr(const T *d_org, const T *d_dst, int N) {
  auto zip_it = thrust::make_zip_iterator(thrust::make_tuple(d_org, d_dst));

  auto d_in = thrust::make_transform_iterator(zip_it, AbsDiffOp<T>{});

  T *d_out;
  cudaMalloc(&d_out, sizeof(T));

  T init = 0;
  void *d_temp = nullptr;
  size_t temp_bytes = 0;

  cub::DeviceReduce::Max(d_temp, temp_bytes, d_in, d_out, N);

  cudaMalloc(&d_temp, temp_bytes);
  cub::DeviceReduce::Max(d_temp, temp_bytes, d_in, d_out, N);

  T h_result;
  cudaMemcpy(&h_result, d_out, sizeof(T), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_out);
  cudaFree(d_temp);

  return h_result;
}

// ==============================================================================
// Particle cell partitioning kernels and functions
// ==============================================================================

static __global__ void fillCellParticles_kernel(const int *d_cell_ids,
                                                int *d_cell_offsets,
                                                int *d_cell_pts_sorted, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int cell_id = d_cell_ids[idx];
    int pos = atomicAdd(&d_cell_offsets[cell_id], 1);
    d_cell_pts_sorted[pos] = idx;
  }
}

template <typename T>
__global__ void computeParticleCellIndices2D_kernel(
    const T *__restrict__ d_org_xx, const T *__restrict__ d_org_yy,
    int *d_cell_ids, int *d_nums_cell_pts, T min_x, T min_y, T grid_len,
    int grid_dim_x, int grid_dim_y, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int id_x = min(static_cast<int>(
                       max_templated((d_org_xx[idx] - min_x) / grid_len, T(0))),
                   grid_dim_x - 1);
    int id_y = min(static_cast<int>(
                       max_templated((d_org_yy[idx] - min_y) / grid_len, T(0))),
                   grid_dim_y - 1);

    int cell_id = id_y * grid_dim_x + id_x;
    d_cell_ids[idx] = cell_id;

    if (d_nums_cell_pts) {
      atomicAdd(&d_nums_cell_pts[cell_id], 1);
    }
  }
}

template <typename T>
__global__ void computeParticleCellIndices3D_kernel(
    const T *__restrict__ d_org_xx, const T *__restrict__ d_org_yy,
    const T *__restrict__ d_org_zz, int *d_cell_ids, int *d_nums_cell_pts,
    T min_x, T min_y, T min_z, T grid_len, int grid_dim_x, int grid_dim_y,
    int grid_dim_z, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int id_x = min(static_cast<int>(
                       max_templated((d_org_xx[idx] - min_x) / grid_len, T(0))),
                   grid_dim_x - 1);
    int id_y = min(static_cast<int>(
                       max_templated((d_org_yy[idx] - min_y) / grid_len, T(0))),
                   grid_dim_y - 1);
    int id_z = min(static_cast<int>(
                       max_templated((d_org_zz[idx] - min_z) / grid_len, T(0))),
                   grid_dim_z - 1);

    int cell_id = id_z * grid_dim_x * grid_dim_y + id_y * grid_dim_x + id_x;
    d_cell_ids[idx] = cell_id;

    if (d_nums_cell_pts) {
      atomicAdd(&d_nums_cell_pts[cell_id], 1);
    }
  }
}

/**
 * Partition 2D particles into grid cells and get sorted indices
 * Allocates d_cell_start and d_cell_pts_sorted on GPU (caller must free)
 */
template <typename T>
void particlePartition2D(const T *d_org_xx, const T *d_org_yy, T min_x, T min_y,
                         T grid_len, int grid_dim_x, int grid_dim_y, int N,
                         int **d_cell_start_out, int **d_cell_pts_sorted_out) {
  int num_cells = grid_dim_x * grid_dim_y;

  int *d_nums_cell_pts;
  int *d_cell_pts_sorted, *d_cell_ids, *d_cell_start, *d_cell_offsets;
  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;
  CUDA_CHECK(cudaMalloc(&d_nums_cell_pts, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_pts_sorted, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_ids, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_start, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_offsets, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_nums_cell_pts, 0, num_cells * sizeof(int)));

  int num_blocks = (N + num_threads - 1) / num_threads;
  computeParticleCellIndices2D_kernel<<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_cell_ids, d_nums_cell_pts, min_x, min_y, grid_len,
      grid_dim_x, grid_dim_y, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes,
                                d_nums_cell_pts, d_cell_start, num_cells);
  CUDA_CHECK(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes,
                                d_nums_cell_pts, d_cell_start, num_cells);

  CUDA_CHECK(cudaMemcpy(d_cell_offsets, d_cell_start, num_cells * sizeof(int),
                        cudaMemcpyDeviceToDevice));

  fillCellParticles_kernel<<<num_blocks, num_threads>>>(
      d_cell_ids, d_cell_offsets, d_cell_pts_sorted, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_cell_ids));
  CUDA_CHECK(cudaFree(d_nums_cell_pts));
  CUDA_CHECK(cudaFree(d_cell_offsets));
  CUDA_CHECK(cudaFree(d_tmp_storage));

  *d_cell_start_out = d_cell_start;
  *d_cell_pts_sorted_out = d_cell_pts_sorted;
}

/**
 * Partition 3D particles into grid cells and get sorted indices
 * Allocates d_cell_start and d_cell_pts_sorted on GPU (caller must free)
 */
template <typename T>
void particlePartition3D(const T *d_org_xx, const T *d_org_yy,
                         const T *d_org_zz, T min_x, T min_y, T min_z,
                         T grid_len, int grid_dim_x, int grid_dim_y,
                         int grid_dim_z, int N, int **d_cell_start_out,
                         int **d_cell_pts_sorted_out) {
  int num_cells = grid_dim_x * grid_dim_y * grid_dim_z;

  int *d_nums_cell_pts;
  int *d_cell_pts_sorted, *d_cell_ids, *d_cell_start, *d_cell_offsets;
  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;
  CUDA_CHECK(cudaMalloc(&d_nums_cell_pts, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_pts_sorted, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_ids, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_start, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cell_offsets, num_cells * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_nums_cell_pts, 0, num_cells * sizeof(int)));

  int num_blocks = (N + num_threads - 1) / num_threads;
  computeParticleCellIndices3D_kernel<<<num_blocks, num_threads>>>(
      d_org_xx, d_org_yy, d_org_zz, d_cell_ids, d_nums_cell_pts, min_x, min_y,
      min_z, grid_len, grid_dim_x, grid_dim_y, grid_dim_z, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes,
                                d_nums_cell_pts, d_cell_start, num_cells);
  CUDA_CHECK(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(d_tmp_storage, tmp_storage_bytes,
                                d_nums_cell_pts, d_cell_start, num_cells);

  CUDA_CHECK(cudaMemcpy(d_cell_offsets, d_cell_start, num_cells * sizeof(int),
                        cudaMemcpyDeviceToDevice));

  fillCellParticles_kernel<<<num_blocks, num_threads>>>(
      d_cell_ids, d_cell_offsets, d_cell_pts_sorted, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(d_cell_ids));
  CUDA_CHECK(cudaFree(d_nums_cell_pts));
  CUDA_CHECK(cudaFree(d_cell_offsets));
  CUDA_CHECK(cudaFree(d_tmp_storage));

  *d_cell_start_out = d_cell_start;
  *d_cell_pts_sorted_out = d_cell_pts_sorted;
}
