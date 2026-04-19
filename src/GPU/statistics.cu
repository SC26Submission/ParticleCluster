#include "particle_compression.cuh"
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

// Functor for absolute value (for CUB)
template <typename T> struct AbsOp {
  __device__ __forceinline__ T operator()(const T &a) const {
    return a < T(0) ? -a : a;
  }
};

// Functor for squaring (for CUB)
template <typename T> struct SquareOp {
  __device__ __forceinline__ T operator()(const T &a) const { return a * a; }
};

template <typename T>
__global__ void difference_kernel(const T *org_data, T *error, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    error[idx] -= org_data[idx];
}

// Find max absolute value using CUB
template <typename T> T findMaxAbs(const T *d_arr, int N) {
  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;

  T *d_result;
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

  AbsOp<T> abs_op;
  auto abs_itr = thrust::make_transform_iterator(d_arr, abs_op);

  cub::DeviceReduce::Max(nullptr, tmp_storage_bytes, abs_itr, d_result, N);
  CUDA_CHECK(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceReduce::Max(d_tmp_storage, tmp_storage_bytes, abs_itr, d_result,
                         N);

  T result;
  CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_tmp_storage));
  CUDA_CHECK(cudaFree(d_result));
  return result;
}

// Calculate sum of squared errors for one array using CUB
template <typename T> T sumSquaredErrors(const T *d_err, int N) {
  SquareOp<T> sq_op;
  auto sq_itr = thrust::make_transform_iterator(d_err, sq_op);

  void *d_tmp_storage = nullptr;
  size_t tmp_storage_bytes = 0;
  T *d_result;
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

  cub::DeviceReduce::Sum(nullptr, tmp_storage_bytes, sq_itr, d_result, N);
  CUDA_CHECK(cudaMalloc(&d_tmp_storage, tmp_storage_bytes));
  cub::DeviceReduce::Sum(d_tmp_storage, tmp_storage_bytes, sq_itr, d_result, N);

  T result;
  CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_tmp_storage));
  CUDA_CHECK(cudaFree(d_result));
  return result;
}

template <typename T>
T calculateMSE2D(const T *d_err_xx, const T *d_err_yy, int N) {
  T mse = sumSquaredErrors(d_err_xx, N) + sumSquaredErrors(d_err_yy, N);
  return mse / (2 * N);
}

template <typename T>
T calculateMSE3D(const T *d_err_xx, const T *d_err_yy, const T *d_err_zz,
                 int N) {
  T mse = sumSquaredErrors(d_err_xx, N) + sumSquaredErrors(d_err_yy, N) +
          sumSquaredErrors(d_err_zz, N);
  return mse / (3 * N);
}

template <typename T>
void calculateStatistics2D(const T *d_org_xx, const T *d_org_yy,
                           const T *d_decomp_xx, const T *d_decomp_yy,
                           T range_x, T range_y, int N) {
  T *d_err_xx, *d_err_yy;
  CUDA_CHECK(cudaMalloc(&d_err_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_err_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_err_xx, d_decomp_xx, N * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_err_yy, d_decomp_yy, N * sizeof(T),
                        cudaMemcpyDeviceToDevice));

  int num_blocks = (N + num_threads - 1) / num_threads;
  difference_kernel<T><<<num_blocks, num_threads>>>(d_org_xx, d_err_xx, N);
  difference_kernel<T><<<num_blocks, num_threads>>>(d_org_yy, d_err_yy, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  T mae_x = findMaxAbs(d_err_xx, N);
  T mae_y = findMaxAbs(d_err_yy, N);
  T mae = std::max(mae_x, mae_y);
  T mse = calculateMSE2D(d_err_xx, d_err_yy, N);
  T rmse = std::sqrt(mse);
  T nrmse = rmse / std::max(range_x, range_y);
  T psnr = -20 * std::log10(nrmse);

  CUDA_CHECK(cudaFree(d_err_xx));
  CUDA_CHECK(cudaFree(d_err_yy));

  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

template <typename T>
void calculateStatistics3D(const T *d_org_xx, const T *d_org_yy,
                           const T *d_org_zz, const T *d_decomp_xx,
                           const T *d_decomp_yy, const T *d_decomp_zz,
                           T range_x, T range_y, T range_z, int N) {
  T *d_err_xx, *d_err_yy, *d_err_zz;
  CUDA_CHECK(cudaMalloc(&d_err_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_err_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_err_zz, N * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_err_xx, d_decomp_xx, N * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_err_yy, d_decomp_yy, N * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_err_zz, d_decomp_zz, N * sizeof(T),
                        cudaMemcpyDeviceToDevice));

  int num_blocks = (N + num_threads - 1) / num_threads;
  difference_kernel<T><<<num_blocks, num_threads>>>(d_org_xx, d_err_xx, N);
  difference_kernel<T><<<num_blocks, num_threads>>>(d_org_yy, d_err_yy, N);
  difference_kernel<T><<<num_blocks, num_threads>>>(d_org_zz, d_err_zz, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  T mae_x = findMaxAbs(d_err_xx, N);
  T mae_y = findMaxAbs(d_err_yy, N);
  T mae_z = findMaxAbs(d_err_zz, N);
  T mae = std::max(mae_x, std::max(mae_y, mae_z));
  T mse = calculateMSE3D(d_err_xx, d_err_yy, d_err_zz, N);
  T rmse = std::sqrt(mse);
  T nrmse = rmse / std::max(range_x, std::max(range_y, range_z));
  T psnr = -20 * std::log10(nrmse);

  CUDA_CHECK(cudaFree(d_err_xx));
  CUDA_CHECK(cudaFree(d_err_yy));
  CUDA_CHECK(cudaFree(d_err_zz));

  printf("MAE: %f\n", mae);
  printf("MSE: %f\n", mse);
  printf("RMSE: %f\n", rmse);
  printf("NRMSE: %f\n", nrmse);
  printf("PSNR: %f dB\n", psnr);
}

// Explicit template instantiations
template void calculateStatistics2D<float>(const float *, const float *,
                                           const float *, const float *, float,
                                           float, int);
template void calculateStatistics2D<double>(const double *, const double *,
                                            const double *, const double *,
                                            double, double, int);
template void calculateStatistics3D<float>(const float *, const float *,
                                           const float *, const float *,
                                           const float *, const float *, float,
                                           float, float, int);
template void calculateStatistics3D<double>(const double *, const double *,
                                            const double *, const double *,
                                            const double *, const double *,
                                            double, double, double, int);