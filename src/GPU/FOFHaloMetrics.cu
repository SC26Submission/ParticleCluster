#include "FOFHaloMetrics.cuh"

// Compute C(n,2) = n*(n-1)/2 for each count in an array.
__global__ void chooseTwo_kernel(const int *counts, long long *results,
                                 int num_entries) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_entries) {
    long long n = counts[idx];
    results[idx] = n * (n - 1) / 2;
  }
}

// Pack two int32 roots into one int64 key for contingency table.
__global__ void packPairKeys_kernel(const int *org_roots,
                                    const int *decomp_roots, long long *keys,
                                    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    keys[idx] = ((long long)(unsigned int)org_roots[idx] << 32) |
                (unsigned int)decomp_roots[idx];
  }
}

static long long sumChooseTwoFromRoots(const int *d_roots, int N) {
  int *d_sorted = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sorted, N * sizeof(int)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_roots, d_sorted, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRadixSort::SortKeys(d_tmp, tmp_bytes, d_roots, d_sorted, N);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  int *d_unique = nullptr;
  int *d_counts = nullptr;
  int *d_num_clusters = nullptr;
  CUDA_CHECK(cudaMalloc(&d_unique, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_counts, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_num_clusters, sizeof(int)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_sorted, d_unique,
                                       d_counts, d_num_clusters, N);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_tmp, tmp_bytes, d_sorted, d_unique,
                                       d_counts, d_num_clusters, N);
    CUDA_CHECK(cudaFree(d_tmp));
  }
  CUDA_CHECK(cudaFree(d_sorted));
  CUDA_CHECK(cudaFree(d_unique));

  int h_num_clusters = 0;
  CUDA_CHECK(cudaMemcpy(&h_num_clusters, d_num_clusters, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_num_clusters));

  long long *d_c2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_c2, h_num_clusters * sizeof(long long)));
  int c2_blocks = (h_num_clusters + num_threads - 1) / num_threads;
  chooseTwo_kernel<<<c2_blocks, num_threads>>>(d_counts, d_c2,
                                               h_num_clusters);
  CUDA_CHECK(cudaFree(d_counts));

  long long *d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(long long)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_c2, d_sum, h_num_clusters);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_c2, d_sum, h_num_clusters);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  long long h_sum = 0;
  CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_c2));
  CUDA_CHECK(cudaFree(d_sum));
  return h_sum;
}

static long long sumChooseTwoFromRootPairs(const int *d_org_roots,
                                           const int *d_decomp_roots, int N) {
  int blocks = (N + num_threads - 1) / num_threads;

  long long *d_pair_keys = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pair_keys, N * sizeof(long long)));
  packPairKeys_kernel<<<blocks, num_threads>>>(d_org_roots, d_decomp_roots,
                                               d_pair_keys, N);

  long long *d_pair_keys_sorted = nullptr;
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

  long long *d_pair_unique = nullptr;
  int *d_pair_counts = nullptr;
  int *d_num_pairs = nullptr;
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

  int h_num_pairs = 0;
  CUDA_CHECK(cudaMemcpy(&h_num_pairs, d_num_pairs, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_num_pairs));

  long long *d_pair_c2 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pair_c2, h_num_pairs * sizeof(long long)));
  int c2_blocks = (h_num_pairs + num_threads - 1) / num_threads;
  chooseTwo_kernel<<<c2_blocks, num_threads>>>(d_pair_counts, d_pair_c2,
                                               h_num_pairs);
  CUDA_CHECK(cudaFree(d_pair_counts));

  long long *d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(long long)));
  {
    void *d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_pair_c2, d_sum, h_num_pairs);
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_bytes));
    cub::DeviceReduce::Sum(d_tmp, tmp_bytes, d_pair_c2, d_sum, h_num_pairs);
    CUDA_CHECK(cudaFree(d_tmp));
  }

  long long h_sum = 0;
  CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_pair_c2));
  CUDA_CHECK(cudaFree(d_sum));
  return h_sum;
}

void calculateARIFromRoots(const int *d_org_roots, const int *d_decomp_roots,
                           int N, long long &h_tp, long long &h_tn,
                           long long &h_fp, long long &h_fn) {
  if (N == 0) {
    h_tp = 0;
    h_tn = 0;
    h_fp = 0;
    h_fn = 0;
    return;
  }

  long long h_sum_org_c2 = sumChooseTwoFromRoots(d_org_roots, N);
  long long h_sum_decomp_c2 = sumChooseTwoFromRoots(d_decomp_roots, N);
  long long h_sum_pair_c2 =
      sumChooseTwoFromRootPairs(d_org_roots, d_decomp_roots, N);

  long long total_pairs = (long long)N * (N - 1) / 2;
  h_tp = h_sum_pair_c2;
  h_fn = h_sum_org_c2 - h_tp;
  h_fp = h_sum_decomp_c2 - h_tp;
  h_tn = total_pairs - h_tp - h_fp - h_fn;
}

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

  int *d_org_roots = nullptr;
  computeFoFRoots2D(d_org_xx, d_org_yy, min_x, range_x, min_y, range_y, N, b,
                    &d_org_roots);

  T min_x_decomp, max_x_decomp, range_x_decomp, min_y_decomp, max_y_decomp,
      range_y_decomp;
  getRange(d_decomp_xx, N, min_x_decomp, max_x_decomp, range_x_decomp);
  getRange(d_decomp_yy, N, min_y_decomp, max_y_decomp, range_y_decomp);

  int *d_decomp_roots = nullptr;
  computeFoFRoots2D(d_decomp_xx, d_decomp_yy, min_x_decomp, range_x_decomp,
                    min_y_decomp, range_y_decomp, N, b, &d_decomp_roots);

  calculateARIFromRoots(d_org_roots, d_decomp_roots, N, h_tp, h_tn, h_fp,
                        h_fn);

  CUDA_CHECK(cudaFree(d_org_roots));
  CUDA_CHECK(cudaFree(d_decomp_roots));
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

  int *d_org_roots = nullptr;
  computeFoFRoots3D(d_org_xx, d_org_yy, d_org_zz, min_x, range_x, min_y,
                    range_y, min_z, range_z, N, b, &d_org_roots);

  T min_x_decomp, min_y_decomp, min_z_decomp, max_x_decomp, max_y_decomp,
      max_z_decomp, range_x_decomp, range_y_decomp, range_z_decomp;
  getRange(d_decomp_xx, N, min_x_decomp, max_x_decomp, range_x_decomp);
  getRange(d_decomp_yy, N, min_y_decomp, max_y_decomp, range_y_decomp);
  getRange(d_decomp_zz, N, min_z_decomp, max_z_decomp, range_z_decomp);

  int *d_decomp_roots = nullptr;
  computeFoFRoots3D(d_decomp_xx, d_decomp_yy, d_decomp_zz, min_x_decomp,
                    range_x_decomp, min_y_decomp, range_y_decomp,
                    min_z_decomp, range_z_decomp, N, b, &d_decomp_roots);

  calculateARIFromRoots(d_org_roots, d_decomp_roots, N, h_tp, h_tn, h_fp,
                        h_fn);

  CUDA_CHECK(cudaFree(d_org_roots));
  CUDA_CHECK(cudaFree(d_decomp_roots));
}

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
