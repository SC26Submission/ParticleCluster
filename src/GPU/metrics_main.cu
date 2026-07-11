#include "FOFHaloMetrics.cuh"
#include "fileIO.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

struct MetricOptions {
  std::vector<std::string> original_files;
  std::vector<std::string> decompressed_files;
  int D = 0;
  size_t N = 0;
  bool is_double = false;
  bool absolute_b = false;
  double b = 0.0;
  double linking_parameter = 0.2;
};

int target_cell_occupancy = 0; // 0 means default: 8 in 2D, 4 in 3D

static void parseError(const char *error) {
  std::fprintf(stderr, "%s\n", error);
  std::fprintf(stderr, "Usage:\n");
  std::fprintf(stderr,
               "  fofpz_metrics -i <orig_x> <orig_y> [<orig_z>] "
               "-o <decomp_x> <decomp_y> [<decomp_z>] -D <2|3> "
               "[-N <n>] (-f|-d) [-B <dimensionless_b> | -b <absolute_b>] "
               "[-occ <n>]\n");
  std::fprintf(stderr,
               "  A single file after -i or -o is treated as interleaved "
               "coordinates; use -D in that case.\n");
  std::exit(EXIT_FAILURE);
}

static MetricOptions parseArgs(int argc, char **argv) {
  MetricOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-i") {
      while (i + 1 < argc && argv[i + 1][0] != '-')
        opts.original_files.push_back(argv[++i]);
    } else if (arg == "-o") {
      while (i + 1 < argc && argv[i + 1][0] != '-')
        opts.decompressed_files.push_back(argv[++i]);
    } else if (arg == "-D") {
      if (i + 1 >= argc)
        parseError("Missing dimension after -D");
      opts.D = std::stoi(argv[++i]);
    } else if (arg == "-N") {
      if (i + 1 >= argc)
        parseError("Missing particle count after -N");
      opts.N = static_cast<size_t>(std::stoull(argv[++i]));
    } else if (arg == "-f") {
      opts.is_double = false;
    } else if (arg == "-d") {
      opts.is_double = true;
    } else if (arg == "-B") {
      if (i + 1 >= argc)
        parseError("Missing dimensionless linking parameter after -B");
      opts.linking_parameter = std::stod(argv[++i]);
      opts.absolute_b = false;
    } else if (arg == "-b") {
      if (i + 1 >= argc)
        parseError("Missing absolute linking length after -b");
      opts.b = std::stod(argv[++i]);
      opts.absolute_b = true;
    } else if (arg == "-occ") {
      if (i + 1 >= argc)
        parseError("Missing target cell occupancy after -occ");
      target_cell_occupancy = std::stoi(argv[++i]);
      if (target_cell_occupancy <= 0)
        parseError("Target cell occupancy must be positive");
    } else {
      parseError("Unknown argument");
    }
  }

  if (opts.original_files.empty())
    parseError("Missing original coordinate files");
  if (opts.decompressed_files.empty())
    parseError("Missing decompressed coordinate files");
  if (opts.original_files.size() > 1)
    opts.D = static_cast<int>(opts.original_files.size());
  else if (opts.decompressed_files.size() > 1)
    opts.D = static_cast<int>(opts.decompressed_files.size());
  if (opts.D != 2 && opts.D != 3)
    parseError("Dimension must be 2 or 3");
  if (!(opts.original_files.size() == 1 ||
        opts.original_files.size() == static_cast<size_t>(opts.D)))
    parseError("Original file count must be 1 interleaved file or D files");
  if (!(opts.decompressed_files.size() == 1 ||
        opts.decompressed_files.size() == static_cast<size_t>(opts.D)))
    parseError("Decompressed file count must be 1 interleaved file or D files");

  return opts;
}

template <typename T>
static size_t inferParticleCount(const MetricOptions &opts) {
  size_t elements = getFileElementCount<T>(opts.original_files[0]);
  if (opts.original_files.size() == 1) {
    if (elements % static_cast<size_t>(opts.D) != 0)
      parseError("Interleaved original file length is not divisible by D");
    return elements / static_cast<size_t>(opts.D);
  }
  return elements;
}

template <typename T> static void upload(const std::vector<T> &h, T **d_out) {
  CUDA_CHECK(cudaMalloc(d_out, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(*d_out, h.data(), h.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
}

struct PointwiseStats {
  double max_abs_error = 0.0;
  double mean_abs_error = 0.0;
  double mse = 0.0;
  double rmse = 0.0;
  double nrmse = 0.0;
  double psnr = std::numeric_limits<double>::infinity();
};

struct IoUStats {
  double mean_iou = 0.0;
  double mass_weighted_mean_iou = 0.0;
  double iou_at_1 = 0.0;
  double iou_at_095 = 0.0;
};

template <typename T>
static void accumulateError(T org, T dec, long double &sum_abs,
                            long double &sum_sq, double &max_abs) {
  double err = static_cast<double>(dec) - static_cast<double>(org);
  double abs_err = std::abs(err);
  sum_abs += abs_err;
  sum_sq += err * err;
  max_abs = std::max(max_abs, abs_err);
}

static PointwiseStats finalizePointwiseStats(long double sum_abs,
                                             long double sum_sq, double max_abs,
                                             size_t num_values,
                                             double norm_range) {
  PointwiseStats stats;
  if (num_values == 0)
    return stats;

  stats.max_abs_error = max_abs;
  stats.mean_abs_error = static_cast<double>(sum_abs / num_values);
  stats.mse = static_cast<double>(sum_sq / num_values);
  stats.rmse = std::sqrt(stats.mse);
  if (norm_range > 0) {
    stats.nrmse = stats.rmse / norm_range;
    stats.psnr = stats.nrmse == 0.0 ? std::numeric_limits<double>::infinity()
                                    : -20.0 * std::log10(stats.nrmse);
  } else {
    stats.nrmse =
        stats.rmse == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    stats.psnr = stats.rmse == 0.0 ? std::numeric_limits<double>::infinity()
                                   : -std::numeric_limits<double>::infinity();
  }
  return stats;
}

template <typename T>
static PointwiseStats computePointwiseStats2D(const std::vector<T> &org_x,
                                              const std::vector<T> &org_y,
                                              const std::vector<T> &dec_x,
                                              const std::vector<T> &dec_y,
                                              T range_x, T range_y) {
  long double sum_abs = 0.0;
  long double sum_sq = 0.0;
  double max_abs = 0.0;
  for (size_t i = 0; i < org_x.size(); ++i) {
    accumulateError(org_x[i], dec_x[i], sum_abs, sum_sq, max_abs);
    accumulateError(org_y[i], dec_y[i], sum_abs, sum_sq, max_abs);
  }
  double norm_range = std::max(std::abs(static_cast<double>(range_x)),
                               std::abs(static_cast<double>(range_y)));
  return finalizePointwiseStats(sum_abs, sum_sq, max_abs, 2 * org_x.size(),
                                norm_range);
}

template <typename T>
static PointwiseStats computePointwiseStats3D(const std::vector<T> &org_x,
                                              const std::vector<T> &org_y,
                                              const std::vector<T> &org_z,
                                              const std::vector<T> &dec_x,
                                              const std::vector<T> &dec_y,
                                              const std::vector<T> &dec_z,
                                              T range_x, T range_y, T range_z) {
  long double sum_abs = 0.0;
  long double sum_sq = 0.0;
  double max_abs = 0.0;
  for (size_t i = 0; i < org_x.size(); ++i) {
    accumulateError(org_x[i], dec_x[i], sum_abs, sum_sq, max_abs);
    accumulateError(org_y[i], dec_y[i], sum_abs, sum_sq, max_abs);
    accumulateError(org_z[i], dec_z[i], sum_abs, sum_sq, max_abs);
  }
  double norm_range =
      std::max(std::abs(static_cast<double>(range_x)),
               std::max(std::abs(static_cast<double>(range_y)),
                        std::abs(static_cast<double>(range_z))));
  return finalizePointwiseStats(sum_abs, sum_sq, max_abs, 3 * org_x.size(),
                                norm_range);
}

static std::uint64_t packRootPair(int org_root, int dec_root) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(org_root))
          << 32) |
         static_cast<std::uint32_t>(dec_root);
}

static IoUStats calculateIoUStats(const std::vector<int> &org_roots,
                                  const std::vector<int> &dec_roots) {
  IoUStats stats;
  size_t N = org_roots.size();
  if (N == 0)
    return stats;

  std::unordered_map<int, size_t> org_sizes;
  std::unordered_map<int, size_t> dec_sizes;
  std::unordered_map<std::uint64_t, size_t> intersections;
  org_sizes.reserve(N);
  dec_sizes.reserve(N);
  intersections.reserve(N);

  for (size_t i = 0; i < N; ++i) {
    int org_root = org_roots[i];
    int dec_root = dec_roots[i];
    ++org_sizes[org_root];
    ++dec_sizes[dec_root];
    ++intersections[packRootPair(org_root, dec_root)];
  }

  std::unordered_map<int, double> best_iou;
  best_iou.reserve(org_sizes.size());
  for (const auto &entry : org_sizes)
    best_iou[entry.first] = 0.0;

  for (const auto &entry : intersections) {
    int org_root = static_cast<int>(entry.first >> 32);
    int dec_root = static_cast<int>(entry.first & 0xffffffffu);
    double intersection = static_cast<double>(entry.second);
    double union_size = static_cast<double>(org_sizes[org_root] +
                                            dec_sizes[dec_root] - entry.second);
    double iou = union_size > 0.0 ? intersection / union_size : 0.0;
    if (iou > best_iou[org_root])
      best_iou[org_root] = iou;
  }

  double sum_iou = 0.0;
  double weighted_sum_iou = 0.0;
  size_t count_at_1 = 0;
  size_t count_at_095 = 0;
  constexpr double exact_tol = 1e-12;
  for (const auto &entry : org_sizes) {
    double iou = best_iou[entry.first];
    sum_iou += iou;
    weighted_sum_iou += iou * static_cast<double>(entry.second);
    if (iou >= 1.0 - exact_tol)
      ++count_at_1;
    if (iou >= 0.95)
      ++count_at_095;
  }

  double num_halos = static_cast<double>(org_sizes.size());
  stats.mean_iou = sum_iou / num_halos;
  stats.mass_weighted_mean_iou = weighted_sum_iou / static_cast<double>(N);
  stats.iou_at_1 = static_cast<double>(count_at_1) / num_halos;
  stats.iou_at_095 = static_cast<double>(count_at_095) / num_halos;
  return stats;
}

static void printPointwiseStats(const PointwiseStats &stats) {
  std::printf("Max absolute error: %.17g\n", stats.max_abs_error);
  std::printf("MAE: %.17g\n", stats.mean_abs_error);
  std::printf("MSE: %.17g\n", stats.mse);
  std::printf("RMSE: %.17g\n", stats.rmse);
  std::printf("NRMSE: %.17g\n", stats.nrmse);
  std::printf("PSNR: %.17g dB\n", stats.psnr);
}

static void printIoUStats(const IoUStats &stats) {
  std::printf("Mean IoU: %.17g\n", stats.mean_iou);
  std::printf("Mass-weighted mean IoU: %.17g\n", stats.mass_weighted_mean_iou);
  std::printf("IoU@1: %.17g\n", stats.iou_at_1);
  std::printf("IoU@0.95: %.17g\n", stats.iou_at_095);
}

template <typename T> static void run2D(const MetricOptions &opts) {
  size_t N = opts.N == 0 ? inferParticleCount<T>(opts) : opts.N;
  std::vector<T> org_x(N), org_y(N), dec_x(N), dec_y(N);
  readCoordFiles2D<T>(opts.original_files, N, org_x.data(), org_y.data());
  readCoordFiles2D<T>(opts.decompressed_files, N, dec_x.data(), dec_y.data());

  T *d_org_x = nullptr, *d_org_y = nullptr;
  T *d_dec_x = nullptr, *d_dec_y = nullptr;
  upload(org_x, &d_org_x);
  upload(org_y, &d_org_y);
  upload(dec_x, &d_dec_x);
  upload(dec_y, &d_dec_y);

  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  getRange(d_org_x, static_cast<int>(N), min_x, max_x, range_x);
  getRange(d_org_y, static_cast<int>(N), min_y, max_y, range_y);
  T b = opts.absolute_b ? static_cast<T>(opts.b)
                        : static_cast<T>(opts.linking_parameter) *
                              std::sqrt(range_x * range_y / static_cast<T>(N));

  auto start = std::chrono::high_resolution_clock::now();
  PointwiseStats pointwise =
      computePointwiseStats2D(org_x, org_y, dec_x, dec_y, range_x, range_y);

  int n_int = static_cast<int>(N);
  int *d_org_roots = nullptr;
  computeFoFRoots2D(d_org_x, d_org_y, min_x, range_x, min_y, range_y, n_int, b,
                    &d_org_roots);

  T min_x_dec, max_x_dec, range_x_dec;
  T min_y_dec, max_y_dec, range_y_dec;
  getRange(d_dec_x, n_int, min_x_dec, max_x_dec, range_x_dec);
  getRange(d_dec_y, n_int, min_y_dec, max_y_dec, range_y_dec);

  int *d_dec_roots = nullptr;
  computeFoFRoots2D(d_dec_x, d_dec_y, min_x_dec, range_x_dec, min_y_dec,
                    range_y_dec, n_int, b, &d_dec_roots);

  long long tp = 0, tn = 0, fp = 0, fn = 0;
  calculateARIFromRoots(d_org_roots, d_dec_roots, n_int, tp, tn, fp, fn);

  std::vector<int> org_roots(N);
  std::vector<int> dec_roots(N);
  CUDA_CHECK(cudaMemcpy(org_roots.data(), d_org_roots, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dec_roots.data(), d_dec_roots, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  IoUStats iou = calculateIoUStats(org_roots, dec_roots);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  T ari = calculateARIFromPairCounts<T>(tp, tn, fp, fn);
  std::chrono::duration<double> elapsed = end - start;
  std::printf("N: %zu\n", N);
  std::printf("Linking length b: %.17g\n", static_cast<double>(b));
  printPointwiseStats(pointwise);
  std::printf("ARI: %.17g\n", static_cast<double>(ari));
  printIoUStats(iou);
  std::printf("Metric wall-clock time: %f seconds\n", elapsed.count());

  CUDA_CHECK(cudaFree(d_org_roots));
  CUDA_CHECK(cudaFree(d_dec_roots));
  CUDA_CHECK(cudaFree(d_org_x));
  CUDA_CHECK(cudaFree(d_org_y));
  CUDA_CHECK(cudaFree(d_dec_x));
  CUDA_CHECK(cudaFree(d_dec_y));
}

template <typename T> static void run3D(const MetricOptions &opts) {
  size_t N = opts.N == 0 ? inferParticleCount<T>(opts) : opts.N;
  std::vector<T> org_x(N), org_y(N), org_z(N), dec_x(N), dec_y(N), dec_z(N);
  readCoordFiles3D<T>(opts.original_files, N, org_x.data(), org_y.data(),
                      org_z.data());
  readCoordFiles3D<T>(opts.decompressed_files, N, dec_x.data(), dec_y.data(),
                      dec_z.data());

  T *d_org_x = nullptr, *d_org_y = nullptr, *d_org_z = nullptr;
  T *d_dec_x = nullptr, *d_dec_y = nullptr, *d_dec_z = nullptr;
  upload(org_x, &d_org_x);
  upload(org_y, &d_org_y);
  upload(org_z, &d_org_z);
  upload(dec_x, &d_dec_x);
  upload(dec_y, &d_dec_y);
  upload(dec_z, &d_dec_z);

  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  T min_z, max_z, range_z;
  getRange(d_org_x, static_cast<int>(N), min_x, max_x, range_x);
  getRange(d_org_y, static_cast<int>(N), min_y, max_y, range_y);
  getRange(d_org_z, static_cast<int>(N), min_z, max_z, range_z);
  T b = opts.absolute_b
            ? static_cast<T>(opts.b)
            : static_cast<T>(opts.linking_parameter) *
                  std::cbrt(range_x * range_y * range_z / static_cast<T>(N));

  auto start = std::chrono::high_resolution_clock::now();
  PointwiseStats pointwise = computePointwiseStats3D(
      org_x, org_y, org_z, dec_x, dec_y, dec_z, range_x, range_y, range_z);

  int n_int = static_cast<int>(N);
  int *d_org_roots = nullptr;
  computeFoFRoots3D(d_org_x, d_org_y, d_org_z, min_x, range_x, min_y, range_y,
                    min_z, range_z, n_int, b, &d_org_roots);

  T min_x_dec, min_y_dec, min_z_dec;
  T max_x_dec, max_y_dec, max_z_dec;
  T range_x_dec, range_y_dec, range_z_dec;
  getRange(d_dec_x, n_int, min_x_dec, max_x_dec, range_x_dec);
  getRange(d_dec_y, n_int, min_y_dec, max_y_dec, range_y_dec);
  getRange(d_dec_z, n_int, min_z_dec, max_z_dec, range_z_dec);

  int *d_dec_roots = nullptr;
  computeFoFRoots3D(d_dec_x, d_dec_y, d_dec_z, min_x_dec, range_x_dec,
                    min_y_dec, range_y_dec, min_z_dec, range_z_dec, n_int, b,
                    &d_dec_roots);

  long long tp = 0, tn = 0, fp = 0, fn = 0;
  calculateARIFromRoots(d_org_roots, d_dec_roots, n_int, tp, tn, fp, fn);

  std::vector<int> org_roots(N);
  std::vector<int> dec_roots(N);
  CUDA_CHECK(cudaMemcpy(org_roots.data(), d_org_roots, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dec_roots.data(), d_dec_roots, N * sizeof(int),
                        cudaMemcpyDeviceToHost));
  IoUStats iou = calculateIoUStats(org_roots, dec_roots);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  T ari = calculateARIFromPairCounts<T>(tp, tn, fp, fn);
  std::chrono::duration<double> elapsed = end - start;
  std::printf("N: %zu\n", N);
  std::printf("Linking length b: %.17g\n", static_cast<double>(b));
  printPointwiseStats(pointwise);
  std::printf("ARI: %.17g\n", static_cast<double>(ari));
  printIoUStats(iou);
  std::printf("Metric wall-clock time: %f seconds\n", elapsed.count());

  CUDA_CHECK(cudaFree(d_org_roots));
  CUDA_CHECK(cudaFree(d_dec_roots));
  CUDA_CHECK(cudaFree(d_org_x));
  CUDA_CHECK(cudaFree(d_org_y));
  CUDA_CHECK(cudaFree(d_org_z));
  CUDA_CHECK(cudaFree(d_dec_x));
  CUDA_CHECK(cudaFree(d_dec_y));
  CUDA_CHECK(cudaFree(d_dec_z));
}

int main(int argc, char **argv) {
  MetricOptions opts = parseArgs(argc, argv);
  if (opts.is_double) {
    if (opts.D == 2)
      run2D<double>(opts);
    else
      run3D<double>(opts);
  } else {
    if (opts.D == 2)
      run2D<float>(opts);
    else
      run3D<float>(opts);
  }
  return 0;
}
