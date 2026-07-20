#include "fileIO.h"
#include "mpi_dist.h"
#include "particle_compression.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <string>
#include <type_traits>
#include <vector>


std::vector<std::string> inputFiles;
std::vector<std::string> baseDecompFiles;
std::string outputDir;
size_t D = 0;   // dimension
size_t N = 0;   // number of particles per rank (0 = auto-detect from file)
double xi = 0;  // coordinate-wise absolute error bound
double b = 0;   // linking length
double d = 0.2; // dimensionless linking length parameter
bool isDouble = false;
bool isABS = true;
bool isEdit = true;
bool isPGD = true;
OrderMode mode = OrderMode::MORTON_CODE;
FoFConstraintStrategy fof_constraint_strategy =
    FoFConstraintStrategy::PAIRWISE_VULNERABILITY;
size_t max_iter = 1000;
double lr = 0.01;
double target_cell_occupancy =
    0; // 0 means finest dense grid up to SIZE_MAX cells

void parseError(const char error[]);

FoFConstraintStrategy parseFoFConstraintStrategy(const std::string &value) {
  if (value == "1" || value == "pairwise" ||
      value == "pairwise-vulnerability") {
    return FoFConstraintStrategy::PAIRWISE_VULNERABILITY;
  }
  if (value == "2" || value == "safe-filter" ||
      value == "safe-component-filtering") {
    return FoFConstraintStrategy::SAFE_COMPONENT_FILTERING;
  }
  if (value == "3" || value == "contracted-forest" ||
      value == "contracted-halo-forest") {
    return FoFConstraintStrategy::CONTRACTED_HALO_FOREST;
  }
  parseError("FoF constraint strategy must be 1/pairwise-vulnerability, "
             "2/safe-component-filtering, or 3/contracted-halo-forest");
  return FoFConstraintStrategy::PAIRWISE_VULNERABILITY;
}

void parseError(const char error[]) {
  fprintf(stderr, "%s\n", error);
  fprintf(stderr, "Usage (MPI version):\n");
  fprintf(stderr, "  -i <x> <y> [<z>]: Input data files (per-rank or shared)\n");
  fprintf(stderr, "  -e <x> <y> [<z>]: Base-decompressed files (optional)\n");
  fprintf(stderr, "  -O <dir>         : Output directory for per-rank files\n");
  fprintf(stderr, "  -D <d>           : Dimensionality (2 or 3)\n");
  fprintf(stderr, "  -N <n>           : Number of particles per rank\n");
  fprintf(stderr, "  -f               : Use float data type\n");
  fprintf(stderr, "  -d               : Use double data type\n");
  fprintf(stderr, "  -M ABS <xi>      : Absolute error bound\n");
  fprintf(stderr, "  -M REL <xi>      : Relative error bound\n");
  fprintf(stderr, "  -B <d>           : Linking length parameter\n");
  fprintf(stderr, "  -KD              : k-d tree reordering\n");
  fprintf(stderr, "  -MC              : Morton code reordering\n");
  fprintf(stderr, "  -CS <strategy>   : FoF constraint strategy: "
                  "1/pairwise-vulnerability, 2/safe-component-filtering, "
                  "3/contracted-halo-forest\n");
  fprintf(stderr, "  -occ <n>         : Target particles per grid cell "
                  "(default: finest dense grid)\n");
  fprintf(stderr, "  -lr              : PGD learning rate\n");
  fprintf(stderr, "  -iter            : PGD max iterations\n");
  fprintf(stderr, "  -c               : Compression only\n");
  fprintf(stderr, "  -l               : Lossless edit (no PGD)\n");
  exit(EXIT_FAILURE);
}

void substituteRank(std::vector<std::string> &files, int rank) {
  char rank_str[16];
  snprintf(rank_str, sizeof(rank_str), "%d", rank);
  for (auto &f : files) {
    size_t pos;
    while ((pos = f.find("%r")) != std::string::npos)
      f.replace(pos, 2, rank_str);
  }
}

void Parsing(int argc, char *argv[]) {
  bool originalFileSpecified = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-i") {
      while (i + 1 < argc && argv[i + 1][0] != '-')
        inputFiles.push_back(argv[++i]);
      if (inputFiles.empty())
        parseError("Missing input file path(s)");
      originalFileSpecified = true;
    } else if (arg == "-e") {
      while (i + 1 < argc && argv[i + 1][0] != '-')
        baseDecompFiles.push_back(argv[++i]);
      if (baseDecompFiles.empty())
        parseError("Missing base decompressed file path(s)");
    } else if (arg == "-O") {
      if (i + 1 >= argc)
        parseError("Missing output directory");
      outputDir = argv[++i];
    } else if (arg == "-D") {
      D = std::stoull(argv[++i]);
    } else if (arg == "-N") {
      N = std::stoull(argv[++i]);
    } else if (arg == "-f") {
      isDouble = false;
    } else if (arg == "-d") {
      isDouble = true;
    } else if (arg == "-M") {
      isABS = std::strcmp(argv[++i], "ABS") == 0;
      if (i + 1 >= argc)
        parseError("Missing error bound value");
      xi = std::stod(argv[++i]);
    } else if (arg == "-B") {
      d = std::stod(argv[++i]);
    } else if (arg == "-KD") {
      mode = OrderMode::KD_TREE;
    } else if (arg == "-MC") {
      mode = OrderMode::MORTON_CODE;
    } else if (arg == "-CS" || arg == "-cs") {
      if (i + 1 >= argc)
        parseError("Missing FoF constraint strategy");
      fof_constraint_strategy = parseFoFConstraintStrategy(argv[++i]);
    } else if (arg == "-occ") {
      if (i + 1 >= argc)
        parseError("Missing target cell occupancy");
      target_cell_occupancy = std::stod(argv[++i]);
      if (target_cell_occupancy <= 0)
        parseError("Target cell occupancy must be positive");
    } else if (arg == "-lr") {
      lr = std::stof(argv[++i]);
    } else if (arg == "-iter") {
      max_iter = std::stoull(argv[++i]);
    } else if (arg == "-c") {
      isEdit = false;
    } else if (arg == "-l") {
      isPGD = false;
    } else {
      parseError("Unknown argument");
    }
  }

  if (!originalFileSpecified)
    parseError("MPI mode requires input files (-i)");
  if (outputDir.empty())
    parseError("MPI mode requires output directory (-O)");
  if (!isEdit && !isPGD)
    parseError("Compression mode; no need to avoid PGD");

  if (inputFiles.size() > 1)
    D = inputFiles.size();
  else if (baseDecompFiles.size() > 1)
    D = baseDecompFiles.size();
  if (D != 2 && D != 3)
    parseError("Dimension must be 2 or 3");
}

template <typename T>
void getRange(const T *arr, size_t N, T &minVal, T &maxVal, T &rangeVal) {
  minVal = arr[0];
  maxVal = arr[0];
  for (size_t i = 0; i < N; ++i) {
    if (arr[i] < minVal) minVal = arr[i];
    if (arr[i] > maxVal) maxVal = arr[i];
  }
  rangeVal = maxVal - minVal;
}

// #3: Fused range computation — single pass over all axes
template <typename T>
void getRange3D(const T *xx, const T *yy, const T *zz, size_t N,
                T &min_x, T &max_x, T &range_x,
                T &min_y, T &max_y, T &range_y,
                T &min_z, T &max_z, T &range_z) {
  min_x = max_x = xx[0];
  min_y = max_y = yy[0];
  min_z = max_z = zz[0];
  for (size_t i = 1; i < N; ++i) {
    if (xx[i] < min_x) min_x = xx[i]; else if (xx[i] > max_x) max_x = xx[i];
    if (yy[i] < min_y) min_y = yy[i]; else if (yy[i] > max_y) max_y = yy[i];
    if (zz[i] < min_z) min_z = zz[i]; else if (zz[i] > max_z) max_z = zz[i];
  }
  range_x = max_x - min_x;
  range_y = max_y - min_y;
  range_z = max_z - min_z;
}

template <typename T>
void getRange2D(const T *xx, const T *yy, size_t N,
                T &min_x, T &max_x, T &range_x,
                T &min_y, T &max_y, T &range_y) {
  min_x = max_x = xx[0];
  min_y = max_y = yy[0];
  for (size_t i = 1; i < N; ++i) {
    if (xx[i] < min_x) min_x = xx[i]; else if (xx[i] > max_x) max_x = xx[i];
    if (yy[i] < min_y) min_y = yy[i]; else if (yy[i] > max_y) max_y = yy[i];
  }
  range_x = max_x - min_x;
  range_y = max_y - min_y;
}

enum MpiTimerField {
  T_IO_READ_ORIGINAL,
  T_IO_READ_BASE,
  T_IO_WRITE_DECOMP,
  T_IO_WRITE_COMPRESSED,
  T_COMM_GLOBAL_GEOMETRY,
  T_COMM_AUTO_XI,
  T_COMM_BBOX_EXCHANGE,
  T_COMM_GHOST_ORIGINAL,
  T_COMM_GHOST_BASE,
  T_COMM_PGD_ALLREDUCE,
  T_LOCAL_RANGE,
  T_LOCAL_BBOX,
  T_LOCAL_EXT_RANGE,
  T_LOCAL_ERROR_SCAN,
  T_LOCAL_DEVICE_BUFFER,
  T_LOCAL_EDIT_FOF_SETUP,
  T_LOCAL_EDIT_PARTITION,
  T_LOCAL_EDIT_PAIR_GENERATION,
  T_LOCAL_EDIT_UNION_FIND,
  T_LOCAL_EDIT_MODE_FILTERING,
  T_LOCAL_EDIT_EDITABLE_TABLE,
  T_LOCAL_EDIT_FOF_OTHER,
  T_LOCAL_EDIT_PGD,
  T_LOCAL_EDIT_ENCODING,
  T_LOCAL_COMPRESSION,
  T_LOCAL_CLEANUP,
  T_H2D_ORIGINAL,
  T_H2D_BASE,
  T_H2D_GHOST_ORIGINAL,
  T_H2D_BASE_GHOSTS,
  T_D2H_DECOMP,
  T_TOTAL,
  T_COUNT
};

static const char *kTimerNames[T_COUNT] = {"I/O/read original",
                                           "I/O/read base",
                                           "I/O/write decompressed",
                                           "I/O/write compressed edits",
                                           "Comm/global geometry",
                                           "Comm/auto xi",
                                           "Comm/bounding boxes",
                                           "Comm/original ghosts",
                                           "Comm/base ghosts",
                                           "Comm/PGD convergence",
                                           "Local/range",
                                           "Local/bounding box",
                                           "Local/extended range",
                                           "Local/error scan",
                                           "Local/device buffer prep",
                                           "Local/FoF setup",
                                           "Local/spatial partition",
                                           "Local/find vulnerable pairs",
                                           "Local/union-find",
                                           "Local/mode filtering",
                                           "Local/editable table",
                                           "Local/FoF other",
                                           "Local/PGD kernels",
                                           "Local/edit encoding",
                                           "Local/compression",
                                           "Local/cleanup",
                                           "H2D/original",
                                           "H2D/base",
                                           "H2D/original ghosts",
                                           "H2D/base ghosts",
                                           "D2H/decompressed",
                                           "Total rank time"};

enum MpiTimerCategory { C_IO, C_COMM, C_LOCAL_COMPUTE, C_TRANSFER, C_TOTAL, C_COUNT };

static const char *kCategoryNames[C_COUNT] = {
    "I/O", "Communication", "Local compute", "H2D/D2H", "Total"};

struct MpiTimingBreakdown {
  double t[T_COUNT] = {};
  void add(MpiTimerField field, double seconds) { t[field] += seconds; }
  void set(MpiTimerField field, double seconds) { t[field] = seconds; }
  double category(MpiTimerCategory category) const {
    switch (category) {
    case C_IO:
      return t[T_IO_READ_ORIGINAL] + t[T_IO_READ_BASE] + t[T_IO_WRITE_DECOMP] +
             t[T_IO_WRITE_COMPRESSED];
    case C_COMM:
      return t[T_COMM_GLOBAL_GEOMETRY] + t[T_COMM_AUTO_XI] +
             t[T_COMM_BBOX_EXCHANGE] + t[T_COMM_GHOST_ORIGINAL] +
             t[T_COMM_GHOST_BASE] + t[T_COMM_PGD_ALLREDUCE];
    case C_LOCAL_COMPUTE:
      return t[T_LOCAL_RANGE] + t[T_LOCAL_BBOX] + t[T_LOCAL_EXT_RANGE] +
             t[T_LOCAL_ERROR_SCAN] + t[T_LOCAL_DEVICE_BUFFER] +
             t[T_LOCAL_EDIT_PARTITION] + t[T_LOCAL_EDIT_PAIR_GENERATION] +
             t[T_LOCAL_EDIT_UNION_FIND] + t[T_LOCAL_EDIT_MODE_FILTERING] +
             t[T_LOCAL_EDIT_EDITABLE_TABLE] + t[T_LOCAL_EDIT_FOF_OTHER] +
             t[T_LOCAL_EDIT_PGD] + t[T_LOCAL_EDIT_ENCODING] +
             t[T_LOCAL_COMPRESSION] + t[T_LOCAL_CLEANUP];
    case C_TRANSFER:
      return t[T_H2D_ORIGINAL] + t[T_H2D_BASE] + t[T_H2D_GHOST_ORIGINAL] +
             t[T_H2D_BASE_GHOSTS] + t[T_D2H_DECOMP];
    case C_TOTAL:
      return t[T_TOTAL];
    default:
      return 0.0;
    }
  }
};

static double nonnegativeTime(double seconds) {
  return seconds > 0.0 ? seconds : 0.0;
}

static void addFoFEditTimingBreakdown(MpiTimingBreakdown &timing,
                                      const FoFEditTiming &edit_timing) {
  double pair_generation = edit_timing.fof_vulnerable_pairs;
  double detailed_fof = edit_timing.fof_spatial_partition + pair_generation +
                        edit_timing.fof_union_find +
                        edit_timing.fof_mode_filtering +
                        edit_timing.fof_editable_table;
  double fof_other =
      edit_timing.fof_other +
      nonnegativeTime(edit_timing.fof_setup - detailed_fof -
                      edit_timing.fof_other);

  timing.add(T_LOCAL_EDIT_FOF_SETUP, edit_timing.fof_setup);
  timing.add(T_LOCAL_EDIT_PARTITION, edit_timing.fof_spatial_partition);
  timing.add(T_LOCAL_EDIT_PAIR_GENERATION, pair_generation);
  timing.add(T_LOCAL_EDIT_UNION_FIND, edit_timing.fof_union_find);
  timing.add(T_LOCAL_EDIT_MODE_FILTERING, edit_timing.fof_mode_filtering);
  timing.add(T_LOCAL_EDIT_EDITABLE_TABLE, edit_timing.fof_editable_table);
  timing.add(T_LOCAL_EDIT_FOF_OTHER, fof_other);
}

static void reportMpiTimings(const MpiTimingBreakdown &timing, MPI_Comm comm) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  double max_times[T_COUNT] = {};
  double sum_times[T_COUNT] = {};
  MPI_Reduce(timing.t, max_times, T_COUNT, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(timing.t, sum_times, T_COUNT, MPI_DOUBLE, MPI_SUM, 0, comm);

  double local_categories[C_COUNT] = {};
  for (int i = 0; i < C_COUNT; ++i)
    local_categories[i] = timing.category(static_cast<MpiTimerCategory>(i));
  double max_categories[C_COUNT] = {};
  double sum_categories[C_COUNT] = {};
  MPI_Reduce(local_categories, max_categories, C_COUNT, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(local_categories, sum_categories, C_COUNT, MPI_DOUBLE, MPI_SUM, 0, comm);

  if (rank != 0)
    return;

  printf("[MPI Timer] Component breakdown (max/avg over ranks, seconds):\n");
  for (int i = 0; i < T_COUNT; ++i) {
    if (max_times[i] == 0.0)
      continue;
    printf("[MPI Timer]   %-28s max %.6f avg %.6f\n", kTimerNames[i],
           max_times[i], sum_times[i] / size);
  }
  printf("[MPI Timer] Category breakdown (max/avg over ranks, seconds):\n");
  for (int i = 0; i < C_COUNT; ++i) {
    printf("[MPI Timer]   %-28s max %.6f avg %.6f\n", kCategoryNames[i],
           max_categories[i], sum_categories[i] / size);
  }
}

template <typename T> MPI_Datatype mpiScalarType() {
  return std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;
}

template <typename T>
void computeGlobalGeometry3D(T local_min_x, T local_max_x, T local_min_y,
                             T local_max_y, T local_min_z, T local_max_z,
                             size_t local_N, DistributedContext &ctx) {
  double local_min[3] = {(double)local_min_x, (double)local_min_y,
                         (double)local_min_z};
  double local_max[3] = {(double)local_max_x, (double)local_max_y,
                         (double)local_max_z};
  double global_min[3], global_max[3];
  MPI_Allreduce(local_min, global_min, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local_max, global_max, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  long long local_count = static_cast<long long>(local_N);
  long long global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  double global_range_x = global_max[0] - global_min[0];
  double global_range_y = global_max[1] - global_min[1];
  double global_range_z = global_max[2] - global_min[2];
  if (!isABS)
    xi *= std::min({global_range_x, global_range_y, global_range_z});
  b = d * std::cbrt(global_range_x * global_range_y * global_range_z /
                    static_cast<double>(global_count));

  if (ctx.rank == 0) {
    printf("Global FoF geometry: N=%lld, ranges=(%e, %e, %e), b=%e, xi=%e\n",
           global_count, global_range_x, global_range_y, global_range_z,
           static_cast<double>(b), static_cast<double>(xi));
  }
}

template <typename T>
void computeGlobalGeometry2D(T local_min_x, T local_max_x, T local_min_y,
                             T local_max_y, size_t local_N,
                             DistributedContext &ctx) {
  double local_min[2] = {(double)local_min_x, (double)local_min_y};
  double local_max[2] = {(double)local_max_x, (double)local_max_y};
  double global_min[2], global_max[2];
  MPI_Allreduce(local_min, global_min, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local_max, global_max, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  long long local_count = static_cast<long long>(local_N);
  long long global_count = 0;
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  double global_range_x = global_max[0] - global_min[0];
  double global_range_y = global_max[1] - global_min[1];
  if (!isABS)
    xi *= std::min(global_range_x, global_range_y);
  b = d * std::sqrt(global_range_x * global_range_y /
                    static_cast<double>(global_count));

  if (ctx.rank == 0) {
    printf("Global FoF geometry: N=%lld, ranges=(%e, %e), b=%e, xi=%e\n",
           global_count, global_range_x, global_range_y, static_cast<double>(b),
           static_cast<double>(xi));
  }
}

template <typename T, OrderMode Mode>
void run3D_mpi(DistributedContext &ctx) {
  if (N == 0) {
    size_t n = getFileElementCount<T>(inputFiles[0]);
    if (inputFiles.size() == 1)
      n /= 3;
    N = n;
  }
  printf("Rank %d: N = %zu, reading files...\n", ctx.rank, N);

  double t0, t1;
  double t_rank_start = MPI_Wtime();
  MpiTimingBreakdown timing;

  T *local_xx = new T[N];
  T *local_yy = new T[N];
  T *local_zz = new T[N];

  t0 = MPI_Wtime();
  readCoordFiles3D<T>(inputFiles, N, local_xx, local_yy, local_zz);
  t1 = MPI_Wtime();
  timing.add(T_IO_READ_ORIGINAL, t1 - t0);
  printf("[Timer] I/O read input: %f seconds\n", t1 - t0);

  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  T min_z, max_z, range_z;
  t0 = MPI_Wtime();
  getRange3D(local_xx, local_yy, local_zz, N,
             min_x, max_x, range_x, min_y, max_y, range_y,
             min_z, max_z, range_z);
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_RANGE, t1 - t0);

  t0 = MPI_Wtime();
  computeGlobalGeometry3D(min_x, max_x, min_y, max_y, min_z, max_z, N, ctx);
  t1 = MPI_Wtime();
  timing.add(T_COMM_GLOBAL_GEOMETRY, t1 - t0);

  T *base_xx_early = nullptr;
  T *base_yy_early = nullptr;
  T *base_zz_early = nullptr;
  if (!baseDecompFiles.empty() && xi == 0) {
    base_xx_early = new T[N];
    base_yy_early = new T[N];
    base_zz_early = new T[N];
    t0 = MPI_Wtime();
    readCoordFiles3D<T>(baseDecompFiles, N, base_xx_early, base_yy_early,
                        base_zz_early);
    t1 = MPI_Wtime();
    timing.add(T_IO_READ_BASE, t1 - t0);
    printf("[Timer] I/O read base (early): %f seconds\n", t1 - t0);

    t0 = MPI_Wtime();
    T local_xi = 0;
    for (size_t i = 0; i < N; ++i) {
      T ex = std::abs(local_xx[i] - base_xx_early[i]);
      T ey = std::abs(local_yy[i] - base_yy_early[i]);
      T ez = std::abs(local_zz[i] - base_zz_early[i]);
      local_xi = std::max(local_xi, std::max(ex, std::max(ey, ez)));
    }
    t1 = MPI_Wtime();
    timing.add(T_LOCAL_ERROR_SCAN, t1 - t0);

    T global_xi;
    t0 = MPI_Wtime();
    MPI_Allreduce(&local_xi, &global_xi, 1, mpiScalarType<T>(), MPI_MAX,
                  MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    timing.add(T_COMM_AUTO_XI, t1 - t0);
    xi = static_cast<double>(global_xi);
    if (ctx.rank == 0)
      printf("Auto-detected xi = %e\n", static_cast<double>(xi));
  }

  double ghost_width = b + 2.0 * std::sqrt(3.0) * xi;
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  ctx.local_bboxes.resize(1);
  computeLocalBBox3D(local_xx, local_yy, local_zz, N, ctx.local_bboxes[0]);
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_BBOX, t1 - t0);

  t0 = MPI_Wtime();
  discoverNeighbors(ctx, ghost_width, 3, MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  timing.add(T_COMM_BBOX_EXCHANGE, t1 - t0);
  printf("[Timer] Neighbor discovery: %f seconds\n", t1 - t0);

  std::vector<GhostBuffer<T>> ghosts;
  size_t total_ghost_count = 0;
  t0 = MPI_Wtime();
  auto pending = beginGhostExchange3D(local_xx, local_yy, local_zz, N, ctx,
                                      ghosts, total_ghost_count, MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  timing.add(T_COMM_GHOST_ORIGINAL, t1 - t0);

  t0 = MPI_Wtime();
  size_t N_ext = N + total_ghost_count;
  T *ext_pool = new T[3 * N_ext];
  T *ext_xx = ext_pool;
  T *ext_yy = ext_pool + N_ext;
  T *ext_zz = ext_pool + 2 * N_ext;
  std::memcpy(ext_xx, local_xx, N * sizeof(T));
  std::memcpy(ext_yy, local_yy, N * sizeof(T));
  std::memcpy(ext_zz, local_zz, N * sizeof(T));
  std::vector<std::vector<size_t>> ghost_send_indices = pending.send_indices;
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_DEVICE_BUFFER, t1 - t0);

  t0 = MPI_Wtime();
  completeGhostExchange(pending, ghosts, total_ghost_count);
  t1 = MPI_Wtime();
  timing.add(T_COMM_GHOST_ORIGINAL, t1 - t0);
  printf("[Timer] Ghost exchange: %f seconds (N_ext=%zu)\n", t1 - t0, N_ext);

  t0 = MPI_Wtime();
  size_t offset = N;
  for (const auto &g : ghosts) {
    std::memcpy(ext_xx + offset, g.xx.data(), g.count * sizeof(T));
    std::memcpy(ext_yy + offset, g.yy.data(), g.count * sizeof(T));
    std::memcpy(ext_zz + offset, g.zz.data(), g.count * sizeof(T));
    offset += g.count;
  }
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_DEVICE_BUFFER, t1 - t0);

  T ext_min_x, ext_max_x, ext_range_x;
  T ext_min_y, ext_max_y, ext_range_y;
  T ext_min_z, ext_max_z, ext_range_z;
  t0 = MPI_Wtime();
  getRange3D(ext_xx, ext_yy, ext_zz, N_ext,
             ext_min_x, ext_max_x, ext_range_x,
             ext_min_y, ext_max_y, ext_range_y,
             ext_min_z, ext_max_z, ext_range_z);
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_EXT_RANGE, t1 - t0);

  CompressionResults3D<T> result;
  CompressedData<T> compressed;

  if (!baseDecompFiles.empty()) {
    T *base_xx = base_xx_early;
    T *base_yy = base_yy_early;
    T *base_zz = base_zz_early;
    if (base_xx == nullptr) {
      base_xx = new T[N];
      base_yy = new T[N];
      base_zz = new T[N];
      t0 = MPI_Wtime();
      readCoordFiles3D<T>(baseDecompFiles, N, base_xx, base_yy, base_zz);
      t1 = MPI_Wtime();
      timing.add(T_IO_READ_BASE, t1 - t0);
      printf("[Timer] I/O read base: %f seconds\n", t1 - t0);
    }

    std::vector<GhostBuffer<T>> base_ghosts;
    t0 = MPI_Wtime();
    reexchangeGhostData3D(ghost_send_indices, base_xx, base_yy, base_zz,
                          ghosts, ctx, base_ghosts, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    timing.add(T_COMM_GHOST_BASE, t1 - t0);
    printf("[Timer] Ghost exchange (base): %f seconds\n", t1 - t0);

    t0 = MPI_Wtime();
    T *ext_base_xx, *ext_base_yy, *ext_base_zz;
    size_t base_N_ext;
    buildExtendedArrays3D(base_xx, base_yy, base_zz, N, base_ghosts,
                          total_ghost_count, ext_base_xx, ext_base_yy,
                          ext_base_zz, base_N_ext);
    result.decomp_xx = new T[N_ext];
    result.decomp_yy = new T[N_ext];
    result.decomp_zz = new T[N_ext];
    std::memcpy(result.decomp_xx, ext_base_xx, N_ext * sizeof(T));
    std::memcpy(result.decomp_yy, ext_base_yy, N_ext * sizeof(T));
    std::memcpy(result.decomp_zz, ext_base_zz, N_ext * sizeof(T));
    t1 = MPI_Wtime();
    timing.add(T_LOCAL_DEVICE_BUFFER, t1 - t0);

    t0 = MPI_Wtime();
    FoFEditTiming edit_timing;
    editParticles3D<T>(ext_xx, ext_yy, ext_zz, ext_min_x, ext_range_x,
                       ext_min_y, ext_range_y, ext_min_z, ext_range_z, N_ext,
                       xi, b, isPGD, result, compressed, N, MPI_COMM_WORLD,
                       &edit_timing);
    t1 = MPI_Wtime();
    addFoFEditTimingBreakdown(timing, edit_timing);
    timing.add(T_LOCAL_EDIT_PGD,
               edit_timing.pgd_total - edit_timing.pgd_allreduce);
    timing.add(T_COMM_PGD_ALLREDUCE, edit_timing.pgd_allreduce);
    timing.add(T_LOCAL_EDIT_ENCODING, edit_timing.edit_encoding);
    printf("[Timer] Edit (FOF+PGD): %f seconds\n", t1 - t0);
    fflush(stdout);

    freeExtendedArrays(ext_base_xx, ext_base_yy, ext_base_zz);
    delete[] base_xx;
    delete[] base_yy;
    delete[] base_zz;
  } else {
    t0 = MPI_Wtime();
    if (isEdit) {
      compressWithEditParticles3D<T, Mode>(
          ext_xx, ext_yy, ext_zz, ext_min_x, ext_range_x, ext_min_y,
          ext_range_y, ext_min_z, ext_range_z, N_ext, xi, b, isPGD, result,
          compressed, N);
    } else {
      compressParticles3D<T, Mode>(ext_xx, ext_yy, ext_zz, ext_min_x,
                                   ext_range_x, ext_min_y, ext_range_y,
                                   ext_min_z, ext_range_z, N_ext, xi, b,
                                   result, compressed);
      compressed.N_local = N;
    }
    t1 = MPI_Wtime();
    timing.add(T_LOCAL_COMPRESSION, t1 - t0);
    printf("[Timer] Compression: %f seconds\n", t1 - t0);
  }

  if (result.decomp_xx && result.decomp_yy && result.decomp_zz) {
    t0 = MPI_Wtime();
    char decomp_file[512];
    snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.xx.out",
             outputDir.c_str(), ctx.rank);
    writeRawArrayBinary(result.decomp_xx, N, decomp_file);
    snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.yy.out",
             outputDir.c_str(), ctx.rank);
    writeRawArrayBinary(result.decomp_yy, N, decomp_file);
    snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.zz.out",
             outputDir.c_str(), ctx.rank);
    writeRawArrayBinary(result.decomp_zz, N, decomp_file);
    t1 = MPI_Wtime();
    timing.add(T_IO_WRITE_DECOMP, t1 - t0);
  }

  t0 = MPI_Wtime();
  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s/rank_%03d.fofpz", outputDir.c_str(),
           ctx.rank);
  writeCompressedFile(outfile, compressed);
  t1 = MPI_Wtime();
  timing.add(T_IO_WRITE_COMPRESSED, t1 - t0);

  t0 = MPI_Wtime();
  delete[] ext_pool;
  delete[] local_xx;
  delete[] local_yy;
  delete[] local_zz;
  delete[] result.decomp_xx;
  delete[] result.decomp_yy;
  delete[] result.decomp_zz;
  result.decomp_xx = nullptr;
  result.decomp_yy = nullptr;
  result.decomp_zz = nullptr;
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_CLEANUP, t1 - t0);

  timing.set(T_TOTAL, MPI_Wtime() - t_rank_start);
  printf("[Timer] I/O write compressed: %f seconds\n", timing.t[T_IO_WRITE_COMPRESSED]);
  printf("Rank %d: wrote %s\n", ctx.rank, outfile);
  printf("[Timer] Total rank time: %f seconds\n", timing.t[T_TOTAL]);
  fflush(stdout);

  reportMpiTimings(timing, MPI_COMM_WORLD);
}

template <typename T, OrderMode Mode>
void run2D_mpi(DistributedContext &ctx) {
  if (N == 0) {
    size_t n = getFileElementCount<T>(inputFiles[0]);
    if (inputFiles.size() == 1)
      n /= 2;
    N = n;
  }

  double t_rank_start = MPI_Wtime();
  double t0, t1;
  MpiTimingBreakdown timing;

  T *local_xx = new T[N];
  T *local_yy = new T[N];

  t0 = MPI_Wtime();
  readCoordFiles2D<T>(inputFiles, N, local_xx, local_yy);
  t1 = MPI_Wtime();
  timing.add(T_IO_READ_ORIGINAL, t1 - t0);

  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  t0 = MPI_Wtime();
  getRange2D(local_xx, local_yy, N, min_x, max_x, range_x, min_y, max_y,
             range_y);
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_RANGE, t1 - t0);

  t0 = MPI_Wtime();
  computeGlobalGeometry2D(min_x, max_x, min_y, max_y, N, ctx);
  t1 = MPI_Wtime();
  timing.add(T_COMM_GLOBAL_GEOMETRY, t1 - t0);

  T *base_xx_early = nullptr;
  T *base_yy_early = nullptr;
  if (!baseDecompFiles.empty() && xi == 0) {
    base_xx_early = new T[N];
    base_yy_early = new T[N];
    t0 = MPI_Wtime();
    readCoordFiles2D<T>(baseDecompFiles, N, base_xx_early, base_yy_early);
    t1 = MPI_Wtime();
    timing.add(T_IO_READ_BASE, t1 - t0);

    t0 = MPI_Wtime();
    T local_xi = 0;
    for (size_t i = 0; i < N; ++i) {
      T ex = std::abs(local_xx[i] - base_xx_early[i]);
      T ey = std::abs(local_yy[i] - base_yy_early[i]);
      local_xi = std::max(local_xi, std::max(ex, ey));
    }
    t1 = MPI_Wtime();
    timing.add(T_LOCAL_ERROR_SCAN, t1 - t0);

    T global_xi;
    t0 = MPI_Wtime();
    MPI_Allreduce(&local_xi, &global_xi, 1, mpiScalarType<T>(), MPI_MAX,
                  MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    timing.add(T_COMM_AUTO_XI, t1 - t0);
    xi = static_cast<double>(global_xi);
    if (ctx.rank == 0)
      printf("Auto-detected xi = %e\n", static_cast<double>(xi));
  }

  double ghost_width = b + 2.0 * std::sqrt(2.0) * xi;
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  ctx.local_bboxes.resize(1);
  computeLocalBBox2D(local_xx, local_yy, N, ctx.local_bboxes[0]);
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_BBOX, t1 - t0);

  t0 = MPI_Wtime();
  discoverNeighbors(ctx, ghost_width, 2, MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  timing.add(T_COMM_BBOX_EXCHANGE, t1 - t0);

  std::vector<GhostBuffer<T>> ghosts;
  size_t total_ghost_count = 0;
  t0 = MPI_Wtime();
  auto pending = beginGhostExchange2D(local_xx, local_yy, N, ctx, ghosts,
                                      total_ghost_count, MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  timing.add(T_COMM_GHOST_ORIGINAL, t1 - t0);

  t0 = MPI_Wtime();
  size_t N_ext = N + total_ghost_count;
  T *ext_pool = new T[2 * N_ext];
  T *ext_xx = ext_pool;
  T *ext_yy = ext_pool + N_ext;
  std::memcpy(ext_xx, local_xx, N * sizeof(T));
  std::memcpy(ext_yy, local_yy, N * sizeof(T));
  std::vector<std::vector<size_t>> ghost_send_indices = pending.send_indices;
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_DEVICE_BUFFER, t1 - t0);

  t0 = MPI_Wtime();
  completeGhostExchange(pending, ghosts, total_ghost_count);
  t1 = MPI_Wtime();
  timing.add(T_COMM_GHOST_ORIGINAL, t1 - t0);

  t0 = MPI_Wtime();
  size_t offset = N;
  for (const auto &g : ghosts) {
    std::memcpy(ext_xx + offset, g.xx.data(), g.count * sizeof(T));
    std::memcpy(ext_yy + offset, g.yy.data(), g.count * sizeof(T));
    offset += g.count;
  }
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_DEVICE_BUFFER, t1 - t0);

  T ext_min_x, ext_max_x, ext_range_x;
  T ext_min_y, ext_max_y, ext_range_y;
  t0 = MPI_Wtime();
  getRange2D(ext_xx, ext_yy, N_ext, ext_min_x, ext_max_x, ext_range_x,
             ext_min_y, ext_max_y, ext_range_y);
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_EXT_RANGE, t1 - t0);

  CompressionResults2D<T> result;
  CompressedData<T> compressed;

  if (!baseDecompFiles.empty()) {
    T *base_xx = base_xx_early;
    T *base_yy = base_yy_early;
    if (base_xx == nullptr) {
      base_xx = new T[N];
      base_yy = new T[N];
      t0 = MPI_Wtime();
      readCoordFiles2D<T>(baseDecompFiles, N, base_xx, base_yy);
      t1 = MPI_Wtime();
      timing.add(T_IO_READ_BASE, t1 - t0);
    }

    std::vector<GhostBuffer<T>> base_ghosts;
    t0 = MPI_Wtime();
    reexchangeGhostData2D(ghost_send_indices, base_xx, base_yy, ghosts, ctx,
                          base_ghosts, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    timing.add(T_COMM_GHOST_BASE, t1 - t0);

    t0 = MPI_Wtime();
    T *ext_base_xx, *ext_base_yy;
    size_t base_N_ext;
    buildExtendedArrays2D(base_xx, base_yy, N, base_ghosts, total_ghost_count,
                          ext_base_xx, ext_base_yy, base_N_ext);
    result.decomp_xx = new T[N_ext];
    result.decomp_yy = new T[N_ext];
    std::memcpy(result.decomp_xx, ext_base_xx, N_ext * sizeof(T));
    std::memcpy(result.decomp_yy, ext_base_yy, N_ext * sizeof(T));
    t1 = MPI_Wtime();
    timing.add(T_LOCAL_DEVICE_BUFFER, t1 - t0);

    t0 = MPI_Wtime();
    FoFEditTiming edit_timing;
    editParticles2D<T>(ext_xx, ext_yy, ext_min_x, ext_range_x, ext_min_y,
                       ext_range_y, N_ext, xi, b, isPGD, result, compressed, N,
                       MPI_COMM_WORLD, &edit_timing);
    t1 = MPI_Wtime();
    addFoFEditTimingBreakdown(timing, edit_timing);
    timing.add(T_LOCAL_EDIT_PGD,
               edit_timing.pgd_total - edit_timing.pgd_allreduce);
    timing.add(T_COMM_PGD_ALLREDUCE, edit_timing.pgd_allreduce);
    timing.add(T_LOCAL_EDIT_ENCODING, edit_timing.edit_encoding);
    printf("[Timer] Edit (FOF+PGD): %f seconds\n", t1 - t0);
    fflush(stdout);

    freeExtendedArrays(ext_base_xx, ext_base_yy);
    delete[] base_xx;
    delete[] base_yy;
  } else {
    t0 = MPI_Wtime();
    if (isEdit) {
      compressWithEditParticles2D<T, Mode>(ext_xx, ext_yy, ext_min_x,
                                           ext_range_x, ext_min_y, ext_range_y,
                                           N_ext, xi, b, isPGD, result,
                                           compressed, N);
    } else {
      compressParticles2D<T, Mode>(ext_xx, ext_yy, ext_min_x, ext_range_x,
                                   ext_min_y, ext_range_y, N_ext, xi, b,
                                   result, compressed);
      compressed.N_local = N;
    }
    t1 = MPI_Wtime();
    timing.add(T_LOCAL_COMPRESSION, t1 - t0);
  }

  if (result.decomp_xx && result.decomp_yy) {
    t0 = MPI_Wtime();
    char decomp_file[512];
    snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.xx.out",
             outputDir.c_str(), ctx.rank);
    writeRawArrayBinary(result.decomp_xx, N, decomp_file);
    snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.yy.out",
             outputDir.c_str(), ctx.rank);
    writeRawArrayBinary(result.decomp_yy, N, decomp_file);
    t1 = MPI_Wtime();
    timing.add(T_IO_WRITE_DECOMP, t1 - t0);
  }

  t0 = MPI_Wtime();
  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s/rank_%03d.fofpz", outputDir.c_str(),
           ctx.rank);
  writeCompressedFile(outfile, compressed);
  t1 = MPI_Wtime();
  timing.add(T_IO_WRITE_COMPRESSED, t1 - t0);

  t0 = MPI_Wtime();
  delete[] ext_pool;
  delete[] local_xx;
  delete[] local_yy;
  delete[] result.decomp_xx;
  delete[] result.decomp_yy;
  result.decomp_xx = nullptr;
  result.decomp_yy = nullptr;
  t1 = MPI_Wtime();
  timing.add(T_LOCAL_CLEANUP, t1 - t0);

  timing.set(T_TOTAL, MPI_Wtime() - t_rank_start);
  printf("Rank %d: wrote %s\n", ctx.rank, outfile);
  printf("[Timer] Total rank time: %f seconds\n", timing.t[T_TOTAL]);
  fflush(stdout);

  reportMpiTimings(timing, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  setbuf(stdout, NULL); // Unbuffered stdout so prints appear immediately

  DistributedContext ctx;
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

  Parsing(argc, argv);
  substituteRank(inputFiles, ctx.rank);
  substituteRank(baseDecompFiles, ctx.rank);

  if (ctx.rank == 0) {
    if (N > 0)
      printf("MPI FOFPz: %d ranks, D=%zu, N=%zu per rank\n", ctx.size, D, N);
    else
      printf("MPI FOFPz: %d ranks, D=%zu, N=auto-detect\n", ctx.size, D);
  }

  if (isDouble) {
    if (D == 2) {
      if (mode == OrderMode::KD_TREE)
        run2D_mpi<double, OrderMode::KD_TREE>(ctx);
      else
        run2D_mpi<double, OrderMode::MORTON_CODE>(ctx);
    } else {
      if (mode == OrderMode::KD_TREE)
        run3D_mpi<double, OrderMode::KD_TREE>(ctx);
      else
        run3D_mpi<double, OrderMode::MORTON_CODE>(ctx);
    }
  } else {
    if (D == 2) {
      if (mode == OrderMode::KD_TREE)
        run2D_mpi<float, OrderMode::KD_TREE>(ctx);
      else
        run2D_mpi<float, OrderMode::MORTON_CODE>(ctx);
    } else {
      if (mode == OrderMode::KD_TREE)
        run3D_mpi<float, OrderMode::KD_TREE>(ctx);
      else
        run3D_mpi<float, OrderMode::MORTON_CODE>(ctx);
    }
  }

  MPI_Finalize();
  return 0;
}
