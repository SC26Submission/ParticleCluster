#include "fileIO.h"
#include "fof_ari.h"
#include "mpi_dist.h"
#include "particle_compression.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>


std::vector<std::string> inputFiles;
std::vector<std::string> baseDecompFiles;
std::string outputDir;
size_t D;      // dimension
size_t N = 0;  // number of particles per rank (0 = auto-detect from file)
float xi;      // coordinate-wise absolute error bound
float b;       // linking length
float d = 0.2; // dimensionless linking length parameter
bool isDouble;
bool isABS;
bool isEdit = true;
bool isPGD = true;
OrderMode mode = OrderMode::MORTON_CODE;
size_t max_iter = 1000;
double lr = 0.01;

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
      xi = std::stof(argv[++i]);
    } else if (arg == "-B") {
      d = std::stof(argv[++i]);
    } else if (arg == "-KD") {
      mode = OrderMode::KD_TREE;
    } else if (arg == "-MC") {
      mode = OrderMode::MORTON_CODE;
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

template <typename T, OrderMode Mode>
void run3D_mpi(DistributedContext &ctx) {
  // Auto-detect N from first input file if not specified
  if (N == 0) {
    size_t n = getFileElementCount<T>(inputFiles[0]);
    if (inputFiles.size() == 1) n /= 3;
    N = n;
  }
  printf("Rank %d: N = %zu, reading files...\n", ctx.rank, N);

  double t0, t1;
  double t_rank_start = MPI_Wtime();

  // Load local particles
  T *local_xx = new T[N];
  T *local_yy = new T[N];
  T *local_zz = new T[N];

  t0 = MPI_Wtime();
  readCoordFiles3D<T>(inputFiles, N, local_xx, local_yy, local_zz);
  t1 = MPI_Wtime();
  printf("[Timer] I/O read input: %f seconds\n", t1 - t0);

  // #3: Fused range computation — single pass over all axes
  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  T min_z, max_z, range_z;
  getRange3D(local_xx, local_yy, local_zz, N,
             min_x, max_x, range_x, min_y, max_y, range_y, min_z, max_z, range_z);

  if (!isABS) {
    xi *= std::min({range_x, range_y, range_z});
  }
  b = d * std::cbrt(range_x * range_y * range_z / N);

  // Early xi detection from base files — must be done BEFORE computing
  // ghost_width, otherwise xi=0 (draco) gives ghost_width=b which is too
  // small and causes millions of clusters, hanging discoverNeighbors.
  T *base_xx_early = nullptr;
  T *base_yy_early = nullptr;
  T *base_zz_early = nullptr;
  if (!baseDecompFiles.empty() && xi == 0) {
    base_xx_early = new T[N];
    base_yy_early = new T[N];
    base_zz_early = new T[N];
    t0 = MPI_Wtime();
    readCoordFiles3D<T>(baseDecompFiles, N, base_xx_early, base_yy_early, base_zz_early);
    t1 = MPI_Wtime();
    printf("[Timer] I/O read base (early): %f seconds\n", t1 - t0);
    float local_xi = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      float ex = std::abs((float)local_xx[i] - (float)base_xx_early[i]);
      float ey = std::abs((float)local_yy[i] - (float)base_yy_early[i]);
      float ez = std::abs((float)local_zz[i] - (float)base_zz_early[i]);
      if (ex > local_xi) local_xi = ex;
      if (ey > local_xi) local_xi = ey;
      if (ez > local_xi) local_xi = ez;
    }
    float global_xi;
    t0 = MPI_Wtime();
    MPI_Allreduce(&local_xi, &global_xi, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    printf("[Timer] MPI_Allreduce (xi): %f seconds\n", t1 - t0);
    xi = static_cast<float>(global_xi);
    if (ctx.rank == 0)
      printf("Auto-detected xi = %e\n", (double)xi);
  }

  printf("Rank %d: range_x=%.2f range_y=%.2f range_z=%.2f b=%.6f xi=%.6f\n",
         ctx.rank, (double)range_x, (double)range_y, (double)range_z,
         (double)b, (double)xi);

  // Detect spatial clusters and discover neighbors
  double ghost_width = b + 2.0 * std::sqrt(3.0) * xi;
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  ctx.local_bboxes.resize(1);
  computeLocalBBox3D(local_xx, local_yy, local_zz, N, ctx.local_bboxes[0]);
  discoverNeighbors(ctx, ghost_width, 3, MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  printf("[Timer] Neighbor discovery: %f seconds\n", t1 - t0);

  // Ghost exchange (double-buffered: overlap MPI with local array prep)
  std::vector<GhostBuffer<T>> ghosts;
  size_t total_ghost_count;
  t0 = MPI_Wtime();
  auto pending = beginGhostExchange3D(local_xx, local_yy, local_zz, N, ctx,
                                      ghosts, total_ghost_count, MPI_COMM_WORLD);

  // #6: Allocate extended arrays as single contiguous block for locality
  size_t N_ext = N + total_ghost_count;
  T *ext_pool = new T[3 * N_ext];
  T *ext_xx = ext_pool;
  T *ext_yy = ext_pool + N_ext;
  T *ext_zz = ext_pool + 2 * N_ext;
  std::memcpy(ext_xx, local_xx, N * sizeof(T));
  std::memcpy(ext_yy, local_yy, N * sizeof(T));
  std::memcpy(ext_zz, local_zz, N * sizeof(T));

  // B2: Wait for ghost data with progressive completion
  completeGhostExchange(pending, ghosts, total_ghost_count);
  t1 = MPI_Wtime();
  printf("[Timer] Ghost exchange: %f seconds (N_ext=%zu)\n", t1 - t0, N_ext);
  size_t offset = N;
  for (const auto &g : ghosts) {
    std::memcpy(ext_xx + offset, g.xx.data(), g.count * sizeof(T));
    std::memcpy(ext_yy + offset, g.yy.data(), g.count * sizeof(T));
    std::memcpy(ext_zz + offset, g.zz.data(), g.count * sizeof(T));
    offset += g.count;
  }

  // #3: Fused range on extended data
  T ext_min_x, ext_max_x, ext_range_x;
  T ext_min_y, ext_max_y, ext_range_y;
  T ext_min_z, ext_max_z, ext_range_z;
  getRange3D(ext_xx, ext_yy, ext_zz, N_ext,
             ext_min_x, ext_max_x, ext_range_x,
             ext_min_y, ext_max_y, ext_range_y,
             ext_min_z, ext_max_z, ext_range_z);
  printf("Rank %d: starting compression (ext_range: %.2f x %.2f x %.2f)...\n",
         ctx.rank, (double)ext_range_x, (double)ext_range_y, (double)ext_range_z);

  // Run compression on extended arrays with N_local filtering
  CompressionResults3D<T> result;
  CompressedData<T> compressed;

  if (!baseDecompFiles.empty()) {
    // Reuse early-loaded base data if available, otherwise load now
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
      printf("[Timer] I/O read base: %f seconds\n", t1 - t0);
    }

    // Re-exchange base decompressed data using saved send_indices so ghost
    // ordering matches the original exchange (local base coords in ghost slots)
    std::vector<GhostBuffer<T>> base_ghosts;
    t0 = MPI_Wtime();
    reexchangeGhostData3D(pending.send_indices, base_xx, base_yy, base_zz,
                          ghosts, ctx, base_ghosts, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    printf("[Timer] Ghost exchange (base): %f seconds\n", t1 - t0);

    // Build extended base arrays with correct base ghost positions
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

    editParticles3D<T>(ext_xx, ext_yy, ext_zz, ext_min_x, ext_range_x,
                       ext_min_y, ext_range_y, ext_min_z, ext_range_z, N_ext,
                       xi, b, isPGD, result, compressed, N, MPI_COMM_WORLD);

    freeExtendedArrays(ext_base_xx, ext_base_yy, ext_base_zz);
    delete[] base_xx;
    delete[] base_yy;
    delete[] base_zz;
  } else {
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
  }

  // Write per-rank compressed file
  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s/rank_%03d.fofpz", outputDir.c_str(),
           ctx.rank);
  t0 = MPI_Wtime();
  writeCompressedFile(outfile, compressed);
  t1 = MPI_Wtime();
  printf("[Timer] I/O write output: %f seconds\n", t1 - t0);
  printf("[Timer] Total rank time: %f seconds\n", MPI_Wtime() - t_rank_start);
  printf("Rank %d: wrote %s\n", ctx.rank, outfile);
  fflush(stdout);

  // Compute ARI (requires decompressed arrays from edit mode)
  if (isEdit && result.decomp_xx && result.decomp_yy && result.decomp_zz) {
    printf("Rank %d: starting ARI computation...\n", ctx.rank);
    distributedARI3D(ext_xx, ext_yy, ext_zz, result.decomp_xx, result.decomp_yy,
                     result.decomp_zz, N, N_ext, static_cast<T>(b), ctx.rank,
                     ctx.size, MPI_COMM_WORLD);
  }

  // Cleanup
  delete[] ext_pool; // #6: single pool deallocation
  delete[] local_xx;
  delete[] local_yy;
  delete[] local_zz;
}

template <typename T, OrderMode Mode>
void run2D_mpi(DistributedContext &ctx) {
  if (N == 0) {
    size_t n = getFileElementCount<T>(inputFiles[0]);
    if (inputFiles.size() == 1) n /= 2;
    N = n;
  }

  double t_rank_start_2d = MPI_Wtime();
  T *local_xx = new T[N];
  T *local_yy = new T[N];

  readCoordFiles2D<T>(inputFiles, N, local_xx, local_yy);

  // #3: Fused range computation
  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  getRange2D(local_xx, local_yy, N, min_x, max_x, range_x, min_y, max_y, range_y);

  if (!isABS) {
    xi *= std::min(range_x, range_y);
  }
  b = d * std::sqrt(range_x * range_y / N);

  // Early xi detection from base files before computing ghost_width
  T *base_xx_early = nullptr;
  T *base_yy_early = nullptr;
  if (!baseDecompFiles.empty() && xi == 0) {
    base_xx_early = new T[N];
    base_yy_early = new T[N];
    readCoordFiles2D<T>(baseDecompFiles, N, base_xx_early, base_yy_early);
    float local_xi = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      float ex = std::abs((float)local_xx[i] - (float)base_xx_early[i]);
      float ey = std::abs((float)local_yy[i] - (float)base_yy_early[i]);
      if (ex > local_xi) local_xi = ex;
      if (ey > local_xi) local_xi = ey;
    }
    float global_xi;
    MPI_Allreduce(&local_xi, &global_xi, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    xi = static_cast<float>(global_xi);
    if (ctx.rank == 0)
      printf("Auto-detected xi = %e\n", (double)xi);
  }

  double ghost_width = b + 2.0 * std::sqrt(2.0) * xi;
  ctx.local_bboxes.resize(1);
  computeLocalBBox2D(local_xx, local_yy, N, ctx.local_bboxes[0]);
  discoverNeighbors(ctx, ghost_width, 2, MPI_COMM_WORLD);

  // Ghost exchange (double-buffered)
  std::vector<GhostBuffer<T>> ghosts;
  size_t total_ghost_count;
  auto pending = beginGhostExchange2D(local_xx, local_yy, N, ctx, ghosts,
                                      total_ghost_count, MPI_COMM_WORLD);

  // #6: Pooled extended array allocation for 2D
  size_t N_ext = N + total_ghost_count;
  T *ext_pool = new T[2 * N_ext];
  T *ext_xx = ext_pool;
  T *ext_yy = ext_pool + N_ext;
  std::memcpy(ext_xx, local_xx, N * sizeof(T));
  std::memcpy(ext_yy, local_yy, N * sizeof(T));

  // B2: Wait for ghost data with progressive completion
  completeGhostExchange(pending, ghosts, total_ghost_count);
  size_t offset = N;
  for (const auto &g : ghosts) {
    std::memcpy(ext_xx + offset, g.xx.data(), g.count * sizeof(T));
    std::memcpy(ext_yy + offset, g.yy.data(), g.count * sizeof(T));
    offset += g.count;
  }

  // #3: Fused range on extended data
  T ext_min_x, ext_max_x, ext_range_x;
  T ext_min_y, ext_max_y, ext_range_y;
  getRange2D(ext_xx, ext_yy, N_ext, ext_min_x, ext_max_x, ext_range_x,
             ext_min_y, ext_max_y, ext_range_y);

  CompressionResults2D<T> result;
  CompressedData<T> compressed;

  if (!baseDecompFiles.empty()) {
    T *base_xx = base_xx_early;
    T *base_yy = base_yy_early;
    if (base_xx == nullptr) {
      base_xx = new T[N];
      base_yy = new T[N];
      readCoordFiles2D<T>(baseDecompFiles, N, base_xx, base_yy);
    }

    std::vector<GhostBuffer<T>> base_ghosts;
    reexchangeGhostData2D(pending.send_indices, base_xx, base_yy,
                          ghosts, ctx, base_ghosts, MPI_COMM_WORLD);

    T *ext_base_xx, *ext_base_yy;
    size_t base_N_ext;
    buildExtendedArrays2D(base_xx, base_yy, N, base_ghosts, total_ghost_count,
                          ext_base_xx, ext_base_yy, base_N_ext);

    result.decomp_xx = new T[N_ext];
    result.decomp_yy = new T[N_ext];
    std::memcpy(result.decomp_xx, ext_base_xx, N_ext * sizeof(T));
    std::memcpy(result.decomp_yy, ext_base_yy, N_ext * sizeof(T));

    editParticles2D<T>(ext_xx, ext_yy, ext_min_x, ext_range_x, ext_min_y,
                       ext_range_y, N_ext, xi, b, isPGD, result, compressed, N,
                       MPI_COMM_WORLD);

    freeExtendedArrays(ext_base_xx, ext_base_yy);
    delete[] base_xx;
    delete[] base_yy;
  } else {
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
  }

  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s/rank_%03d.fofpz", outputDir.c_str(),
           ctx.rank);
  writeCompressedFile(outfile, compressed);

  printf("[Timer] Total rank time: %f seconds\n", MPI_Wtime() - t_rank_start_2d);
  printf("Rank %d: wrote %s\n", ctx.rank, outfile);
  fflush(stdout);

  // Compute ARI (requires decompressed arrays from edit mode)
  if (isEdit && result.decomp_xx && result.decomp_yy) {
    printf("Rank %d: starting ARI computation...\n", ctx.rank);
    distributedARI2D(ext_xx, ext_yy, result.decomp_xx, result.decomp_yy, N,
                     N_ext, static_cast<T>(b), ctx.rank, ctx.size,
                     MPI_COMM_WORLD);
  }

  delete[] ext_pool; // #6: single pool deallocation
  delete[] local_xx;
  delete[] local_yy;
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
