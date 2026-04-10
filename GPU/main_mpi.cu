#include "fileIO.h"
#include "mpi_dist.cuh"
#include "particle_compression.cuh"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

std::vector<std::string> inputFiles;
std::vector<std::string> baseDecompFiles;
std::string outputDir;
int D;         // dimension
int N = 0;     // number of particles per rank (0 = auto-detect from file)
float xi = 0;  // coordinate-wise absolute error bound
float b;       // linking length
float d = 0.2; // dimensionless linking length parameter
bool isDouble;
bool isABS;
bool isEdit = true;
OrderMode mode = OrderMode::MORTON_CODE;
int max_iter = 1000;
double lr = 0.01;

void parseError(const char error[]) {
  fprintf(stderr, "%s\n", error);
  fprintf(stderr, "Usage (MPI GPU version):\n");
  fprintf(stderr, "  -i <x> <y> [<z>]: Input data files (per-rank)\n");
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
      D = std::stoi(argv[++i]);
    } else if (arg == "-N") {
      N = std::stoi(argv[++i]);
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
      max_iter = std::stoi(argv[++i]);
    } else if (arg == "-c") {
      isEdit = false;
    } else {
      parseError("Unknown argument");
    }
  }

  if (!originalFileSpecified)
    parseError("MPI mode requires input files (-i)");
  if (outputDir.empty())
    parseError("MPI mode requires output directory (-O)");

  if (inputFiles.size() > 1)
    D = inputFiles.size();
  else if (baseDecompFiles.size() > 1)
    D = baseDecompFiles.size();
  if (D != 2 && D != 3)
    parseError("Dimension must be 2 or 3");
}

template <typename T> std::string suffix(const std::string &base) {
  if constexpr (std::is_same_v<T, float>)
    return base + ".f32";
  else if constexpr (std::is_same_v<T, double>)
    return base + ".d64";
  else
    static_assert(sizeof(T) == 0, "Unsupported type");
}

template <typename T, OrderMode Mode> void run3D_mpi(DistributedContext &ctx) {
  // Auto-detect N from first input file if not specified
  if (N == 0) {
    size_t n = getFileElementCount<T>(inputFiles[0]);
    if (inputFiles.size() == 1)
      n /= 3; // interleaved
    N = static_cast<int>(n);
  }

  double t0, t1;
  double t_rank_start = MPI_Wtime();

  // Load local particles into pinned host memory
  T *h_local_xx = nullptr, *h_local_yy = nullptr, *h_local_zz = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_local_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMallocHost(&h_local_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMallocHost(&h_local_zz, N * sizeof(T)));

  t0 = MPI_Wtime();
  readCoordFiles3D<T>(inputFiles, N, h_local_xx, h_local_yy, h_local_zz);
  t1 = MPI_Wtime();
  printf("[Timer] I/O read input: %f seconds\n", t1 - t0);
  fflush(stdout);

  // Upload to device for range computation
  T *d_local_xx, *d_local_yy, *d_local_zz;
  CUDA_CHECK(cudaMalloc(&d_local_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_local_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_local_zz, N * sizeof(T)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMemcpyAsync(d_local_xx, h_local_xx, N * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_local_yy, h_local_yy, N * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_local_zz, h_local_zz, N * sizeof(T),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // Compute ranges on local data
  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  T min_z, max_z, range_z;
  getRange(d_local_xx, N, min_x, max_x, range_x);
  getRange(d_local_yy, N, min_y, max_y, range_y);
  getRange(d_local_zz, N, min_z, max_z, range_z);

  if (!isABS) {
    xi *= std::min({range_x, range_y, range_z});
  }
  b = d * std::cbrt(range_x * range_y * range_z / N);

  // If base files provided and xi still 0, auto-detect xi NOW before ghost
  // exchange so ghost_width is correct. Base host arrays are kept alive for
  // reuse in the edit path below (avoids reading them twice).
  T *h_base_xx_early = nullptr, *h_base_yy_early = nullptr,
    *h_base_zz_early = nullptr;
  if (!baseDecompFiles.empty() && xi == 0) {
    CUDA_CHECK(cudaMallocHost(&h_base_xx_early, N * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_base_yy_early, N * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_base_zz_early, N * sizeof(T)));
    readCoordFiles3D<T>(baseDecompFiles, N, h_base_xx_early, h_base_yy_early,
                        h_base_zz_early);
    T *d_tmp_base_xx, *d_tmp_base_yy, *d_tmp_base_zz;
    CUDA_CHECK(cudaMalloc(&d_tmp_base_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmp_base_yy, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_tmp_base_zz, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_tmp_base_xx, h_base_xx_early, N * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmp_base_yy, h_base_yy_early, N * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tmp_base_zz, h_base_zz_early, N * sizeof(T),
                          cudaMemcpyHostToDevice));
    T local_xi = getMaxAbsErr(d_local_xx, d_tmp_base_xx, N);
    local_xi = std::max(local_xi, getMaxAbsErr(d_local_yy, d_tmp_base_yy, N));
    local_xi = std::max(local_xi, getMaxAbsErr(d_local_zz, d_tmp_base_zz, N));
    CUDA_CHECK(cudaFree(d_tmp_base_xx));
    CUDA_CHECK(cudaFree(d_tmp_base_yy));
    CUDA_CHECK(cudaFree(d_tmp_base_zz));
    float global_xi;
    MPI_Allreduce(&local_xi, &global_xi, 1, MPI_FLOAT, MPI_MAX,
                  MPI_COMM_WORLD);
    xi = global_xi;
    if (ctx.rank == 0)
      printf("Auto-detected xi = %e\n", (double)xi);
  }

  // Detect spatial clusters and discover neighbors
  double ghost_width = b + 2.0 * std::sqrt(3.0) * xi;
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  ctx.local_bboxes.resize(1);
  computeLocalBBox3D_GPU(d_local_xx, d_local_yy, d_local_zz, N,
                         ctx.local_bboxes[0]);
  discoverNeighbors(ctx, ghost_width, 3, MPI_COMM_WORLD);
  t1 = MPI_Wtime();
  printf("[Timer] Neighbor discovery: %f seconds\n", t1 - t0);
  fflush(stdout);

  // Ghost exchange: C2 GPU ghost ID, A3 packed buffers, B1 pipelined sends
  std::vector<GhostBuffer<T>> ghosts;
  size_t total_ghost_count;
  t0 = MPI_Wtime();
  auto pending = beginGhostExchange3D(d_local_xx, d_local_yy, d_local_zz,
                                      static_cast<size_t>(N), ctx, ghosts,
                                      total_ghost_count, MPI_COMM_WORLD);

  // While MPI transfers ghost data: allocate extended arrays, copy local (D2D)
  int N_ext = N + static_cast<int>(total_ghost_count);
  T *d_ext_xx, *d_ext_yy, *d_ext_zz;
  CUDA_CHECK(cudaMalloc(&d_ext_xx, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_ext_yy, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_ext_zz, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_ext_xx, d_local_xx, (size_t)N * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ext_yy, d_local_yy, (size_t)N * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ext_zz, d_local_zz, (size_t)N * sizeof(T),
                        cudaMemcpyDeviceToDevice));

  // Free local device arrays (data is now in extended arrays)
  CUDA_CHECK(cudaFree(d_local_xx));
  CUDA_CHECK(cudaFree(d_local_yy));
  CUDA_CHECK(cudaFree(d_local_zz));

  // Save send indices before completeGhostExchange clears pending
  std::vector<std::vector<int>> ghost_send_indices = pending.send_indices;

  // B2: Wait for ghost data with progressive completion
  completeGhostExchange(pending, ghosts, total_ghost_count);
  t1 = MPI_Wtime();
  printf("[Timer] Ghost exchange: %f seconds\n", t1 - t0);
  fflush(stdout);

  // B3: Copy ghost data to device using async streams
  t0 = MPI_Wtime();
  {
    size_t offset = N;
    int num_nonempty = 0;
    for (const auto &g : ghosts)
      if (g.count > 0)
        num_nonempty++;

    std::vector<cudaStream_t> streams(num_nonempty);
    for (int i = 0; i < num_nonempty; ++i)
      CUDA_CHECK(cudaStreamCreate(&streams[i]));

    int si = 0;
    for (const auto &g : ghosts) {
      if (g.count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_ext_xx + offset, g.xx.data(),
                                   g.count * sizeof(T), cudaMemcpyHostToDevice,
                                   streams[si]));
        CUDA_CHECK(cudaMemcpyAsync(d_ext_yy + offset, g.yy.data(),
                                   g.count * sizeof(T), cudaMemcpyHostToDevice,
                                   streams[si]));
        CUDA_CHECK(cudaMemcpyAsync(d_ext_zz + offset, g.zz.data(),
                                   g.count * sizeof(T), cudaMemcpyHostToDevice,
                                   streams[si]));
        offset += g.count;
        si++;
      }
    }
    for (int i = 0; i < num_nonempty; ++i) {
      CUDA_CHECK(cudaStreamSynchronize(streams[i]));
      CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
  }
  t1 = MPI_Wtime();
  printf("[Timer] Ghost H2D copy: %f seconds (N_ext=%d)\n", t1 - t0, N_ext);
  fflush(stdout);

  // Compute ranges on extended data (grid must cover ghosts too)
  T ext_min_x, ext_max_x, ext_range_x;
  T ext_min_y, ext_max_y, ext_range_y;
  T ext_min_z, ext_max_z, ext_range_z;
  getRange(d_ext_xx, N_ext, ext_min_x, ext_max_x, ext_range_x);
  getRange(d_ext_yy, N_ext, ext_min_y, ext_max_y, ext_range_y);
  getRange(d_ext_zz, N_ext, ext_min_z, ext_max_z, ext_range_z);

  // Run compression on extended arrays with N_local=N filtering
  CompressedData<T> compressed;
  CompressionState3D<T> state;

  if (!baseDecompFiles.empty()) {
    // Edit-only mode — reuse early-loaded base arrays if available
    T *h_base_xx = h_base_xx_early;
    T *h_base_yy = h_base_yy_early;
    T *h_base_zz = h_base_zz_early;
    t0 = MPI_Wtime();
    if (h_base_xx == nullptr) {
      CUDA_CHECK(cudaMallocHost(&h_base_xx, N * sizeof(T)));
      CUDA_CHECK(cudaMallocHost(&h_base_yy, N * sizeof(T)));
      CUDA_CHECK(cudaMallocHost(&h_base_zz, N * sizeof(T)));
      readCoordFiles3D<T>(baseDecompFiles, N, h_base_xx, h_base_yy, h_base_zz);
    }
    t1 = MPI_Wtime();
    printf("[Timer] I/O read base: %f seconds\n", t1 - t0);
    fflush(stdout);

    // Upload base to device then build extended
    T *d_base_xx, *d_base_yy, *d_base_zz;
    CUDA_CHECK(cudaMalloc(&d_base_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_base_yy, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_base_zz, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_base_xx, h_base_xx, N * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_yy, h_base_yy, N * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_zz, h_base_zz, N * sizeof(T),
                          cudaMemcpyHostToDevice));

    // Exchange base-decompressed ghost data using the same send indices
    // as the original exchange — guarantees identical ghost ordering/count.
    t0 = MPI_Wtime();
    std::vector<GhostBuffer<T>> base_ghosts;
    reexchangeGhostData3D(ghost_send_indices, h_base_xx, h_base_yy, h_base_zz,
                          ghosts, ctx, base_ghosts, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    printf("[Timer] Ghost exchange (base): %f seconds\n", t1 - t0);
    fflush(stdout);

    T *d_ext_base_xx, *d_ext_base_yy, *d_ext_base_zz;
    int base_dummy;
    buildExtendedDeviceArrays3D(d_base_xx, d_base_yy, d_base_zz, N, base_ghosts,
                                total_ghost_count, d_ext_base_xx, d_ext_base_yy,
                                d_ext_base_zz, base_dummy);

    if (xi == 0) {
      t0 = MPI_Wtime();
      T local_xi = getMaxAbsErr(d_ext_xx, d_base_xx, N);
      local_xi = std::max(local_xi, getMaxAbsErr(d_ext_yy, d_base_yy, N));
      local_xi = std::max(local_xi, getMaxAbsErr(d_ext_zz, d_base_zz, N));
      float global_xi;
      MPI_Allreduce(&local_xi, &global_xi, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      xi = global_xi;
      t1 = MPI_Wtime();
      printf("[Timer] MPI_Allreduce (xi): %f seconds\n", t1 - t0);
      if (ctx.rank == 0)
        printf("Auto-detected xi = %e\n", (double)xi);
    }

    CUDA_CHECK(cudaFree(d_base_xx));
    CUDA_CHECK(cudaFree(d_base_yy));
    CUDA_CHECK(cudaFree(d_base_zz));

    t0 = MPI_Wtime();
    editParticles3D<T, Mode>(d_ext_xx, d_ext_yy, d_ext_zz, d_ext_base_xx,
                             d_ext_base_yy, d_ext_base_zz, ext_min_x,
                             ext_range_x, ext_min_y, ext_range_y, ext_min_z,
                             ext_range_z, N_ext, xi, b, state, compressed, N,
                             MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    printf("[Timer] Edit (FOF+PGD): %f seconds\n", t1 - t0);
    fflush(stdout);
    // NOTE: editParticles3D assigns d_ext_base_* to state.d_decomp_* and runs
    // PGD on them. state.free() will free these pointers — do NOT free them
    // here to avoid use-after-free in the export section and double-free.
    destroyHashTable(state.d_editable_pts_ht);

    CUDA_CHECK(cudaFreeHost(h_base_xx));
    CUDA_CHECK(cudaFreeHost(h_base_yy));
    CUDA_CHECK(cudaFreeHost(h_base_zz));
  } else {
    t0 = MPI_Wtime();
    if (isEdit) {
      compressWithEditParticles3D<T, Mode>(d_ext_xx, d_ext_yy, d_ext_zz,
                                           ext_min_x, ext_range_x, ext_min_y,
                                           ext_range_y, ext_min_z, ext_range_z,
                                           N_ext, xi, b, state, compressed, N);
      destroyHashTable(state.d_editable_pts_ht);
    } else {
      compressParticles3D<T, Mode>(
          d_ext_xx, d_ext_yy, d_ext_zz, ext_min_x, ext_range_x, ext_min_y,
          ext_range_y, ext_min_z, ext_range_z, N_ext, xi, b, state, compressed);
      compressed.N_local = static_cast<size_t>(N);
    }
    t1 = MPI_Wtime();
    printf("[Timer] Compression: %f seconds\n", t1 - t0);
    fflush(stdout);
  }

  t0 = MPI_Wtime();
  ////////////////// Export Decomp Start //////////////////
  // Copy only N local particles (state holds N_ext including ghosts)
  T *decomp_xx = new T[N];
  T *decomp_yy = new T[N];
  T *decomp_zz = new T[N];
  CUDA_CHECK(cudaMemcpy(decomp_xx, state.d_decomp_xx, N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(decomp_yy, state.d_decomp_yy, N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(decomp_zz, state.d_decomp_zz, N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  char decomp_file[512];
  snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.xx.out",
           outputDir.c_str(), ctx.rank);
  writeRawArrayBinary(decomp_xx, N, decomp_file);
  snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.yy.out",
           outputDir.c_str(), ctx.rank);
  writeRawArrayBinary(decomp_yy, N, decomp_file);
  snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.zz.out",
           outputDir.c_str(), ctx.rank);
  writeRawArrayBinary(decomp_zz, N, decomp_file);
  delete[] decomp_xx;
  delete[] decomp_yy;
  delete[] decomp_zz;
  /////////////////// Export Decomp End ///////////////////

  state.free();
  CUDA_CHECK(cudaFree(d_ext_xx));
  CUDA_CHECK(cudaFree(d_ext_yy));
  CUDA_CHECK(cudaFree(d_ext_zz));

  // Write per-rank compressed file
  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s/rank_%03d.fofpz", outputDir.c_str(),
           ctx.rank);
  writeCompressedFile(outfile, compressed);
  t1 = MPI_Wtime();
  printf("[Timer] I/O write output: %f seconds\n", t1 - t0);
  printf("Rank %d: wrote %s\n", ctx.rank, outfile);
  printf("[Timer] Total rank time: %f seconds\n", t1 - t_rank_start);
  fflush(stdout);

  CUDA_CHECK(cudaFreeHost(h_local_xx));
  CUDA_CHECK(cudaFreeHost(h_local_yy));
  CUDA_CHECK(cudaFreeHost(h_local_zz));
}

template <typename T, OrderMode Mode> void run2D_mpi(DistributedContext &ctx) {
  double t_rank_start = MPI_Wtime();

  if (N == 0) {
    size_t n = getFileElementCount<T>(inputFiles[0]);
    if (inputFiles.size() == 1)
      n /= 2;
    N = static_cast<int>(n);
  }

  T *h_local_xx = nullptr, *h_local_yy = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_local_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMallocHost(&h_local_yy, N * sizeof(T)));

  readCoordFiles2D<T>(inputFiles, N, h_local_xx, h_local_yy);

  T *d_local_xx, *d_local_yy;
  CUDA_CHECK(cudaMalloc(&d_local_xx, N * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_local_yy, N * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_local_xx, h_local_xx, N * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_local_yy, h_local_yy, N * sizeof(T),
                        cudaMemcpyHostToDevice));

  T min_x, max_x, range_x;
  T min_y, max_y, range_y;
  getRange(d_local_xx, N, min_x, max_x, range_x);
  getRange(d_local_yy, N, min_y, max_y, range_y);

  if (!isABS) {
    xi *= std::min(range_x, range_y);
  }
  b = d * std::sqrt(range_x * range_y / N);

  double ghost_width = b + 2.0 * std::sqrt(2.0) * xi;
  ctx.local_bboxes.resize(1);
  computeLocalBBox2D_GPU(d_local_xx, d_local_yy, N, ctx.local_bboxes[0]);
  discoverNeighbors(ctx, ghost_width, 2, MPI_COMM_WORLD);

  // Ghost exchange: C2 GPU ghost ID, A3 packed buffers, B1 pipelined sends
  std::vector<GhostBuffer<T>> ghosts;
  size_t total_ghost_count;
  auto pending =
      beginGhostExchange2D(d_local_xx, d_local_yy, static_cast<size_t>(N), ctx,
                           ghosts, total_ghost_count, MPI_COMM_WORLD);

  // While MPI transfers: allocate extended arrays, copy local (D2D)
  int N_ext = N + static_cast<int>(total_ghost_count);
  T *d_ext_xx, *d_ext_yy;
  CUDA_CHECK(cudaMalloc(&d_ext_xx, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_ext_yy, (size_t)N_ext * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_ext_xx, d_local_xx, (size_t)N * sizeof(T),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_ext_yy, d_local_yy, (size_t)N * sizeof(T),
                        cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaFree(d_local_xx));
  CUDA_CHECK(cudaFree(d_local_yy));

  // Save send indices before completeGhostExchange clears pending
  std::vector<std::vector<int>> ghost_send_indices = pending.send_indices;

  // B2: Wait for ghost data with progressive completion
  completeGhostExchange(pending, ghosts, total_ghost_count);

  // B3: Copy ghost data to device using async streams
  {
    size_t offset = N;
    int num_nonempty = 0;
    for (const auto &g : ghosts)
      if (g.count > 0)
        num_nonempty++;

    std::vector<cudaStream_t> streams(num_nonempty);
    for (int i = 0; i < num_nonempty; ++i)
      CUDA_CHECK(cudaStreamCreate(&streams[i]));

    int si = 0;
    for (const auto &g : ghosts) {
      if (g.count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_ext_xx + offset, g.xx.data(),
                                   g.count * sizeof(T), cudaMemcpyHostToDevice,
                                   streams[si]));
        CUDA_CHECK(cudaMemcpyAsync(d_ext_yy + offset, g.yy.data(),
                                   g.count * sizeof(T), cudaMemcpyHostToDevice,
                                   streams[si]));
        offset += g.count;
        si++;
      }
    }
    for (int i = 0; i < num_nonempty; ++i) {
      CUDA_CHECK(cudaStreamSynchronize(streams[i]));
      CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
  }

  T ext_min_x, ext_max_x, ext_range_x;
  T ext_min_y, ext_max_y, ext_range_y;
  getRange(d_ext_xx, N_ext, ext_min_x, ext_max_x, ext_range_x);
  getRange(d_ext_yy, N_ext, ext_min_y, ext_max_y, ext_range_y);

  CompressedData<T> compressed;
  CompressionState2D<T> state;

  if (!baseDecompFiles.empty()) {
    T *h_base_xx = nullptr, *h_base_yy = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_base_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&h_base_yy, N * sizeof(T)));
    readCoordFiles2D<T>(baseDecompFiles, N, h_base_xx, h_base_yy);

    T *d_base_xx, *d_base_yy;
    CUDA_CHECK(cudaMalloc(&d_base_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_base_yy, N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_base_xx, h_base_xx, N * sizeof(T),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_base_yy, h_base_yy, N * sizeof(T),
                          cudaMemcpyHostToDevice));

    std::vector<GhostBuffer<T>> base_ghosts;
    reexchangeGhostData2D(ghost_send_indices, h_base_xx, h_base_yy,
                          ghosts, ctx, base_ghosts, MPI_COMM_WORLD);

    T *d_ext_base_xx, *d_ext_base_yy;
    int base_dummy;
    buildExtendedDeviceArrays2D(d_base_xx, d_base_yy, N, base_ghosts,
                                total_ghost_count, d_ext_base_xx, d_ext_base_yy,
                                base_dummy);

    if (xi == 0) {
      T local_xi = getMaxAbsErr(d_ext_xx, d_base_xx, N);
      local_xi = std::max(local_xi, getMaxAbsErr(d_ext_yy, d_base_yy, N));
      float global_xi;
      MPI_Allreduce(&local_xi, &global_xi, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      xi = global_xi;
      if (ctx.rank == 0)
        printf("Auto-detected xi = %e\n", (double)xi);
    }

    CUDA_CHECK(cudaFree(d_base_xx));
    CUDA_CHECK(cudaFree(d_base_yy));

    editParticles2D<T, Mode>(d_ext_xx, d_ext_yy, d_ext_base_xx, d_ext_base_yy,
                             ext_min_x, ext_range_x, ext_min_y, ext_range_y,
                             N_ext, xi, b, state, compressed, N,
                             MPI_COMM_WORLD);
    // NOTE: editParticles2D assigns d_ext_base_* to state.d_decomp_* —
    // state.free() will free these; do NOT free them here.
    destroyHashTable(state.d_editable_pts_ht);

    CUDA_CHECK(cudaFreeHost(h_base_xx));
    CUDA_CHECK(cudaFreeHost(h_base_yy));
  } else {
    if (isEdit) {
      compressWithEditParticles2D<T, Mode>(d_ext_xx, d_ext_yy, ext_min_x,
                                           ext_range_x, ext_min_y, ext_range_y,
                                           N_ext, xi, b, state, compressed, N);
      destroyHashTable(state.d_editable_pts_ht);
    } else {
      compressParticles2D<T, Mode>(d_ext_xx, d_ext_yy, ext_min_x, ext_range_x,
                                   ext_min_y, ext_range_y, N_ext, xi, b, state,
                                   compressed);
      compressed.N_local = static_cast<size_t>(N);
    }
  }

  ////////////////// Export Decomp Start //////////////////
  T *decomp_xx = new T[N];
  T *decomp_yy = new T[N];
  CUDA_CHECK(cudaMemcpy(decomp_xx, state.d_decomp_xx, N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(decomp_yy, state.d_decomp_yy, N * sizeof(T),
                        cudaMemcpyDeviceToHost));
  char decomp_file[512];
  snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.xx.out",
           outputDir.c_str(), ctx.rank);
  writeRawArrayBinary(decomp_xx, N, decomp_file);
  snprintf(decomp_file, sizeof(decomp_file), "%s/rank_%03d.yy.out",
           outputDir.c_str(), ctx.rank);
  writeRawArrayBinary(decomp_yy, N, decomp_file);
  delete[] decomp_xx;
  delete[] decomp_yy;
  /////////////////// Export Decomp End ///////////////////

  state.free();
  CUDA_CHECK(cudaFree(d_ext_xx));
  CUDA_CHECK(cudaFree(d_ext_yy));

  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s/rank_%03d.fofpz", outputDir.c_str(),
           ctx.rank);
  writeCompressedFile(outfile, compressed);
  double t_rank_end = MPI_Wtime();
  printf("Rank %d: wrote %s\n", ctx.rank, outfile);
  printf("[Timer] Total rank time: %f seconds\n", t_rank_end - t_rank_start);
  fflush(stdout);

  CUDA_CHECK(cudaFreeHost(h_local_xx));
  CUDA_CHECK(cudaFreeHost(h_local_yy));
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  DistributedContext ctx;
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

  // Assign each rank to its own GPU based on node-local rank
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, ctx.rank,
                      MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  cudaSetDevice(local_rank % num_devices);
  MPI_Comm_free(&local_comm);

  Parsing(argc, argv);
  substituteRank(inputFiles, ctx.rank);
  substituteRank(baseDecompFiles, ctx.rank);

  if (ctx.rank == 0) {
    if (N > 0)
      printf("MPI GPU FOFPz: %d ranks, D=%d, N=%d per rank\n", ctx.size, D, N);
    else
      printf("MPI GPU FOFPz: %d ranks, D=%d, N=auto-detect\n", ctx.size, D);
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
