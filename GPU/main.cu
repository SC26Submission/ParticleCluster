#include "fileIO.h"
#include "particle_compression.cuh"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

std::vector<std::string> inputFiles;
std::vector<std::string> baseDecompFiles;
std::string compressedFile;
std::string decompressedFile;
int D;         // dimension
int N;         // number of particles
float xi = 0;  // coordinate-wise absolute error bound
float b;       // linking length
float d = 0.2; // dimensionless linking length parameter
bool isDouble;
bool isABS;
bool isEdit = true; // Compress and edit, or compress only
OrderMode mode = OrderMode::MORTON_CODE;
int max_iter = 1000;
double lr = 0.01;

void parseError(const char error[]) {
  fprintf(stderr, "%s\n", error);
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  -i <x> <y> [<z>]: Specify input data files (2 or 3 files "
                  "determines dimension)\n");
  fprintf(stderr, "  -e <x> <y> [<z>]: Specify base-decompressed files "
                  "(optional; same count as -i)\n");
  fprintf(stderr,
          "  -z <file_path>  : Specify the compressed file (optional)\n");
  fprintf(stderr,
          "  -o <file_path>  : Specify the decompressed file (optional)\n");
  fprintf(stderr, "  -D <d>          : dimensionality (2 or 3 for 2D or 3D)\n");
  fprintf(stderr, "  -N <n>          : number of particles\n");
  fprintf(stderr, "  -f              : Use float data type\n");
  fprintf(stderr, "  -d              : Use double data type\n");
  fprintf(stderr, "  -M ABS <xi>     : Specify the absolute error bound\n");
  fprintf(stderr, "  -M REL <xi>     : Specify the relative error bound\n");
  fprintf(stderr, "  -B <d>          : Specify the linking length parameter\n");
  fprintf(stderr, "  -KD             : Use k-d tree as reordering method\n");
  fprintf(stderr, "  -MC             : Use Morton code as reordering method\n");
  fprintf(stderr, "  -lr             : Specify the learning rate in PGD\n");
  fprintf(stderr, "  -iter           : Specify the max iter count in PGD\n");
  fprintf(stderr, "  -c              : Compression only\n");
  fprintf(stderr,
          "  -l              : Losslessly store vulnerable pairs; avoid PGD\n");
  exit(EXIT_FAILURE);
}

void Parsing(int argc, char *argv[]) {
  bool originalFileSpecified = false;
  bool CompressedFileSpecified = false;

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
    } else if (arg == "-z") {
      if (i + 1 >= argc)
        parseError("Missing compressed file path");
      compressedFile = argv[++i];
      CompressedFileSpecified = true;
    } else if (arg == "-o") {
      if (i + 1 >= argc)
        parseError("Missing decompressed file path");
      decompressedFile = argv[++i];
    } else if (arg == "-D") {
      D = std::stoi(argv[++i]);
    } else if (arg == "-N") {
      N = std::stoi(argv[++i]);
    } else if (arg == "-f") {
      isDouble = false; // Use float type
    } else if (arg == "-d") {
      isDouble = true; // Use double type
    } else if (arg == "-M") {
      isABS = std::strcmp(argv[++i], "ABS") == 0;
      if (i + 1 >= argc)
        parseError("Missing relative error bound");
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

  if (!originalFileSpecified && !CompressedFileSpecified) {
    parseError("At least one of original data (-i) and compressed data "
               "(-z) should be identified");
  }

  // Derive D from file count; single file requires -D; -z alone also uses -D
  if (inputFiles.size() > 1)
    D = inputFiles.size();
  else if (baseDecompFiles.size() > 1)
    D = baseDecompFiles.size();
  if (D != 2 && D != 3)
    parseError(
        "Dimension must be 2 or 3: provide 2 or 3 files to -i/-e, or use -D");
}

template <typename T> std::string suffix(const std::string &base) {
  if constexpr (std::is_same_v<T, float>) {
    return base + ".f32";
  } else if constexpr (std::is_same_v<T, double>) {
    return base + ".d64";
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type");
  }
}

template <typename T, OrderMode Mode> void run2D() {
  if (inputFiles.empty()) {
    // Decompression mode
    CompressedData<T> compressed = readCompressedFile<T>(compressedFile);
    T dec_xi = compressed.xi;
    T dec_b = compressed.b;

    T *decomp_xx = new T[N];
    T *decomp_yy = new T[N];

    if (!baseDecompFiles.empty()) {
      // Edit only
      readCoordFiles2D<T>(baseDecompFiles, N, decomp_xx, decomp_yy);
      reconstructEditParticles2D<T>(compressed, decomp_xx, decomp_yy, N,
                                    dec_xi);
    } else {
      if (isEdit) {
        // Base and PGD edit
        decompressWithEditParticles2D<T, Mode>(compressed, decomp_xx, decomp_yy,
                                               N, dec_xi, dec_b);
      } else {
        // Base only
        decompressParticles2D<T, Mode>(compressed, decomp_xx, decomp_yy, N,
                                       dec_xi, dec_b);
      }
    }
    writeRawArrayBinary(decomp_xx, N, decompressedFile + suffix<T>("xx"));
    writeRawArrayBinary(decomp_yy, N, decompressedFile + suffix<T>("yy"));
    delete[] decomp_xx;
    delete[] decomp_yy;
  } else {
    // Compression mode
    T *org_xx = nullptr;
    T *org_yy = nullptr;
    CUDA_CHECK(cudaMallocHost(&org_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&org_yy, N * sizeof(T)));

    readCoordFiles2D<T>(inputFiles, N, org_xx, org_yy);

    T *d_org_xx, *d_org_yy;
    CUDA_CHECK(cudaMalloc(&d_org_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_org_yy, N * sizeof(T)));

    // Async copies
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_org_xx, org_xx, N * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_org_yy, org_yy, N * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    T min_x, max_x, range_x;
    T min_y, max_y, range_y;

    getRange(d_org_xx, N, min_x, max_x, range_x);
    getRange(d_org_yy, N, min_y, max_y, range_y);

    if (!isABS) {
      xi *= std::min(range_x, range_y);
    }
    b = d * std::sqrt(range_x * range_y / N);

    CompressedData<T> compressed;
    CompressionState2D<T> state;
    if (!baseDecompFiles.empty()) {
      // Edit-only: upload base decompressed data and run edit
      T *base_xx = nullptr;
      T *base_yy = nullptr;
      CUDA_CHECK(cudaMallocHost(&base_xx, N * sizeof(T)));
      CUDA_CHECK(cudaMallocHost(&base_yy, N * sizeof(T)));
      readCoordFiles2D<T>(baseDecompFiles, N, base_xx, base_yy);
      T *d_base_xx = nullptr;
      T *d_base_yy = nullptr;
      CUDA_CHECK(cudaMalloc(&d_base_xx, N * sizeof(T)));
      CUDA_CHECK(cudaMalloc(&d_base_yy, N * sizeof(T)));
      cudaStream_t stream_edit;
      CUDA_CHECK(cudaStreamCreate(&stream_edit));
      CUDA_CHECK(cudaMemcpyAsync(d_base_xx, base_xx, N * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_edit));
      CUDA_CHECK(cudaMemcpyAsync(d_base_yy, base_yy, N * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_edit));
      CUDA_CHECK(cudaStreamSynchronize(stream_edit));
      CUDA_CHECK(cudaStreamDestroy(stream_edit));

      if (xi == 0) {
        T max_ae_xx = getMaxAbsErr(d_org_xx, d_base_xx, N);
        T max_ae_yy = getMaxAbsErr(d_org_yy, d_base_yy, N);
        xi = std::max(max_ae_xx, max_ae_yy);
      }

      editParticles2D<T, Mode>(d_org_xx, d_org_yy, d_base_xx, d_base_yy, min_x,
                               range_x, min_y, range_y, N, xi, b, state,
                               compressed);
      destroyHashTable(state.d_editable_pts_ht);

      CUDA_CHECK(cudaFreeHost(base_xx));
      CUDA_CHECK(cudaFreeHost(base_yy));
    } else {
      if (isEdit) {
        compressWithEditParticles2D<T, Mode>(d_org_xx, d_org_yy, min_x, range_x,
                                             min_y, range_y, N, xi, b, state,
                                             compressed);
      } else {
        // Compress only
        compressParticles2D<T, Mode>(d_org_xx, d_org_yy, min_x, range_x, min_y,
                                     range_y, N, xi, b, state, compressed);
      }
      ////////////////// Export Order Start //////////////////
      int *order = new int[N];
      getVisitOrder(state, order);
      writeRawArrayBinary(order, N, decompressedFile + "order.dat");
      /////////////////// Export Order End ///////////////////
    }
    state.free();
    CUDA_CHECK(cudaFree(d_org_xx));
    CUDA_CHECK(cudaFree(d_org_yy));

    writeCompressedFile(compressedFile, compressed);

    CUDA_CHECK(cudaFreeHost(org_xx));
    CUDA_CHECK(cudaFreeHost(org_yy));
  }
}

template <typename T, OrderMode Mode> void run3D() {
  if (inputFiles.empty()) {
    // Decompression mode
    CompressedData<T> compressed = readCompressedFile<T>(compressedFile);
    T dec_xi = compressed.xi;
    T dec_b = compressed.b;

    T *decomp_xx = new T[N];
    T *decomp_yy = new T[N];
    T *decomp_zz = new T[N];

    if (!baseDecompFiles.empty()) {
      // Edit only
      readCoordFiles3D<T>(baseDecompFiles, N, decomp_xx, decomp_yy, decomp_zz);
      reconstructEditParticles3D<T>(compressed, decomp_xx, decomp_yy, decomp_zz,
                                    N, dec_xi);
    } else {
      if (isEdit) {
        // Base and PGD edit
        decompressWithEditParticles3D<T, Mode>(compressed, decomp_xx, decomp_yy,
                                               decomp_zz, N, dec_xi, dec_b);
      } else {
        decompressParticles3D<T, Mode>(compressed, decomp_xx, decomp_yy,
                                       decomp_zz, N, dec_xi, dec_b);
      }
    }
    writeRawArrayBinary(decomp_xx, N, decompressedFile + suffix<T>("xx"));
    writeRawArrayBinary(decomp_yy, N, decompressedFile + suffix<T>("yy"));
    writeRawArrayBinary(decomp_zz, N, decompressedFile + suffix<T>("zz"));
    delete[] decomp_xx;
    delete[] decomp_yy;
    delete[] decomp_zz;
  } else {
    // Compression mode
    T *org_xx = nullptr;
    T *org_yy = nullptr;
    T *org_zz = nullptr;
    CUDA_CHECK(cudaMallocHost(&org_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&org_yy, N * sizeof(T)));
    CUDA_CHECK(cudaMallocHost(&org_zz, N * sizeof(T)));

    readCoordFiles3D<T>(inputFiles, N, org_xx, org_yy, org_zz);

    T *d_org_xx, *d_org_yy, *d_org_zz;
    CUDA_CHECK(cudaMalloc(&d_org_xx, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_org_yy, N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_org_zz, N * sizeof(T)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpyAsync(d_org_xx, org_xx, N * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_org_yy, org_yy, N * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_org_zz, org_zz, N * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    T min_x, max_x, range_x;
    T min_y, max_y, range_y;
    T min_z, max_z, range_z;

    getRange(d_org_xx, N, min_x, max_x, range_x);
    getRange(d_org_yy, N, min_y, max_y, range_y);
    getRange(d_org_zz, N, min_z, max_z, range_z);

    if (!isABS) {
      xi *= std::min({range_x, range_y, range_z});
    }
    b = d * std::cbrt(range_x * range_y * range_z / N);

    CompressedData<T> compressed;
    CompressionState3D<T> state;
    if (!baseDecompFiles.empty()) {
      // Edit-only
      T *base_xx = nullptr;
      T *base_yy = nullptr;
      T *base_zz = nullptr;
      CUDA_CHECK(cudaMallocHost(&base_xx, N * sizeof(T)));
      CUDA_CHECK(cudaMallocHost(&base_yy, N * sizeof(T)));
      CUDA_CHECK(cudaMallocHost(&base_zz, N * sizeof(T)));
      readCoordFiles3D<T>(baseDecompFiles, N, base_xx, base_yy, base_zz);
      T *d_base_xx = nullptr;
      T *d_base_yy = nullptr;
      T *d_base_zz = nullptr;
      CUDA_CHECK(cudaMalloc(&d_base_xx, N * sizeof(T)));
      CUDA_CHECK(cudaMalloc(&d_base_yy, N * sizeof(T)));
      CUDA_CHECK(cudaMalloc(&d_base_zz, N * sizeof(T)));
      cudaStream_t stream_edit;
      CUDA_CHECK(cudaStreamCreate(&stream_edit));
      CUDA_CHECK(cudaMemcpyAsync(d_base_xx, base_xx, N * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_edit));
      CUDA_CHECK(cudaMemcpyAsync(d_base_yy, base_yy, N * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_edit));
      CUDA_CHECK(cudaMemcpyAsync(d_base_zz, base_zz, N * sizeof(T),
                                 cudaMemcpyHostToDevice, stream_edit));
      CUDA_CHECK(cudaStreamSynchronize(stream_edit));
      CUDA_CHECK(cudaStreamDestroy(stream_edit));

      if (xi == 0) {
        T max_ae_xx = getMaxAbsErr(d_org_xx, d_base_xx, N);
        T max_ae_yy = getMaxAbsErr(d_org_yy, d_base_yy, N);
        xi = std::max(max_ae_xx, max_ae_yy);
        T max_ae_zz = getMaxAbsErr(d_org_zz, d_base_zz, N);
        xi = std::max(xi, static_cast<float>(max_ae_zz));
      }

      editParticles3D<T, Mode>(
          d_org_xx, d_org_yy, d_org_zz, d_base_xx, d_base_yy, d_base_zz, min_x,
          range_x, min_y, range_y, min_z, range_z, N, xi, b, state, compressed);
      destroyHashTable(state.d_editable_pts_ht);

      CUDA_CHECK(cudaFreeHost(base_xx));
      CUDA_CHECK(cudaFreeHost(base_yy));
      CUDA_CHECK(cudaFreeHost(base_zz));
    } else {
      if (isEdit) {
        compressWithEditParticles3D<T, Mode>(
            d_org_xx, d_org_yy, d_org_zz, min_x, range_x, min_y, range_y, min_z,
            range_z, N, xi, b, state, compressed);
      } else {
        // Compress only
        compressParticles3D<T, Mode>(d_org_xx, d_org_yy, d_org_zz, min_x,
                                     range_x, min_y, range_y, min_z, range_z, N,
                                     xi, b, state, compressed);
      }
    }

    ////////////////// Export Decomp Start //////////////////
    T *decomp_xx = new T[N];
    T *decomp_yy = new T[N];
    T *decomp_zz = new T[N];
    getDecompressedCoords3D(state, decomp_xx, decomp_yy, decomp_zz);
    writeRawArrayBinary(decomp_xx, N, decompressedFile + "xx.out");
    writeRawArrayBinary(decomp_yy, N, decompressedFile + "yy.out");
    writeRawArrayBinary(decomp_zz, N, decompressedFile + "zz.out");
    /////////////////// Export Decomp End ///////////////////

    // ////////////////// Export Order Start //////////////////
    // int *order = new int[N];
    // getVisitOrder(state, order);
    // writeRawArrayBinary(order, N, decompressedFile + "order.dat");
    // /////////////////// Export Order End ///////////////////

    state.free();
    CUDA_CHECK(cudaFree(d_org_xx));
    CUDA_CHECK(cudaFree(d_org_yy));
    CUDA_CHECK(cudaFree(d_org_zz));

    writeCompressedFile(compressedFile, compressed);

    CUDA_CHECK(cudaFreeHost(org_xx));
    CUDA_CHECK(cudaFreeHost(org_yy));
    CUDA_CHECK(cudaFreeHost(org_zz));
  }
}

int main(int argc, char *argv[]) {
  Parsing(argc, argv);

  if (isDouble) {
    if (D == 2) {
      if (mode == OrderMode::KD_TREE) {
        run2D<double, OrderMode::KD_TREE>();
      } else {
        run2D<double, OrderMode::MORTON_CODE>();
      }
    } else {
      if (mode == OrderMode::KD_TREE) {
        run3D<double, OrderMode::KD_TREE>();
      } else {
        run3D<double, OrderMode::MORTON_CODE>();
      }
    }
  } else {
    if (D == 2) {
      if (mode == OrderMode::KD_TREE) {
        run2D<float, OrderMode::KD_TREE>();
      } else {
        run2D<float, OrderMode::MORTON_CODE>();
      }
    } else {
      if (mode == OrderMode::KD_TREE) {
        run3D<float, OrderMode::KD_TREE>();
      } else {
        run3D<float, OrderMode::MORTON_CODE>();
      }
    }
  }

  return 0;
}