#include "fileIO.h"
#include "particle_compression.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

std::vector<std::string> inputFiles;
std::vector<std::string> baseDecompFiles;
std::string compressedFile;
std::string decompressedFile;
size_t D;      // dimension
size_t N;      // number of particles
float xi;      // coordinate-wise absolute error bound
float b;       // linking length
float d = 0.2; // dimensionless linking length parameter
bool isDouble;
bool isABS;
bool isEdit = true; // Compress and edit, or compress only
bool isPGD = true;  // PGD for or losslessly store vulnerable pairs
OrderMode mode = OrderMode::MORTON_CODE;
size_t max_iter = 1000;
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
      D = std::stoull(argv[++i]);
    } else if (arg == "-N") {
      N = std::stoull(argv[++i]);
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
      max_iter = std::stoull(argv[++i]);
    } else if (arg == "-c") {
      isEdit = false;
    } else if (arg == "-l") {
      isPGD = false;
    } else {
      parseError("Unknown argument");
    }
  }

  if (!originalFileSpecified && !CompressedFileSpecified) {
    parseError("At least one of original data (-i) and compressed data "
               "(-z) should be identified");
  }
  if (!isEdit && !isPGD) {
    parseError("Compression mode; no need to avoid PGD");
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

template <typename T>
void getRange(const T *arr, size_t N, T &minVal, T &maxVal, T &rangeVal) {
  minVal = arr[0];
  maxVal = arr[0];
  for (size_t i = 0; i < N; ++i) {
    if (arr[i] < minVal)
      minVal = arr[i];
    if (arr[i] > maxVal)
      maxVal = arr[i];
  }
  rangeVal = maxVal - minVal;
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
      if (isPGD) {
        reconstructEditParticles2D<T>(compressed, decomp_xx, decomp_yy, N,
                                      dec_xi);
      } else {
        auto decomp_start = std::chrono::high_resolution_clock::now();
        applyLosslessEdits2D<T>(compressed, decomp_xx, decomp_yy, N);
        auto decomp_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decomp_time = decomp_end - decomp_start;
        printf("Decompression time: %f seconds\n", decomp_time.count());
      }
    } else {
      if (isEdit && isPGD) {
        // Base and PGD edit
        decompressWithEditParticles2D<T, Mode>(compressed, decomp_xx, decomp_yy,
                                               N, dec_xi, dec_b);
      } else {
        // Base only (or base + lossless edit override)
        auto decomp_start = std::chrono::high_resolution_clock::now();
        decompressParticles2D<T, Mode>(compressed, decomp_xx, decomp_yy, N,
                                       dec_xi, dec_b);
        if (isEdit && !isPGD) {
          applyLosslessEdits2D<T>(compressed, decomp_xx, decomp_yy, N);
        }
        auto decomp_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decomp_time = decomp_end - decomp_start;
        printf("Decompression time: %f seconds\n", decomp_time.count());
      }
    }
    writeRawArrayBinary(decomp_xx, N, decompressedFile + suffix<T>("xx"));
    writeRawArrayBinary(decomp_yy, N, decompressedFile + suffix<T>("yy"));
    delete[] decomp_xx;
    delete[] decomp_yy;
  } else {
    // Compression mode
    T *org_xx = new T[N];
    T *org_yy = new T[N];

    readCoordFiles2D<T>(inputFiles, N, org_xx, org_yy);

    T min_x, max_x, range_x;
    T min_y, max_y, range_y;

    getRange(org_xx, N, min_x, max_x, range_x);
    getRange(org_yy, N, min_y, max_y, range_y);

    if (!isABS) {
      xi *= std::min(range_x, range_y);
    }
    b = d * std::sqrt(range_x * range_y / N);

    CompressionResults2D<T> result;
    CompressedData<T> compressed;
    if (!baseDecompFiles.empty()) {
      // Edit only
      result.decomp_xx = new T[N];
      result.decomp_yy = new T[N];
      readCoordFiles2D<T>(baseDecompFiles, N, result.decomp_xx,
                          result.decomp_yy);
      editParticles2D<T>(org_xx, org_yy, min_x, range_x, min_y, range_y, N, xi,
                         b, isPGD, result, compressed);
    } else {
      if (isEdit) {
        // Compress & edit
        compressWithEditParticles2D<T, Mode>(org_xx, org_yy, min_x, range_x,
                                             min_y, range_y, N, xi, b, isPGD,
                                             result, compressed);
      } else {
        // Compress only
        compressParticles2D<T, Mode>(org_xx, org_yy, min_x, range_x, min_y,
                                     range_y, N, xi, b, result, compressed);
      }
    }
    writeCompressedFile(compressedFile, compressed);

    // //////////////////////// Debug, remove later ////////////////////////
    // writeVectorBinary(result.decomp_xx, decompressedFile + suffix<T>("xx"));
    // writeVectorBinary(result.decomp_yy, decompressedFile + suffix<T>("yy"));
    // writeVectorBinary(result.visit_order, decompressedFile + "order.uint64");

    delete[] org_xx;
    delete[] org_yy;
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
      if (isPGD) {
        reconstructEditParticles3D<T>(compressed, decomp_xx, decomp_yy,
                                      decomp_zz, N, dec_xi);
      } else {
        auto decomp_start = std::chrono::high_resolution_clock::now();
        applyLosslessEdits3D<T>(compressed, decomp_xx, decomp_yy, decomp_zz, N);
        auto decomp_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decomp_time = decomp_end - decomp_start;
        printf("Decompression time: %f seconds\n", decomp_time.count());
      }
    } else {
      if (isEdit && isPGD) {
        // Base and PGD edit
        decompressWithEditParticles3D<T, Mode>(compressed, decomp_xx, decomp_yy,
                                               decomp_zz, N, dec_xi, dec_b);
      } else {
        // Base only (or base + lossless edit override)
        auto decomp_start = std::chrono::high_resolution_clock::now();
        decompressParticles3D<T, Mode>(compressed, decomp_xx, decomp_yy,
                                       decomp_zz, N, dec_xi, dec_b);
        if (isEdit && !isPGD) {
          applyLosslessEdits3D<T>(compressed, decomp_xx, decomp_yy, decomp_zz,
                                  N);
        }
        auto decomp_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> decomp_time = decomp_end - decomp_start;
        printf("Decompression time: %f seconds\n", decomp_time.count());
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
    T *org_xx = new T[N];
    T *org_yy = new T[N];
    T *org_zz = new T[N];

    readCoordFiles3D<T>(inputFiles, N, org_xx, org_yy, org_zz);

    T min_x, max_x, range_x;
    T min_y, max_y, range_y;
    T min_z, max_z, range_z;

    getRange(org_xx, N, min_x, max_x, range_x);
    getRange(org_yy, N, min_y, max_y, range_y);
    getRange(org_zz, N, min_z, max_z, range_z);

    if (!isABS) {
      xi *= std::min({range_x, range_y, range_z});
    }
    b = d * std::cbrt(range_x * range_y * range_z / N);

    CompressionResults3D<T> result;
    CompressedData<T> compressed;
    if (!baseDecompFiles.empty()) {
      // Edit only
      result.decomp_xx = new T[N];
      result.decomp_yy = new T[N];
      result.decomp_zz = new T[N];
      readCoordFiles3D<T>(baseDecompFiles, N, result.decomp_xx,
                          result.decomp_yy, result.decomp_zz);
      editParticles3D<T>(org_xx, org_yy, org_zz, min_x, range_x, min_y, range_y,
                         min_z, range_z, N, xi, b, isPGD, result, compressed);
    } else {
      if (isEdit) {
        // Compress & edit
        compressWithEditParticles3D<T, Mode>(
            org_xx, org_yy, org_zz, min_x, range_x, min_y, range_y, min_z,
            range_z, N, xi, b, isPGD, result, compressed);
      } else {
        // Compress only
        compressParticles3D<T, Mode>(org_xx, org_yy, org_zz, min_x, range_x,
                                     min_y, range_y, min_z, range_z, N, xi, b,
                                     result, compressed);
      }
    }
    writeCompressedFile(compressedFile, compressed);

    // //////////////////////// Debug, remove later ////////////////////////
    // writeVectorBinary(result.decomp_xx, decompressedFile + suffix<T>("xx"));
    // writeVectorBinary(result.decomp_yy, decompressedFile + suffix<T>("yy"));
    // writeVectorBinary(result.decomp_zz, decompressedFile + suffix<T>("zz"));
    // writeVectorBinary(result.visit_order, decompressedFile + "order.uint64");

    delete[] org_xx;
    delete[] org_yy;
    delete[] org_zz;
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
}