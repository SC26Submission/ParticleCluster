#pragma once

#include "config.cuh"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

enum class DataType { FLOAT, DOUBLE, SIZE_T, INT };

void writeMap(const std::map<int, std::pair<uint32_t, int>> &m,
              const std::string &filename);

std::map<int, std::pair<uint32_t, int>> readMap(const std::string &filename);

void readRawArrayBinary(const std::string &fileName, void *data, std::size_t N,
                        DataType type);

template <typename T>
void writeRawArrayBinary(const T *data, size_t N, const std::string &filename) {
  static_assert(std::is_trivially_copyable<T>::value,
                "Type must be trivially copyable.");

  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for writing." << std::endl;
    return;
  }

  // Write the data to the binary file
  file.write(reinterpret_cast<const char *>(data), N * sizeof(T));

  if (!file) {
    std::cerr << "Error: Failed to write to file." << std::endl;
  }

  file.close();
}

template <typename T>
std::vector<T> readVectorBinary(const std::string &filename) {
  static_assert(std::is_trivially_copyable<T>::value,
                "Type must be trivially copyable.");

  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (!ifs) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::streamsize file_size = ifs.tellg();
  if (file_size % sizeof(T) != 0) {
    throw std::runtime_error("File size is not a multiple of element size: " +
                             filename);
  }

  size_t N = static_cast<size_t>(file_size / sizeof(T));
  std::vector<T> vec(N);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(vec.data()), file_size);

  if (!ifs) {
    throw std::runtime_error("Error reading file: " + filename);
  }

  return vec;
}

template <typename T>
void writeVectorBinary(const std::vector<T> &vec, const std::string &filename) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  ofs.write(reinterpret_cast<const char *>(vec.data()), sizeof(T) * vec.size());
}

template <typename T>
void writeCompressedFile(const std::string &filename,
                         const CompressedData<T> &data) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }

  ofs.write(reinterpret_cast<const char *>(&data.size_flag),
            sizeof(data.size_flag));
  ofs.write(reinterpret_cast<const char *>(&data.size_quant),
            sizeof(data.size_quant));
  ofs.write(reinterpret_cast<const char *>(&data.size_edit),
            sizeof(data.size_edit));

  // Write code_table_flag
  size_t map_size = data.code_table_flag.size();
  ofs.write(reinterpret_cast<const char *>(&map_size), sizeof(map_size));
  for (const auto &[key, val] : data.code_table_flag) {
    ofs.write(reinterpret_cast<const char *>(&key), sizeof(key));
    ofs.write(reinterpret_cast<const char *>(&val.first), sizeof(val.first));
    ofs.write(reinterpret_cast<const char *>(&val.second), sizeof(val.second));
  }

  // Write code_table_quant
  map_size = data.code_table_quant.size();
  ofs.write(reinterpret_cast<const char *>(&map_size), sizeof(map_size));
  for (const auto &[key, val] : data.code_table_quant) {
    ofs.write(reinterpret_cast<const char *>(&key), sizeof(key));
    ofs.write(reinterpret_cast<const char *>(&val.first), sizeof(val.first));
    ofs.write(reinterpret_cast<const char *>(&val.second), sizeof(val.second));
  }

  // Write code_table_edit
  map_size = data.code_table_edit.size();
  ofs.write(reinterpret_cast<const char *>(&map_size), sizeof(map_size));
  for (const auto &[key, val] : data.code_table_edit) {
    ofs.write(reinterpret_cast<const char *>(&key), sizeof(key));
    ofs.write(reinterpret_cast<const char *>(&val.first), sizeof(val.first));
    ofs.write(reinterpret_cast<const char *>(&val.second), sizeof(val.second));
  }

  ofs.write(reinterpret_cast<const char *>(&data.bit_stream_size_flag),
            sizeof(data.bit_stream_size_flag));
  ofs.write(reinterpret_cast<const char *>(&data.bit_stream_size_quant),
            sizeof(data.bit_stream_size_quant));
  ofs.write(reinterpret_cast<const char *>(&data.bit_stream_size_edit),
            sizeof(data.bit_stream_size_edit));

  auto writeVec = [&](const std::vector<uint8_t> &v) {
    size_t sz = v.size();
    ofs.write(reinterpret_cast<const char *>(&sz), sizeof(sz));
    ofs.write(reinterpret_cast<const char *>(v.data()), sz);
  };
  writeVec(data.compressed_lossless_flag);
  writeVec(data.compressed_quant_codes);
  writeVec(data.compressed_quant_edits);

  // Write new fields
  size_t lossless_size = data.lossless_values.size();
  ofs.write(reinterpret_cast<const char *>(&lossless_size),
            sizeof(lossless_size));
  ofs.write(reinterpret_cast<const char *>(data.lossless_values.data()),
            sizeof(T) * lossless_size);

  ofs.write(reinterpret_cast<const char *>(&data.grid_min_x),
            sizeof(data.grid_min_x));
  ofs.write(reinterpret_cast<const char *>(&data.grid_min_y),
            sizeof(data.grid_min_y));
  ofs.write(reinterpret_cast<const char *>(&data.grid_min_z),
            sizeof(data.grid_min_z));
  ofs.write(reinterpret_cast<const char *>(&data.grid_dim_x),
            sizeof(data.grid_dim_x));
  ofs.write(reinterpret_cast<const char *>(&data.grid_dim_y),
            sizeof(data.grid_dim_y));
  ofs.write(reinterpret_cast<const char *>(&data.grid_dim_z),
            sizeof(data.grid_dim_z));

  ofs.write(reinterpret_cast<const char *>(&data.xi), sizeof(data.xi));
  ofs.write(reinterpret_cast<const char *>(&data.b), sizeof(data.b));

  // Write lossless edit mask (Huffman+ZSTD compressed bitmask)
  ofs.write(reinterpret_cast<const char *>(&data.size_edit_flag),
            sizeof(data.size_edit_flag));
  ofs.write(reinterpret_cast<const char *>(&data.bit_stream_size_edit_flag),
            sizeof(data.bit_stream_size_edit_flag));
  size_t mask_map_size = data.code_table_edit_flag.size();
  ofs.write(reinterpret_cast<const char *>(&mask_map_size),
            sizeof(mask_map_size));
  for (const auto &[key, val] : data.code_table_edit_flag) {
    ofs.write(reinterpret_cast<const char *>(&key), sizeof(key));
    ofs.write(reinterpret_cast<const char *>(&val.first), sizeof(val.first));
    ofs.write(reinterpret_cast<const char *>(&val.second), sizeof(val.second));
  }
  size_t compressed_mask_size = data.compressed_lossless_edit_flag.size();
  ofs.write(reinterpret_cast<const char *>(&compressed_mask_size),
            sizeof(compressed_mask_size));
  ofs.write(
      reinterpret_cast<const char *>(data.compressed_lossless_edit_flag.data()),
      compressed_mask_size);
  size_t edit_val_count = data.lossless_edit_values.size();
  ofs.write(reinterpret_cast<const char *>(&edit_val_count),
            sizeof(edit_val_count));
  ofs.write(reinterpret_cast<const char *>(data.lossless_edit_values.data()),
            sizeof(T) * edit_val_count);

  // Write sparse edit mapping
  size_t evp_count = data.editable_visit_positions.size();
  ofs.write(reinterpret_cast<const char *>(&evp_count), sizeof(evp_count));
  if (evp_count > 0)
    ofs.write(
        reinterpret_cast<const char *>(data.editable_visit_positions.data()),
        sizeof(int) * evp_count);

  // Write N_local for distributed mode (0 = all local)
  ofs.write(reinterpret_cast<const char *>(&data.N_local),
            sizeof(data.N_local));
}

template <typename T>
CompressedData<T> readCompressedFile(const std::string &filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Could not open file for reading: " + filename);
  }
  CompressedData<T> data;

  ifs.read(reinterpret_cast<char *>(&data.size_flag), sizeof(data.size_flag));
  ifs.read(reinterpret_cast<char *>(&data.size_quant), sizeof(data.size_quant));
  ifs.read(reinterpret_cast<char *>(&data.size_edit), sizeof(data.size_edit));

  // Read code_table_flag
  size_t map_size;
  ifs.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));
  for (size_t i = 0; i < map_size; ++i) {
    uint8_t key;
    uint32_t code;
    int len;
    ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
    ifs.read(reinterpret_cast<char *>(&code), sizeof(code));
    ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
    data.code_table_flag[key] = {code, len};
  }

  // Read code_table_quant
  ifs.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));
  for (size_t i = 0; i < map_size; ++i) {
    UInt key;
    uint32_t code;
    int len;
    ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
    ifs.read(reinterpret_cast<char *>(&code), sizeof(code));
    ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
    data.code_table_quant[key] = {code, len};
  }

  // Read code_table_edit
  ifs.read(reinterpret_cast<char *>(&map_size), sizeof(map_size));
  for (size_t i = 0; i < map_size; ++i) {
    UInt2 key;
    uint32_t code;
    int len;
    ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
    ifs.read(reinterpret_cast<char *>(&code), sizeof(code));
    ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
    data.code_table_edit[key] = {code, len};
  }

  ifs.read(reinterpret_cast<char *>(&data.bit_stream_size_flag),
           sizeof(data.bit_stream_size_flag));
  ifs.read(reinterpret_cast<char *>(&data.bit_stream_size_quant),
           sizeof(data.bit_stream_size_quant));
  ifs.read(reinterpret_cast<char *>(&data.bit_stream_size_edit),
           sizeof(data.bit_stream_size_edit));

  auto readVec = [&](std::vector<uint8_t> &v) {
    size_t sz;
    ifs.read(reinterpret_cast<char *>(&sz), sizeof(sz));
    v.resize(sz);
    ifs.read(reinterpret_cast<char *>(v.data()), sz);
  };
  readVec(data.compressed_lossless_flag);
  readVec(data.compressed_quant_codes);
  readVec(data.compressed_quant_edits);

  // Read new fields
  size_t lossless_size;
  ifs.read(reinterpret_cast<char *>(&lossless_size), sizeof(lossless_size));
  data.lossless_values.resize(lossless_size);
  ifs.read(reinterpret_cast<char *>(data.lossless_values.data()),
           sizeof(T) * lossless_size);

  ifs.read(reinterpret_cast<char *>(&data.grid_min_x), sizeof(data.grid_min_x));
  ifs.read(reinterpret_cast<char *>(&data.grid_min_y), sizeof(data.grid_min_y));
  ifs.read(reinterpret_cast<char *>(&data.grid_min_z), sizeof(data.grid_min_z));
  ifs.read(reinterpret_cast<char *>(&data.grid_dim_x), sizeof(data.grid_dim_x));
  ifs.read(reinterpret_cast<char *>(&data.grid_dim_y), sizeof(data.grid_dim_y));
  ifs.read(reinterpret_cast<char *>(&data.grid_dim_z), sizeof(data.grid_dim_z));

  ifs.read(reinterpret_cast<char *>(&data.xi), sizeof(data.xi));
  ifs.read(reinterpret_cast<char *>(&data.b), sizeof(data.b));

  // Read lossless edit mask (Huffman+ZSTD compressed bitmask)
  ifs.read(reinterpret_cast<char *>(&data.size_edit_flag),
           sizeof(data.size_edit_flag));
  ifs.read(reinterpret_cast<char *>(&data.bit_stream_size_edit_flag),
           sizeof(data.bit_stream_size_edit_flag));
  size_t mask_map_size;
  ifs.read(reinterpret_cast<char *>(&mask_map_size), sizeof(mask_map_size));
  for (size_t i = 0; i < mask_map_size; ++i) {
    uint8_t key;
    uint32_t code;
    int len;
    ifs.read(reinterpret_cast<char *>(&key), sizeof(key));
    ifs.read(reinterpret_cast<char *>(&code), sizeof(code));
    ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
    data.code_table_edit_flag[key] = {code, len};
  }
  size_t compressed_mask_size;
  ifs.read(reinterpret_cast<char *>(&compressed_mask_size),
           sizeof(compressed_mask_size));
  data.compressed_lossless_edit_flag.resize(compressed_mask_size);
  ifs.read(reinterpret_cast<char *>(data.compressed_lossless_edit_flag.data()),
           compressed_mask_size);
  size_t edit_val_count;
  ifs.read(reinterpret_cast<char *>(&edit_val_count), sizeof(edit_val_count));
  data.lossless_edit_values.resize(edit_val_count);
  ifs.read(reinterpret_cast<char *>(data.lossless_edit_values.data()),
           sizeof(T) * edit_val_count);

  // Read sparse edit mapping
  size_t evp_count = 0;
  if (ifs.read(reinterpret_cast<char *>(&evp_count), sizeof(evp_count))) {
    data.editable_visit_positions.resize(evp_count);
    if (evp_count > 0)
      ifs.read(reinterpret_cast<char *>(data.editable_visit_positions.data()),
               sizeof(int) * evp_count);
  }

  // Read N_local for distributed mode (optional, default 0 = all local)
  ifs.read(reinterpret_cast<char *>(&data.N_local), sizeof(data.N_local));
  if (!ifs) data.N_local = 0;

  return data;
}

template <typename T> constexpr DataType dtype() {
  if constexpr (std::is_same_v<T, float>)
    return DataType::FLOAT;
  else if constexpr (std::is_same_v<T, double>)
    return DataType::DOUBLE;
  else
    static_assert(sizeof(T) == 0, "Unsupported type for dtype()");
}

// #5: Streaming block-wise deinterleave — reads in L1-friendly blocks
template <typename T>
void readInterleavedBinary(const std::string &filename, size_t N, size_t dim,
                           T **outs) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("Could not open file: " + filename);

  constexpr size_t BLOCK = 4096;
  std::vector<T> buf(BLOCK * dim);

  for (size_t off = 0; off < N; off += BLOCK) {
    size_t chunk = std::min(BLOCK, N - off);
    ifs.read(reinterpret_cast<char *>(buf.data()), chunk * dim * sizeof(T));
    if (!ifs)
      throw std::runtime_error("Error reading file: " + filename);
    for (size_t i = 0; i < chunk; ++i)
      for (size_t d = 0; d < dim; ++d)
        outs[d][off + i] = buf[i * dim + d];
  }
}

// #5: Streaming block-wise deinterleave for vectors
template <typename T>
void readInterleavedBinaryVecs(const std::string &filename, size_t dim,
                               std::vector<T> *vecs) {
  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (!ifs)
    throw std::runtime_error("Could not open file: " + filename);
  size_t total = static_cast<size_t>(ifs.tellg()) / sizeof(T);
  size_t N = total / dim;
  for (size_t d = 0; d < dim; ++d)
    vecs[d].resize(N);

  ifs.seekg(0, std::ios::beg);
  constexpr size_t BLOCK = 4096;
  std::vector<T> buf(BLOCK * dim);

  for (size_t off = 0; off < N; off += BLOCK) {
    size_t chunk = std::min(BLOCK, N - off);
    ifs.read(reinterpret_cast<char *>(buf.data()), chunk * dim * sizeof(T));
    if (!ifs)
      throw std::runtime_error("Error reading file: " + filename);
    for (size_t i = 0; i < chunk; ++i)
      for (size_t d = 0; d < dim; ++d)
        vecs[d][off + i] = buf[i * dim + d];
  }
}

// Infer element count from file size.
template <typename T>
size_t getFileElementCount(const std::string &fileName) {
  std::ifstream f(fileName, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    std::cerr << "Error: Cannot open " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }
  size_t bytes = static_cast<size_t>(f.tellg());
  if (bytes % sizeof(T) != 0) {
    std::cerr << "Error: File size not a multiple of element size: " << fileName
              << std::endl;
    exit(EXIT_FAILURE);
  }
  return bytes / sizeof(T);
}

// #10: Parallel file reads — separate files are read concurrently via threads.
template <typename T>
void readCoordFiles2D(const std::vector<std::string> &files, size_t N, T *xx,
                      T *yy) {
  if (files.size() == 1) {
    T *ptrs[] = {xx, yy};
    readInterleavedBinary<T>(files[0], N, 2, ptrs);
  } else {
    std::thread t0([&]() { readRawArrayBinary(files[0], xx, N, dtype<T>()); });
    std::thread t1([&]() { readRawArrayBinary(files[1], yy, N, dtype<T>()); });
    t0.join();
    t1.join();
  }
}

template <typename T>
void readCoordFiles3D(const std::vector<std::string> &files, size_t N, T *xx,
                      T *yy, T *zz) {
  if (files.size() == 1) {
    T *ptrs[] = {xx, yy, zz};
    readInterleavedBinary<T>(files[0], N, 3, ptrs);
  } else {
    std::thread t0([&]() { readRawArrayBinary(files[0], xx, N, dtype<T>()); });
    std::thread t1([&]() { readRawArrayBinary(files[1], yy, N, dtype<T>()); });
    std::thread t2([&]() { readRawArrayBinary(files[2], zz, N, dtype<T>()); });
    t0.join();
    t1.join();
    t2.join();
  }
}
