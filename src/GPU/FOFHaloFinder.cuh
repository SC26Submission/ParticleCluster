#pragma once

#include "util.cuh"
#include <cuda_runtime.h>

// Union-Find structure
struct UnionFind {
  int *parent; // parent array for union-find
  int *rank;   // rank array for union-find
  int capacity;

  UnionFind() : parent(nullptr), rank(nullptr), capacity(0) {}
};

// Create Union-Find structure (not initialize)
inline UnionFind createUnionFind(int N) {
  UnionFind uf;
  uf.capacity = N;
  CUDA_CHECK(cudaMalloc(&uf.parent, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&uf.rank, N * sizeof(int)));
  return uf;
}

// Cleanup Union-Find structure
inline void destroyUnionFind(UnionFind &uf) {
  if (uf.parent)
    CUDA_CHECK(cudaFree(uf.parent));
  if (uf.rank)
    CUDA_CHECK(cudaFree(uf.rank));
  uf.parent = nullptr;
  uf.rank = nullptr;
  uf.capacity = 0;
}

// Find operation
__device__ __forceinline__ int find_uf(int *parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]]; // path compression (halving)
    x = parent[x];
  }
  return x;
}

__device__ __forceinline__ int find_uf_readonly(int *parent, int x) {
  int p = parent[x];
  while (p != x) {
    x = p;
    p = parent[x];
  }
  return x;
}

// Union operation
__device__ __forceinline__ void union_uf(int *parent, int *rank, int x, int y) {
  (void)rank;
  while (true) {
    x = find_uf_readonly(parent, x);
    y = find_uf_readonly(parent, y);
    if (x == y)
      return;

    int hi = max(x, y);
    int lo = min(x, y);
    int old = atomicCAS(&parent[hi], hi, lo);
    if (old == hi)
      return;
  }
}

// Forward declarations for kernels (defined in FOFHaloFinder.cu)
__global__ void initUnionFind_kernel(int *parent, int *rank, int N);
__global__ void flattenRoots_kernel(int *parent, int *roots, int N);

// Public API functions (implemented in FOFHaloFinder.cu)
template <typename T>
void computeFoFRoots2D(const T *d_xx, const T *d_yy, T min_x, T range_x,
                       T min_y, T range_y, int N, T b, int **d_roots_out);

template <typename T>
void computeFoFRoots3D(const T *d_xx, const T *d_yy, const T *d_zz, T min_x,
                       T range_x, T min_y, T range_y, T min_z, T range_z, int N,
                       T b, int **d_roots_out);
