#pragma once

#include "util.cuh"
#include <cuda_runtime.h>

// Custom atomicAdd for long long (64-bit signed)
__device__ __forceinline__ long long atomicAdd(long long *address,
                                               long long val) {
  unsigned long long *address_as_ull = (unsigned long long *)address;
  unsigned long long old = *address_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    (unsigned long long)(val + (long long)assumed));
  } while (assumed != old);
  return (long long)old;
}

// Forward declarations for kernels defined in FOFHaloFinder.cu
__global__ void initUnionFind_kernel(int *parent, int *rank, int N);

// ============================================================================
// ARI (Adjusted Rand Index) Calculation
// ============================================================================

// GPU Union-Find structure
struct UnionFind {
  int *parent; // parent array for union-find
  int *rank;   // rank array for union-find
  int capacity;

  UnionFind() : parent(nullptr), rank(nullptr), capacity(0) {}
};

// Create and initialize Union-Find structure
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
    parent[x] = parent[parent[x]]; // path compression
    x = parent[x];
  }
  return x;
}

// Union operation
__device__ __forceinline__ void union_uf(int *parent, int *rank, int x, int y) {
  x = find_uf(parent, x);
  y = find_uf(parent, y);
  if (x == y)
    return;

  if (rank[x] < rank[y]) {
    int tmp = x;
    x = y;
    y = tmp;
  }
  parent[y] = x;
  if (rank[x] == rank[y]) {
    rank[x]++;
  }
}

// Edge structure for linking particles
struct Edge {
  int src;
  int dst;
};

// Public API functions (implementations in FOFHaloFinder.cu)
template <typename T>
void calculateARI2D(const T *d_org_xx, const T *d_org_yy, T *d_decomp_xx,
                    T *d_decomp_yy, T min_x, T range_x, T min_y, T range_y,
                    int N, T b, long long &h_tp, long long &h_tn,
                    long long &h_fp, long long &h_fn);

template <typename T>
void calculateARI3D(const T *d_org_xx, const T *d_org_yy, const T *d_org_zz,
                    T *d_decomp_xx, T *d_decomp_yy, T *d_decomp_zz, T min_x,
                    T range_x, T min_y, T range_y, T min_z, T range_z, int N,
                    T b, long long &h_tp, long long &h_tn, long long &h_fp,
                    long long &h_fn);
