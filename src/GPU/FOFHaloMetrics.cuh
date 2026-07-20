#pragma once

#include "FOFHaloFinder.cuh"

template <typename T>
T calculateARIFromPairCounts(long long tp, long long tn, long long fp,
                             long long fn) {
  T tp_v = static_cast<T>(tp);
  T tn_v = static_cast<T>(tn);
  T fp_v = static_cast<T>(fp);
  T fn_v = static_cast<T>(fn);

  T denom = (tp_v + fn_v) * (fn_v + tn_v) +
            (tp_v + fp_v) * (fp_v + tn_v);
  if (denom == T(0))
    return T(1);
  return 2 * (tp_v * tn_v - fp_v * fn_v) / denom;
}

void calculateARIFromRoots(const int *d_org_roots, const int *d_decomp_roots,
                           int N, long long &h_tp, long long &h_tn,
                           long long &h_fp, long long &h_fn);

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
