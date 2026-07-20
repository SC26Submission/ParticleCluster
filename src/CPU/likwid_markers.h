#pragma once
// Thin wrapper around the LIKWID marker API for per-component CPU roofline
// profiling (memory bandwidth + FLOP/s -> arithmetic intensity).
//
// Build with -DUSE_LIKWID and link -llikwid to enable; otherwise every macro
// compiles to a no-op so the normal build and runtime are unaffected.
//
// Usage under likwid-perfctr:
//   likwid-perfctr -C 0-31 -g MEM      -m -- fofpz_cpu ...   # bandwidth (bytes)
//   likwid-perfctr -C 0-31 -g FLOPS_SP -m -- fofpz_cpu ...   # single-precision FLOPs
//   AI (FLOP/byte) = FLOPs(FLOPS_SP) / bytes(MEM) per region.
//
// OpenMP note: for parallel regions the START/STOP calls must run on every
// thread (i.e. inside the parallel region), so LIKWID reads each core's
// counters. For serial regions the master-thread markers are correct because
// only one core is active.

#ifdef USE_LIKWID
#ifndef LIKWID_PERFMON
#define LIKWID_PERFMON
#endif
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#define LIKWID_MARKER_RESET(regionTag)
#define LIKWID_MARKER_CLOSE
#endif
