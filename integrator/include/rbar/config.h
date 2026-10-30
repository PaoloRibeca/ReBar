/*
    config.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Portability layer for the single-source (C++17 host / CUDA device)
    R-bar kernel.

    Everything in include/rbar/ that is marked RB_HD compiles both as
    ordinary C++17 (CPU, OpenMP driver) and as CUDA __host__ __device__
    code.  Device-side code uses no dynamic memory, no exceptions, no STL,
    and only fixed-size per-thread arrays whose bounds are the compile-time
    limits below.
    Host-only code (graph building, forests, symbolic expansion, drivers)
    lives in src/.

    This program was designed and developed by the author(s),
    with the assistance of the following AI tool(s):
      2026 Claude (Anthropic).
    The final logic and implementation were reviewed and verified in
    their entirety by the author(s).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include <stdint.h>
#include <math.h>

#if defined(__CUDACC__)
#define RB_HD __host__ __device__
#define RB_FI __forceinline__
#else
#define RB_HD
#define RB_FI inline
#endif
/* Free functions of the headers have INTERNAL linkage (static): every object file keeps its own
   copy, compiled by its own compiler with its own -D settings.  (With plain inline/template
   functions the linker keeps one weak copy for the whole program, so e.g. the nvcc-compiled
   emulator could silently run the g++-compiled copy.)  RB_HDI: small, force-inlined; RB_HDS:
   larger, inline hint only. */
#define RB_HDI static RB_HD RB_FI
#define RB_HDS static RB_HD inline

/* Read-only gathers of class tables through the non-coherent (texture) path on the device */
#if defined(__CUDA_ARCH__)
#define RB_LDG(p) __ldg(p)
#else
#define RB_LDG(p) (*(p))
#endif

#if defined(__CUDA_ARCH__)
#define RB_CTZ(x) (__ffs((int)(x)) - 1)
#define RB_POPC(x) __popc(x)
#else
#define RB_CTZ(x) __builtin_ctz(x)
#define RB_POPC(x) __builtin_popcount(x)
#endif

/* Compile-time limits (override with -D...).
   RB_MAXE: lines per graph (16 = 8-loop Gamma4; Gamma2 at 8 loops has 15)
   RB_MAXV: vertices of an atom (block of a minor) <= vertices of the graph (9 at 8-loop Gamma4)
   RB_MAXLOOP: loops of an atom (<= loops of the graph)
   RB_MAXATOMS: distinct polynomial atoms per graph class (per-thread scratch array of doubles) */
#ifndef RB_MAXE
#define RB_MAXE 16
#endif
#ifndef RB_MAXV
#define RB_MAXV 12
#endif
#ifndef RB_MAXLOOP
#define RB_MAXLOOP 9
#endif
#ifndef RB_MAXATOMS
#define RB_MAXATOMS 768
#endif

/* Precision-generic elementary functions. */
#if defined(RB_LIBM_PERTURB) && !defined(__CUDA_ARCH__)
/* TEST-ONLY (host): emulate a different libm, as on the GPU (CUDA's exp/log differ from glibc's by
   up to 1-2 ulp), by moving every exp/log result by k ulp, k in [-RB_LIBM_PERTURB,
   +RB_LIBM_PERTURB], a deterministic hash of the argument bits.  Used to calibrate the CPU-vs-GPU
   acceptance tolerance. */
#include <string.h>
static RB_FI double rb_perturb(double y, double x) {
  uint64_t b; memcpy(&b, &x, 8);
  b ^= b >> 31; b *= 0xbf58476d1ce4e5b9ULL; b ^= b >> 29; b *= 0x94d049bb133111ebULL; b ^= b >> 32;
  int k = (int)(b % (2u * RB_LIBM_PERTURB + 1u)) - RB_LIBM_PERTURB;
  for (; k > 0; --k) y = nextafter(y, INFINITY);
  for (; k < 0; ++k) y = nextafter(y, -INFINITY);
  return y;
}
static RB_FI double rb_exp(double x) { return rb_perturb(exp(x), x); }
static RB_FI double rb_log(double x) { return rb_perturb(log(x), x); }
#else
RB_HDI double rb_exp(double x) { return exp(x); }
RB_HDI double rb_log(double x) { return log(x); }
#endif
RB_HDI double rb_sqrt(double x) { return sqrt(x); }
RB_HDI double rb_fabs(double x) { return fabs(x); }
RB_HDI bool rb_isfinite(double x) { return isfinite(x); }
/* Single precision (--fp32: evaluation of the BARE kinds only; accumulation stays FP64) */
RB_HDI float rb_exp(float x) { return expf(x); }
RB_HDI float rb_log(float x) { return logf(x); }
RB_HDI float rb_sqrt(float x) { return sqrtf(x); }
RB_HDI float rb_fabs(float x) { return fabsf(x); }
RB_HDI bool rb_isfinite(float x) { return isfinite(x); }
/* Precision of the per-sample bookkeeping (log of the density, mass factor): never below double */
template <class R> struct rb_wide { typedef R T; };
template <> struct rb_wide<float> { typedef double T; };
/* Fast path of the atom evaluation (x = exp(lx) precomputed) only when every lx exceeds this (no
   underflow) */
#ifndef RB_LXMIN
#define RB_LXMIN (-600.0)
#endif
#ifndef RB_LXMIN_F32
#define RB_LXMIN_F32 (-80.0)
#endif
RB_HDI double rb_lxmin(double) { return RB_LXMIN; }
RB_HDI double rb_lxmin(float) { return RB_LXMIN_F32; }
#if !defined(__CUDA_ARCH__)
/* long double (x87 80-bit on x86-64: 64-bit mantissa) -- host-only reference precision for paired
   checks */
static inline long double rb_exp(long double x) { return expl(x); }
static inline long double rb_log(long double x) { return logl(x); }
static inline long double rb_sqrt(long double x) { return sqrtl(x); }
static inline long double rb_fabs(long double x) { return fabsl(x); }
static inline bool rb_isfinite(long double x) { return isfinite(x); }
static inline double rb_lxmin(long double) { return RB_LXMIN; }
#endif
