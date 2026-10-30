/*
    batch.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Single-source (C++17 / CUDA) batched Monte Carlo: the ONE definition of
    how samples are grouped, evaluated and reduced, shared by
      - the CUDA kernel (cuda/device_cuda.cu: one CUDA block = one "block"
        here),
      - the host GPU emulator (src/device_emu.cpp: lanes run one after the
        other),
      - the CPU (OpenMP) backend (src/backend.cpp: one OpenMP thread = one
        block).
    All three run rb_block_run<RbExec> with the same executor RbExec, whose
    methods are __host__ __device__ and switch on __CUDA_ARCH__ only at the
    level of "one lane per CUDA thread + __syncthreads" vs "loop over the
    lanes", atomicAdd vs a host atomic, clock64 vs a host clock.
    Because all three run the same template (rb_block_run) with the same
    constants, a class's accumulator depends only on (seed, stream, sample
    range) -- never on the backend, the grid size, the number of GPUs or of
    threads, or the order in which chunks are executed -- up to the last
    bits of the per-sample weights (device libm vs host libm, FMA
    contraction).

    Canonical grouping:
      chunk = RB_CHUNK = RB_BLOCK * RB_SPT consecutive sample indices of
        one class (the last chunk of a range may be partial).  Chunk
        accumulators are merged in increasing chunk order on the host.
      lane t = samples first + j*RB_BLOCK + t, j = 0..RB_SPT-1, accumulated
        sequentially (rb_acc_add);
      block = the RB_BLOCK lane accumulators of a chunk merged by a fixed
        binary tree (s = RB_BLOCK/2, ..., 1: acc[t] += acc[t+s] for t < s).
      unit = RB_UNIT_CHUNKS consecutive chunks of a task (sample range),
        aligned at the task's first sample; its chunk accumulators are
        merged in chunk order (from zero), and the units of a task in unit
        order.  The unit is also the scheduling granularity of the
        multi-device backend.  (Compensated merges are not associative bit
        for bit, so this two-level order is part of the definition.)
    A launch processes a list of work items (contiguous sample ranges of
    possibly different classes); its chunks are numbered 0..nchunks-1 and
    handed out dynamically (atomic counter) to persistent blocks.

    Executor interface (X = RbExec), written so that a missing barrier is
    impossible by construction:
      RbBlockShared &shared(): block-shared state (__shared__ on the
        device)
      void par(F f): f(t) for every lane t of the block, then a block
        barrier
      void one(F f): f() on lane 0 only, then a block barrier
      unsigned long long next_chunk(): atomic fetch-and-increment of the
        launch's chunk counter
      unsigned long long ticks(): clock (device: clock64 cycles; host: ns)
      R *scratch<R>(int t); int stride(): per-lane logP scratch (element a
        at scratch<R>(t)[a*stride()])

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
#include "kernel.h"

#ifndef RB_BLOCK
/* Lanes (CUDA threads) per block -- part of the canonical reduction order */
#define RB_BLOCK 128
#endif
#ifndef RB_SPT
/* Samples per lane per chunk */
#define RB_SPT 32
#endif
#define RB_CHUNK ((uint64_t)RB_BLOCK * (uint64_t)RB_SPT)
#ifndef RB_UNIT_CHUNKS
/* Chunks per unit (canonical second reduction level) */
#define RB_UNIT_CHUNKS 16
#endif
#define RB_UNIT ((uint64_t)RB_UNIT_CHUNKS * RB_CHUNK)
static_assert((RB_BLOCK & (RB_BLOCK - 1)) == 0 && RB_BLOCK >= 32,
  "RB_BLOCK must be a power of two >= 32");

/* One contiguous sample range of one class inside a launch. */
struct RbWorkItem {
  int cls; /* slot in the views array */
  uint32_t stream; /* Philox counter word 2 (kind << 26; legacy scheme: | class index) */
  uint64_t key; /* Philox key of the class (rb_class_key(seed, salt); legacy scheme: the seed) */
  uint64_t idx0; /* first sample index */
  uint64_t n; /* number of samples (> 0) */
  uint64_t chunk0; /* launch-chunk index of the item's first chunk */
};
/* Result of one chunk. */
struct RbChunkOut {
  RbAcc acc;
  /* Duration of the chunk on its block (cycles on the GPU, ns on the host) */
  unsigned long long ticks;
};
struct RbBlockShared {
  RbAcc acc[RB_BLOCK];
  RbClassView P;
  unsigned long long k, t0;
  int item;
  int bad; /* the class view of the current chunk failed the sanity check */
};

RB_HDI uint64_t rb_nchunks(uint64_t n) { return (n + RB_CHUNK - 1) / RB_CHUNK; }

/* Sanity check of a class view before any sample is drawn from it.  A zeroed view (class not
   resident, an upload not yet visible to the kernel, an evicted slot) would otherwise give l = 0,
   ngroup = 0, i.e. 4096 VALID ZERO weights per chunk -- silently biased integrals.  A failing
   chunk is reported with n = NaN, so the host's per-unit sample-count check fails and the device
   is dropped. */
RB_HDI bool rb_view_ok(const RbClassView &P) {
  return P.l >= 1 && P.l <= RB_MAXE && P.natoms >= 0 && P.natoms <= RB_MAXATOMS && P.ngroup >= 1
      && P.omega && P.Jt && P.grp_off && P.grp_foff && P.comp_coeff && P.comp_d0 && P.comp_d1
      && (P.natoms == 0 || (P.atom_off && P.atom_nv && P.atom_loc && P.ae_u && P.ae_v));
}

/* Work item containing launch chunk k (items sorted by chunk0, items[0].chunk0 == 0). */
RB_HDI int rb_find_item(const RbWorkItem *it, int nitems, uint64_t k) {
  int lo = 0, hi = nitems - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (it[mid].chunk0 <= k) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/* Lane t of a chunk starting at sample index `first`, with `nleft` samples left in its work item.
   R = precision of the evaluation (double; float only with --fp32 for the bare kinds); accumulation
   is always FP64 and is kept in a local accumulator (registers), stored once at the end. */
template <class R>
RB_HDS void rb_lane(const RbClassView &P, uint64_t key, uint32_t stream, uint64_t first,
    uint64_t nleft, int t, R *logP, int lps, RbAcc *acc_out) {
  RbAcc acc;
  rb_acc_zero(&acc);
  for (int j = 0; j < RB_SPT; ++j) {
    const uint64_t off = (uint64_t)j * RB_BLOCK + (uint64_t)t;
    if (off >= nleft) break;
    R sabs;
    const R w = rb_sample<R>(P, key, stream, first + off, logP, &sabs, lps);
    rb_acc_add(&acc, (double)w, (double)sabs);
  }
  *acc_out = acc;
}

/* The block program: persistent loop over the launch's chunks.  views[] may be in device or host
   memory. */
template <class X, class R = double>
RB_HDS void rb_block_run(X &x, const RbClassView *views, const RbWorkItem *items, int nitems,
    uint64_t nchunks, RbChunkOut *out) {
  RbBlockShared &S = x.shared();
  for (;;) {
    x.one([&]() {
      S.k = x.next_chunk();
      if (S.k < nchunks) {
        S.item = rb_find_item(items, nitems, S.k);
        S.P = views[items[S.item].cls];
        S.bad = rb_view_ok(S.P) ? 0 : 1;
        S.t0 = x.ticks();
      }
    });
    if (S.k >= nchunks) break; /* uniform across the block: S.k was written before the barrier */
    if (S.bad) { /* uniform as well; no sample is drawn from a bad view */
      x.one([&]() { RbAcc z; rb_acc_zero(&z); z.n = NAN; out[S.k].acc = z; out[S.k].ticks = 0; });
      continue;
    }
    const RbWorkItem it = items[S.item];
    const uint64_t c = S.k - it.chunk0;
    const uint64_t first = it.idx0 + c * RB_CHUNK, nleft = it.n - c * RB_CHUNK;
    x.par([&](int t) {
      rb_lane<R>(S.P, it.key, it.stream, first, nleft, t, x.template scratch<R>(t), x.stride(),
        &S.acc[t]);
    });
    for (int s = RB_BLOCK / 2; s > 0; s >>= 1)
      x.par([&](int t) { if (t < s) rb_acc_merge(&S.acc[t], &S.acc[t + s]); });
    x.one([&]() { out[S.k].acc = S.acc[0]; out[S.k].ticks = x.ticks() - S.t0; });
  }
}

/* Diagnostics shared by the device self-test and the host reference.
   Hash of the sampled Hepp sector (edge order): identical on every backend, since the sector choice
   uses only correctly rounded +,*,/ on bit-identical Philox uniforms and tables. */
RB_HDI uint32_t rb_ord_hash(const signed char *ord, int l) {
  uint32_t h = 2166136261u;
  for (int i = 0; i < l; ++i) { h ^= (uint32_t)(unsigned char)ord[i]; h *= 16777619u; }
  return h;
}
struct RbSampleDbg { double w, sabs, lvs; uint32_t hash; uint32_t pad; };
RB_HDS void rb_sample_dbg(const RbClassView &P, uint64_t key, uint32_t stream, uint64_t idx,
    double *logP, int lps, RbSampleDbg *o) {
  RbStream rs;
  rb_stream_init(&rs, key, stream, idx);
  signed char ord[RB_MAXE];
  double lx[RB_MAXE];
  const double lvs = rb_point(P, &rs, ord, lx);
  double sabs;
  o->w = rb_weight_at<double>(P, ord, lx, lvs, logP, &sabs, lps);
  o->sabs = sabs;
  o->lvs = lvs;
  o->hash = rb_ord_hash(ord, P.l);
  o->pad = 0;
}

/* Fixed inputs for the start-up FP64 / Philox sanity test (same table on host and device). */
#define RB_NFPTEST 16
RB_HDI double rb_fptest_arg(int i) {
  const double a[RB_NFPTEST] = {1.0, 0.5, 2.0, 1e-300, 1e300, 3.141592653589793, -7.25, 0.1, -700.0,
    700.0, 1.0000000000000002, 123456.789, -1e-10, 37.5, 0.999999999, 4.0e-5};
  return a[i];
}
/* out[4*i+0..3] = exp(-|a| or a), log(|a|), sqrt(|a|), fma(a, 1/3, a*a) */
RB_HDS void rb_fptest(int i, double *out) {
  const double a = rb_fptest_arg(i);
  const double e = (rb_fabs(a) < 700.5) ? a : -rb_fabs(a);
  out[4 * i + 0] = rb_exp(e);
  out[4 * i + 1] = rb_log(rb_fabs(a));
  out[4 * i + 2] = rb_sqrt(rb_fabs(a));
  out[4 * i + 3] = fma(a, 1.0 / 3.0, a * 1e-3);
}

/* Philox4x32-10 known-answer vectors (Random123 kat_vectors, commit 9545ff64): input i ->
   out[4*i..4*i+3] */
#define RB_NKAT 3
RB_HDS void rb_kat(int i, uint32_t *out) {
  const uint32_t c[RB_NKAT][4] = {{0u, 0u, 0u, 0u},
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
    {0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}};
  const uint32_t k[RB_NKAT][2] = {{0u, 0u}, {0xffffffffu, 0xffffffffu}, {0xa4093822u, 0x299f31d0u}};
  RbPhilox4 x;
  for (int j = 0; j < 4; ++j) x.v[j] = c[i][j];
  const RbPhilox4 o = rb_philox4x32_10(x, k[i][0], k[i][1]);
  for (int j = 0; j < 4; ++j) out[4 * i + j] = o.v[j];
}
static const uint32_t rb_kat_expected[RB_NKAT * 4] = {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu,
  0x9b00dbd8u, 0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu, 0xd16cfe09u, 0x94fdccebu,
  0x5001e420u, 0x24126ea1u};

/* Build signature: every compile-time constant and struct layout that host and device must agree
   on.  The device computes it in its start-up test (rb_fptest_kernel) and the host compares it with
   its own: a -D option that reached only some compiles (e.g. -DRB_MAXE=14 in NVCC_EXTRA but not for
   the host builder, which lays out atom_loc with stride RB_MAXE) makes the device fail its
   self-test instead of producing garbage. */
RB_HDI uint64_t rb_build_signature() {
  const double lxm = RB_LXMIN, lxf = RB_LXMIN_F32;
  uint64_t v[16] = {(uint64_t)RB_MAXE, (uint64_t)RB_MAXV, (uint64_t)RB_MAXLOOP,
    (uint64_t)RB_MAXATOMS, (uint64_t)RB_BLOCK, (uint64_t)RB_SPT, (uint64_t)RB_UNIT_CHUNKS,
    (uint64_t)(int64_t)(lxm * 1024.0), (uint64_t)(int64_t)(lxf * 1024.0),
    (uint64_t)sizeof(RbClassView), (uint64_t)sizeof(RbWorkItem), (uint64_t)sizeof(RbChunkOut),
    (uint64_t)sizeof(RbAcc), (uint64_t)sizeof(RbBlockShared), 3u /* format version */, 0u};
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < 16; ++i) { h ^= v[i]; h *= 1099511628211ULL; h = rb_mix64(h); }
  return h;
}

/* The executor. */
#if !defined(__CUDA_ARCH__)
#include <chrono>
inline unsigned long long rb_host_ns() {
  return (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
#endif
/* Device: one lane per CUDA thread (threadIdx.x), barrier = __syncthreads(), `sh` points to
     __shared__ memory.
   Host: one block per host thread; the lanes of a phase run one after the other -- in forward
     order, or in reverse order (race test: a phase whose result depends on the lane order hides a
     missing barrier); the barrier is the end of that loop. */
struct RbExec {
  RbBlockShared *sh;
  /* Launch-wide chunk counter (device global memory, or host memory) */
  unsigned long long *counter;
  void *base; /* scratch of lane t: (R *)base + t * lane_step, element a at [a * lps] */
  long long lane_step; /* in elements of the evaluation type R */
  int lps;
  int reverse; /* host only */
  RB_HD RbBlockShared &shared() { return *sh; }
  template <class F> RB_HD void par(F f) {
#if defined(__CUDA_ARCH__)
    f((int)threadIdx.x);
    __syncthreads();
#else
    if (!reverse) for (int t = 0; t < RB_BLOCK; ++t) f(t);
    else for (int t = RB_BLOCK - 1; t >= 0; --t) f(t);
#endif
  }
  template <class F> RB_HD void one(F f) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0) f();
    __syncthreads();
#else
    f();
#endif
  }
  RB_HD unsigned long long next_chunk() {
#if defined(__CUDA_ARCH__)
    return atomicAdd(counter, 1ull);
#else
    return __atomic_fetch_add(counter, 1ull, __ATOMIC_RELAXED);
#endif
  }
  RB_HD unsigned long long ticks() {
#if defined(__CUDA_ARCH__)
    return (unsigned long long)clock64();
#else
    return rb_host_ns();
#endif
  }
  template <class R> RB_HD R *scratch(int t) { return (R *)base + (long long)t * lane_step; }
  RB_HD int stride() const { return lps; }
};
/* Host-side block: owns its shared state (11 KB).  Visible in both nvcc passes: host code of a CUDA
   translation unit is parsed in the device pass too (it is never compiled for the device). */
#include <memory>
struct RbHostBlock {
  std::unique_ptr<RbBlockShared> mem{new RbBlockShared};
  RbExec x;
  RbHostBlock(unsigned long long *counter, void *base, long long lane_step, int lps, bool reverse) {
    x.sh = mem.get(); x.counter = counter; x.base = base; x.lane_step = lane_step; x.lps = lps;
    x.reverse = reverse ? 1 : 0;
  }
};
