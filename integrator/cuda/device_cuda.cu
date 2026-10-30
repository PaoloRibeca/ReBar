/*
    device_cuda.cu -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    The CUDA device behind the "cuda" backend (src/backend.h).

    Kernel: rb_batch_kernel<R> = the single-source block program
    rb_block_run<RbExec, R> (include/rbar/batch.h), the same program the
    emulator and the CPU backend run on the host: one CUDA block of
    RB_BLOCK threads = one block of the canonical algorithm, persistent
    (grid = the number of co-resident blocks from the occupancy API),
    chunks handed out by a launch-wide atomic counter, FP64 per-thread
    accumulation (Neumaier-compensated, in registers), FP64 block tree
    reduction in shared memory, one RbChunkOut per chunk (accumulator +
    clock64 cycles), merged on the host in chunk order.  R = double
    (production) or float (--fp32: evaluation of the bare kinds only).
    Per-thread logP scratch is interleaved over all threads of the grid
    (element a of thread g at [a * (grid*RB_BLOCK) + g]) so that the
    accesses of a warp coalesce.

    Host side: one Device per GPU (the scheduler drives each from its own
    host thread).  All copies and memsets use the device's own
    (non-blocking) stream followed by a synchronisation, so every class
    image and view is in place before the next kernel on that stream
    (a pageable cudaMemcpy on the legacy stream does not order the view
    writes before kernels on a non-blocking stream).
    Images and views are verified by read-back; a launch is refused for a
    class that is not resident; the kernel itself refuses invalid views
    (rb_view_ok).
    Every wait on the device goes through a watchdog (no blocking
    synchronisation without a timeout), every CUDA call is checked, and the
    device is dropped (its work requeued) on any error.

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

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "backend.h"

#ifndef RB_MINBLOCKS
// __launch_bounds__ min blocks per SM (register cap).  8 since v0.4.0 (H200, 5 loops: +30 % over 1,
// the v0.3.0 value; 6: +20 %); the Makefile passes MINBLOCKS (default 8)
#define RB_MINBLOCKS 8
#endif
#ifndef RB_FMAD
// nvcc --fmad setting as passed by the Makefile (1 true, 0 false; -1 unknown): reporting only
#define RB_FMAD -1
#endif

template <class R>
__global__ void __launch_bounds__(RB_BLOCK, RB_MINBLOCKS)
rb_batch_kernel(const RbClassView *views, const RbWorkItem *items, int nitems,
    unsigned long long nchunks, double *scratch, unsigned long long *counter, RbChunkOut *out) {
  __shared__ RbBlockShared sh;
  RbExec x;
  x.sh = &sh;
  x.counter = counter;
  // Thread t of block b: element a at [a * grid*RB_BLOCK + b*RB_BLOCK + t]
  x.base = (R *)scratch + (size_t)blockIdx.x * RB_BLOCK;
  x.lane_step = 1;
  x.lps = (int)(gridDim.x * RB_BLOCK);
  x.reverse = 0;
  rb_block_run<RbExec, R>(x, views, items, nitems, (uint64_t)nchunks, out);
}

__global__ void rb_fptest_kernel(double *fp, uint32_t *kat, unsigned long long *sig) {
  const int i = threadIdx.x;
  if (i < RB_NFPTEST) rb_fptest(i, fp);
  if (i < RB_NKAT) rb_kat(i, kat);
  if (i == 0) *sig = (unsigned long long)rb_build_signature();
}

__global__ void
rb_dbg_kernel(const RbClassView *views, int cls, unsigned long long key, unsigned int stream,
    unsigned long long idx0, int n, double *scratch, RbSampleDbg *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const RbClassView P = views[cls];
  if (!rb_view_ok(P)) {
    out[i].w = NAN;
    out[i].sabs = NAN;
    out[i].lvs = NAN;
    out[i].hash = 0xffffffffu;
    out[i].pad = 1;
    return;
  }
  rb_sample_dbg(P, (uint64_t)key, (uint32_t)stream, (uint64_t)idx0 + (uint64_t)i, scratch + i, n,
    &out[i]);
}

#define CK(call) \
  do { \
    const cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
      if (err) \
        *err = std::string(#call) + " -> " + cudaGetErrorName(e_) + ": " + cudaGetErrorString(e_); \
      return false; \
    } \
  } while (0)

namespace {
struct CudaFree { void operator()(void *p) const { if (p) cudaFree(p); } };
template <class T> using DevPtr = std::unique_ptr<T, CudaFree>;
template <class T> static bool dev_alloc(DevPtr<T> &p, size_t count, std::string *err) {
  void *q = nullptr;
  CK(cudaMalloc(&q, sizeof(T) * count));
  p.reset((T *)q);
  return true;
}

class CudaDevice : public Device {
  int ord_;
  double mem_frac_, hang_s_;
  bool fp32_;
  cudaDeviceProp prop_;
  cudaFuncAttributes fa_;
  int bpsm_ = 0, G_ = 0, clock_khz_ = 0, fp64_ratio_ = 0;
  size_t free0_ = 0, total0_ = 0, budget_ = 0, stack_ = 0;
  int ncls_ = 0, natoms_max_ = 1;
  RbClassView *d_views_ = nullptr;
  double *d_scratch_ = nullptr;
  unsigned long long *d_counter_ = nullptr;
  RbWorkItem *d_items_ = nullptr;
  size_t items_cap_ = 0;
  RbChunkOut *d_out_ = nullptr;
  size_t out_cap_ = 0;
  std::map<int, void *> img_;
  cudaStream_t st_ = nullptr;
  cudaEvent_t e0_ = nullptr, e1_ = nullptr, ew_ = nullptr;
  bool opened_ = false, hung_ = false;
  // Wait for everything queued on st_ so far, polling an event; a device that does not finish
  // within `limit_s` is declared hung (the process then leaves with _Exit instead of waiting on the
  // context forever).
  bool wait_or_hang(double limit_s, const char *what, std::string *err) {
    CK(cudaEventRecord(ew_, st_));
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      const cudaError_t q = cudaEventQuery(ew_);
      if (q == cudaSuccess) return true;
      if (q != cudaErrorNotReady) CK(q);
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > limit_s) {
        *err = std::string(what) + " did not complete within " + std::to_string(limit_s)
            + " s (hung device?)";
        hung_ = true;
        g_rb_hung = true;
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }
  bool check_launch(std::string *err) { CK(cudaGetLastError()); return true; }
  // Device interface.
 public:
  CudaDevice(int ord, double mem_frac, double hang_s, bool fp32) : ord_(ord), mem_frac_(mem_frac),
      hang_s_(hang_s), fp32_(fp32) {
    memset(&prop_, 0, sizeof prop_);
    memset(&fa_, 0, sizeof fa_);
  }
  ~CudaDevice() override {
    if (!opened_ || hung_ || g_rb_hung) return; // a hung context cannot be torn down safely
    if (cudaSetDevice(ord_) != cudaSuccess) return;
    for (auto &p : img_) cudaFree(p.second);
    cudaFree(d_views_); cudaFree(d_scratch_); cudaFree(d_counter_); cudaFree(d_items_);
    cudaFree(d_out_);
    if (e0_) cudaEventDestroy(e0_);
    if (e1_) cudaEventDestroy(e1_);
    if (ew_) cudaEventDestroy(ew_);
    if (st_) cudaStreamDestroy(st_);
  }
  std::string label() const override {
    return "cuda:" + std::to_string(ord_) + " " + std::string(prop_.name);
  }
  bool bind(std::string *err) override { CK(cudaSetDevice(ord_)); return true; }
  int blocks() const override { return G_; }
  double tick_hz() const override { return clock_khz_ > 0 ? 1e3 * (double)clock_khz_ : 1e9; }
  size_t mem_budget() const override { return budget_; }
  // Opening of the device.
  bool open(int nclasses, int natoms_max, std::string *err) override {
    CK(cudaSetDevice(ord_));
    CK(cudaGetDeviceProperties(&prop_, ord_));
    CK(cudaDeviceGetAttribute(&clock_khz_, cudaDevAttrClockRate, ord_));
    CK(cudaDeviceGetAttribute(&fp64_ratio_, cudaDevAttrSingleToDoublePrecisionPerfRatio, ord_));
    if (prop_.major < 6) {
      *err = "compute capability " + std::to_string(prop_.major) + "." + std::to_string(prop_.minor)
          + " < 6.0";
      return false;
    }
    {
      const cudaError_t e = fp32_ ? cudaFuncGetAttributes(&fa_, rb_batch_kernel<float>)
          : cudaFuncGetAttributes(&fa_, rb_batch_kernel<double>);
      if (e != cudaSuccess) {
        *err = std::string("kernel not loadable on this device (binary built for other "
          "architectures? rebuild with CUDA_ARCH including ") + std::to_string(prop_.major)
            + std::to_string(prop_.minor) + "): " + cudaGetErrorString(e);
        cudaGetLastError();
        return false;
      }
    }
    if (fp32_)
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm_, rb_batch_kernel<float>, RB_BLOCK,
        0));
    else
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm_, rb_batch_kernel<double>, RB_BLOCK,
        0));
    if (bpsm_ < 1) {
      *err = "occupancy calculator: kernel cannot be resident (registers/shared memory)";
      return false;
    }
    G_ = bpsm_ * prop_.multiProcessorCount;
    // Per-thread stack: the kernel's frame is known statically (ptxas: ~1.1 KB > the 1024 B default
    // limit); make the limit explicit with a small margin (it is reserved for every resident
    // thread: ~2048 x SMs x limit bytes)
    CK(cudaDeviceGetLimit(&stack_, cudaLimitStackSize));
    const size_t need = ((size_t)fa_.localSizeBytes + 512 + 255) & ~(size_t)255;
    if (stack_ < need) {
      CK(cudaDeviceSetLimit(cudaLimitStackSize, need));
      CK(cudaDeviceGetLimit(&stack_, cudaLimitStackSize));
    }
    CK(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking));
    CK(cudaEventCreate(&e0_));
    CK(cudaEventCreate(&e1_));
    CK(cudaEventCreateWithFlags(&ew_, cudaEventDisableTiming));
    opened_ = true;
    ncls_ = std::max(1, nclasses);
    natoms_max_ = std::max(1, natoms_max);
    CK(cudaMalloc(&d_views_, sizeof(RbClassView) * (size_t)ncls_));
    CK(cudaMemsetAsync(d_views_, 0, sizeof(RbClassView) * (size_t)ncls_, st_));
    CK(cudaMalloc(&d_scratch_, sizeof(double) * (size_t)G_ * RB_BLOCK * (size_t)natoms_max_));
    CK(cudaMalloc(&d_counter_, sizeof(unsigned long long)));
    if (!wait_or_hang(hang_s_, "initialisation", err)) return false;
    CK(cudaMemGetInfo(&free0_, &total0_));
    budget_ = (size_t)(mem_frac_ * (double)free0_);
    return true;
  }
  // Class upload and eviction.
  bool upload(int cls, const PackedClass &pc, std::string *err) override {
    if (cls < 0 || cls >= ncls_) { *err = "class slot out of range"; return false; }
    if (img_.count(cls)) { *err = "class already resident"; return false; }
    void *p = nullptr;
    CK(cudaMalloc(&p, pc.bytes()));
    img_[cls] = p;
    const RbClassView v = view_at(pc.rel, p);
    std::vector<double> back(pc.buf.size());
    RbClassView vback;
    // All on st_, then one synchronisation: the image and the view are in place before any later
    // kernel on st_
    CK(cudaMemcpyAsync(p, pc.buf.data(), pc.bytes(), cudaMemcpyHostToDevice, st_));
    CK(cudaMemcpyAsync(d_views_ + cls, &v, sizeof v, cudaMemcpyHostToDevice, st_));
    // Before the (blocking) read-backs
    if (!wait_or_hang(hang_s_, "class upload", err)) return false;
    CK(cudaMemcpyAsync(back.data(), p, pc.bytes(), cudaMemcpyDeviceToHost, st_));
    CK(cudaMemcpyAsync(&vback, d_views_ + cls, sizeof vback, cudaMemcpyDeviceToHost, st_));
    if (!wait_or_hang(hang_s_, "class read-back", err)) return false;
    if (memcmp(back.data(), pc.buf.data(), pc.bytes()) != 0) {
      *err = "class image read-back mismatch";
      return false;
    }
    if (memcmp(&vback, &v, sizeof v) != 0) { *err = "class view read-back mismatch"; return false; }
    return true;
  }
  bool evict(int cls, std::string *err) override {
    auto it = img_.find(cls);
    if (it == img_.end()) return true;
    CK(cudaMemsetAsync(d_views_ + cls, 0, sizeof(RbClassView), st_));
    if (!wait_or_hang(hang_s_, "eviction", err)) return false;
    void *p = it->second;
    img_.erase(it);
    CK(cudaFree(p));
    return true;
  }
  // Launch.
  bool launch(const std::vector<RbWorkItem> &items, uint64_t nchunks, std::vector<RbChunkOut> &out,
      LaunchStats *st, double expected_s, std::string *err) override {
    if (items.empty() || nchunks == 0) {
      out.clear();
      st->elapsed_s = 0;
      st->ticks = 0;
      return true;
    }
    for (const RbWorkItem &it : items)
      if (it.cls < 0 || it.cls >= ncls_ || !img_.count(it.cls)) {
        *err = "work item for a non-resident class " + std::to_string(it.cls);
        return false;
      }
    if (items.size() > items_cap_) {
      if (d_items_) { CK(cudaFree(d_items_)); d_items_ = nullptr; }
      items_cap_ = items.size() + items.size() / 2 + 16;
      CK(cudaMalloc(&d_items_, sizeof(RbWorkItem) * items_cap_));
    }
    if (nchunks > out_cap_) {
      if (d_out_) { CK(cudaFree(d_out_)); d_out_ = nullptr; }
      out_cap_ = (size_t)nchunks + (size_t)nchunks / 2 + 16;
      CK(cudaMalloc(&d_out_, sizeof(RbChunkOut) * out_cap_));
    }
    CK(cudaMemcpyAsync(d_items_, items.data(), sizeof(RbWorkItem) * items.size(),
      cudaMemcpyHostToDevice, st_));
    CK(cudaMemsetAsync(d_counter_, 0, sizeof(unsigned long long), st_));
    // NaN poison: unwritten chunks fail the count check
    CK(cudaMemsetAsync(d_out_, 0xff, sizeof(RbChunkOut) * (size_t)nchunks, st_));
    CK(cudaEventRecord(e0_, st_));
    CK(cudaGetLastError()); // a stale non-sticky error must not be attributed to this launch
    if (fp32_)
      rb_batch_kernel<float><<<G_, RB_BLOCK, 0, st_>>>(d_views_, d_items_, (int)items.size(),
        (unsigned long long)nchunks, d_scratch_, d_counter_, d_out_);
    else
      rb_batch_kernel<double><<<G_, RB_BLOCK, 0, st_>>>(d_views_, d_items_, (int)items.size(),
        (unsigned long long)nchunks, d_scratch_, d_counter_, d_out_);
    if (!check_launch(err)) return false;
    CK(cudaEventRecord(e1_, st_));
    // Watchdog: poll instead of blocking, so that a hung launch removes the device instead of the
    // whole run.
    // Calibrated launches may legitimately take long (high loop orders): the limit scales with the
    // expected time.
    const double limit = std::max(hang_s_, 20.0 * expected_s);
    if (!wait_or_hang(limit, "launch", err)) return false;
    out.resize(nchunks);
    CK(cudaMemcpyAsync(out.data(), d_out_, sizeof(RbChunkOut) * (size_t)nchunks,
      cudaMemcpyDeviceToHost, st_));
    if (!wait_or_hang(hang_s_, "read-back of the chunk results", err)) return false;
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0_, e1_));
    st->elapsed_s = 1e-3 * ms;
    st->ticks = 0;
    for (auto &c : out) st->ticks += c.ticks;
    return true;
  }
  // FP64/Philox test.
  bool fptest(std::vector<double> &fp, std::vector<uint32_t> &kat, uint64_t *signature,
      std::string *err) override {
    fp.assign(4 * RB_NFPTEST, 0);
    kat.assign(4 * RB_NKAT, 0);
    DevPtr<double> dfp;
    DevPtr<uint32_t> dk;
    DevPtr<unsigned long long> ds;
    if (!dev_alloc(dfp, fp.size(), err) || !dev_alloc(dk, kat.size(), err)
        || !dev_alloc(ds, 1, err))
      return false;
    CK(cudaGetLastError());
    rb_fptest_kernel<<<1, 32, 0, st_>>>(dfp.get(), dk.get(), ds.get());
    if (!check_launch(err)) return false;
    // Wait (with the watchdog) BEFORE any device-to-host copy: a copy to pageable memory blocks the
    // host thread until the kernel has finished, i.e. forever if the kernel hangs
    if (!wait_or_hang(hang_s_, "FP64/Philox test kernel", err)) return false;
    unsigned long long sig = 0;
    CK(cudaMemcpyAsync(fp.data(), dfp.get(), sizeof(double) * fp.size(), cudaMemcpyDeviceToHost,
      st_));
    CK(cudaMemcpyAsync(kat.data(), dk.get(), sizeof(uint32_t) * kat.size(), cudaMemcpyDeviceToHost,
      st_));
    CK(cudaMemcpyAsync(&sig, ds.get(), sizeof sig, cudaMemcpyDeviceToHost, st_));
    if (!wait_or_hang(hang_s_, "read-back of the test kernel", err)) return false;
    *signature = (uint64_t)sig;
    return true;
  }
  // Per-sample debug kernel.
  bool debug_samples(int cls, const RbStreamId &id, uint64_t idx0, int n,
      std::vector<RbSampleDbg> &out, std::string *err) override {
    out.assign(n, RbSampleDbg());
    DevPtr<double> scr;
    DevPtr<RbSampleDbg> dout;
    if (!dev_alloc(scr, (size_t)n * natoms_max_, err) || !dev_alloc(dout, (size_t)n, err))
      return false;
    CK(cudaGetLastError());
    rb_dbg_kernel<<<(n + 127) / 128, 128, 0, st_>>>(d_views_, cls, (unsigned long long)id.key,
      id.stream, (unsigned long long)idx0, n, scr.get(), dout.get());
    if (!check_launch(err)) return false;
    // Before the (blocking) read-back
    if (!wait_or_hang(hang_s_, "per-sample debug kernel", err)) return false;
    CK(cudaMemcpyAsync(out.data(), dout.get(), sizeof(RbSampleDbg) * (size_t)n,
      cudaMemcpyDeviceToHost, st_));
    if (!wait_or_hang(hang_s_, "read-back of the debug kernel", err)) return false;
    return true;
  }
  // Device information in JSON.
  void info_json(JsonW &J) const override {
    J.begin_obj();
    J.kv("type", "cuda");
    J.kvi("ordinal", ord_); J.kv("name", std::string(prop_.name));
    J.kv("compute_capability", std::to_string(prop_.major) + "." + std::to_string(prop_.minor));
    J.kvi("sms", prop_.multiProcessorCount); J.kvi("clock_khz", clock_khz_);
    J.kvi("single_to_double_perf_ratio", fp64_ratio_);
    J.kv("global_mem_bytes", (double)prop_.totalGlobalMem);
    J.kv("free_mem_at_open", (double)free0_);
    J.kv("class_mem_budget", (double)budget_);
    J.kvi("l2_bytes", prop_.l2CacheSize); J.kvi("ecc", prop_.ECCEnabled);
    J.kv("kernel_precision", fp32_ ? "fp32" : "fp64");
    J.kvi("kernel_regs", fa_.numRegs); J.kvi("kernel_local_bytes", (long long)fa_.localSizeBytes);
    J.kvi("kernel_static_shared_bytes", (long long)fa_.sharedSizeBytes);
    J.kvi("kernel_max_threads", fa_.maxThreadsPerBlock);
    J.kvi("kernel_binary_version", fa_.binaryVersion); J.kvi("kernel_ptx_version", fa_.ptxVersion);
    J.kvi("min_blocks_per_sm_bound", RB_MINBLOCKS);
    J.kvi("fmad", RB_FMAD);
    J.kvi("blocks_per_sm", bpsm_); J.kvi("grid", G_); J.kvi("threads_per_block", RB_BLOCK);
    J.kv("stack_limit_bytes", (double)stack_);
    J.kv("hang_timeout_s", hang_s_);
    J.kv("scratch_bytes", (double)sizeof(double) * (double)G_ * RB_BLOCK * (double)natoms_max_);
    J.end_obj();
  }
};
} // namespace

CudaProbe cuda_probe() {
  CudaProbe p;
  p.compiled = true;
  if (cudaDriverGetVersion(&p.driver) != cudaSuccess) p.driver = 0;
  if (cudaRuntimeGetVersion(&p.runtime) != cudaSuccess) p.runtime = 0;
  const cudaError_t e = cudaGetDeviceCount(&p.count);
  if (e != cudaSuccess) {
    p.err = std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
    p.count = 0;
  }
  cudaGetLastError();
  return p;
}

std::unique_ptr<Device> make_cuda_device(int ordinal, const BackendOptions &o) {
  const double hang = o.hang_s > 0 ? o.hang_s : std::max(300.0, 50.0 * o.launch_s);
  return std::unique_ptr<Device>(new CudaDevice(ordinal, o.mem_frac, hang, o.fp32));
}
