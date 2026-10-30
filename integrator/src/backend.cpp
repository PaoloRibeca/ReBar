/*
    backend.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    CPU backend, class images, and the multi-device scheduler shared by the
    CUDA and emulated devices (see backend.h).

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

#include "backend.h"
#include <omp.h>
#include <signal.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

std::atomic<bool> g_rb_stop{false};
std::atomic<bool> g_rb_hung{false};
std::atomic<long long> g_rb_test_stop_units{-1}, g_rb_test_kill_units{-1};
static std::atomic<long long> g_test_units_done{0};
// TEST hooks: count evaluated units (or tasks) and stop / kill the process when a limit is reached
static void test_hooks_count(long long k) {
  const long long d = (g_test_units_done += k);
  const long long ks = g_rb_test_kill_units.load(), ss = g_rb_test_stop_units.load();
  if (ks >= 0 && d >= ks) {
    fprintf(stderr, "TEST: SIGKILL after %lld units\n", d);
    fflush(stderr);
    raise(SIGKILL);
  }
  if (ss >= 0 && d >= ss && !g_rb_stop) {
    fprintf(stderr, "TEST: stop requested after %lld units\n", d);
    g_rb_stop = true;
  }
}

double rb_now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double thread_cpu_s() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

// CPU.
// Chunk accumulators of the sample range [idx0, idx0 + n): OpenMP threads = persistent blocks of
// rb_block_run
static void cpu_chunks(const ClassData &cd, const RbStreamId &id, uint64_t idx0, uint64_t n,
    std::vector<RbChunkOut> &out, double *core_s, bool fp32) {
  const RbClassView v = cd.view();
  const RbWorkItem it{0, id.stream, id.key, idx0, n, 0};
  const uint64_t nch = rb_nchunks(n);
  out.assign(nch, RbChunkOut());
  unsigned long long counter = 0;
  double cpu = 0;
#pragma omp parallel reduction(+ : cpu)
  {
    const double t0 = thread_cpu_s();
    std::vector<double> lp(std::max(1, cd.natoms));
    // One OpenMP thread = one persistent block; lanes run one after the other, so they share one
    // compact scratch
    RbHostBlock b(&counter, lp.data(), 0, 1, false);
    if (fp32) rb_block_run<RbExec, float>(b.x, &v, &it, 1, nch, out.data());
    else rb_block_run<RbExec, double>(b.x, &v, &it, 1, nch, out.data());
    cpu += thread_cpu_s() - t0;
  }
  if (core_s) *core_s = cpu;
}
// First level of the canonical merge: chunks in order, from zero
static RbAcc merge_chunks(const RbChunkOut *c, uint64_t n) {
  RbAcc a;
  rb_acc_zero(&a);
  for (uint64_t k = 0; k < n; ++k) rb_acc_merge(&a, &c[k].acc);
  return a;
}
RbAcc cpu_run_range(const ClassData &cd, const RbStreamId &id, uint64_t idx0, uint64_t n,
    double *core_s, bool fp32, const RbAcc *init) {
  RbAcc r;
  if (init) r = *init; else rb_acc_zero(&r);
  if (core_s) *core_s = 0;
  if (n == 0) return r;
  std::vector<RbChunkOut> out;
  cpu_chunks(cd, id, idx0, n, out, core_s, fp32);
  // Canonical two-level merge (rbar/batch.h): chunks in order within units of RB_UNIT_CHUNKS, units
  // in order
  for (uint64_t u0 = 0; u0 < out.size(); u0 += RB_UNIT_CHUNKS) {
    const RbAcc ua = merge_chunks(out.data() + u0,
      std::min<uint64_t>(RB_UNIT_CHUNKS, out.size() - u0));
    rb_acc_merge(&r, &ua);
  }
  return r;
}
// One unit (n <= RB_UNIT): the unit-level accumulator itself (host fallback of the multi-device
// backend)
static RbAcc cpu_run_unit(const ClassData &cd, const RbStreamId &id, uint64_t idx0, uint64_t n,
    double *core_s, bool fp32) {
  std::vector<RbChunkOut> out;
  cpu_chunks(cd, id, idx0, n, out, core_s, fp32);
  return merge_chunks(out.data(), out.size());
}

namespace {
class CpuBackend : public Backend {
  std::vector<const ClassData *> cls_;
  std::vector<RbStreamId> ids_;
  bool fp32_;
  double samples_ = 0, core_s_ = 0, wall_s_ = 0;
  long long tasks_ = 0;
  // Backend interface.
 public:
  explicit CpuBackend(bool fp32) : fp32_(fp32) {}
  std::string name() const override { return "cpu"; }
  bool prepare(const std::vector<const ClassData *> &cls, const std::vector<RbStreamId> &ids,
      std::string *) override {
    cls_ = cls; ids_ = ids;
    return true;
  }
  void run(const std::vector<SampleTask> &tasks, double deadline,
      std::vector<SampleOut> &out) override {
    out.assign(tasks.size(), SampleOut());
    // Tasks not run: done = 0, acc = init
    for (size_t i = 0; i < tasks.size(); ++i) out[i].acc = tasks[i].init;
    const double w0 = rb_now_s();
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (g_rb_stop || rb_now_s() > deadline) break;
      const SampleTask &t = tasks[i];
      double cpu = 0;
      out[i].acc = cpu_run_range(*cls_[t.cls], ids_[t.cls], t.idx0, t.n, &cpu, fp32_, &t.init);
      out[i].cost_s = cpu;
      out[i].done = t.n;
      samples_ += (double)t.n; core_s_ += cpu; ++tasks_;
      test_hooks_count(1);
    }
    wall_s_ += rb_now_s() - w0;
  }
  double capacity() const override { return omp_get_max_threads(); }
  const char *cost_unit() const override { return "core-s"; }
  void info_json(JsonW &J) const override {
    J.begin_obj();
    J.kv("kind", "cpu"); J.kvi("threads", omp_get_max_threads());
    J.kv("precision", fp32_ ? "fp32-eval/fp64-accum" : "fp64");
    J.kvi("lanes_per_block", RB_BLOCK); J.kvi("samples_per_lane", RB_SPT);
    J.kvi("chunk", (long long)RB_CHUNK);
    J.kvi("unit_chunks", RB_UNIT_CHUNKS);
    J.kv("samples", samples_); J.kv("core_s", core_s_); J.kv("wall_s", wall_s_);
    J.kvi("tasks", tasks_);
    J.kv("samples_per_wall_s", wall_s_ > 0 ? samples_ / wall_s_ : 0.0);
    J.end_obj();
  }
};
} // namespace

std::unique_ptr<Backend> make_cpu_backend(bool fp32) {
  return std::unique_ptr<Backend>(new CpuBackend(fp32));
}

// Class images.
PackedClass pack_class(const ClassData &cd) {
  PackedClass pc;
  struct Seg { const void *src; size_t bytes, off; };
  std::vector<Seg> segs;
  size_t off = 0;
  auto add = [&](const void *src, size_t bytes) -> size_t {
    const size_t o = off;
    segs.push_back({src, bytes, o});
    off += (bytes + 15) & ~(size_t)15;
    return o;
  };
  RbClassView r = cd.view();
#define RB_SEG(field, vec) r.field = (decltype(r.field))(uintptr_t)add(cd.vec.data(), \
    cd.vec.size() * sizeof(cd.vec[0]))
  RB_SEG(omega, omega); RB_SEG(Jt, Jt);
  RB_SEG(atom_off, atom_off); RB_SEG(atom_nv, atom_nv);
  RB_SEG(ae_u, ae_u); RB_SEG(ae_v, ae_v); RB_SEG(atom_loc, atom_loc);
  RB_SEG(grp_off, grp_off); RB_SEG(grp_foff, grp_foff); RB_SEG(cf_atom, cf_atom);
  RB_SEG(cf_p2, cf_p2);
  RB_SEG(comp_coeff, comp_coeff); RB_SEG(comp_d0, comp_d0); RB_SEG(comp_d1, comp_d1);
#undef RB_SEG
  pc.buf.assign((off + sizeof(double) - 1) / sizeof(double) + 1, 0.0);
  for (auto &s : segs) if (s.bytes) memcpy((char *)pc.buf.data() + s.off, s.src, s.bytes);
  pc.rel = r;
  return pc;
}

size_t packed_bytes(const ClassData &cd) {
  size_t off = 0;
  auto add = [&](size_t bytes) { off += (bytes + 15) & ~(size_t)15; };
#define RB_SZ(vec) add(cd.vec.size() * sizeof(cd.vec[0]))
  RB_SZ(omega); RB_SZ(Jt); RB_SZ(atom_off); RB_SZ(atom_nv); RB_SZ(ae_u); RB_SZ(ae_v);
  RB_SZ(atom_loc);
  RB_SZ(grp_off); RB_SZ(grp_foff); RB_SZ(cf_atom); RB_SZ(cf_p2); RB_SZ(comp_coeff); RB_SZ(comp_d0);
  RB_SZ(comp_d1);
#undef RB_SZ
  return ((off + sizeof(double) - 1) / sizeof(double) + 1) * sizeof(double);
}

RbClassView view_at(const RbClassView &rel, const void *base) {
  RbClassView v = rel;
  const char *b = (const char *)base;
#define RB_FIX(field) v.field = (decltype(v.field))(b + (uintptr_t)rel.field)
  RB_FIX(omega); RB_FIX(Jt); RB_FIX(atom_off); RB_FIX(atom_nv); RB_FIX(ae_u); RB_FIX(ae_v);
  RB_FIX(atom_loc);
  RB_FIX(grp_off); RB_FIX(grp_foff); RB_FIX(cf_atom); RB_FIX(cf_p2); RB_FIX(comp_coeff);
  RB_FIX(comp_d0); RB_FIX(comp_d1);
#undef RB_FIX
  return v;
}

// Multi-device backend.
namespace {
static int ulp_dist(double a, double b) {
  if (a == b) return 0;
  if (!std::isfinite(a) || !std::isfinite(b)) return 1 << 30;
  int64_t ia, ib;
  memcpy(&ia, &a, 8); memcpy(&ib, &b, 8);
  if (ia < 0) ia = INT64_MIN - ia;
  if (ib < 0) ib = INT64_MIN - ib;
  const int64_t d = ia > ib ? ia - ib : ib - ia;
  return d > (1 << 30) ? (1 << 30) : (int)d;
}

struct Unit { int task, cls; uint64_t idx0, n; };

struct DevState {
  std::unique_ptr<Device> d;
  bool alive = true;
  std::string why_dead;
  std::string selftest; // JSON fragment
  bool selftest_ok = true;
  std::map<int, unsigned long long> resident; // class -> last-use stamp
  std::map<int, size_t> res_bytes;
  size_t used = 0;
  unsigned long long stamp = 0;
  long long launches = 0, chunks = 0, uploads = 0, evictions = 0, round_launches = 0;
  double samples = 0, kernel_s = 0, min_s = 1e300, max_s = 0, upload_bytes = 0, round_samples = 0,
      round_kernel_s = 0;
  double fill_sum = 0, fill_min = 1; // launch fill factor = chunks / (waves x grid)
  // Device-seconds (wave-normalised) and samples (for the device-wide mean cost)
  double cost_s = 0, cost_n = 0;
  double max_chunk_s = 0; // longest chunk seen (s): lower bound of a launch's duration
  std::map<int, std::pair<double, double>> cost; // class -> (device-s, samples)
};

struct LaunchRec {
  int dev;
  double t, elapsed_s, samples, fill, expected_s;
  long long chunks, items;
};

class MultiBackend : public Backend {
  BackendOptions o_;
  std::string kind_;
  std::vector<DevState> dev_;
  std::vector<const ClassData *> cls_;
  std::vector<RbStreamId> ids_;
  std::vector<size_t> img_bytes_;
  std::vector<double> ccost_s_, ccost_n_; // per class, over all devices
  std::vector<LaunchRec> log_;
  double t0_ = 0;
  long long fallback_units_ = 0;
  double fallback_samples_ = 0, fallback_core_s_ = 0;
  bool degraded_ = false;
  std::string probe_json_;
  mutable std::mutex mu_;
  // Backend interface.
 public:
  MultiBackend(const BackendOptions &o, const std::string &kind,
      std::vector<std::unique_ptr<Device>> devs, const std::string &probe)
      : o_(o), kind_(kind), probe_json_(probe) {
    for (auto &d : devs) { DevState s; s.d = std::move(d); dev_.push_back(std::move(s)); }
    t0_ = rb_now_s();
  }
  std::string name() const override { return degraded_ ? kind_ + "->cpu" : kind_; }
  double capacity() const override {
    int n = 0;
    for (auto &d : dev_) n += d.alive;
    return n > 0 ? (double)n : (double)omp_get_max_threads();
  }
  const char *cost_unit() const override { return "device-s"; }
  // Residency.
  bool ensure_resident(DevState &D, const std::vector<int> &need, std::string *err) {
    ++D.stamp;
    std::set<int> needset(need.begin(), need.end());
    for (int c : need) {
      auto it = D.resident.find(c);
      if (it != D.resident.end()) { it->second = D.stamp; continue; }
      const size_t b = img_bytes_[c];
      while (D.used + b > D.d->mem_budget()) {
        int victim = -1;
        unsigned long long best = ~0ull;
        for (auto &r : D.resident)
          if (!needset.count(r.first) && r.second < best) { best = r.second; victim = r.first; }
        if (victim < 0) {
          *err = "class images of one launch exceed the device memory budget";
          return false;
        }
        if (!D.d->evict(victim, err)) return false;
        D.used -= D.res_bytes[victim];
        D.res_bytes.erase(victim);
        D.resident.erase(victim);
        ++D.evictions;
      }
      PackedClass pc = pack_class(*cls_[c]);
      if (!D.d->upload(c, pc, err)) return false;
      D.resident[c] = D.stamp;
      D.res_bytes[c] = pc.bytes();
      D.used += pc.bytes();
      ++D.uploads;
      D.upload_bytes += (double)pc.bytes();
    }
    return true;
  }
  // Self-test.
  void selftest(DevState &D) {
    JsonW J;
    J.begin_obj();
    bool ok = true;
    std::string err;
    // (1) FP64 elementary functions, Philox and the build signature on the device vs the host
    std::vector<double> fp;
    std::vector<uint32_t> kat;
    uint64_t sig = 0;
    if (!D.d->fptest(fp, kat, &sig, &err)) {
      J.kv("error", err); ok = false;
    } else {
      int kbad = 0;
      for (int i = 0; i < 4 * RB_NKAT; ++i) kbad += kat[i] != rb_kat_expected[i];
      double hfp[4 * RB_NFPTEST];
      for (int i = 0; i < RB_NFPTEST; ++i) rb_fptest(i, hfp);
      int uexp = 0, ulog = 0, usqrt = 0, ufma = 0;
      for (int i = 0; i < RB_NFPTEST; ++i) {
        uexp = std::max(uexp, ulp_dist(fp[4 * i], hfp[4 * i]));
        ulog = std::max(ulog, ulp_dist(fp[4 * i + 1], hfp[4 * i + 1]));
        usqrt = std::max(usqrt, ulp_dist(fp[4 * i + 2], hfp[4 * i + 2]));
        ufma = std::max(ufma, ulp_dist(fp[4 * i + 3], hfp[4 * i + 3]));
      }
      const uint64_t hsig = rb_build_signature();
      char b1[32], b2[32];
      snprintf(b1, sizeof b1, "%016llx", (unsigned long long)sig);
      snprintf(b2, sizeof b2, "%016llx", (unsigned long long)hsig);
      J.kvi("philox_kat_mismatches", kbad);
      J.kvi("exp_max_ulp", uexp); J.kvi("log_max_ulp", ulog); J.kvi("sqrt_max_ulp", usqrt);
      J.kvi("fma_max_ulp", ufma);
      J.kv("build_signature_device", std::string(b1));
      J.kv("build_signature_host", std::string(b2));
      J.kvb("build_signature_match", sig == hsig);
      if (sig != hsig) {
        err = "build signature mismatch (compile-time limits / struct layouts differ between host "
            "and device code: pass -D options through DEFS, not NVCC_EXTRA)";
        J.kv("error", err);
        ok = false;
      }
      if (kbad || uexp > 2 || ulog > 2 || usqrt > 0 || ufma > 0) ok = false;
    }
    // (2) sample by sample on selected classes, (3) one chunk through the batch kernel vs the host
    std::vector<int> pick;
    {
      int best_atoms = -1, best_groups = -1, ia = -1, ig = -1;
      for (size_t c = 0; c < cls_.size(); ++c) {
        if (cls_[c]->natoms > best_atoms) { best_atoms = cls_[c]->natoms; ia = (int)c; }
        if (cls_[c]->ngroup > best_groups) { best_groups = cls_[c]->ngroup; ig = (int)c; }
      }
      for (int c : {0, ia, ig, (int)cls_.size() / 2, (int)cls_.size() - 1})
        if (c >= 0 && c < (int)cls_.size() && std::find(pick.begin(), pick.end(), c) == pick.end()
            && (int)pick.size() < o_.selftest_classes)
          pick.push_back(c);
    }
    // FP32 evaluation: eps_32 = 6e-8 per sample
    const double chunk_tol = o_.fp32 ? 1e-4 : o_.selftest_tol;
    J.kv("chunk_precision", o_.fp32 ? "fp32" : "fp64");
    J.key("classes"); J.begin_arr();
    for (int c : pick) {
      J.begin_obj();
      J.kvi("class", c); J.kv("nickel", cls_[c]->G.name);
      if (ok && !ensure_resident(D, {c}, &err)) {
        J.kv("error", err);
        ok = false;
        J.end_obj();
        continue;
      }
      const RbStreamId id = ids_[c];
      const int n = 4096;
      std::vector<RbSampleDbg> dv;
      if (!ok || !D.d->debug_samples(c, id, 0, n, dv, &err)) {
        if (ok) J.kv("error", err);
        ok = false; J.end_obj(); continue;
      }
      const RbClassView hv = cls_[c]->view();
      std::vector<double> lp(std::max(1, cls_[c]->natoms));
      int hash_bad = 0, nonfinite = 0;
      double wdev = 0, lvsdev = 0;
      for (int i = 0; i < n; ++i) {
        RbSampleDbg h;
        rb_sample_dbg(hv, id.key, id.stream, (uint64_t)i, lp.data(), 1, &h);
        hash_bad += h.hash != dv[i].hash;
        if (!std::isfinite(dv[i].w) || !std::isfinite(h.w)) {
          nonfinite += std::isfinite(dv[i].w) != std::isfinite(h.w);
          continue;
        }
        wdev = std::max(wdev, std::fabs(dv[i].w - h.w) / std::max(h.sabs, 1e-300));
        lvsdev = std::max(lvsdev,
          std::fabs(dv[i].lvs - h.lvs) / std::max(std::fabs(h.lvs), 1e-300));
      }
      J.kvi("samples", n); J.kvi("sector_mismatches", hash_bad);
      J.kvi("nonfinite_mismatches", nonfinite);
      J.kv("max_abs_dw_over_sum_abs_terms", wdev); J.kv("max_rel_dlogv", lvsdev);
      if (hash_bad || nonfinite || !(wdev <= o_.selftest_tol)) ok = false;
      // One full chunk through the batch kernel (production precision), compared with the host
      // canonical reduction
      std::vector<RbWorkItem> items{RbWorkItem{c, id.stream, id.key, 0, RB_CHUNK, 0}};
      std::vector<RbChunkOut> out;
      LaunchStats st;
      if (!D.d->launch(items, 1, out, &st, 0.0, &err)) {
        J.kv("error", err);
        ok = false;
        J.end_obj();
        continue;
      }
      RbAcc ha;
      {
        unsigned long long counter = 0;
        std::vector<RbChunkOut> ho(1);
        RbHostBlock b(&counter, lp.data(), 0, 1, false);
        const RbWorkItem hit{0, id.stream, id.key, 0, RB_CHUNK, 0}; // single host view: slot 0
        if (o_.fp32) rb_block_run<RbExec, float>(b.x, &hv, &hit, 1, 1, ho.data());
        else rb_block_run<RbExec, double>(b.x, &hv, &hit, 1, 1, ho.data());
        ha = ho[0].acc;
      }
      const RbAcc &da = out[0].acc;
      const double bound = chunk_tol * std::sqrt(ha.n * ha.sabs2);
      const double ds = std::fabs((da.s + da.sc) - (ha.s + ha.sc)),
          dq = std::fabs((da.q + da.qc) - (ha.q + ha.qc));
      const bool bitwise = memcmp(&da, &ha, sizeof(RbAcc)) == 0;
      J.kv("chunk_n_dev", da.n); J.kv("chunk_n_host", ha.n);
      J.kv("chunk_abs_dsum", ds); J.kv("chunk_bound", bound);
      J.kv("chunk_rel_dsumsq", dq / std::max(ha.q, 1e-300));
      J.kvb("chunk_bitwise_equal", bitwise);
      J.kv("chunk_ticks", (double)out[0].ticks);
      J.kv("chunk_s", (double)out[0].ticks / D.d->tick_hz());
      if (da.n != ha.n || da.nbad != ha.nbad || !(ds <= bound)) ok = false;
      J.end_obj();
    }
    J.end_arr();
    J.kvb("ok", ok);
    J.end_obj();
    D.selftest = J.s;
    D.selftest_ok = ok;
  }
  // Preparation.
  bool prepare(const std::vector<const ClassData *> &cls, const std::vector<RbStreamId> &ids,
      std::string *err) override {
    cls_ = cls; ids_ = ids;
    int natoms_max = 1;
    img_bytes_.resize(cls.size());
    for (size_t c = 0; c < cls.size(); ++c) {
      natoms_max = std::max(natoms_max, cls[c]->natoms);
      img_bytes_[c] = packed_bytes(*cls[c]);
    }
    ccost_s_.assign(cls.size(), 0); ccost_n_.assign(cls.size(), 0);
    int nok = 0;
    for (auto &D : dev_) {
      std::string e;
      if (g_rb_stop) { D.alive = false; D.why_dead = "interrupted"; continue; }
      if (!D.d->bind(&e) || !D.d->open((int)cls.size(), natoms_max, &e)) {
        D.alive = false; D.why_dead = "open: " + e;
        fprintf(stderr, "[%s] %s: %s\n", kind_.c_str(), D.d->label().c_str(), D.why_dead.c_str());
        continue;
      }
      if (o_.selftest) {
        selftest(D);
        if (!D.selftest_ok) { D.alive = false; D.why_dead = "start-up self-test failed"; }
        if (D.selftest_ok)
          fprintf(stderr, "[%s] %s: start-up self-test passed\n", kind_.c_str(),
            D.d->label().c_str());
        else
          fprintf(stderr, "[%s] %s: start-up self-test FAILED, device not used: %s\n",
            kind_.c_str(), D.d->label().c_str(), D.selftest.c_str());
      }
      D.d->arm();
      nok += D.alive;
    }
    if (nok == 0) { *err = "no usable " + kind_ + " device (see backend_info)"; return false; }
    return true;
  }
  // Cost estimates.
  // Device-seconds per sample at FULL occupancy (launch time x fill factor, apportioned by block
  // cycles)
  double est_cost(const DevState &D, int c) const {
    auto it = D.cost.find(c);
    if (it != D.cost.end() && it->second.second > 0) return it->second.first / it->second.second;
    if (D.cost_n > 0) return D.cost_s / D.cost_n;
    if (ccost_n_[c] > 0) return ccost_s_[c] / ccost_n_[c];
    return 0;
  }
  // Expected duration of a launch of nch chunks whose full-occupancy cost is est: a partial last
  // wave still costs a whole wave, and no launch is shorter than its longest chunk
  static double wall_est(double est, uint64_t nch, uint64_t G, double max_chunk_s) {
    if (nch == 0) return 0;
    const uint64_t waves = (nch + G - 1) / G;
    const double fill = (double)nch / (double)(waves * G);
    return std::max(est / fill, max_chunk_s);
  }
  // Run.
  void run(const std::vector<SampleTask> &tasks, double deadline,
      std::vector<SampleOut> &out) override {
    out.assign(tasks.size(), SampleOut());
    // Units are merged into it in order
    for (size_t t = 0; t < tasks.size(); ++t) out[t].acc = tasks[t].init;
    std::vector<Unit> units;
    std::vector<std::pair<size_t, size_t>> trange(tasks.size());
    const uint64_t usz = RB_UNIT; // canonical unit (the merge order depends on it)
    for (size_t t = 0; t < tasks.size(); ++t) {
      trange[t].first = units.size();
      for (uint64_t a = 0; a < tasks[t].n; a += usz)
        units.push_back(Unit{(int)t, tasks[t].cls, tasks[t].idx0 + a,
          std::min<uint64_t>(usz, tasks[t].n - a)});
      trange[t].second = units.size();
    }
    std::vector<RbAcc> uacc(units.size());
    std::vector<double> ucost(units.size(), 0);
    std::vector<char> udone(units.size(), 0);
    std::deque<size_t> q;
    for (size_t u = 0; u < units.size(); ++u) q.push_back(u);
    bool stop = false;
    for (auto &D : dev_) { D.round_launches = 0; D.round_samples = 0; D.round_kernel_s = 0; }
    auto worker = [&](DevState &D, int di) {
      {
        std::string e;
        if (!D.d->bind(&e)) {
          std::lock_guard<std::mutex> lk(mu_);
          D.alive = false; D.why_dead = "bind: " + e;
          fprintf(stderr, "[%s] %s FAILED, removed from the pool: %s\n", kind_.c_str(),
            D.d->label().c_str(), D.why_dead.c_str());
          return;
        }
      }
      const uint64_t G = (uint64_t)std::max(1, D.d->blocks());
      for (;;) {
        std::vector<size_t> batch;
        uint64_t nch = 0;
        double est = 0, expected = 0;
        {
          std::lock_guard<std::mutex> lk(mu_);
          if (stop || g_rb_stop || q.empty() || !D.alive) return;
          const bool calibrated = D.cost_n > 0;
          const double remaining = deadline < 1e299 ? deadline - rb_now_s() : 1e300;
          if (remaining <= 0) { stop = true; return; }
          // Launch sizing.  Calibrated: up to --launch-s of work, but never fewer than two full
          // waves of the grid while work remains (otherwise, when one chunk takes longer than
          // --launch-s, the measured cost per sample inflates and launches shrink to a few chunks),
          // and never past the deadline.  First launch of a device: one wave (calibration).
          const uint64_t min_nch = 2 * G;
          // Classes of the batch and their image bytes: they must fit the device together
          std::set<int> bcls;
          size_t bbytes = 0;
          while (!q.empty()) {
            const Unit &U = units[q.front()];
            const uint64_t c = rb_nchunks(U.n);
            const double e = (double)U.n * est_cost(D, U.cls);
            const bool newc = !bcls.count(U.cls);
            const double wnew = calibrated ? wall_est(est + e, nch + c, G, D.max_chunk_s) : 0;
            if (!batch.empty()) {
              if ((long long)(nch + c) > o_.max_launch_chunks) break;
              if (calibrated) {
                if (wnew > remaining) break;
                if (wnew > o_.launch_s && nch >= min_nch) break;
              } else if (nch >= G) break;
              if (newc && bbytes + img_bytes_[U.cls] > D.d->mem_budget()) break;
            } else if (calibrated && wnew > remaining) {
              stop = true; // not even one unit (one wave) fits before the deadline
              return;
            }
            if (newc) { bcls.insert(U.cls); bbytes += img_bytes_[U.cls]; }
            batch.push_back(q.front()); q.pop_front();
            nch += c; est += e;
          }
          expected = calibrated ? wall_est(est, nch, G, D.max_chunk_s) : 0;
        }
        // Residency, work items
        std::vector<int> need;
        for (size_t u : batch) need.push_back(units[u].cls);
        std::sort(need.begin(), need.end());
        need.erase(std::unique(need.begin(), need.end()), need.end());
        std::vector<RbWorkItem> items;
        uint64_t c0 = 0;
        for (size_t u : batch) {
          const Unit &U = units[u];
          items.push_back(RbWorkItem{U.cls, ids_[U.cls].stream, ids_[U.cls].key, U.idx0, U.n, c0});
          c0 += rb_nchunks(U.n);
        }
        std::string err;
        std::vector<RbChunkOut> co;
        LaunchStats st;
        const double tl = rb_now_s();
        bool ok = ensure_resident(D, need, &err)
            && D.d->launch(items, nch, co, &st, expected, &err);
        // Merge chunks into units (chunk order) and verify that every sample was accounted for
        std::vector<RbAcc> ua(batch.size());
        std::vector<unsigned long long> utk(batch.size(), 0);
        unsigned long long tk_all = 0, tk_max = 0;
        if (ok) {
          if (co.size() != nch) {
            ok = false;
            err = "launch returned a wrong number of chunk results";
          }
          size_t k = 0;
          for (size_t i = 0; ok && i < batch.size(); ++i) {
            rb_acc_zero(&ua[i]);
            const uint64_t nc = rb_nchunks(units[batch[i]].n);
            for (uint64_t j = 0; j < nc; ++j, ++k) {
              rb_acc_merge(&ua[i], &co[k].acc);
              utk[i] += co[k].ticks;
              tk_max = std::max(tk_max, co[k].ticks);
            }
            tk_all += utk[i];
            if (!(ua[i].n + ua[i].nbad == (double)units[batch[i]].n)) {
              ok = false;
              err = "sample count check failed (chunks lost, corrupted, or drawn from an invalid "
                  "class view): unit of class " + std::to_string(units[batch[i]].cls);
            }
          }
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (!ok) {
          // Requeue, order preserved
          for (auto it = batch.rbegin(); it != batch.rend(); ++it) q.push_front(*it);
          D.alive = false;
          D.why_dead = err;
          fprintf(stderr, "[%s] %s FAILED, removed from the pool (%zu units requeued): %s\n",
            kind_.c_str(), D.d->label().c_str(), batch.size(), err.c_str());
          return;
        }
        double ns = 0;
        for (size_t u : batch) ns += (double)units[u].n;
        const uint64_t waves = (nch + G - 1) / G;
        const double fill = (double)nch / (double)(waves * G);
        const double dev_s = st.elapsed_s * fill; // full-occupancy device time
        for (size_t i = 0; i < batch.size(); ++i) {
          const size_t u = batch[i];
          const double share = tk_all > 0 ? (double)utk[i] / (double)tk_all
              : (double)units[u].n / ns;
          uacc[u] = ua[i];
          ucost[u] = dev_s * share;
          udone[u] = 1;
          const int c = units[u].cls;
          D.cost[c].first += ucost[u]; D.cost[c].second += (double)units[u].n;
          ccost_s_[c] += ucost[u]; ccost_n_[c] += (double)units[u].n;
        }
        D.cost_s += dev_s; D.cost_n += ns;
        D.max_chunk_s = std::max(D.max_chunk_s, (double)tk_max / D.d->tick_hz());
        ++D.launches; ++D.round_launches;
        D.chunks += (long long)nch;
        D.samples += ns; D.round_samples += ns;
        D.kernel_s += st.elapsed_s; D.round_kernel_s += st.elapsed_s;
        D.fill_sum += fill; D.fill_min = std::min(D.fill_min, fill);
        D.min_s = std::min(D.min_s, st.elapsed_s); D.max_s = std::max(D.max_s, st.elapsed_s);
        if (log_.size() < 100000)
          log_.push_back(LaunchRec{di, tl - t0_, st.elapsed_s, ns, fill, expected, (long long)nch,
            (long long)items.size()});
        test_hooks_count((long long)batch.size());
      }
    };
    {
      std::vector<std::thread> th;
      for (size_t d = 0; d < dev_.size(); ++d)
        if (dev_[d].alive) th.emplace_back(worker, std::ref(dev_[d]), (int)d);
      for (auto &t : th) t.join();
    }
    bool any_alive = false;
    for (auto &D : dev_) any_alive = any_alive || D.alive;
    // Units not evaluated: (a) requeued after a device failure while the others had stopped; (b)
    // everything left when no device is alive.  Holes before an evaluated unit are filled (on the
    // host), so that every task's evaluated samples form a prefix; the rest is run on the host only
    // if no device is left.  After a stop request (SIGTERM: a scheduler's grace period is short)
    // nothing is run on the host: the units after a hole are dropped (the merge below stops at the
    // first hole) and are recomputed by the resumed run.
    for (size_t t = 0; t < tasks.size(); ++t) {
      size_t last = trange[t].first;
      bool anyd = false;
      for (size_t u = trange[t].first; u < trange[t].second; ++u)
        if (udone[u]) { last = u; anyd = true; }
      // Mandatory: holes before an evaluated unit
      const size_t holes_end = anyd ? last + 1 : trange[t].first;
      for (size_t u = trange[t].first; u < trange[t].second; ++u) {
        if (udone[u]) continue;
        if (g_rb_stop) break;
        // Optional: only without devices
        if (u >= holes_end && (any_alive || g_rb_stop || rb_now_s() >= deadline)) break;
        double cpu = 0;
        const Unit &U = units[u];
        uacc[u] = cpu_run_unit(*cls_[U.cls], ids_[U.cls], U.idx0, U.n, &cpu, o_.fp32);
        ucost[u] = cpu; udone[u] = 1;
        ++fallback_units_; fallback_samples_ += (double)U.n; fallback_core_s_ += cpu;
        if (!any_alive) degraded_ = true;
      }
    }
    for (size_t t = 0; t < tasks.size(); ++t) {
      for (size_t u = trange[t].first; u < trange[t].second && udone[u]; ++u) {
        rb_acc_merge(&out[t].acc, &uacc[u]);
        out[t].cost_s += ucost[u];
        out[t].done += units[u].n;
      }
    }
  }
  // Round report.
  void round_report(FILE *f) const override {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &D : dev_)
      fprintf(f, "    %-40s %s launches %lld  samples %.3g  kernel %.2fs  %.3g samples/s  mean "
        "fill %.2f%s\n", D.d->label().c_str(), D.alive ? "" : "[DEAD]", D.round_launches,
        D.round_samples, D.round_kernel_s,
        D.round_kernel_s > 0 ? D.round_samples / D.round_kernel_s : 0.0,
        D.launches ? D.fill_sum / D.launches : 0.0, D.alive ? "" : (" -- " + D.why_dead).c_str());
    if (fallback_units_)
      fprintf(f, "    host fallback: %lld units, %.3g samples, %.1f core-s\n", fallback_units_,
        fallback_samples_, fallback_core_s_);
  }
  // JSON report.
  void info_json(JsonW &J) const override {
    std::lock_guard<std::mutex> lk(mu_);
    J.begin_obj();
    J.kv("kind", kind_);
    J.kv("precision", o_.fp32 ? "fp32-eval/fp64-accum" : "fp64");
    J.kvb("degraded_to_cpu", degraded_);
    J.key("probe"); J.raw(probe_json_.empty() ? "null" : probe_json_);
    J.kvi("lanes_per_block", RB_BLOCK); J.kvi("samples_per_lane", RB_SPT);
    J.kvi("chunk", (long long)RB_CHUNK);
    J.kv("launch_target_s", o_.launch_s); J.kvi("unit_chunks", RB_UNIT_CHUNKS);
    J.kvi("max_launch_chunks", o_.max_launch_chunks);
    {
      char b[32];
      snprintf(b, sizeof b, "%016llx", (unsigned long long)rb_build_signature());
      J.kv("build_signature", std::string(b));
    }
    int alive = 0, active = 0;
    for (auto &D : dev_) { alive += D.alive; active += D.samples > 0; }
    J.kvi("devices_total", (long long)dev_.size()); J.kvi("devices_alive", alive);
    J.kvi("devices_with_samples", active);
    J.key("devices"); J.begin_arr();
    for (auto &D : dev_) {
      J.begin_obj();
      J.kv("label", D.d->label());
      J.key("properties"); D.d->info_json(J);
      J.kvb("alive", D.alive); J.kv("why_dead", D.why_dead);
      J.key("selftest"); J.raw(D.selftest.empty() ? "null" : D.selftest);
      J.kvi("launches", D.launches); J.kvi("chunks", D.chunks); J.kv("samples", D.samples);
      J.kv("kernel_s", D.kernel_s); J.kv("launch_min_s", D.launches ? D.min_s : 0.0);
      J.kv("launch_max_s", D.max_s);
      J.kv("samples_per_s", D.kernel_s > 0 ? D.samples / D.kernel_s : 0.0);
      J.kv("mean_fill", D.launches ? D.fill_sum / D.launches : 0.0);
      J.kv("min_fill", D.launches ? D.fill_min : 0.0);
      J.kv("max_chunk_s", D.max_chunk_s);
      J.kvi("uploads", D.uploads); J.kv("upload_bytes", D.upload_bytes);
      J.kvi("evictions", D.evictions);
      J.kvi("resident_classes", (long long)D.resident.size());
      J.kv("resident_bytes", (double)D.used);
      J.end_obj();
    }
    J.end_arr();
    J.kvi("fallback_units", fallback_units_); J.kv("fallback_samples", fallback_samples_);
    J.kv("fallback_core_s", fallback_core_s_);
    // Per class device cost (for tuning): ns per sample at full occupancy, over all devices
    J.key("class_device_ns_per_sample"); J.begin_arr();
    for (size_t c = 0; c < ccost_n_.size(); ++c)
      J.num(ccost_n_[c] > 0 ? 1e9 * ccost_s_[c] / ccost_n_[c] : NAN);
    J.end_arr();
    J.key("launch_log"); J.begin_arr();
    const size_t first = log_.size() > 2000 ? log_.size() - 2000 : 0; // the last 2000 launches
    for (size_t i = first; i < log_.size(); ++i) {
      const LaunchRec &r = log_[i];
      J.begin_arr(); J.integer(r.dev); J.num(r.t); J.num(r.elapsed_s); J.num(r.samples);
      J.integer(r.chunks); J.integer(r.items);
      J.num(r.fill); J.num(r.expected_s); J.end_arr();
    }
    J.end_arr();
    J.kv("launch_log_columns", "device, t_start_s, elapsed_s, samples, chunks, work_items, fill "
      "(chunks/(waves*grid)), expected_s");
    J.kvi("launch_log_total", (long long)log_.size());
    J.end_obj();
  }
};
} // namespace

std::unique_ptr<Backend> make_multi_backend(const BackendOptions &o, std::string *err) {
  std::vector<std::unique_ptr<Device>> devs;
  std::string probe;
  if (o.kind == "emu") {
    for (int i = 0; i < std::max(1, o.emu_devices); ++i) devs.push_back(make_emu_device(i, o));
    return std::unique_ptr<Backend>(new MultiBackend(o, "emu", std::move(devs), probe));
  }
  if (o.kind != "cuda") { *err = "unknown backend " + o.kind; return nullptr; }
  CudaProbe p = cuda_probe();
  {
    JsonW J;
    J.begin_obj(); J.kvb("compiled_with_cuda", p.compiled); J.kvi("device_count", p.count);
    J.kvi("driver_version", p.driver); J.kvi("runtime_version", p.runtime); J.kv("error", p.err);
    J.end_obj();
    probe = J.s;
  }
  if (!p.compiled) { *err = p.err; return nullptr; }
  if (p.count <= 0) {
    *err = "no CUDA device: " + (p.err.empty() ? std::string("device count 0") : p.err);
    return nullptr;
  }
  std::vector<int> ords;
  if (!o.devices.empty()) {
    std::stringstream ss(o.devices);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (tok.empty()) continue;
      const int d = atoi(tok.c_str());
      if (d < 0 || d >= p.count) { *err = "--devices: no CUDA device " + tok; return nullptr; }
      ords.push_back(d);
    }
  } else for (int d = 0; d < p.count; ++d) ords.push_back(d);
  for (int d : ords) {
    std::unique_ptr<Device> x = make_cuda_device(d, o);
    if (!x) { *err = "cannot create CUDA device " + std::to_string(d); return nullptr; }
    devs.push_back(std::move(x));
  }
  return std::unique_ptr<Backend>(new MultiBackend(o, "cuda", std::move(devs), probe));
}
