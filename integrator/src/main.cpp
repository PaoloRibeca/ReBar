/*
    main.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Driver for the single-source R-bar kernel (CPU build: bin/rebar_cpu;
    CUDA build: bin/rebar_gpu).

      rebar_* gen --legs 4|2 --loops L [--selfloops]
    list 1PI graphs (Nickel-like strings)
      rebar_* info --kind K (--list FILE | --gen) --loops L
    per-class combinatorics, JSON
      rebar_* run --kind K (--list FILE | --gen) --loops L [--N 1]
        [--seed S] [--pilot n] [--samples NTOT | --seconds T]
        [--alloc time|samples|equal] [--out res.json]
        [--backend cpu|cuda|emu|auto] [--devices 0,1,..] [--launch-s 0.5]
        ...
    Monte Carlo estimate of the order and of each of its classes, JSON
      rebar_* devices
    CUDA probe + device properties (JSON)
      rebar_* selftest [--kind K --loops L ...] [--out st.json]
    device start-up diagnostics only
      rebar_* check --what vanish|prune|twins|fp|faces|philox|emu ...
    self-diagnostics (see TECHNICAL.md)
      rebar_* bench --kind K ... [--n n]
    single-thread ns/sample per class

    All Monte Carlo streams are Philox4x32-10 with a per-class key =
    rb_class_key(seed, FNV-1a("kind|L|nickel")) and counter = (sample, kind
    << 26, call) (stream scheme "salted-v1"; --streams legacy: key = seed,
    stream = (kind << 26) | class index).  A class's random numbers depend
    only on (seed, kind, L, Nickel index), never on its position in the
    list or on other runs.  Samples are grouped and reduced by the
    canonical batch algorithm of include/rbar/batch.h on every backend, so
    per-class results are bitwise reproducible for any thread count, any
    number of (emulated) devices and any grid size; a real GPU differs only
    by device-libm round-off / FMA.

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

#include <errno.h>
#include <omp.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "backend.h"
#include "cache.h"
#include "build.h"
#include "json.h"
#include "jsonr.h"
#include "refs.h"

#ifndef RB_VERSION
#define RB_VERSION "0.4.3"
#endif
#ifndef RB_BUILD_INFO
// Build stamp (Makefile, from tools/build-info): "<release>-<count> <date> <commit>"; empty outside
// a git checkout
#define RB_BUILD_INFO ""
#endif
#ifndef RB_VARIANT
// Compile-time limit variant (Makefile: e16 = v0.3.0 limits, e18 = RB_MAXE 18 ...)
#define RB_VARIANT "e16"
#endif
#ifndef RB_PROGRAM
#define RB_PROGRAM "rebar_cpu"
#endif
#ifndef RB_DEFAULT_BACKEND
#define RB_DEFAULT_BACKEND "cpu"
#endif

// Options.
struct Args {
  std::map<std::string, std::string> kv;
  std::string cmd;
  bool has(const std::string &k) const { return kv.count(k) > 0; }
  std::string get(const std::string &k, const std::string &d = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  double getd(const std::string &k, double d) const { return has(k) ? atof(get(k).c_str()) : d; }
  long long getl(const std::string &k, long long d) const {
    return has(k) ? (long long)atof(get(k).c_str()) : d;
  }
};
static Args parse_args(int argc, char **argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      std::string k = s.substr(2), v = "1";
      auto eq = k.find('=');
      if (eq != std::string::npos) { v = k.substr(eq + 1); k = k.substr(0, eq); }
      else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) v = argv[++i];
      a.kv[k] = v;
    }
  }
  return a;
}
static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double thread_cpu_s() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}
static std::string utc_now() {
  time_t t = time(nullptr);
  char b[64];
  strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
  return b;
}

// Graphs.
static bool load_list(const std::string &fn, std::vector<Graph> &out, std::string *err) {
  std::ifstream f(fn);
  if (!f) { *err = "cannot open " + fn; return false; }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::string tok, nk;
    while (std::getline(ss, tok, '\t')) {
      std::string t;
      for (char c : tok) if (!isspace((unsigned char)c)) t += c;
      if (t.find('|') != std::string::npos) { nk = t; break; }
    }
    if (nk.empty()) continue;
    Graph G;
    if (!parse_nickel(nk, G, err)) return false;
    out.push_back(G);
  }
  return true;
}
static bool get_graphs(const Args &a, Kind kind, int L, std::vector<Graph> &gs, std::string *err) {
  if (a.has("list")) return load_list(a.get("list"), gs, err);
  int E = kind_legs(kind);
  int V = E == 4 ? L + 1 : L;
  bool sl = a.has("selfloops") || kind == Kind::C;
  gs = generate_graphs(V, E, sl);
  return true;
}

// Limit variants.
// The compile-time limits (RB_MAXE lines, RB_MAXLOOP loops of an atom, RB_MAXV vertices,
// RB_MAXATOMS) size per-thread arrays of the kernel; the e16 binaries (v0.3.0 limits) are the
// fastest and hold every order up to 8 loops, the e18 ones (<name>-e18) 9-loop Gamma2 and Gamma4.
// An e16 binary given graphs it cannot hold re-executes the e18 binary next to it with the same
// arguments (REBAR_NO_REEXEC=1 disables this); the numbers do not depend on it.
static char **g_argv = nullptr;
static std::string self_exe() {
  char b[4096];
  const ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
  if (n <= 0) return "";
  b[n] = 0;
  return b;
}
static void reexec_bigger(const std::string &why) {
  if (std::string(RB_VARIANT) != "e16" || getenv("REBAR_NO_REEXEC") || !g_argv) return;
  const std::string self = self_exe();
  if (self.empty()) return;
  const std::string sib = self + "-e18";
  if (access(sib.c_str(), X_OK) != 0) {
    fprintf(stderr, "error: %s; the larger-limit binary %s does not exist (make cpu / make gpu "
      "build both variants)\n", why.c_str(), sib.c_str());
    return;
  }
  fprintf(stderr, "note: %s: re-executing as %s\n", why.c_str(), sib.c_str());
  fflush(stdout);
  fflush(stderr);
  execv(sib.c_str(), g_argv);
  fprintf(stderr, "error: cannot execute %s: %s\n", sib.c_str(), strerror(errno));
}
static bool deriv_kind(const std::string &k) { return k == "h" || k == "bare2d" || k == "int2d"; }
// 0 if this binary can hold the command's graphs; otherwise re-executes the e18 binary (no return)
// or returns 2
static int precheck_limits(const Args &a) {
  const std::string kname = a.get("kind", "b");
  Kind kind;
  if (!kind_from_string(kname, &kind)) return 0; // the command reports it
  int lines = 0, loops = 0, verts = 0;
  auto add = [&](int l, int L, int V) {
    lines = std::max(lines, l);
    loops = std::max(loops, L);
    verts = std::max(verts, V);
  };
  if (a.cmd == "eval") {
    Graph G; std::string err;
    if (parse_nickel(a.get("nickel"), G, &err)) add(G.l(), G.loops(), G.V);
  } else if (a.has("list")) {
    std::vector<Graph> gs; std::string err;
    if (!load_list(a.get("list"), gs, &err)) return 0; // the command reports it
    for (auto &G : gs) add(G.l(), G.loops(), G.V);
  } else {
    const int E = kind_legs(kind);
    std::vector<int> Ls = {(int)a.getl("loops", a.cmd == "selftest" ? 4 : 2)};
    if (a.has("loops2")) Ls.push_back((int)a.getl("loops2", 0));
    else if (a.cmd == "check" && a.get("what") == "streams") Ls.push_back(Ls[0] + 1);
    for (int L : Ls) { const int V = E == 4 ? L + 1 : L; add(E == 4 ? 2 * L : 2 * L - 1, L, V); }
  }
  const int needloop = loops + (deriv_kind(kname) ? 1 : 0);
  if (lines <= RB_MAXE && needloop <= RB_MAXLOOP && verts <= RB_MAXV) return 0;
  char why[320];
  snprintf(why, sizeof why, "the graphs need %d lines, atoms of %d loops, %d vertices; this binary "
    "(variant %s) has RB_MAXE %d, RB_MAXLOOP %d, RB_MAXV %d", lines, needloop, verts, RB_VARIANT,
    RB_MAXE, RB_MAXLOOP, RB_MAXV);
  reexec_bigger(why); // returns only if that is impossible
  fprintf(stderr, "error: %s\n", why);
  return 2;
}

// MC core.
static void acc_stats(const RbAcc &a, double *mean, double *sd) {
  double n = a.n, s = a.s + a.sc, q = a.q + a.qc;
  *mean = n > 0 ? s / n : NAN;
  *sd = n > 1 ? std::sqrt(std::max(0.0, (q - s * s / n) / (n - 1))) : NAN;
}
// Kurtosis of w and the resulting relative standard error of the sd estimate, sqrt((kappa - 1)/(4
// n))
static double acc_kurtosis(const RbAcc &a) {
  double n = a.n;
  if (n < 4) return NAN;
  double m = (a.s + a.sc) / n, e2 = (a.q + a.qc) / n, e3 = a.q3 / n, e4 = a.q4 / n;
  double mu2 = e2 - m * m, mu4 = e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m * m * m * m;
  return mu2 > 0 ? mu4 / (mu2 * mu2) : NAN;
}
static double acc_sd_relerr(const RbAcc &a) {
  double k = acc_kurtosis(a);
  return std::sqrt(std::max(0.0, k - 1) / (4 * a.n));
}

// Classes.
struct ClassRun {
  ClassData cd;
  double build_s = 0;
  // Device image (pack_class) and approximate host memory of the class
  size_t image_bytes = 0, host_bytes = 0;
  double SN = 0; // symmetry x O(N) factor at the requested N
  RbAcc acc{}; // value-initialised: all zero
  std::vector<std::string> merged; // Whitney twins merged into this class (--dedup)
  // Weight polynomial P(N) = Sum over merged twins of S_j * on_j(N); SN = P(N)
  std::vector<double> wpoly;
  // (nickel, S) of every member (representative first)
  std::vector<std::pair<std::string, double>> twin_S;
  std::vector<std::vector<double>> twin_on; // their O(N) polynomials
  uint64_t salt = 0; // FNV-1a 64 of "kind|L|nickel": per-class part of the Philox key
  RbStreamId id{0, 0}; // Philox key and counter word of the class in this run
  double cpu = 0; // cost spent: core-seconds (cpu backend) or device-seconds (cuda, emu)
  uint64_t nalloc = 0; // target total samples (pilot + main)
  uint64_t next = 0; // first sample index not yet merged into acc (committed state)
  // Round targets: base + (nalloc - base) * r / rounds (base = pilot size; or the sample count at
  // the last --extend)
  uint64_t base = 0;
  // Task of the current round interrupted after a prefix of its units (time box, SIGTERM): its
  // accumulator so far (the canonical in-order merge of the units [next, next + part_done)) is kept
  // apart from acc, so that a resumed run continues that merge exactly (bitwise the same as an
  // uninterrupted run); the JSON shows acc + part.
  RbAcc part{};
  uint64_t part_done = 0;
  RbAcc shown() const {
    if (part_done == 0) return acc;
    RbAcc x = acc;
    rb_acc_merge(&x, &part);
    return x;
  }
};
static int series_sign(Kind k, int L) {
  if (k == Kind::B || k == Kind::C) return (L + 1) % 2 ? -1 : 1; // (-1)^(L+1)
  if (k == Kind::H) return L % 2 ? -1 : 1; // (-1)^L
  return 1;
}

static std::string hex64(uint64_t v) {
  char b[32];
  snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
  return b;
}
static double acc_roundoff_rms_over_sd(const RbAcc &a, double sd) {
  return 2.2e-16 * std::sqrt(a.sabs2 / a.n) / sd;
}
static double acc_roundoff_max_over_sd(const RbAcc &a, double sd) {
  return 2.2e-16 * a.maxsabs / sd;
}

static void write_class_json(JsonW &J, const ClassRun &c, int idx) {
  const ClassData &d = c.cd;
  J.begin_obj();
  J.kvi("idx", idx);
  J.kv("nickel", d.G.name);
  J.kvi("V", d.G.V); J.kvi("l", d.l); J.kvi("L", d.L);
  J.kv("S", d.S); J.kvarr("on_poly", d.on); J.kv("SN", c.SN);
  // SN = weight_poly(N): use this (not S * on_poly) to recombine at other N
  J.kvarr("weight_poly", c.wpoly);
  J.kv("salt", hex64(c.salt)); J.kv("philox_key", hex64(c.id.key));
  J.kvi("philox_stream", (long long)c.id.stream);
  J.kvb("is1vr", d.is1vr); J.kvb("selfloop", d.selfloop);
  J.key("merged_twins"); J.begin_arr(); for (auto &m : c.merged) J.str(m); J.end_arr();
  if (!c.merged.empty()) {
    J.key("twins"); J.begin_arr();
    for (size_t k = 0; k < c.twin_S.size(); ++k) {
      J.begin_obj(); J.kv("nickel", c.twin_S[k].first); J.kv("S", c.twin_S[k].second);
      J.kvarr("on_poly", c.twin_on[k]); J.end_obj();
    }
    J.end_arr();
  }
  J.kvi("nparts", d.nparts); J.kvi("ndots", d.ndots); J.kvi("nforests", d.nforests);
  J.kvi("nterms_raw", d.nterms_raw); J.kvi("ncomp", d.ncomp); J.kvi("ngroup", d.ngroup);
  J.kvi("natoms", d.natoms);
  J.kvi("max_atom_loops", d.max_atom_loops); J.kvi("nshift", d.nshift);
  J.kv("omega_min", d.omega_min);
  J.kvi("omega2_pre_min", d.omega2_pre_min); J.kvi("nneg_faces", d.nneg_faces);
  J.kv("pw", d.pw); J.kv("pref", d.pref); J.kv("logJE", d.logJE); J.kv("build_s", c.build_s);
  J.kv("build_expand_s", d.t_expand); J.kv("build_tables_s", d.t_tables);
  J.kv("image_bytes", (double)c.image_bytes); J.kv("host_bytes", (double)c.host_bytes);
  const RbAcc acc = c.shown();
  if (acc.n > 0) {
    double m, sd; acc_stats(acc, &m, &sd);
    double a = d.pref;
    J.kv("n", acc.n);
    J.kv("mean_w", m); J.kv("sd_w", sd);
    J.kv("value", a * m); J.kv("error", a * sd / std::sqrt(acc.n));
    J.kv("contrib", c.SN * a * m); J.kv("contrib_err", std::fabs(c.SN) * a * sd / std::sqrt(acc.n));
    J.kv("rel_sd_per_sample", sd / std::fabs(m));
    J.kv("abs_sd_per_sample", std::fabs(c.SN) * a * sd);
    J.kv("maxw_over_absmean", acc.maxw / std::fabs(m));
    J.kv("nbad", acc.nbad); J.kv("ncancel_1e-8", acc.ncanc);
    J.kv("roundoff_rms_over_sd", acc_roundoff_rms_over_sd(acc, sd));
    // Worst single sample: eps * max Sum|terms| / sd
    J.kv("roundoff_max_over_sd", acc_roundoff_max_over_sd(acc, sd));
    J.kv("core_ns_per_sample", 1e9 * c.cpu / acc.n);
    J.kv("kurtosis", acc_kurtosis(acc));
    J.kv("sd_relerr", acc_sd_relerr(acc));
  }
  J.end_obj();
}

struct OrderSummary {
  double sum = 0, err = 0, err_relerr = 0, rho_neyman = 0, rho_time = 0, sumabs = 0, n = 0, cpu = 0,
      mean_cost = 0;
  double nbad = 0, roundoff_max_over_sd = 0, roundoff_rms_over_sd = 0;
  int classes_with_nbad = 0;
};
static OrderSummary summarise(const std::vector<ClassRun> &cr) {
  OrderSummary o;
  double var = 0, ny = 0, nt = 0;
  for (auto &c : cr) {
    const RbAcc acc = c.shown();
    o.nbad += acc.nbad;
    o.classes_with_nbad += acc.nbad > 0;
    if (acc.n <= 0) continue;
    double m, sd; acc_stats(acc, &m, &sd);
    double a = c.SN * c.cd.pref;
    o.sum += a * m;
    o.sumabs += std::fabs(a * m);
    var += a * a * sd * sd / acc.n;
    {
      double vh = a * a * sd * sd / acc.n, r = acc_sd_relerr(acc);
      if (std::isfinite(r)) o.err_relerr += (2 * r * vh) * (2 * r * vh);
    }
    if (sd > 0) {
      o.roundoff_max_over_sd = std::max(o.roundoff_max_over_sd, acc_roundoff_max_over_sd(acc, sd));
      o.roundoff_rms_over_sd = std::max(o.roundoff_rms_over_sd, acc_roundoff_rms_over_sd(acc, sd));
    }
    double cost = c.cpu / acc.n;
    ny += std::fabs(a) * sd;
    nt += std::fabs(a) * sd * std::sqrt(cost);
    o.n += acc.n; o.cpu += c.cpu;
  }
  o.err = std::sqrt(var);
  // Relative standard error of err
  o.err_relerr = var > 0 ? std::sqrt(o.err_relerr) / (2 * var) : NAN;
  o.rho_neyman = ny / std::fabs(o.sum);
  o.rho_time = nt / std::fabs(o.sum);
  // Mean cost per sample under cost-aware Neyman ~ (sum a s sqrt c / sum a s)^2
  o.mean_cost = (nt > 0) ? (nt * nt) / (ny * ny) : 0;
  return o;
}
// Warnings that a run's numbers need attention (also printed on stderr at the end of the run)
static std::vector<std::string> run_warnings(const OrderSummary &o, const std::string &be_note) {
  std::vector<std::string> w;
  if (o.nbad > 0)
    w.push_back(std::to_string((long long)o.nbad) + " non-finite weight(s) in "
        + std::to_string(o.classes_with_nbad)
        + " class(es) were EXCLUDED from the sums (biased if not negligible): see classes[].nbad");
  if (o.roundoff_max_over_sd > 1e-3)
    w.push_back("a single sample's round-off scale reached "
        + std::to_string(o.roundoff_max_over_sd) + " of the per-sample sd "
        "(classes[].roundoff_max_over_sd): cancellation near a face; check with long double");
  if (!be_note.empty()) w.push_back(be_note);
  return w;
}

// Run state of the driver (what a checkpoint must hold besides the per-class accumulators).
struct RunSession {
  std::string utc, program, version, backend, resumed_from;
  double wall_s = 0, samples = 0;
  int round_from = 0, round_to = 0;
  bool extended = false;
};
struct RunState {
  bool pilot_done = false;
  int round_done = 0; // main-phase rounds completed
  int rounds = 0; // planned main-phase rounds (0: not planned yet)
  uint64_t npilot = 0;
  std::string alloc; // samples | time | equal
  double total = 0; // main-phase samples of the (last) plan
  bool complete = false; // every planned round done
  int extensions = 0;
  double wall_prev = 0; // wall-clock seconds of the earlier sessions
  std::vector<RunSession> sessions; // earlier sessions + this one (last)
};

static std::string hex_words(const void *p, size_t nbytes) {
  std::string r;
  const unsigned char *b = (const unsigned char *)p;
  for (size_t i = 0; i + 8 <= nbytes; i += 8) { uint64_t w; memcpy(&w, b + i, 8); r += hex64(w); }
  return r;
}
static bool unhex_words(const std::string &h, void *p, size_t nbytes) {
  if (h.size() != 2 * nbytes) return false;
  unsigned char *b = (unsigned char *)p;
  for (size_t i = 0; i < nbytes / 8; ++i) {
    char *end = nullptr;
    const std::string w = h.substr(16 * i, 16);
    const uint64_t v = strtoull(w.c_str(), &end, 16);
    if (!end || *end) return false;
    memcpy(b + 8 * i, &v, 8);
  }
  return true;
}
static_assert(sizeof(RbAcc) % 8 == 0, "RbAcc must be a whole number of 64-bit words");

// "checkpoint": everything a later `run --resume FILE` needs to continue this run exactly
// (rbar-checkpoint-1).
// Accumulators and cost as hex bit patterns (exact); sample counters as integers (exact below
// 2^53).
static void write_checkpoint_json(JsonW &J, const std::vector<ClassRun> &cr, const RunState &rs) {
  J.key("checkpoint"); J.begin_obj();
  J.kv("format", "rbar-checkpoint-1");
  J.kvb("complete", rs.complete);
  J.kvb("pilot_done", rs.pilot_done);
  J.kvi("round_done", rs.round_done); J.kvi("rounds", rs.rounds);
  J.kvi("npilot", (long long)rs.npilot);
  J.kv("alloc", rs.alloc); J.kv("total", hex_words(&rs.total, 8));
  J.kvi("extensions", rs.extensions);
  J.kv("wall_s_previous", hex_words(&rs.wall_prev, 8));
  J.key("sessions"); J.begin_arr();
  for (auto &x : rs.sessions) {
    J.begin_obj();
    J.kv("utc", x.utc); J.kv("program", x.program); J.kv("version", x.version);
    J.kv("backend", x.backend);
    J.kv("resumed_from", x.resumed_from); J.kvb("extended", x.extended);
    J.kv("wall_s", x.wall_s); J.kv("samples", x.samples); J.kvi("round_from", x.round_from);
    J.kvi("round_to", x.round_to);
    J.end_obj();
  }
  J.end_arr();
  J.kv("class_columns", "nickel, salt, ncomp, natoms, next, part_done, base, nalloc, cost_s (hex), "
    "acc (RbAcc, hex words), part (RbAcc, hex words)");
  J.key("classes"); J.begin_arr();
  for (auto &c : cr) {
    J.begin_arr();
    J.str(c.cd.G.name); J.str(hex64(c.salt)); J.integer(c.cd.ncomp); J.integer(c.cd.natoms);
    J.integer((long long)c.next); J.integer((long long)c.part_done); J.integer((long long)c.base);
    J.integer((long long)c.nalloc);
    J.str(hex_words(&c.cpu, 8)); J.str(hex_words(&c.acc, sizeof(RbAcc)));
    J.str(hex_words(&c.part, sizeof(RbAcc)));
    J.end_arr();
  }
  J.end_arr();
  J.end_obj();
}

static void write_run_json(const std::string &fn, const Args &a, Kind kind, int L, double N,
    uint64_t seed, const std::vector<ClassRun> &cr,
    const std::vector<std::pair<std::string, std::string>> &dropped, double wall, bool final_,
    const Backend *be = nullptr, const std::string &be_note = "", const RunState *rs = nullptr) {
  JsonW J;
  J.begin_obj();
  J.kv("program", RB_PROGRAM); J.kv("version", RB_VERSION); J.kv("build", RB_BUILD_INFO);
  J.kv("utc", utc_now());
  J.kvb("final", final_);
  if (rs) J.kvb("complete", rs->complete);
  J.key("limits"); J.begin_obj();
  J.kv("variant", RB_VARIANT); J.kvi("RB_MAXE", RB_MAXE); J.kvi("RB_MAXV", RB_MAXV);
  J.kvi("RB_MAXLOOP", RB_MAXLOOP); J.kvi("RB_MAXATOMS", RB_MAXATOMS);
  J.end_obj();
  J.kv("kind", kind_name(kind)); J.kvi("L", L); J.kv("N", N); J.kv("seed", std::to_string(seed));
  J.kv("streams", a.get("streams", "salted-v1"));
  J.kv("precision", a.has("fp32") ? "fp32-eval/fp64-accum" : "fp64");
  J.kvi("threads", omp_get_max_threads());
  J.kv("omega_scale", a.getd("omega-scale", 0.7)); J.kvi("omega_shift2", a.getl("omega-shift", 2));
  {
    char b[32];
    snprintf(b, sizeof b, "%016llx", (unsigned long long)rb_build_signature());
    J.kv("build_signature", std::string(b));
  }
  J.key("args"); J.begin_obj(); for (auto &p : a.kv) J.kv(p.first, p.second); J.end_obj();
  J.key("classes"); J.begin_arr();
  for (size_t i = 0; i < cr.size(); ++i) write_class_json(J, cr[i], (int)i);
  J.end_arr();
  J.key("dropped"); J.begin_arr();
  for (auto &d : dropped) {
    J.begin_obj();
    J.kv("nickel", d.first);
    J.kv("reason", d.second);
    J.end_obj();
  }
  J.end_arr();
  OrderSummary o = summarise(cr);
  J.key("order"); J.begin_obj();
  J.kv("sum", o.sum); J.kv("err", o.err);
  J.kv("err_relerr", o.err_relerr);
  int sg = series_sign(kind, L);
  J.kv("series_coeff", sg * o.sum); J.kv("series_err", o.err);
  J.kv("rho_neyman_per_sample", o.rho_neyman);
  J.kv("rho_time_sqrt_core_s", o.rho_time);
  J.kv("mean_core_ns_per_sample_neyman_time", 1e9 * o.mean_cost);
  J.kv("sum_abs_over_abs_sum", o.sumabs / std::fabs(o.sum));
  J.kv("samples", o.n); J.kv("core_s", o.cpu); J.kv("wall_s", wall);
  J.kv("nbad", o.nbad); J.kvi("classes_with_nbad", o.classes_with_nbad);
  J.kv("roundoff_max_over_sd", o.roundoff_max_over_sd);
  J.kv("roundoff_rms_over_sd", o.roundoff_rms_over_sd);
  double rv, re;
  std::string kn = kind_name(kind);
  bool have = find_ref(kn.c_str(), N, L, &rv, &re);
  if (kind == Kind::INT4 && have) {
    double f = std::pow(2.0, L) / 3.0;
    J.kv("phi4tools_S_L", f * o.sum);
    J.kv("phi4tools_S_L_err", f * o.err);
  }
  if (have) {
    double val = (kind == Kind::INT4) ? std::pow(2.0, L) / 3.0 * o.sum : o.sum;
    double er = (kind == Kind::INT4) ? std::pow(2.0, L) / 3.0 * o.err : o.err;
    J.kv("ref", rv); J.kv("ref_err", re);
    J.kv("pull", (val - rv) / std::sqrt(er * er + re * re));
    J.kv("rel_dev", (val - rv) / rv);
  }
  J.end_obj();
  J.key("warnings"); J.begin_arr(); for (auto &w : run_warnings(o, be_note)) J.str(w); J.end_arr();
  // Backend (CPU / CUDA / emulated): which one ran, why (fallback), devices, self-tests, per-launch
  // timing.
  // With cuda/emu, "core_ns_per_sample" and "core_s" are DEVICE time (each launch's elapsed time x
  // fill factor, apportioned to its classes by their block cycles); with cpu they are thread CPU
  // time, as before.
  J.kv("backend", be ? be->name() : std::string("none"));
  J.kv("backend_note", be_note);
  J.kv("cost_unit", be ? be->cost_unit() : "core-s");
  J.key("backend_info");
  if (be) be->info_json(J); else J.raw("null");
  if (rs) write_checkpoint_json(J, cr, *rs);
  J.end_obj();
  if (!J.write(fn)) fprintf(stderr, "cannot write %s\n", fn.c_str());
}

// Build options from the command line: ONE place, used by every command (run, info, bench, check,
// eval)
static BuildOptions make_build_options(const Args &a, Kind kind) {
  BuildOptions bo;
  bo.kind = kind;
  bo.prune = !a.has("noprune");
  bo.omega_shift = (int)a.getl("omega-shift", 2);
  // Default from the omega scan at 4-5 loops
  bo.omega_scale = a.getd("omega-scale", 0.7);
  bo.allow_negative_faces = a.has("allow-negative-faces");
  return bo;
}

static bool build_all(const Args &a, Kind kind, int L, double N, std::vector<ClassRun> &cr,
    std::vector<std::pair<std::string, std::string>> &dropped, bool tables) {
  std::vector<Graph> gs;
  std::string err;
  if (!get_graphs(a, kind, L, gs, &err)) {
    fprintf(stderr, "error: %s\n", err.c_str());
    return false;
  }
  BuildOptions bo = make_build_options(a, kind);
  bo.need_tables = tables;
  bool drop_struct = a.has("drop-1vr");
  // Self-loop (tadpole) graphs vanish identically for b, h and c at L >= 2 (check I2 of
  // ../checks/); --drop-selfloops skips them without building them, also for c, where --drop-1vr is
  // ignored (below); one-vertex-reducible graphs vanish for c too, but only after algebraic pruning
  // (c 6: 13 of 75, c 7: 52 of 317)
  bool drop_sl = a.has("drop-selfloops") && L >= 2
      && (kind == Kind::B || kind == Kind::H || kind == Kind::C);
  for (auto &G : gs)
    if (G.loops() != L) {
      fprintf(stderr, "error: %s has %d loops, expected %d\n", G.name.c_str(), G.loops(), L);
      return false;
    }
  // Classes are built in parallel (the builder has no shared state) and collected in list order
  std::vector<ClassRun> tmp(gs.size());
  // 0 keep, 1 dropped (structural), 2 dropped (tadpole), 3 vanishes, 4 error
  std::vector<int> st(gs.size(), 0);
  std::vector<std::string> errs(gs.size());
  // --class-cache FILE: built classes are taken from the file when it exists and was built with the
  // same code, limits and options; the classes built here are written to it afterwards
  // (`rebar build` does only that)
  std::map<std::string, CachedClass> cache;
  std::vector<char> fromcache(gs.size(), 0);
  // An existing file built differently is left alone (never overwritten)
  bool cache_refused = false;
  const std::string cfp = cache_fingerprint(bo, tables, RB_VERSION);
  if (a.has("class-cache")) {
    std::vector<CachedClass> cc; std::string cerr;
    std::ifstream probe(a.get("class-cache"), std::ios::binary);
    if (probe) {
      probe.close();
      if (!cache_read(a.get("class-cache"), cfp, cc, &cerr)) {
        cache_refused = true;
        fprintf(stderr, "class cache not used (and not overwritten): %s\n", cerr.c_str());
      } else {
        for (auto &x : cc) cache[x.cd.G.name] = std::move(x);
        fprintf(stderr, "class cache: %zu classes read from %s\n", cache.size(),
          a.get("class-cache").c_str());
      }
    }
  }
  const double tb0 = now_s();
  // Progress of a long build (every 60 s, from whichever thread finishes a class first after that):
  // classes done, elapsed, estimated remaining; classes vary a lot in cost, so the estimate is
  // rough
  long long ndone = 0; double tlast = tb0;
#pragma omp parallel for schedule(dynamic, 1)
  for (long long gi = 0; gi < (long long)gs.size(); ++gi) {
    if (gs.size() > 200) {
      long long k;
#pragma omp atomic capture
      k = ++ndone;
      const double tn = now_s();
      if (tn - tlast > 60.0) {
#pragma omp critical(build_progress)
        if (tn - tlast > 60.0) {
          tlast = tn;
          const double el = tn - tb0;
          fprintf(stderr, "build: %lld/%zu classes started, %.0f s elapsed, about %.0f s to go\n",
            k, gs.size(), el, el * ((double)gs.size() / (double)k - 1.0));
        }
      }
    }
    const Graph &G = gs[gi];
    if (drop_struct && (kind == Kind::B || kind == Kind::H) && (is_1vr(G) || has_selfloop(G))) {
      st[gi] = 1;
      continue;
    }
    if (drop_sl && has_selfloop(G)) { st[gi] = 5; continue; }
    if ((kind == Kind::INT4 || kind == Kind::INT2D || kind == Kind::INT2) && has_selfloop(G)) {
      st[gi] = 2;
      continue;
    }
    ClassRun &c = tmp[gi];
    if (!cache.empty()) {
      auto it = cache.find(G.name);
      if (it != cache.end()) {
        c.cd = it->second.cd; c.build_s = it->second.build_s;
        c.wpoly = c.cd.on;
        for (double &x : c.wpoly) x *= c.cd.S;
        c.SN = c.cd.S * eval_poly(c.cd.on, N);
        c.twin_S = {{G.name, c.cd.S}};
        c.twin_on = {c.cd.on};
        if (c.cd.ncomp == 0) st[gi] = 3;
        fromcache[gi] = 1;
        continue;
      }
    }
    double t0 = now_s();
    if (!build_class(G, bo, c.cd, &errs[gi])) {
      st[gi] = (a.has("skip-divergent") && errs[gi].rfind("divergent", 0) == 0) ? 6 : 4;
      continue;
    }
    c.build_s = now_s() - t0;
    c.image_bytes = packed_bytes(c.cd);
    c.host_bytes = c.cd.host_bytes();
    c.wpoly = c.cd.on;
    for (double &x : c.wpoly) x *= c.cd.S;
    c.SN = c.cd.S * eval_poly(c.cd.on, N); // = weight_poly(N), but exactly S at N = 1
    c.twin_S = {{G.name, c.cd.S}};
    c.twin_on = {c.cd.on};
    if (c.cd.ncomp == 0) st[gi] = 3;
  }
  if (a.has("class-cache")) {
    size_t nbuilt = 0, ncached = 0;
    for (size_t gi = 0; gi < gs.size(); ++gi)
      if (st[gi] == 0 || st[gi] == 3) { if (fromcache[gi]) ++ncached; else ++nbuilt; }
    if (nbuilt > 0 && !cache_refused) {
      std::vector<CachedClass> cc; cc.reserve(nbuilt + ncached);
      for (size_t gi = 0; gi < gs.size(); ++gi)
        if (st[gi] == 0 || st[gi] == 3) cc.push_back({tmp[gi].cd, tmp[gi].build_s});
      std::string cerr;
      if (!cache_write(a.get("class-cache"), cfp, cc, &cerr))
        fprintf(stderr, "class cache not written: %s\n", cerr.c_str());
      else
        fprintf(stderr,
          "class cache: %zu classes written to %s (%zu built now, %zu from the file)\n", cc.size(),
          a.get("class-cache").c_str(), nbuilt, ncached);
    } else if (ncached > 0)
      fprintf(stderr, "class cache: all %zu classes from %s\n", ncached,
        a.get("class-cache").c_str());
  }
  for (size_t gi = 0; gi < gs.size(); ++gi) {
    const Graph &G = gs[gi];
    if (st[gi] == 4) {
      const std::string &e = errs[gi];
      if (e.rfind("too many atoms", 0) == 0 || e.rfind("too many lines", 0) == 0
          || e.rfind("atom with too many", 0) == 0)
        reexec_bigger("class " + G.name + ": " + e);
      fprintf(stderr, "error building %s: %s\n", G.name.c_str(), e.c_str());
      return false;
    }
    if (st[gi] == 1) {
      dropped.push_back({G.name, "structural: 1VR/tadpole (vanishes identically for b,h)"});
      continue;
    }
    if (st[gi] == 2) {
      dropped.push_back({G.name, "tadpole (zero in the intermediate scheme)"});
      continue;
    }
    if (st[gi] == 5) {
      dropped.push_back({G.name, "structural: self-loop (vanishes identically at L >= 2)"});
      continue;
    }
    if (st[gi] == 6) { dropped.push_back({G.name, "--skip-divergent: " + errs[gi]}); continue; }
    if (st[gi] == 3) {
      dropped.push_back({G.name, std::string("vanishes identically after algebraic pruning")
          + (tmp[gi].cd.is1vr ? " [1VR]" : "") + (tmp[gi].cd.selfloop ? " [self-loop]" : "")});
      continue;
    }
    cr.push_back(std::move(tmp[gi]));
  }
  if (gs.size() > 50)
    fprintf(stderr, "built %zu graphs in %.1fs (%d threads)\n", gs.size(), now_s() - tb0,
      omp_get_max_threads());
  if (a.has("dedup")) {
    // Merge cycle-matroid (Whitney) twins, but only after verifying pointwise equality of the
    // subtracted, dot-summed integrands under the matroid edge map (long double; 12 points in
    // [0.05,1]^l and 4 near random faces; the difference must be small relative to |f| itself, not
    // only to Sum|terms|).  The twins' O(N) polynomials usually differ, so the merged class carries
    // the weight polynomial P(N) = Sum_j S_j on_j(N).
    std::mt19937_64 rng(4242);
    std::vector<char> gone(cr.size(), 0);
    int nmerge = 0, nrefuse = 0;
    for (size_t i = 0; i < cr.size(); ++i) {
      if (gone[i]) continue;
      for (size_t j = i + 1; j < cr.size(); ++j) {
        if (gone[j] || cr[j].cd.l != cr[i].cd.l) continue;
        std::vector<int> emap;
        if (!matroid_iso(cr[i].cd.G, cr[j].cd.G, &emap)) continue;
        long double worst = 0, scale = 0, fmax = 0;
        const int l = cr[i].cd.l;
        for (int k = 0; k < 16; ++k) {
          double lx[RB_MAXE], ly[RB_MAXE];
          std::uniform_real_distribution<double> U(std::log(0.05), 0.0);
          for (int e = 0; e < l; ++e) lx[e] = U(rng);
          if (k >= 12) { // near a random face: a random subset of lines scaled by 1e-4
            const uint32_t S = (uint32_t)(rng() % ((1ull << l) - 2)) + 1u;
            for (int e = 0; e < l; ++e) if ((S >> e) & 1u) lx[e] += std::log(1e-4);
          }
          for (int e = 0; e < l; ++e) ly[emap[e]] = lx[e];
          long double sa, sb;
          long double va = eval_point<long double>(cr[i].cd, lx, &sa),
              vb = eval_point<long double>(cr[j].cd, ly, &sb);
          worst = std::max(worst, fabsl(va - vb)); scale = std::max(scale, std::max(sa, sb));
          fmax = std::max(fmax, fabsl(va));
        }
        if (worst <= 1e-15L * scale && worst <= 1e-11L * fmax) {
          for (size_t q = 0; q < cr[j].wpoly.size(); ++q) {
            if (cr[i].wpoly.size() <= q) cr[i].wpoly.resize(q + 1, 0.0);
            cr[i].wpoly[q] += cr[j].wpoly[q];
          }
          cr[i].SN += cr[j].SN; // = Sum_j S_j on_j(N) = weight_poly(N)
          cr[i].merged.push_back(cr[j].cd.G.name);
          for (auto &m : cr[j].merged) cr[i].merged.push_back(m);
          for (size_t q = 0; q < cr[j].twin_S.size(); ++q) {
            cr[i].twin_S.push_back(cr[j].twin_S[q]);
            cr[i].twin_on.push_back(cr[j].twin_on[q]);
          }
          dropped.push_back({cr[j].cd.G.name, "merged into Whitney twin " + cr[i].cd.G.name
              + " (pointwise-equal integrands verified)"});
          gone[j] = 1; ++nmerge;
        } else {
          fprintf(stderr, "warning: twins %s %s NOT pointwise equal (%.3Le vs scale %.3Le, |f| "
            "%.3Le): kept separate\n", cr[i].cd.G.name.c_str(), cr[j].cd.G.name.c_str(), worst,
            scale, fmax);
          ++nrefuse;
        }
      }
    }
    std::vector<ClassRun> keep;
    for (size_t i = 0; i < cr.size(); ++i) if (!gone[i]) keep.push_back(std::move(cr[i]));
    cr = std::move(keep);
    fprintf(stderr, "dedup: %d twins merged, %d refused\n", nmerge, nrefuse);
  }
  return true;
}

// Commands.
static int cmd_gen(const Args &a) {
  int E = (int)a.getl("legs", 4), L = (int)a.getl("loops", 1);
  int V = E == 4 ? L + 1 : L;
  auto gs = generate_graphs(V, E, a.has("selfloops"));
  int k = 0;
  for (auto &G : gs)
    printf("%d\t%s\t%.12g\t1VR=%d\tselfloop=%d\n", ++k, G.name.c_str(), sym_factor(G),
      (int)is_1vr(G), (int)has_selfloop(G));
  fprintf(stderr, "%zu graphs (V=%d, E=%d, L=%d, selfloops %s)\n", gs.size(), V, E, L,
    a.has("selfloops") ? "allowed" : "excluded");
  return 0;
}

static bool assign_streams(const Args &a, Kind kind, int L, uint64_t seed,
    std::vector<ClassRun> &cr, std::vector<RbStreamId> *ids, std::string *err);
// build: construct the classes of an order and write them to --class-cache FILE (no sampling); a
// later run or info with the same --class-cache, code and options loads them instead of building
static int cmd_build(const Args &a) {
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  if (!a.has("class-cache")) { fprintf(stderr, "build needs --class-cache FILE\n"); return 1; }
  // build replaces the file; a run never overwrites one it refused
  remove(a.get("class-cache").c_str());
  int L = (int)a.getl("loops", 2);
  double N = a.getd("N", 1.0);
  std::vector<ClassRun> cr;
  std::vector<std::pair<std::string, std::string>> dropped;
  if (!build_all(a, kind, L, N, cr, dropped, true)) return 2;
  size_t img = 0; int maxat = 0; double bs = 0;
  for (auto &c : cr) {
    img += packed_bytes(c.cd);
    maxat = std::max(maxat, c.cd.natoms);
    bs += c.build_s;
  }
  printf("build: %zu classes (%zu dropped), %.1f class-seconds, max atoms %d, device images %.2f "
    "GB -> %s\n", cr.size(), dropped.size(), bs, maxat, img / 1073741824.0,
    a.get("class-cache").c_str());
  return 0;
}

static int cmd_info(const Args &a) {
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  int L = (int)a.getl("loops", 2);
  double N = a.getd("N", 1.0);
  std::vector<ClassRun> cr;
  std::vector<std::pair<std::string, std::string>> dropped;
  if (!build_all(a, kind, L, N, cr, dropped, true)) return 2;
  {
    std::vector<RbStreamId> ids;
    std::string err;
    if (!assign_streams(a, kind, L, (uint64_t)a.getl("seed", 12345), cr, &ids, &err)) {
      fprintf(stderr, "error: %s\n", err.c_str());
      return 2;
    }
  }
  for (size_t i = 0; i < cr.size(); ++i) {
    auto &d = cr[i].cd;
    printf("%3zu %-28s S=%-9.6g SN=%-9.6g 1VR=%d parts=%-3d forests=%-5ld raw=%-6ld comp=%-5d "
      "atoms=%-4d maxloop=%d shift=%d omin=%.1f build=%.2fs\n", i, d.G.name.c_str(), d.S, cr[i].SN,
      (int)d.is1vr, d.nparts, d.nforests, d.nterms_raw, d.ncomp, d.natoms, d.max_atom_loops,
      d.nshift, d.omega_min, cr[i].build_s);
  }
  for (auto &d : dropped) printf("dropped %-28s %s\n", d.first.c_str(), d.second.c_str());
  // --atoms: every atom (a 2-connected block of a minor) as "ATOM <class> <nv> <u-v ...>" in local
  // vertices
  if (a.has("atoms")) {
    for (size_t i = 0; i < cr.size(); ++i)
      for (auto &A : cr[i].cd.atoms) {
        printf("ATOM %zu %d", i, A.nv);
        for (size_t k = 0; k < A.u.size(); ++k) printf(" %d-%d", A.u[k], A.v[k]);
        printf("\n");
      }
  }
  if (a.has("out")) write_run_json(a.get("out"), a, kind, L, N, 0, cr, dropped, 0, true);
  return 0;
}

// Backends.
static BackendOptions backend_options(const Args &a) {
  BackendOptions o;
  o.kind = a.get("backend", RB_DEFAULT_BACKEND);
  o.devices = a.get("devices", "");
  o.launch_s = a.getd("launch-s", o.launch_s);
  o.max_launch_chunks = a.getl("max-launch-chunks", o.max_launch_chunks);
  o.selftest_classes = (int)a.getl("selftest-classes", o.selftest_classes);
  o.selftest_tol = a.getd("selftest-tol", o.selftest_tol);
  o.selftest = !a.has("no-selftest");
  o.mem_frac = a.getd("mem-frac", o.mem_frac);
  o.hang_s = a.getd("hang-s", o.hang_s);
  o.emu_devices = (int)a.getl("emu-devices", o.emu_devices);
  o.emu_blocks = (int)a.getl("emu-blocks", o.emu_blocks);
  o.emu_reverse = a.has("emu-reverse");
  o.emu_fail_device = (int)a.getl("emu-fail-device", o.emu_fail_device);
  o.emu_fail_after = (int)a.getl("emu-fail-after", o.emu_fail_after);
  o.emu_mem = a.getd("emu-mem", o.emu_mem);
  o.emu_fail_mode = (int)a.getl("emu-fail-mode", o.emu_fail_mode);
  o.fp32 = a.has("fp32");
  return o;
}

// Philox streams: per-class keys (stream scheme "salted-v1") or the legacy (seed, kind << 26 |
// index) scheme.
static uint64_t fnv1a64(const std::string &x) {
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : x) { h ^= c; h *= 1099511628211ULL; }
  return h;
}
static uint64_t class_salt(Kind kind, int L, const std::string &nickel) {
  return fnv1a64(std::string(kind_name(kind)) + "|" + std::to_string(L) + "|" + nickel);
}
static bool assign_streams(const Args &a, Kind kind, int L, uint64_t seed,
    std::vector<ClassRun> &cr, std::vector<RbStreamId> *ids, std::string *err) {
  const std::string scheme = a.get("streams", "salted-v1");
  const uint32_t kbase = (uint32_t)kind << 26;
  ids->clear();
  std::set<std::pair<uint64_t, uint32_t>> seen;
  for (size_t i = 0; i < cr.size(); ++i) {
    ClassRun &c = cr[i];
    c.salt = class_salt(kind, L, c.cd.G.name);
    if (scheme == "legacy") c.id = RbStreamId{seed, kbase | (uint32_t)i};
    else if (scheme == "salted-v1") c.id = RbStreamId{rb_class_key(seed, c.salt), kbase};
    else { *err = "unknown --streams " + scheme + " (salted-v1 | legacy)"; return false; }
    if (!seen.insert({c.id.key, c.id.stream}).second) {
      *err = "Philox key collision for class " + c.cd.G.name + " (duplicate graph in the list?)";
      return false;
    }
    ids->push_back(c.id);
  }
  return true;
}

// Create and prepare the backend for the given classes.  auto: CUDA if a device passes its
// self-test, else CPU -- but only when there is no GPU at all; when GPUs exist and every one fails
// its self-test, auto refuses to run on the CPU unless --allow-cpu-fallback is given (a GPU-sized
// run would otherwise crawl on the host for days).
// On failure returns nullptr; *failed then holds the backend whose devices failed (for its
// diagnostics).
static std::unique_ptr<Backend> open_backend(const Args &a, const std::vector<ClassRun> &cr,
    const std::vector<RbStreamId> &ids, std::string *note,
    std::unique_ptr<Backend> *failed = nullptr) {
  BackendOptions o = backend_options(a);
  const bool autob = o.kind == "auto";
  if (autob) o.kind = "cuda";
  std::vector<const ClassData *> ptrs;
  for (auto &c : cr) ptrs.push_back(&c.cd);
  std::unique_ptr<Backend> be;
  std::string err;
  if (o.kind != "cpu") {
    be = make_multi_backend(o, &err);
    if (be && !be->prepare(ptrs, ids, &err)) {
      if (!autob || !a.has("allow-cpu-fallback")) {
        if (autob)
          fprintf(stderr, "error: CUDA device(s) present but none is usable (%s); not falling back "
            "to the CPU without --allow-cpu-fallback\n", err.c_str());
        else fprintf(stderr, "error: %s\n", err.c_str());
        if (failed) *failed = std::move(be);
        return nullptr;
      }
      *note = "auto: CUDA backend not usable (" + err
          + "); running on the CPU (--allow-cpu-fallback)";
      be.reset();
    } else if (!be) {
      if (!autob) { fprintf(stderr, "error: %s\n", err.c_str()); return nullptr; }
      *note = "auto: no usable CUDA device (" + err + "); running on the CPU";
    }
    if (!be) fprintf(stderr, "WARNING: %s\n", note->c_str());
  }
  if (!be) {
    be = make_cpu_backend(o.fp32);
    be->prepare(ptrs, ids, &err);
  }
  fprintf(stderr, "backend: %s%s\n", be->name().c_str(),
    o.fp32 ? " (FP32 evaluation, FP64 accumulation)" : "");
  return be;
}

// --fp32 is meant for the bare-route kinds only (all-positive or t0-subtracted integrands); the
// R-bar kinds cancel strongly near sunset faces and need FP64
static bool check_fp32(const Args &a, Kind kind) {
  if (!a.has("fp32")) return true;
  if (kind == Kind::B || kind == Kind::H || kind == Kind::C) {
    if (a.has("force-fp32")) {
      fprintf(stderr, "WARNING: --fp32 with an R-bar kind (forced): results are NOT reliable\n");
      return true;
    }
    fprintf(stderr, "error: --fp32 is for the bare kinds (bare4, int4, bare2d, int2d) only; use "
      "--force-fp32 to override\n");
    return false;
  }
  return true;
}

// SIGINT / SIGTERM: the first one asks the run to finish its launches and write the JSON; a second
// one leaves immediately (a thread may be blocked in a CUDA call that never returns).
// Async-signal-safe.
static void on_signal(int) {
  if (g_rb_stop.exchange(true)) _Exit(130);
}

// Checkpoint / resume.
// Options that may change between the sessions of one run: they decide where and for how long it
// runs, never which samples are drawn or how they are merged.  They are not inherited from the
// checkpoint (each session gives its own, or gets the defaults).  Every other option is inherited;
// giving one with a different value is an error (a resumed run is the same computation, continued).
static bool session_key(const std::string &k) {
  static const std::set<std::string> s = {"backend", "devices", "seconds", "out", "threads",
    "launch-s", "max-launch-chunks", "class-cache", "mem-frac", "hang-s", "selftest-classes",
    "selftest-tol", "no-selftest", "allow-cpu-fallback", "emu-devices", "emu-blocks", "emu-reverse",
    "emu-mem", "emu-fail-device", "emu-fail-after", "emu-fail-mode", "resume", "extend", "round-s",
    "verbose", "test-stop-after-units", "test-kill-after-units"};
  return s.count(k) > 0;
}
// Never inherited: one-shot actions and test hooks
static bool one_shot_key(const std::string &k) {
  return k == "resume" || k == "extend" || k.rfind("test-", 0) == 0;
}

static bool merge_resume_args(const Args &cl, const JV &rec, Args *out, std::string *err) {
  const JV *ra = rec.get("args");
  if (!ra || ra->t != JV::OBJ) { *err = "the file has no \"args\" object"; return false; }
  Args m;
  m.cmd = cl.cmd;
  // Inherited: everything that defines the computation; NOT inherited: the session options (where
  // and how long the earlier session ran: --threads is applied in main() from the command line,
  // --seconds was that session's budget)
  for (auto &p : ra->o)
    if (!one_shot_key(p.first) && !session_key(p.first) && p.second.t == JV::STR)
      m.kv[p.first] = p.second.s;
  const bool ext = cl.has("extend");
  for (auto &p : cl.kv) {
    const std::string &k = p.first;
    // A new plan, with --extend only
    const bool plan_key = k == "samples" || k == "rounds" || k == "alloc";
    if (session_key(k) || (ext && plan_key)) { m.kv[k] = p.second; continue; }
    auto it = m.kv.find(k);
    if (it == m.kv.end() || it->second != p.second) {
      *err = "--" + k + " " + p.second + " differs from the run being resumed ("
          + (it == m.kv.end() ? std::string("not set") : "--" + k + " " + it->second) + "); a "
          "resume inherits every option except where and how long it runs (--backend, --devices, "
          "--seconds, --out, ...)"
          + (plan_key ? "; a new --samples / --rounds / --alloc needs --extend" : "");
      return false;
    }
  }
  if (ext) { // a new budget replaces the old one
    if (cl.has("samples") && !cl.has("seconds")) m.kv.erase("seconds");
    if (cl.has("seconds") && !cl.has("samples")) m.kv.erase("samples");
  }
  if (!cl.has("out")) m.kv["out"] = cl.get("resume");
  *out = m;
  return true;
}

static bool restore_checkpoint(const JV &rec, std::vector<ClassRun> &cr, RunState *rs,
    std::string *err) {
  const JV *ck = rec.get("checkpoint");
  if (!ck || ck->t != JV::OBJ) {
    *err = "no \"checkpoint\" object (a run JSON of rebar >= 0.4.0 is needed)";
    return false;
  }
  if (ck->str("format") != "rbar-checkpoint-1") {
    *err = "unknown checkpoint format " + ck->str("format");
    return false;
  }
  rs->complete = ck->boolean("complete");
  rs->pilot_done = ck->boolean("pilot_done");
  rs->round_done = (int)ck->num("round_done");
  rs->rounds = (int)ck->num("rounds");
  rs->npilot = (uint64_t)ck->num("npilot");
  rs->alloc = ck->str("alloc");
  rs->extensions = (int)ck->num("extensions");
  if (!unhex_words(ck->str("total"), &rs->total, 8)
      || !unhex_words(ck->str("wall_s_previous"), &rs->wall_prev, 8)) {
    *err = "bad checkpoint header";
    return false;
  }
  if (const JV *ss = ck->get("sessions"))
    for (auto &x : ss->a) {
      RunSession q;
      q.utc = x.str("utc"); q.program = x.str("program"); q.version = x.str("version");
      q.backend = x.str("backend");
      q.resumed_from = x.str("resumed_from"); q.extended = x.boolean("extended");
      q.wall_s = x.num("wall_s"); q.samples = x.num("samples");
      q.round_from = (int)x.num("round_from"); q.round_to = (int)x.num("round_to");
      rs->sessions.push_back(q);
    }
  // The wall time of the earlier sessions is the sum of what each recorded at its last save
  // (wall_s_previous, written by the saving session, holds the sessions before *it* and would drop
  // the last one)
  rs->wall_prev = 0;
  for (auto &q : rs->sessions) rs->wall_prev += q.wall_s;
  const JV *cl = ck->get("classes");
  if (!cl || cl->t != JV::ARR) { *err = "checkpoint without classes"; return false; }
  if (cl->a.size() != cr.size()) {
    *err = "the checkpoint has " + std::to_string(cl->a.size()) + " classes, this build "
        + std::to_string(cr.size()) + " (other list or options?)";
    return false;
  }
  for (size_t i = 0; i < cr.size(); ++i) {
    const JV &r = cl->a[i];
    ClassRun &c = cr[i];
    if (r.t != JV::ARR || r.a.size() != 11) {
      *err = "bad checkpoint row " + std::to_string(i);
      return false;
    }
    if (r.a[0].s != c.cd.G.name || r.a[1].s != hex64(c.salt)) {
      *err = "class " + std::to_string(i) + " is " + c.cd.G.name + " here but " + r.a[0].s
          + " in the checkpoint";
      return false;
    }
    if ((int)r.a[2].n != c.cd.ncomp || (int)r.a[3].n != c.cd.natoms) {
      *err = "class " + c.cd.G.name + ": the builder gives " + std::to_string(c.cd.ncomp)
          + " terms / " + std::to_string(c.cd.natoms) + " atoms, the checkpoint "
          + std::to_string((int)r.a[2].n) + " / " + std::to_string((int)r.a[3].n)
          + " (other code version?)";
      return false;
    }
    c.next = (uint64_t)r.a[4].n; c.part_done = (uint64_t)r.a[5].n; c.base = (uint64_t)r.a[6].n;
    c.nalloc = (uint64_t)r.a[7].n;
    if (!unhex_words(r.a[8].s, &c.cpu, 8) || !unhex_words(r.a[9].s, &c.acc, sizeof(RbAcc))
        || !unhex_words(r.a[10].s, &c.part, sizeof(RbAcc))) {
      *err = "bad accumulator in checkpoint row " + std::to_string(i);
      return false;
    }
  }
  return true;
}

// Neyman weights of the classes from their current statistics (the allocation of the main phase)
static std::vector<double> alloc_weights(const std::vector<ClassRun> &cr,
    const std::string &alloc) {
  std::vector<double> wgt(cr.size());
  for (size_t i = 0; i < cr.size(); ++i) {
    const RbAcc acc = cr[i].shown();
    double m, sd; acc_stats(acc, &m, &sd);
    double c = acc.n > 0 ? cr[i].cpu / acc.n : 0;
    double s = std::fabs(cr[i].SN * cr[i].cd.pref) * (std::isfinite(sd) ? sd : 0);
    wgt[i] = alloc == "equal" ? 1.0 : alloc == "time" ? (c > 0 ? s / std::sqrt(c) : s) : s;
  }
  double wsum = 0; for (double w : wgt) wsum += w;
  if (!(wsum > 0) || !std::isfinite(wsum)) for (double &w : wgt) w = 1.0; // zero-variance classes
  return wgt;
}
// Rounds of a plan: --rounds (default 8), raised so that a round holds <= 2e6 units of work (host
// memory of the multi-device scheduler) and, in a time-boxed plan, lasts <= --round-s (default 1800
// s: a checkpoint every 30 min)
static int plan_rounds(const Args &a, const std::vector<ClassRun> &cr, double main_s) {
  int rounds = (int)a.getl("rounds", 8);
  double units = 0;
  for (auto &c : cr)
    units += std::ceil((double)(c.nalloc > c.base ? c.nalloc - c.base : 0) / (double)RB_UNIT);
  const int need = (int)std::min(1e6, std::ceil(units / 2e6));
  if (need > rounds) {
    fprintf(stderr, "rounds raised from %d to %d (%.3g units of work)\n", rounds, need, units);
    rounds = need;
  }
  if (main_s > 0) {
    const double rsec = a.getd("round-s", 1800);
    const int nt = (int)std::min(1e5, std::ceil(main_s / std::max(1.0, rsec)));
    if (nt > rounds) {
      fprintf(stderr, "rounds raised from %d to %d (rounds of <= %g s: checkpoints)\n", rounds, nt,
        rsec);
      rounds = nt;
    }
  }
  return std::max(1, rounds);
}

static int cmd_run(const Args &a0) {
  // --resume FILE: continue the run whose last JSON is FILE (its options, plan, accumulators and
  // Philox counters)
  Args a = a0;
  JV rec;
  const bool resume = a0.has("resume");
  if (resume) {
    std::string err;
    if (!json_read_file(a0.get("resume"), &rec, &err,
        {"classes", "backend_info", "dropped", "warnings", "order"})) {
      fprintf(stderr, "error: %s\n", err.c_str());
      return 2;
    }
    if (!merge_resume_args(a0, rec, &a, &err)) {
      fprintf(stderr, "error: --resume %s: %s\n", a0.get("resume").c_str(), err.c_str());
      return 2;
    }
    if (rec.str("version") != RB_VERSION)
      fprintf(stderr, "WARNING: resuming a run of %s %s with %s %s\n", rec.str("program").c_str(),
        rec.str("version").c_str(), RB_PROGRAM, RB_VERSION);
    if (int rc = precheck_limits(a)) return rc; // the list comes from the checkpoint
  }
  if (a.has("test-stop-after-units")) g_rb_test_stop_units = a.getl("test-stop-after-units", -1);
  if (a.has("test-kill-after-units")) g_rb_test_kill_units = a.getl("test-kill-after-units", -1);
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  int L = (int)a.getl("loops", 2);
  double N = a.getd("N", 1.0);
  uint64_t seed = (uint64_t)a.getl("seed", 12345);
  std::string out = a.get("out",
    std::string("rebar_") + kind_name(kind) + "_L" + std::to_string(L) + ".json");
  double t_start = now_s();
  double deadline = a.has("seconds") ? t_start + a.getd("seconds", 0) : 1e300;
  if (!check_fp32(a, kind)) return 1;
  std::vector<ClassRun> cr;
  std::vector<std::pair<std::string, std::string>> dropped;
  if (!build_all(a, kind, L, N, cr, dropped, true)) return 2;
  fprintf(stderr, "[%s L=%d N=%g] %zu classes, %zu dropped; build %.1fs\n", kind_name(kind), L, N,
    cr.size(), dropped.size(), now_s() - t_start);
  if (cr.empty()) { fprintf(stderr, "no classes\n"); return 2; }
  std::vector<RbStreamId> ids;
  {
    std::string err;
    if (!assign_streams(a, kind, L, seed, cr, &ids, &err)) {
      fprintf(stderr, "error: %s\n", err.c_str());
      return 2;
    }
  }
  RunState rs;
  if (resume) {
    std::string err;
    if (!restore_checkpoint(rec, cr, &rs, &err)) {
      fprintf(stderr, "error: --resume %s: %s\n", a0.get("resume").c_str(), err.c_str());
      return 2;
    }
    rec = JV();
    uint64_t done = 0;
    for (auto &c : cr) done += c.next + c.part_done;
    fprintf(stderr,
      "resuming %s: %s, %d/%d rounds done, %.4g samples so far, %zu earlier session(s)\n",
      a0.get("resume").c_str(), rs.pilot_done ? "pilot done" : "pilot incomplete", rs.round_done,
      rs.rounds, (double)done, rs.sessions.size());
    if (rs.complete && !a.has("extend")) {
      fprintf(stderr, "nothing to do: the run is complete (add samples with --extend --samples N "
        "or --extend --seconds T)\n");
      return 0;
    }
  } else {
    rs.npilot = (uint64_t)a.getl("pilot", 20000);
    rs.alloc = a.get("alloc", a.has("seconds") ? "time" : "samples");
    for (auto &c : cr) c.base = rs.npilot;
  }
  const uint64_t npilot = rs.npilot;
  {
    RunSession me;
    me.utc = utc_now(); me.program = RB_PROGRAM; me.version = RB_VERSION;
    me.resumed_from = resume ? a0.get("resume") : ""; me.extended = a.has("extend");
    me.round_from = rs.round_done;
    rs.sessions.push_back(me);
  }
  std::string note;
  std::unique_ptr<Backend> failed;
  std::unique_ptr<Backend> be = open_backend(a, cr, ids, &note, &failed);
  if (!be) {
    if (failed && !resume)
      write_run_json(out, a, kind, L, N, seed, cr, dropped, now_s() - t_start, true, failed.get(),
        "no usable device; nothing was run", &rs);
    return 3;
  }
  rs.sessions.back().backend = be->name();
  double session_samples = 0;
  auto wall = [&]() { return rs.wall_prev + (now_s() - t_start); };
  auto save = [&](bool final_) {
    RunSession &me = rs.sessions.back();
    me.wall_s = now_s() - t_start; me.samples = session_samples; me.round_to = rs.round_done;
    me.backend = be->name();
    rs.complete = rs.pilot_done && rs.rounds > 0 && rs.round_done >= rs.rounds;
    write_run_json(out, a, kind, L, N, seed, cr, dropped, wall(), final_, be.get(), note, &rs);
  };
  // A task that ran to its end is committed into acc; one cut short keeps its prefix accumulator
  // apart (part)
  auto apply = [&](const std::vector<SampleTask> &tasks, const std::vector<SampleOut> &res) {
    bool complete = true;
    for (size_t t = 0; t < tasks.size(); ++t) {
      ClassRun &c = cr[tasks[t].cls];
      c.cpu += res[t].cost_s;
      session_samples += (double)res[t].done;
      if (res[t].done == tasks[t].n) {
        rb_acc_merge(&c.acc, &res[t].acc);
        c.next += c.part_done + res[t].done;
        c.part = RbAcc{};
        c.part_done = 0;
      } else {
        c.part = res[t].acc;
        c.part_done += res[t].done;
        complete = false;
      }
    }
    return complete;
  };
  // Pilot (not time-boxed: it is needed for the allocation)
  if (!rs.pilot_done) {
    std::vector<SampleTask> tasks;
    for (size_t i = 0; i < cr.size(); ++i) {
      // A class whose pilot task is already committed has next = npilot
      const uint64_t start = cr[i].next + cr[i].part_done;
      if (start < npilot) tasks.push_back(SampleTask{(int)i, start, npilot - start, cr[i].part});
    }
    std::vector<SampleOut> res;
    be->run(tasks, 1e300, res);
    rs.pilot_done = apply(tasks, res);
    save(!rs.pilot_done);
    OrderSummary o = summarise(cr);
    fprintf(stderr,
      "pilot%s: sum %.10g +- %.3g  rho_neyman %.4g  rho_time %.4g sqrt(%s)  (%.1fs)\n",
      rs.pilot_done ? "" : " (INCOMPLETE)", o.sum, o.err, o.rho_neyman, o.rho_time, be->cost_unit(),
      now_s() - t_start);
    be->round_report(stderr);
    if (!rs.pilot_done) {
      fprintf(stderr, "interrupted during the pilot; continue with: run --resume %s\n",
        out.c_str());
      return 130;
    }
  }
  // Allocation (once per plan): Neyman by samples, by measured cost (time) or equal; then the
  // rounds of the plan
  if (rs.rounds == 0 || a.has("extend")) {
    const bool ext = a.has("extend");
    if (ext) {
      // Fold every interrupted task into the committed state (statistically exact; the old plan is
      // abandoned)
      for (auto &c : cr)
        if (c.part_done) {
          rb_acc_merge(&c.acc, &c.part);
          c.next += c.part_done;
          c.part = RbAcc{};
          c.part_done = 0;
        }
      if (a0.has("alloc")) rs.alloc = a0.get("alloc");
      if (!a.has("samples") && !a.has("seconds")) {
        fprintf(stderr, "error: --extend needs a budget: --samples N or --seconds T\n");
        return 2;
      }
    }
    std::vector<double> wgt = alloc_weights(cr, rs.alloc);
    double wsum = 0; for (double w : wgt) wsum += w;
    double total = 0, main_s = 0;
    if (a.has("seconds")) {
      // Cost-seconds available for the main phase (core-s or device-s), divided by the expected
      // cost per sample
      main_s = std::max(0.0, deadline - now_s()) * 0.9;
      double avail = main_s * be->capacity();
      double cost_per_unit = 0;
      for (size_t i = 0; i < cr.size(); ++i) {
        const RbAcc acc = cr[i].shown();
        cost_per_unit += wgt[i] / wsum * (acc.n > 0 ? cr[i].cpu / acc.n : 0);
      }
      total = cost_per_unit > 0 ? avail / cost_per_unit : 0;
    } else total = (double)a.getl("samples", 0);
    if (!ext) {
      for (size_t i = 0; i < cr.size(); ++i) {
        double x = total * wgt[i] / wsum;
        cr[i].nalloc = std::max<uint64_t>(npilot,
          std::isfinite(x) && x > 0 ? (uint64_t)std::min(x, 1e15) : 0);
        cr[i].base = npilot;
      }
    } else {
      // Top-up: the target totals (done + budget) follow the weights; classes already above their
      // target get nothing
      double done = 0;
      for (auto &c : cr) done += (double)c.next;
      std::vector<double> extra(cr.size(), 0);
      double esum = 0;
      for (size_t i = 0; i < cr.size(); ++i) {
        extra[i] = std::max(0.0, (done + total) * wgt[i] / wsum - (double)cr[i].next);
        esum += extra[i];
      }
      for (size_t i = 0; i < cr.size(); ++i) {
        const double x = esum > 0 ? total * extra[i] / esum : 0;
        cr[i].base = cr[i].next;
        cr[i].nalloc = cr[i].next + (std::isfinite(x) && x > 0 ? (uint64_t)std::min(x, 1e15) : 0);
      }
      rs.extensions++;
      fprintf(stderr, "extension %d: %.4g more samples (%s allocation)\n", rs.extensions, total,
        rs.alloc.c_str());
    }
    rs.total = total;
    rs.rounds = plan_rounds(a, cr, main_s);
    rs.round_done = 0;
    rs.complete = false;
    rs.sessions.back().round_from = 0;
    save(false); // the plan is part of the checkpoint
  }
  // Main phase in rounds (checkpointed after each); the backend stops at the deadline (a round cut
  // short is resumed exactly).  The multi-device scheduler materialises a round's work as units of
  // RB_UNIT samples.
  const int rounds = rs.rounds;
  bool stopped = false;
  for (int r = rs.round_done + 1; r <= rounds && !g_rb_stop; ++r) {
    double tr0 = now_s();
    std::vector<SampleTask> tasks;
    for (size_t i = 0; i < cr.size(); ++i) {
      const ClassRun &c = cr[i];
      const uint64_t target = c.base + (uint64_t)((double)(c.nalloc - c.base) * r / rounds);
      const uint64_t start = c.next + c.part_done;
      if (target > start) tasks.push_back(SampleTask{(int)i, start, target - start, c.part});
    }
    bool complete = true;
    if (!tasks.empty()) {
      std::vector<SampleOut> res;
      be->run(tasks, deadline, res);
      complete = apply(tasks, res);
    }
    if (complete) rs.round_done = r;
    OrderSummary o = summarise(cr);
    fprintf(stderr,
      "round %d/%d%s: sum %.10g +- %.3g  rho_neyman %.4g  nbad %.0f  (%.1fs, round %.1fs)\n", r,
      rounds, complete ? "" : " (cut short)", o.sum, o.err, o.rho_neyman, o.nbad, now_s() - t_start,
      now_s() - tr0);
    be->round_report(stderr);
    const bool stop = !complete || now_s() > deadline || g_rb_stop;
    save(r == rounds || stop);
    if (stop) {
      if (rs.round_done < rounds)
        fprintf(stderr, "%s in round %d of %d; continue with: run --resume %s\n",
          g_rb_stop ? "interrupted" : "time budget reached", r, rounds, out.c_str());
      stopped = true;
      break;
    }
  }
  if (g_rb_stop && !stopped) { // the signal arrived between two rounds (or after the last one)
    if (rs.round_done < rounds)
      fprintf(stderr, "interrupted after round %d of %d; continue with: run --resume %s\n",
        rs.round_done, rounds, out.c_str());
    save(true);
  }
  OrderSummary o = summarise(cr);
  for (auto &w : run_warnings(o, note)) fprintf(stderr, "WARNING: %s\n", w.c_str());
  double rv, re;
  printf("%s L=%d N=%g: sum = %.10g +- %.3g (error of the error %.2g%%)", kind_name(kind), L, N,
    o.sum, o.err, 100 * o.err_relerr);
  if (kind == Kind::INT4)
    printf("  [S_L = %.10g +- %.3g]", std::pow(2.0, L) / 3 * o.sum, std::pow(2.0, L) / 3 * o.err);
  if (find_ref(kind_name(kind), N, L, &rv, &re)) {
    double val = kind == Kind::INT4 ? std::pow(2.0, L) / 3 * o.sum : o.sum,
        er = kind == Kind::INT4 ? std::pow(2.0, L) / 3 * o.err : o.err;
    printf("  ref %.10g +- %.2g  pull %+.2f", rv, re, (val - rv) / std::sqrt(er * er + re * re));
  }
  printf("\n"
    "  backend %s  rho_neyman %.4g  rho_time %.4g sqrt(%s)  sum|c|/|sum| %.3g  samples %.3g  nbad "
    "%.0f  %s %.1f  wall %.1fs%s -> %s\n", be->name().c_str(), o.rho_neyman, o.rho_time,
    be->cost_unit(), o.sumabs / std::fabs(o.sum), o.n, o.nbad, be->cost_unit(), o.cpu, wall(),
    rs.complete ? "" : " [run not complete: resumable]", out.c_str());
  return g_rb_stop ? 130 : 0;
}

static int cmd_bench(const Args &a) {
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  int L = (int)a.getl("loops", 2);
  uint64_t n = (uint64_t)a.getl("n", 20000);
  std::vector<ClassRun> cr;
  std::vector<std::pair<std::string, std::string>> dropped;
  if (!build_all(a, kind, L, 1.0, cr, dropped, true)) return 2;
  double tot = 0;
  for (size_t i = 0; i < cr.size(); ++i) {
    RbClassView v = cr[i].cd.view();
    std::vector<double> logP(std::max(1, cr[i].cd.natoms));
    double t0 = thread_cpu_s(), acc = 0, sabs;
    for (uint64_t k = 0; k < n; ++k)
      acc += rb_sample<double>(v, 7, (uint32_t)i, k, logP.data(), &sabs);
    double dt = (thread_cpu_s() - t0) / n;
    tot += dt;
    printf("%3zu %-28s comps=%-5d groups=%-4d atoms=%-4d %8.0f ns/sample (single thread)%s\n", i,
      cr[i].cd.G.name.c_str(), cr[i].cd.ncomp, cr[i].cd.ngroup, cr[i].cd.natoms, 1e9 * dt,
      acc == 1.2345 ? "?" : "");
  }
  printf("mean %.0f ns/sample over %zu classes\n", 1e9 * tot / cr.size(), cr.size());
  return 0;
}

// Devices / selftest.
static int cmd_devices(const Args &a) {
  CudaProbe p = cuda_probe();
  JsonW J;
  J.begin_obj();
  J.kv("program", RB_PROGRAM); J.kv("version", RB_VERSION); J.kv("utc", utc_now());
  J.kvb("compiled_with_cuda", p.compiled); J.kvi("device_count", p.count);
  J.kvi("driver_version", p.driver); J.kvi("runtime_version", p.runtime); J.kv("error", p.err);
  J.kv("build_signature_host", hex64(rb_build_signature()));
  J.key("devices"); J.begin_arr();
  BackendOptions o = backend_options(a);
  int nok = 0;
  for (int d = 0; d < p.count && !g_rb_stop; ++d) {
    std::unique_ptr<Device> dev = make_cuda_device(d, o);
    J.begin_obj();
    std::string err;
    bool ok = dev && dev->bind(&err) && dev->open(1, 1, &err);
    J.kvb("opened", ok); J.kv("error", err);
    if (dev) { J.key("properties"); dev->info_json(J); }
    if (ok) {
      std::vector<double> fp; std::vector<uint32_t> kat; uint64_t sig = 0;
      if (dev->fptest(fp, kat, &sig, &err)) {
        int kbad = 0; for (int i = 0; i < 4 * RB_NKAT; ++i) kbad += kat[i] != rb_kat_expected[i];
        J.kvi("philox_kat_mismatches", kbad);
        J.kv("build_signature_device", hex64(sig));
        J.kvb("build_signature_match", sig == rb_build_signature());
        nok += kbad == 0 && sig == rb_build_signature();
      } else J.kv("fptest_error", err);
    }
    J.end_obj();
    if (g_rb_hung) break;
  }
  J.end_arr();
  J.kvi("devices_ok", nok);
  J.end_obj();
  if (a.has("out")) J.write(a.get("out"));
  printf("%s\n", J.s.c_str());
  return p.count > 0 && nok == p.count ? 0 : 1;
}

// Start-up diagnostics only: build the classes, open the backend (self-tests run in prepare), write
// the JSON.
static int cmd_selftest(const Args &a0) {
  Args a = a0;
  if (!a.has("kind")) a.kv["kind"] = "b";
  if (!a.has("loops")) a.kv["loops"] = "4";
  if (!a.has("list")) a.kv["gen"] = "1";
  if (!a.has("backend")) a.kv["backend"] = "auto";
  Kind kind;
  if (!kind_from_string(a.get("kind"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  int L = (int)a.getl("loops", 4);
  uint64_t seed = (uint64_t)a.getl("seed", 2026);
  std::vector<ClassRun> cr;
  std::vector<std::pair<std::string, std::string>> dropped;
  if (!build_all(a, kind, L, 1.0, cr, dropped, true) || cr.empty()) return 2;
  std::vector<RbStreamId> ids;
  std::string err;
  if (!assign_streams(a, kind, L, seed, cr, &ids, &err)) {
    fprintf(stderr, "error: %s\n", err.c_str());
    return 2;
  }
  std::string note;
  std::unique_ptr<Backend> failed;
  std::unique_ptr<Backend> be = open_backend(a, cr, ids, &note, &failed);
  const std::string fn = a.get("out", "selftest.json");
  if (!be) {
    if (failed) {
      write_run_json(fn, a, kind, L, 1.0, seed, cr, dropped, 0, true, failed.get(),
        "self-test FAILED on every device");
      JsonW J; failed->info_json(J);
      printf("self-test FAILED\n%s\n", J.s.c_str());
    }
    return 3;
  }
  write_run_json(fn, a, kind, L, 1.0, seed, cr, dropped, 0, true, be.get(), note);
  JsonW J; be->info_json(J);
  printf("backend %s%s%s\n%s\n", be->name().c_str(), note.empty() ? "" : ": ", note.c_str(),
    J.s.c_str());
  return note.empty() ? 0 : 4;
}

// GPU-emulation unit test: every backend configuration must reproduce the CPU accumulators bit for
// bit.
static bool acc_equal(const RbAcc &x, const RbAcc &y) { return memcmp(&x, &y, sizeof(RbAcc)) == 0; }
// true if some occurrence of "key": in the JSON text has a value that does not start with 0 (or
// false)
static bool json_any_nonzero(const std::string &js, const std::string &key) {
  const std::string k = "\"" + key + "\":";
  for (size_t p = js.find(k); p != std::string::npos; p = js.find(k, p + 1)) {
    const char c = js[p + k.size()];
    if (c != '0' && c != 'f' && c != 'n') return true;
  }
  return false;
}
static int cmd_check_emu(const Args &a) {
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  int L = (int)a.getl("loops", 3);
  uint64_t seed = (uint64_t)a.getl("seed", 777);
  const bool fp32 = a.has("fp32");
  std::vector<ClassRun> cr;
  std::vector<std::pair<std::string, std::string>> dropped;
  Args ab = a;
  if (!ab.has("list")) ab.kv["gen"] = "1";
  if (!build_all(ab, kind, L, 1.0, cr, dropped, true) || cr.empty()) return 2;
  std::vector<RbStreamId> ids;
  {
    std::string err;
    if (!assign_streams(a, kind, L, seed, cr, &ids, &err)) {
      fprintf(stderr, "error: %s\n", err.c_str());
      return 2;
    }
  }
  printf("# emulation test: kind %s L=%d, %zu classes, streams %s, precision %s, omega-scale %g\n",
    kind_name(kind), L, cr.size(), a.get("streams", "salted-v1").c_str(), fp32 ? "fp32" : "fp64",
    a.getd("omega-scale", 0.7));
  // Enough units (>= 12 in total) for the fault-injection configurations to reach their failing
  // launch
  const uint64_t units_per_task = std::max<uint64_t>(2, (12 + cr.size() - 1) / cr.size());
  uint64_t base_n = (uint64_t)a.getl("n", units_per_task * RB_UNIT + 3 * RB_CHUNK + 1234);
  if (cr.size() * ((base_n + RB_UNIT - 1) / RB_UNIT) < 12) {
    base_n = units_per_task * RB_UNIT + 1234;
    printf("(--n raised to %llu so that the fault-injection configurations reach their failing "
      "launch)\n", (unsigned long long)base_n);
  }
  int fails = 0;
  // tasks: odd offsets and lengths (partial chunks, several units per task)
  std::vector<SampleTask> tasks;
  for (size_t i = 0; i < cr.size(); ++i)
    tasks.push_back(SampleTask{(int)i, 1000 * (uint64_t)i + 7, base_n + 17 * (uint64_t)i});
  // (0) per-sample: rb_sample (compact scratch) == rb_sample_dbg (interleaved scratch), bitwise
  {
    long long bad = 0, tot = 0;
    for (size_t i = 0; i < cr.size(); ++i) {
      const RbClassView v = cr[i].cd.view();
      const int nl = 37;
      std::vector<double> lp(std::max(1, cr[i].cd.natoms)),
          li((size_t)nl * std::max(1, cr[i].cd.natoms), NAN);
      for (uint64_t k = 0; k < 256; ++k) {
        double sa;
        const double w = rb_sample<double>(v, ids[i].key, ids[i].stream, k, lp.data(), &sa);
        RbSampleDbg d;
        rb_sample_dbg(v, ids[i].key, ids[i].stream, k, li.data() + (k % nl), nl, &d);
        bad += !(memcmp(&w, &d.w, 8) == 0 && memcmp(&sa, &d.sabs, 8) == 0);
        ++tot;
      }
    }
    printf("%-58s %lld/%lld samples differ  %s\n", "per-sample: compact vs interleaved scratch",
      bad, tot, bad ? "FAIL" : "PASS");
    fails += bad != 0;
  }
  // (1) reference: CPU backend with the current thread count, then with 1 thread
  std::vector<RbAcc> ref(tasks.size());
  for (size_t t = 0; t < tasks.size(); ++t)
    ref[t] = cpu_run_range(cr[tasks[t].cls].cd, ids[tasks[t].cls], tasks[t].idx0, tasks[t].n,
      nullptr, fp32);
  {
    const int nt = omp_get_max_threads();
    omp_set_num_threads(1);
    int bad = 0;
    for (size_t t = 0; t < tasks.size(); ++t)
      bad += !acc_equal(ref[t], cpu_run_range(cr[tasks[t].cls].cd, ids[tasks[t].cls], tasks[t].idx0,
        tasks[t].n, nullptr, fp32));
    omp_set_num_threads(nt);
    printf("%-58s %d/%zu classes differ  %s\n",
      ("cpu: 1 thread vs " + std::to_string(nt) + " threads").c_str(), bad, tasks.size(),
      bad ? "FAIL" : "PASS");
    fails += bad != 0;
  }
  // (2) emulated multi-device configurations
  size_t maxb = 0;
  for (auto &c : cr) maxb = std::max(maxb, pack_class(c.cd).bytes());
  struct Cfg {
    const char *name;
    int ndev, blocks;
    bool reverse;
    int fail_dev, fail_after, fail_mode;
    double mem;
    double launch_s;
    long long maxch;
  };
  std::vector<Cfg> cfgs = {
    {"emu: 1 device, 1 block, one launch", 1, 1, false, -1, 0, 0, 0, 1e3, 1 << 19},
    {"emu: 3 devices, 5 blocks, many small launches", 3, 5, false, -1, 0, 0, 0, 1e-4, 1 << 19},
    {"emu: 2 devices, 7 blocks, reversed lanes", 2, 7, true, -1, 0, 0, 0, 1e-3, 1 << 19},
    {"emu: 4 devices, 3 blocks, launches of <= 8 chunks", 4, 3, false, -1, 0, 0, 0, 1e-2, 8},
    {"emu: 3 devices, device 1 fails at its 2nd launch", 3, 4, false, 1, 2, 0, 0, 1e-4, 1 << 19},
    {"emu: 2 devices, device 0 reads a zeroed view (guard)", 2, 4, false, 0, 2, 1, 0, 1e-4,
      1 << 19},
    {"emu: 2 devices, memory for 2.5 class images (eviction)", 2, 3, false, -1, 0, 0,
      2.5 * (double)maxb, 1e-4, 1 << 19},
    {"emu: 1 device failing at launch 3 (host fallback)", 1, 4, false, 0, 3, 0, 0, 1e-4, 1 << 19},
  };
  for (const Cfg &c : cfgs) {
    BackendOptions o;
    o.kind = "emu"; o.emu_devices = c.ndev; o.emu_blocks = c.blocks; o.emu_reverse = c.reverse;
    o.emu_fail_device = c.fail_dev; o.emu_fail_after = c.fail_after; o.emu_fail_mode = c.fail_mode;
    o.emu_mem = c.mem; o.launch_s = c.launch_s;
    o.max_launch_chunks = c.maxch;
    o.selftest_classes = 2;
    o.fp32 = fp32;
    std::string err;
    std::unique_ptr<Backend> be = make_multi_backend(o, &err);
    std::vector<const ClassData *> ptrs;
    for (auto &x : cr) ptrs.push_back(&x.cd);
    if (!be || !be->prepare(ptrs, ids, &err)) {
      printf("%-58s prepare failed: %s  FAIL\n", c.name, err.c_str());
      ++fails;
      continue;
    }
    std::vector<SampleOut> res;
    be->run(tasks, 1e300, res);
    int bad = 0, incomplete = 0;
    for (size_t t = 0; t < tasks.size(); ++t) {
      bad += !acc_equal(ref[t], res[t].acc);
      incomplete += res[t].done != tasks[t].n;
    }
    JsonW J; be->info_json(J);
    const bool st_ok = J.s.find("\"ok\":false") == std::string::npos;
    // With fewer classes than fit in the budget nothing is ever evicted
    const bool ev_possible = cr.size() >= 3;
    const bool ev = (c.mem > 0 && ev_possible) ? json_any_nonzero(J.s, "evictions") : true;
    const bool expect_dead = c.fail_dev >= 0;
    const bool dead_ok = !expect_dead || J.s.find("\"alive\":false") != std::string::npos;
    const bool guard_ok = c.fail_mode != 1
        || J.s.find("sample count check failed") != std::string::npos;
    const bool fb_ok = std::string(c.name).find("host fallback") == std::string::npos
        || json_any_nonzero(J.s, "fallback_units");
    const bool ok = !bad && !incomplete && st_ok && ev && dead_ok && fb_ok && guard_ok;
    printf("%-58s %d/%zu classes differ, %d incomplete, selftest %s%s%s%s  %s\n", c.name, bad,
      tasks.size(), incomplete, st_ok ? "ok" : "FAILED",
      expect_dead ? (dead_ok ? ", failed device removed" : ", failed device NOT removed") : "",
      c.fail_mode == 1
          ? (guard_ok ? " (zeroed view caught by the count check)" : " (zeroed view NOT caught)")
          : "",
      c.mem > 0 ? (!ev_possible ? ", eviction n/a (< 3 classes)" : ev ? ", evictions seen"
          : ", NO evictions") : "",
      ok ? "PASS" : "FAIL");
    if (a.has("verbose")) printf("%s\n", J.s.c_str());
    fails += !ok;
  }
  // (3) time box: a deadline in the past runs nothing; a short one returns a prefix of every task
  {
    BackendOptions o;
    o.kind = "emu"; o.emu_devices = 2; o.emu_blocks = 2; o.selftest = false; o.launch_s = 1e-4;
    o.fp32 = fp32;
    std::string err;
    std::unique_ptr<Backend> be = make_multi_backend(o, &err);
    std::vector<const ClassData *> ptrs;
    for (auto &x : cr) ptrs.push_back(&x.cd);
    be->prepare(ptrs, ids, &err);
    std::vector<SampleOut> res;
    be->run(tasks, rb_now_s() - 1, res);
    uint64_t done = 0;
    for (auto &r : res) done += r.done;
    // Calibrate, then give a deadline of a few launches
    be->run(tasks, 1e300, res);
    be->run(tasks, rb_now_s() + 0.02, res);
    int badprefix = 0, partial = 0;
    for (size_t t = 0; t < tasks.size(); ++t) {
      const uint64_t d = res[t].done;
      if (d != tasks[t].n && d % RB_UNIT != 0) ++badprefix;
      if (d != tasks[t].n) ++partial;
      if (d > 0) {
        RbAcc r = cpu_run_range(cr[tasks[t].cls].cd, ids[tasks[t].cls], tasks[t].idx0, d, nullptr,
          fp32);
        badprefix += !acc_equal(r, res[t].acc);
      }
    }
    const bool ok = done == 0 && badprefix == 0;
    printf("%-58s past deadline: %llu samples; short deadline: %d/%zu tasks partial, %d bad "
      "prefixes  %s\n", "time box (deadline) semantics", (unsigned long long)done, partial,
      tasks.size(), badprefix, ok ? "PASS" : "FAIL");
    fails += !ok;
  }
  printf("%s: %d failure(s)\n", fails ? "EMULATION TEST FAILED" : "emulation test passed", fails);
  return fails ? 6 : 0;
}

// Stream independence: per-sample weight correlation between class i of loop order L and class i of
// L2, both with the same --seed (two separate runs, as in practice), in the current and in the
// legacy stream scheme.
static double pearson(const std::vector<double> &x, const std::vector<double> &y) {
  double mx = 0, my = 0;
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  double sxy = 0, sxx = 0, syy = 0;
  for (size_t i = 0; i < n; ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  return sxy / std::sqrt(sxx * syy);
}
static std::vector<double> ranks(const std::vector<double> &a) {
  std::vector<size_t> id(a.size());
  for (size_t i = 0; i < id.size(); ++i) id[i] = i;
  std::sort(id.begin(), id.end(), [&](size_t x, size_t y) { return a[x] < a[y]; });
  std::vector<double> r(a.size());
  for (size_t i = 0; i < id.size(); ++i) r[id[i]] = (double)i;
  return r;
}
static int cmd_check_streams(const Args &a) {
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  const int L1 = (int)a.getl("loops", 4), L2 = (int)a.getl("loops2", L1 + 1);
  const uint64_t seed = (uint64_t)a.getl("seed", 12345);
  const uint64_t n = (uint64_t)a.getl("n", 200000);
  const int nc = (int)a.getl("classes", 8);
  std::vector<ClassRun> c1, c2;
  std::vector<std::pair<std::string, std::string>> d1, d2;
  Args a1 = a, a2 = a;
  a1.kv["gen"] = "1"; a2.kv["gen"] = "1"; a1.kv["loops"] = std::to_string(L1);
  a2.kv["loops"] = std::to_string(L2);
  if (!build_all(a1, kind, L1, 1.0, c1, d1, true) || !build_all(a2, kind, L2, 1.0, c2, d2, true))
    return 2;
  printf("# check --what streams: kind %s, class i of L=%d vs class i of L=%d, both with seed "
    "%llu, %llu samples\n", kind_name(kind), L1, L2, (unsigned long long)seed,
    (unsigned long long)n);
  int status = 0;
  for (const char *scheme : {"legacy", "salted-v1"}) {
    Args as = a; as.kv["streams"] = scheme;
    std::vector<RbStreamId> i1, i2;
    std::string err;
    if (!assign_streams(as, kind, L1, seed, c1, &i1, &err)
        || !assign_streams(as, kind, L2, seed, c2, &i2, &err)) {
      fprintf(stderr, "%s\n", err.c_str());
      return 2;
    }
    double worst_p = 0, worst_s = 0;
    for (int i = 0; i < nc && i < (int)c1.size() && i < (int)c2.size(); ++i) {
      const RbClassView v1 = c1[i].cd.view(), v2 = c2[i].cd.view();
      std::vector<double> p1(std::max(1, c1[i].cd.natoms)), p2(std::max(1, c2[i].cd.natoms)), w1(n),
          w2(n);
      for (uint64_t k = 0; k < n; ++k) {
        double sa;
        w1[k] = rb_sample<double>(v1, i1[i].key, i1[i].stream, k, p1.data(), &sa);
        w2[k] = rb_sample<double>(v2, i2[i].key, i2[i].stream, k, p2.data(), &sa);
      }
      const double rp = pearson(w1, w2), rs = pearson(ranks(w1), ranks(w2));
      worst_p = std::max(worst_p, std::fabs(rp)); worst_s = std::max(worst_s, std::fabs(rs));
      printf("  %-9s class %2d  %-26s %-28s pearson %+.4f  spearman %+.4f\n", scheme, i,
        c1[i].cd.G.name.c_str(), c2[i].cd.G.name.c_str(), rp, rs);
    }
    const double lim = 5.0 / std::sqrt((double)n); // |rho| of independent streams: ~ 1/sqrt(n)
    const bool ok = std::string(scheme) == "legacy" || (worst_p < lim && worst_s < lim);
    printf("%-9s max |pearson| %.4f  max |spearman| %.4f  (independence: < %.4f)  %s\n", scheme,
      worst_p, worst_s, lim,
      std::string(scheme) == "legacy" ? "(legacy scheme, for comparison)" : (ok ? "PASS" : "FAIL"));
    if (!ok) status = 7;
  }
  return status;
}

// Checks.
static void random_point(std::mt19937_64 &rng, int l, double *lx, double lo = std::log(1e-3)) {
  std::uniform_real_distribution<double> U(lo, 0.0);
  for (int e = 0; e < l; ++e) lx[e] = U(rng);
}

static int cmd_check(const Args &a) {
  std::string what = a.get("what", "philox");
  if (what == "emu") return cmd_check_emu(a);
  if (what == "streams") return cmd_check_streams(a);
  if (what == "philox") {
    // Known-answer tests from Random123 kat_vectors (philox4x32_10)
    struct KAT { uint32_t c[4], k[2], r[4]; } kat[] = {
      {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
      {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu},
        {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
      {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u},
        {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}}};
    int bad = 0;
    for (auto &t : kat) {
      RbPhilox4 c; for (int i = 0; i < 4; ++i) c.v[i] = t.c[i];
      RbPhilox4 o = rb_philox4x32_10(c, t.k[0], t.k[1]);
      bool ok = true; for (int i = 0; i < 4; ++i) ok = ok && o.v[i] == t.r[i];
      printf("philox KAT: %08x %08x %08x %08x  %s\n", o.v[0], o.v[1], o.v[2], o.v[3],
        ok ? "OK" : "MISMATCH");
      bad += !ok;
    }
    return bad ? 3 : 0;
  }
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  int L = (int)a.getl("loops", 2);
  std::vector<Graph> gs;
  std::string err;
  if (!get_graphs(a, kind, L, gs, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
  std::mt19937_64 rng((uint64_t)a.getl("seed", 1));
  int npts = (int)a.getl("points", 64);
  // The production density (--omega-scale, --omega-shift) as in run
  BuildOptions bo = make_build_options(a, kind);
  bo.need_tables = false;
  printf("# check --what %s: kind %s L=%d, omega-scale %g, omega-shift %d%s (%s %s)\n",
    what.c_str(), kind_name(kind), L, bo.omega_scale, bo.omega_shift, bo.prune ? "" : ", unpruned",
    RB_PROGRAM, RB_VERSION);
  int status = 0;
  if (what == "vanish" || what == "prune") {
    // vanish: structural 1VR/tadpole graphs must come out with no terms (b, h); report the others.
    // prune: pruned and unpruned expansions must agree pointwise (long double), and report the
    //   reduction.
    for (auto &G : gs) {
      ClassData P, Q;
      if (!build_class(G, bo, P, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
      bool structural = is_1vr(G) || has_selfloop(G);
      if (what == "vanish") {
        bool ok = !structural || P.ncomp == 0 || kind == Kind::C;
        printf("%-28s 1VR=%d selfloop=%d raw_terms=%-6ld pruned_terms=%-5d %s\n", G.name.c_str(),
          (int)is_1vr(G), (int)has_selfloop(G), P.nterms_raw, P.ncomp,
          structural ? (P.ncomp == 0 ? "VANISHES (algebraic)"
              : (kind == Kind::C ? "nonzero (c: allowed)" : "NONZERO -> FAIL"))
              : (P.ncomp == 0 ? "vanishes (non-structural!)" : ""));
        if (!ok) status = 4;
        continue;
      }
      BuildOptions bq = bo; bq.prune = false;
      if (!build_class(G, bq, Q, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
      // |pruned - raw| relative to Sum|raw terms| (round-off scale) AND relative to |f| (what
      // matters), at points in [0.05,1]^l (base) and a quarter of them near random faces (lines of
      // a random subset scaled by 1e-4)
      // Criteria: everywhere |pruned - raw| <= 1e-15 Sum|raw terms| (long double round-off of the
      // raw expansion); at base points (no cancellation to speak of) also |pruned - raw| <= 1e-10
      // |f|.  Near faces |f| << Sum|raw| and the |f|-relative difference only measures the
      // round-off of the RAW expansion (reported, not tested).
      long double worst = 0, worst_cancel = 1, worst_f = 0, worst_f_face = 0;
      for (int k = 0; k < npts; ++k) {
        double lx[RB_MAXE]; random_point(rng, G.l(), lx, std::log(0.05));
        const bool face = 4 * k >= 3 * npts;
        if (face) {
          const uint32_t S = (uint32_t)(rng() % ((1ull << G.l()) - 2)) + 1u;
          for (int e = 0; e < G.l(); ++e) if ((S >> e) & 1u) lx[e] += std::log(1e-4);
        }
        long double sp = 0, sq = 0;
        long double vp = P.ncomp ? eval_point<long double>(P, lx, &sp) : 0.0L;
        long double vq = eval_point<long double>(Q, lx, &sq);
        long double d = fabsl(vp - vq) / std::max(sq, (long double)1e-300);
        worst = std::max(worst, d);
        const long double df = fabsl(vp - vq) / std::max(fabsl(vq), (long double)1e-300);
        if (face) worst_f_face = std::max(worst_f_face, df); else worst_f = std::max(worst_f, df);
        if (sq > 0) worst_cancel = std::min(worst_cancel, fabsl(vq) / sq);
      }
      const bool ok = worst < 1e-15 && (P.ncomp == 0 || worst_f < 1e-10);
      printf("%-28s raw=%-6d pruned=%-5d atoms=%-4d max|pruned-raw|/sum|raw| = %.2Le  base points: "
        "max|pruned-raw|/|f| = %.2Le  (near faces %.2Le)  min|sum|/sum|raw terms| = %.2Le %s\n",
        G.name.c_str(), Q.ncomp, P.ncomp, P.natoms, worst, P.ncomp ? worst_f : 0.0L,
        P.ncomp ? worst_f_face : 0.0L, worst_cancel, ok ? "OK" : "FAIL");
      if (!ok) status = 4;
    }
    return status;
  }
  if (what == "twins") {
    // Cycle-matroid (Whitney) twins: compare the dot-summed integrands pointwise under the matroid
    // edge map
    std::vector<int> done(gs.size(), 0);
    int npairs = 0;
    for (size_t i = 0; i < gs.size(); ++i)
      for (size_t j = i + 1; j < gs.size(); ++j) {
        std::vector<int> emap;
        if (!matroid_iso(gs[i], gs[j], &emap)) continue;
        ++npairs;
        ClassData A, B;
        if (!build_class(gs[i], bo, A, &err) || !build_class(gs[j], bo, B, &err)) {
          fprintf(stderr, "%s\n", err.c_str());
          return 2;
        }
        long double worst = 0, scale = 0, fmax = 0;
        for (int k = 0; k < npts; ++k) {
          double lx[RB_MAXE], ly[RB_MAXE];
          random_point(rng, gs[i].l(), lx, std::log(0.05));
          for (int e = 0; e < gs[i].l(); ++e) ly[emap[e]] = lx[e];
          long double sa = 0, sb = 0;
          long double va = A.ncomp ? eval_point<long double>(A, lx, &sa) : 0.0L;
          long double vb = B.ncomp ? eval_point<long double>(B, ly, &sb) : 0.0L;
          worst = std::max(worst, fabsl(va - vb));
          scale = std::max(scale, std::max(sa, sb));
          fmax = std::max(fmax, fabsl(va));
        }
        bool eq = worst <= 1e-15 * std::max(scale, (long double)1e-300)
            && (fmax == 0 || worst <= 1e-11L * fmax);
        printf("twins %-26s %-26s S=%.6g,%.6g comps=%d,%d  max|fA-fB|=%.2Le (scale %.2Le, max|f| "
          "%.2Le) %s\n", gs[i].name.c_str(), gs[j].name.c_str(), A.S, B.S, A.ncomp, B.ncomp, worst,
          scale, fmax, eq ? "EQUAL pointwise" : "DIFFERENT");
      }
    printf("%d matroid-isomorphic pairs\n", npairs);
    return 0;
  }
  if (what == "topw") {
    // Locate the largest weights of one class: sector flags S_j with their log-gaps y_j = lx(next)
    // - lx(this)
    bo.need_tables = true;
    uint64_t n = (uint64_t)a.getl("n", 1000000);
    int top = (int)a.getl("top", 12);
    for (size_t gi = 0; gi < gs.size(); ++gi) {
      if (a.has("nickel") && gs[gi].name != a.get("nickel")) continue;
      ClassData cd;
      if (!build_class(gs[gi], bo, cd, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
      if (cd.ncomp == 0) continue;
      RbClassView v = cd.view();
      std::vector<std::pair<double, uint64_t>> best;
      double s1 = 0;
#pragma omp parallel
      {
        std::vector<double> lp(cd.natoms + 1);
        std::vector<std::pair<double, uint64_t>> loc;
        double ls = 0;
#pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n; ++i) {
          double sa;
          double w = std::fabs(rb_sample<double>(v, 777, (uint32_t)gi, (uint64_t)i, lp.data(),
            &sa));
          ls += w; loc.push_back({w, (uint64_t)i});
          if (loc.size() > 4096) {
            std::nth_element(loc.begin(), loc.begin() + top, loc.end(),
              std::greater<std::pair<double, uint64_t>>());
            loc.resize(top);
          }
        }
#pragma omp critical
        { best.insert(best.end(), loc.begin(), loc.end()); s1 += ls; }
      }
      std::sort(best.begin(), best.end(), std::greater<std::pair<double, uint64_t>>());
      double mean = s1 / n;
      printf("== %s  mean|w| %.4g  (lines:", cd.G.name.c_str(), mean);
      for (int e = 0; e < cd.l; ++e) printf(" %d=%d-%d", e, cd.G.eu[e], cd.G.ev[e]);
      printf(")\n");
      for (int r = 0; r < top && r < (int)best.size(); ++r) {
        RbStream rs; rb_stream_init(&rs, 777, (uint32_t)gi, best[r].second);
        signed char ord[RB_MAXE]; double lx[RB_MAXE];
        rb_point(v, &rs, ord, lx);
        printf("  w/mean %9.3g  flags(gap>3):", best[r].first / mean);
        uint32_t S = 0;
        for (int j = 0; j + 1 < cd.l; ++j) {
          S |= 1u << ord[j];
          double gap = lx[ord[j + 1]] - lx[ord[j]];
          if (gap > 3 || a.has("allgaps")) {
            printf(" {");
            for (int e = 0; e < cd.l; ++e) if ((S >> e) & 1u) printf("%d,", e);
            printf("} y=%.1f om=%.1f%s;", gap, cd.omega[S], cd.omega2[S] == 2 && 0 ? "" : "");
          }
        }
        printf("\n");
        if (a.has("terms")) {
          // Per-group contributions at this point
          double lxd[RB_MAXE]; RbStream r2; rb_stream_init(&r2, 777, (uint32_t)gi, best[r].second);
          signed char o2[RB_MAXE]; double lvs = rb_point(v, &r2, o2, lxd);
          std::vector<double> lp(cd.natoms + 1);
          double sa; double w = rb_weight_at<double>(v, o2, lxd, lvs, lp.data(), &sa);
          printf("     w %.4g  sum|terms| %.4g  logP:", w, sa);
          for (int q = 0; q < cd.natoms; ++q) printf(" %.2f", lp[q]);
          printf("\n");
          // Largest groups
          double sumlx = 0, sx = 0;
          for (int e = 0; e < cd.l; ++e) { sumlx += lxd[e]; sx += std::exp(lxd[e]); }
          double logT = cd.logJE + sumlx - lvs - cd.pw * std::log(sx);
          std::vector<std::pair<double, int>> gv;
          for (int g = 0; g < cd.ngroup; ++g) {
            double e = logT;
            for (int f = cd.grp_foff[g]; f < cd.grp_foff[g + 1]; ++f)
              e += 0.5 * cd.cf_p2[f] * lp[cd.cf_atom[f]];
            double t = 0;
            for (int k = cd.grp_off[g]; k < cd.grp_off[g + 1]; ++k) {
              double m = cd.comp_coeff[k];
              if (cd.comp_d0[k] >= 0) m *= std::exp(lxd[cd.comp_d0[k]]);
              if (cd.comp_d1[k] >= 0) m *= std::exp(lxd[cd.comp_d1[k]]);
              t += m;
            }
            gv.push_back({t * std::exp(e), g});
          }
          std::sort(gv.begin(), gv.end(), [](auto &x, auto &y) {
            return std::fabs(x.first) > std::fabs(y.first);
          });
          for (int q = 0; q < 4 && q < (int)gv.size(); ++q) {
            int g = gv[q].second;
            printf("     group %3d value %+.4g  factors:", g, gv[q].first);
            for (int f = cd.grp_foff[g]; f < cd.grp_foff[g + 1]; ++f) {
              const AtomDef &A = cd.atoms[cd.cf_atom[f]];
              printf(" [");
              for (size_t z = 0; z < A.g.size(); ++z)
                printf("%d%s", A.g[z], z + 1 < A.g.size() ? "," : "");
              printf("]^%g", 0.5 * cd.cf_p2[f]);
            }
            printf("  dots:");
            for (int k = cd.grp_off[g]; k < cd.grp_off[g + 1]; ++k)
              printf(" %+g*x%d", cd.comp_coeff[k], cd.comp_d0[k]);
            printf("\n");
          }
        }
      }
    }
    return 0;
  }
  if (what == "tail") {
    // Hill estimator of the tail index alpha of |w| (top k order statistics) per class: alpha <= 2
    // means infinite variance (error bars unreliable); bounded weights give large alpha.  Also the
    // share of the second moment carried by the top 1e-4 of the samples.
    bo.need_tables = true;
    uint64_t n = (uint64_t)a.getl("n", 2000000);
    int k = (int)a.getl("k", 200);
    for (size_t gi = 0; gi < gs.size(); ++gi) {
      ClassData cd;
      if (!build_class(gs[gi], bo, cd, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
      if (cd.ncomp == 0) continue;
      RbClassView v = cd.view();
      const int nt = omp_get_max_threads();
      std::vector<std::vector<double>> keep(nt);
      std::vector<double> s1(nt, 0), s2(nt, 0);
#pragma omp parallel
      {
        int t = omp_get_thread_num();
        std::vector<double> lp(cd.natoms + 1);
        std::vector<double> &kp = keep[t];
#pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n; ++i) {
          double sa;
          double w = std::fabs(rb_sample<double>(v, 777, (uint32_t)gi, (uint64_t)i, lp.data(),
            &sa));
          s1[t] += w; s2[t] += w * w;
          kp.push_back(w);
          if (kp.size() > (size_t)(4 * k + 64)) {
            std::nth_element(kp.begin(), kp.begin() + 2 * k, kp.end(), std::greater<double>());
            kp.resize(2 * k + 1);
          }
        }
      }
      std::vector<double> all;
      double S1 = 0, S2 = 0;
      for (int t = 0; t < nt; ++t) {
        all.insert(all.end(), keep[t].begin(), keep[t].end());
        S1 += s1[t];
        S2 += s2[t];
      }
      std::sort(all.begin(), all.end(), std::greater<double>());
      double hill = 0;
      for (int i = 0; i < k; ++i) hill += std::log(all[i] / all[k]);
      double alpha = k / hill;
      double top = 0;
      size_t ntop = std::max<size_t>(1, n / 10000);
      for (size_t i = 0; i < ntop && i < all.size(); ++i) top += all[i] * all[i];
      double mean = S1 / n;
      printf("%-28s alpha(Hill, k=%d) = %6.2f  max|w|/mean|w| = %9.3g  top-1e-4 share of sum w^2 = "
        "%.3f  rel sd |w| = %.3g\n", cd.G.name.c_str(), k, alpha, all[0] / mean, top / S2,
        std::sqrt(std::max(0.0, S2 / n - mean * mean)) / mean);
    }
    return 0;
  }
  if (what == "fp32") {
    // Paired FP32-evaluation vs FP64 at identical sampled points (the --fp32 path of the bare
    // kinds)
    bo.need_tables = true;
    uint64_t n = (uint64_t)a.getl("n", 20000);
    double worst_shift = 0;
    for (size_t gi = 0; gi < gs.size(); ++gi) {
      ClassData cd;
      if (!build_class(gs[gi], bo, cd, &err)) {
        if (a.has("skip-divergent") && err.rfind("divergent", 0) == 0) continue;
        fprintf(stderr, "%s\n", err.c_str());
        return 2;
      }
      if (cd.ncomp == 0) continue;
      RbClassView v = cd.view();
      std::vector<double> lp(cd.natoms + 1);
      std::vector<float> lf(cd.natoms + 1);
      double s1 = 0, s2 = 0, d1 = 0, d2 = 0, dmax = 0, nb = 0;
      for (uint64_t k = 0; k < n; ++k) {
        double sa; float sf;
        const double w64 = rb_sample<double>(v, 99, (uint32_t)gi, k, lp.data(), &sa);
        const double w32 = (double)rb_sample<float>(v, 99, (uint32_t)gi, k, lf.data(), &sf);
        if (!std::isfinite(w32)) { ++nb; continue; }
        const double d = w32 - w64;
        s1 += w64; s2 += w64 * w64; d1 += d; d2 += d * d;
        dmax = std::max(dmax, std::fabs(d) / std::max(std::fabs(w64), 1e-300));
      }
      const double m = s1 / n, sd = std::sqrt(std::max(0.0, s2 / n - m * m));
      // Systematic shift of the mean in units of the per-sample sd
      const double shift = std::fabs(d1 / n) / sd;
      worst_shift = std::max(worst_shift, shift);
      printf("%-28s mean(w64) %+.6e sd %.3e | FP32-FP64: mean %+.2e (%.1e sd)  rms/sd %.1e  max "
        "rel %.1e  nonfinite %.0f\n", cd.G.name.c_str(), m, sd, d1 / n, shift,
        std::sqrt(d2 / n) / sd, dmax, nb);
      if (nb > 0) status = 5;
    }
    printf("worst |mean(FP32-FP64)|/sd = %.2e\n", worst_shift);
    return status;
  }
  if (what == "fp" || what == "faces") {
    bo.need_tables = true;
    uint64_t n = (uint64_t)a.getl("n", 20000);
    for (size_t gi = 0; gi < gs.size(); ++gi) {
      ClassData cd;
      if (!build_class(gs[gi], bo, cd, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
      if (cd.ncomp == 0) continue;
      RbClassView v = cd.view();
      if (what == "fp") {
        // Paired FP64 vs long double at identical points; |diff| relative to the per-sample sd
        std::vector<double> lp(cd.natoms + 1);
        std::vector<long double> lq(cd.natoms + 1);
        double s1 = 0, s2 = 0, d1 = 0, d2 = 0, dmax = 0;
        for (uint64_t k = 0; k < n; ++k) {
          RbStream rs; rb_stream_init(&rs, 99, (uint32_t)gi, k);
          signed char ord[RB_MAXE]; double lx[RB_MAXE];
          double lvs = rb_point(v, &rs, ord, lx);
          double sa; long double sb;
          double w64 = rb_weight_at<double>(v, ord, lx, lvs, lp.data(), &sa);
          long double w80 = rb_weight_at<long double>(v, ord, lx, lvs, lq.data(), &sb);
          double d = (double)(w64 - w80);
          s1 += (double)w80; s2 += (double)(w80 * w80); d1 += d; d2 += d * d;
          dmax = std::max(dmax, std::fabs(d));
        }
        double m = s1 / n, sd = std::sqrt(std::max(0.0, s2 / n - m * m));
        printf("%-28s mean(w80) %+.6e sd %.3e | FP64-FP80: mean %+.2e  rms %.2e  max %.2e  (rms/sd "
          "%.1e, |mean|/sd*sqrt(n) %.2f)\n", cd.G.name.c_str(), m, sd, d1 / n, std::sqrt(d2 / n),
          dmax, std::sqrt(d2 / n) / sd, std::fabs(d1 / n) / sd * std::sqrt((double)n));
      } else {
        // Faces whose omega was shifted: estimate the true exponent of the pruned sum in long
        // double
        const uint32_t NS = 1u << cd.l;
        int nbad = 0, nchk = 0;
        double worst = 1e9;
        for (uint32_t S = 1; S + 1 < NS; ++S) {
          // Recompute min exponent to detect shifted faces
          bool shifted = false;
          {
            // omega2 >= 2 at shifted faces by construction; re-derive via a quick scan of the
            // component minimum
            ClassData &c = cd;
            int best = 1 << 30;
            for (auto &comp : c.comps) {
              int d = 0;
              for (auto &f : comp.f) {
                const AtomDef &A = c.atoms[f.first];
                int p[RB_MAXV]; for (int u = 0; u < A.nv; ++u) p[u] = u;
                int r = 0, cnt = 0;
                for (size_t q = 0; q < A.g.size(); ++q)
                  if ((S >> A.g[q]) & 1u) {
                    ++cnt;
                    int x = A.u[q], y = A.v[q];
                    while (p[x] != x) x = p[x];
                    while (p[y] != y) y = p[y];
                    if (x != y) { p[x] = y; ++r; }
                  }
                d += f.second * (cnt - r);
              }
              if (comp.d0 >= 0 && ((S >> comp.d0) & 1u)) d += 2;
              if (comp.d1 >= 0 && ((S >> comp.d1) & 1u)) d += 2;
              best = std::min(best, d);
            }
            shifted = 2 * __builtin_popcount(S) + best <= 0;
          }
          if (!shifted && !a.has("allfaces")) continue;
          ++nchk;
          double est = 1e9;
          for (int k = 0; k < 4; ++k) {
            double lx[RB_MAXE]; random_point(rng, cd.l, lx, std::log(0.2));
            double r1 = 1e-3, r2 = 1e-5;
            double l1[RB_MAXE], l2[RB_MAXE];
            for (int e = 0; e < cd.l; ++e) {
              l1[e] = lx[e] + (((S >> e) & 1u) ? std::log(r1) : 0);
              l2[e] = lx[e] + (((S >> e) & 1u) ? std::log(r2) : 0);
            }
            long double sa, sb;
            long double f1 = eval_point<long double>(cd, l1, &sa),
                f2 = eval_point<long double>(cd, l2, &sb);
            // eval_point returns f * Prod x (measure included): exponent of rho directly
            double ex = (double)(logl(fabsl(f2) / fabsl(f1)) / logl((long double)(r2 / r1)));
            est = std::min(est, ex);
          }
          worst = std::min(worst, est - cd.omega[S]);
          bool ok = est >= cd.omega[S] - 0.05;
          if (!ok) ++nbad;
          if (a.has("verbose") || !ok)
            printf("  face %#x |S|=%d omega=%.1f true~%.2f %s\n", S, __builtin_popcount(S),
              cd.omega[S], est, ok ? "" : "VIOLATED");
        }
        printf("%-28s shifted faces %d checked %d violated %d (min true-omega margin %.2f)\n",
          cd.G.name.c_str(), cd.nshift, nchk, nbad, nchk ? worst : 0.0);
        if (nbad) status = 5;
      }
    }
    return status;
  }
  fprintf(stderr, "unknown check\n");
  return 1;
}

// eval: read points (l log-coordinates per line) from stdin, print f(x)*Prod x (long double) for
// one graph
static int cmd_eval(const Args &a) {
  Kind kind;
  if (!kind_from_string(a.get("kind", "b"), &kind)) { fprintf(stderr, "bad --kind\n"); return 1; }
  Graph G; std::string err;
  if (!parse_nickel(a.get("nickel"), G, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
  BuildOptions bo = make_build_options(a, kind);
  bo.need_tables = false;
  ClassData cd;
  if (!build_class(G, bo, cd, &err)) { fprintf(stderr, "%s\n", err.c_str()); return 2; }
  std::string line;
  while (std::getline(std::cin, line)) {
    std::stringstream ss(line);
    double lx[RB_MAXE]; int n = 0;
    while (n < cd.l && ss >> lx[n]) ++n;
    if (n != cd.l) continue;
    long double sa = 0;
    long double v = cd.ncomp ? eval_point<long double>(cd, lx, &sa) : 0.0L;
    printf("%.18Le %.6Le\n", v, sa);
  }
  return 0;
}

static int dispatch(const Args &a, const char *argv0) {
  if ((a.cmd == "info" || a.cmd == "bench" || a.cmd == "check" || a.cmd == "selftest"
      || a.cmd == "eval" || (a.cmd == "run" && !a.has("resume"))))
    if (int rc = precheck_limits(a)) return rc;
  if (a.cmd == "gen") return cmd_gen(a);
  if (a.cmd == "info") return cmd_info(a);
  if (a.cmd == "build") return cmd_build(a);
  if (a.cmd == "run") return cmd_run(a);
  if (a.cmd == "bench") return cmd_bench(a);
  if (a.cmd == "check") return cmd_check(a);
  if (a.cmd == "eval") return cmd_eval(a);
  if (a.cmd == "devices") return cmd_devices(a);
  if (a.cmd == "selftest") return cmd_selftest(a);
  if (a.cmd == "version") {
    printf("%s %s%s%s variant %s (RB_MAXE %d, RB_MAXLOOP %d, RB_MAXV %d, RB_MAXATOMS %d) build "
      "signature %s\n", RB_PROGRAM, RB_VERSION, *RB_BUILD_INFO ? " build " : "", RB_BUILD_INFO,
      RB_VARIANT, RB_MAXE, RB_MAXLOOP, RB_MAXV, RB_MAXATOMS, hex64(rb_build_signature()).c_str());
    return 0;
  }
  fprintf(stderr,
    "usage: %s gen|info|run|bench|check|devices|selftest|version [--options]  (see README.md)\n",
    argv0);
  return 1;
}

int main(int argc, char **argv) {
  g_argv = argv;
  Args a = parse_args(argc, argv);
  if (a.has("threads")) omp_set_num_threads((int)a.getl("threads", 1));
  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  const int rc = dispatch(a, argv[0]);
  // Never wait on a hung GPU context (static destructors / cudart teardown would block forever)
  if (g_rb_hung) { fflush(stdout); fflush(stderr); std::_Exit(rc ? rc : 5); }
  return rc;
}
