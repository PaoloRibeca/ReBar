/*
    cache.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Binary reader and writer of the class cache (cache.h): a magic string,
    the fingerprint of the build, the number of classes, then every field
    of every ClassData with explicit lengths. The graph of a class is
    restored from its Nickel index. Little-endian native layout; the file
    is meant for the machine (or cluster) that wrote it.

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

#include "cache.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "graph.h"
#include "rbar/config.h"

static const char MAGIC[8] = {'R', 'B', 'C', 'A', 'C', 'H', 'E', '1'};

std::string cache_fingerprint(const BuildOptions &bo, bool tables, const char *version) {
  char buf[256];
  snprintf(buf, sizeof buf, "%s|%d|%d|%d|%s|%d|%d|%d|%.17g|%d|%d", version, RB_MAXE, RB_MAXLOOP,
    RB_MAXATOMS, kind_name(bo.kind), (int)bo.prune, (int)bo.vertex_disjoint, bo.omega_shift,
    bo.omega_scale, (int)bo.allow_negative_faces, (int)tables);
  return buf;
}

namespace {
struct W {
  std::ofstream f;
  bool ok = true;
  void raw(const void *p, size_t n) {
    f.write((const char *)p, (std::streamsize)n);
    ok = ok && f.good();
  }
  void u64(uint64_t x) { raw(&x, 8); }
  void i32(int32_t x) { raw(&x, 4); }
  void dbl(double x) { raw(&x, 8); }
  void str(const std::string &s) { u64(s.size()); raw(s.data(), s.size()); }
  template <class T> void vec(const std::vector<T> &v) {
    u64(v.size());
    if (!v.empty()) raw(v.data(), v.size() * sizeof(T));
  }
};
struct R {
  std::ifstream f;
  bool ok = true;
  void raw(void *p, size_t n) {
    f.read((char *)p, (std::streamsize)n);
    ok = ok && (size_t)f.gcount() == n;
  }
  uint64_t u64() { uint64_t x = 0; raw(&x, 8); return x; }
  int32_t i32() { int32_t x = 0; raw(&x, 4); return x; }
  double dbl() { double x = 0; raw(&x, 8); return x; }
  std::string str() {
    uint64_t n = u64();
    std::string s;
    if (!ok || n > (1u << 30)) { ok = false; return s; }
    s.resize(n);
    raw(&s[0], n);
    return s;
  }
  template <class T> void vec(std::vector<T> &v) {
    uint64_t n = u64(); if (!ok || n > (1u << 31)) { ok = false; return; }
    v.resize(n); if (n) raw(v.data(), n * sizeof(T));
  }
};
} // namespace

bool cache_write(const std::string &fn, const std::string &fp, const std::vector<CachedClass> &cs,
    std::string *err) {
  const std::string tmp = fn + ".tmp";
  W w; w.f.open(tmp, std::ios::binary);
  if (!w.f) { *err = "cannot write " + tmp; return false; }
  w.raw(MAGIC, 8); w.str(fp); w.u64(cs.size());
  for (const CachedClass &c : cs) {
    const ClassData &d = c.cd;
    w.str(d.G.name); w.dbl(c.build_s);
    w.i32((int32_t)d.kind); w.i32(d.l); w.i32(d.L); w.dbl(d.pw); w.dbl(d.pref); w.dbl(d.S);
    w.vec(d.on);
    w.i32(d.is1vr); w.i32(d.selfloop); w.i32(d.nparts); w.i32(d.ndots);
    w.u64((uint64_t)d.nforests); w.u64((uint64_t)d.nterms_raw);
    w.u64((uint64_t)d.nterms_nonzero_forests);
    w.i32(d.ncomp); w.i32(d.natoms); w.i32(d.max_atom_loops); w.i32(d.nshift); w.dbl(d.omega_min);
    w.i32(d.omega2_pre_min); w.i32(d.nneg_faces);
    w.dbl(d.t_expand); w.dbl(d.t_tables);
    w.u64(d.atoms.size());
    for (const AtomDef &a : d.atoms) {
      w.vec(a.g);
      w.vec(a.u);
      w.vec(a.v);
      w.i32(a.nv);
      w.i32(a.nloops);
      w.u64(a.mask);
    }
    w.u64(d.comps.size());
    for (const CompDef &cp : d.comps) { w.dbl(cp.c); w.i32(cp.d0); w.i32(cp.d1); w.vec(cp.f); }
    w.vec(d.omega2); w.vec(d.omega); w.vec(d.Jt); w.dbl(d.logJE); w.i32(d.ngroup);
    w.vec(d.atom_off); w.vec(d.atom_nv); w.vec(d.grp_off); w.vec(d.grp_foff); w.vec(d.cf_atom);
    w.vec(d.ae_u); w.vec(d.ae_v); w.vec(d.atom_loc); w.vec(d.cf_p2); w.vec(d.comp_d0);
    w.vec(d.comp_d1); w.vec(d.comp_coeff);
  }
  w.f.close();
  if (!w.ok) { *err = "write error on " + tmp; return false; }
  if (rename(tmp.c_str(), fn.c_str()) != 0) {
    *err = "cannot rename " + tmp + " to " + fn;
    return false;
  }
  return true;
}

bool cache_read(const std::string &fn, const std::string &fp, std::vector<CachedClass> &cs,
    std::string *err) {
  R r; r.f.open(fn, std::ios::binary);
  if (!r.f) { *err = "cannot open " + fn; return false; }
  char magic[8]; r.raw(magic, 8);
  if (!r.ok || memcmp(magic, MAGIC, 8) != 0) { *err = fn + " is not a class cache"; return false; }
  const std::string got = r.str();
  if (got != fp) {
    *err = "built with a different code or options: file '" + got + "', this run '" + fp + "'";
    return false;
  }
  const uint64_t n = r.u64();
  if (!r.ok || n > (1u << 26)) { *err = "corrupt header in " + fn; return false; }
  cs.clear(); cs.reserve(n);
  for (uint64_t i = 0; i < n && r.ok; ++i) {
    CachedClass c; ClassData &d = c.cd;
    const std::string name = r.str(); c.build_s = r.dbl();
    d.kind = (Kind)r.i32(); d.l = r.i32(); d.L = r.i32(); d.pw = r.dbl(); d.pref = r.dbl();
    d.S = r.dbl(); r.vec(d.on);
    d.is1vr = r.i32() != 0; d.selfloop = r.i32() != 0; d.nparts = r.i32(); d.ndots = r.i32();
    d.nforests = (long)r.u64(); d.nterms_raw = (long)r.u64();
    d.nterms_nonzero_forests = (long)r.u64();
    d.ncomp = r.i32(); d.natoms = r.i32(); d.max_atom_loops = r.i32(); d.nshift = r.i32();
    d.omega_min = r.dbl(); d.omega2_pre_min = r.i32(); d.nneg_faces = r.i32();
    d.t_expand = r.dbl(); d.t_tables = r.dbl();
    uint64_t na = r.u64(); if (!r.ok || na > (1u << 24)) { r.ok = false; break; }
    d.atoms.resize(na);
    for (AtomDef &a : d.atoms) {
      r.vec(a.g);
      r.vec(a.u);
      r.vec(a.v);
      a.nv = r.i32();
      a.nloops = r.i32();
      a.mask = (uint32_t)r.u64();
    }
    uint64_t nc = r.u64(); if (!r.ok || nc > (1u << 28)) { r.ok = false; break; }
    d.comps.resize(nc);
    for (CompDef &cp : d.comps) { cp.c = r.dbl(); cp.d0 = r.i32(); cp.d1 = r.i32(); r.vec(cp.f); }
    r.vec(d.omega2); r.vec(d.omega); r.vec(d.Jt); d.logJE = r.dbl(); d.ngroup = r.i32();
    r.vec(d.atom_off); r.vec(d.atom_nv); r.vec(d.grp_off); r.vec(d.grp_foff); r.vec(d.cf_atom);
    r.vec(d.ae_u); r.vec(d.ae_v); r.vec(d.atom_loc); r.vec(d.cf_p2); r.vec(d.comp_d0);
    r.vec(d.comp_d1); r.vec(d.comp_coeff);
    std::string perr;
    if (!parse_nickel(name, d.G, &perr)) {
      *err = "class " + name + " in " + fn + ": " + perr;
      return false;
    }
    cs.push_back(std::move(c));
  }
  if (!r.ok) { *err = "truncated or corrupt class cache " + fn; return false; }
  return true;
}
