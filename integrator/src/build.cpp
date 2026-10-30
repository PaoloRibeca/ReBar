/*
    build.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Implementation of build.h.

    Parametric R-bar (zero-momentum BPHZ, Lagrangian forests) for one graph
    with phi^2 "dot" insertions.
    Every forest term lives on the graph's simplex with the common
    prefactor Gamma(pw) pi^(-L/2):
      term(F) = (-1)^|F| * xdot
                * Prod_{rho in F u {G}} U_{rho/F}(eps)^(-3/2)
                [ * W_G U_G^-1 at the root for h ]
    where rho/F is rho with its maximal F-children contracted (t^0
    children) or replaced by an effective massless line of parameter eps_c
    = W_c/U_c (t^2 children; U and W are multilinear in eps_c by
    deletion-contraction:
    U(eps) = U_contracted + eps U_split), expanded to first order in each
    eps_c (jet variables).
    Every polynomial is factorised into the Kirchhoff polynomials of its
    2-connected blocks ("atoms"), so that identical products from different
    forests get identical keys; summing the coefficients of identical keys
    performs all exact cancellations algebraically (e.g. tadpole-type
    reduced graphs, cut-vertex factorisation, one-vertex-reducible graphs,
    which then come out with no terms at all).

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

#include "build.h"
#include <algorithm>
#include <time.h>
#include <climits>
#include <cmath>
#include <functional>
#include <map>
#include <set>

const char *kind_name(Kind k) {
  switch (k) {
    case Kind::B: return "b"; case Kind::H: return "h"; case Kind::C: return "c";
    case Kind::BARE4: return "bare4"; case Kind::INT4: return "int4";
    case Kind::BARE2D: return "bare2d"; case Kind::INT2D: return "int2d";
    case Kind::INT2: return "int2";
  }
  return "?";
}
bool kind_from_string(const std::string &s, Kind *k) {
  for (Kind x : {Kind::B, Kind::H, Kind::C, Kind::BARE4, Kind::INT4, Kind::BARE2D, Kind::INT2D,
      Kind::INT2})
    if (s == kind_name(x)) { *k = x; return true; }
  return false;
}
int kind_legs(Kind k) { return (k == Kind::B || k == Kind::BARE4 || k == Kind::INT4) ? 4 : 2; }

namespace {
struct JKey {
  uint32_t m;
  Fac f;
  bool operator<(const JKey &o) const { return m != o.m ? m < o.m : f < o.f; }
};
using Jet = std::map<JKey, double>;

Fac fac_mul(const Fac &a, const Fac &b) {
  Fac r;
  size_t i = 0, j = 0;
  while (i < a.size() || j < b.size()) {
    if (j == b.size() || (i < a.size() && a[i].first < b[j].first)) r.push_back(a[i++]);
    else if (i == a.size() || b[j].first < a[i].first) r.push_back(b[j++]);
    else { int p = a[i].second + b[j].second; if (p) r.push_back({a[i].first, p}); ++i; ++j; }
  }
  return r;
}
void jet_clean(Jet &J) {
  for (auto it = J.begin(); it != J.end();) { if (it->second == 0.0) it = J.erase(it); else ++it; }
}
Jet jet_mul(const Jet &A, const Jet &B) {
  Jet r;
  for (auto &a : A)
    for (auto &b : B) {
      if (a.first.m & b.first.m) continue; // first order in every jet variable
      r[JKey{a.first.m | b.first.m, fac_mul(a.first.f, b.first.f)}] += a.second * b.second;
    }
  jet_clean(r);
  return r;
}
void jet_add(Jet &acc, const Jet &x) { for (auto &t : x) acc[t.first] += t.second; jet_clean(acc); }
Jet jet_shift(const Jet &J, uint32_t bit) {
  Jet r;
  for (auto &t : J) r[JKey{t.first.m | bit, t.first.f}] = t.second;
  return r;
}

// J^(p2/2): J = f0 (mask 0, a single product of atoms with exponent 1, coefficient 1) + n
bool jet_pow(const Jet &J, int p2, Jet *out, std::string *err) {
  const JKey *k0 = nullptr;
  double c0 = 0;
  uint32_t vars = 0;
  for (auto &t : J) {
    if (t.first.m == 0) {
      if (k0) { *err = "jet_pow: several constant terms"; return false; }
      k0 = &t.first;
      c0 = t.second;
    }
    vars |= t.first.m;
  }
  if (!k0 || c0 != 1.0) {
    *err = "jet_pow: constant term missing or coefficient != 1";
    return false;
  }
  Fac f0p, f0inv;
  for (auto &x : k0->f) {
    if (x.second != 2) {
      *err = "jet_pow: constant term not a product of distinct atoms";
      return false;
    }
    f0p.push_back({x.first, p2}); f0inv.push_back({x.first, -2});
  }
  Jet q;
  for (auto &t : J) if (t.first.m) q[JKey{t.first.m, fac_mul(t.first.f, f0inv)}] += t.second;
  jet_clean(q);
  Jet res; res[JKey{0, {}}] = 1.0;
  Jet qk; qk[JKey{0, {}}] = 1.0;
  const double p = 0.5 * p2;
  double binom = 1.0;
  int nv = __builtin_popcount(vars);
  for (int k = 1; k <= nv; ++k) {
    qk = jet_mul(qk, q);
    if (qk.empty()) break;
    binom *= (p - k + 1) / k; // dyadic rationals: exact in double
    for (auto &t : qk) res[t.first] += binom * t.second;
  }
  jet_clean(res);
  Jet f0j; f0j[JKey{0, f0p}] = 1.0;
  *out = jet_mul(res, f0j);
  return true;
}

// mode 0: contract; 1: split (effective line a-b)
struct ChildSpec { uint32_t S, Vm; int mode, a, b; };

class Builder {
 public:
  Builder(const Graph &G, std::vector<AtomDef> &atoms, int maxloops_cap) : G(G), atoms(atoms),
      cap(maxloops_cap) {}
  std::string err;
  // Kirchhoff polynomial (as a constant jet) of the minor of rho with children contracted/split
  // and, if ia>=0, vertices ia and ib identified (2-forest polynomial).  Empty jet = zero
  // polynomial.
  const Jet &poly(uint32_t rho, const std::vector<ChildSpec> &ch, int ia, int ib) {
    std::vector<int> key = {(int)rho, ia, ib};
    for (auto &c : ch) { key.push_back((int)c.S); key.push_back(c.mode); }
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    Jet &J = cache[key];
    int p[32];
    for (int v = 0; v < G.V; ++v) p[v] = v;
    auto fnd = [&](int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; };
    auto un = [&](int x, int y) { x = fnd(x); y = fnd(y); if (x != y) p[x] = y; };
    uint32_t kept = rho;
    for (auto &c : ch) {
      kept &= ~c.S;
      for (int v = 0; v < G.V; ++v)
        if ((c.Vm >> v) & 1u) {
          if (c.mode == 1 && v == c.b) continue;
          un(v, c.mode == 1 ? c.a : (int)__builtin_ctz(c.Vm));
        }
    }
    if (ia >= 0) { if (fnd(ia) == fnd(ib)) return J; un(ia, ib); }
    uint32_t vm = vmask(G, rho);
    std::map<int, int> cls;
    for (int v = 0; v < G.V; ++v)
      if ((vm >> v) & 1u) {
        int r = fnd(v);
        if (!cls.count(r)) { int id = (int)cls.size(); cls[r] = id; }
      }
    const int n = (int)cls.size();
    std::vector<int> eg, ea, eb;
    for (int i = 0; i < G.l(); ++i)
      if ((kept >> i) & 1u) {
        eg.push_back(i);
        ea.push_back(cls[fnd(G.eu[i])]);
        eb.push_back(cls[fnd(G.ev[i])]);
      }
    // Connectivity
    {
      std::vector<int> q(n);
      for (int i = 0; i < n; ++i) q[i] = i;
      std::function<int(int)> f2 = [&](int x) { return q[x] == x ? x : q[x] = f2(q[x]); };
      int nc = n;
      for (size_t k = 0; k < eg.size(); ++k) {
        int x = f2(ea[k]), y = f2(eb[k]);
        if (x != y) { q[x] = y; --nc; }
      }
      if (nc != 1) return J; // disconnected minor: no spanning tree, polynomial 0
    }
    // Blocks (Tarjan, multigraph aware; self-loops are their own blocks)
    std::vector<std::vector<std::pair<int, int>>> adj(n);
    std::vector<std::vector<int>> blocks;
    for (size_t k = 0; k < eg.size(); ++k) {
      if (ea[k] == eb[k]) { blocks.push_back({(int)k}); continue; }
      adj[ea[k]].push_back({eb[k], (int)k}); adj[eb[k]].push_back({ea[k], (int)k});
    }
    std::vector<int> disc(n, 0), low(n, 0), st;
    int timer = 0;
    std::function<void(int, int)> dfs = [&](int u, int pe) {
      disc[u] = low[u] = ++timer;
      for (auto [w, k] : adj[u]) {
        if (k == pe) continue;
        if (!disc[w]) {
          st.push_back(k);
          dfs(w, k);
          low[u] = std::min(low[u], low[w]);
          if (low[w] >= disc[u]) {
            std::vector<int> b;
            while (true) { int x = st.back(); st.pop_back(); b.push_back(x); if (x == k) break; }
            blocks.push_back(b);
          }
        } else if (disc[w] < disc[u]) {
          st.push_back(k);
          low[u] = std::min(low[u], disc[w]);
        }
      }
    };
    for (int u = 0; u < n; ++u) if (!disc[u]) dfs(u, -1);
    Fac f;
    for (auto &b : blocks) {
      if (b.size() == 1 && ea[b[0]] != eb[b[0]]) continue; // bridge: factor 1
      std::vector<int> ks = b;
      std::sort(ks.begin(), ks.end(), [&](int x, int y) { return eg[x] < eg[y]; });
      std::map<int, int> lv;
      std::vector<int> akey;
      AtomDef A;
      for (int k : ks) {
        for (int c : {ea[k], eb[k]}) if (!lv.count(c)) { int id = (int)lv.size(); lv[c] = id; }
        akey.push_back(eg[k]); akey.push_back(lv[ea[k]]); akey.push_back(lv[eb[k]]);
        A.g.push_back(eg[k]); A.u.push_back(lv[ea[k]]); A.v.push_back(lv[eb[k]]);
        A.mask |= 1u << eg[k];
      }
      A.nv = (int)lv.size();
      A.nloops = (int)ks.size() - A.nv + 1;
      int id;
      auto ai = atom_index.find(akey);
      if (ai == atom_index.end()) {
        id = (int)atoms.size();
        atoms.push_back(A);
        atom_index[akey] = id;
      } else id = ai->second;
      f.push_back({id, 2});
    }
    std::sort(f.begin(), f.end());
    for (size_t i = 1; i < f.size(); ++i)
      if (f[i].first == f[i - 1].first) { err = "repeated atom in one minor"; }
    J[JKey{0, f}] = 1.0;
    return J;
  }
  // Builder state.
 private:
  const Graph &G;
  std::vector<AtomDef> &atoms;
  int cap;
  std::map<std::vector<int>, int> atom_index;
  std::map<std::vector<int>, Jet> cache;
};

struct RPart { Part p; int deg; }; // deg 0 (t^0) or 2 (t^2)
} // namespace

// Construction of one class.
// Thread CPU seconds (a class is built by one thread): the cost measure of the builder, independent
// of the load
static double bclock() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}
bool build_class(const Graph &G, const BuildOptions &opt, ClassData &cd, std::string *err) {
  cd = ClassData();
  const double tb0 = bclock();
  cd.G = G;
  cd.kind = opt.kind;
  cd.l = G.l();
  cd.L = G.loops();
  const int l = cd.l, L = cd.L;
  if (G.E() != kind_legs(opt.kind)) {
    *err = "graph " + G.name + " has " + std::to_string(G.E()) + " legs, kind needs "
        + std::to_string(kind_legs(opt.kind));
    return false;
  }
  if (l > RB_MAXE) { *err = "too many lines for RB_MAXE"; return false; }
  if (!is_1pi(G)) { *err = "graph " + G.name + " is not 1PI"; return false; }
  cd.S = sym_factor(G);
  cd.on = on_factor(G);
  cd.is1vr = is_1vr(G);
  cd.selfloop = has_selfloop(G);
  const bool rbar = (opt.kind == Kind::B || opt.kind == Kind::H || opt.kind == Kind::C);
  const bool deriv = (opt.kind == Kind::H || opt.kind == Kind::BARE2D || opt.kind == Kind::INT2D);
  const bool sunsets_only = (opt.kind == Kind::INT4 || opt.kind == Kind::INT2D
      || opt.kind == Kind::INT2);
  // int2: the whole graph is not subtracted, so the two-loop sunset itself (Gamma(0) prefactor) is
  // outside the kind; in the intermediate scheme its subtracted value is zero and route B starts at
  // three loops
  if (opt.kind == Kind::INT2 && L < 3) {
    *err = "int2 starts at three loops (the subtracted sunset vanishes)";
    return false;
  }
  const int ndot = opt.kind == Kind::C ? 2 : (opt.kind == Kind::B || opt.kind == Kind::H) ? 1 : 0;
  cd.pw = l + ndot - 1.5 * L + (deriv ? 1 : 0);
  cd.pref = std::tgamma(cd.pw) * std::pow(M_PI, -0.5 * L);
  // Candidate renormalisation parts: proper, connected, bridgeless line subsets with 2 or 4 legs.
  // In the intermediate-scheme kinds the only parts are the sunsets, triple lines between two
  // vertices: found directly, without the search over the 2^l subsets, which was most of the build
  // time of those kinds at high orders
  std::vector<Part> cand;
  std::vector<uint32_t> subsets;
  if (sunsets_only) {
    for (int u = 0; u < G.V; ++u) for (int v = u + 1; v < G.V; ++v) {
      uint32_t S = 0; int m = 0;
      for (int i = 0; i < l; ++i)
        if ((G.eu[i] == u && G.ev[i] == v) || (G.eu[i] == v && G.ev[i] == u)) { S |= 1u << i; ++m; }
      if (m == 3 && S != G.full()) subsets.push_back(S);
    }
  } else if (rbar) {
    for (uint32_t S = 1; S < G.full(); ++S) subsets.push_back(S);
  }
  if (rbar || sunsets_only) {
    for (uint32_t S : subsets) {
      if (!connected(G, S) || !bridgeless(G, S)) continue;
      int legs = legs_of(G, S);
      if (legs != 2 && legs != 4) continue;
      Part P; P.S = S; P.Vm = vmask(G, S); P.legs = legs;
      if (sunsets_only
          && !(legs == 2 && __builtin_popcount(S) == 3 && __builtin_popcount(P.Vm) == 2))
        continue;
      if (legs == 2) {
        std::vector<int> t;
        for (int v = 0; v < G.V; ++v) if ((P.Vm >> v) & 1u) {
          int d = G.ext[v];
          for (int i = 0; i < l; ++i)
            if (!((S >> i) & 1u)) { if (G.eu[i] == v) ++d; if (G.ev[i] == v) ++d; }
          for (int k = 0; k < d; ++k) t.push_back(v);
        }
        if (t.size() != 2) { *err = "internal: 2-point part with != 2 legs"; return false; }
        P.a = t[0]; P.b = t[1];
      }
      cand.push_back(P);
    }
  }
  cd.nparts = (int)cand.size();
  // Root terminals (p^2 derivative)
  int ra = -1, rb = -1;
  if (deriv) {
    std::vector<int> t;
    for (int v = 0; v < G.V; ++v) for (int k = 0; k < G.ext[v]; ++k) t.push_back(v);
    ra = t[0]; rb = t[1];
  }
  // Dot configurations
  struct DotCfg { int d0, d1; double mult; };
  std::vector<DotCfg> dots;
  if (ndot == 1) for (int e = 0; e < l; ++e) dots.push_back({e, -1, 1.0});
  else if (ndot == 2)
    for (int e = 0; e < l; ++e)
      for (int f = e; f < l; ++f) dots.push_back({e, f, e == f ? 1.0 : 2.0});
  else dots.push_back({-1, -1, 1.0});
  cd.ndots = (int)dots.size();
  Builder B(G, cd.atoms, RB_MAXLOOP);
  std::map<std::pair<std::pair<int, int>, Fac>, double> total;
  std::vector<CompDef> raw;
  for (auto &dc : dots) {
    std::vector<RPart> parts;
    for (auto &P : cand) {
      int k = 0;
      if (dc.d0 >= 0 && ((P.S >> dc.d0) & 1u)) ++k;
      if (dc.d1 >= 0 && ((P.S >> dc.d1) & 1u)) ++k;
      int deg = -1;
      if (sunsets_only) deg = 0;
      else if (P.legs == 4 && k == 0) deg = 0;
      else if (P.legs == 2 && k == 0) deg = 2;
      else if (P.legs == 2 && k == 1) deg = 0;
      if (deg >= 0) parts.push_back({P, deg});
    }
    const int np = (int)parts.size();
    auto compat = [&](int i, int j) {
      uint32_t a = parts[i].p.S, b = parts[j].p.S;
      if ((a & ~b) == 0 || (b & ~a) == 0) return true;
      if (opt.vertex_disjoint) return (parts[i].p.Vm & parts[j].p.Vm) == 0;
      return (a & b) == 0;
    };
    std::vector<int> cur;
    std::function<void(int)> rec;
    bool fail = false;
    rec = [&](int i) {
      if (fail) return;
      if (i == np) {
        // Forest term.
        cd.nforests++;
        std::vector<int> nodes = cur;
        std::sort(nodes.begin(), nodes.end(), [&](int x, int y) {
          return __builtin_popcount(parts[x].p.S) < __builtin_popcount(parts[y].p.S);
        });
        const int nn = (int)nodes.size();
        auto Sof = [&](int k) { return k < nn ? parts[nodes[k]].p.S : G.full(); };
        std::vector<int> parent(nn + 1, -1);
        for (int k = 0; k < nn; ++k) {
          int best = nn;
          for (int j = 0; j < nn; ++j) {
            if (j == k) continue;
            uint32_t a = Sof(k), b = Sof(j);
            if ((a & ~b) == 0 && a != b && __builtin_popcount(b) < __builtin_popcount(Sof(best)))
              best = j;
          }
          parent[k] = best;
        }
        std::vector<int> bit(nn + 1, -1);
        int nbits = 0;
        for (int k = 0; k < nn; ++k) {
          const auto &rp = parts[nodes[k]];
          if (rp.deg == 2 && rp.p.a != rp.p.b) bit[k] = nbits++;
        }
        std::vector<Jet> eps(nn + 1);
        Jet term; term[JKey{0, {}}] = (cur.size() % 2) ? -1.0 : 1.0;
        for (int k = 0; k <= nn; ++k) {
          std::vector<int> ch;
          for (int j = 0; j < nn; ++j) if (parent[j] == k) ch.push_back(j);
          std::vector<int> ch2;
          for (int j : ch) if (bit[j] >= 0) ch2.push_back(j);
          const bool isroot = (k == nn);
          const bool needW = isroot ? deriv : (bit[k] >= 0);
          int ta = isroot ? ra : parts[nodes[k]].p.a, tb = isroot ? rb : parts[nodes[k]].p.b;
          Jet U, W;
          for (uint32_t s = 0; s < (1u << ch2.size()); ++s) {
            std::vector<ChildSpec> spec;
            Jet pre; pre[JKey{0, {}}] = 1.0;
            for (int j : ch) {
              const auto &rp = parts[nodes[j]].p;
              int pos = (int)(std::find(ch2.begin(), ch2.end(), j) - ch2.begin());
              bool split = pos < (int)ch2.size() && ((s >> pos) & 1u);
              spec.push_back({rp.S, rp.Vm, split ? 1 : 0, rp.a, rp.b});
              if (split) pre = jet_mul(pre, eps[j]);
            }
            if (pre.empty()) continue;
            const Jet &pu = B.poly(Sof(k), spec, -1, -1);
            jet_add(U, jet_mul(pre, pu));
            if (needW && ta != tb) {
              const Jet &pwj = B.poly(Sof(k), spec, ta, tb);
              jet_add(W, jet_mul(pre, pwj));
            }
          }
          if (!B.err.empty()) { *err = B.err; fail = true; return; }
          Jet Um3;
          if (!jet_pow(U, -3, &Um3, err)) { fail = true; return; }
          Jet fk = Um3;
          if (isroot && deriv) {
            Jet Um2;
            if (!jet_pow(U, -2, &Um2, err)) { fail = true; return; }
            fk = jet_mul(fk, jet_mul(W, Um2));
          }
          if (!isroot && bit[k] >= 0) {
            Jet Um2;
            if (!jet_pow(U, -2, &Um2, err)) { fail = true; return; }
            eps[k] = jet_shift(jet_mul(W, Um2), 1u << bit[k]);
          }
          term = jet_mul(term, fk);
          if (term.empty()) break;
        }
        if (!term.empty()) cd.nterms_nonzero_forests++;
        for (auto &t : term) {
          cd.nterms_raw++;
          if (opt.prune) total[{{dc.d0, dc.d1}, t.first.f}] += dc.mult * t.second;
          else raw.push_back(CompDef{dc.mult * t.second, dc.d0, dc.d1, t.first.f});
        }
        return;
      }
      rec(i + 1);
      for (int j : cur) if (!compat(i, j)) return;
      cur.push_back(i);
      rec(i + 1);
      cur.pop_back();
    };
    rec(0);
    if (fail) return false;
  }
  // Components
  if (opt.prune) {
    for (auto &t : total)
      if (t.second != 0.0)
        cd.comps.push_back(CompDef{t.second, t.first.first.first, t.first.first.second,
          t.first.second});
  } else cd.comps = raw;
  // Keep only referenced atoms, re-index
  std::vector<int> remap(cd.atoms.size(), -1);
  std::vector<AtomDef> used;
  for (auto &c : cd.comps)
    for (auto &f : c.f)
      if (remap[f.first] < 0) {
        remap[f.first] = (int)used.size();
        used.push_back(cd.atoms[f.first]);
      }
  for (auto &c : cd.comps) {
    for (auto &f : c.f) f.first = remap[f.first];
    std::sort(c.f.begin(), c.f.end());
  }
  cd.atoms = used;
  cd.ncomp = (int)cd.comps.size();
  cd.natoms = (int)cd.atoms.size();
  for (auto &a : cd.atoms) cd.max_atom_loops = std::max(cd.max_atom_loops, a.nloops);
  if (cd.natoms > RB_MAXATOMS) {
    *err = "too many atoms (" + std::to_string(cd.natoms) + ") for RB_MAXATOMS";
    return false;
  }
  if (cd.max_atom_loops > RB_MAXLOOP) { *err = "atom with too many loops"; return false; }
  for (auto &a : cd.atoms)
    if (a.nv > RB_MAXV) { *err = "atom with too many vertices"; return false; }
  // Flat arrays
  cd.atom_off.push_back(0);
  for (auto &a : cd.atoms) {
    for (size_t k = 0; k < a.g.size(); ++k) {
      cd.ae_u.push_back((signed char)a.u[k]);
      cd.ae_v.push_back((signed char)a.v[k]);
    }
    cd.atom_off.push_back((int)cd.ae_u.size());
    cd.atom_nv.push_back(a.nv);
    std::vector<signed char> loc(RB_MAXE, -1);
    for (size_t k = 0; k < a.g.size(); ++k) loc[a.g[k]] = (signed char)k;
    cd.atom_loc.insert(cd.atom_loc.end(), loc.begin(), loc.end());
  }
  // Group terms with identical atom products (sorted by Fac, then dots)
  std::stable_sort(cd.comps.begin(), cd.comps.end(), [](const CompDef &x, const CompDef &y) {
    if (x.f != y.f) return x.f < y.f;
    if (x.d0 != y.d0) return x.d0 < y.d0;
    return x.d1 < y.d1;
  });
  cd.grp_off.push_back(0);
  cd.grp_foff.push_back(0);
  for (size_t k = 0; k < cd.comps.size(); ++k) {
    const auto &c = cd.comps[k];
    if (k > 0 && c.f != cd.comps[k - 1].f) {
      cd.grp_off.push_back((int)k);
      for (auto &f : cd.comps[k - 1].f) {
        cd.cf_atom.push_back(f.first);
        cd.cf_p2.push_back((signed char)f.second);
      }
      cd.grp_foff.push_back((int)cd.cf_atom.size());
    }
    cd.comp_coeff.push_back(c.c);
    cd.comp_d0.push_back((signed char)c.d0);
    cd.comp_d1.push_back((signed char)c.d1);
  }
  if (!cd.comps.empty()) {
    cd.grp_off.push_back((int)cd.comps.size());
    for (auto &f : cd.comps.back().f) {
      cd.cf_atom.push_back(f.first);
      cd.cf_p2.push_back((signed char)f.second);
    }
    cd.grp_foff.push_back((int)cd.cf_atom.size());
  }
  cd.ngroup = (int)cd.grp_off.size() - 1;
  if (cd.ngroup < 0) cd.ngroup = 0;
  const double tb1 = bclock();
  cd.t_expand = tb1 - tb0;
  if (cd.ncomp == 0 || !opt.need_tables) return true;
  // Forest-aware exponent table: 2 omega(S) = 2|S| + min_k (Sum_a p2_ka null_a(S) + 2 #dots_k in S)
  const uint32_t NS = 1u << l;
  cd.omega2.assign(NS, 0);
  cd.omega.assign(NS, 0.0);
  cd.Jt.assign(NS, 0.0);
  std::vector<int> nul(cd.natoms);
  cd.omega_min = 1e300;
  cd.omega2_pre_min = INT_MAX;
  cd.nneg_faces = 0;
  for (uint32_t S = 1; S + 1 < NS; ++S) {
    for (int a = 0; a < cd.natoms; ++a) {
      const AtomDef &A = cd.atoms[a];
      int p[RB_MAXV];
      for (int v = 0; v < A.nv; ++v) p[v] = v;
      int r = 0, cnt = 0;
      for (size_t k = 0; k < A.g.size(); ++k) {
        if (!((S >> A.g[k]) & 1u)) continue;
        ++cnt;
        int x = A.u[k], y = A.v[k];
        while (p[x] != x) x = p[x];
        while (p[y] != y) y = p[y];
        if (x != y) { p[x] = y; ++r; }
      }
      nul[a] = cnt - r;
    }
    int best = INT_MAX;
    for (auto &c : cd.comps) {
      int d = 0;
      for (auto &f : c.f) d += f.second * nul[f.first];
      if (c.d0 >= 0 && ((S >> c.d0) & 1u)) d += 2;
      if (c.d1 >= 0 && ((S >> c.d1) & 1u)) d += 2;
      best = std::min(best, d);
    }
    int w2 = 2 * __builtin_popcount(S) + best;
    cd.omega2_pre_min = std::min(cd.omega2_pre_min, w2);
    // A face with pre-shift 2 omega < 0 (possible with nested t^2 subtractions at 7-8 loops; none
    // exists up to 6 loops) is integrable only through the cancellation inside the t^2 jets, which
    // is evaluated as a sum of separately exponentiated terms: the round-off error eps * Sum|terms|
    // ~ lambda^omega_pre near the face then gives weights with infinite variance.  Refused unless
    // explicitly allowed.
    if (w2 < 0) cd.nneg_faces++;
    // No subtraction in the bare kinds: a face with 2 omega <= 0 is a genuine (sunset)
    // subdivergence
    if (w2 <= 0 && (opt.kind == Kind::BARE4 || opt.kind == Kind::BARE2D)) {
      *err = "divergent bare integral (face with 2*omega = " + std::to_string(w2)
          + ", e.g. a sunset subgraph): use int4 / int2d";
      return false;
    }
    if (w2 <= 0) { w2 += opt.omega_shift; cd.nshift++; }
    if (w2 <= 0) {
      *err = "non-integrable face in the density (2 omega = " + std::to_string(w2) + ")";
      return false;
    }
    cd.omega2[S] = w2;
    cd.omega[S] = 0.5 * w2 * opt.omega_scale;
    cd.omega_min = std::min(cd.omega_min, cd.omega[S]);
  }
  if (cd.nneg_faces > 0 && !opt.allow_negative_faces) {
    *err = std::to_string(cd.nneg_faces) + " face(s) with pre-shift 2*omega = "
        + std::to_string(cd.omega2_pre_min) + " < 0: the FP64 evaluation of the t^2 jets is not "
        "stable there (use --allow-negative-faces to run anyway)";
    return false;
  }
  if (NS <= 2) cd.omega2_pre_min = 0;
  for (uint32_t S = 1; S < NS; ++S) {
    if (__builtin_popcount(S) == 1) { cd.Jt[S] = 1.0; continue; }
    double j = 0;
    for (uint32_t T = S; T; T &= T - 1) {
      uint32_t Sb = S & ~(T & (~T + 1));
      j += cd.Jt[Sb] / cd.omega[Sb];
    }
    cd.Jt[S] = j;
  }
  cd.logJE = std::log(cd.Jt[NS - 1]);
  if (NS <= 2) cd.omega_min = 0; // l = 1: no proper faces
  cd.t_tables = bclock() - tb1;
  return true;
}

size_t ClassData::host_bytes() const {
  size_t b = sizeof(ClassData) + omega2.capacity() * sizeof(int)
      + (omega.capacity() + Jt.capacity()) * sizeof(double);
  for (auto &c : comps) b += sizeof(CompDef) + c.f.capacity() * sizeof(c.f[0]) + 16;
  for (auto &a : atoms)
    b += sizeof(AtomDef) + (a.g.capacity() + a.u.capacity() + a.v.capacity()) * sizeof(int) + 48;
  b += (atom_off.capacity() + atom_nv.capacity() + grp_off.capacity() + grp_foff.capacity()
      + cf_atom.capacity()) * sizeof(int);
  b += ae_u.capacity() + ae_v.capacity() + atom_loc.capacity() + cf_p2.capacity()
      + comp_d0.capacity() + comp_d1.capacity();
  b += comp_coeff.capacity() * sizeof(double) + on.capacity() * sizeof(double);
  return b;
}

RbClassView ClassData::view() const {
  RbClassView v;
  v.l = l; v.natoms = natoms; v.ncomp = ncomp; v.pw = pw; v.logJE = logJE;
  v.omega = omega.data(); v.Jt = Jt.data();
  v.atom_off = atom_off.data(); v.atom_nv = atom_nv.data(); v.ae_u = ae_u.data();
  v.ae_v = ae_v.data();
  v.atom_loc = atom_loc.data();
  v.ngroup = ngroup; v.grp_off = grp_off.data(); v.grp_foff = grp_foff.data();
  v.cf_atom = cf_atom.data(); v.cf_p2 = cf_p2.data();
  v.comp_coeff = comp_coeff.data(); v.comp_d0 = comp_d0.data(); v.comp_d1 = comp_d1.data();
  return v;
}

template <class R> R eval_point(const ClassData &cd, const double *lx0, R *sabs) {
  const int l = cd.l;
  double mx = -1e300;
  for (int e = 0; e < l; ++e) mx = std::max(mx, lx0[e]);
  double lxd[RB_MAXE];
  for (int e = 0; e < l; ++e) lxd[e] = lx0[e] - mx;
  signed char ord[RB_MAXE];
  {
    std::vector<int> o(l);
    for (int e = 0; e < l; ++e) o[e] = e;
    std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return lxd[a] < lxd[b]; });
    for (int e = 0; e < l; ++e) ord[e] = (signed char)o[e];
  }
  RbClassView v = cd.view();
  v.logJE = 0;
  std::vector<R> logP(std::max(1, cd.natoms));
  return rb_weight_at<R>(v, ord, lxd, 0.0, logP.data(), sabs);
}
template double eval_point<double>(const ClassData &, const double *, double *);
template long double eval_point<long double>(const ClassData &, const double *, long double *);
