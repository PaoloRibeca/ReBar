/*
    graph.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Implementation of graph.h.

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

#include "graph.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <set>

static int vlabel(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'Z') return 10 + c - 'A';
  if (c >= 'a' && c <= 'z' && c != 'e') return 10 + c - 'a';
  return -1;
}
static char vchar(int v) { return v < 10 ? (char)('0' + v) : (char)('A' + v - 10); }

bool parse_nickel(const std::string &s0, Graph &G, std::string *err) {
  std::string s;
  for (char c : s0) if (!isspace((unsigned char)c)) s += c;
  std::vector<std::string> parts;
  std::string cur;
  for (char c : s) { if (c == '|') { parts.push_back(cur); cur.clear(); } else cur += c; }
  if (!cur.empty()) parts.push_back(cur); // tolerate a missing final '|'
  G = Graph();
  G.V = (int)parts.size();
  G.ext.assign(G.V, 0);
  G.name = s0;
  for (int i = 0; i < G.V; ++i)
    for (char c : parts[i]) {
      if (c == 'e') { G.ext[i]++; continue; }
      int j = vlabel(c);
      if (j < 0 || j >= G.V) {
        if (err) *err = "bad vertex label '" + std::string(1, c) + "' in " + s0;
        return false;
      }
      G.eu.push_back(i); G.ev.push_back(j);
    }
  if (G.l() > 32) { if (err) *err = "more than 32 lines"; return false; }
  return degree4(G, err);
}

bool degree4(const Graph &G, std::string *err) {
  std::vector<int> d(G.ext);
  for (int i = 0; i < G.l(); ++i) { d[G.eu[i]]++; d[G.ev[i]]++; }
  for (int v = 0; v < G.V; ++v)
    if (d[v] != 4) {
      if (err)
        *err = "vertex " + std::to_string(v) + " has degree " + std::to_string(d[v]) + " in "
            + G.name;
      return false;
    }
  return true;
}

std::string nickel_string(const Graph &G) {
  std::vector<std::vector<int>> nb(G.V);
  for (int i = 0; i < G.l(); ++i) {
    int a = std::min(G.eu[i], G.ev[i]), b = std::max(G.eu[i], G.ev[i]);
    nb[a].push_back(b);
  }
  std::string s;
  for (int v = 0; v < G.V; ++v) {
    s += std::string(G.ext[v], 'e');
    std::sort(nb[v].begin(), nb[v].end());
    for (int w : nb[v]) s += vchar(w);
    s += '|';
  }
  return s;
}

Graph relabel(const Graph &G, const std::vector<int> &p) {
  Graph H = G;
  for (int v = 0; v < G.V; ++v) H.ext[p[v]] = G.ext[v];
  for (int i = 0; i < G.l(); ++i) { H.eu[i] = p[G.eu[i]]; H.ev[i] = p[G.ev[i]]; }
  return H;
}

// Nickel-like canonical string: lexicographic minimum over relabellings with 'e' < vertex labels
// and the separator '|' sorting before everything (a shorter vertex list precedes its extensions).
static int nk_rank(char c) {
  return c == '|' ? 0 : c == 'e' ? 1 : 2 + (vlabel(c) >= 0 ? vlabel(c) : 99);
}
static bool nk_less(const std::string &a, const std::string &b) {
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return nk_rank(a[i]) < nk_rank(b[i]);
  return a.size() < b.size();
}
std::string nickel_canonical(const Graph &G) {
  std::vector<int> p(G.V);
  std::iota(p.begin(), p.end(), 0);
  std::string best;
  bool have = false;
  do {
    std::string s = nickel_string(relabel(G, p));
    if (!have || nk_less(s, best)) { best = s; have = true; }
  } while (std::next_permutation(p.begin(), p.end()));
  return best;
}

// Subgraph predicates on line bitmasks.
uint32_t vmask(const Graph &G, uint32_t S) {
  uint32_t m = 0;
  for (int i = 0; i < G.l(); ++i) if ((S >> i) & 1u) m |= (1u << G.eu[i]) | (1u << G.ev[i]);
  return m;
}
static int uf_find(int *p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
int ncomponents(const Graph &G, uint32_t S, uint32_t verts) {
  int p[32];
  for (int v = 0; v < G.V; ++v) p[v] = v;
  int nc = __builtin_popcount(verts);
  for (int i = 0; i < G.l(); ++i)
    if ((S >> i) & 1u) {
      int a = uf_find(p, G.eu[i]), b = uf_find(p, G.ev[i]);
      if (a != b) { p[a] = b; --nc; }
    }
  return nc;
}
bool connected(const Graph &G, uint32_t S) { return S && ncomponents(G, S, vmask(G, S)) == 1; }
bool bridgeless(const Graph &G, uint32_t S) {
  uint32_t vm = vmask(G, S);
  int c0 = ncomponents(G, S, vm);
  for (int i = 0; i < G.l(); ++i)
    if ((S >> i) & 1u) {
      if (G.eu[i] == G.ev[i]) continue;
      if (ncomponents(G, S & ~(1u << i), vm) != c0) return false;
    }
  return true;
}
int rank_of(const Graph &G, uint32_t S) {
  uint32_t vm = vmask(G, S);
  return __builtin_popcount(vm) - ncomponents(G, S, vm);
}
int nullity(const Graph &G, uint32_t S) { return __builtin_popcount(S) - rank_of(G, S); }
int legs_of(const Graph &G, uint32_t S) {
  return 4 * __builtin_popcount(vmask(G, S)) - 2 * __builtin_popcount(S);
}
bool is_1pi(const Graph &G) {
  uint32_t all = (G.V >= 32) ? 0xffffffffu : ((1u << G.V) - 1u);
  return ncomponents(G, G.full(), all) == 1 && bridgeless(G, G.full());
}
bool is_1vr(const Graph &G) {
  if (G.V < 3) return false;
  for (int v = 0; v < G.V; ++v) {
    uint32_t S = 0;
    for (int i = 0; i < G.l(); ++i) if (G.eu[i] != v && G.ev[i] != v) S |= 1u << i;
    uint32_t verts = ((1u << G.V) - 1u) & ~(1u << v);
    if (ncomponents(G, S, verts) != 1) return true;
  }
  return false;
}
bool has_selfloop(const Graph &G) {
  for (int i = 0; i < G.l(); ++i) if (G.eu[i] == G.ev[i]) return true;
  return false;
}

// Symmetry and tensor factors.
static double factorial(int n) { double r = 1; for (int i = 2; i <= n; ++i) r *= i; return r; }

double aut_order(const Graph &G) {
  const int V = G.V;
  std::vector<std::vector<int>> m(V, std::vector<int>(V, 0));
  std::vector<int> loops(V, 0);
  for (int i = 0; i < G.l(); ++i) {
    if (G.eu[i] == G.ev[i]) loops[G.eu[i]]++;
    else { m[G.eu[i]][G.ev[i]]++; m[G.ev[i]][G.eu[i]]++; }
  }
  // Count vertex permutations preserving ext, loops, multiplicities (backtracking)
  std::vector<int> img(V, -1), used(V, 0);
  double naut = 0;
  std::function<void(int)> bt = [&](int k) {
    if (k == V) { naut += 1; return; }
    for (int w = 0; w < V; ++w) {
      if (used[w] || G.ext[w] != G.ext[k] || loops[w] != loops[k]) continue;
      bool ok = true;
      for (int j = 0; j < k && ok; ++j) if (m[k][j] != m[w][img[j]]) ok = false;
      if (!ok) continue;
      img[k] = w; used[w] = 1;
      bt(k + 1);
      used[w] = 0; img[k] = -1;
    }
  };
  bt(0);
  double a = naut;
  for (int i = 0; i < V; ++i)
    for (int j = i + 1; j < V; ++j) a *= factorial(m[i][j]);
  for (int v = 0; v < V; ++v)
    a *= std::pow(2.0, loops[v]) * factorial(loops[v]) * factorial(G.ext[v]);
  return a;
}
double sym_factor(const Graph &G) { return factorial(G.E()) / aut_order(G); }

// O(N) factor: each vertex carries S_ijkl = (d_ij d_kl + d_ik d_jl + d_il d_jk)/3; sum over the 3^V
// pairings, N per closed index loop; the external tensor is projected on the symmetric structure
// (Gamma4) or delta (Gamma2), which after symmetrisation over leg labellings amounts to summing all
// open-strand patterns.
std::vector<double> on_factor(const Graph &G) {
  const int V = G.V, l = G.l(), E = G.E();
  const int H = 2 * l + E;
  std::vector<std::vector<int>> slots(V);
  for (int i = 0; i < l; ++i) {
    slots[G.eu[i]].push_back(2 * i);
    slots[G.ev[i]].push_back(2 * i + 1);
  }
  int k = 0;
  for (int v = 0; v < V; ++v) for (int j = 0; j < G.ext[v]; ++j) slots[v].push_back(2 * l + k++);
  std::vector<double> c(l + 2, 0.0);
  std::vector<int> choice(V, 0);
  const double w = std::pow(3.0, -V);
  long total = 1;
  for (int v = 0; v < V; ++v) total *= 3;
  std::vector<int> p(H);
  for (long t = 0; t < total; ++t) {
    long x = t;
    for (int v = 0; v < V; ++v) { choice[v] = (int)(x % 3); x /= 3; }
    std::iota(p.begin(), p.end(), 0);
    auto un = [&](int a, int b) {
      a = uf_find(p.data(), a);
      b = uf_find(p.data(), b);
      if (a != b) p[a] = b;
    };
    for (int i = 0; i < l; ++i) un(2 * i, 2 * i + 1);
    static const int pr[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    for (int v = 0; v < V; ++v) {
      const auto &s = slots[v];
      un(s[pr[choice[v]][0]], s[pr[choice[v]][1]]);
      un(s[pr[choice[v]][2]], s[pr[choice[v]][3]]);
    }
    std::vector<char> open(H, 0);
    for (int h = 2 * l; h < H; ++h) open[uf_find(p.data(), h)] = 1;
    int closed = 0;
    for (int h = 0; h < H; ++h) if (uf_find(p.data(), h) == h && !open[h]) ++closed;
    c[closed] += w;
  }
  while (c.size() > 1 && c.back() == 0) c.pop_back();
  return c;
}
double eval_poly(const std::vector<double> &c, double N) {
  double r = 0;
  for (int i = (int)c.size() - 1; i >= 0; --i) r = r * N + c[i];
  return r;
}

// Graph generator.
std::vector<Graph> generate_graphs(int V, int E, bool selfloops) {
  std::vector<Graph> out;
  std::set<std::vector<uint8_t>> seen;
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < V; ++i) for (int j = i + 1; j < V; ++j) pairs.push_back({i, j});
  std::vector<int> ext(V), loops(V), rem(V);
  std::vector<int> mult(pairs.size());
  std::vector<std::vector<int>> perms;
  {
    std::vector<int> p(V);
    std::iota(p.begin(), p.end(), 0);
    do perms.push_back(p); while (std::next_permutation(p.begin(), p.end()));
  }
  auto key_of = [&](const std::vector<int> &pm) {
    std::vector<int> inv(V);
    for (int i = 0; i < V; ++i) inv[pm[i]] = i;
    std::vector<uint8_t> k;
    for (int v = 0; v < V; ++v) k.push_back((uint8_t)ext[inv[v]]);
    for (int v = 0; v < V; ++v) k.push_back((uint8_t)loops[inv[v]]);
    std::vector<std::vector<int>> m(V, std::vector<int>(V, 0));
    for (size_t q = 0; q < pairs.size(); ++q) {
      m[pairs[q].first][pairs[q].second] = m[pairs[q].second][pairs[q].first] = mult[q];
    }
    for (int a = 0; a < V; ++a)
      for (int b = a + 1; b < V; ++b) k.push_back((uint8_t)m[inv[a]][inv[b]]);
    return k;
  };
  std::function<void(size_t)> rec_mult = [&](size_t q) {
    if (q == pairs.size()) {
      for (int v = 0; v < V; ++v) if (rem[v] != 0) return;
      Graph G; G.V = V; G.ext = ext;
      for (int v = 0; v < V; ++v)
        for (int s = 0; s < loops[v]; ++s) { G.eu.push_back(v); G.ev.push_back(v); }
      for (size_t r = 0; r < pairs.size(); ++r)
        for (int s = 0; s < mult[r]; ++s) {
          G.eu.push_back(pairs[r].first);
          G.ev.push_back(pairs[r].second);
        }
      if (G.l() > 32 || !is_1pi(G)) return;
      std::vector<uint8_t> best;
      for (auto &pm : perms) { auto k = key_of(pm); if (best.empty() || k < best) best = k; }
      if (!seen.insert(best).second) return;
      // Rebuild graph in canonical labelling
      Graph C; C.V = V; C.ext.assign(best.begin(), best.begin() + V);
      for (int v = 0; v < V; ++v)
        for (int s = 0; s < best[V + v]; ++s) { C.eu.push_back(v); C.ev.push_back(v); }
      size_t idx = 2 * V;
      for (int a = 0; a < V; ++a)
        for (int b = a + 1; b < V; ++b) {
          for (int s = 0; s < best[idx]; ++s) { C.eu.push_back(a); C.ev.push_back(b); }
          ++idx;
        }
      C.name = nickel_canonical(C);
      out.push_back(C);
      return;
    }
    int i = pairs[q].first, j = pairs[q].second;
    // Vertex i must be saturated after its last pair
    for (int m = std::min(rem[i], rem[j]); m >= 0; --m) {
      rem[i] -= m; rem[j] -= m; mult[q] = m;
      bool ok = true;
      if (j == V - 1 && rem[i] != 0) ok = false;
      if (ok) rec_mult(q + 1);
      rem[i] += m; rem[j] += m;
    }
  };
  std::function<void(int, int)> rec_loops = [&](int v, int dummy) {
    (void)dummy;
    if (v == V) {
      for (int u = 0; u < V; ++u) rem[u] = 4 - ext[u] - 2 * loops[u];
      rec_mult(0);
      return;
    }
    int mx = selfloops ? (4 - ext[v]) / 2 : 0;
    for (int s = 0; s <= mx; ++s) { loops[v] = s; rec_loops(v + 1, 0); }
  };
  std::function<void(int, int, int)> rec_ext = [&](int v, int left, int maxv) {
    if (v == V) { if (left == 0) rec_loops(0, 0); return; }
    for (int x = std::min(left, maxv); x >= 0; --x) { ext[v] = x; rec_ext(v + 1, left - x, x); }
  };
  rec_ext(0, E, V == 1 ? 4 : 2);
  std::sort(out.begin(), out.end(), [](const Graph &a, const Graph &b) {
    return nk_less(a.name, b.name);
  });
  return out;
}

// Cycle matroid.
std::vector<uint32_t> circuits(const Graph &G) {
  std::set<uint32_t> out;
  std::vector<std::vector<std::pair<int, int>>> adj(G.V);
  for (int i = 0; i < G.l(); ++i) {
    if (G.eu[i] == G.ev[i]) { out.insert(1u << i); continue; }
    adj[G.eu[i]].push_back({G.ev[i], i}); adj[G.ev[i]].push_back({G.eu[i], i});
  }
  for (int s = 0; s < G.V; ++s) {
    std::function<void(int, uint32_t, uint32_t)> dfs = [&](int v, uint32_t usedv, uint32_t usede) {
      for (auto [w, i] : adj[v]) {
        if ((usede >> i) & 1u) continue;
        if (w < s) continue;
        if (w == s) { if (usede) out.insert(usede | (1u << i)); continue; }
        if ((usedv >> w) & 1u) continue;
        dfs(w, usedv | (1u << w), usede | (1u << i));
      }
    };
    dfs(s, 1u << s, 0);
  }
  return std::vector<uint32_t>(out.begin(), out.end());
}

bool matroid_iso(const Graph &A, const Graph &B, std::vector<int> *emap) {
  const int l = A.l();
  if (B.l() != l) return false;
  auto CA = circuits(A), CB = circuits(B);
  if (CA.size() != CB.size()) return false;
  auto profile = [&](const std::vector<uint32_t> &C) {
    std::vector<std::vector<int>> P(l);
    for (uint32_t c : C)
      for (int e = 0; e < l; ++e) if ((c >> e) & 1u) P[e].push_back(__builtin_popcount(c));
    for (auto &p : P) std::sort(p.begin(), p.end());
    return P;
  };
  auto PA = profile(CA), PB = profile(CB);
  {
    auto a = PA, b = PB;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    if (a != b) return false;
  }
  std::set<uint32_t> SB(CB.begin(), CB.end());
  std::vector<int> order(l);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
    return PA[x].size() > PA[y].size();
  });
  std::vector<int> pos(l);
  for (int k = 0; k < l; ++k) pos[order[k]] = k;
  std::vector<std::vector<uint32_t>> check(l); // circuits completed when edge order[k] is assigned
  for (uint32_t c : CA) {
    int mx = -1, me = -1;
    for (int e = 0; e < l; ++e) if (((c >> e) & 1u) && pos[e] > mx) { mx = pos[e]; me = e; }
    check[me].push_back(c);
  }
  std::vector<int> phi(l, -1), used(l, 0);
  std::function<bool(int)> bt = [&](int k) {
    if (k == l) return true;
    int e = order[k];
    for (int f = 0; f < l; ++f) {
      if (used[f] || PB[f] != PA[e]) continue;
      phi[e] = f; used[f] = 1;
      bool ok = true;
      for (uint32_t c : check[e]) {
        uint32_t im = 0;
        for (int x = 0; x < l; ++x) if ((c >> x) & 1u) im |= 1u << phi[x];
        if (!SB.count(im)) { ok = false; break; }
      }
      if (ok && bt(k + 1)) return true;
      used[f] = 0; phi[e] = -1;
    }
    return false;
  };
  if (!bt(0)) return false;
  if (emap) *emap = phi;
  return true;
}
