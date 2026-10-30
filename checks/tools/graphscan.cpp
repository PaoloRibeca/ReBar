/*
    graphscan.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Fast scan of a graph list for the reproducibility checks (checks/G1,
    G2, G3): reads plain phi^4 graphs as GAU strings (the format of
    graphs/lists, e.g. "6(.2 1^2.2 1^6.1 1^2)") or as Nickel indices
    (the first tab-separated field containing '|', as rebar reads
    integrator/data/lists), and for every graph computes the structure
    (vertex degrees, connectedness, one-particle irreducibility, loops,
    legs, self-loops), the order of its automorphism group at the level of
    half-edges with unlabelled legs, |Aut G| = |Aut_V| prod_(i<j) m_ij!
    prod_i 2^(m_ii) m_ii! prod_i legs_i!, and its canonical Nickel index
    (the lexicographic minimum over vertex labellings with '|' < 'e' < '0'
    < ... < '9' < 'A' < ..., by branch and bound, the convention of
    graphs/tools/gaugraph.py and src/graph.cpp). It prints a summary (one
    "key value..." line per item, see the end of main) and, with
    --nickel-out FILE, the canonical Nickel index of every graph in input
    order.
    With --on it also counts, for every graph, the 3^V vertex pairings of
    its O(N) factor by closed index loops (by a transfer matrix over the
    vertices, see OnCount) and prints their sums by (legs, vertices, |Aut
    G|); --on-out FILE writes the counts of every graph in input order (as
    integrator/tools/lists/listcheck does).
    It is a compiled companion of gaugraph.py for the 9- and 10-loop lists
    (1.6 million graphs), which the Python tools would take hours to scan;
    checks/G1 verifies that both give the same Nickel indices and weights
    on every list they both read, checks/G2 that the pairing counts equal
    listcheck's on every graph it counts.

    usage: graphscan [--nickel-out FILE] [--on] [--on-out FILE] LIST
      (OpenMP: OMP_NUM_THREADS threads)
    build: g++ -O2 -std=c++17 -fopenmp -o graphscan graphscan.cpp

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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

static const int MAXV = 16;

struct G {
  int V = 0;
  int m[MAXV][MAXV]; // line multiplicities; m[i][i] = self-loop lines at i
  int legs[MAXV];
};

// Readers.
static bool read_int(const std::string &s, size_t &i, long &v) {
  while (i < s.size() && isspace((unsigned char)s[i])) ++i;
  size_t j = i;
  while (j < s.size() && isdigit((unsigned char)s[j])) ++j;
  if (j == i) return false;
  v = atol(s.substr(i, j - i).c_str());
  i = j;
  return true;
}

// Run-length encoded section up to the closing character: ".k" = k zeros, "v" = one v, "v^k" = k
// copies of v
static bool read_runs(const std::string &s, size_t &i, char close, std::vector<long> &out,
    std::string &err) {
  for (;;) {
    while (i < s.size() && isspace((unsigned char)s[i])) ++i;
    if (i >= s.size()) { err = "unterminated section"; return false; }
    if (s[i] == close) { ++i; return true; }
    if (s[i] == '.') {
      ++i;
      long k;
      if (!read_int(s, i, k)) { err = "bad '.k'"; return false; }
      out.insert(out.end(), k, 0);
      continue;
    }
    long v, k = 1;
    if (!read_int(s, i, v)) { err = std::string("unexpected '") + s[i] + "'"; return false; }
    if (v == 0) { err = "literal 0"; return false; }
    size_t j = i;
    while (j < s.size() && isspace((unsigned char)s[j])) ++j;
    if (j < s.size() && s[j] == '^') {
      i = j + 1;
      if (!read_int(s, i, k)) { err = "bad '^k'"; return false; }
    }
    out.insert(out.end(), k, v);
  }
}

static bool parse_gau(const std::string &s, G &g, std::string &err) {
  size_t i = 0;
  long V;
  if (!read_int(s, i, V) || V < 1 || V > MAXV) { err = "bad vertex count"; return false; }
  while (i < s.size() && s[i] != '(') ++i;
  if (i >= s.size()) { err = "no '('"; return false; }
  ++i;
  g.V = (int)V;
  memset(g.m, 0, sizeof g.m);
  while (i < s.size() && isspace((unsigned char)s[i])) ++i;
  if (i < s.size() && s[i] == '<') { err = "coloured vertices (not a plain list)"; return false; }
  std::vector<long> diag, up;
  if (i < s.size() && s[i] == '[') { ++i; if (!read_runs(s, i, ']', diag, err)) return false; }
  if (!read_runs(s, i, ')', up, err)) return false;
  if ((long)up.size() != V * (V - 1) / 2 || (!diag.empty() && (long)diag.size() != V)) {
    err = "wrong section length";
    return false;
  }
  for (size_t k = 0; k < diag.size(); ++k) {
    if (diag[k] > 4) { err = "typed lines (code > 4)"; return false; }
    g.m[k][k] = (int)diag[k];
  }
  size_t q = 0;
  for (int a = 0; a < V; ++a)
    for (int b = a + 1; b < V; ++b) {
      const long c = up[q++];
      if (c > 4) { err = "typed lines (code > 4)"; return false; }
      g.m[a][b] = g.m[b][a] = (int)c;
    }
  for (int a = 0; a < V; ++a) {
    int d = 2 * g.m[a][a];
    for (int b = 0; b < V; ++b) if (b != a) d += g.m[a][b];
    if (d > 4) { err = "vertex of degree > 4"; return false; }
    g.legs[a] = 4 - d;
  }
  return true;
}

static int label_of(char c) {
  return c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'Z' ? c - 'A' + 10 : -1;
}
static char label_char(int x) { return x < 10 ? (char)('0' + x) : (char)('A' + x - 10); }

static bool parse_nickel(const std::string &s, G &g, std::string &err) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : s) { if (c == '|') { parts.push_back(cur); cur.clear(); } else cur += c; }
  if (!cur.empty()) parts.push_back(cur);
  if (parts.empty() || (int)parts.size() > MAXV) { err = "bad vertex count"; return false; }
  g.V = (int)parts.size();
  memset(g.m, 0, sizeof g.m);
  for (int a = 0; a < g.V; ++a) {
    g.legs[a] = 0;
    for (char c : parts[a]) {
      if (c == 'e') { ++g.legs[a]; continue; }
      const int b = label_of(c);
      if (b < 0 || b >= g.V) { err = std::string("bad label '") + c + "'"; return false; }
      if (a == b) ++g.m[a][a]; else { ++g.m[a][b]; ++g.m[b][a]; }
    }
  }
  for (int a = 0; a < g.V; ++a) {
    int d = 2 * g.m[a][a] + g.legs[a];
    for (int b = 0; b < g.V; ++b) if (b != a) d += g.m[a][b];
    if (d != 4) {
      err = "vertex " + std::to_string(a) + " of degree " + std::to_string(d);
      return false;
    }
  }
  return true;
}

// Structure.
static int components(const G &g, int skip_a = -1, int skip_b = -1) {
  int par[MAXV];
  for (int v = 0; v < g.V; ++v) par[v] = v;
  auto f = [&](int x) { while (par[x] != x) x = par[x] = par[par[x]]; return x; };
  for (int a = 0; a < g.V; ++a)
    for (int b = a + 1; b < g.V; ++b) {
      int k = g.m[a][b] - (a == skip_a && b == skip_b ? 1 : 0);
      if (k > 0) par[f(a)] = f(b);
    }
  int n = 0;
  for (int v = 0; v < g.V; ++v) n += f(v) == v;
  return n;
}

static bool is_1pi(const G &g) {
  if (components(g) != 1) return false;
  for (int a = 0; a < g.V; ++a)
    for (int b = a + 1; b < g.V; ++b)
      if (g.m[a][b] == 1 && components(g, a, b) > 1) return false;
  return true;
}

// Automorphisms.
static uint64_t fact(int n) { uint64_t r = 1; for (int k = 2; k <= n; ++k) r *= k; return r; }

struct Aut {
  const G *g;
  int col[MAXV], ord[MAXV], img[MAXV];
  bool used[MAXV];
  uint64_t count;
  void bt(int k) {
    if (k == g->V) { ++count; return; }
    const int u = ord[k];
    for (int w = 0; w < g->V; ++w) {
      if (used[w] || col[w] != col[u]) continue;
      if (g->m[u][u] != g->m[w][w]) continue;
      bool ok = true;
      for (int j = 0; j < k && ok; ++j) ok = g->m[u][ord[j]] == g->m[w][img[ord[j]]];
      if (!ok) continue;
      used[w] = true; img[u] = w;
      bt(k + 1);
      used[w] = false;
    }
  }
  uint64_t vertex_group(const G &gg) {
    g = &gg;
    const int V = gg.V;
    // Colour refinement: start from (legs, self-loops, multiset of the row), refine by the
    // neighbours' colours
    std::vector<std::vector<int>> key(V);
    for (int v = 0; v < V; ++v) {
      key[v] = {gg.legs[v], gg.m[v][v]};
      std::vector<int> row;
      for (int w = 0; w < V; ++w) if (w != v && gg.m[v][w]) row.push_back(gg.m[v][w]);
      std::sort(row.begin(), row.end());
      key[v].insert(key[v].end(), row.begin(), row.end());
    }
    int ncol = 0;
    for (int round = 0; round <= V; ++round) {
      std::map<std::vector<int>, int> id;
      for (int v = 0; v < V; ++v) id.emplace(key[v], 0);
      int c = 0;
      for (auto &x : id) x.second = c++;
      for (int v = 0; v < V; ++v) col[v] = id[key[v]];
      if (c == ncol) break;
      ncol = c;
      for (int v = 0; v < V; ++v) {
        std::vector<int> nb;
        for (int w = 0; w < V; ++w)
          if (w != v && gg.m[v][w]) nb.push_back(col[w] * 8 + gg.m[v][w]);
        std::sort(nb.begin(), nb.end());
        key[v] = {col[v]};
        key[v].insert(key[v].end(), nb.begin(), nb.end());
      }
    }
    // Order: breadth first from a vertex of the smallest colour class, so that adjacency prunes
    // early
    std::vector<int> size(V, 0);
    for (int v = 0; v < V; ++v) ++size[col[v]];
    bool seen[MAXV] = {false};
    int n = 0;
    while (n < V) {
      int s = -1;
      for (int v = 0; v < V; ++v) if (!seen[v] && (s < 0 || size[col[v]] < size[col[s]])) s = v;
      seen[s] = true; ord[n++] = s;
      for (int h = n - 1; h < n; ++h)
        for (int w = 0; w < V; ++w)
          if (!seen[w] && gg.m[ord[h]][w] && w != ord[h]) { seen[w] = true; ord[n++] = w; }
    }
    for (int v = 0; v < V; ++v) used[v] = false;
    count = 0;
    bt(0);
    return count;
  }
};

static bool aut_order(const G &g, uint64_t &out) {
  Aut A;
  unsigned __int128 a = A.vertex_group(g);
  for (int v = 0; v < g.V; ++v) {
    a *= fact(g.legs[v]);
    a *= ((uint64_t)1 << g.m[v][v]) * fact(g.m[v][v]);
    for (int w = v + 1; w < g.V; ++w) a *= fact(g.m[v][w]);
  }
  if (a >> 63) return false;
  out = (uint64_t)a;
  return true;
}

// O(N) factor.
// Pairing counts of the O(N) factor: every vertex carries S_ijkl = (d_ij d_kl + d_ik d_jl + d_il
// d_jk)/3, so T_G(N) = 3^-V sum over the 3^V choices of one pairing of the four half-edges at every
// vertex of N^(closed index loops) (the strands through the external legs are open; src/graph.cpp
// on_factor).  cnt[k] = number of choices with k closed loops, computed by a transfer matrix over
// the vertices instead of the 3^V enumeration: after the first t vertices of an order, a state is
// the way the processed strands join the cut lines (those from the t vertices to the others) in
// pairs, a cut line whose strand runs to an external leg being "open", and each state carries its
// counts by closed loops so far.  Adding a vertex joins its four half-edges in one of the three
// ways; a join of the two ends of one strand closes a loop.  States with the same joins merge, so
// the work grows with the number of such matchings of the cut lines (at most 15 lines wide here, 4
// bits per line in the key), not with 3^V; checks/G2 compares the counts with listcheck's
// enumeration graph by graph.
static const int ON_MAXK = 2 * MAXV + 2; // closed loops: at most two per vertex
struct OnCount {
  struct St { uint64_t key; uint32_t c[ON_MAXK]; };
  std::vector<St> cur, nxt;
  std::vector<int> table, stamp;
  int gen = 0;
  // The order of the vertices: greedy (most lines into the processed set, then fewest out), from
  // every start; the one with the smallest largest cut, then the smallest total cut
  int order(const G &g, int *best) {
    const int V = g.V;
    int bmax = 1 << 30, bsum = 1 << 30;
    for (int s0 = 0; s0 < V; ++s0) {
      int ord[MAXV], in[MAXV] = {0}, deg[MAXV] = {0};
      bool done[MAXV] = {false};
      for (int a = 0; a < V; ++a) for (int b = 0; b < V; ++b) if (b != a) deg[a] += g.m[a][b];
      int cut = 0, mx = 0, sm = 0;
      for (int t = 0; t < V; ++t) {
        int v = s0;
        if (t) {
          int bk = -(1 << 30);
          for (int w = 0; w < V; ++w)
            if (!done[w]) {
              const int k = 8 * in[w] - (deg[w] - in[w]);
              if (k > bk) { bk = k; v = w; }
            }
        }
        done[v] = true; ord[t] = v;
        cut += deg[v] - 2 * in[v];
        for (int w = 0; w < V; ++w) if (!done[w]) in[w] += g.m[v][w];
        mx = std::max(mx, cut); sm += cut;
      }
      if (mx < bmax || (mx == bmax && sm < bsum)) {
        bmax = mx;
        bsum = sm;
        std::copy(ord, ord + V, best);
      }
    }
    return bmax;
  }
  // cnt[0..ON_MAXK): pairing counts by closed loops; false if a cut is wider than 15 lines (4-bit
  // positions)
  bool run(const G &g, uint64_t *cnt) {
    const int V = g.V;
    int ord[MAXV];
    if (order(g, ord) > 15) return false;
    static const int pr[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    if (table.empty()) { table.assign(1 << 12, 0); stamp.assign(1 << 12, 0); }
    cur.clear();
    St s0; s0.key = 0; std::fill(s0.c, s0.c + ON_MAXK, 0); s0.c[0] = 1;
    cur.push_back(s0);
    int cutv[16], w = 0, kmax = 0; // the unprocessed vertex of each cut line, by position
    bool done[MAXV] = {false};
    // Half-edge ids: 0-15 old cut positions, 16-31 their open ends, 32-35 v's halves of its new
    // lines, 36-39 their far halves (the new cut positions), 40-43 v's legs, 44-47 their open ends,
    // 48-55 v's self-loops
    int partner[56], posmap[56];
    for (int t = 0; t < V; ++t) {
      const int v = ord[t];
      done[v] = true;
      int slot[4], ns = 0, np = 0, ncutv[16], nk = 0, nx = 0, nl = 0;
      std::fill(posmap, posmap + 56, 15);
      for (int p = 0; p < w; ++p)
        if (cutv[p] == v) slot[ns++] = p; else { ncutv[np] = cutv[p]; posmap[p] = np++; }
      for (int u = 0; u < V; ++u)
        if (!done[u]) for (int r = 0; r < g.m[v][u]; ++r) {
          slot[ns++] = 32 + nk;
          ncutv[np] = u;
          posmap[36 + nk] = np++;
          ++nk;
        }
      for (int r = 0; r < g.legs[v]; ++r) slot[ns++] = 40 + nx++;
      for (int r = 0; r < g.m[v][v]; ++r) {
        slot[ns++] = 48 + 2 * nl;
        slot[ns++] = 49 + 2 * nl;
        ++nl;
      }
      if (ns != 4 || np > 15) return false;
      ++gen;
      nxt.clear();
      const int mask = (int)table.size() - 1;
      for (const St &st : cur) {
        for (int c = 0; c < 3; ++c) {
          for (int p = 0; p < w; ++p) {
            const int x = (int)((st.key >> (4 * p)) & 15);
            if (x == 15) { partner[p] = 16 + p; partner[16 + p] = p; } else partner[p] = x;
          }
          for (int k = 0; k < nk; ++k) { partner[32 + k] = 36 + k; partner[36 + k] = 32 + k; }
          for (int k = 0; k < nx; ++k) { partner[40 + k] = 44 + k; partner[44 + k] = 40 + k; }
          for (int k = 0; k < nl; ++k) {
            partner[48 + 2 * k] = 49 + 2 * k;
            partner[49 + 2 * k] = 48 + 2 * k;
          }
          int closed = 0;
          for (int j = 0; j < 2; ++j) {
            const int a = slot[pr[c][2 * j]], b = slot[pr[c][2 * j + 1]], pa = partner[a],
                pb = partner[b];
            if (pa == b) ++closed; else { partner[pa] = pb; partner[pb] = pa; }
          }
          uint64_t key = 0;
          for (int p = 0; p < w; ++p)
            if (cutv[p] != v) key |= (uint64_t)posmap[partner[p]] << (4 * posmap[p]);
          for (int k = 0; k < nk; ++k)
            key |= (uint64_t)posmap[partner[36 + k]] << (4 * posmap[36 + k]);
          uint64_t h = key * 0x9E3779B97F4A7C15ull;
          int i = (int)(h >> 52) & mask, id = -1;
          for (;; i = (i + 1) & mask) {
            if (stamp[i] != gen) {
              stamp[i] = gen;
              table[i] = id = (int)nxt.size();
              nxt.emplace_back();
              nxt.back().key = key;
              std::fill(nxt.back().c, nxt.back().c + ON_MAXK, 0);
              break;
            }
            if (nxt[table[i]].key == key) { id = table[i]; break; }
          }
          if (2 * nxt.size() > table.size()) return false;
          uint32_t *d = nxt[id].c;
          for (int k = 0; k <= kmax; ++k) d[k + closed] += st.c[k];
        }
      }
      kmax += 2;
      cur.swap(nxt);
      w = np;
      std::copy(ncutv, ncutv + np, cutv);
    }
    if (cur.size() != 1 || cur[0].key != 0 || w != 0) return false;
    for (int k = 0; k < ON_MAXK; ++k) cnt[k] = cur[0].c[k];
    return true;
  }
};

// Nickel index.
struct Nickel {
  const G *g;
  int V, pos[MAXV], order[MAXV], n;
  std::vector<int> best, key;
  bool have;
  // Exact key of the placed prefix: per placed vertex its legs (1), the sorted labels >= its own of
  // its neighbours (2 + label; unplaced neighbours will get larger labels, so the key stops there),
  // then '|' (0)
  bool prefix_key() {
    key.clear();
    for (int k = 0; k < n; ++k) {
      const int v = order[k];
      for (int i = 0; i < g->legs[v]; ++i) key.push_back(1);
      int labs[4 * MAXV], nl = 0;
      bool unplaced = false;
      for (int w = 0; w < V; ++w) {
        if (!g->m[v][w]) continue;
        if (pos[w] < 0) { unplaced = true; continue; }
        if (pos[w] >= k) for (int c = 0; c < g->m[v][w]; ++c) labs[nl++] = pos[w];
      }
      std::sort(labs, labs + nl);
      for (int i = 0; i < nl; ++i) key.push_back(2 + labs[i]);
      if (unplaced) return false;
      key.push_back(0);
    }
    return n == V;
  }
  void rec() {
    const bool complete = prefix_key();
    if (have) {
      const size_t L = std::min(key.size(), best.size());
      int cmp = 0;
      for (size_t i = 0; i < L && !cmp; ++i) cmp = key[i] < best[i] ? -1 : key[i] > best[i] ? 1 : 0;
      if (cmp > 0) return;
      if (complete) { if (cmp < 0 || key.size() < best.size()) best = key; return; }
    } else if (complete) { best = key; have = true; return; }
    int cand[MAXV], nc = 0;
    for (int k = 0; k < n && !nc; ++k)
      for (int w = 0; w < V; ++w) if (g->m[order[k]][w] && pos[w] < 0) cand[nc++] = w;
    if (!nc) for (int w = 0; w < V; ++w) if (pos[w] < 0) cand[nc++] = w;
    for (int i = 0; i < nc; ++i) {
      const int w = cand[i];
      pos[w] = n; order[n++] = w;
      rec();
      pos[w] = -1; --n;
    }
  }
  std::string run(const G &gg) {
    g = &gg; V = gg.V; n = 0; have = false;
    for (int v = 0; v < V; ++v) pos[v] = -1;
    rec();
    std::string s;
    for (int r : best) s += r == 0 ? '|' : r == 1 ? 'e' : label_char(r - 2);
    return s;
  }
};

// Main.
struct Row {
  bool ok = false, nickel_input = false;
  int V = 0, loops = 0, E = 0, selfloops = 0, comps = 0;
  bool pi = false, aut_ok = false;
  uint64_t aut = 0;
  std::string input, nickel, err, on; // on: pairing counts c0,c1,... (with --on-out)
};

int main(int argc, char **argv) {
  std::string list, nkout, onout;
  bool on = false;
  const char *usage = "usage: graphscan [--nickel-out FILE] [--on] [--on-out FILE] LIST\n";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--nickel-out") && i + 1 < argc) nkout = argv[++i];
    else if (!strcmp(argv[i], "--on")) on = true;
    else if (!strcmp(argv[i], "--on-out") && i + 1 < argc) { onout = argv[++i]; on = true; }
    else if (argv[i][0] == '-') { fprintf(stderr, "%s", usage); return 2; }
    else list = argv[i];
  }
  if (list.empty()) { fprintf(stderr, "%s", usage); return 2; }
  std::ifstream f(list);
  if (!f) { fprintf(stderr, "cannot open %s\n", list.c_str()); return 2; }
  std::vector<Row> rows;
  std::string line;
  while (std::getline(f, line)) {
    Row r;
    size_t d = line.find_first_not_of(" \t");
    size_t e = d == std::string::npos ? d : line.find_first_not_of("0123456789", d);
    // GAU string: the vertex count, then '('
    if (e != std::string::npos && e > d && line.find_first_not_of(" ", e) != std::string::npos
        && line[line.find_first_not_of(" ", e)] == '(') {
      size_t a = line.find_first_not_of(" \t\r"), b = line.find_last_not_of(" \t\r");
      r.input = line.substr(a, b - a + 1);
    } else { // Nickel list: the first tab-separated field containing '|'
      if (line.empty() || line[0] == '#') continue;
      size_t a = 0;
      while (a <= line.size()) {
        size_t b = line.find('\t', a);
        if (b == std::string::npos) b = line.size();
        std::string t;
        for (size_t k = a; k < b; ++k) if (!isspace((unsigned char)line[k])) t += line[k];
        if (t.find('|') != std::string::npos) { r.input = t; break; }
        a = b + 1;
      }
      r.nickel_input = true;
    }
    if (!r.input.empty()) rows.push_back(std::move(r));
  }
  const long n = (long)rows.size();
  struct OnSum { long graphs = 0; uint64_t c[ON_MAXK] = {0}; };
  // (E, V, |Aut G|) -> graphs, summed pairing counts
  std::map<std::tuple<int, int, uint64_t>, OnSum> on_sums;
  long on_failed = 0;
#pragma omp parallel
  {
    OnCount oc;
    std::map<std::tuple<int, int, uint64_t>, OnSum> local;
    long failed = 0;
#pragma omp for schedule(dynamic, 256)
    for (long i = 0; i < n; ++i) {
      Row &r = rows[i];
      G g;
      if (!(r.nickel_input ? parse_nickel(r.input, g, r.err) : parse_gau(r.input, g, r.err)))
        continue;
      r.ok = true;
      r.V = g.V;
      int nl = 0;
      for (int a = 0; a < g.V; ++a) {
        r.E += g.legs[a]; r.selfloops += g.m[a][a];
        for (int b = a; b < g.V; ++b) nl += g.m[a][b];
      }
      r.comps = components(g);
      r.pi = r.comps == 1 && is_1pi(g);
      r.loops = nl - g.V + r.comps;
      r.aut_ok = aut_order(g, r.aut);
      Nickel NK;
      r.nickel = NK.run(g);
      if (on) {
        uint64_t c[ON_MAXK], tot = 0, all = 1;
        bool ok = oc.run(g, c);
        for (int k = 0; ok && k < ON_MAXK; ++k) tot += c[k];
        for (int v = 0; v < g.V; ++v) all *= 3;
        if (!ok || tot != all) { ++failed; r.on = "FAILED"; continue; }
        int top = ON_MAXK - 1;
        while (top > 0 && !c[top]) --top;
        if (!onout.empty())
          for (int k = 0; k <= top; ++k) r.on += (k ? "," : "") + std::to_string(c[k]);
        if (r.aut_ok) {
          OnSum &t = local[std::make_tuple(r.E, r.V, r.aut)];
          ++t.graphs;
          for (int k = 0; k <= top; ++k) t.c[k] += c[k];
        }
      }
    }
#pragma omp critical
    {
      on_failed += failed;
      for (auto &x : local) {
        OnSum &t = on_sums[x.first];
        t.graphs += x.second.graphs;
        for (int k = 0; k < ON_MAXK; ++k) t.c[k] += x.second.c[k];
      }
    }
  }
  long errors = 0, disconnected = 0, not1pi = 0, canon_listed = 0, nickel_input = 0, overflow = 0;
  std::map<int, long> loops, legs, selfloops, vertices;
  std::map<std::pair<int, uint64_t>, long> auts; // (E, |Aut|) -> graphs
  std::unordered_set<std::string> inputs, nickels;
  FILE *fo = nkout.empty() ? nullptr : fopen(nkout.c_str(), "w");
  if (!nkout.empty() && !fo) { fprintf(stderr, "cannot write %s\n", nkout.c_str()); return 2; }
  FILE *fon = onout.empty() ? nullptr : fopen(onout.c_str(), "w");
  if (!onout.empty() && !fon) { fprintf(stderr, "cannot write %s\n", onout.c_str()); return 2; }
  for (long i = 0; i < n; ++i) {
    const Row &r = rows[i];
    if (!r.ok) {
      if (errors < 5)
        fprintf(stderr, "graph %ld (%s): %s\n", i + 1, r.input.c_str(), r.err.c_str());
      ++errors;
      if (fo) fprintf(fo, "ERROR\n");
      if (fon) fprintf(fon, "ERROR\n");
      continue;
    }
    inputs.insert(r.input);
    nickels.insert(r.nickel);
    if (r.comps != 1) ++disconnected; else if (!r.pi) ++not1pi;
    ++loops[r.loops]; ++legs[r.E]; ++selfloops[r.selfloops]; ++vertices[r.V];
    if (r.aut_ok) ++auts[{r.E, r.aut}]; else ++overflow;
    if (r.nickel_input) { ++nickel_input; canon_listed += r.nickel == r.input; }
    if (fo) fprintf(fo, "%s\n", r.nickel.c_str());
    if (fon) fprintf(fon, "%s\n", r.on.c_str());
  }
  if (fo) fclose(fo);
  if (fon) fclose(fon);
  auto hist = [](const char *name, const std::map<int, long> &h) {
    printf("%s", name);
    for (auto &x : h) printf(" %d:%ld", x.first, x.second);
    printf("\n");
  };
  printf("graphs %ld\nerrors %ld\ndisconnected %ld\nnot_1pi %ld\naut_overflow %ld\n", n, errors,
    disconnected, not1pi, overflow);
  hist("vertices", vertices); hist("loops", loops); hist("legs", legs);
  hist("selfloops", selfloops);
  printf("distinct_inputs %zu\ndistinct_nickel %zu\nnickel_inputs %ld\ncanonical_as_listed %ld\n",
    inputs.size(), nickels.size(), nickel_input, canon_listed);
  for (auto &x : auts)
    printf("aut %d %llu %ld\n", x.first.first, (unsigned long long)x.first.second, x.second);
  // on E V |Aut| graphs c0,c1,...: the pairing counts summed over the graphs with these E, V and
  // |Aut|
  if (on) {
    printf("on_failed %ld\n", on_failed);
    for (auto &x : on_sums) {
      int top = ON_MAXK - 1;
      while (top > 0 && !x.second.c[top]) --top;
      printf("on %d %d %llu %ld ", std::get<0>(x.first), std::get<1>(x.first),
        (unsigned long long)std::get<2>(x.first), x.second.graphs);
      for (int k = 0; k <= top; ++k)
        printf("%s%llu", k ? "," : "", (unsigned long long)x.second.c[k]);
      printf("\n");
    }
  }
  return 0;
}
