/*
    topologies.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Generates the topologies of the Feynman graphs with a given list of
    vertex valences by vertex insertion.  A vertex of valence d has
    d - deg - 2 t external legs, deg its line ends and t its self-loops,
    so the legs are implicit and the class grown is every connected graph
    with the given valences and any number of legs.  Every such graph with
    V vertices comes from one with V - 1 by inserting a vertex w on a
    slots (a line or a self-loop, split into two lines to w; the same slot
    several times if its multiplicity allows), b legs (moved to w) and t
    self-loops at w: removing a leaf of a spanning tree and re-pairing its
    half-lines is the inverse step, and re-pairing them as far as possible
    leaves at most one leg, so b <= 1 suffices.  When every valence is even
    and the number of legs E is 0, 2 or 4 the class with E legs is grown
    alone, with the insertions that preserve E (a + b + t = d/2): a leaf
    whose legs exceed its line ends cannot be re-paired, and two such
    leaves need E >= 6.  Growth proceeds by levels over the sub-multisets
    of the valences, one hash set of canonical forms per sub-multiset;
    candidates are canonicalised by colour refinement with backtracking, so
    the cost is the number of graphs times a few hundred insertions, not
    the number of labelled pairings that the generator of 2003 enumerated
    and screened.  The program derives from that generator (P. Ribeca,
    2000-2003), whose compact output format and graph classes it keeps.
      topologies [-l L] [-i] [-t] [-c] [-o FILE] [-j THREADS] [-N] [-w]
          DEGREES
    grows the connected graphs with the valences DEGREES (comma-separated,
    1 to 8, at most 12 entries) and writes them in the compact format of
    graphs/lists, with the legs as negative colours when some valence
    differs from 4: -l keeps the graphs with L loops, -i the
    one-particle-irreducible ones, -t includes the graphs with self-loops,
    -c is accepted for compatibility (only connected graphs are grown), -o
    writes to a file instead of standard output, -N writes the integrator
    list (Nickel indices, weights, flags) instead, -w prints the sum of
    E!/|Aut G| over the graphs written, per number of legs;
      topologies -E 2|4 -V VMAX [-o DIR] [-j THREADS] [-N] [-w] [-a] [-G]
    is the phi^4 mode: for every V <= VMAX, with L = V + 1 - E/2 loops, it
    writes DIR/<L>-loop-2 (1PI, self-loops included: the -cit lists) and
    DIR/<L>-loop-2-ci (no self-loops), or DIR/<L>-loop-4 (no self-loops:
    the -ci lists) and DIR/<L>-loop-4-cit, in the labelling of its own
    canonical form; -N adds DIR/gamma<E>_L<L>.txt and _with_selfloops.txt,
    the integrator's lists; -w the zero-dimensional sum rule of every
    order against the exact coefficients it carries; -a every connected
    graph of the order in DIR/<L>-loop-<E>-all; -G relabels the output
    with the getCanonical of the 2003 generator, so that the strings of
    graphs without self-loops and their order equal the stored lists (it
    costs far more than the generation and serves byte comparisons only);
      topologies -r [-x] < LIST
    prints the canonical key of every graph of a list (compact format, or
    the plain integers of gaugraph.py adj) in hexadecimal, so that two
    lists are compared as sets of graphs with sort and cmp;
      topologies -w [-x] < LIST
    prints the sum of E!/|Aut G| over the graphs of a list, per number of
    legs, as an exact fraction (tools/sumrules.py convention);
      topologies -N [-x] < LIST
    prints a list as an integrator list.  A list without a colour section
    is read with the phi^4 convention (4 - deg legs), with -x with no legs
    (the line-end reading of the 2003 generator's degree lists).

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
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <getopt.h>
#include <sys/resource.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "handle.hpp"
#include "permutations.hpp"
#include "partitions.hpp"
#include "generators.hpp"

namespace {
constexpr int MAXV = 13; // vertices; twelve are eleven loops with four legs in phi^4
constexpr int DMAX = 8; // the largest valence
constexpr int WORDS = 6; // 64-bit words of a key
bool zerolegs = false; // -x: a list without colour section has no legs

struct Graph {
  int V = 0;
  uint8_t val[MAXV] = {}; // valences
  uint8_t m[MAXV][MAXV] = {}; // line multiplicities, symmetric, zero diagonal
  uint8_t tad[MAXV] = {}; // self-loops per vertex
  int deg(int i) const { int d = 0; for (int j = 0; j < V; ++j) d += m[i][j]; return d; }
  int legs(int i) const { return val[i] - deg(i) - 2 * tad[i]; }
  int E() const { int e = 0; for (int i = 0; i < V; ++i) e += legs(i); return e; }
  bool connected_without(int a, int b) const { // ignoring one line a-b: is the graph connected?
    bool seen[MAXV] = {}; int stack[MAXV], sp = 0; seen[0] = true; stack[sp++] = 0; int n = 1;
    while (sp) {
      int u = stack[--sp];
      for (int v = 0; v < V; ++v) {
        int k = m[u][v];
        if (k && ((u == a && v == b) || (u == b && v == a))) --k;
        if (k && !seen[v]) { seen[v] = true; stack[sp++] = v; ++n; }
      }
    }
    return n == V;
  }
  bool opi() const { // no bridge: cutting any line of multiplicity one keeps the graph connected
    for (int i = 0; i < V; ++i) for (int j = i + 1; j < V; ++j)
      if (m[i][j] == 1 && !connected_without(i, j)) return false;
    return true;
  }
  bool selfloops() const { for (int i = 0; i < V; ++i) if (tad[i]) return true; return false; }
  bool allfour() const { for (int i = 0; i < V; ++i) if (val[i] != 4) return false; return true; }
};

// A canonical form, bit-packed: for every vertex its valence less one (four bits) and its
//  self-loops (three bits), then four bits per pair of the upper triangle row by row (a pair
//  carries at most eight lines), in the canonical labelling, as a big-endian bit string over six
//  64-bit words (348 bits at twelve vertices); the legs follow from the valence.  The words
//  compare as the string does, and the set of a class keys on them.  With the vertex fields first
//  and all valences equal the order of the keys is that of the former phi^4 layout (one self-loop
//  bit per vertex, two bits per pair), so the phi^4 lists keep their labelling and their order.
struct Key {
  uint8_t V = 0; uint64_t w[WORDS] = {};
  void set(int bit, int nbits, unsigned value) { // bits counted from the most significant end
    int i = bit >> 6, end = (bit & 63) + nbits;
    if (end <= 64) w[i] |= (uint64_t)value << (64 - end);
    else {
      w[i] |= (uint64_t)value >> (end - 64);
      w[i + 1] |= (uint64_t)value << (128 - end);
    }
  }
  unsigned get(int bit, int nbits) const {
    int i = bit >> 6, end = (bit & 63) + nbits; uint64_t mask = (1ULL << nbits) - 1;
    if (end <= 64) return (unsigned)(w[i] >> (64 - end) & mask);
    return (unsigned)((w[i] << (end - 64) | w[i + 1] >> (128 - end)) & mask);
  }
  bool operator<(const Key &o) const {
    for (int i = 0; i < WORDS; ++i) if (w[i] != o.w[i]) return w[i] < o.w[i];
    return V < o.V;
  }
  bool operator==(const Key &o) const {
    if (V != o.V) return false;
    for (int i = 0; i < WORDS; ++i) if (w[i] != o.w[i]) return false;
    return true;
  }
};
struct KeyHash {
  size_t operator()(const Key &k) const {
    uint64_t h = k.V * 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < WORDS; ++i) {
      h ^= (k.w[i] + 0x632BE59BD9B4E019ULL) * 0xBF58476D1CE4E5B9ULL;
      h ^= h >> 29;
      h *= 0x94D049BB133111EBULL;
    }
    return (size_t)(h ^ h >> 31);
  }
};
static_assert(7 * (MAXV - 1) + 4 * (MAXV - 1) * (MAXV - 2) / 2 <= 64 * WORDS,
  "the key must hold seven bits per vertex and four bits per pair");
static_assert(DMAX <= 16 && DMAX / 2 <= 7, "the key fields must hold valence and self-loops");

Graph decode(const Key &k) {
  Graph g; g.V = k.V; int b = 0;
  for (int i = 0; i < g.V; ++i) {
    g.val[i] = (uint8_t)(k.get(b, 4) + 1);
    g.tad[i] = (uint8_t)k.get(b + 4, 3);
    b += 7;
  }
  for (int i = 0; i < g.V; ++i) for (int j = i + 1; j < g.V; ++j) {
    g.m[i][j] = g.m[j][i] = (uint8_t)k.get(b, 4);
    b += 4;
  }
  return g;
}

// Canonical labelling.
// Colour refinement on neighbour bitmasks, then backtracking over the smallest non-singleton cell;
//  no heap allocation.  The signature of a vertex towards a colour class is the multiset of the
//  multiplicities of its lines to the class, as a mixed-radix code with radix floor(DMAX / k) + 1
//  for multiplicity k, injective since sum_k k count_k <= DMAX; for multiplicities up to three it
//  orders the count vectors as the former code 1 + 5 m2 + 25 m3 did, which keeps the phi^4
//  labelling.
typedef std::array<int8_t, MAXV> Col;
constexpr uint16_t WEIGHT[DMAX + 1] = {0, 1, 9, 45, 135, 405, 810, 1620, 3240};
static_assert(DMAX == 8, "WEIGHT lists the mixed-radix weights for DMAX = 8");
struct Canon {
  const Graph &g; int V; int mmax = 0;
  uint16_t nb[DMAX + 1][MAXV]; // nb[k][i]: the vertices joined to i by exactly k lines
  int legs[MAXV];
  Key best; bool have = false;
  // Leaves with the best key: the vertex automorphisms that keep valences, self-loops and legs
  long naut = 0;
  explicit Canon(const Graph &gg) : g(gg), V(gg.V) {
    for (int k = 0; k <= DMAX; ++k) for (int i = 0; i < V; ++i) nb[k][i] = 0;
    for (int i = 0; i < V; ++i) {
      legs[i] = g.legs(i);
      for (int j = 0; j < V; ++j) if (g.m[i][j]) {
        nb[g.m[i][j]][i] |= (uint16_t)(1u << j);
        mmax = std::max<int>(mmax, g.m[i][j]);
      }
    }
  }
  // Refine col (values 0..k-1, ordered) to an equitable colouring; returns the number of colours
  int refine(Col &col) const {
    int k = 0; for (int i = 0; i < V; ++i) k = std::max(k, col[i] + 1);
    for (;;) {
      uint16_t cm[MAXV] = {}; for (int i = 0; i < V; ++i) cm[col[i]] |= (uint16_t)(1u << i);
      uint16_t sig[MAXV][MAXV + 1];
      for (int i = 0; i < V; ++i) {
        sig[i][0] = (uint16_t)col[i];
        for (int c = 0; c < k; ++c) {
          unsigned s = 0;
          for (int x = 1; x <= mmax; ++x) s += WEIGHT[x] * __builtin_popcount(nb[x][i] & cm[c]);
          sig[i][1 + c] = (uint16_t)s;
        }
      }
      auto cmp = [&](int a, int b) {
        for (int c = 0; c <= k; ++c)
          if (sig[a][c] != sig[b][c]) return sig[a][c] < sig[b][c] ? -1 : 1;
        return 0;
      };
      int idx[MAXV]; for (int i = 0; i < V; ++i) idx[i] = i;
      for (int i = 1; i < V; ++i) {
        int x = idx[i], j = i;
        while (j > 0 && cmp(x, idx[j - 1]) < 0) { idx[j] = idx[j - 1]; --j; }
        idx[j] = x;
      }
      int nk = 0; Col ncol{};
      for (int r = 0; r < V; ++r) {
        if (r && cmp(idx[r], idx[r - 1])) ++nk;
        ncol[idx[r]] = (int8_t)nk;
      }
      ++nk;
      col = ncol;
      if (nk == k) return k;
      k = nk;
    }
  }
  // The key of the graph in the labelling where vertex i goes to position col[i] (col discrete)
  Key encode(const Col &col) const {
    int at[MAXV]; for (int i = 0; i < V; ++i) at[col[i]] = i;
    Key s; s.V = (uint8_t)V; int b = 0;
    for (int q = 0; q < V; ++q) {
      s.set(b, 4, g.val[at[q]] - 1u);
      s.set(b + 4, 3, g.tad[at[q]]);
      b += 7;
    }
    for (int q = 0; q < V; ++q) for (int r = q + 1; r < V; ++r) {
      s.set(b, 4, g.m[at[q]][at[r]]);
      b += 4;
    }
    return s;
  }
  void search(Col col) {
    int k = refine(col);
    if (k == V) {
      Key s = encode(col);
      if (!have || s < best) { best = s; have = true; naut = 1; } else if (s == best) ++naut;
      return;
    }
    int cnt[MAXV] = {}; for (int i = 0; i < V; ++i) ++cnt[col[i]];
    int cell = -1;
    for (int c = 0; c < k; ++c) if (cnt[c] > 1 && (cell < 0 || cnt[c] < cnt[cell])) cell = c;
    for (int v = 0; v < V; ++v) if (col[v] == cell) {
      Col c2 = col;
      for (int u = 0; u < V; ++u) if (c2[u] > cell) ++c2[u]; // open a colour just above the cell
      c2[v] = (int8_t)(cell + 1); // individualise v
      search(c2);
    }
  }
  // The initial colouring is by (valence, legs, self-loops), in that order
  Key run() {
    Col col; int raw[MAXV], vals[MAXV], nv = 0;
    for (int i = 0; i < V; ++i) raw[i] = g.val[i] << 7 | legs[i] << 3 | g.tad[i];
    for (int i = 0; i < V; ++i) {
      int j = 0;
      while (j < nv && vals[j] != raw[i]) ++j;
      if (j == nv) vals[nv++] = raw[i];
    }
    std::sort(vals, vals + nv);
    for (int i = 0; i < V; ++i) col[i] = (int8_t)(std::lower_bound(vals, vals + nv, raw[i]) - vals);
    search(col);
    return best;
  }
};

// The insertions.
// An insertion (a, b, t) of a vertex w of valence d splits a slots (lines or self-loops, two
//  half-lines of w each), takes b legs (one half-line each) and puts t self-loops at w, leaving
//  d - 2a - b - 2t legs at w.  In the general mode every (a, b, t) with b <= 1, a + b >= 1 and
//  2a + b + 2t <= d is used (taking two legs is the same as splitting the line they would form in
//  the parent that has it); when E is fixed the insertions with a + b + t = d/2 and a + b >= 1,
//  which preserve E, are used instead.
struct Slot { int kind; int u, v; }; // kind 0: line u-v (u < v); 1: self-loop at u; 2: leg at u
struct Ins { int a, b, t; };

std::vector<Ins> insertions(int d, bool fixedE) {
  std::vector<Ins> out;
  for (int a = 0; 2 * a <= d; ++a)
    for (int b = 0; 2 * a + b <= d; ++b) {
      if (a + b < 1) continue;
      if (fixedE) {
        int t = d / 2 - a - b;
        if (t >= 0) out.push_back({a, b, t});
      } else if (b <= 1) for (int t = 0; 2 * a + b + 2 * t <= d; ++t) out.push_back({a, b, t});
    }
  return out;
}
// The slots of a graph: the lines and the self-loops, then the legs
void slots(const Graph &g, std::vector<Slot> &lines, std::vector<Slot> &legs) {
  lines.clear();
  legs.clear();
  for (int i = 0; i < g.V; ++i) {
    for (int j = i + 1; j < g.V; ++j) if (g.m[i][j]) lines.push_back({0, i, j});
    if (g.tad[i]) lines.push_back({1, i, i});
    if (g.legs(i)) legs.push_back({2, i, i});
  }
}
int multiplicity(const Graph &g, const Slot &s) {
  return s.kind == 0 ? g.m[s.u][s.v] : s.kind == 1 ? g.tad[s.u] : g.legs(s.u);
}
// Split slot s and attach the new vertex w to its ends
void split(Graph &g, const Slot &s, int w) {
  if (s.kind == 0) {
    --g.m[s.u][s.v];
    --g.m[s.v][s.u];
    ++g.m[s.u][w];
    ++g.m[w][s.u];
    ++g.m[s.v][w];
    ++g.m[w][s.v];
  } else if (s.kind == 1) { --g.tad[s.u]; g.m[s.u][w] += 2; g.m[w][s.u] += 2; }
  else { ++g.m[s.u][w]; ++g.m[w][s.u]; } // the leg moves to w (leg counts follow from the valence)
}

// The set of the graphs of a class, sharded by hash so that the threads rarely wait for each other
struct Level {
  static constexpr int SHARDS = 1024;
  std::unordered_set<Key, KeyHash> set[SHARDS];
  std::mutex lock[SHARDS];
  void insert(const Key &k) {
    size_t h = KeyHash()(k) % SHARDS;
    std::lock_guard<std::mutex> l(lock[h]);
    set[h].insert(k);
  }
  size_t size() const { size_t n = 0; for (auto &s : set) n += s.size(); return n; }
  std::vector<Key> sorted() const {
    std::vector<Key> v; v.reserve(size());
    for (auto &s : set) v.insert(v.end(), s.begin(), s.end());
    std::sort(v.begin(), v.end());
    return v;
  }
};

// The children of a parent p by a vertex of valence d, through the insertions ins, kept when
//  their number of legs lies in [Emin, Emax]; the slots of a multiset are chosen in non-decreasing
//  order, each as often as the multiplicity it has left allows
struct Inserter {
  const Graph &p; int d; Level &out; int Emin, Emax;
  std::vector<Slot> lines, legs; int w, t = 0;
  Inserter(const Graph &pp, int dd, Level &o, int lo, int hi)
      : p(pp), d(dd), out(o), Emin(lo), Emax(hi), w(pp.V) { slots(p, lines, legs); }
  void run(const std::vector<Ins> &ins) {
    int E = p.E();
    for (const Ins &x : ins) {
      int lw = d - 2 * x.a - x.b - 2 * x.t, Ec = E - x.b + lw;
      if (Ec < Emin || Ec > Emax) continue;
      Graph g = p; g.V = w + 1; g.val[w] = (uint8_t)d; t = x.t;
      pick(g, lines, 0, x.a, x.b);
    }
  }
  void pick(Graph &g, const std::vector<Slot> &S, size_t from, int left, int b) {
    if (left == 0) {
      if (b > 0) pick(g, legs, 0, b, 0);
      else { g.tad[w] = (uint8_t)t; out.insert(Canon(g).run()); }
      return;
    }
    for (size_t s = from; s < S.size(); ++s) {
      if (multiplicity(g, S[s]) < 1) continue;
      Graph h = g; split(h, S[s], w);
      pick(h, S, s, left - 1, b);
    }
  }
};

// Output in the labelling of the 2003 generator (-G), through its own classes: the compact string
//  of the graph relabelled by getCanonical, with the sort key of the order of the stored lists.
//  Those were written one GAU degree list at a time, in the order of scripts/gau_lists.sh (the
//  entries 4 - legs of the vertices with legs, ascending), each list in the reverse of the order
//  of std::set<topology>: self-loop vector ascending, then upper triangle descending.  The
//  strings of the graphs without self-loops are those of the stored lists; with self-loops they
//  are not, since getCanonical does not reproduce the labelling of the generator's own screen.
struct Relabelled {
  std::vector<int> group, tad, tri; std::string s; bool selfloops = false;
  bool operator<(const Relabelled &o) const {
    if (group != o.group) return group < o.group;
    if (tad != o.tad) return tad < o.tad;
    return tri > o.tri;
  }
};
Relabelled relabel(const Graph &g) {
  int **mat = new int *[g.V]; int *tad = new int[g.V]; Relabelled r;
  for (int i = 0; i < g.V; ++i) {
    mat[i] = new int[g.V];
    for (int j = 0; j < g.V; ++j) mat[i][j] = g.m[i][j];
    tad[i] = g.tad[i];
  }
  std::ostringstream os;
  {
    handledPermutatedTopology *t;
#pragma omp critical(topologies)
    t = new handledPermutatedTopology(g.V, mat, tad); // numOfConnectedGroups keeps static buffers
    // One vertex: only the identity labelling, which the 2003 classes cannot sort
    if (g.V > 1) t->getCanonical();
    for (int i = 0; i < g.V; ++i) {
      int l = g.legs(t->rPermutation()->rCurrentPermutation()[i]);
      if (l) r.group.push_back(4 - l);
      r.tad.push_back(t->rTadpoles(i));
      for (int j = i + 1; j < g.V; ++j) r.tri.push_back(t->rMatrix(i, j));
    }
    std::sort(r.group.begin(), r.group.end());
    topology h(t);
#pragma omp critical(topologies)
    os << h; // ends with its own newline
  }
  for (int i = 0; i < g.V; ++i) delete[] mat[i];
  delete[] mat; delete[] tad;
  r.s = os.str(); r.selfloops = g.selfloops();
  return r;
}

// The compact string of a graph in the labelling of its key, V(<colours>[self-loops]upper
//  triangle), run-length encoded as the 2003 generator wrote it: ".k" k zeros, "v" one v, "v^k" k
//  copies, runs separated by blanks.  The colour section carries the legs as negative colours
//  (colour -k: k legs, the convention of GAU and of tools/gaugraph.py) and is written only when
//  some valence differs from 4, so that the phi^4 lists are unchanged; the self-loop section only
//  when there are self-loops.
void rle(const int *v, int n, std::string &s) {
  for (int i = 0; i < n;) {
    int j = i; while (j < n && v[j] == v[i]) ++j;
    if (v[i] == 0) s += "." + std::to_string(j - i);
    else {
      if (i) s.push_back(' ');
      s += std::to_string(v[i]);
      if (j - i > 1) s += "^" + std::to_string(j - i);
    }
    i = j;
  }
}
std::string compact(const Key &k) {
  Graph g = decode(k); int V = g.V, tri[MAXV * (MAXV - 1) / 2], col[MAXV], tad[MAXV], n = 0;
  bool any = false, four = true;
  for (int i = 0; i < V; ++i) {
    any |= g.tad[i] != 0;
    four &= g.val[i] == 4;
    col[i] = -g.legs(i);
    tad[i] = g.tad[i];
  }
  for (int i = 0; i < V; ++i) for (int j = i + 1; j < V; ++j) tri[n++] = g.m[i][j];
  std::string s = std::to_string(V) + "(";
  if (!four) { s.push_back('<'); rle(col, V, s); s.push_back('>'); }
  if (any) { s.push_back('['); rle(tad, V, s); s.push_back(']'); }
  rle(tri, n, s);
  s += ")\n";
  return s;
}

// Writes the 1PI graphs of a phi^4 order, with and without self-loops, the strings made in
//  parallel block by block; with -G, relabelled and in the order of the stored lists
void write(const std::vector<Key> &level, std::ostream &owith, std::ostream &owithout,
    bool canonical, long &n1, long &n2) {
  n1 = n2 = 0;
  if (canonical) {
    std::vector<Relabelled> rs(level.size()); std::vector<char> keep(level.size(), 0);
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < level.size(); ++i) {
      Graph g = decode(level[i]);
      if (!g.opi()) continue;
      rs[i] = relabel(g);
      keep[i] = 1;
    }
    std::vector<Relabelled> kept;
    for (size_t i = 0; i < level.size(); ++i) if (keep[i]) kept.push_back(std::move(rs[i]));
    std::sort(kept.begin(), kept.end());
    for (const Relabelled &r : kept) {
      owith << r.s; ++n1;
      if (!r.selfloops) { owithout << r.s; ++n2; }
    }
    return;
  }
  const size_t B = 4096; size_t nb = (level.size() + B - 1) / B;
  std::vector<std::string> with(nb), without(nb); std::vector<long> c1(nb, 0), c2(nb, 0);
#pragma omp parallel for schedule(dynamic)
  for (size_t b = 0; b < nb; ++b)
    for (size_t i = b * B; i < std::min(level.size(), (b + 1) * B); ++i) {
      Graph g = decode(level[i]);
      if (!g.opi()) continue;
      std::string s = compact(level[i]);
      with[b] += s; ++c1[b];
      if (!g.selfloops()) { without[b] += s; ++c2[b]; }
    }
  for (size_t b = 0; b < nb; ++b) {
    owith << with[b];
    owithout << without[b];
    n1 += c1[b];
    n2 += c2[b];
  }
}

// Reading lists.
// The compact format (V(<colours>[self-loops]upper triangle), run-length encoded: ".k" k zeros,
//  "v" one v, "v^k" k copies) or plain integers (V, self-loops per vertex, upper triangle:
//  gaugraph.py adj).  The legs of a vertex come from its colour when the section is present (-k:
//  k legs); otherwise every vertex has valence 4 (the phi^4 lists), or no legs with -x.
struct Reader {
  const std::string &s; size_t i = 0;
  explicit Reader(const std::string &t) : s(t) {}
  void blanks() { while (i < s.size() && isspace((unsigned char)s[i])) ++i; }
  int ch() { blanks(); return i < s.size() ? s[i++] : -1; }
  int num() {
    blanks(); size_t j = i;
    while (j < s.size()
        && (isdigit((unsigned char)s[j]) || (j == i && (s[j] == '-' || s[j] == '+'))))
      ++j;
    if (j == i) throw std::runtime_error("integer expected");
    int v = std::stoi(s.substr(i, j - i)); i = j; return v;
  }
  std::vector<int> runs(char close) {
    std::vector<int> out;
    for (int c = ch(); c != close; c = ch()) {
      if (c < 0) throw std::runtime_error("unexpected end");
      if (c == '.') { int n = num(); out.insert(out.end(), n, 0); continue; }
      --i; int v = num(); int n = 1;
      if (ch() == '^') n = num(); else --i;
      out.insert(out.end(), n, v);
    }
    return out;
  }
};

// The valences of a parsed graph from its legs (-1: the phi^4 convention); false if some vertex
//  does not fit its valence
bool valences(Graph &g, const std::vector<int> &legs) {
  for (int i = 0; i < g.V; ++i) {
    int d = g.deg(i) + 2 * g.tad[i], l = legs.empty() ? (zerolegs ? 0 : 4 - d) : legs[i];
    if (l < 0 || d + l > DMAX) return false;
    g.val[i] = (uint8_t)(d + l);
  }
  return true;
}
bool parse(const std::string &line, Graph &g) {
  try {
    if (line.find('(') == std::string::npos) { // plain integers
      Reader r(line); int V = r.num(); if (V < 1 || V >= MAXV) return false;
      g = Graph(); g.V = V;
      for (int i = 0; i < V; ++i) g.tad[i] = (uint8_t)r.num();
      for (int i = 0; i < V; ++i) for (int j = i + 1; j < V; ++j)
        g.m[i][j] = g.m[j][i] = (uint8_t)r.num();
      return valences(g, {});
    }
    Reader r(line); int V = r.num(); if (V < 1 || V >= MAXV) return false;
    while (r.ch() != '(') {}
    std::vector<int> diag(V, 0), tri, legs;
    int c = r.ch();
    if (c == '<') {
      legs = r.runs('>');
      if ((int)legs.size() != V) return false;
      for (int &x : legs) { if (x > 0) return false; x = -x; } // positive colours: GAU's vertices
      c = r.ch();
    }
    if (c == '[') { diag = r.runs(']'); } else --r.i;
    tri = r.runs(')');
    if ((int)diag.size() != V || (int)tri.size() != V * (V - 1) / 2) return false;
    // Without a colour section and without -x the list is a phi^4 one, in which codes above 4 are
    //  lines with numerators: not handled
    int top = legs.empty() && !zerolegs ? 4 : DMAX;
    for (int x : diag) if (x < 0 || 2 * x > top) return false;
    for (int x : tri) if (x < 0 || x > top) return false;
    g = Graph(); g.V = V; size_t k = 0;
    for (int i = 0; i < V; ++i) g.tad[i] = (uint8_t)diag[i];
    for (int i = 0; i < V; ++i) for (int j = i + 1; j < V; ++j) {
      g.m[i][j] = g.m[j][i] = (uint8_t)tri[k++];
    }
    return valences(g, legs);
  } catch (...) { return false; }
}

// The lists are read in blocks of lines, which the threads process together
constexpr size_t BLOCK = 1 << 18;
bool read_block(std::vector<std::string> &lines, long &n) {
  lines.clear(); std::string line;
  while (lines.size() < BLOCK && std::getline(std::cin, line)) {
    ++n;
    if (line.find_first_not_of(" \t\r") != std::string::npos) lines.push_back(line);
  }
  return !lines.empty();
}

// Nickel indices: the lists of the integrator.
// The Nickel index of a graph in the convention of Phi4tools and of tools/gaugraph.py: the vertices
//  are numbered 0, 1, ...; for each in turn, 'e' per leg, then the sorted labels >= its own of the
//  vertices it is joined to, once per line (its own label once per self-loop), then '|'; the index
//  is the minimum of the string over the numberings, with '|' < 'e' < '0' < ... < '9' < 'A' < ...,
//  found by branch and bound as gaugraph.py does it (the next vertex is an unplaced neighbour of
//  the first placed vertex that has one, which the minimum always satisfies).
struct Nickel {
  const Graph &g; int V; int legs[MAXV];
  std::vector<uint8_t> best; bool have = false;
  int order[MAXV], pos[MAXV]; int n = 0;
  explicit Nickel(const Graph &gg) : g(gg), V(gg.V) {
    for (int i = 0; i < V; ++i) { legs[i] = g.legs(i); pos[i] = -1; }
  }
  // The key of the placed prefix: 0 '|', 1 'e', 2 + label; false if a placed vertex has an unplaced
  //  neighbour (then the key stops before that vertex's separator)
  bool prefix(std::vector<uint8_t> &key) const {
    key.clear();
    for (int k = 0; k < n; ++k) {
      int v = order[k];
      for (int e = 0; e < legs[v]; ++e) key.push_back(1);
      int labs[DMAX]; int nl = 0; bool unplaced = false;
      for (int t = 0; t < g.tad[v]; ++t) labs[nl++] = k;
      for (int w = 0; w < V; ++w) if (g.m[v][w]) {
        if (pos[w] < 0) unplaced = true;
        else if (pos[w] >= k) for (int t = 0; t < g.m[v][w]; ++t) labs[nl++] = pos[w];
      }
      std::sort(labs, labs + nl);
      for (int t = 0; t < nl; ++t) key.push_back((uint8_t)(2 + labs[t]));
      if (unplaced) return false;
      key.push_back(0);
    }
    return true;
  }
  void rec() {
    std::vector<uint8_t> key; bool complete = prefix(key) && n == V;
    if (have && key.size() <= best.size() && std::lexicographical_compare(best.begin(),
        best.begin() + key.size(), key.begin(), key.end()))
      return;
    if (have && key.size() > best.size()
        && !(std::lexicographical_compare(key.begin(), key.end(), best.begin(), best.end())))
      return;
    if (complete) { if (!have || key < best) { best = key; have = true; } return; }
    int cand[MAXV]; int nc = 0;
    for (int k = 0; k < n && !nc; ++k) {
      int v = order[k];
      for (int w = 0; w < V; ++w) if (g.m[v][w] && pos[w] < 0) cand[nc++] = w;
    }
    if (!nc) for (int w = 0; w < V; ++w) if (pos[w] < 0) cand[nc++] = w;
    for (int c = 0; c < nc; ++c) {
      int w = cand[c];
      order[n] = w;
      pos[w] = n;
      ++n;
      rec();
      --n;
      pos[w] = -1;
    }
  }
  std::string run() {
    rec(); std::string s;
    for (uint8_t r : best)
      s.push_back(r == 0 ? '|' : r == 1 ? 'e' : r - 2 < 10 ? (char)('0' + r - 2)
          : (char)('A' + r - 12));
    return s;
  }
};

// A row of the integrator's lists: Nickel index, S = E!/|Aut G|, the flags; the rows sort by the
//  Nickel key
struct Row { std::vector<uint8_t> key; std::string text; };
// One-vertex reducible: some vertex disconnects the others (self-loops ignored)
bool onevr(const Graph &g) {
  for (int cut = 0; cut < g.V; ++cut) {
    if (g.V < 2) return false;
    int start = cut == 0 ? 1 : 0; bool seen[MAXV] = {}; int stack[MAXV], sp = 0, cnt = 1;
    seen[start] = true; stack[sp++] = start;
    while (sp) {
      int u = stack[--sp];
      for (int v = 0; v < g.V; ++v) if (v != cut && g.m[u][v] && !seen[v]) {
        seen[v] = true;
        stack[sp++] = v;
        ++cnt;
      }
    }
    if (cnt < g.V - 1) return true;
  }
  return false;
}
Row row(const Graph &g) {
  Nickel nk(g); std::string s = nk.run();
  Canon c(g); c.run(); double aut = (double)c.naut; int E = 0;
  for (int a = 0; a < g.V; ++a) {
    for (int b = a + 1; b < g.V; ++b) for (int k = 2; k <= g.m[a][b]; ++k) aut *= k;
    for (int k = 1; k <= g.tad[a]; ++k) aut *= 2 * k;
    int l = g.legs(a); E += l; for (int k = 2; k <= l; ++k) aut *= k;
  }
  double ef = 1; for (int k = 2; k <= E; ++k) ef *= k;
  char buf[64]; std::snprintf(buf, sizeof buf, "%.12g", ef / aut);
  Row r; r.key = nk.best;
  r.text = s + "\t" + buf + "\t1VR=" + (onevr(g) ? "1" : "0") + "\tselfloop="
      + (g.selfloops() ? "1" : "0");
  return r;
}
void write_rows(std::vector<Row> &rows, std::ostream &os) {
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.key < b.key; });
  long i = 0; for (auto &r : rows) os << ++i << '\t' << r.text << '\n';
}
// The integrator lists of a phi^4 order: without self-loops (the lists in use) and with them
void write_nickel(const std::vector<Key> &level, std::ostream &owithout, std::ostream &owith,
    long &n1, long &n2) {
  std::vector<Row> with(level.size()); std::vector<char> keep(level.size(), 0);
#pragma omp parallel for schedule(dynamic, 256)
  for (size_t i = 0; i < level.size(); ++i) {
    Graph g = decode(level[i]);
    if (!g.opi()) continue;
    with[i] = row(g);
    keep[i] = 1;
  }
  std::vector<Row> all, nosl;
  for (size_t i = 0; i < level.size(); ++i) if (keep[i]) {
    if (with[i].text.back() == '0') nosl.push_back(with[i]);
    all.push_back(std::move(with[i]));
  }
  n1 = (long)all.size(); n2 = (long)nosl.size();
  write_rows(nosl, owithout); write_rows(all, owith);
}
// -N on a list: print it as an integrator list (every graph of the input, sorted by Nickel index)
int nickel_list() {
  std::vector<std::string> lines; long n = 0; bool bad = false; std::vector<Row> rows;
  while (read_block(lines, n)) {
    std::vector<Row> out(lines.size());
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < lines.size(); ++i) {
      Graph g;
      if (!parse(lines[i], g)) { bad = true; continue; }
      out[i] = row(g);
    }
    if (bad) {
      std::cerr << "topologies: cannot read a line among the first " << n << "\n";
      return 1;
    }
    for (auto &r : out) rows.push_back(std::move(r));
  }
  write_rows(rows, std::cout);
  return 0;
}

// -r: print the canonical key of every graph of the input, in hexadecimal, so that two lists are
//  compared as sets of graphs with sort and cmp
std::string hex(const Key &k) {
  char h[8 + 16 * WORDS]; int n = std::snprintf(h, sizeof h, "%02x", k.V);
  for (int i = 0; i < WORDS; ++i)
    n += std::snprintf(h + n, sizeof h - n, "%016llx", (unsigned long long)k.w[i]);
  return h;
}
int keys() {
  std::vector<std::string> lines; long n = 0; bool bad = false;
  while (read_block(lines, n)) {
    std::vector<std::string> out(lines.size());
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < lines.size(); ++i) {
      Graph g; if (!parse(lines[i], g)) { bad = true; continue; }
      out[i] = hex(Canon(g).run());
    }
    if (bad) {
      std::cerr << "topologies: cannot read a line among the first " << n << "\n";
      return 1;
    }
    for (auto &o : out) { std::fputs(o.c_str(), stdout); std::fputc('\n', stdout); }
  }
  return 0;
}

// -w: the zero-dimensional sum rule, sum over the graphs of the input of S_G = E!/|Aut G|, with
//  |Aut G| the vertex automorphisms times m! per line of multiplicity m, 2^t t! per vertex with t
//  self-loops and l! per vertex with l legs (tools/sumrules.py), as an exact fraction
std::string str128(__int128 x) {
  if (!x) return "0";
  std::string s;
  bool neg = x < 0;
  if (neg) x = -x;
  while (x) { s.push_back('0' + (int)(x % 10)); x /= 10; }
  if (neg) s.push_back('-');
  std::reverse(s.begin(), s.end());
  return s;
}
__int128 gcd128(__int128 a, __int128 b) {
  while (b) { __int128 t = a % b; a = b; b = t; }
  return a;
}
struct Frac {
  __int128 num = 0, den = 1;
  void add(__int128 n, __int128 d) {
    num = num * d + n * den;
    den *= d;
    __int128 g = gcd128(num < 0 ? -num : num, den);
    if (g > 1) { num /= g; den /= g; }
  }
};
std::string frac(const Frac &f) {
  return f.den == 1 ? str128(f.num) : str128(f.num) + "/" + str128(f.den);
}
// The exact zero-dimensional sum rules of phi^4: |coefficient of g^L| in 1/<x^2> (two legs, L = V)
//  and in -W_4/<x^2>^4 (four legs, L = V - 1) for the model x^2/2 + g x^4/24, which the sum of
//  E!/|Aut G| over the 1PI graphs with L loops and E legs, self-loops included, must equal.
//  Computed once with tools/zerodim_coeff.py (exact rationals).
const char *ZERODIM[2][15] = {
  {"", "1/2", "5/12", "5/6", "115/48", "625/72", "10775/288", "80275/432", "21615925/20736",
    "67531025/10368", "620085875/13824", "7001454875/20736", "1373072329375/497664",
    "18162979371125/746496", "2063837204439125/8957952"},
  {"1", "3/2", "21/4", "45/2", "1775/16", "4905/8", "119825/32", "3605575/144", "419516275/2304",
    "1649976125/1152", "18633428725/1536", "253871896625/2304", "59129851501625/55296",
    "2749622377511375/248832", "13404440892476875/110592"}};
// The sum rule of a set of graphs with the same E: the sum, whether every graph has V vertices
//  of valence 4 (then the phi^4 table applies), and the verdict against the table
struct SumRule { Frac total; int E = -1, V = -1; bool phi4 = true; long n = 0; };
void sumrule(const std::vector<Graph> &gs, SumRule &sr) {
#pragma omp parallel
  {
    Frac mine; int myV = -1; bool myphi4 = true;
#pragma omp for schedule(dynamic, 256)
    for (size_t i = 0; i < gs.size(); ++i) {
      const Graph &g = gs[i]; Canon c(g); c.run(); __int128 aut = c.naut; int legs = 0;
      for (int a = 0; a < g.V; ++a) {
        for (int b = a + 1; b < g.V; ++b) for (int k = 2; k <= g.m[a][b]; ++k) aut *= k;
        for (int k = 1; k <= g.tad[a]; ++k) aut *= 2 * k;
        int l = g.legs(a); legs += l; for (int k = 2; k <= l; ++k) aut *= k;
      }
      if (myV < 0) myV = g.V; else if (myV != g.V) myphi4 = false;
      if (!g.allfour()) myphi4 = false;
      __int128 ef = 1; for (int k = 2; k <= legs; ++k) ef *= k;
      mine.add(ef, aut);
    }
#pragma omp critical
    {
      if (!myphi4) sr.phi4 = false;
      if (myV >= 0) {
        if (sr.V < 0) sr.V = myV;
        else if (sr.V != myV) sr.phi4 = false;
      }
      sr.total.add(mine.num, mine.den);
    }
  }
  sr.n += (long)gs.size();
}
// The verdict of a sum rule against the table ("" when the table does not know the answer)
std::string verdict(const SumRule &sr) {
  int L = sr.E == 2 ? sr.V : sr.V - 1;
  if (!sr.phi4 || (sr.E != 2 && sr.E != 4) || L < 0 || L > 14) return "";
  std::string got = frac(sr.total), want = ZERODIM[sr.E == 4][L];
  return got == want ? "EXACT" : "WRONG, " + want + " expected";
}
// The sum rules of the graphs of a list, per number of legs, with the verdict when the list is a
//  complete 1PI phi^4 one
void report(std::vector<Graph> &gs) {
  std::map<int, std::vector<Graph>> byE;
  for (Graph &g : gs) byE[g.E()].push_back(g);
  for (auto &x : byE) {
    SumRule sr; sr.E = x.first; sumrule(x.second, sr);
    std::string v = verdict(sr);
    std::printf("%ld graphs with %d legs: sum of E!/|Aut| = %s%s\n", sr.n, sr.E,
      frac(sr.total).c_str(),
      v == "EXACT" ? ", the zero-dimensional coefficient of the complete list"
          : v.empty() ? "" : (" (a complete list, self-loops included, sums to "
              + v.substr(7, v.size() - 7 - 9) + ")").c_str());
  }
}
int weights() {
  std::vector<std::string> lines; long n = 0; std::vector<Graph> gs;
  while (read_block(lines, n)) {
    std::vector<Graph> block(lines.size()); bool bad = false;
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < lines.size(); ++i) if (!parse(lines[i], block[i])) bad = true;
    if (bad) {
      std::cerr << "topologies: cannot read a line among the first " << n << "\n";
      return 1;
    }
    gs.insert(gs.end(), block.begin(), block.end());
  }
  report(gs);
  return 0;
}

double cpu() {
  struct rusage r;
  getrusage(RUSAGE_SELF, &r);
  return r.ru_utime.tv_sec + 1e-6 * r.ru_utime.tv_usec + r.ru_stime.tv_sec
      + 1e-6 * r.ru_stime.tv_usec;
}

// The growth.
// The valences are a multiset with distinct values vals[j] of multiplicities n[j]; a sub-multiset
//  is a vector of counts c[j] <= n[j], numbered in mixed radix, and the classes of a level (a size
//  of the sub-multiset) are grown from those of the level before: the class of c receives, for
//  every j with c[j] > 0, the children of the class of c - e_j by a vertex of valence vals[j].
struct Growth {
  std::vector<int> vals, n, stride; int V = 0, total = 1;
  bool fixedE = false; int E = -1; // the number of legs when it is fixed, or selected with -l
  std::vector<std::vector<Ins>> ins; // per distinct valence
  std::vector<std::vector<Key>> cls; // per sub-multiset, the sorted keys of its class
  explicit Growth(const std::vector<int> &degrees) {
    std::vector<int> d = degrees; std::sort(d.begin(), d.end()); V = (int)d.size();
    for (int x : d) {
      if (vals.empty() || vals.back() != x) { vals.push_back(x); n.push_back(0); }
      ++n.back();
    }
    for (size_t j = 0; j < vals.size(); ++j) { stride.push_back(total); total *= n[j] + 1; }
    cls.resize(total);
  }
  bool alleven() const { for (int x : vals) if (x & 1) return false; return true; }
  int sum() const {
    int s = 0; for (size_t j = 0; j < vals.size(); ++j) s += vals[j] * n[j]; return s;
  }
  // The mode: E fixed (every valence even, E in {0, 2, 4}) or every E, with the window of -l
  void mode(int legs) {
    E = legs;
    fixedE = legs >= 0 && legs <= 4 && alleven();
    ins.clear();
    for (int x : vals) ins.push_back(insertions(x, fixedE));
  }
  std::vector<int> counts(int id) const {
    std::vector<int> c(vals.size());
    for (size_t j = 0; j < vals.size(); ++j) c[j] = id / stride[j] % (n[j] + 1);
    return c;
  }
  std::string name(const std::vector<int> &c) const {
    std::string s;
    for (size_t j = 0; j < vals.size(); ++j) for (int k = 0; k < c[j]; ++k)
      s += (s.empty() ? "" : ",") + std::to_string(vals[j]);
    return s;
  }
  // The window of the legs of the ancestors of the target graphs at the sub-multiset c: removing a
  //  leaf of valence d changes E by at most +1 and at least 2 - d
  void window(const std::vector<int> &c, int &lo, int &hi) const {
    if (E < 0) { lo = 0; hi = 2 * sum(); return; }
    lo = hi = E;
    for (size_t j = 0; j < vals.size(); ++j) {
      hi += n[j] - c[j];
      lo -= (n[j] - c[j]) * (vals[j] - 2);
    }
    if (fixedE) lo = hi = E;
  }
  // Grows the class of id from its parents (level >= 2) or from the one-vertex graphs
  size_t grow(int id) {
    std::vector<int> c = counts(id); int lo, hi; window(c, lo, hi); int size = 0;
    for (int x : c) size += x;
    Level *next = new Level;
    for (size_t j = 0; j < vals.size(); ++j) if (c[j] > 0) {
      if (size == 1) { // the seeds: one vertex of valence vals[j] with t self-loops
        for (int t = 0; 2 * t <= vals[j]; ++t) {
          int legs = vals[j] - 2 * t;
          if (legs < lo || legs > hi) continue;
          Graph g; g.V = 1; g.val[0] = (uint8_t)vals[j]; g.tad[0] = (uint8_t)t;
          next->insert(Canon(g).run());
        }
        continue;
      }
      const std::vector<Key> &parents = cls[id - stride[j]];
#pragma omp parallel for schedule(dynamic, 64)
      for (size_t i = 0; i < parents.size(); ++i)
        Inserter(decode(parents[i]), vals[j], *next, lo, hi).run(ins[j]);
    }
    cls[id] = next->sorted();
    delete next;
    return cls[id].size();
  }
  // Grows every level up to the whole multiset; after a level is complete, hook(level) runs on
  //  its classes and the level before is freed
  template <class F> void run(F hook, FILE *log) {
    double c0 = cpu();
    for (int size = 1; size <= V; ++size) {
      auto t0 = std::chrono::steady_clock::now();
      std::vector<int> ids;
      for (int id = 0; id < total; ++id) {
        std::vector<int> c = counts(id); int s = 0; for (int x : c) s += x;
        if (s == size) ids.push_back(id);
      }
      for (int id : ids) grow(id);
      double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      for (int id : ids) {
        if (ids.size() > 1)
          std::fprintf(log, "V=%2d {%s}: %9zu connected\n", size, name(counts(id)).c_str(),
            cls[id].size());
        hook(size, id, tg, c0);
      }
      for (int id = 0; id < total; ++id) {
        std::vector<int> c = counts(id); int s = 0; for (int x : c) s += x;
        if (s == size - 1) { cls[id].clear(); cls[id].shrink_to_fit(); }
      }
      std::fflush(log);
    }
  }
};

// The graphs of a class that pass the filters: the legs with -l, 1PI with -i, no self-loops
//  without -t
std::vector<Key> select(const std::vector<Key> &level, int E, bool opi, bool tads) {
  std::vector<char> keep(level.size(), 0);
#pragma omp parallel for schedule(dynamic, 256)
  for (size_t i = 0; i < level.size(); ++i) {
    Graph g = decode(level[i]);
    keep[i] = (E < 0 || g.E() == E) && (tads || !g.selfloops()) && (!opi || g.opi());
  }
  std::vector<Key> out;
  for (size_t i = 0; i < level.size(); ++i) if (keep[i]) out.push_back(level[i]);
  return out;
}
// The compact strings of a list, made in parallel block by block
void write_list(const std::vector<Key> &list, std::ostream &os) {
  const size_t B = 4096; size_t nb = (list.size() + B - 1) / B; std::vector<std::string> s(nb);
#pragma omp parallel for schedule(dynamic)
  for (size_t b = 0; b < nb; ++b)
    for (size_t i = b * B; i < std::min(list.size(), (b + 1) * B); ++i) s[b] += compact(list[i]);
  for (auto &x : s) os << x;
}
// The integrator list of a list
void write_nickel_list(const std::vector<Key> &list, std::ostream &os) {
  std::vector<Row> rows(list.size());
#pragma omp parallel for schedule(dynamic, 256)
  for (size_t i = 0; i < list.size(); ++i) rows[i] = row(decode(list[i]));
  write_rows(rows, os);
}

void usage() {
  std::cerr << "usage: topologies [-l L] [-i] [-t] [-c] [-o FILE] [-j THREADS] [-N] [-w] DEGREES\n"
      "       topologies -E 2|4 -V VMAX [-o DIR] [-j THREADS] [-N] [-w] [-a] [-G]\n"
      "       topologies -r [-x] < LIST    (canonical keys)\n"
      "       topologies -w [-x] < LIST    (zero-dimensional sum rules, per number of legs)\n"
      "       topologies -N [-x] < LIST    (integrator list: Nickel indices, weights, flags)\n"
      "DEGREES: the valences of the vertices, comma-separated, 1 to 8, at most 12 entries;\n"
      "-l L keeps the graphs with L loops, -i the one-particle-irreducible ones, -t includes the\n"
      "graphs with self-loops, -c is accepted (only connected graphs are grown). In the phi^4\n"
      "mode every order up to VMAX vertices is written to DIR as <L>-loop-2 and <L>-loop-2-ci,\n"
      "or <L>-loop-4 and <L>-loop-4-cit, with -N the integrator's gamma<E>_L<L>.txt and\n"
      "_with_selfloops.txt, with -a every connected graph in <L>-loop-<E>-all, with -G in the\n"
      "labelling and the order of the 2003 generator's lists.\n";
}

// An output file of the phi^4 mode; the run stops when it cannot be created (a missing directory)
std::ofstream output(const std::string &path) {
  std::ofstream f(path);
  if (!f) { std::fprintf(stderr, "topologies: cannot write %s\n", path.c_str()); std::exit(1); }
  return f;
}

// The phi^4 mode: the files of every order
int phi4(int E, int Vmax, const std::string &dir, bool nickel, bool sums, bool all,
    bool canonical) {
  std::vector<int> degrees(Vmax, 4); Growth G(degrees); G.mode(E);
  G.run([&](int V, int id, double tg, double &c0) {
    const std::vector<Key> &level = G.cls[id]; int L = V + 1 - E / 2;
    std::string base = dir + "/" + std::to_string(L) + "-loop-" + std::to_string(E);
    std::ofstream owith = output(E == 2 ? base : base + "-cit"),
      owithout = output(E == 2 ? base + "-ci" : base);
    long n1, n2;
    write(level, owith, owithout, canonical, n1, n2);
    if (nickel) { // the integrator's lists too, in the names of integrator/data/lists
      std::string nb = dir + "/gamma" + std::to_string(E) + "_L" + std::to_string(L);
      std::ofstream nwithout = output(nb + ".txt"), nwith = output(nb + "_with_selfloops.txt");
      write_nickel(level, nwithout, nwith, n1, n2);
    }
    double c = cpu();
    std::printf("V=%2d L=%2d: %9zu connected, %9ld 1PI, %9ld 1PI without self-loops  (%.1f s "
      "growing; %.1f s CPU)\n", V, L, level.size(), n1, n2, tg, c - c0);
    c0 = c;
    // Every connected graph of the order, 1PI or not (for checks of sums over the connected class)
    if (all) {
      std::ofstream oall = output(base + "-all"); for (auto &k : level) oall << compact(k);
    }
    if (sums) { // the zero-dimensional sum rule of the complete 1PI list, self-loops included
      std::vector<Graph> gs; gs.reserve(n1);
      for (auto &k : level) { Graph g = decode(k); if (g.opi()) gs.push_back(g); }
      SumRule sr; sr.E = E; sumrule(gs, sr);
      std::printf("          sum rule of the %ld 1PI graphs: %s %s\n", n1, frac(sr.total).c_str(),
        verdict(sr).c_str());
    }
  }, stdout);
  return 0;
}

// The general mode: the class of the whole multiset, filtered, to a file or to standard output
int general(const std::vector<int> &degrees, int L, bool opi, bool tads, const std::string &out,
    bool nickel, bool sums) {
  Growth G(degrees); int V = G.V, sum = G.sum(), E = -1;
  if (L >= 0) {
    E = sum - 2 * (L + V - 1);
    if (E < 0) {
      std::fprintf(stderr, "topologies: %d loops are impossible with the valences %s (%d to %d)\n",
        L, G.name(G.n).c_str(), 0, sum / 2 - V + 1);
      return 2;
    }
  }
  G.mode(E);
  std::fprintf(stderr, "topologies: valences {%s}, %s%s\n", G.name(G.n).c_str(),
    E < 0 ? "every number of legs" : (std::to_string(E) + " legs").c_str(),
    G.fixedE ? " (grown at fixed E)" : "");
  std::vector<Key> list;
  G.run([&](int size, int id, double tg, double &c0) {
    if (size < V) return;
    list = select(G.cls[id], E, opi, tads);
    double c = cpu();
    std::fprintf(stderr, "V=%2d: %9zu connected, %9zu written  (%.1f s growing; %.1f s CPU)\n",
      size, G.cls[id].size(), list.size(), tg, c - c0);
    c0 = c;
  }, stderr);
  std::ofstream file; std::ostream *os = &std::cout;
  if (!out.empty()) {
    file.open(out);
    if (!file) { std::fprintf(stderr, "topologies: cannot write %s\n", out.c_str()); return 1; }
    os = &file;
  }
  if (nickel) write_nickel_list(list, *os); else write_list(list, *os);
  if (sums) {
    std::vector<Graph> gs; gs.reserve(list.size());
    for (auto &k : list) gs.push_back(decode(k));
    std::fflush(stdout);
    report(gs);
  }
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  int E = 0, Vmax = 0, threads = 0, loops = -1;
  bool canonical = false, nickel = false, sums = false, all = false, opi = false, tads = false,
      readkeys = false;
  std::string out;
  for (int c; (c = getopt(argc, argv, "E:V:l:o:j:acitGNrwxh")) != -1;) {
    if (c == 'E') E = atoi(optarg);
    else if (c == 'V') Vmax = atoi(optarg);
    else if (c == 'l') loops = atoi(optarg);
    else if (c == 'o') out = optarg;
    else if (c == 'j') threads = atoi(optarg);
    else if (c == 'a') all = true;
    else if (c == 'c') {} // only connected graphs are grown
    else if (c == 'i') opi = true;
    else if (c == 't') tads = true;
    else if (c == 'G') canonical = true;
    else if (c == 'N') nickel = true;
    else if (c == 'r') readkeys = true;
    else if (c == 'w') sums = true;
    else if (c == 'x') zerolegs = true;
    else { usage(); return c == 'h' ? 0 : 2; }
  }
#ifdef _OPENMP
  if (threads > 0) omp_set_num_threads(threads);
#endif
  bool grow = E != 0 || Vmax != 0 || optind < argc;
  // The read modes: keys, sum rules or an integrator list of the list on standard input
  if (readkeys) return keys();
  if (!grow) {
    if (nickel) return nickel_list();
    if (sums) return weights();
    usage();
    return 2;
  }
  if (E != 0 || Vmax != 0) { // the phi^4 mode
    if ((E != 2 && E != 4) || Vmax < 1 || Vmax > MAXV - 1 || optind < argc) { usage(); return 2; }
    createPartitionTable();
    return phi4(E, Vmax, out.empty() ? "." : out, nickel, sums, all, canonical);
  }
  if (optind != argc - 1 || canonical || all) { usage(); return 2; }
  std::vector<int> degrees; std::string s = argv[optind];
  for (size_t i = 0, j; i < s.size(); i = j + 1) {
    j = s.find(',', i); if (j == std::string::npos) j = s.size();
    int d = atoi(s.substr(i, j - i).c_str());
    if (d < 1 || d > DMAX) {
      std::fprintf(stderr, "topologies: a valence must lie between 1 and %d\n", DMAX);
      return 2;
    }
    degrees.push_back(d);
  }
  if (degrees.empty() || (int)degrees.size() > MAXV - 1) {
    std::fprintf(stderr, "topologies: 1 to %d vertices\n", MAXV - 1);
    return 2;
  }
  return general(degrees, loops, opi, tads, out, nickel, sums);
}
