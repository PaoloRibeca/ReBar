/*
    listcheck.cpp -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Structural checks and exact weight data for a graph list, through the
    integrator's own graph code (src/graph.cpp: the parser and degree-4
    check, is_1pi, is_1vr, has_selfloop, aut_order, sym_factor, on_factor,
    nickel_canonical), plus an independent exact branch-and-bound Nickel
    canonicaliser.

    build: g++ -O2 -std=c++17 -fopenmp -Isrc -o listcheck
      tools/lists/listcheck.cpp src/graph.cpp
    usage: listcheck LIST [--brute all|none|K] [--proto-on all|none|K]
      --brute: brute-force canonical check (the integrator's
        nickel_canonical with its V! relabellings) of every K-th graph
      --proto-on: O(N) factor from the integrator's own on_factor for every
        K-th graph (default all), compared bit for bit with listcheck's
        fast exact count (below); 2.5 s and 20 ms per graph at 9 vertices

    Reads LIST as rebar does (src/main.cpp load_list: first tab-separated
    field containing '|'; '#' lines ignored).
    stdout, one row per graph:
      idx nickel V E L legs 1PI 1VR selfloop canon_bb brute aut S on_counts
        on_double proto_on
      legs: external legs per vertex, sorted descending (e.g. 2,1,1)
      canon_bb: 1 if the exact branch-and-bound canonical Nickel index
        equals the string as listed
      brute: 1/0: same test with the integrator's brute-force
        nickel_canonical (all V! relabellings); '-' if not run
      aut: |Aut G| at the half-edge level (src/graph.cpp aut_order), an
        integer
      S: E!/|Aut G| as the integrator computes it (sym_factor, double,
        %.17g)
      on_counts: number of the 3^V vertex pairings with k closed index
        loops, k = 0, 1, ... (exact integers, counted here by the same
        union-find walk as on_factor, without its floating point)
      on_double: the integrator's on_factor(G) (double, %.17g),
        coefficients of N^0, N^1, ...: on_factor adds w = pow(3, -V) once
        per pairing to the coefficient of its loop count, so its result is
        fixed by on_counts; it is rebuilt from them by the same additions
      proto_on: 1/0: the integrator's on_factor(G) equals on_double bit for
        bit; '-' if not run (--proto-on)
    stderr: a summary (counts, distinct canonical forms, failures). Exit
      status 1 if any structural check fails.

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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static char vch(int v) { return v < 10 ? (char)('0' + v) : (char)('A' + v - 10); }

// Exact canonical Nickel index (minimum over relabellings, '|' < 'e' < '0' < '1' < ...), branch and
// bound.
// Vertices get labels 0, 1, 2, ... in turn.  The key of a placed prefix is exact: a vertex's list
// holds its legs, then the sorted labels >= its own of its neighbours; neighbours not yet placed
// get labels larger than every placed one, so the sorted placed labels are a prefix of the final
// list.  The next label always goes to a neighbour of the first placed vertex that still has
// unplaced neighbours: any other choice puts a larger label at that position of that vertex's list,
// which is strictly worse.  Without such a vertex every unplaced vertex is a candidate.
struct BB {
  int V;
  std::vector<std::vector<int>> m; // line multiplicities, m[v][v] = self-loops at v
  std::vector<int> ext;
  std::vector<int> best;
  bool have = false;
  std::vector<int> order, pos;
  void key(std::vector<int> &k, bool &complete) {
    k.clear();
    complete = (int)order.size() == V;
    for (int q = 0; q < (int)order.size(); ++q) {
      int v = order[q];
      for (int i = 0; i < ext[v]; ++i) k.push_back(1);
      std::vector<int> labs;
      bool unplaced = false;
      for (int w = 0; w < V; ++w) {
        if (!m[v][w]) continue;
        if (pos[w] < 0) { unplaced = true; continue; }
        if (pos[w] >= q) for (int c = 0; c < m[v][w]; ++c) labs.push_back(pos[w]);
      }
      std::sort(labs.begin(), labs.end());
      for (int x : labs) k.push_back(2 + x);
      if (unplaced) { complete = false; return; }
      k.push_back(0);
    }
  }
  void rec() {
    std::vector<int> k;
    bool complete;
    key(k, complete);
    if (have) {
      // Compare k with the prefix of best of the same length
      size_t n = std::min(k.size(), best.size());
      for (size_t i = 0; i < n; ++i) {
        if (k[i] < best[i]) break;
        if (k[i] > best[i]) return;
      }
      if (complete) { if (k < best) best = k; return; }
    } else if (complete) { best = k; have = true; return; }
    std::vector<int> cand;
    for (int v : order) {
      for (int w = 0; w < V; ++w) if (m[v][w] && pos[w] < 0) cand.push_back(w);
      if (!cand.empty()) break;
    }
    if (cand.empty()) for (int w = 0; w < V; ++w) if (pos[w] < 0) cand.push_back(w);
    for (int w : cand) {
      pos[w] = (int)order.size(); order.push_back(w);
      rec();
      order.pop_back(); pos[w] = -1;
    }
  }
  std::string run(const Graph &G) {
    V = G.V;
    m.assign(V, std::vector<int>(V, 0));
    for (int i = 0; i < G.l(); ++i) {
      int a = G.eu[i], b = G.ev[i];
      if (a == b) m[a][a]++; else { m[a][b]++; m[b][a]++; }
    }
    ext = G.ext;
    pos.assign(V, -1); order.clear(); have = false;
    rec();
    std::string s;
    for (int r : best) s += r == 0 ? '|' : r == 1 ? 'e' : vch(r - 2);
    return s;
  }
};

// Pairing counts: vertex v pairs its 4 half-edges in one of 3 ways; with the lines' own pairings
// they close index loops; strands through an external leg are open.  Same enumeration as on_factor
// (src/graph.cpp), integer counts.
static std::vector<long> on_counts(const Graph &G) {
  const int V = G.V, l = G.l(), E = G.E(), H = 2 * l + E;
  std::vector<std::vector<int>> slots(V);
  for (int i = 0; i < l; ++i) {
    slots[G.eu[i]].push_back(2 * i);
    slots[G.ev[i]].push_back(2 * i + 1);
  }
  int k = 0;
  for (int v = 0; v < V; ++v) for (int j = 0; j < G.ext[v]; ++j) slots[v].push_back(2 * l + k++);
  static const int pr[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
  std::vector<long> cnt(l + 2, 0);
  long total = 1;
  for (int v = 0; v < V; ++v) total *= 3;
  int par[96];
  char open[96];
  auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
  auto un = [&](int a, int b) { a = find(a); b = find(b); if (a != b) par[a] = b; };
  for (long t = 0; t < total; ++t) {
    for (int h = 0; h < H; ++h) { par[h] = h; open[h] = 0; }
    for (int i = 0; i < l; ++i) un(2 * i, 2 * i + 1);
    long x = t;
    for (int v = 0; v < V; ++v) {
      const int c = (int)(x % 3); x /= 3;
      const auto &s = slots[v];
      un(s[pr[c][0]], s[pr[c][1]]); un(s[pr[c][2]], s[pr[c][3]]);
    }
    for (int h = 2 * l; h < H; ++h) open[find(h)] = 1;
    int closed = 0;
    for (int h = 0; h < H; ++h) if (find(h) == h && !open[h]) ++closed;
    cnt[closed]++;
  }
  while (cnt.size() > 1 && cnt.back() == 0) cnt.pop_back();
  return cnt;
}
// on_factor's doubles from the counts: c[k] = w + w + ... (cnt[k] times), w = pow(3, -V), trailing
// zeros dropped
static std::vector<double> on_double(const Graph &G, const std::vector<long> &cnt) {
  const double w = std::pow(3.0, -G.V);
  std::vector<double> c(cnt.size(), 0.0);
  for (size_t k = 0; k < cnt.size(); ++k) for (long i = 0; i < cnt[k]; ++i) c[k] += w;
  while (c.size() > 1 && c.back() == 0) c.pop_back();
  return c;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: listcheck LIST [--brute all|none|K]\n"); return 2; }
  std::string brute = "none", proto_on = "all";
  for (int i = 2; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--brute") brute = argv[i + 1];
    if (std::string(argv[i]) == "--proto-on") proto_on = argv[i + 1];
  }
  std::ifstream f(argv[1]);
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  std::vector<std::string> names;
  std::string line;
  while (std::getline(f, line)) { // as src/main.cpp load_list
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::string tok, nk;
    while (std::getline(ss, tok, '\t')) {
      std::string t;
      for (char c : tok) if (!isspace((unsigned char)c)) t += c;
      if (t.find('|') != std::string::npos) { nk = t; break; }
    }
    if (!nk.empty()) names.push_back(nk);
  }
  const long n = (long)names.size();
  std::vector<std::string> rows(n);
  std::vector<int> bad(n, 0), canon(n, 0), brute_ok(n, -1), pon_ok(n, -1), pi(n, 0), vr(n, 0),
      sl(n, 0), Ls(n, 0), Es(n, 0);
  std::vector<std::string> legs(n), cform(n);
  long kbrute = brute == "all" ? 1 : brute == "none" ? 0 : atol(brute.c_str());
  long kpon = proto_on == "all" ? 1 : proto_on == "none" ? 0 : atol(proto_on.c_str());
#pragma omp parallel for schedule(dynamic, 4)
  for (long gi = 0; gi < n; ++gi) {
    Graph G;
    std::string err;
    if (!parse_nickel(names[gi], G, &err)) {
      rows[gi] = names[gi] + "\tPARSE-ERROR " + err;
      bad[gi] = 1;
      continue;
    }
    std::vector<int> lg;
    for (int x : G.ext) if (x) lg.push_back(x);
    std::sort(lg.rbegin(), lg.rend());
    std::string ls;
    for (size_t i = 0; i < lg.size(); ++i) ls += (i ? "," : "") + std::to_string(lg[i]);
    legs[gi] = ls;
    pi[gi] = is_1pi(G); vr[gi] = is_1vr(G); sl[gi] = has_selfloop(G); Ls[gi] = G.loops();
    Es[gi] = G.E();
    BB bb;
    cform[gi] = bb.run(G);
    canon[gi] = cform[gi] == names[gi];
    if (kbrute > 0 && gi % kbrute == 0) brute_ok[gi] = nickel_canonical(G) == names[gi];
    double aut = aut_order(G), S = sym_factor(G);
    std::vector<long> nc = on_counts(G);
    std::vector<double> on = on_double(G, nc);
    if (kpon > 0 && gi % kpon == 0) pon_ok[gi] = on_factor(G) == on; // bit for bit
    std::string cnt, ond;
    for (size_t k = 0; k < on.size(); ++k) {
      char b[64];
      snprintf(b, sizeof b, "%s%ld", k ? "," : "", nc[k]); cnt += b;
      snprintf(b, sizeof b, "%s%.17g", k ? "," : "", on[k]); ond += b;
    }
    if (nc.size() != on.size() || pon_ok[gi] == 0) bad[gi] = 1;
    char b[512];
    snprintf(b, sizeof b, "%d\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%s\t%.0f\t%.17g\t", G.V, G.E(),
      G.loops(), ls.c_str(), pi[gi], vr[gi], sl[gi], canon[gi],
      brute_ok[gi] < 0 ? "-" : brute_ok[gi] ? "1" : "0", aut, S);
    rows[gi] = names[gi] + "\t" + b + cnt + "\t" + ond + "\t"
        + (pon_ok[gi] < 0 ? "-" : pon_ok[gi] ? "1" : "0");
  }
  long nbad = 0, npi = 0, nvr = 0, nsl = 0, ncanon = 0, nbr = 0, nbr_ok = 0, npon = 0, npon_ok = 0;
  std::map<std::string, long> lcount, Lcount, Ecount;
  std::set<std::string> distinct_names, distinct_canon;
  for (long gi = 0; gi < n; ++gi) {
    printf("%ld\t%s\n", gi + 1, rows[gi].c_str());
    nbad += bad[gi]; npi += pi[gi]; nvr += vr[gi]; nsl += sl[gi]; ncanon += canon[gi];
    if (brute_ok[gi] >= 0) { ++nbr; nbr_ok += brute_ok[gi]; }
    if (pon_ok[gi] >= 0) { ++npon; npon_ok += pon_ok[gi]; }
    lcount[legs[gi]]++; Lcount[std::to_string(Ls[gi])]++; Ecount[std::to_string(Es[gi])]++;
    distinct_names.insert(names[gi]); distinct_canon.insert(cform[gi]);
  }
  fprintf(stderr, "%s: %ld graphs; distinct strings %zu, distinct canonical forms %zu; canonical "
    "as listed %ld/%ld; brute-force checked %ld, equal %ld\n", argv[1], n, distinct_names.size(),
    distinct_canon.size(), ncanon, n, nbr, nbr_ok);
  fprintf(stderr,
    "  integrator on_factor run on %ld graphs, bit-identical to the exact counts for %ld\n", npon,
    npon_ok);
  fprintf(stderr, "  1PI %ld/%ld, 1VR %ld, with self-loops %ld, parse/count errors %ld; loops", npi,
    n, nvr, nsl, nbad);
  for (auto &kv : Lcount) fprintf(stderr, " L=%s:%ld", kv.first.c_str(), kv.second);
  fprintf(stderr, "; external legs");
  for (auto &kv : Ecount) fprintf(stderr, " E=%s:%ld", kv.first.c_str(), kv.second);
  fprintf(stderr, "; leg partitions");
  for (auto &kv : lcount) fprintf(stderr, " (%s):%ld", kv.first.c_str(), kv.second);
  fprintf(stderr, "\n");
  bool ok = nbad == 0 && npi == n && ncanon == n && nbr_ok == nbr && npon_ok == npon
      && (long)distinct_canon.size() == n;
  fprintf(stderr, "  -> %s\n", ok ? "OK" : "FAILED");
  return ok ? 0 : 1;
}
