/*
    graph.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Host-only graph utilities: Nickel indices, subgraph predicates on edge
    bitmasks, symmetry and O(N) factors, a brute-force generator of 1PI
    phi^4 graphs, cycle-matroid isomorphism (Whitney twins).

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
#include <cstdint>
#include <string>
#include <vector>

struct Graph {
  int V = 0;
  std::vector<int> eu, ev; // lines (eu[i] <= ev[i] is NOT required; self-loop: eu == ev)
  std::vector<int> ext; // external legs per vertex
  std::string name; // Nickel index as read / generated
  int l() const { return (int)eu.size(); }
  int E() const { int s = 0; for (int x : ext) s += x; return s; }
  int loops() const { return l() - V + 1; } // connected graphs only
  uint32_t full() const { return l() >= 32 ? 0xffffffffu : ((1u << l()) - 1u); }
};

bool parse_nickel(const std::string &s, Graph &G, std::string *err);
std::string nickel_string(const Graph &G); // in the graph's own labelling
std::string nickel_canonical(const Graph &G); // min over all relabellings
Graph relabel(const Graph &G, const std::vector<int> &perm); // vertex v -> perm[v]

// Subgraph predicates on line bitmasks.
uint32_t vmask(const Graph &G, uint32_t S);
int ncomponents(const Graph &G, uint32_t S, uint32_t verts); // components of (verts, S)
bool connected(const Graph &G, uint32_t S); // (V(S), S) connected
bool bridgeless(const Graph &G, uint32_t S); // no bridge in (V(S), S)
int rank_of(const Graph &G, uint32_t S);
int nullity(const Graph &G, uint32_t S);
int legs_of(const Graph &G, uint32_t S); // 4|V(S)| - 2|S|
bool is_1pi(const Graph &G);
bool is_1vr(const Graph &G);
bool has_selfloop(const Graph &G);
bool degree4(const Graph &G, std::string *err);

// Symmetry and tensor factors.
// Half-edge level |Aut| (vertex perms x multi-edge x self-loop x ext legs)
double aut_order(const Graph &G);
// E! / |Aut|  (N = 1 weight, Phi4tools convention: W = S/3 for Gamma4)
double sym_factor(const Graph &G);
// O(N) factor as coefficients of N^k (equals 1 at N = 1)
std::vector<double> on_factor(const Graph &G);
double eval_poly(const std::vector<double> &c, double N);

// Generator: all 1PI graphs with V vertices and E external legs (no self-loops unless asked).
std::vector<Graph> generate_graphs(int V, int E, bool selfloops);

// Cycle matroid.
std::vector<uint32_t> circuits(const Graph &G);
bool matroid_iso(const Graph &A, const Graph &B, std::vector<int> *edge_map); // edge_map[eA] = eB
