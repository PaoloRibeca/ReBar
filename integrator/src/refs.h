/*
    refs.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Reference values for validation (N = 0, 1, 2, 3).
      B, H, C: CS-route per-order sums from the Phi4tools O(N) series
        (arXiv:2311.12576 app. D) via tools/refs.py; uncertainties
        propagated from the quoted Phi4tools errors.
      INT4: Phi4tools Gamma4 order sums S_L = sum_G V_G (intermediate
        scheme, N = 1), from the per-graph Phi4tools values at 5 and 6 loops
        (S_5 uncertainty = quadrature of the per-graph errors).

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
#include <cstring>

struct RefVal { const char *kind; int N, L; double v, e; };

static const RefVal kRefs[] = {
  // N = 1
  {"b", 1, 1, 0.75, 0}, {"b", 1, 2, 0.5, 0}, {"b", 1, 3, 0.5986037059, 1e-10},
    {"b", 1, 4, 0.9671926870, 1e-9},
  {"b", 1, 5, 1.896525357, 4.9e-7}, {"b", 1, 6, 4.300683598, 4.9e-6},
    {"b", 1, 7, 10.9175936, 4.5e-5},
  {"h", 1, 2, 0.012345679012, 0}, {"h", 1, 3, -0.001542750058, 1e-12},
    {"h", 1, 4, 0.004699104918, 1e-12},
  {"h", 1, 5, 0.002443913312, 1.0e-8}, {"h", 1, 6, 0.008020563982, 8.9e-8},
    {"h", 1, 7, 0.01454955429, 8.0e-7},
  {"c", 1, 1, 0.25, 0}, {"c", 1, 2, 0.083333333333, 0}, {"c", 1, 3, 0.07785997192, 1e-11},
    {"c", 1, 4, 0.1006770275, 1e-10},
  {"c", 1, 5, 0.1705519189, 8.2e-8}, {"c", 1, 6, 0.3459107299, 8.6e-7},
  // N = 0
  {"b", 0, 1, 0.6666666667, 1e-10}, {"b", 0, 2, 0.4074074074, 1e-10},
    {"b", 0, 3, 0.4657891194, 1e-10}, {"b", 0, 4, 0.7146383217, 1e-10},
  {"b", 0, 5, 1.343451256, 3.3e-7},
  {"h", 0, 2, 0.008230452675, 1e-12}, {"h", 0, 3, -0.0009142222565, 1e-12},
    {"h", 0, 4, 0.002580128416, 1e-12}, {"h", 0, 5, 0.001376862082, 5.8e-9},
  {"c", 0, 1, 0.1666666667, 1e-10}, {"c", 0, 2, 0.05555555556, 1e-11},
    {"c", 0, 3, 0.04376258412, 1e-11}, {"c", 0, 4, 0.05462562208, 1e-11},
  // N = 2
  {"b", 2, 1, 0.8333333333, 1e-10}, {"b", 2, 2, 0.5925925926, 1e-10},
    {"b", 2, 3, 0.7381207135, 1e-10}, {"b", 2, 4, 1.248598841, 1e-9},
  {"b", 2, 5, 2.539047034, 6.8e-7},
  {"h", 2, 2, 0.01646090535, 1e-11}, {"h", 2, 3, -0.002285555641, 1e-12},
    {"h", 2, 4, 0.007357422444, 1e-12}, {"h", 2, 5, 0.003694852476, 1.6e-8},
  {"c", 2, 1, 0.3333333333, 1e-10}, {"c", 2, 2, 0.1111111111, 1e-10},
    {"c", 2, 3, 0.1201014235, 1e-10}, {"c", 2, 4, 0.1584286558, 1e-10},
  // N = 3
  {"b", 3, 1, 0.9166666667, 1e-10}, {"b", 3, 2, 0.6851851852, 1e-10},
    {"b", 3, 3, 0.8843401418, 1e-10}, {"b", 3, 4, 1.558448023, 1e-9},
  {"b", 3, 5, 3.272480446, 9.1e-7},
  {"h", 3, 2, 0.02057613169, 1e-11}, {"h", 3, 3, -0.003142639007, 1e-12},
    {"h", 3, 4, 0.01054513062, 1e-11}, {"h", 3, 5, 0.005090922124, 2.3e-8},
  {"c", 3, 1, 0.4166666667, 1e-10}, {"c", 3, 2, 0.1388888889, 1e-10},
    {"c", 3, 3, 0.170486939, 1e-9}, {"c", 3, 4, 0.2272863767, 1e-10},
  // Phi4tools Gamma4 order sums (N = 1, intermediate scheme), S_L = (2^L/3) sum_G S_G I_G
  {"int4", 1, 1, 1.0, 0}, {"int4", 1, 4, 75.2084920642, 1e-9},
    {"int4", 1, 5, 400.174143593, 1.8e-6},
};

static inline bool find_ref(const char *kind, double N, int L, double *v, double *e) {
  for (const auto &r : kRefs)
    if (!strcmp(r.kind, kind) && (double)r.N == N && r.L == L) { *v = r.v; *e = r.e; return true; }
  return false;
}
