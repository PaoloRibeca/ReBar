/*
    build.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Host-side construction of one class (graph + all dot insertions) for
    the kernel: renormalisation parts, Zimmermann forests (Lagrangian rule:
    nested or vertex-disjoint), parametric forest terms with t^2 jets
    (effective massless lines by deletion-contraction), full symbolic
    expansion into products of block Kirchhoff polynomials ("atoms"), exact
    algebraic pruning, forest-aware exponent table.

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
#include <map>
#include <string>
#include <vector>
#include "graph.h"
#include "rbar/kernel.h"

enum class Kind {
  B, // btilde: Gamma4, one dot summed over lines, R-bar (CS route)
  H, // h: Gamma2, one dot, p^2-derivative, R-bar
  C, // c: Gamma2, two dots (ordered pairs), R-bar
  BARE4, // bare Gamma4 (no dots, no subtraction): I_G = Gamma(L/2) pi^(-L/2) Int U^-3/2
  // Phi4tools/intermediate scheme Gamma4: sunsets subtracted at zero momentum (t^0), tadpoles zero
  INT4,
  BARE2D, // bare Gamma2 p^2-derivative (no dots, no subtraction)
  INT2D, // Gamma2 p^2-derivative with sunsets t^0-subtracted (intermediate scheme)
  // Gamma2 at zero momentum, undotted, sunsets t^0-subtracted (intermediate scheme; route B, from 3
  // loops)
  INT2
};
const char *kind_name(Kind k);
bool kind_from_string(const std::string &s, Kind *k);
int kind_legs(Kind k);

struct BuildOptions {
  Kind kind = Kind::B;
  // Combine identical products (exact cancellations); false = keep raw forest terms
  bool prune = true;
  bool vertex_disjoint = true; // forest compatibility: nested or vertex-disjoint (Lagrangian rule)
  // Added to 2*omega where the min-over-terms exponent is <= 0 (t^2-protected faces)
  int omega_shift = 2;
  bool need_tables = true; // build omega / J tables (2^l entries)
  // Density exponents omega(S) -> scale * omega(S) (scale < 1: heavier, safer tails);
  // 0.7 = production default (from the omega scan at 4-5 loops); every command uses it
  double omega_scale = 0.7;
  bool allow_negative_faces = false; // accept faces whose pre-shift 2*omega_min < 0 (see build.cpp)
};

struct Part {
  uint32_t S = 0, Vm = 0;
  int legs = 0;
  int a = -1, b = -1; // terminals of a 2-point part (a == b: tadpole-type)
};

using Fac = std::vector<std::pair<int, int>>; // sorted (atom id, 2*exponent), exponents != 0

struct AtomDef {
  std::vector<int> g, u, v; // global line ids, local endpoints
  int nv = 0, nloops = 0;
  uint32_t mask = 0;
};

struct CompDef {
  double c = 0;
  int d0 = -1, d1 = -1;
  Fac f;
};

struct ClassData {
  Graph G;
  Kind kind = Kind::B;
  int l = 0, L = 0;
  double pw = 0, pref = 0; // mass exponent, Gamma(pw) pi^(-L/2)
  double S = 0; // symmetry factor E!/|Aut| (N = 1)
  std::vector<double> on; // O(N) factor polynomial
  bool is1vr = false, selfloop = false;
  // Combinatorics / pruning statistics
  int nparts = 0, ndots = 0;
  long nforests = 0, nterms_raw = 0, nterms_nonzero_forests = 0;
  int ncomp = 0, natoms = 0, max_atom_loops = 0, nshift = 0;
  double omega_min = 0; // min over proper S of omega(S)
  int omega2_pre_min = 0; // min over proper S of 2*omega(S) BEFORE the shift of t^2-protected faces
  int nneg_faces = 0; // faces with pre-shift 2*omega < 0 (refused unless allow_negative_faces)
  // Builder thread-CPU seconds: forest expansion + pruning + flat arrays; omega / J tables
  double t_expand = 0, t_tables = 0;
  std::vector<AtomDef> atoms;
  std::vector<CompDef> comps;
  std::vector<int> omega2; // 2*omega(S) (exact half-integers)
  std::vector<double> omega, Jt;
  double logJE = 0;
  // Flat arrays for the kernel view
  int ngroup = 0;
  std::vector<int> atom_off, atom_nv, grp_off, grp_foff, cf_atom;
  std::vector<signed char> ae_u, ae_v, atom_loc, cf_p2, comp_d0, comp_d1;
  std::vector<double> comp_coeff;
  RbClassView view() const;
  // Approximate host memory held by the class (tables, terms, atoms, flat arrays)
  size_t host_bytes() const;
};

bool build_class(const Graph &G, const BuildOptions &opt, ClassData &cd, std::string *err);

// Host evaluation of the measure-weighted projective integrand f(x) * Prod_e x_e at an arbitrary
// point (lx = log x, any scale), in double or long double; returns also Sum|terms|.
template <class R> R eval_point(const ClassData &cd, const double *lx, R *sabs);
