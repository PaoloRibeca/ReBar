/*
    kernel.h -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Single-source (C++17 / CUDA) sampling and evaluation of one graph
    class's R-bar integrand.

    A "class" is one graph (with all its dot insertions summed) whose
    projective integrand has been expanded on the host (src/build.cpp) into
        f(x) = (Sum_e x_e)^(-pw) * Sum_k c_k * xdot_k
               * Prod_a P_a(x)^(p_ka / 2)
    where the P_a ("atoms") are Kirchhoff polynomials of 2-connected blocks
    of graph minors (one per distinct block over all forests, all dots, all
    jet orders), xdot_k is the dot monomial (x_e, x_e x_f or x_e^2) and the
    coefficients c_k are exact dyadic rationals left after algebraic
    pruning of exactly-cancelling terms.

    Sampling: forest-aware tropical (Hepp-sector) density built from
    omega(S) = |S| + min_k deg_S(term k) (+1 on faces where that minimum is
    <= 0, i.e. faces protected by a t^2 subtraction), via Borinsky's
    recursion J(S) = Sum_{e in S} J(S\e) / omega(S\e).  One sample = 2(l-1)
    Philox uniforms -> (sector, point) -> weight w = J(E) * f(x) * Prod_e
    x_e / density, so that Int_simplex f = E[w].

    Arithmetic: point generation in double; evaluation templated on R
    (double on the device; double or long double on the host -- the latter
    is used only for paired round-off checks).  Each atom is evaluated in
    log space as log P = log P^trop + log det(M), M = rescaled loop matrix
    of the Kruskal (minimum) spanning forest in the sampled order: entries
    are exp(lx_e - lx_f) <= 1, det(M) in [1, #spanning trees], so no
    overflow/underflow for any point the sampler can produce.  Terms are
    summed with Neumaier compensation.

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
#include "config.h"
#include "philox.h"

/* Flat, pointer-based view of one class.  The pointers may refer to host or device memory. */
struct RbClassView {
  int l; /* lines of the graph (<= RB_MAXE) */
  int natoms, ncomp;
  double pw; /* mass-factor exponent: (Sum x)^(-pw) */
  double logJE; /* log J(E): normalisation of the tropical density */
  const double *omega; /* [2^l] density exponents omega(S) */
  const double *Jt; /* [2^l] J(S) */
  const int *atom_off; /* [natoms+1] offsets into ae_u / ae_v */
  const int *atom_nv; /* [natoms] vertices of the atom graph */
  const signed char *ae_u, *ae_v; /* local endpoints of the atom's edges (local edge order) */
  const signed char *atom_loc; /* [natoms * RB_MAXE] global edge -> local edge index, or -1 */
  int ngroup; /* terms grouped by identical atom product (they differ only by the dots) */
  const int *grp_off; /* [ngroup+1] term ranges */
  const int *grp_foff; /* [ngroup+1] factor ranges into cf_atom / cf_p2 */
  const int *cf_atom; /* atom index of each factor */
  const signed char *cf_p2; /* 2 x exponent of each factor (half-integer exponents) */
  const double *comp_coeff; /* c_k */
  const signed char *comp_d0, *comp_d1; /* dot edges of term k (-1 = none; d0 == d1 means x_e^2) */
};

/* Sample one point.  ord[0..l-1]: edges in ascending order of x (ord[l-1] = largest, lx = 0).
   Returns Sum_j log v_j (log of the radial uniforms), from which the density is reconstructed. */
RB_HDS double rb_point(const RbClassView &P, RbStream *rs, signed char *ord, double *lx) {
  const int l = P.l;
  uint32_t S = (l >= 32) ? 0xffffffffu : ((1u << l) - 1u);
  double cum = 0, lvs = 0;
  for (int k = l - 1; k >= 1; --k) {
    const double target = rb_next(rs) * RB_LDG(P.Jt + S);
    uint32_t T = S;
    int e = -1;
    double acc = 0;
    while (T) {
      const int b = RB_CTZ(T);
      T &= T - 1;
      const uint32_t Sb = S & ~(1u << b);
      acc += RB_LDG(P.Jt + Sb) / RB_LDG(P.omega + Sb);
      if (acc >= target || T == 0) { e = b; break; }
    }
    ord[k] = (signed char)e;
    lx[e] = cum;
    S &= ~(1u << e);
    const double lv = rb_log(rb_next(rs));
    cum += lv / RB_LDG(P.omega + S);
    lvs += lv;
  }
  const int e0 = RB_CTZ(S);
  ord[0] = (signed char)e0;
  lx[e0] = cum;
  return lvs;
}

/* log P_a at the point: Kruskal minimum spanning forest in ascending-x order, fundamental cycles as
   bitmasks over local tree edges (+ orientation), rescaled loop matrix M_ij = Sum_{e in C_i n C_j}
   +- x_e/sqrt(x_fi x_fj) (M_ii = 1 + Sum x_e/x_fi <= ... all entries bounded by the Kruskal
   property), det(M) = U/U^trop by LDL^T.
   Fast path: x[] = exp(lx) and isq[] = x^(-1/2) precomputed per sample (valid when all lx >
   RB_LXMIN); otherwise (xs == 0) every ratio is formed as exp(lx_e - lx_f), which cannot overflow
   or underflow harmfully. */
template <class R>
RB_HDS R rb_atom_logP(const RbClassView &P, int a, const signed char *ord, const R *lx, const R *xs,
    const R *isq) {
  const int off = P.atom_off[a];
  const int nv = P.atom_nv[a];
  const signed char *loc = P.atom_loc + (size_t)a * RB_MAXE;
  const signed char *eu = P.ae_u + off, *ev = P.ae_v + off;
  signed char uf[RB_MAXV];
  for (int v = 0; v < nv; ++v) uf[v] = (signed char)v;
  uint32_t intree = 0;
  signed char nt[RB_MAXLOOP];
  signed char ntg[RB_MAXLOOP];
  signed char g_of[RB_MAXE];
  int nnt = 0;
  R lt = 0;
  for (int i = 0; i < P.l; ++i) {
    const int g = ord[i], le = loc[g];
    if (le < 0) continue;
    g_of[le] = (signed char)g;
    int x = eu[le], y = ev[le];
    while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; }
    while (uf[y] != y) { uf[y] = uf[uf[y]]; y = uf[y]; }
    if (x != y) { uf[x] = (signed char)y; intree |= 1u << le; }
    else { nt[nnt] = (signed char)le; ntg[nnt] = (signed char)g; ++nnt; lt += lx[g]; }
  }
  if (nnt == 0) return lt;
  /* Tree as parent pointers (BFS from every unvisited vertex) */
  uint32_t adj[RB_MAXV];
  signed char par[RB_MAXV], pe[RB_MAXV], dep[RB_MAXV], q[RB_MAXV];
  for (int v = 0; v < nv; ++v) { adj[v] = 0; par[v] = -2; }
  const int ne = P.atom_off[a + 1] - off;
  for (int le = 0; le < ne; ++le)
    if ((intree >> le) & 1u) { adj[eu[le]] |= 1u << le; adj[ev[le]] |= 1u << le; }
  for (int r = 0; r < nv; ++r) {
    if (par[r] != -2) continue;
    int h = 0, t = 0;
    q[t++] = (signed char)r; par[r] = -1; pe[r] = -1; dep[r] = 0;
    while (h < t) {
      const int x = q[h++];
      uint32_t m = adj[x];
      while (m) {
        const int le = RB_CTZ(m);
        m &= m - 1;
        const int y = eu[le] == x ? ev[le] : eu[le];
        if (par[y] == -2) {
          par[y] = (signed char)x;
          pe[y] = (signed char)le;
          dep[y] = (signed char)(dep[x] + 1);
          q[t++] = (signed char)y;
        }
      }
    }
  }
  /* Fundamental cycles: tree-edge bitmask + orientation bitmask */
  uint32_t cm[RB_MAXLOOP], sm[RB_MAXLOOP];
  for (int i = 0; i < nnt; ++i) {
    const int f = nt[i];
    int A = ev[f], B = eu[f];
    uint32_t c = 0, s = 0;
    while (A != B) {
      if (dep[A] >= dep[B]) {
        const int e = pe[A];
        c |= 1u << e;
        if (eu[e] != A) s |= 1u << e;
        A = par[A];
      } else { const int e = pe[B]; c |= 1u << e; if (ev[e] != B) s |= 1u << e; B = par[B]; }
    }
    cm[i] = c; sm[i] = s;
  }
  R M[RB_MAXLOOP * (RB_MAXLOOP + 1) / 2];
  if (xs) {
    for (int i = 0; i < nnt; ++i) {
      const R ii = isq[ntg[i]], ii2 = ii * ii;
      R d = 1;
      uint32_t m = cm[i];
      while (m) { const int e = RB_CTZ(m); m &= m - 1; d += xs[g_of[e]] * ii2; }
      M[i * (i + 1) / 2 + i] = d;
      for (int j = 0; j < i; ++j) {
        uint32_t mm = cm[i] & cm[j];
        if (!mm) { M[i * (i + 1) / 2 + j] = 0; continue; }
        const uint32_t sg = sm[i] ^ sm[j];
        R o = 0;
        while (mm) {
          const int e = RB_CTZ(mm);
          mm &= mm - 1;
          const R v = xs[g_of[e]];
          o += ((sg >> e) & 1u) ? -v : v;
        }
        M[i * (i + 1) / 2 + j] = o * ii * isq[ntg[j]];
      }
    }
  } else {
    for (int i = 0; i < nnt; ++i) {
      const R lfi = lx[ntg[i]];
      R d = 1;
      uint32_t m = cm[i];
      while (m) { const int e = RB_CTZ(m); m &= m - 1; d += rb_exp(lx[g_of[e]] - lfi); }
      M[i * (i + 1) / 2 + i] = d;
      for (int j = 0; j < i; ++j) {
        uint32_t mm = cm[i] & cm[j];
        const uint32_t sg = sm[i] ^ sm[j];
        const R half = (R)0.5 * (lfi + lx[ntg[j]]);
        R o = 0;
        while (mm) {
          const int e = RB_CTZ(mm);
          mm &= mm - 1;
          const R v = rb_exp(lx[g_of[e]] - half);
          o += ((sg >> e) & 1u) ? -v : v;
        }
        M[i * (i + 1) / 2 + j] = o;
      }
    }
  }
  /* LDL^T (packed): L_ij stored below the diagonal, D_j on the diagonal; det = Prod D_j in [1,
     #trees] */
  R det = 1;
  for (int j = 0; j < nnt; ++j) {
    R d = M[j * (j + 1) / 2 + j];
    for (int k = 0; k < j; ++k) {
      const R v = M[j * (j + 1) / 2 + k];
      d -= v * v * M[k * (k + 1) / 2 + k];
    }
    M[j * (j + 1) / 2 + j] = d;
    det *= d;
    const R id = (R)1 / d;
    for (int i = j + 1; i < nnt; ++i) {
      R s = M[i * (i + 1) / 2 + j];
      for (int k = 0; k < j; ++k)
        s -= M[i * (i + 1) / 2 + k] * M[j * (j + 1) / 2 + k] * M[k * (k + 1) / 2 + k];
      M[i * (i + 1) / 2 + j] = s * id;
    }
  }
  return lt + rb_log(det);
}

/* Evaluate Sum_g exp(logT + Sum_a p_ga/2 log P_a) * Sum_{k in g} c_k xdot_k with Neumaier summation
   over groups.
   xs = exp(lx) (fast path) or null (then every term is exponentiated in log space).
   Returns the sum; *sabs = Sum_k |term_k| (for cancellation / round-off diagnostics). */
template <class R>
RB_HDS R rb_terms(const RbClassView &P, const R *lx, const R *xs, const R *logP, int lps, R logT,
    R *sabs) {
  R s = 0, c = 0, sa = 0;
  for (int g = 0; g < P.ngroup; ++g) {
    R e = logT;
    for (int f = P.grp_foff[g]; f < P.grp_foff[g + 1]; ++f)
      e += (R)(0.5 * P.cf_p2[f]) * logP[P.cf_atom[f] * lps];
    R t = 0, ta = 0;
    if (xs) {
      const R E = rb_exp(e);
      for (int k = P.grp_off[g]; k < P.grp_off[g + 1]; ++k) {
        const int d0 = P.comp_d0[k], d1 = P.comp_d1[k];
        R m = (R)P.comp_coeff[k];
        if (d0 >= 0) m *= xs[d0];
        if (d1 >= 0) m *= xs[d1];
        t += m; ta += rb_fabs(m);
      }
      t *= E; ta *= E;
    } else {
      for (int k = P.grp_off[g]; k < P.grp_off[g + 1]; ++k) {
        const int d0 = P.comp_d0[k], d1 = P.comp_d1[k];
        R ek = e;
        if (d0 >= 0) ek += lx[d0];
        if (d1 >= 0) ek += lx[d1];
        const R m = (R)P.comp_coeff[k] * rb_exp(ek);
        t += m; ta += rb_fabs(m);
      }
    }
    const R y = s + t;
    c += (rb_fabs(s) >= rb_fabs(t)) ? (s - y) + t : (t - y) + s;
    s = y;
    sa += ta;
  }
  *sabs = sa;
  return s + c;
}

/* Integrand weight at a given point (lx in log coordinates, any chart; ord = ascending order of
   lx).
   lvs = Sum log v of the radial uniforms that produced the point (density).
   logP: per-thread scratch of natoms values, element a at logP[a * lps] (lps = 1 on the CPU; on the
   GPU the scratch of all threads is interleaved, lps = number of threads, so that a warp's accesses
   coalesce). */
template <class R>
RB_HDS R rb_weight_at(const RbClassView &P, const signed char *ord, const double *lxd, double lvs,
    R *logP, R *sabs, int lps = 1) {
  /* W: precision of the per-sample bookkeeping (log density, mass factor) = R, but double when R =
     float */
  typedef typename rb_wide<R>::T W;
  R lx[RB_MAXE], xs[RB_MAXE], isq[RB_MAXE];
  W sumlx = 0, sx = 0;
  bool fast = true;
  const double lxmin = rb_lxmin((R)0);
  for (int e = 0; e < P.l; ++e) {
    lx[e] = (R)lxd[e]; sumlx += (W)lxd[e];
    xs[e] = rb_exp(lx[e]); sx += (W)xs[e];
    if (lxd[e] < lxmin) fast = false;
  }
  if (fast) for (int e = 0; e < P.l; ++e) isq[e] = (R)1 / rb_sqrt(xs[e]);
  for (int a = 0; a < P.natoms; ++a)
    logP[a * lps] = rb_atom_logP<R>(P, a, ord, lx, fast ? xs : (const R *)0, isq);
  const R logT = (R)((W)P.logJE + sumlx - (W)lvs - (W)P.pw * rb_log(sx));
  return rb_terms<R>(P, lx, fast ? xs : (const R *)0, logP, lps, logT, sabs);
}

/* One Monte Carlo sample: Philox stream (key, stream, idx) -> point -> weight.  E[w] = Int_simplex
   f. */
template <class R>
RB_HDS R rb_sample(const RbClassView &P, uint64_t key, uint32_t stream, uint64_t idx, R *logP,
    R *sabs, int lps = 1) {
  RbStream rs;
  rb_stream_init(&rs, key, stream, idx);
  signed char ord[RB_MAXE];
  double lx[RB_MAXE];
  const double lvs = rb_point(P, &rs, ord, lx);
  return rb_weight_at<R>(P, ord, lx, lvs, logP, sabs, lps);
}

/* Per-stream accumulator (FP64, Neumaier-compensated first and second moments + diagnostics). */
struct RbAcc {
  double n; /* samples */
  double s, sc; /* Sum w (sum + compensation) */
  double q, qc; /* Sum w^2 */
  double q3, q4; /* Sum w^3, Sum w^4 (kurtosis -> reliability of the sd estimate) */
  double sabs2; /* Sum (Sum_k |term_k|)^2: round-off scale, error ~ eps * sqrt(sabs2/n) */
  double maxw; /* max |w| */
  double nbad; /* non-finite weights (excluded from the sums, reported) */
  double ncanc; /* samples with |w| < 1e-8 * Sum|terms| (deep cancellation) */
  double maxsabs; /* max Sum|terms|: worst single-sample round-off scale (eps * maxsabs) */
};
RB_HDI void rb_acc_zero(RbAcc *a) {
  a->n = a->s = a->sc = a->q = a->qc = a->q3 = a->q4 = a->sabs2 = a->maxw = a->nbad = a->ncanc
      = a->maxsabs = 0;
}
RB_HDI void rb_neu(double *s, double *c, double t) {
  const double y = *s + t;
  *c += (fabs(*s) >= fabs(t)) ? (*s - y) + t : (t - y) + *s;
  *s = y;
}
RB_HDI void rb_acc_add(RbAcc *a, double w, double sabs) {
  if (!isfinite(w) || !isfinite(sabs)) { a->nbad += 1; return; }
  a->n += 1;
  rb_neu(&a->s, &a->sc, w);
  rb_neu(&a->q, &a->qc, w * w);
  a->q3 += w * w * w;
  a->q4 += (w * w) * (w * w);
  a->sabs2 += sabs * sabs;
  if (fabs(w) > a->maxw) a->maxw = fabs(w);
  if (fabs(w) < 1e-8 * sabs) a->ncanc += 1;
  if (sabs > a->maxsabs) a->maxsabs = sabs;
}
RB_HDI void rb_acc_merge(RbAcc *a, const RbAcc *b) {
  a->n += b->n;
  rb_neu(&a->s, &a->sc, b->s); rb_neu(&a->s, &a->sc, b->sc);
  rb_neu(&a->q, &a->qc, b->q); rb_neu(&a->q, &a->qc, b->qc);
  a->q3 += b->q3; a->q4 += b->q4;
  a->sabs2 += b->sabs2;
  if (b->maxw > a->maxw) a->maxw = b->maxw;
  a->nbad += b->nbad; a->ncanc += b->ncanc;
  if (b->maxsabs > a->maxsabs) a->maxsabs = b->maxsabs;
}
