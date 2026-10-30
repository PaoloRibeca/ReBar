#!/usr/bin/env python3
# S5_largeN.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check S5: the large-N structure of the series -- the leading powers of N of the order sums B_L(N), H_L(N), C_L(N)
# and of the polynomials p_X,k(N) agree with the 1/N expansion in the same scheme.
#
# This program was designed and developed by the author(s),
# with the assistance of the following AI tool(s):
#   2026 Claude (Anthropic).
# The final logic and implementation were reviewed and verified in
# their entirety by the author(s).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""S5 -- the leading powers of N of the order sums and of the polynomials p_X,k(N) follow the 1/N expansion.

Claim (paper, Sec. II E, "New orders", item "Large N"; Tables tab:newpoly and tab:newpolynu): the leading power of N
of B_L(N), H_L(N) and C_L(N) at each order agrees with the 1/N expansion in the same scheme (massive, zero momentum,
d = 3, ubar = (N+8) g/(48 pi)): the N^L terms vanish, as the limits N -> infinity at fixed ubar,
    beta -> -ubar + ubar^2,   eta -> 0,   eta_phi2 -> -ubar,   nu -> 1/(2 - ubar)
(the bubble chains) require, and the N^(L-1) terms agree with the 1/N terms of the RG functions; in the polynomials
of Eq. series-form, p_beta,k has degree k-2 (k >= 3), p_eta,k and p_eta_phi2,k (k >= 2) degree k-1, and
[N^k] p_nu,k = 2^-(k+1) exactly.  The text quotes the numbers of paper/generated/newpoly.tex (\\largeN...).

The 1/N terms are computed by series/polynomials.py (large_n; the derivation is in its header): Z from the
sigma-exchange self-energy, Gamma^(4)(0) and Gamma^(2,1)(0;0) from the effective potential at order 1/N, with the
gap mass re-expressed through the renormalized one, as one-dimensional integrals.  Through the Callan-Symanzik
relations, the two leading coefficients of the polynomials p_X,k fix exactly the two leading coefficients of the sums
(the polynomials truncated to them give the sums' N^L and N^(L-1) coefficients through the inverse relations).

Inputs: integrator/tools/refs.py (Phi4tools' published polynomials, in exact arithmetic), the ledger (every class:
weight polynomial, pooled value and error), series/coefficients.py and series/polynomials.py,
paper/generated/newpoly.tex.
What runs:
  (a) the 1/N coefficients against the leading coefficients of Phi4tools' polynomials, beta k = 3..8, eta k = 2..8,
      nu k = 1..8, whose degrees and exact N^k coefficients of nu (2^-(k+1)) must be as predicted;
  (b) the predicted N^L and N^(L-1) coefficients of B_L, H_L, C_L against the exact B_1, B_2, H_2, C_1, C_2;
  (c) every class of every order of the ledger has a weight polynomial of degree at most L-1 (so [N^L] X_L = 0
      exactly, whatever the Monte Carlo values: the N = infinity limits hold identically), and the N^(L-1)
      coefficient of every order of the ledger (L = 2..10, also those the coefficients take from Phi4tools or exact
      values) against the prediction, with its Monte Carlo error (independent classes): pulls and chi^2;
  (d) the polynomials of series/polynomials.py: their degrees, [N^k] p_nu,k = 2^-(k+1) exactly with zero propagated
      variance, [N] p_eta_phi2,1 = -1, and their 1/N coefficients against the expansion (pulls; informational, they
      combine the sums of (c));
  (e) the macros \\largeN... of paper/generated/newpoly.tex against the numbers recomputed here.
PASS: (a) to 1e-8 relative, degrees and exact coefficients as predicted; (b) to 1e-12; (c) every class of degree at
most L-1, every |pull| < 3.5 and the chi^2 probability above 1e-3; (d) exactly; (e) every macro reproduced.
"""
import os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the leading powers of N of B_L, H_L, C_L and of the polynomials p_X,k follow the 1/N expansion in the same scheme (Sec. II E, Tables tab:newpoly, tab:newpolynu)"
QUICK = False


def main():
    PO = C.series_module("polynomials")
    ok = True
    led = PO.ledger_sums(C.LEDGER)
    res = PO.compute(C.LEDGER, "best", led=led, check=False)
    T = PO.large_n_tests(res)

    a = T["phi4tools"]
    print("(a) Phi4tools' polynomials (refs.py) against the 1/N expansion:")
    for f, k, deg, exact, j, got, want in a["rows"]:
        print("    %s_%d: degree %d%s; N^%d %+.15e, 1/N expansion %+.15e (rel. %.1e)"
              % (f, k, deg, "".join(", N^%d = %s" % (i, x) for i, x in sorted(exact.items(), reverse=True)), j, got, want,
                 abs(got - want) / abs(got)))
    oka = a["worst"] < 1e-8 and a["exact"]
    print("    largest relative difference %.1e; degrees and exact leading coefficients %s  %s"
          % (a["worst"], "as predicted" if a["exact"] else "NOT as predicted", "OK" if oka else "FAIL"))
    okb = T["exact_sums"] < 1e-12
    print("(b) the predicted N^L and N^(L-1) coefficients of B1, B2, H2, C1, C2 against the exact ones: largest difference "
          "%.1e  %s" % (T["exact_sums"], "OK" if okb else "FAIL"))
    c = T["ledger"]
    print("(c) the ledger: degree of the weight polynomials (largest over the classes) and the N^(L-1) coefficient of each order")
    for o, n, deg, L, top, err, want, z in c["rows"]:
        print("    %-4s %8d classes, degree %d (N^%d absent: %s); N^%d %+.10e +- %.2e, 1/N expansion %+.10e, pull %+.2f"
              % (o.upper(), n, deg, L, "yes" if deg <= L - 1 else "NO", L - 1, top, err, want, z))
    okc = c["degrees"] and c["zmax"] < 3.5 and c["p"] > 1e-3
    print("    every class of degree <= L-1: %s; chi^2 = %.1f for %d orders (p = %.2f), largest |pull| %.2f  %s"
          % ("yes" if c["degrees"] else "NO", c["chi2"], c["dof"], c["p"], c["zmax"], "OK" if okc else "FAIL"))
    d = T["polynomials"]
    print("(d) the polynomials p_X,k (series/polynomials.py), the terms of the tables (all %d checked):" % len(res["terms"]))
    for nm, deg, exact, j, got, err, want, this in d["rows"]:
        if nm in [PO.name(t) for t in PO.TABLE] or not this:
            print("    %-11s degree %d%s; N^%d %+.8e +- %.1e, 1/N expansion %+.8e, %s%s"
                  % (nm, deg, "".join(", N^%d = %s (error 0)" % (i, x) for i, x in sorted(exact.items(), reverse=True)), j, got,
                     err, want, ("pull %+.2f" % ((got - want) / err)) if err else ("rel. %.1e" % (abs(got - want) / abs(want))),
                     "" if this else "  DEGREE OR EXACT COEFFICIENT NOT AS PREDICTED"))
    print("    degrees and exact leading coefficients: %s  %s" % ("as predicted" if d["exact"] else "NOT as predicted",
                                                                 "OK" if d["exact"] else "FAIL"))
    want = dict(PO.large_n_macros(T))
    path = os.path.join(C.PAPER, "generated", "newpoly.tex")
    got = {}
    if os.path.exists(path):
        with open(path) as fh:
            for m in re.finditer(r"\\newcommand\*\{\\(largeN\w+)\}\{(.*)\}", fh.read()):
                got[m.group(1)] = m.group(2)
    bad = [k for k in want if got.get(k) != want[k]]
    oke = not bad
    print("(e) paper/generated/newpoly.tex: %s  %s" % (", ".join("\\%s = %s%s" % (k, want[k], "" if got.get(k) == want[k] else
                                                                  " (file: %s)" % got.get(k)) for k in want), "OK" if oke else "FAIL"))
    ok = oka and okb and okc and d["exact"] and oke
    C.verdict("S5", ok, "1/N expansion = Phi4tools' leading coefficients to %.0e; N^L absent from every class; N^(L-1) of "
              "%d ledger orders against it: chi^2 = %.1f (p = %.2f), largest |pull| %.2f; degrees and 2^-(k+1) of p_nu,k exact"
              % (a["worst"], c["dof"], c["chi2"], c["p"], c["zmax"]))


if __name__ == "__main__":
    main()
