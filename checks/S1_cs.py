#!/usr/bin/env python3
# S1_cs.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check S1: the Callan-Symanzik relations map Phi4tools' sums back to Phi4tools' coefficients, and the exact
# two-loop sums to the exact low-order coefficients.
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
"""S1 -- the Callan-Symanzik relations map Phi4tools' sums to Phi4tools' coefficients to 1e-15.

Claim (paper, App. A; series/README.md "From sums to coefficients"): with u = g/8pi, ubar = (N+8)u/6 and
    eta = 2h/(1+h),  beta_u = -u + 2 eta u + (2 - eta) btilde,  1/nu = (2 - eta)(1 - c),
the per-order sums B_L, H_L, C_L that integrator/tools/refs.py derives from Phi4tools' series of beta, eta and nu
(Sberveglieri and Spada, JHEP 05 (2024) 073, App. D), fed to the relations of series/coefficients.py, give back
Phi4tools' coefficients to round-off (1e-15 relative).

Inputs: integrator/tools/refs.py (Phi4tools' O(N) series, transcribed), series/coefficients.py (the relations).
What runs: (a) coefficients.compute(policy="refs"): beta, eta, nu in ubar through ubar^8 at N = 0..4, against
refs.py's own transcription of Phi4tools; (b) an independent exact implementation of the same relations in
rational arithmetic (written here) on the exact low orders B1 = (N+8)/12, B2 = (5N+22)/54, H2 = (N+2)/243,
C1 = (N+2)/12, C2 = (N+2)/36, against the exact coefficients beta_1 = -1, beta_2 = 1,
beta_3 = -4(41N+190)/(27(N+8)^2), eta_2 = 8(N+2)/(27(N+8)^2), nu_0 = 1/2, nu_1 = (N+2)/(4(N+8)),
nu_2 = (27N^2+16N-76)/(216(N+8)^2).
PASS: (a) max relative difference <= 1e-15 over every nonzero coefficient; (b) exact equality for N = 0..4.
"""
import os, sys
from fractions import Fraction as F

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the Callan-Symanzik relations map Phi4tools' sums to Phi4tools' coefficients to 1e-15 (App. A)"
QUICK = True
K = 4    # exact series through u^3


def mul(a, b):
    return [sum(a[i] * b[n - i] for i in range(n + 1)) for n in range(K)]


def inv(a):
    r = [F(1) / a[0]]
    for n in range(1, K):
        r.append(-sum(a[j] * r[n - j] for j in range(1, n + 1)) / a[0])
    return r


def exact_cs(N):
    """beta, eta, nu in ubar from the exact two-loop sums, in rational arithmetic"""
    B = {1: F(N + 8, 12), 2: F(5 * N + 22, 54)}
    H = {2: F(N + 2, 243)}
    Cs = {1: F(N + 2, 12), 2: F(N + 2, 36)}
    bt = [F(0)] * K
    h = [F(0)] * K
    c = [F(0)] * K
    for L, v in B.items():
        bt[L + 1] = (-1) ** (L + 1) * v
    for L, v in H.items():
        h[L] = (-1) ** L * v
    for L, v in Cs.items():
        c[L] = (-1) ** (L + 1) * v
    one_h = [F(1)] + h[1:]
    eta = [2 * x for x in mul(h, inv(one_h))]
    tme = [2 - eta[0]] + [-x for x in eta[1:]]
    beta = mul(tme, bt)
    for i in range(K - 1):
        beta[i + 1] += 2 * eta[i]
    beta[1] -= 1
    omc = [1 - c[0]] + [-x for x in c[1:]]
    nu = inv(mul(tme, omc))
    k = F(N + 8, 6)
    return ([beta[n] * k ** (1 - n) for n in range(K)], [eta[n] / k ** n for n in range(K)], [nu[n] / k ** n for n in range(K)])


def main():
    CO = C.series_module("coefficients")
    refs = C.load("refs", os.path.join(C.ITOOLS, "refs.py"))
    res = CO.compute(policy="refs")
    worst, where, n = 0.0, None, 0
    for N in range(5):
        pt = {"beta": refs.beta_coeffs(N), "eta": refs.eta_coeffs(N), "nu": refs.nu_coeffs(N)}
        for f, cs in pt.items():
            for k in range(1, len(cs)):
                v = cs[k][0]
                if not v:
                    continue
                w = res["coefficients"][f][str(k)]["N"][str(N)][0]
                d = abs(w - v) / abs(v)
                n += 1
                if d > worst:
                    worst, where = d, "%s_%d at N = %d" % (f, k, N)
    print("(a) Phi4tools -> B_L, H_L, C_L (refs.py) -> CS relations (coefficients.py) -> beta, eta, nu: %d coefficients "
          "(ubar^1..8, N = 0..4), max relative difference %.1e (%s)" % (n, worst, where))
    bad = []
    for N in range(5):
        b, e, v = exact_cs(N)
        want = {"beta_1": (b[1], F(-1)), "beta_2": (b[2], F(1)), "beta_3": (b[3], F(-4 * (41 * N + 190), 27 * (N + 8) ** 2)),
                "eta_2": (e[2], F(8 * (N + 2), 27 * (N + 8) ** 2)), "nu_0": (v[0], F(1, 2)), "nu_1": (v[1], F(N + 2, 4 * (N + 8))),
                "nu_2": (v[2], F(27 * N * N + 16 * N - 76, 216 * (N + 8) ** 2))}
        for name, (got, exp) in want.items():
            if got != exp:
                bad.append("%s at N = %d: %s, expected %s" % (name, N, got, exp))
        print("(b) N = %d: beta_3 = %s, eta_2 = %s, nu_2 = %s (exact, from B1, B2, H2, C1, C2)" % (N, b[3], e[2], v[2]))
    for x in bad:
        print("    MISMATCH " + x)
    C.verdict("S1", worst <= 1e-15 and not bad, "round trip through the CS relations: %d coefficients to %.1e; exact low orders "
              "(beta_1-3, eta_2, nu_0-2, N = 0..4) %s" % (n, worst, "equal" if not bad else "DIFFER (%d)" % len(bad)))


if __name__ == "__main__":
    main()
