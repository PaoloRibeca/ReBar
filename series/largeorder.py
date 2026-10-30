#!/usr/bin/env python3
"""
    largeorder.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    The large-order test of the series: every coefficient of beta, eta and eta_phi2 in Nickel's coupling
    gN = (N+8) g/(48 pi) (Phi4tools' ubar) is divided by its leading large-order estimate
    c (-1/A)^k Gamma(k+b) with the instanton constants of d = 3, as implemented by Riccardo Guida in
    ribeta.asymptotics.m (2003-04) for the seven-loop computation: Brezin-Parisi constants, A = 113.383507815
    (Zinn-Justin, Quantum Field Theory and Critical Phenomena, 3rd ed., Eq. (28.73): a(d = 3) = 0.147774232 * 9/(N+8)),
    and the exponents
    b_beta = 3 + N/2, b_eta = 2 + N/2, b_eta2 = 3 + N/2 in the Gamma(k+b+1) convention (a = 48 pi/(9 A) =
    0.14777423 at N = 1). The ratios must approach 1; the test catches an error of a per cent in a new
    coefficient, not a smaller one (thesis section 6.3.2.2; the 2004 criterion: the last three ratios decrease
    towards 1).

    usage: largeorder.py [--ledger DIR] [--N 0,1,2,3,4] [--tex FILE] [--fit-tex FILE]
    --fit-tex: the ratios of beta and eta_phi2 to their leading estimates are fitted, for N = 0..4, with
    c0 (1 + d1/k + d2/k^2) from k = 6 on, each of the last three orders is predicted from the lower ones, and the
    numbers the text quotes are written as macros (paper/generated/largeorder-fit.tex).
    Reads the coefficients through coefficients.py; --tex writes the paper's table (a ruledtabular block, N = 1).

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
"""
import os
import sys
import math
from math import exp, factorial, gamma, pi, sqrt

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coefficients as CO  # noqa: E402

# d = 3 constants (Brezin-Parisi; 48 pi/(9 A) = a(d = 3) of Zinn-Justin's book, 3rd ed., Eq. (28.73)), as in ribeta.asymptotics.m
DBAR1, DBAR13, I1, AA, I6, H3 = 10.544, 1.4571, 31.691522, 113.383507815, 659.868352, 13.563312
D = 3
PM = -1   # sign of the instanton integral from the square root of the determinant


def nickel(n): return (n + 8) / (48 * pi)
def bb(e, n): return (n - 1 + D + e) / 2
def kk(e, n): return pi ** (n / 2) * factorial(e) / (factorial(e // 2) * 2 ** (e - 1) * gamma((n + e) / 2))


def dz(n):
    i4 = 2 / 3 * AA
    return D ** (-D / 2) * (2 * AA / pi) ** bb(0, n) * DBAR1 ** -0.5 * DBAR13 ** (-(n - 1) / 2) * (I6 / i4 - 1) ** (D / 2)


def dzscaled(e, n): return (PM / (2 * pi)) / (AA ** bb(e, n)) * dz(n) * kk(e, n)
def ff(e): return -(sqrt(6) * I1) ** e
def fac(k, b): return -(-1 / AA) ** k * gamma(k + b)


def asymptotic(k, f, n):
    """Leading large-order estimate of the coefficient of gN^k in f = beta, eta or eta_phi2 at N = n."""
    nk = nickel(n)
    if f == "beta":
        return (-(4 - D) * exp(-nk * AA)) * dzscaled(4, n) * ff(4) * fac(k, bb(4, n) + 1) * nk ** (1 - k)
    if f == "eta":
        return ((4 - D) * exp(-nk * AA)) * dzscaled(2, n) * (6 * H3 * I1 / D) * fac(k, bb(2, n) + 1) * nk ** (-k)
    if f == "eta_phi2":
        return (-(4 - D) ** 2 / 2 * exp(-nk * AA)) * dzscaled(2, n) * ff(2) * fac(k, bb(2, n) + 2) * nk ** (-k)
    raise ValueError(f)


def ratios(res, f, n):
    """(k, coefficient, error, estimate, ratio) for every available order k >= 3 of f at N = n."""
    out = []
    for k in sorted(int(x) for x in res["coefficients"][f]):
        if k < 3:
            continue
        v, e = res["coefficients"][f][str(k)]["N"][str(n)][:2]
        a = asymptotic(k, f, n)
        out.append((k, v, e, a, v / a))
    return out


# ------------------------------------------------------------------------------------------- subleading corrections
KMIN = 6          # first order of the fits (the ratios settle from k = 6 on)


def lsq(rows, ys):
    """least squares for a few parameters (normal equations, Gaussian elimination with pivoting)"""
    k = len(rows[0])
    A = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda q: abs(A[q][i])); A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        for q in range(k):
            if q != i:
                f = A[q][i] / A[i][i]; A[q] = [x - f * y for x, y in zip(A[q], A[i])]; b[q] -= f * b[i]
    return [b[i] / A[i][i] for i in range(k)]


def fit(rs, kmin=KMIN, kmax=None):
    """fit ratio_k = c0 (1 + d1/k + d2/k^2) to the orders kmin..kmax of {k: ratio}; (c0, d1, d2, max |residual|)"""
    ks = [k for k in sorted(rs) if k >= kmin and (kmax is None or k <= kmax)]
    c = lsq([[1.0, 1.0 / k, 1.0 / k ** 2] for k in ks], [rs[k] for k in ks])
    res = max(abs(rs[k] - (c[0] + c[1] / k + c[2] / k ** 2)) / abs(rs[k]) for k in ks)
    return c[0], c[1] / c[0], c[2] / c[0], res


def predictions(rs, last=3, kmin=KMIN):
    """{K: predicted/computed - 1} for the last orders K, each predicted by the fit to kmin..K-1 (at least 4 orders)"""
    out = {}
    for K in sorted(rs)[-last:]:
        if len([k for k in rs if kmin <= k < K]) < 4:
            continue
        c0, d1, d2, _ = fit(rs, kmin, K - 1)
        out[K] = c0 * (1 + d1 / K + d2 / K ** 2) / rs[K] - 1
    return out


def fit_report(res, ns=(0, 1, 2, 3, 4)):
    """per N and function (beta, eta_phi2): the fit of the ratios and the leave-out predictions of the last orders"""
    rep = {}
    for n in ns:
        for f in ("beta", "eta_phi2"):
            rs = {k: r for k, _, _, _, r in ratios(res, f, n)}
            c0, d1, d2, mres = fit(rs)
            rep[(n, f)] = {"c0": c0, "d1": d1, "d2": d2, "res": mres, "kmax": max(rs), "pred": predictions(rs)}
    return rep


def fit_tex(rep, path):
    """macros for the text: \\lofit... (see the comments in the file)"""
    def sci1(x):
        e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
        return "%.0f\\times10^{%d}" % (m, e) if round(m) < 10 else "10^{%d}" % (e + 1)
    beta = [v for (n, f), v in rep.items() if f == "beta"]; phi = [v for (n, f), v in rep.items() if f == "eta_phi2"]
    pb = [abs(x) for v in beta for k, x in v["pred"].items() if k >= 9]
    pb1 = [abs(x) for (n, f), v in rep.items() if f == "beta" and n == 1 for k, x in v["pred"].items() if k >= 9]
    pp = [abs(x) for v in phi for k, x in v["pred"].items() if k >= 9]
    L = ["% generated/largeorder-fit.tex -- the subleading fit of the large-order ratios (Sec. III C), N = 0..4.",
         "% WRITTEN BY series/largeorder.py --fit-tex (make -C ../series): do not edit by hand.",
         "%% ratio_k = c0 (1 + d1/k + d2/k^2) fitted to k = %d..kmax; predictions: each of the last orders from the lower ones." % KMIN,
         "\\newcommand*{\\lofitkmin}{%d}" % KMIN,
         "\\newcommand*{\\lofitkmaxbeta}{%d}\\newcommand*{\\lofitkmaxphi}{%d}" % (max(v["kmax"] for v in beta), max(v["kmax"] for v in phi)),
         "\\newcommand*{\\lofitresbeta}{%s}\\newcommand*{\\lofitresphi}{%s}" % (sci1(max(v["res"] for v in beta)), sci1(max(v["res"] for v in phi))),
         "\\newcommand*{\\lofitdonemin}{%.1f}\\newcommand*{\\lofitdonemax}{%.1f}" % (min(v["d1"] for v in beta + phi), max(v["d1"] for v in beta + phi)),
         "\\newcommand*{\\lofitdtwomin}{%.0f}\\newcommand*{\\lofitdtwomax}{%.0f}" % (min(v["d2"] for v in beta + phi), max(v["d2"] for v in beta + phi)),
         "\\newcommand*{\\lofitcomin}{%.1f}\\newcommand*{\\lofitcomax}{%.1f}" % (min(v["c0"] for v in beta + phi), max(v["c0"] for v in beta + phi)),
         "\\newcommand*{\\lofitpredbeta}{%s}\\newcommand*{\\lofitpredbetaone}{%s}\\newcommand*{\\lofitpredphi}{%s}" % (sci1(max(pb)), sci1(max(pb1)), sci1(max(pp))),
         ""]
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def main(argv):
    ledger = None; ns = [1]; tex = None; fitpath = None
    it = iter(argv)
    for x in it:
        if x == "--ledger": ledger = next(it)
        elif x == "--N": ns = [int(t) for t in next(it).split(",")]
        elif x == "--tex": tex = next(it)
        elif x == "--fit-tex": fitpath = next(it)
        else: raise SystemExit(__doc__)
    res = CO.compute(ledger) if ledger else CO.compute()
    rows = {}
    for n in ns:
        print("N = %d: ratio of each coefficient of gN^k to its leading large-order estimate" % n)
        print("  k   beta                 eta                  eta_phi2")
        rb, re_, r2 = (dict((r[0], r) for r in ratios(res, f, n)) for f in ("beta", "eta", "eta_phi2"))
        for k in sorted(set(rb) | set(re_) | set(r2)):
            cells = []
            for r in (rb, re_, r2):
                cells.append("%8.4f" % r[k][4] if k in r else "        ")
            print("  %d   %s" % (k, "   ".join(cells)))
        rows[n] = (rb, re_, r2)
    if tex:
        rb, re_, r2 = rows[1] if 1 in rows else rows[ns[0]]
        L = ["% generated/largeorder.tex -- body of Table tab:largeorder: coefficient / leading large-order estimate at N = 1.",
             "% WRITTEN BY series/largeorder.py (make -C ../series): do not edit by hand.",
             "\\begin{ruledtabular}", "\\begin{tabular}{lrrr}",
             "$k$ & $\\beta$ & $\\eta$ & $\\eta_{\\phi^2}$ \\\\", "\\hline"]
        for k in sorted(set(rb) | set(re_) | set(r2)):
            c = []
            for r in (rb, re_, r2):
                if k in r:
                    s = "%.3f" % r[k][4]
                    c.append(s)
                else:
                    c.append("---")
            L.append("%d & %s \\\\" % (k, " & ".join(c)))
        L += ["\\end{tabular}", "\\end{ruledtabular}", ""]
        with open(tex, "w") as f:
            f.write("\n".join(L))
        print("wrote", os.path.relpath(tex, os.path.dirname(HERE)))
    if fitpath:
        rep = fit_report(res)
        print("subleading fit ratio = c0 (1 + d1/k + d2/k^2), k = %d..kmax; predicted/computed - 1 for the last orders" % KMIN)
        for (n, f), v in sorted(rep.items()):
            print("  N=%d %-8s c0 %.4f  d1 %+.3f  d2 %+.3f  max residual %.1e   %s" % (n, f, v["c0"], v["d1"], v["d2"], v["res"],
                  "  ".join("k=%d: %+.1e" % (k, x) for k, x in sorted(v["pred"].items()))))
        fit_tex(rep, fitpath)
        print("wrote", os.path.relpath(fitpath, os.path.dirname(HERE)))


if __name__ == "__main__":
    main(sys.argv[1:])
