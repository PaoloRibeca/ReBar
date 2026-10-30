#!/usr/bin/env python3
"""
    resum.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    The resummation of the fixed-dimension series (paper, Sec. "Resummation"): Borel-Leroy transform with the
    conformal mapping of the Borel plane (Le Guillou & Zinn-Justin 1977, 1980; Guida & Zinn-Justin 1998).
    For a series F(u) = sum_k F_k u^k and parameters (b, alpha),
        F(u) ~ sum_{k<=K} d_k int_0^inf dt t^b e^-t w(ut)^k / (1 - w(ut))^alpha,   w(x) = (sqrt(1+ax) - 1)/(sqrt(1+ax) + 1),
    with the d_k fixed by F_0 .. F_K and a = 48 pi / ((N+8) A), A = 113.383507815 the instanton action (largeorder.py),
    in the coupling ubar = (N+8) u/6 of the paper.  alpha sets the growth x^(alpha/2) of the approximant of the Borel
    transform at large x (alpha = 2 lambda of Kompaniets & Panzer 2017); a shift F -> F/u^s is the same as
    (b, alpha) -> (b - s, alpha + 2s), so the scan over alpha contains the shifted series of Guida & Zinn-Justin.

    Resummed functions: beta (its zero ubar* and its slope omega = beta'(ubar*)), eta at ubar*, and 1/nu =
    2 - eta + eta_phi2 at ubar*, resummed as one series: the Callan-Symanzik relations give 1/nu directly, as a product
    of series, while nu needs the inversion of a series, which turns any zero of 1/nu in the complex coupling plane into
    a pole; the series of nu itself gives a second determination (Guida & Zinn-Justin summed S and 1/S).
    Scan: b = 0, 0.5, .., 12 and alpha = -2, -1.5, .., 10 (beyond alpha ~ 10 the truncations oscillate with growing
    amplitude).  For every (b, alpha), F_K' is the result truncated at the order K'; over the neighbourhood
    |b' - b| <= 1, |alpha' - alpha| <= 1 of the point,
        delta = the largest of max(|F_K - F_(K-1)|, |F_(K-1) - F_(K-2)|)      (stability against the truncation)
        sigma = half the range of F_K                                        (stability against the parameters)
    (taking delta over the neighbourhood discards the isolated points where successive truncations cross by
    accident).  The central (b*, alpha*) minimises delta + sigma (points whose neighbourhood leaves the grid are not
    eligible); the region R is the set of points with delta + sigma at most twice the minimum.  Resummation error at
    fixed ubar*: the largest of delta*, sigma* and max_R |F_K - F_K*| (for nu also the difference from the series of
    nu); to it the error of ubar* is added linearly, |F(ubar* +- d ubar*) - F(ubar*)| at (b*, alpha*), as in Guida &
    Zinn-Justin.
    Statistical error: the per-order sums are linear in the class integrals, drawn independently from Gaussians, so
    each sum at fixed N is drawn from a Gaussian with its propagated error (refs.py and Phi4tools inputs with their
    quoted errors); every draw rebuilds the coefficients (coefficients.rg) and is resummed with the central (b, alpha);
    the standard deviation over the draws (fixed seed) is the statistical error.  At N = 1 the report compares it with
    a linearised propagation and, for the ledger inputs of the eighth-order series, with resampling the classes.

    Rows: (a) the eighth-order series of Phi4tools (beta, eta, nu to ubar^8, 1/nu from nu); (b) our coefficients to
    ubar^8; (c) to ubar^9; (d) eta and nu to ubar^10, beta to ubar^11; and the variant d10 of (d) with beta to ubar^10.
    The report (stdout) gives every component of the errors, the central parameters and the last truncations.

    usage: resum.py [--json FILE] [--ledger DIR] [--N 0,1,2,3] [--draws D] [--seed S] [--tex FILE]
    --json: the coefficients (default generated/coefficients.json, written by coefficients.py); --ledger recomputes
    them from a ledger instead (slow); --draws 0 skips the statistical errors; --tex writes the body of Table
    tab:exponents and the macros of the text (paper/generated/exponents.tex).
    Deterministic: the same coefficients and the same seed give the same output, byte for byte.

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
import json
import math
import os
import random
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coefficients as CO  # noqa: E402
import rebar_ledger as RL  # noqa: E402

AA = 113.383507815          # instanton action: 48 pi/(9 A) = a(d = 3) of Zinn-Justin's book, 3rd ed., Eq. (28.73)
B_GRID = [0.5 * i for i in range(25)]                 # 0 .. 12
ALPHA_GRID = [-2 + 0.5 * j for j in range(25)]        # -2 .. 10
BOX = 2                     # neighbourhood for sigma: +-2 grid steps = +-1 in b and alpha
REGION = 2.0                # R: delta + sigma <= REGION * minimum
DRAWS, SEED = 200, 20261018
FUNCS = ("ustar", "nu", "eta", "omega")
JSON_DEFAULT = os.path.join(HERE, "generated", "coefficients.json")

def quoted(text):
    """'0.629971(4)' -> (0.629971, 4e-6, '0.629971(4)'): a value as its source prints it"""
    v, e = text.rstrip(")").split("(")
    d = len(v.split(".")[1]) if "." in v else 0
    return float(v), int(e) * 10.0 ** -d, text


def derived(v, e, text=None):
    """a value obtained from quoted ones (scaling relations; errors in quadrature), shown with two digits"""
    return v, e, text or RL.fmt_ve(v, e, 2)


# Benchmarks, each checked in its primary source (arXiv versions as of October 2026; see the comments): (value,
# error, text).  "derived" values follow from the quoted ones by the relations in the comments.
BENCHMARKS = {
    # Guida & Zinn-Justin, J. Phys. A 31 (1998) 8103, arXiv:cond-mat/9803240v3, Table 1 (d = 3 expansion; their
    # g*_Ni is ubar*).  v1 and v2 differ only by \draft; against v3 they had, in Table 1, ubar* = 1.391(4) at N = 3
    # and the errors of nu 0.0013 (N = 2) and 0.0030 (N = 3); every other entry used here is the same in all three.
    # All versions print omega = 0.782 +- 0.0013 at N = 3, a misprint for 0.013 (the error 0.012 of theta = omega nu
    # requires it); we quote 0.782(13).
    "GZJ98": {N: {f: quoted(t) for f, t in zip(("ustar", "nu", "eta", "omega"), row)} for N, row in (
        (0, ("1.413(6)", "0.5882(11)", "0.0284(25)", "0.812(16)")),
        (1, ("1.411(4)", "0.6304(13)", "0.0335(25)", "0.799(11)")),
        (2, ("1.403(3)", "0.6703(15)", "0.0354(25)", "0.789(11)")),
        (3, ("1.390(4)", "0.7073(35)", "0.0355(25)", "0.782(13)")))},
    # N = 0, self-avoiding walks: nu = 0.58759700(40) and Delta_1 = 0.528(8), Clisby & Duenweg, PRE 94 (2016) 052102,
    # arXiv:2001.03138 (abstract; Delta_1 in Sec. V and the table of results); gamma = 1.15695300(95), Clisby,
    # J. Phys. A 50 (2017) 264003, arXiv:1701.08415 (abstract).  Derived: eta = 2 - gamma/nu, omega = Delta_1/nu.
    # N = 2: eta = 0.03810(8), nu = 0.67169(7), omega = 0.789(4), Hasenbusch, PRB 100 (2019) 224517, arXiv:1910.05916
    # (abstract).  N = 3: nu = 0.71164(10), eta = 0.03784(5), omega = 0.759(2), Hasenbusch, PRB 102 (2020) 024406,
    # arXiv:2005.04448 (abstract).
    "MC": {0: {"nu": quoted("0.58759700(40)"),
               "eta": derived(2 - 1.15695300 / 0.58759700,
                              math.hypot(0.00000095 / 0.58759700, 1.15695300 * 0.00000040 / 0.58759700 ** 2)),
               "omega": derived(0.528 / 0.58759700, 0.008 / 0.58759700)},
           2: {"nu": quoted("0.67169(7)"), "eta": quoted("0.03810(8)"), "omega": quoted("0.789(4)")},
           3: {"nu": quoted("0.71164(10)"), "eta": quoted("0.03784(5)"), "omega": quoted("0.759(2)")}},
    # N = 1, conformal bootstrap: Delta_sigma = 0.518148806(24), Delta_eps = 1.41262528(29), Chang et al., JHEP 03
    # (2025) 136, arXiv:2411.15300, Table 1, so that nu = 1/(3 - Delta_eps) = 0.62997097(12) and eta = 2 Delta_sigma - 1
    # = 0.036297612(48) (earlier: Kos, Poland, Simmons-Duffin & Vichi, JHEP 08 (2016) 036, 0.629971(4), 0.0362978(20));
    # omega = Delta_eps' - 3, Delta_eps' = 3.82968(23), Simmons-Duffin, JHEP 03 (2017) 086, arXiv:1612.08471, Table 2
    # (non-rigorous error; the rigorous bound of Reehorst, JHEP 09 (2022) 177, is 3.82951(61)).
    # N = 2: Delta_phi = 0.519088(22), Delta_s = 1.51136(22), Chester et al., JHEP 06 (2020) 142, arXiv:1912.03324,
    # Table 1 (rigorous); Delta_s' = 3.794(8), ibid. Table 4 (extremal functional, non-rigorous).
    # N = 3: Delta_phi = 0.518936(67), Delta_s = 1.59488(81), Chester et al., PRD 104 (2021) 105013, arXiv:2011.14647,
    # Table 1 (rigorous); no omega.  Derived: eta = 2 Delta_phi - 1, nu = 1/(3 - Delta_s), omega = Delta_s' - 3.
    "CB": {1: {"nu": quoted("0.62997097(12)"), "eta": quoted("0.036297612(48)"), "omega": quoted("0.82968(23)")},
           2: {"nu": derived(1 / (3 - 1.51136), 0.00022 / (3 - 1.51136) ** 2, "0.67175(10)"),
               "eta": derived(2 * 0.519088 - 1, 2 * 0.000022), "omega": quoted("0.794(8)")},
           3: {"nu": derived(1 / (3 - 1.59488), 0.00081 / (3 - 1.59488) ** 2),
               "eta": derived(2 * 0.518936 - 1, 2 * 0.000067)}},
}
BENCH_CITE = {("GZJ98", N): "GuidaZinnJustin1998" for N in range(4)}
BENCH_CITE.update({("MC", 0): "ClisbyDunweg2016,Clisby2017", ("CB", 1): "ChangEtAl2025,SimmonsDuffin2017,Reehorst2022",
                   ("MC", 2): "Hasenbusch2019", ("CB", 2): "ChesterEtAl2020",
                   ("MC", 3): "Hasenbusch2020", ("CB", 3): "ChesterEtAl2021"})
PRIMARY = {0: "MC", 1: "CB", 2: "MC", 3: "MC"}     # the benchmark the text compares with


# ---------------------------------------------------------------------------------------- power series
def series_mul(p, q, n):
    return [sum(p[j] * q[k - j] for j in range(k + 1) if j < len(p) and k - j < len(q)) for k in range(n + 1)]


def series_pow(g, p, n):
    """g^p for a series g with g[0] = 1, to order n (J. C. P. Miller's recurrence)"""
    out = [1.0]
    for k in range(1, n + 1):
        out.append(sum((p * m - k + m) * g[m] * out[k - m] for m in range(1, k + 1)) / k)
    return out


def series_inv(g):
    out = [1 / g[0]]
    for n in range(1, len(g)):
        out.append(-sum(g[j] * out[n - j] for j in range(1, n + 1)) / g[0])
    return out


def w_series(a, n):
    """w(x) = (sqrt(1 + a x) - 1)/(sqrt(1 + a x) + 1) to order n"""
    s = [1.0]
    for k in range(1, n + 1):
        s.append(s[-1] * (0.5 - (k - 1)) / k * a)
    num, den = [0.0] + s[1:], [2.0] + s[1:]
    out = []
    for k in range(n + 1):
        out.append((num[k] - sum(den[j] * out[k - j] for j in range(1, k + 1))) / den[0])
    return out


# ---------------------------------------------------------------------------------------- the resummation
class Quadrature:
    """int_0^inf t^b e^-t h(t) dt = int_0^S 2 s^(2b+1) e^(-s^2) h(s^2) ds, 8-point Gauss-Legendre on 16 panels
    (relative accuracy better than 1e-13 on the whole grid)"""
    GL = [(-0.9602898564975363, 0.1012285362903763), (-0.7966664774136267, 0.2223810344533745),
          (-0.5255324099163290, 0.3137066458778873), (-0.1834346424956498, 0.3626837833783620),
          (0.1834346424956498, 0.3626837833783620), (0.5255324099163290, 0.3137066458778873),
          (0.7966664774136267, 0.2223810344533745), (0.9602898564975363, 0.1012285362903763)]

    def __init__(self, panels=16, smax=9.5):
        self.t, self.ws = [], []
        h = smax / panels
        for p in range(panels):
            for x, wgt in self.GL:
                s = h * (p + 0.5 * (1 + x))
                self.t.append(s * s); self.ws.append((s, wgt * h / 2))
        self._w = {}

    def weights(self, b):
        if b not in self._w:
            self._w[b] = [2 * wgt * s ** (2 * b + 1) * math.exp(-s * s) for s, wgt in self.ws]
        return self._w[b]


QUAD = Quadrature()
_PMAT = {}


def pmatrix(a, alpha, n):
    """P[k][m]: coefficient of x^m in w(x)^k (1 - w(x))^-alpha"""
    key = (a, alpha, n)
    if key not in _PMAT:
        w = w_series(a, n)
        pref = series_pow([1.0] + [-x for x in w[1:]], -alpha, n)
        P, wk = [], [1.0] + [0.0] * n
        for _ in range(n + 1):
            P.append(series_mul(wk, pref, n)); wk = series_mul(wk, w, n)
        _PMAT[key] = P
    return _PMAT[key]


class Resummer:
    """Borel-Leroy + conformal mapping of one series F_0 .. F_K with parameters (a, b, alpha); the d_k of a lower
    truncation are the first ones of a higher one, so every call returns all the truncations at once"""

    def __init__(self, F, a, b, alpha):
        self.K = len(F) - 1; self.a, self.b, self.alpha = a, b, alpha
        P = pmatrix(a, alpha, self.K)
        B = [F[k] / math.gamma(k + b + 1) for k in range(self.K + 1)]
        d = []
        for k in range(self.K + 1):
            d.append((B[k] - sum(d[j] * P[j][k] for j in range(k))) / P[k][k])
        self.d = d
        self.wq = QUAD.weights(b)

    def truncations(self, u, kmax=None):
        """[F resummed and truncated at 0, 1, .., kmax] at coupling u"""
        n = self.K if kmax is None else kmax
        if not 0.2 < u < 5:
            raise ValueError("coupling out of range")
        a, alpha = self.a, self.alpha
        M = [0.0] * (n + 1)
        for t, wq in zip(QUAD.t, self.wq):
            r = math.sqrt(1 + a * u * t)
            w = (r - 1) / (r + 1)
            c = wq * (1 - w) ** (-alpha)
            for k in range(n + 1):
                M[k] += c; c *= w
        out, acc = [], 0.0
        for dk, mk in zip(self.d, M):
            acc += dk * mk; out.append(acc)
        return out

    def slopes(self, u, h=1e-4):
        p, m = self.truncations(u + h), self.truncations(u - h)
        return [(x - y) / (2 * h) for x, y in zip(p, m)]

    def zero(self, k, x0=1.38, x1=1.44, tol=1e-12):
        """zero of the truncation at order k (secant)"""
        f0, f1 = self.truncations(x0, k)[k], self.truncations(x1, k)[k]
        for _ in range(60):
            if f1 == f0:
                break
            x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0), f1
            f1 = self.truncations(x1, k)[k]
            if abs(x1 - x0) < tol:
                return x1
        raise ValueError("no zero")


def coupling_a(N):
    return 48 * math.pi / ((N + 8) * AA)


# ---------------------------------------------------------------------------------------- scan and selection
def scan(F, a, kind, u=None, last=3, alphas=None):
    """{(i, j): [F_(K-last+1), .., F_K]} on the grid (b = B_GRID[i], alpha = alphas[j], by default ALPHA_GRID);
    kind: 'zero', 'value', 'slope' (value and slope at the coupling u)"""
    K = len(F) - 1
    out = {}
    for j, alpha in enumerate(alphas or ALPHA_GRID):
        for i, b in enumerate(B_GRID):
            try:
                R = Resummer(F, a, b, alpha)
                if kind == "zero":
                    v = [R.zero(k) for k in range(K - last + 1, K + 1)]
                elif kind == "slope":
                    v = R.slopes(u)[K - last + 1:]
                else:
                    v = R.truncations(u)[K - last + 1:]
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            if all(math.isfinite(x) for x in v):
                out[(i, j)] = v
    return out


def select(grid, alphas=None):
    """central point, its value and the resummation error at fixed coupling, with the components"""
    alphas = alphas or ALPHA_GRID
    def delta(p):
        v = grid[p]
        return max(abs(v[-1] - v[-2]), abs(v[-2] - v[-3]))

    def box(p):
        i, j = p
        return [(i + di, j + dj) for di in range(-BOX, BOX + 1) for dj in range(-BOX, BOX + 1)]
    dmax, sigma, score = {}, {}, {}
    for p in sorted(grid):
        bx = box(p)
        if any(q not in grid for q in bx):
            continue
        vals = [grid[q][-1] for q in bx]
        dmax[p], sigma[p] = max(delta(q) for q in bx), 0.5 * (max(vals) - min(vals))
        score[p] = dmax[p] + sigma[p]
    best = min(sorted(score), key=lambda p: score[p])
    region = [p for p in sorted(score) if score[p] <= REGION * score[best]]
    spread = max(abs(grid[p][-1] - grid[best][-1]) for p in region)
    err = max(dmax[best], sigma[best], spread)
    return {"best": best, "b": B_GRID[best[0]], "alpha": alphas[best[1]], "value": grid[best][-1], "err": err,
            "delta": dmax[best], "sigma": sigma[best], "spread": spread, "region": len(region), "trunc": grid[best]}


# ---------------------------------------------------------------------------------------- the series
def load(path=JSON_DEFAULT, ledger=None):
    if ledger:
        return json.loads(RL.dump_json(CO.compute(ledger)))
    with open(path) as fh:
        return json.load(fh)


def inputs(res, N):
    """{(kind, L): (value, error)} of the per-order sums at N, from the sources chosen by coefficients.py"""
    refN = RL.references(N)
    out = {}
    for o, s in sorted(res["sources"].items()):
        key = (o[0], int(o[1:]))
        out[key] = tuple(res["sums"][o]["N"][str(N)]) if s == "ledger" else (refN[key][0], refN[key][1])
    return out


def ours(vals, N):
    """beta, eta, nu, 1/nu (lists of coefficients of ubar^0 ..) from the per-order sums"""
    c = CO.rg(vals, N)
    return {"beta": c["beta"], "eta": c["eta"], "nu": c["nu"], "inu": series_inv(c["nu"])}


def phi4tools(N, rng=None):
    p = CO.phi4tools(N)
    s = {f: [v + (e * rng.gauss(0, 1) if rng and e else 0.0) for v, e in p[f]] for f in ("beta", "eta", "nu")}
    s["inu"] = series_inv(s["nu"])
    return s


ROWS = (("a", "8th order, Phi4tools", {"beta": 8, "eta": 8, "nu": 8}),
        ("b", "8th order, this work", {"beta": 8, "eta": 8, "nu": 8}),
        ("c", "9th order, this work", {"beta": 9, "eta": 9, "nu": 9}),
        ("d", "10th order, this work", {"beta": 11, "eta": 10, "nu": 10}))


def cut(s, K):
    return {"beta": s["beta"][:K["beta"] + 1], "eta": s["eta"][:K["eta"] + 1], "nu": s["nu"][:K["nu"] + 1],
            "inu": s["inu"][:K["nu"] + 1]}


def analyse(s, N):
    """the full analysis of one set of truncated series"""
    a = coupling_a(N)
    out = {}
    u = select(scan(s["beta"], a, "zero"))
    du = u["err"]
    out["ustar"] = dict(u, total=du, du=0.0)
    ustar = u["value"]

    def at(f, kind):
        r = select(scan(s[f], a, kind, ustar))
        R = Resummer(s[f], a, r["b"], r["alpha"])
        g = R.slopes if kind == "slope" else R.truncations
        r["du"] = max(abs(g(ustar + du)[-1] - r["value"]), abs(g(ustar - du)[-1] - r["value"]))
        r["total"] = r["err"] + r["du"]
        return r
    out["omega"] = at("beta", "slope")
    out["eta"] = at("eta", "value")
    out["nu_direct"] = at("nu", "value")
    inu = at("inu", "value")
    sc = inu["value"] ** -2           # nu = 1/(1/nu): the errors scale with 1/(1/nu)^2
    nu = dict(inu, value=1 / inu["value"], trunc=[1 / x for x in inu["trunc"]],
              **{k: inu[k] * sc for k in ("err", "delta", "sigma", "spread", "du", "total")})
    d = abs(nu["value"] - out["nu_direct"]["value"])
    nu["inverse"] = d
    if d > nu["err"]:
        nu["total"] += d - nu["err"]; nu["err"] = d
    out["nu"] = nu
    return out


def evaluate(s, N, o):
    """exponents of the series s with the central parameters of the analysis o"""
    a = coupling_a(N)
    p = o["ustar"]
    u = Resummer(s["beta"], a, p["b"], p["alpha"]).zero(len(s["beta"]) - 1)
    r = {"ustar": u}
    p = o["omega"]; r["omega"] = Resummer(s["beta"], a, p["b"], p["alpha"]).slopes(u)[-1]
    p = o["eta"]; r["eta"] = Resummer(s["eta"], a, p["b"], p["alpha"]).truncations(u)[-1]
    p = o["nu"]; r["nu"] = 1 / Resummer(s["inu"], a, p["b"], p["alpha"]).truncations(u)[-1]
    return r


def statistics(draws, N, o):
    """standard deviations of the exponents over the draws (lists of series)"""
    vals = [evaluate(s, N, o) for s in draws]
    out = {}
    for f in ("ustar", "omega", "eta", "nu"):
        x = [v[f] for v in vals]
        m = sum(x) / len(x)
        out[f] = math.sqrt(sum((y - m) ** 2 for y in x) / (len(x) - 1))
    return out


def linearised(inp, N, K, o):
    """first-order propagation of the input errors through the resummation, at the central parameters"""
    vals = {k: v for k, (v, e) in inp.items()}
    c0 = evaluate(cut(ours(vals, N), K), N, o)
    var = {f: 0.0 for f in c0}
    for key, (v, e) in sorted(inp.items()):
        if not e:
            continue
        p = dict(vals); p[key] = v + e
        c = evaluate(cut(ours(p, N), K), N, o)
        for f in c0:
            var[f] += (c[f] - c0[f]) ** 2
    return {f: math.sqrt(x) for f, x in var.items()}


def class_check(res, o, ndraw, seed, N=1, K=None, ledger=RL.LEDGER):
    """Resampling the class integrals against resampling the per-order sums: the ledger inputs of the eighth-order
    series (row b) at N, each way with ndraw draws; {order: s.d. of the sum / propagated error}, {quantity: s.d. from
    the classes / s.d. from the sums}"""
    K = K or ROWS[1][2]
    inp = inputs(res, N)
    base = {k: v for k, (v, e) in inp.items()}
    orders = [o_ for o_, s in sorted(res["sources"].items()) if s == "ledger" and int(o_[1:]) <= (7 if o_[0] == "b" else 8)]
    cls = {o_: [(RL.poly(r["weight_poly"], N), r["value"], r["error"]) for r in
                RL.iter_classes(ledger, o_, fields=("weight_poly", "value", "error"))] for o_ in orders}
    rng_c, rng_s = random.Random(seed + 7), random.Random(seed + 8)
    ev_c, ev_s, sums = [], [], {o_: [] for o_ in orders}
    for _ in range(ndraw):
        vc, vs = dict(base), dict(base)
        for o_ in orders:
            key = (o_[0], int(o_[1:]))
            x = sum(w * (v + e * rng_c.gauss(0, 1)) for w, v, e in cls[o_])
            vc[key] = x; sums[o_].append(x)
            vs[key] = inp[key][0] + inp[key][1] * rng_s.gauss(0, 1)
        ev_c.append(evaluate(cut(ours(vc, N), K), N, o)); ev_s.append(evaluate(cut(ours(vs, N), K), N, o))

    def sd(x):
        m = sum(x) / len(x)
        return math.sqrt(sum((y - m) ** 2 for y in x) / (len(x) - 1))
    return ({o_: sd(sums[o_]) / inp[(o_[0], int(o_[1:]))][1] for o_ in orders},
            {f: sd([e[f] for e in ev_c]) / sd([e[f] for e in ev_s]) for f in FUNCS})


# ---------------------------------------------------------------------------------------- the analysis of all rows
def compute(res, Ns, ndraw, seed, log=print, ledger=None):
    """{(N, row): analysis} for the rows of ROWS and the variant d10 (beta to ubar^10), with statistical errors;
    'lin': the linearised statistical errors of row (d) at N = 1; 'class': class_check at N = 1"""
    table = {}
    for N in Ns:
        inp = inputs(res, N)
        base = ours({k: v for k, (v, e) in inp.items()}, N)
        rng = random.Random(seed + 1000 * N)
        draws_ours = [ours({k: v + (e * rng.gauss(0, 1) if e else 0.0) for k, (v, e) in inp.items()}, N)
                      for _ in range(ndraw)]
        rng = random.Random(seed + 1000 * N + 500)
        draws_p4t = [phi4tools(N, rng) for _ in range(ndraw)]
        for tag, label, K in ROWS + (("d10", "beta to ubar^10", {"beta": 10, "eta": 10, "nu": 10}),):
            s = cut(phi4tools(N) if tag == "a" else base, K)
            o = analyse(s, N)
            st = statistics([cut(x, K) for x in (draws_p4t if tag == "a" else draws_ours)], N, o) if ndraw > 1 \
                else {f: 0.0 for f in FUNCS}
            for f in FUNCS:
                o[f]["stat"] = st[f]
            o["K"] = K
            if tag == "d":
                wide = [-2 + 0.5 * j for j in range(29)]
                o["eta_wide"] = select(scan(s["eta"], coupling_a(N), "value", o["ustar"]["value"], alphas=wide), wide)
            table[(N, tag)] = o
            log("N = %d (%s) done" % (N, tag))
        if N == 1 and ndraw > 1:
            table["lin"] = linearised(inp, N, ROWS[3][2], table[(N, "d")])
            table["class"] = class_check(res, table[(N, "b")], ndraw, seed, N, ledger=ledger or RL.LEDGER)
            log("N = %d class-level check done" % N)
    return table


def report(table, Ns, ndraw, seed):
    L = ["Borel-Leroy + conformal mapping; grid b = %g..%g, alpha = %g..%g (step 0.5); %d draws, seed %d"
         % (B_GRID[0], B_GRID[-1], ALPHA_GRID[0], ALPHA_GRID[-1], ndraw, seed),
         "total resummation error = max(delta, sigma, spread over R) + ubar* term; stat = s.d. over the draws"]
    for N in Ns:
        L.append("\nN = %d   a = %.9f" % (N, coupling_a(N)))
        for tag, label, K in ROWS + (("d10", "beta to ubar^10", None),):
            o = table[(N, tag)]
            L.append("  (%s) %s: beta to ubar^%d, eta and nu to ubar^%d" % (tag, label, o["K"]["beta"], o["K"]["eta"]))
            for f in FUNCS:
                r = o[f]
                L.append("    %-6s %-13s res %.1e (delta %.1e sigma %.1e spread %.1e |R| %3d; ubar* %.1e)  stat %.1e"
                         "  b=%g alpha=%g; last truncations %s%s"
                         % (f, fmt(r["value"], r["total"]), r["total"], r["delta"], r["sigma"], r["spread"],
                            r["region"], r["du"], r["stat"], r["b"], r["alpha"],
                            " ".join("%.6f" % x for x in r["trunc"]),
                            "; |nu - nu direct| %.1e" % r["inverse"] if f == "nu" else ""))
            nd = o["nu_direct"]
            L.append("    nu from the series of nu: %s  b=%g alpha=%g" % (fmt(nd["value"], nd["total"]), nd["b"], nd["alpha"]))
            if "eta_wide" in o:
                ew = o["eta_wide"]
                L.append("    eta with alpha up to 12: %.6f (b=%g alpha=%g)" % (ew["value"], ew["b"], ew["alpha"]))
        if N == 1 and "lin" in table:
            L.append("  linearised statistical errors, row (d): " + "  ".join("%s %.2e (draws %.2e)" % (
                f, table["lin"][f], table[(N, "d")][f]["stat"]) for f in FUNCS))
            cs, cf = table["class"]
            L.append("  class-level resampling, row (b): s.d. of the sums / propagated error " + "  ".join(
                "%s %.3f" % kv for kv in sorted(cs.items())))
            L.append("    s.d. from the classes / s.d. from the sums " + "  ".join("%s %.3f" % (f, cf[f]) for f in FUNCS))
        for src in ("GZJ98", "MC", "CB"):
            if N in BENCHMARKS[src]:
                L.append("  %-6s " % src + "  ".join("%s %s" % (f, BENCHMARKS[src][N][f][2]) for f in FUNCS
                                                     if f in BENCHMARKS[src][N]))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------- the paper's fragment
def fmt(v, e, nd=2):
    return RL.fmt_ve(v, e, nd)


NUMS = ("zero", "one", "two", "three", "four")


def sci(x, digits=1):
    """2.3e-4 -> '2\\times10^{-4}' (math mode, no $)"""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10 ** e, digits - 1)
    if abs(m) >= 10:
        m /= 10; e += 1
    return ("%.*f" % (digits - 1, m)) + "\\times10^{%d}" % e


def macro(name, value):
    return "\\newcommand*{\\%s}{%s}" % (name, value)


def signed(x):
    t = "%+.1f" % x
    return "0.0" if t in ("+0.0", "-0.0") else t


def ratio(x):
    """0.104 -> '0.1', 8.3e-4 -> '8\\times10^{-4}' (math mode)"""
    return "%.1g" % x if x >= 0.01 else sci(x)


def tex_text(table, Ns, ndraw, seed):
    """paper/generated/exponents.tex: \\exprows (the body of Table tab:exponents) and the numbers of the text"""
    F = {"ustar": "Ustar", "nu": "Nu", "eta": "Eta", "omega": "Omega"}
    RW = {"a": "A", "b": "B", "c": "C", "d": "D", "d10": "Dten"}
    L = ["% generated/exponents.tex -- body of Table tab:exponents and the resummed numbers of the text, N = 0..3.",
         "%% WRITTEN BY series/resum.py --tex (%d draws, seed %d) from series/generated/coefficients.json:" % (ndraw, seed),
         "% do not edit by hand.  Every line ends with % so that the file can be read anywhere.",
         "%",
         "% Macro contract.  \\exprows: the rows of the table, per N in chronological order of the series (Guida-Zinn-Justin",
         "% 1998, 8th order of Phi4tools and ours, 9th, 10th), then the non-perturbative benchmarks.  Per N (suffix Nzero .. Nthree), row (A: Phi4tools at",
         "% 8th order; B, C, D: this work at 8th, 9th, 10th order, D with beta to ubar^11; Dten: D with beta to ubar^10)",
         "% and quantity (Ustar, Nu, Eta, Omega): \\rs<Quantity><Row>N<n>, value(total resummation error), e.g.",
         "% \\rsNuDNone (nu from the series of 1/nu); \\rsNuDirDN<n>: nu of row D from the series of nu.",
         "% \\rsPull<Quantity>N<n>: (row D - benchmark) / combined error, the benchmark being the Monte Carlo one for",
         "% N = 0, 2, 3 and the bootstrap for N = 1 (signed).  Statistics: \\rsStatMax<Quantity> (largest statistical",
         "% error, rows B-D, all N),",
         "% \\rsStatRatio (largest statistical/resummation error ratio, rows B-D), \\rsStatRatioA (row A),",
         "% \\rsLinDev (largest relative difference between linearised and resampled errors, N = 1, row D),",
         "% \\rsClassDev (largest relative difference between resampling the classes and the sums, N = 1, row B).",
         "% Benchmarks as in the table: \\rsBench<Source><Quantity>N<n>, Source GZJ (Guida-Zinn-Justin 1998), MC, CB.",
         "% \\rsEtaWindow (largest change of eta, row D, when the alpha window is extended to 12), \\rsNuDirMax",
         "% (largest |nu from 1/nu - nu from nu|, rows B-D), \\rsDraws.  Method: \\rsBmax, \\rsAlphaMin, \\rsAlphaMax,",
         "% \\rsAone (a at N = 1), \\rsInstanton (A).",
         "% \\rsABdiff, \\rsABmax: largest |row A - row B|, absolute and in units of the resummation error of row B (rows A",
         "% and B are printed as one when their digits are the same).  \\rsGain<Quantity>: range over N",
         "% of the ratio of the resummation errors of rows B and D.  Scientific values in math mode.",
         ]
    def bench_row(src, N):
        d, cite = BENCHMARKS[src][N], BENCH_CITE[(src, N)]
        lab = {"GZJ98": "7th order", "MC": "self-avoiding walks" if N == 0 else "Monte Carlo", "CB": "conformal bootstrap"}[src]
        lab += ", Ref%s.~\\cite{%s}" % ("s" if "," in cite else "", cite)
        return lab, [d[f][2] if f in d else "---" for f in FUNCS]

    def ours_row(N, tag):
        o = table[(N, tag)]
        return [fmt(o[f]["value"], o[f]["total"]) for f in FUNCS]
    rows = []
    for N in Ns:                  # chronological: GZJ98, the 8th order (Phi4tools, ours), 9th, 10th; then the benchmarks
        if rows:
            rows.append("\\hline")
        lines = [bench_row("GZJ98", N)]
        a, b = ours_row(N, "a"), ours_row(N, "b")
        if a == b:                # the same digits: one row (the text quotes the largest difference)
            lines.append(("8th order, Ref.~\\cite{Phi4tools2024} and this work", a))
        else:
            lines += [("8th order, Ref.~\\cite{Phi4tools2024}", a), ("8th order, this work", b)]
        lines += [(label, ours_row(N, tag)) for tag, label, K in ROWS[2:]]
        lines += [bench_row(src, N) for src in ("MC", "CB") if N in BENCHMARKS[src]]
        for k, (lab, cells) in enumerate(lines):
            rows.append("%s & %s & %s \\\\" % (N if k == 0 else "", lab, " & ".join(cells)))
    L.append("\\newcommand*{\\exprows}{%")
    L += [r + "%" for r in rows]
    L.append("}%")
    for N in Ns:
        n = NUMS[N]
        for tag in ("a", "b", "c", "d", "d10"):
            o = table[(N, tag)]
            L.append("".join(macro("rs%s%sN%s" % (F[f], RW[tag], n), fmt(o[f]["value"], o[f]["total"])) for f in FUNCS))
        o = table[(N, "d")]
        L.append(macro("rsNuDirDN%s" % n, fmt(o["nu_direct"]["value"], o["nu_direct"]["total"])))
        bench = BENCHMARKS[PRIMARY[N]][N]
        L.append("".join(macro("rsPull%sN%s" % (F[f], n), signed((o[f]["value"] - bench[f][0]) / math.hypot(
            o[f]["total"], bench[f][1]))) for f in ("nu", "eta", "omega")))
    for src in ("GZJ98", "MC", "CB"):
        for N in Ns:
            if N in BENCHMARKS[src]:
                L.append("".join(macro("rsBench%s%sN%s" % ("GZJ" if src == "GZJ98" else src, F[f], NUMS[N]), x[2])
                                 for f, x in sorted(BENCHMARKS[src][N].items())))
    main_rows = [(N, t) for N in Ns for t in ("b", "c", "d")]
    for f in FUNCS:
        L.append(macro("rsStatMax%s" % F[f], sci(max(table[k][f]["stat"] for k in main_rows))))
    L.append(macro("rsStatRatio", ratio(max(table[k][f]["stat"] / table[k][f]["total"] for k in main_rows for f in FUNCS))))
    L.append(macro("rsStatRatioA", ratio(max(table[(N, "a")][f]["stat"] / table[(N, "a")][f]["total"]
                                           for N in Ns for f in FUNCS))))
    if "lin" in table:
        L.append(macro("rsLinDev", "%.0f\\%%" % (100 * max(abs(table["lin"][f] / table[(1, "d")][f]["stat"] - 1)
                                                            for f in FUNCS))))
    if "class" in table:
        cs, cf = table["class"]
        L.append(macro("rsClassDev", "%.0f\\%%" % (100 * max(abs(x - 1) for x in list(cs.values()) + list(cf.values())))))
    L.append(macro("rsABdiff", sci(max(abs(table[(N, "a")][f]["value"] - table[(N, "b")][f]["value"])
                                          for N in Ns for f in FUNCS))))
    L.append(macro("rsABmax", sci(max(abs(table[(N, "a")][f]["value"] - table[(N, "b")][f]["value"])
                                           / table[(N, "b")][f]["total"] for N in Ns for f in FUNCS))))
    for f in FUNCS:
        g = [table[(N, "b")][f]["total"] / table[(N, "d")][f]["total"] for N in Ns]
        lo, hi = "%.1f" % min(g), "%.1f" % max(g)
        L.append(macro("rsGain%s" % F[f], lo if lo == hi else lo + "--" + hi))
    L.append(macro("rsEtaWindow", sci(max(abs(table[(N, "d")]["eta_wide"]["value"] - table[(N, "d")]["eta"]["value"])
                                          for N in Ns))))
    L.append(macro("rsNuDirMax", sci(max(table[k]["nu"]["inverse"] for k in main_rows))))
    L.append(macro("rsDraws", "%d" % ndraw))
    L.append(macro("rsBmax", "%g" % B_GRID[-1]) + macro("rsAlphaMin", "%g" % ALPHA_GRID[0]) +
             macro("rsAlphaMax", "%g" % ALPHA_GRID[-1]))
    L.append(macro("rsAone", "%.7f" % coupling_a(1)) + macro("rsInstanton", "%.4f" % AA))
    return "\n".join(x if x.startswith("%") or x.endswith("%") else x + "%" for x in L) + "\n"


def main(argv):
    path, ledger, Ns, ndraw, seed, tex = JSON_DEFAULT, None, [0, 1, 2, 3], DRAWS, SEED, None
    it = iter(argv)
    for x in it:
        if x == "--json": path = next(it)
        elif x == "--ledger": ledger = next(it)
        elif x == "--N": Ns = [int(t) for t in next(it).split(",")]
        elif x == "--draws": ndraw = int(next(it))
        elif x == "--seed": seed = int(next(it))
        elif x == "--tex": tex = next(it)
        else: raise SystemExit(__doc__)
    table = compute(load(path, ledger), Ns, ndraw, seed, log=lambda m: print(m, file=sys.stderr, flush=True),
                    ledger=ledger)
    sys.stdout.write(report(table, Ns, ndraw, seed))
    if tex:
        RL.write_if_changed(tex, tex_text(table, Ns, ndraw, seed))
        print("wrote", os.path.relpath(tex, RL.ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
