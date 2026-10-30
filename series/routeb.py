#!/usr/bin/env python3
"""
    routeb.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    Route B: the RG functions from the bare (intermediate-scheme) vertex functions by renormalization and series
    reversion, as in Nickel et al. and Phi4tools, independently of route A (the Callan-Symanzik sums B_L, H_L, C_L
    of the ledger).  The inputs are three per-order sums, k = number of quartic vertices:
        P4_k  = sum_G S_G(N) V_G over the four-point graphs of Gamma4(0)            (L = k - 1 loops)
        P2_k  = sum_G S_G(N) V_G over the two-point graphs of Gamma2(0)             (L = k loops)
        P2d_k = sum_G S_G(N) V'_G over the two-point graphs of d Gamma2 / d p^2 (0)  (L = k loops)
    in Phi4tools' normalization (tadpoles zero, sunsets subtracted at zero momentum).  They come from Phi4tools'
    per-graph values (results/checks/phi4tools/routeb-sums.json, k <= 8) and, for any order, from runs of the
    integrator with the kinds int4, int2 and int2d, whose order sums X convert as
        P4_k = 2^(k-1)/3 X^int4_(k-1),   P2_k = 2^k X^int2_k,   P2d_k = -2^k X^int2d_k.
    The recipe (from thesis eqs. 2.12-2.14 and Phi4tools 3.1-3.7): with
    v = u0/(48 pi m_I), A = Gamma2/m_I^2, D = Gamma2', G = Gamma4/(48 pi m_I), F = 1 + (N+2) v - 4 (N+2) v^2,
        ubar = G A^(-1/2) D^(-3/2) -> v(ubar) by reversion, Z = 1/D, Zg = v/G,
        beta = -[d/dubar log(ubar Zg / Z^2)]^(-1), eta = beta d log Z/dubar, eta_phi2 = beta d log Z_phi2/dubar,
        Z_phi2^(-1) = (A - (v/2) A')/F, 1/nu = 2 + eta_phi2 - eta,
    and the coefficients in the paper's coupling ubar~ = (N+8) ubar follow by rescaling.  Uncertainties are
    propagated linearly (each input sum varied by its error, in quadrature).  The ninth order of route B is a
    near-cancellation: its lower-order inputs are amplified by factors of several hundred, so route B checks
    route A at 1e-4 to 1e-3, not better.

    usage: routeb.py [--N 0,1,2,3,4] [--K 9] [--run KIND:L:RUN.json[.xz] ...] [--validate] [--tex FILE]
      --validate   route B from Phi4tools' sums alone, against Phi4tools' published coefficients (must agree)
      --run        replace Phi4tools' sum of that order by the integrator's (KIND int4, int2 or int2d, L loops)
      --tex        write the paper's Table tab:routes and the route-B numbers of its text (N = 1) to FILE
    Without --validate, route B (Phi4tools' sums, replaced by the given runs) is compared with route A from the ledger,
    after each run's sum against Phi4tools' where they have it.
    Python >= 3.6, standard library only.

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
import glob
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rebar_ledger as RL  # noqa: E402

SUMS = os.path.join(RL.ROOT, "results", "checks", "phi4tools", "routeb-sums.json")
FUNCS = ("beta", "eta", "eta_phi2", "nu")


class S:
    """Truncated power series in one variable, orders 0..S.K (floats)."""
    K = 9

    def __init__(self, c):
        c = [float(x) for x in list(c)[:S.K + 1]]
        self.c = c + [0.0] * (S.K + 1 - len(c))

    def __add__(s, o):
        o = o if isinstance(o, S) else S([o])
        return S([a + b for a, b in zip(s.c, o.c)])
    __radd__ = __add__

    def __neg__(s): return S([-a for a in s.c])
    def __sub__(s, o): return s + (-(o if isinstance(o, S) else S([o])))

    def __mul__(s, o):
        if not isinstance(o, S):
            return S([a * o for a in s.c])
        r = [0.0] * (S.K + 1)
        for i, a in enumerate(s.c):
            if a:
                for j in range(S.K + 1 - i):
                    r[i + j] += a * o.c[j]
        return S(r)
    __rmul__ = __mul__

    def pw(s, p):
        """s^p for s with a non-zero constant term"""
        a0 = s.c[0]; a = [x / a0 for x in s.c]
        f = [0.0] * (S.K + 1); f[0] = 1.0
        for n in range(1, S.K + 1):
            f[n] = sum((p * k - (n - k)) * a[k] * f[n - k] for k in range(1, n + 1)) / n
        return S(f) * (a0 ** p)

    def inv(s): return s.pw(-1)
    def __truediv__(s, o): return s * (o.inv() if isinstance(o, S) else 1.0 / o)
    def d(s): return S([(k + 1) * s.c[k + 1] for k in range(S.K)] + [0.0])
    def vd(s): return S([k * x for k, x in enumerate(s.c)])

    def log1(s):
        """log(s) - log(s_0)"""
        q = s.vd() / s
        return S([0.0] + [q.c[k] / k for k in range(1, S.K + 1)])

    def compose(s, g):
        r = S([0.0]); p = S([1.0])
        for k in range(S.K + 1):
            r = r + p * s.c[k]; p = p * g
        return r

    def revert(s):
        g = S([0.0, 1.0 / s.c[1]])
        for _ in range(S.K + 1):
            g = g - (s.compose(g) - S([0.0, 1.0])) / s.c[1]
        return g


def series(N, P, K):
    """{func: [coefficient of ubar^k, k = 0..K]} in Nickel's ubar = g/(48 pi), from the sums P[(name, k)]."""
    S.K = K + 1
    t = lambda k: (-1) ** (k - 1) * 3.0 ** k
    A = S([1.0, 0.0] + [t(k) * P.get(("P2", k), 0.0) for k in range(2, K + 1)])
    D = S([1.0, 0.0] + [t(k) * P.get(("P2d", k), 0.0) for k in range(2, K + 1)])
    G = S([0.0, 1.0] + [t(k) * P.get(("P4", k), 0.0) for k in range(2, K + 1)])
    F = S([1.0, N + 2.0, -4.0 * (N + 2.0)])
    ub = G / (D.pw(1.5) * A.pw(0.5))
    v_of = ub.revert()
    Z = D.inv().compose(v_of)
    Zg = S(G.c[1:] + [0.0]).inv().compose(v_of)
    X = Zg / Z.pw(2)
    u = S([0.0, 1.0])
    beta = -(u / (S([1.0]) + u * X.log1().d()))
    G12 = (A - 0.5 * A.vd()) / F
    eta = beta * D.inv().log1().compose(v_of).d()
    etap2 = beta * G12.inv().log1().compose(v_of).d()
    nu = (S([2.0]) + etap2 - eta).inv()
    return {"beta": beta.c[:K + 1], "eta": eta.c[:K + 1], "eta_phi2": etap2.c[:K + 1], "nu": nu.c[:K + 1]}


def in_paper_coupling(res, N):
    """coefficients of ubar~ = (N+8) ubar: beta k -> /(N+8)^(k-1), the others k -> /(N+8)^k"""
    n8 = N + 8.0
    out = {"beta": [c / n8 ** (k - 1) for k, c in enumerate(res["beta"])]}
    for f in ("eta", "eta_phi2", "nu"):
        out[f] = [c / n8 ** k for k, c in enumerate(res[f])]
    return out


def phi4tools_sums(N):
    """{(name, k): (value, error)} at N from Phi4tools' per-graph values"""
    d = json.load(open(SUMS))["sums"]
    name = {"G4": "P4", "G2": "P2", "G2d": "P2d"}
    out = {}
    for key, s in d.items():
        kind, k = key.split("_")
        v = sum(float(c) * N ** j for j, c in enumerate(s["poly"]))
        out[(name[kind], int(k))] = (v, s["err"][int(N)] if 0 <= N <= 4 and N == int(N) else max(s["err"]))
    return out


def load_run(path):
    """an integrator run, plain or xz-compressed (results/checks/routeb*/runs/*.json.xz)"""
    if path.endswith(".xz"):
        import lzma
        with lzma.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def run_sum(kind, L, path, N):
    """{(name, k): (value, error)} from an integrator run of KIND at L loops"""
    R = load_run(path)
    if R.get("kind") != kind or int(R.get("L", -1)) != L:
        raise SystemExit("%s: a %s run at %s loops, not %s at %d" % (path, R.get("kind"), R.get("L"), kind, L))
    x = 0.0; e2 = 0.0
    for c in R["classes"]:
        if not c.get("n"):
            continue
        w = sum(a * N ** j for j, a in enumerate(c["weight_poly"]))
        x += w * c["value"]; e2 += (w * c["error"]) ** 2
    e = math.sqrt(e2)
    if kind == "int4":
        k = L + 1; f = 2.0 ** (k - 1) / 3.0; name = "P4"
    elif kind == "int2":
        k = L; f = 2.0 ** k; name = "P2"
    elif kind == "int2d":
        k = L; f = -(2.0 ** k); name = "P2d"
    else:
        raise SystemExit("route B takes int4, int2 or int2d runs, not " + kind)
    return {(name, k): (f * x, abs(f) * e)}


def with_errors(N, P, K):
    """central coefficients (paper coupling) and their linear-propagation errors"""
    vals = {key: v for key, (v, e) in P.items()}
    c0 = in_paper_coupling(series(N, vals, K), N)
    var = {f: [0.0] * (K + 1) for f in FUNCS}
    for key, (v, e) in P.items():
        if e <= 0 or key[1] > K:
            continue
        up = dict(vals); up[key] = v + e
        c1 = in_paper_coupling(series(N, up, K), N)
        for f in FUNCS:
            for k in range(K + 1):
                var[f][k] += (c1[f][k] - c0[f][k]) ** 2
    return c0, {f: [math.sqrt(x) for x in var[f]] for f in FUNCS}


TEXNAME = {"P4": (r"$\Gamma^{(4)}$", -1), "P2": (r"$\Gamma^{(2)}$", 0), "P2d": (r"$\partial_{p^2}\Gamma^{(2)}$", 0)}
_ks = lambda k: str(k) if k < 10 else "{%d}" % k
TEXF = {"beta": lambda k: r"$\beta_%s$" % _ks(k), "eta": lambda k: r"$10^3\eta_%s$" % _ks(k),
        "eta_phi2": lambda k: r"$\eta_{\phi^2,%d}$" % k, "nu": lambda k: r"$10^2\nu_%s$" % _ks(k)}


def collect_runs(runs, N):
    """{(name, k): (value, error)} from the --run arguments KIND:L:FILES, where FILES is a file, a comma-separated list
    or a glob (the shards of a sharded order: disjoint classes, so their sums add and their errors add in quadrature),
    and the device hours of all the files"""
    R = {}; hours = 0.0
    for r in runs:
        kind, L, p = r.split(":", 2)
        files = sorted(f for x in p.split(",") for f in (glob.glob(x) or [x]))
        for f in files:
            for key, (v, e) in run_sum(kind, int(L), f, N).items():
                R[key] = (R[key][0] + v, math.hypot(R[key][1], e)) if key in R else (v, e)
            hours += load_run(f)["order"]["core_s"] / 3600.0
    return R, hours
SCALE = {"beta": 1.0, "eta": 1e3, "eta_phi2": 1.0, "nu": 1e2}


def write_tex(path, runs):
    """paper/generated/routes.tex: the body of Table tab:routes and the numbers the text quotes, at N = 1.  Route B
    takes Phi4tools' sums below eight vertices and the runs from eight vertices on (to ten, for the tenth order); the
    variant with Phi4tools' sums at eight vertices too is quoted in the text, at the ninth order"""
    import coefficients as CO
    import tables as TB
    N = 1
    fmt = lambda v, e: TB.math_if(TB.fmt_err(v, e))
    ph = phi4tools_sums(N)
    R, hours = collect_runs(runs, N)
    K = max(9, max(k for _, k in R))      # the ninth order, and the tenth when the runs at ten vertices are given
    ours = dict(ph); ours.update(R)
    theirs = dict(ph); theirs.update({k: v for k, v in R.items() if k not in ph})
    cb, eb = with_errors(N, ours, K); cp, ep = with_errors(N, theirs, K)
    res = CO.compute()
    L = ["% generated/routes.tex -- body of Table tab:routes and the route-B numbers of the text, N = 1.",
         "% WRITTEN BY series/routeb.py --tex from the stored route-B runs and the ledger: do not edit by hand.",
         "\\newcommand*{\\routesumrows}{%"]
    zmax = 0.0
    for key in sorted(R, key=lambda x: (x[1], ["P4", "P2", "P2d"].index(x[0]))):
        name, k = key; v, e = R[key]; lab, dl = TEXNAME[name]
        if key in ph:
            pv, pe = ph[key]; z = (v - pv) / math.hypot(e, pe); zmax = max(zmax, abs(z))
            L.append("%s & %d & %s & %s & $%+.2f$ \\\\" % (lab, k + dl, fmt(v, e), fmt(pv, pe), z))
        else:
            L.append("%s & %d & %s & --- & --- \\\\" % (lab, k + dl, fmt(v, e)))
    L += ["}", "\\newcommand*{\\routecoefrows}{%"]
    ratios = []; zph = {}; rels = []
    for k in range(9, K + 1):
        for f in FUNCS:
            a, ae = res["coefficients"][f][str(k)]["N"][str(N)][:2]
            sc = SCALE[f]; z = (a - cb[f][k]) / math.hypot(ae, eb[f][k]); zmax = max(zmax, abs(z))
            ratios.append(eb[f][k] / ae)      # the error ratio of the two routes, over every order of the table
            if cb[f][k]:
                rels.append(eb[f][k] / abs(cb[f][k]))      # the relative precision of route B
            if k == 9:      # the variant with Phi4tools' sums at eight vertices is quoted at the ninth order
                zph[f] = (a - cp[f][k]) / math.hypot(ae, ep[f][k])
            L.append("%s & & %s & %s & $%+.2f$ \\\\" % (TEXF[f](k), fmt(sc * a, sc * ae), fmt(sc * cb[f][k], sc * eb[f][k]), z))
    # Phi4tools' quoted error of eta_8 against the quadrature of their per-graph errors that route B propagates
    q = with_errors(N, ph, 8)[1]["eta"][8]; quoted = CO.phi4tools(N)["eta"][8][1]
    a8, ae8 = res["coefficients"]["eta"]["8"]["N"][str(N)][:2]; z8 = (a8 - cp["eta"][8]) / math.hypot(ae8, ep["eta"][8])
    L += ["}",
          "\\newcommand*{\\routeBhours}{%.1f}" % hours,
          "\\newcommand*{\\routeBzmax}{%.1f}" % zmax,
          "\\newcommand*{\\routeBratiomin}{%d}" % int(round(min(ratios), -1)),
          "\\newcommand*{\\routeBratiomax}{%d}" % int(round(max(ratios), -1)),
          "\\newcommand*{\\routeBrelmin}{$%s$}" % TB.sci(min(rels), 1),
          "\\newcommand*{\\routeBrelmax}{$%s$}" % TB.sci(max(rels), 1),
          "\\newcommand*{\\routeBzphibeta}{$%+.1f$}" % zph["beta"],
          "\\newcommand*{\\routeBzphieta}{$%+.1f$}" % zph["eta"],
          "\\newcommand*{\\routeBzphietaphi}{$%+.1f$}" % zph["eta_phi2"],
          "\\newcommand*{\\routeBzphinu}{$%+.1f$}" % zph["nu"],
          "\\newcommand*{\\routeBzphietaeight}{$%+.1f$}" % z8,
          "\\newcommand*{\\routeBquadratio}{%.1f}" % (quoted / q), ""]
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("wrote", os.path.relpath(path, RL.ROOT))
    return 0


def main(argv):
    Ns = [1]; K = 9; runs = []; validate = False; tex = None
    it = iter(argv)
    for a in it:
        if a == "--N": Ns = [float(x) for x in next(it).split(",")]
        elif a == "--K": K = int(next(it))
        elif a == "--run": runs.append(next(it))
        elif a == "--validate": validate = True
        elif a == "--tex": tex = next(it)
        else: raise SystemExit(__doc__)
    import coefficients as CO
    if validate:
        worst = 0.0
        for N in Ns:
            Kv = min(K, 8)
            c, e = with_errors(N, phi4tools_sums(N), Kv)
            ref = CO.phi4tools(N)
            print("N = %g: route B from Phi4tools' sums against Phi4tools' published coefficients" % N)
            for f in ("beta", "eta", "nu"):
                for k in range(2, Kv + 1):
                    if k >= len(ref[f]):
                        continue
                    rv, re_ = ref[f][k]
                    s = math.hypot(e[f][k], re_)
                    z = (c[f][k] - rv) / s if s > 0 else (0.0 if abs(c[f][k] - rv) < 1e-12 * max(1.0, abs(rv)) else float("inf"))
                    worst = max(worst, abs(z))
                    print("  %-4s k=%d  route B %+.10e +- %.1e   published %+.10e +- %.1e   z %+.2f" % (f, k, c[f][k], e[f][k], rv, re_, z))
        print("worst |z| = %.2f  %s" % (worst, "PASS" if worst < 3 else "FAIL"))
        return 0 if worst < 3 else 1
    if tex:
        return write_tex(tex, runs)
    res = CO.compute()
    for N in Ns:
        P = phi4tools_sums(N); ph = dict(P)
        R = collect_runs(runs, N)[0]
        for key, (v, e) in sorted(R.items(), key=lambda x: (x[0][1], x[0][0])):
            if key in ph:
                pv, pe = ph[key]
                print("N = %g: %s_%d run %+.10e +- %.1e   Phi4tools %+.10e +- %.1e   z %+.2f" % (
                    N, key[0], key[1], v, e, pv, pe, (v - pv) / math.hypot(e, pe)))
            else:
                print("N = %g: %s_%d run %+.10e +- %.1e (relative %.1e)" % (N, key[0], key[1], v, e, e / abs(v)))
        P.update(R)
        Kb = min(K, max(k for _, k in P))
        c, e = with_errors(N, P, Kb)
        print("N = %g: route A (ledger) against route B (%s)" % (N, "Phi4tools' sums" + (" and %d run(s)" % len(runs) if runs else "")))
        for f in FUNCS:
            for k in range(2, Kb + 1):
                a = res["coefficients"].get(f, {}).get(str(k), {}).get("N", {}).get(str(int(N)))
                if not a:
                    continue
                s = math.hypot(a[1], e[f][k])
                print("  %-8s k=%d  route A %+.9e +- %.1e   route B %+.9e +- %.1e   z %+.2f" % (f, k, a[0], a[1], c[f][k], e[f][k], (a[0] - c[f][k]) / s if s else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
