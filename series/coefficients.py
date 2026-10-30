#!/usr/bin/env python3
# coefficients.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# From the ledger to the RG functions: the per-order sums B_L(N), H_L(N), C_L(N) as polynomials in N (exact O(N)
# weights times pooled class means) and at fixed N with their errors; then, through the Callan-Symanzik relations of
# integrator/README.md, the coefficients of beta, eta, eta_phi2 and nu in ubar = (N+8) u / 6 (Phi4tools'
# normalisation) at N = 0..4 with linearly propagated errors, and their numerator polynomials in N. Orders the
# ledger lacks are taken from the exact low orders and integrator/tools/refs.py (Phi4tools).
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
"""coefficients.py -- per-order sums and RG coefficients from the ledger.

usage: coefficients.py [--ledger DIR] [--source best|ledger|refs] [--N 0,1,2,3,4] [--json OUT] [--quiet]

Conventions (integrator/README.md, "What it computes"): u = g/8pi, ubar = (N+8) u/6,
    btilde = sum_L (-1)^(L+1) B_L u^(L+1),  h = sum_L (-1)^L H_L u^L,  c = sum_L (-1)^(L+1) C_L u^L,
    eta = 2h/(1+h),  beta_u = -u + 2 eta u + (2 - eta) btilde,  eta_phi2 = -(2 - eta) c,  1/nu = (2 - eta)(1 - c),
and in ubar: beta(ubar) = sum_n beta_n ubar^n with beta_n = k^(1-n) [u^n] beta_u, eta_n = k^-n [u^n] eta, and so on
(k = (N+8)/6).  Phi4tools' form: beta_n = p_beta,n(N) / (N+8)^(n-1), eta_n = p_eta,n(N) / (N+8)^n, nu_n and
eta_phi2,n = p(N) / (N+8)^n; the numerators p are obtained by exact interpolation at N = 0 .. deg+1 (the extra point
checks the degree).

Sources of the per-order sums (fixed per order, at N = 1, so that every N uses the same one):
  ledger  the pooled ledger wherever it has the order, else exact / refs.py;
  refs    exact low orders (B1, B2, H2, C1, C2) and refs.py only (the Phi4tools round trip);
  best    (default) the smallest error at N = 1 among exact, refs.py and the ledger.
Errors: first-order propagation of every input's error, added in quadrature (the inputs are independent: different
orders and kinds are different integrals sampled with different seeds).  Caveats: (i) the refs.py inputs are derived
from the same Phi4tools coefficients and their errors are treated as independent; (ii) the coefficients of one
function at different orders, and of different functions, share inputs and are correlated, as are the values at
different N (shared classes); use them jointly (e.g. in a resummation) only with the covariance, which is not
produced here; (iii) the polynomial coefficients are given without errors for the same reason.
"""
import json, math, os, sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebar_ledger as RL  # noqa: E402

K = 12                       # series in u through u^11 (beta to ubar^11 needs B_10)
FUNCS = ("beta", "eta", "eta_phi2", "nu")


def ledger_sums(ledger):
    """{(kind, L): (pooled classes, versions)} of the orders with data"""
    orders = RL.load_orders(ledger)
    out = {}
    for o, e in orders["orders"].items():
        if not e.get("contributions"):
            continue
        cl = RL.load_classes(ledger, o, fields=("weight_poly", "value", "error"))
        out[(e["kind"], e["L"])] = (cl, e.get("versions", []))
    return out


def at_N(classes, N):
    s = v = 0.0
    for r in classes:
        w = RL.poly(r["weight_poly"], N)
        s += w * r["value"]
        v += (w * r["error"]) ** 2
    return s, math.sqrt(v)


def choose(led, policy):
    """{(kind, L): 'exact' | 'refs' | 'ledger'}"""
    ref1 = RL.references(1)
    src = {}
    for key in set(ref1) | set(led):
        cands = []
        if key in ref1:
            cands.append((ref1[key][1], 0, "exact" if ref1[key][2] == "exact" else "refs"))
        if key in led and policy != "refs":
            cands.append((at_N(led[key][0], 1)[1], 1, "ledger"))
        if not cands:
            continue
        if policy == "ledger" and key in led:
            src[key] = "ledger"
        else:
            src[key] = min(cands)[2]
    return src


def inputs(led, src, N):
    refN = RL.references(N)
    out = {}
    for key, s in src.items():
        if s == "ledger":
            out[key] = at_N(led[key][0], N) + ("ledger",)
        else:
            out[key] = (refN[key][0], refN[key][1], s)
    return out


# ------------------------------------------------------------------------------------------------- series algebra
def mul(a, b):
    r = [0.0] * K
    for i, x in enumerate(a):
        if x:
            for j in range(K - i):
                r[i + j] += x * b[j]
    return r


def inv(a):
    r = [0.0] * K
    r[0] = 1 / a[0]
    for n in range(1, K):
        r[n] = -sum(a[j] * r[n - j] for j in range(1, n + 1)) / a[0]
    return r


def rg(vals, N):
    """vals {(kind, L): value} -> {func: [coefficient of ubar^n]}"""
    bt, h, c = [0.0] * K, [0.0] * K, [0.0] * K
    for (kind, L), v in vals.items():
        if kind == "b" and L + 1 < K:
            bt[L + 1] = (-1) ** (L + 1) * v
        elif kind == "h" and L < K:
            h[L] = (-1) ** L * v
        elif kind == "c" and L < K:
            c[L] = (-1) ** (L + 1) * v
    one_h = list(h); one_h[0] += 1
    eta = [2 * x for x in mul(h, inv(one_h))]
    tme = [-x for x in eta]; tme[0] += 2                      # 2 - eta
    beta = mul(tme, bt)
    for i in range(K - 1):
        beta[i + 1] += 2 * eta[i]
    beta[1] -= 1
    etaphi = [-x for x in mul(tme, c)]
    omc = [-x for x in c]; omc[0] += 1
    nu = inv(mul(tme, omc))
    k = (N + 8) / 6.0
    return {"beta": [beta[n] * k ** (1 - n) for n in range(K)], "eta": [eta[n] / k ** n for n in range(K)],
            "eta_phi2": [etaphi[n] / k ** n for n in range(K)], "nu": [nu[n] / k ** n for n in range(K)]}


def available(have):
    """highest power of ubar computable for each function from the orders in 'have'"""
    def upto(kind, lo, n):
        return all((kind, L) in have for L in range(lo, n + 1))
    out = {}
    for f in FUNCS:
        n = 0
        while n + 1 < K:
            m = n + 1
            ok = (upto("b", 1, m - 1) and upto("h", 2, m - 1)) if f == "beta" else upto("h", 2, m) if f == "eta" \
                else (upto("h", 2, m) and upto("c", 1, m))
            if not ok:
                break
            n = m
        out[f] = n
    return out


def propagate(inp, N):
    """central values and errors of every coefficient at this N"""
    vals = {k: v[0] for k, v in inp.items()}
    base = rg(vals, N)
    var = {f: [0.0] * K for f in FUNCS}
    for key, (v, e, s) in inp.items():
        if not e:
            continue
        p = dict(vals); p[key] = v + e
        d = rg(p, N)
        for f in FUNCS:
            for n in range(K):
                var[f][n] += (d[f][n] - base[f][n]) ** 2
    return base, {f: [math.sqrt(x) for x in var[f]] for f in FUNCS}


def numerator(f, n, values_at):
    """exact interpolation of the numerator polynomial p_f,n(N) from its values at N = 0 .. deg+1"""
    deg = n - 2 if f == "beta" else n - 1 if f == "eta" else n
    deg = max(deg, 0)
    pw = n - 1 if f == "beta" else n
    pts = [(Fr(N), Fr(values_at(N)) * (N + 8) ** pw) for N in range(deg + 2)]
    # Newton divided differences on the first deg+1 points, check on the last
    xs, ys = [p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]]
    coef = list(ys)
    for j in range(1, len(xs)):
        for i in range(len(xs) - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    p = [Fr(0)] * (deg + 1)                  # expand the Newton form
    for i in range(len(xs) - 1, -1, -1):
        q = [Fr(0)] * (deg + 1)              # p * (x - xs[i]) + coef[i]
        for j in range(deg):
            q[j + 1] += p[j]
            q[j] -= xs[i] * p[j]
        q[0] += coef[i]
        p = q
    xN, yN = pts[-1]
    resid = float(sum(c * xN ** j for j, c in enumerate(p)) - yN) / max(1.0, abs(float(yN)))
    return [float(x) for x in p], resid


def phi4tools(N):
    sys.path.insert(0, RL.TOOLS)
    import refs
    sys.path.pop(0)
    return {"beta": refs.beta_coeffs(float(N)), "eta": refs.eta_coeffs(float(N)), "nu": refs.nu_coeffs(float(N))}


def compute(ledger=RL.LEDGER, policy="best", Ns=(0, 1, 2, 3, 4), led=None):
    if led is None:
        led = ledger_sums(ledger) if policy != "refs" else {}
    elif policy == "refs":
        led = {}
    src = choose(led, policy)
    avail = available(set(src))
    res = {"policy": policy, "N": list(Ns), "sources": {RL.order_name(*k): s for k, s in sorted(src.items())},
           "available": avail, "sums": {}, "coefficients": {f: {} for f in FUNCS}}
    prov = sorted(RL.order_name(*k) for k, s in src.items() if s == "ledger" and any(v.startswith("0.3.") for v in led[k][1]))
    res["provisional_inputs"] = prov
    for key, (cl, vers) in sorted(led.items()):
        c = [0.0] * max(len(r["weight_poly"]) for r in cl)
        e = list(c)
        for r in cl:
            for j, w in enumerate(r["weight_poly"]):
                c[j] += w * r["value"]
                e[j] += (w * r["error"]) ** 2
        res["sums"][RL.order_name(*key)] = {"poly": c, "poly_err": [math.sqrt(x) for x in e], "versions": vers,
                                            "N": {str(N): list(at_N(cl, N)) for N in Ns}}
    cache = {}

    def coeffs(N):
        if N not in cache:
            cache[N] = propagate(inputs(led, src, N), N)
        return cache[N]
    for N in Ns:
        base, err = coeffs(N)
        pt = phi4tools(N)
        for f in FUNCS:
            for n in range(1, avail[f] + 1):
                d = res["coefficients"][f].setdefault(str(n), {"N": {}})
                d["N"][str(N)] = [base[f][n], err[f][n]]
                if f in pt and n < len(pt[f]) and (pt[f][n][0] or n == 0):
                    d.setdefault("phi4tools", {})[str(N)] = [pt[f][n][0], pt[f][n][1]]
    for f in FUNCS:            # numerators and the inputs each coefficient depends on
        for ns, d in res["coefficients"][f].items():
            n = int(ns)
            p, r = numerator(f, n, lambda N: coeffs(N)[0][f][n])
            d["numerator"], d["numerator_check"] = p, r
            d["denominator"] = "(N+8)^%d" % (n - 1 if f == "beta" else n)
            need = [("b", L) for L in range(1, n)] + [("h", L) for L in range(2, n)] if f == "beta" else \
                [("h", L) for L in range(2, n + 1)] + ([("c", L) for L in range(1, n + 1)] if f != "eta" else [])
            d["ledger_inputs"] = [RL.order_name(*k) for k in need if src.get(k) == "ledger"]
            d["provisional"] = any(o in prov for o in d["ledger_inputs"])
    return res


def report(res):
    out = ["Sources (policy %s): %s" % (res["policy"], ", ".join("%s %s" % kv for kv in res["sources"].items()))]
    out.append("\nPer-order sums from the ledger (at N):")
    for o, d in res["sums"].items():
        out.append("  %-4s " % o.upper() + "  ".join("N=%s %s" % (N, RL.fmt_ve(*v)) for N, v in d["N"].items()))
    for f in FUNCS:
        out.append("\n%s: coefficient of ubar^n (this work / Phi4tools, pull)" % f)
        for ns, d in sorted(res["coefficients"][f].items(), key=lambda kv: int(kv[0])):
            cells = []
            for N, (v, e) in d["N"].items():
                c = RL.fmt_ve(v, e) if e else "%.10g" % v
                ph = d.get("phi4tools", {}).get(N)
                if ph and (e or ph[1]) and int(ns) >= 4:
                    c += " [%s, %+.1f]" % (RL.fmt_ve(*ph) if ph[1] else "%.10g" % ph[0], (v - ph[0]) / math.sqrt(e * e + ph[1] ** 2))
                cells.append("N=%s %s" % (N, c))
            tag = " (ledger: %s)" % ",".join(d["ledger_inputs"]) if d["ledger_inputs"] else ""
            out.append("  n=%-2s %s%s" % (ns, "  ".join(cells), tag))
    return "\n".join(out)


def main(argv):
    ledger, policy, Ns, js, quiet = RL.LEDGER, "best", (0, 1, 2, 3, 4), None, False
    it = iter(argv[1:])
    for a in it:
        if a == "--ledger":
            ledger = next(it)
        elif a == "--source":
            policy = next(it)
        elif a == "--N":
            Ns = tuple(int(x) for x in next(it).split(","))
        elif a == "--json":
            js = next(it)
        elif a == "--quiet":
            quiet = True
        else:
            print(__doc__)
            return 2
    if policy not in ("best", "ledger", "refs"):
        print(__doc__)
        return 2
    led = ledger_sums(ledger) if policy != "refs" else None
    res = compute(ledger, policy, Ns, led=led)
    if policy == "best":
        # the same coefficients with every order sum from the ledger (none from refs.py): the largest change of a
        # coefficient of order 8 or higher, in units of its error under the default policy
        alt = compute(ledger, "ledger", Ns, led=led)
        worst = [0.0, None]
        for f, cf in res["coefficients"].items():
            for k, d in cf.items():
                if int(k) < 8 or k not in alt["coefficients"].get(f, {}):
                    continue
                for N, ve in d["N"].items():
                    v, e = ve[0], ve[1]
                    a = alt["coefficients"][f][k]["N"].get(N)
                    if a and e:
                        z = abs(a[0] - v) / e
                        if z > worst[0]:
                            worst = [z, [f, int(k), int(N)]]
        res["ledger_only"] = {"max_shift_in_sigma": worst[0], "at": worst[1], "sources": alt["sources"]}
    if js:
        RL.write_if_changed(js, RL.dump_json(res))
    if not quiet:
        print(report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
