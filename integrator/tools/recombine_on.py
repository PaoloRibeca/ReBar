#!/usr/bin/env python3
"""Recombine the per-class integrals of one run (any N, typically N = 1) at other values of N.

Usage: recombine_on.py RUN.json [--N 0,1,2,3] [--legacy]

Each class contributes  P_G(N) * pref * mean_w  with error  |P_G(N)| * pref * sd_w / sqrt(n), where P_G is the
class's weight polynomial `weight_poly` (= S_G * on_poly_G(N), or for a --dedup class merged from Whitney twins
the sum over the twins of S_j * on_j(N): the twins' O(N) factors usually differ).  --legacy uses S * on_poly of the
representative only (what the v0.2 JSON allowed; wrong for merged classes), for comparison.
References: src/refs.h (Phi4tools O(N) series via the CS relations), when available.  Python 3 stdlib only.
"""
import json, math, os, re, sys


def load_refs():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "refs.h")
    refs = {}
    try:
        for m in re.finditer(r'\{"(\w+)",\s*(\d+),\s*(\d+),\s*([-+0-9.eE]+),\s*([-+0-9.eE]+)\}', open(p).read()):
            refs[(m.group(1), int(m.group(2)), int(m.group(3)))] = (float(m.group(4)), float(m.group(5)))
    except OSError:
        pass
    return refs


def poly(c, N):
    return sum(x * N ** k for k, x in enumerate(c))


def main():
    args, opts = [], {}
    it = iter(sys.argv[1:])
    for x in it:
        if x == "--legacy":
            opts["legacy"] = True
        elif x.startswith("--"):
            opts[x[2:]] = next(it)
        else:
            args.append(x)
    if len(args) != 1:
        print(__doc__)
        return 2
    R = json.load(open(args[0]))
    Ns = [float(x) for x in opts.get("N", "0,1,2,3").split(",")]
    refs = load_refs()
    kind, L = R["kind"], R["L"]
    out = 0
    for N in Ns:
        s = v = 0.0
        for c in R["classes"]:
            if "n" not in c:
                continue
            if opts.get("legacy") or "weight_poly" not in c:
                w = c["S"] * poly(c["on_poly"], N)
            else:
                w = poly(c["weight_poly"], N)
            s += w * c["pref"] * c["mean_w"]
            v += (w * c["pref"] * c["sd_w"]) ** 2 / c["n"]
        e = math.sqrt(v)
        line = f"{kind} L={L} N={N:g}: {s:.10g} +- {e:.3g}"
        r = refs.get((kind, int(N), L)) if N == int(N) else None
        if r:
            pull = (s - r[0]) / math.sqrt(e * e + r[1] * r[1])
            line += f"   ref {r[0]:.10g}   pull {pull:+.2f}   rel.dev {(s - r[0]) / r[0]:+.2e}"
            out = max(out, abs(pull))
        print(line)
    print(f"max |pull| {out:.2f}  ({'legacy S*on_poly of the representative' if opts.get('legacy') else 'weight_poly'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
