#!/usr/bin/env python3
"""Summarise replica ensembles (independent seeds of the same order): are the pulls N(0,1), and does the per-class
scatter match the quoted errors?  Python 3 stdlib only.

Usage: replicas.py DIR [DIR ...]      (each DIR holds r*.json runs of one kind / loop order)
"""
import glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import chi2_sf  # noqa: E402


def main():
    for d in sys.argv[1:]:
        Rs = []
        for f in sorted(glob.glob(os.path.join(d, "r*.json"))):
            try:
                R = json.load(open(f))
            except Exception:
                continue
            if R.get("final") and R["order"].get("pull") is not None:
                Rs.append(R)
        if len(Rs) < 3:
            print(f"{d}: {len(Rs)} complete replicas"); continue
        p = [R["order"]["pull"] for R in Rs]
        n = len(p)
        m = sum(p) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in p) / (n - 1))
        chi2 = sum(x * x for x in p)
        # unweighted mean of the order sums vs the reference, in units of its scatter
        v = [R["order"]["sum"] for R in Rs]
        vm = sum(v) / n
        vs = math.sqrt(sum((x - vm) ** 2 for x in v) / (n - 1) / n)
        ref = Rs[0]["order"]["ref"]
        cls = {}
        for R in Rs:
            for c in R["classes"]:
                if "n" in c and c.get("contrib_err", 0) > 0:
                    cls.setdefault(c["nickel"], []).append((c["contrib"], c["contrib_err"]))
        c2 = 0.0; dof = 0
        for xs in cls.values():
            w = [1 / e ** 2 for _, e in xs]
            mm = sum(wi * x for wi, (x, _) in zip(w, xs)) / sum(w)
            c2 += sum(wi * (x - mm) ** 2 for wi, (x, _) in zip(w, xs)); dof += len(xs) - 1
        out2 = sum(1 for x in p if abs(x) > 2)
        print(f"{os.path.basename(d.rstrip('/')):6s} n={n:3d}  mean pull {m:+.3f} +- {1 / math.sqrt(n):.3f}  sd {sd:.3f}  chi2/n {chi2 / n:.3f} (p={chi2_sf(chi2, n):.3f})  "
              f"|pull|>2: {out2} ({100 * out2 / n:.1f}%, expect 4.6%)  mean of sums vs ref {(vm - ref) / vs:+.2f} scatter-sigma  "
              f"per-class chi2/dof {c2 / max(dof, 1):.3f} (dof {dof}, p={chi2_sf(c2, dof):.3f})")


if __name__ == "__main__":
    main()
