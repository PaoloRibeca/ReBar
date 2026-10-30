#!/usr/bin/env python3
# N6_paired.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N6: FP64 round-off does not bias the order sums (paired FP64/FP80 evaluation on identical samples).
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
"""N6 -- FP64 round-off does not bias the order sums.

Claim (paper, Sec. III A, "Round-off"): every run records, per class, the round-off scale eps_mach sum|terms| of its
samples relative to their spread; it is negligible through seven loops but grows with the order (at ten loops up to
0.6 of the per-sample spread, rms).  Evaluating the same sample points in double and in extended (x87, 64-bit
mantissa) precision measures the actual round-off.  For the 55 classes with the largest scales (every class of
B_8 ... C_10 with an rms scale above 1e-3 or a single-sample scale above 1, two controls per order from six loops)
no class shows a significant FP64-FP80 difference, and summed over the classes of an order the possible bias is
below 1e-5 of the order's statistical error.

Inputs: results/checks/paired-precision/pairs.tsv (per class: its list line and the output of
`rebar_cpu check --what fp --n 200000`, 17 October 2026), the pooled classes of the ledger (prefactor, weight;
series/rebar_ledger.py iter_classes), ledger/orders.json (order errors).
What runs: the stored outputs are re-analysed: per class the mean FP64-FP80 difference in units of its own
standard error (significance), and its effect on the order sum (|w_G(1)| pref_G times the mean difference) against
the order's error; then the four six-loop control classes are recomputed with the CPU binary and must reproduce the
stored output lines byte for byte (the paired sampler uses a fixed Philox stream).
PASS: no |significance| above 4 among the 55 classes, every order's summed bias below 1e-5 of its error, and the
live recomputation identical.
"""
import math, os, re, sys, tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "FP64 round-off does not bias the order sums: paired FP64/FP80 evaluation of the 55 most exposed classes (Sec. III A)"
QUICK = True
N = 200000
PAT = re.compile(r"mean\(w80\) (\S+) sd (\S+) \| FP64-FP80: mean (\S+)\s+rms (\S+)\s+max (\S+)")
LIVE = ("b6", "c6")


def main():
    path = os.path.join(C.RESULTS, "paired-precision", "pairs.tsv")
    rows = [l.rstrip("\n").split("\t") for l in open(path) if l.strip() and not l.startswith("#")]
    RL = C.series_module("rebar_ledger")
    orders = RL.load_orders(C.LEDGER)["orders"]
    meta = {}
    for o in sorted({r[1] for r in rows}):
        want = {line.split("|TAB|")[1] for _, oo, _, _, _, line, _ in rows if oo == o}
        for c in RL.iter_classes(C.LEDGER, o, fields=("nickel", "weight_poly", "pref")):
            if c["nickel"] in want:
                meta[(o, c["nickel"])] = c
    worst = 0.0; per = {}
    for idx, o, kind, L, var, line, out in rows:
        nk = line.split("|TAB|")[1]
        m = PAT.search(out)
        mw, sd, d, rms, mx = map(float, m.groups())
        c = meta[(o, nk)]
        k = abs(sum(c["weight_poly"])) * c["pref"]
        sig = d / (rms / math.sqrt(N)) if rms > 0 else 0.0
        worst = max(worst, abs(sig))
        b = per.setdefault(o, [0.0, 0.0, 0])
        b[0] += k * d; b[1] += (k * rms) ** 2 / N; b[2] += 1
    ok = worst < 4
    worst_order = 0.0
    for o in sorted(per, key=lambda x: (x[0], int(x[1:]))):
        b, v, n = per[o]
        r = abs(b) / orders[o]["err"]
        worst_order = max(worst_order, r)
        print("  %-4s %2d classes: summed bias %+.2e +- %.1e, %.1e of the order error %.2e" % (o, n, b, math.sqrt(v), r, orders[o]["err"]))
    ok = ok and worst_order < 1e-5
    print("  largest |significance| over %d classes: %.2f; largest summed bias: %.1e of an order error" % (len(rows), worst, worst_order))
    exe = C.cpu_binary()
    if not exe:
        C.verdict("N6", ok, "stored outputs re-analysed; live part not run", skip="the CPU binary for the live recomputation")
    same = 0; live = [r for r in rows if r[1] in LIVE]
    with tempfile.TemporaryDirectory() as td:
        for idx, o, kind, L, var, line, out in live:
            lp = os.path.join(td, "class.txt")
            with open(lp, "w") as fh:
                fh.write(line.replace("|TAB|", "\t") + "\n")
            st, so, se = C.run([exe, "check", "--what", "fp", "--kind", kind, "--loops", L, "--list", lp, "--n", str(N)])
            got = [x for x in so.splitlines() if "FP64-FP80" in x]
            hit = bool(got) and got[0] == out
            same += hit
            print("  live %s %-28s %s" % (o, line.split("|TAB|")[1], "identical" if hit else "DIFFERENT: " + (got[0] if got else se[-200:])))
    ok = ok and same == len(live)
    C.verdict("N6", ok, "55 classes: largest |significance| %.2f, largest summed bias %.1e of an order error; %d of %d live recomputations identical"
              % (worst, worst_order, same, len(live)))


if __name__ == "__main__":
    main()
