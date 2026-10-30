#!/usr/bin/env python3
# N3_exact.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N3: the sampled two-loop sums reproduce the exact values B2 = 1/2, H2 = 1/81, C2 = 1/12 within errors.
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
"""N3 -- the exact two-loop sums B2 = 1/2, H2 = 1/81, C2 = 1/12 are reproduced within errors.

Claim (paper, Sec. II B 3 "Integrands" and III A, Table tab:validation): the sampled integrals reproduce the exact
two-loop sums B2 = 1/2, H2 = 1/81 and C2 = 1/12 (N = 1) to a few 1e-7.

Inputs: ledger/b2, h2, c2 (the pooled per-class values, through series/rebar_ledger.py iter_classes; orders.json:
the order sums).
Exact values as polynomials in N (rebar_ledger.exact_low): B2 = (5N+22)/54, H2 = (N+2)/243, C2 = (N+2)/36.
What runs: the order sums are recomputed from the pooled classes, sum_G weight_poly_G(N) value_G, with errors
added in quadrature, and compared with orders.json and with the exact values.
PASS: at N = 1 every pull |z| <= 3 and the relative deviation is below 1e-6; the sums agree with orders.json to 1e-14;
and the weight polynomial of the classes gives the exact N dependence, weight_poly(N)/weight_poly(1) =
exact(N)/exact(1) for N = 0..4 to 1e-14 (so the pulls at N = 0..4 equal those at N = 1).
"""
import math, os, sys
from fractions import Fraction

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the exact two-loop sums B2=1/2, H2=1/81, C2=1/12 are reproduced within errors (Sec. III A)"
QUICK = True
EXACT = {"b2": (Fraction(22, 54), Fraction(5, 54)), "h2": (Fraction(2, 243), Fraction(1, 243)), "c2": (Fraction(2, 36), Fraction(1, 36))}


def main():
    RL = C.series_module("rebar_ledger")
    orders = RL.load_orders(C.LEDGER)["orders"]
    ok, cells = True, []
    for o, (a, b) in EXACT.items():
        cl = RL.load_classes(C.LEDGER, o, fields=("weight_poly", "value", "error"))
        def at(N):
            s = sum(sum(w * N ** k for k, w in enumerate(r["weight_poly"])) * r["value"] for r in cl)
            e = math.sqrt(sum((sum(w * N ** k for k, w in enumerate(r["weight_poly"])) * r["error"]) ** 2 for r in cl))
            return s, e
        s, e = at(1)
        ex = float(a + b)
        z = (s - ex) / e
        rd = (s - ex) / ex
        led = orders[o]
        dled = abs(s - led["sum"]) / abs(led["sum"])
        shape = max(abs(at(N)[0] / s - float((a + b * N) / (a + b))) for N in range(5))
        ok_o = abs(z) <= 3 and abs(rd) < 1e-6 and dled < 1e-14 and shape < 1e-14
        ok = ok and ok_o
        print("%s: %d class(es), %d contribution(s), %.3g samples: %.10f +- %.2e, exact %s = %.10f, pull %+.2f, relative "
              "deviation %+.1e; orders.json %.10f (rel. diff %.0e, pull %+.2f there); N dependence vs exact %s: max %.1e  %s"
              % (o.upper(), len(cl), len(led["contributions"]), led["samples"], s, e, Fraction(a + b), ex, z, rd, led["sum"], dled,
                 led["pull"], "(%s + %s N)" % (a, b), shape, "OK" if ok_o else "FAIL"))
        cells.append("%s %+.2f sigma (rel. %+.1e)" % (o.upper(), z, rd))
    C.verdict("N3", ok, "pulls at N = 0..4: " + ", ".join(cells))


if __name__ == "__main__":
    main()
