#!/usr/bin/env python3
# N4_phi4tools.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N4: the per-order sums at 3-7 loops agree with Phi4tools' within errors; pull statistics.
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
"""N4 -- the per-order sums at 3-7 loops agree with Phi4tools' within errors (Table tab:validation).

Claim (paper, Sec. III A, Table tab:validation): the sums B_L, H_L, C_L at L = 3-7 (N = 1), pooled class by class
from the ledger, agree with the values derived from Phi4tools' series (Sberveglieri and Spada, JHEP 05 (2024) 073)
through the Callan-Symanzik relations (integrator/tools/refs.py), within the combined errors.

Inputs: the pooled classes (series/rebar_ledger.py iter_classes) and orders.json for b3-b7, h3-h7, c3-c7;
integrator/tools/refs.py (through
series/rebar_ledger.py references()); paper/generated/validation.tex (the table body).
What runs: every order sum is recomputed from its pooled classes, the pull z = (sum - ref)/sqrt(err^2 + ref_err^2)
is formed, and the 15 pulls are summarised (mean, standard deviation, chi^2 = sum z^2 and its p-value, signs).
The same pulls at N = 0, 2, 3, 4 are printed (informational: the classes are the same, so they are correlated with
those at N = 1).
PASS: every |z| <= 3 at N = 1, chi^2 p-value > 1e-3, the sums equal orders.json (1e-14), and the z column of
validation.tex is these pulls.  The mean pull and the sign count are reported (error calibration).
"""
import math, os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the per-order sums at 3-7 loops agree with Phi4tools' within errors; pull statistics (Table III)"
QUICK = True


def order_sum(cl, N):
    w = [sum(c * N ** k for k, c in enumerate(r["weight_poly"])) for r in cl]
    return sum(x * r["value"] for x, r in zip(w, cl)), math.sqrt(sum((x * r["error"]) ** 2 for x, r in zip(w, cl)))


def main():
    RL = C.series_module("rebar_ledger")
    orders = RL.load_orders(C.LEDGER)["orders"]
    with open(os.path.join(C.PAPER, "generated", "validation.tex")) as fh:
        table = fh.read()
    kind_row = {}
    cur = None
    for line in table.splitlines():
        m = re.match(r"^(\$([BHC])_L\$)? *& (\d+) & .*& (\\provisional\{)?\$([+-]\d+\.\d+)\$\}?(?: & \d+)? \\\\$", line)
        if m:
            cur = m.group(2) or cur
            kind_row[(cur.lower(), int(m.group(3)))] = m.group(5)
    zs, ok, bad_table, rows = {}, True, [], []
    for N in (1, 0, 2, 3, 4):
        refs = RL.references(N)
        for kind in "bhc":
            for L in range(3, 8):
                o = "%s%d" % (kind, L)
                cl = RL.load_classes(C.LEDGER, o, fields=("weight_poly", "value", "error"))
                s, e = order_sum(cl, N)
                rv, re_, src = refs[(kind, L)]
                z = (s - rv) / math.sqrt(e * e + re_ * re_)
                zs.setdefault(N, []).append(z)
                if N == 1:
                    dled = abs(s - orders[o]["sum"]) / abs(s)
                    tz = kind_row.get((kind, L))
                    if tz != "%+.2f" % z:
                        bad_table.append(o)
                    ok = ok and abs(z) <= 3 and dled < 1e-14
                    rows.append("%s%d %-15s ref %-15s %s  z %+.2f%s" % (kind.upper(), L, RL.fmt_ve(s, e), RL.fmt_ve(rv, re_) if re_ else "%.10g" % rv,
                                                                    src, z, "" if tz == "%+.2f" % z else "  (table: %s)" % tz))
        if N == 1:
            print("N = 1 (Table tab:validation):")
            print("\n".join("  " + r for r in rows))
        z = zs[N]
        n = len(z)
        mean = sum(z) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in z) / (n - 1))
        chi2 = sum(x * x for x in z)
        p = RL.chi2_sf(chi2, n)
        neg = sum(1 for x in z if x < 0)
        print("  N = %d: %d pulls, mean %+.2f (%.1f sigma from 0), sd %.2f, max |z| %.2f, chi2 %.1f / %d (p = %.2f), %d negative"
              % (N, n, mean, mean * math.sqrt(n), sd, max(abs(x) for x in z), chi2, n, p, neg))
        if N == 1:
            stats = (mean, sd, max(abs(x) for x in z), chi2, n, p, neg)
            ok = ok and p > 1e-3
    if bad_table:
        print("validation.tex z column differs for: %s" % ", ".join(bad_table))
    mean, sd, zmax, chi2, n, p, neg = stats
    if neg > n / 2 and mean * math.sqrt(n) < -2:
        print("NOTE: %d of %d pulls are negative and their mean is %.1f sigma below 0: a question of error "
              "calibration (not a failure of this claim, which is agreement within the quoted errors)" % (neg, n, -mean * math.sqrt(n)))
    C.verdict("N4", ok and not bad_table, "15 pulls at N = 1: max |z| %.2f, chi2 %.1f/%d (p = %.2f), mean %+.2f, sd %.2f, %d negative; "
              "Table III z column %s" % (zmax, chi2, n, p, mean, sd, neg, "reproduced" if not bad_table else "DIFFERS"))


if __name__ == "__main__":
    main()
