#!/usr/bin/env python3
# N5_pooling.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N5: the ledger is what its archived runs give (its invariants, and a replay bit for bit), and independent
# runs of a class are consistent (the per-class chi^2 of every merge test, and the scatter of every order).
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
"""N5 -- the ledger is what its runs give, and independent runs of the same class are consistent.

Claim (paper, Sec. II C 4 "Pooling the runs: the ledger" and III A; doc/LEDGER.md): runs with different seeds sample a
class independently, and their statistics pool exactly; every run of the ledger was consistent class by class with the
pool of the runs merged before it (rule 6 of series/rebar-merge.py: chi^2 of z = (v - V)/sqrt(e^2 + E^2) over the
classes in common, p > 1e-6, and the largest |z| with Bonferroni-corrected p > 1e-6).

Inputs: ledger/orders.json (the statistics of every order and the manifest: per run its SHA-256, coverage and merge
test), ledger/<order>/weights-*.tsv and stats-*.tsv (format 3: the exact weights and the folded registers); for (b)
the archive of the merged runs, ledger-archive/<order>/<id>.json.xz next to the ledger (outside git, deposited on
Zenodo).
What runs:
  (a) series/rebar-merge.py --check, the invariants: every file's SHA-256 as recorded; per order the sum of n over
      the classes equal to the samples folded by its runs, the sum of k (from their coverage) to the classes they
      folded; the statistics of orders.json equal to those recomputed from the stats; weights for every class;
  (b) series/rebar-merge.py --replay: the ledger rebuilt from the archived runs, in manifest order, into a temporary
      directory, every rule and merge test run again, and compared with the ledger byte for byte (every stats and
      weights file, and orders.json with the manifest and its merge tests); SKIP when the archive is absent;
  (c) the merge test of every run that had classes in common with the pool before it (its manifest entry,
      check_detail: chi^2, dof, p, largest |z|, Bonferroni p; recomputed by (b)), and no forced merge;
  (d) the scatter between seeds of every order with such tests (orders.json scatter: the sum of their chi^2 over the
      sum of their degrees of freedom, the classes in common; the tests are independent, so the sum is a chi^2).
PASS: (a) every invariant holds, (b) the replay is identical, every test of (c) has p > 1e-6 and Bonferroni
p > 1e-6, every order of (d) has p > 1e-6.
"""
import os, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the ledger replays bit for bit from its archived runs, and independent runs of a class are consistent (per-class chi^2) (Sec. II C 4, III A)"
QUICK = False   # the replay of the archive takes minutes
P_MIN = 1e-6


def main():
    RL = C.series_module("rebar_ledger")
    merge = os.path.join(C.SERIES, "rebar-merge.py")
    top = RL.load_orders(C.LEDGER)
    if top.get("format") != RL.FORMAT3:
        C.verdict("N5", True, "%s is not a format-3 ledger" % C.rel(C.LEDGER),
                  skip="a format-3 ledger (doc/LEDGER.md)")
    orders = {o: e for o, e in top["orders"].items() if e.get("contributions")}
    # (a) the invariants
    st, so, se = C.python(merge, "--ledger", C.LEDGER, "--check")
    lines = so.splitlines()
    print("(a) rebar-merge.py --check: %s" % (lines[-1] if lines else se.strip()[-300:]))
    for x in lines:
        if x.startswith("  "):
            print("   " + x)
    ok = inv = st == 0
    # (b) the replay of the archive
    archive = RL.archive_dir(C.LEDGER)
    skip, nruns = None, sum(len(e["contributions"]) for e in orders.values())
    if not os.path.isdir(archive):
        skip = "the archive of the runs (%s) for the replay" % C.rel(archive)
        print("(b) replay not run: no archive at %s" % archive)
        replay = "replay not run"
    else:
        st, so, se = C.python(merge, "--ledger", C.LEDGER, "--archive", archive, "--replay")
        lines = so.splitlines()
        print("(b) rebar-merge.py --replay: %s" % (lines[-1] if lines else se.strip()[-300:]))
        ok = ok and st == 0
        replay = "replay of %d archived runs %s" % (nruns, "identical bit for bit" if st == 0 else "DIFFERENT")
    # (c) the merge test of every run, (d) the scatter of every order
    ntests, worst, worst_b, forced, scat, nclasses = 0, (1.0, None), (1.0, None), [], [], 0
    print("(c, d) merge tests (each run against the pool before it) and scatter between seeds:")
    for o, e in orders.items():
        nclasses += e.get("classes") or 0
        tests = [(c["id"], c["check_detail"]) for c in e["contributions"] if (c.get("check_detail") or {}).get("dof")]
        forced += [c["id"] for c in e["contributions"] if c.get("forced")]
        line = "  %-4s %2d run(s), %6d classes" % (o, len(e["contributions"]), e.get("classes") or 0)
        if tests:
            pm = min(tests, key=lambda t: t[1]["p"])
            pb = min(tests, key=lambda t: t[1]["p_bonferroni"])
            ntests += len(tests)
            if pm[1]["p"] < worst[0]:
                worst = (pm[1]["p"], pm[0])
            if pb[1]["p_bonferroni"] < worst_b[0]:
                worst_b = (pb[1]["p_bonferroni"], pb[0])
            bad = [i for i, d in tests if d["p"] <= P_MIN or d["p_bonferroni"] <= P_MIN]
            ok = ok and not bad
            sc = e.get("scatter") or {}
            p = RL.chi2_sf(sc["chi2"], sc["dof"]) if sc.get("dof") else None
            line += ": %2d test(s), smallest p %.2g, Bonferroni %.2g, largest |z| %.2f" % (
                len(tests), pm[1]["p"], pb[1]["p_bonferroni"], max(d["maxz"] for _, d in tests))
            if p is not None:
                scat.append((sc["chi2"] / sc["dof"], p, o))
                ok = ok and p > P_MIN
                line += "; scatter chi2 %.1f / %d (chi2/dof %.3f, p = %.2g)" % (sc["chi2"], sc["dof"], sc["chi2"] / sc["dof"], p)
            if bad:
                line += "  FAIL: " + ", ".join(bad)
        else:
            line += ": no merge test with classes in common (one run, or disjoint shards)"
        print(line)
    if forced:
        print("forced merges: " + ", ".join(forced))
    ok = ok and not forced
    if scat:
        pmin = min(scat, key=lambda x: x[1])
        high = ", ".join("%s %.3f" % (o, r) for r, _, o in scat if RL.parse_order(o)[1] >= 7)
        scatter = "scatter chi2/dof %s; smallest p %.2g (%s)" % (high or "---", pmin[1], pmin[2])
    else:
        scatter = "no scatter between seeds"
    C.verdict("N5", ok, "%d orders, %d classes: invariants %s; %s; %d merge tests, smallest p %.2g (%s), Bonferroni %.2g "
              "(%s); %s%s" % (len(orders), nclasses, "hold" if inv else "VIOLATED", replay, ntests, worst[0], worst[1],
                              worst_b[0], worst_b[1], scatter, "; forced: " + ", ".join(forced) if forced else ""), skip=skip)


if __name__ == "__main__":
    main()
