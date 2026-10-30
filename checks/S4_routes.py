#!/usr/bin/env python3
# S4_routes.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check S4: route B (the intermediate-scheme vertex functions through the Z factors) agrees with Phi4tools' sums
# and with route A at the ninth and tenth orders (Table tab:routes).
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
"""S4 -- route B agrees with Phi4tools' sums and with route A at the ninth and tenth orders (Table tab:routes).

Claim (paper, Sec. II E "New orders" and Sec. III C "Two independent routes"): the formulas of route B reproduce
Phi4tools' published coefficients through the eighth order from its per-graph values; the route-B runs at eight
vertices agree with Phi4tools' sums, and the ninth- and tenth-order coefficients of beta, eta, eta_phi2 and nu that
route B gives from the runs at eight to ten vertices agree with those of route A; every number of Table tab:routes and of
the text about route B (paper/generated/routes.tex) is regenerated from the stored runs and the ledger.

Inputs: series/routeb.py, results/checks/phi4tools/routeb-sums.json, the six runs of results/checks/routeb/runs/
(18 October 2026: int4 at 7 and 8 loops, int2 and int2d at 8 and 9) and the three of results/checks/routeb10/runs/
(19 October 2026: int4 at 9 loops, int2 and int2d at 10, each merged from its shards), the ledger (through
series/coefficients.py), paper/generated/routes.tex.
What runs:
  (a) routeb.py --validate at N = 0..4: route B from Phi4tools' sums against Phi4tools' coefficients, k <= 8;
  (b) routeb.py --tex into a temporary directory, against paper/generated/routes.tex byte for byte;
  (c) every difference z of the table, in units of the combined uncertainty.
PASS: (a) worst |z| < 3; (b) identical, and \\input by the paper; (c) the 11 differences (three eight-vertex sums, four
coefficients at each of the two orders), every |z| < 3.
"""
import io, os, re, sys, tempfile
from contextlib import redirect_stdout

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "route B reproduces Phi4tools' sums and coefficients and agrees with route A at the ninth and tenth orders (Table tab:routes)"
QUICK = False
RUNS9 = os.path.join(C.RESULTS, "routeb", "runs")
RUNS10 = os.path.join(C.RESULTS, "routeb10", "runs")
INPUTS = [("int4", 7, RUNS9), ("int2", 8, RUNS9), ("int2d", 8, RUNS9), ("int4", 8, RUNS9), ("int2", 9, RUNS9),
          ("int2d", 9, RUNS9), ("int4", 9, RUNS10), ("int2", 10, RUNS10), ("int2d", 10, RUNS10)]
NDIFF = 3 + 4 * 2      # the eight-vertex sums against Phi4tools, and four coefficients at the ninth and tenth orders


def main():
    RB = C.series_module("routeb")
    # (a)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = RB.main(["--validate", "--N", "0,1,2,3,4"])
    m = re.search(r"worst \|z\| = ([\d.]+)", buf.getvalue())
    worst = float(m.group(1)) if m else float("inf")
    n = len(re.findall(r" z [+-]", buf.getvalue()))
    oka = rc == 0 and worst < 3 and n > 0
    print("(a) route B from Phi4tools' per-graph sums against its published coefficients, N = 0-4, k <= 8: %d coefficients, "
          "worst |z| = %.2f  %s" % (n, worst, "OK" if oka else "MISMATCH"))
    # (b)
    runs = []
    for kind, L, RUNS in INPUTS:
        p = os.path.join(RUNS, "%s_L%d.json.xz" % (kind, L))
        if not os.path.exists(p):
            C.verdict("S4", False, "missing stored run %s" % p)
            return
        runs.append("%s:%d:%s" % (kind, L, p))
    ref = os.path.join(C.PAPER, "generated", "routes.tex")
    with tempfile.TemporaryDirectory(prefix="rebar-S4-") as d:
        out = os.path.join(d, "routes.tex")
        with redirect_stdout(io.StringIO()):
            RB.write_tex(out, runs)
        new = open(out).read()
    old = open(ref).read() if os.path.exists(ref) else ""
    srcs = " ".join(open(os.path.join(C.PAPER, f)).read() for f in ["main.tex"] +
                    [os.path.join("sections", x) for x in sorted(os.listdir(os.path.join(C.PAPER, "sections")))])
    inp = re.search(r"\\input\{generated/routes(\.tex)?\}", srcs) is not None
    okb = new == old and inp
    print("(b) paper/generated/routes.tex regenerated from the %d stored runs and the ledger: %s; \\input by the paper: %s"
          % (len(runs), "byte-identical" if new == old else "DIFFERS", "yes" if inp else "NO"))
    # (c)
    zs = []
    for line in new.splitlines():
        cells = [c.strip() for c in line.rstrip().rstrip("\\").split("&")]
        m = re.match(r"^\$([+-][\d.]+)\$$", cells[-1]) if len(cells) == 5 else None
        if m:
            zs.append(float(m.group(1)))
            print("      %s" % " & ".join(cells))
    okc = len(zs) == NDIFF and max(abs(z) for z in zs) < 3
    print("(c) %d differences in the table, largest |z| = %.2f  %s" % (len(zs), max(abs(z) for z in zs) if zs else float("nan"),
                                                                         "OK" if okc else "MISMATCH"))
    C.verdict("S4", oka and okb and okc,
              "route B reproduces Phi4tools' coefficients (worst |z| %.2f); Table tab:routes regenerated byte for byte; "
              "%d differences, largest |z| %.2f" % (worst, len(zs), max(abs(z) for z in zs) if zs else float("nan")))


if __name__ == "__main__":
    main()
