#!/usr/bin/env python3
# I3_twins.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check I3: Whitney (cycle-matroid) twins have pointwise equal subtracted integrands.
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
"""I3 -- Whitney twins have pointwise equal subtracted integrands.

Claim (paper, Sec. II B 2, "Twins"): 2-isomorphic graphs (twins: the same cycle matroid) have equal
integrals; their subtracted integrands, summed over the positions of the insertion, agree pointwise under the matroid
map of the lines: all 3 pairs at four loops and all 25 non-vanishing pairs at five loops for B_L.

Inputs: integrator/data/lists/gamma4_L4.txt, gamma4_L5.txt (and gamma4_L6.txt, beyond the claim).
What runs: `rebar_cpu check --what twins`: for every pair of graphs of a list with isomorphic cycle matroids
(src/graph.cpp matroid_iso), both classes are built and evaluated in long double at 64 random points with the lines of
the second mapped by the matroid isomorphism; EQUAL means |fA - fB| <= 1e-15 sum|terms| and <= 1e-11 max|f|.
PASS: at L = 4 and 5 every pair is EQUAL, with 3 pairs at L = 4 and 25 pairs with terms (non-vanishing) at L = 5;
the 6-loop pairs are reported and must be EQUAL too.
"""
import os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "Whitney twins have pointwise equal subtracted integrands (Sec. II B 2)"
QUICK = True
EXPECT = {4: 3, 5: 25}     # non-vanishing pairs quoted by the paper


def main():
    exe = C.cpu_binary()
    if not exe:
        C.verdict("I3", True, "nothing run", skip="the CPU binary (make -C integrator cpu VARIANTS=e16 failed)")
    ok, cells = True, []
    for L in (4, 5, 6):
        st, so, se = C.run([exe, "check", "--what", "twins", "--kind", "b", "--loops", str(L), "--list", os.path.join(C.LISTS, "gamma4_L%d.txt" % L)])
        pairs = re.findall(r"^twins (\S+)\s+(\S+)\s+S=\S+ comps=(\d+),(\d+)\s+max\|fA-fB\|=(\S+) \(scale (\S+), max\|f\| (\S+)\) (EQUAL pointwise|DIFFERENT)", so, re.M)
        m = re.search(r"^(\d+) matroid-isomorphic pairs", so, re.M)
        npairs = int(m.group(1)) if m else -1
        nz = [p for p in pairs if int(p[2]) or int(p[3])]
        eq = sum(1 for p in pairs if p[7] == "EQUAL pointwise")
        rel = max([float(p[4]) / float(p[6]) for p in nz if float(p[6]) > 0] or [0.0])
        good = st == 0 and npairs == len(pairs) and eq == len(pairs) and (L not in EXPECT or len(nz) == EXPECT[L])
        print("b L=%d: %d matroid-isomorphic pairs, %d with terms (paper: %s), %d EQUAL pointwise; max |fA - fB| / max|f| = %.1e  %s"
              % (L, npairs, len(nz), EXPECT.get(L, "-"), eq, rel, "OK" if good else "FAIL"))
        ok = ok and good
        cells.append("L=%d %d/%d" % (L, eq, len(pairs)))
    C.verdict("I3", ok, "twin pairs equal pointwise: " + ", ".join(cells) + " (non-vanishing pairs at L = 4, 5: 3 and 25, as quoted)")


if __name__ == "__main__":
    main()
