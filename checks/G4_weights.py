#!/usr/bin/env python3
# G4_weights.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check G4: the weights and O(N) polynomials of rebar equal those of the 2003 implementation (GAU) on all graphs
# with L <= 5 and on its four-loop tables.
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
"""G4 -- weights and O(N) polynomials equal the 2003 implementation's on all graphs with L <= 5 and its four-loop tables.

Claim (paper, Sec. II B 1 "Weights" and II B 3): every graph carries S_G = T_G(N) E!/|Aut G|; the weights and O(N)
polynomials of rebar (integrator/src/graph.cpp: aut_order, sym_factor, on_factor) equal those of the independent
implementation of the thesis and of GAU (Ribeca 2003, Guida and Ribeca 2006) on all graphs with L <= 5 of its lists
and on all 47 graphs of its four-loop tables.

Inputs: graphs/lists/<L>-loop-2, <L>-loop-4 for L = 1-5 (the regenerated lists: 260 graphs, of which 251 were in
GAU's stored lists; the other 9 are the self-loop graphs that the 2003 colouring defect lost); results/checks/gau/
beta.specs (GAU's four-loop tables: label, weight and O(N) polynomial printed by GAU itself, 2005);
integrator/data/lists/gamma4_L1-5.txt, gamma2_L1-5_with_selfloops.txt.
What runs:
  (a) GAU's weight formula (globalSymmetryFactor) and O(N) contraction (symmetrizedONPolynomial), as transcribed
      in graphs/tools/gaugraph.py, on every graph with L <= 5, against rebar's own values for the same Nickel index
      (integrator/tools/lists/listcheck, compiled with src/graph.cpp: |Aut| at the half-edge level and the integer
      pairing counts of on_factor), in exact arithmetic;
  (b) the 47 D20/D40 entries of beta.specs with L >= 1: each matched by its canonical GAU string to a graph of our
      lists, GAU's printed weight and O(N) polynomial against rebar's (and gaugraph's);
  (c) `rebar_cpu gen` regenerates the lists gamma4_L1-5.txt, gamma2_L2-5.txt and gamma2_L1-5_with_selfloops.txt (its
      S column is sym_factor) byte for byte.
PASS: every weight and every O(N) coefficient equal (exactly) in (a) and (b), all 47 entries matched, (c) identical.
"""
import os, re, sys
from fractions import Fraction
from math import factorial

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "weights and O(N) polynomials equal the 2003 implementation's on all graphs with L<=5 and the four-loop tables (Sec. II B 1)"
QUICK = True
BETA = os.path.join(C.RESULTS, "gau", "beta.specs")


def rebar_rows():
    """{nickel: (S as Fraction, O(N) factor {k: Fraction})} from listcheck on the integrator's lists with L <= 5"""
    out = {}
    names = ["gamma4_L%d.txt" % L for L in range(1, 6)] + ["gamma2_L%d_with_selfloops.txt" % L for L in range(1, 6)]
    for n in names:
        E = 4 if n.startswith("gamma4") else 2
        L = int(re.search(r"_L(\d+)", n).group(1))
        V = L + 1 if E == 4 else L
        for line in C.listcheck_rows(os.path.join(C.LISTS, n), V).splitlines():
            if line.startswith("#"):
                continue
            f = line.split("\t")
            aut, counts = int(f[11]), [int(x) for x in f[13].split(",")]
            on = {k: Fraction(c, 3 ** int(f[2])) for k, c in enumerate(counts) if c}
            out[f[1]] = (Fraction(factorial(int(f[3])), aut), on, f[15])
    return out


def main():
    gg = C.gaugraph()
    rb = rebar_rows()
    ok = True
    # (a)
    graphs, bad, nproto = [], [], 0
    for E in (4, 2):
        for L in range(1, 6):
            with open(os.path.join(C.GAU, "%d-loop-%d" % (L, E))) as fh:
                for s in (x.strip() for x in fh if x.strip()):
                    g = gg.parse(s)
                    nk = g.nickel()
                    graphs.append((E, L, s, g, nk))
                    if nk not in rb:
                        bad.append("%s: not in the integrator's lists" % nk)
                        continue
                    S, on, proto = rb[nk]
                    nproto += proto == "1"
                    gon = {k: v for k, v in g.on_factor().items() if v}
                    if g.weight() != S or gon != on:
                        bad.append("%s: GAU weight %s O(N) %s, rebar %s %s" % (nk, g.weight(), gg.format_on(gon), S, gg.format_on(on)))
    print("(a) %d graphs with L <= 5 (Gamma(4) 167, Gamma(2) with self-loops 93): GAU's weight and O(N) contraction (gaugraph.py) vs "
          "rebar's |Aut| and pairing counts (src/graph.cpp): %d differ; rebar's own on_factor bit-identical to the exact counts on %d"
          % (len(graphs), len(bad), nproto))
    for x in bad[:10]:
        print("    " + x)
    ok = ok and not bad and len(graphs) == 260
    # (b)
    canon = {}
    for E, L, s, g, nk in graphs:
        if L <= 4:
            canon[g.amputated().canonical().gau()] = (g, nk)
    entries, bad_b = 0, []
    with open(BETA) as fh:
        for line in fh:
            f = line.split(None, 6)
            if len(f) == 7 and re.fullmatch(r"D[24]0L[1-9]\d*N\d+", f[0]):
                entries += 1
                w, on, s = Fraction(f[2]), {k: v for k, v in gg.parse_on(f[3]).items() if v}, f[6].strip()
                if s not in canon:
                    bad_b.append("%s %s: no graph of our lists has this canonical form" % (f[0], s))
                    continue
                g, nk = canon[s]
                S, ron, _ = rb[nk]
                gon = {k: v for k, v in g.on_factor().items() if v}
                if not (w == S == g.weight() and on == ron == gon):
                    bad_b.append("%s %s (%s): GAU %s %s, rebar %s %s" % (f[0], s, nk, w, f[3], S, gg.format_on(ron)))
    print("(b) beta.specs (GAU, 2005): %d D20/D40 entries with L >= 1, %d matched to our graphs; printed weight and O(N) "
          "polynomial = rebar's = gaugraph's for %d" % (entries, entries - sum(1 for x in bad_b if "no graph" in x), entries - len(bad_b)))
    for x in bad_b[:10]:
        print("    " + x)
    ok = ok and not bad_b and entries == 47
    # (c)
    exe = C.cpu_binary()
    nsame, ntot = 0, 0
    if exe:
        for legs, L, sl, name in [(4, L, False, "gamma4_L%d.txt" % L) for L in range(1, 6)] + [(2, L, False, "gamma2_L%d.txt" % L) for L in range(2, 6)] + \
                [(2, L, True, "gamma2_L%d_with_selfloops.txt" % L) for L in range(1, 6)]:
            st, so, se = C.run([exe, "gen", "--legs", str(legs), "--loops", str(L)] + (["--selfloops"] if sl else []))
            ntot += 1
            nsame += st == 0 and so == open(os.path.join(C.LISTS, name)).read()
        print("(c) rebar_cpu gen regenerates %d of %d lists byte for byte (gamma4_L1-5, gamma2_L2-5, gamma2_L1-5_with_selfloops)" % (nsame, ntot))
        ok = ok and nsame == ntot
    else:
        print("(c) no CPU binary: rebar_cpu gen not run")
    summary = "%d graphs with L <= 5 and %d beta.specs entries: weights and O(N) polynomials equal; rebar_cpu gen %d/%d lists identical" % (
        len(graphs), entries, nsame, ntot)
    C.verdict("G4", ok, summary, skip=None if exe else "the CPU binary for (c)")


if __name__ == "__main__":
    main()
