#!/usr/bin/env python3
# I2_vanish.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check I2: one-vertex-reducible graphs, graphs with self-loops and generalized tadpoles vanish identically after
# the algebraic pruning of the forest expansion.
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
"""I2 -- one-vertex-reducible graphs, graphs with self-loops and generalized tadpoles vanish after pruning.

Claim (paper, Sec. II B 2 "Constructing the integrands"): graphs that vanish identically come out of the pruning
with no terms at all, without any structural rule: graphs with a cut vertex (one-vertex-reducible, 1VR), graphs with
self-loops (L >= 2) and, for H_L, generalized tadpoles (both legs on one vertex, W_G = 0).  Verified explicitly up
to five loops, and for C_L that of the 1VR graphs up to seven.

Inputs: b: the generator of rebar (`--gen --selfloops`, Gamma(4) with self-loops, L = 2-5); h and c:
integrator/data/lists/gamma2_L<L>_with_selfloops.txt (L = 2-5); c: gamma2_L6.txt, gamma2_L7.txt (13 and 52 1VR
graphs, no self-loops).
What runs: `rebar_cpu check --what vanish` on every order: every class is built with the production options and the
number of terms left by the pruning is printed with the 1VR and self-loop flags of src/graph.cpp; then
`rebar_cpu info --drop-selfloops` on the h lists (the production shortcut skips exactly the self-loop graphs).
PASS: every 1VR or self-loop graph has 0 terms (b, h, c); for h every generalized tadpole has 0 terms; no other
graph has 0 terms; --drop-selfloops drops exactly the self-loop graphs.
"""
import os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "one-vertex-reducible graphs, graphs with self-loops and generalized tadpoles vanish after pruning (Sec. II B 2)"
QUICK = False
ROW = re.compile(r"^(\S+)\s+1VR=(\d) selfloop=(\d) raw_terms=(\d+)\s+pruned_terms=(\d+)")


def gtadpole(nk):
    """both external legs on one vertex"""
    return any(p.count("e") == 2 for p in nk.split("|"))


def main():
    exe = C.cpu_binary()
    if not exe:
        C.verdict("I2", True, "nothing run", skip="the CPU binary (make -C integrator cpu VARIANTS=e16 failed)")
    jobs = [("b", L, ["--gen", "--selfloops"]) for L in range(2, 6)]
    jobs += [(k, L, ["--list", os.path.join(C.LISTS, "gamma2_L%d_with_selfloops.txt" % L)]) for k in "hc" for L in range(2, 6)]
    jobs += [("c", L, ["--list", os.path.join(C.LISTS, "gamma2_L%d.txt" % L)]) for L in (6, 7)]
    ok, tot = True, {"1VR": 0, "selfloop": 0, "gtadpole": 0, "graphs": 0, "vanish": 0}
    for kind, L, src in jobs:
        st, so, se = C.run([exe, "check", "--what", "vanish", "--kind", kind, "--loops", str(L)] + src)
        rows = [ROW.match(x) for x in so.splitlines()]
        rows = [(m.group(1), m.group(2) == "1", m.group(3) == "1", int(m.group(4)), int(m.group(5))) for m in rows if m]
        n1vr = sum(1 for r in rows if r[1])
        nsl = sum(1 for r in rows if r[2])
        ngt = sum(1 for r in rows if kind == "h" and gtadpole(r[0]) and not (r[1] or r[2]))
        bad = []
        for nk, vr, sl, raw, pruned in rows:
            expected = vr or sl or (kind == "h" and gtadpole(nk))
            if expected != (pruned == 0):
                bad.append("%s (1VR %d, self-loop %d, %d -> %d terms)" % (nk, vr, sl, raw, pruned))
        van = sum(1 for r in rows if r[4] == 0)
        print("%s L=%d: %4d graphs (%s): %3d 1VR, %3d with self-loops%s; %3d vanish (raw terms of the vanishing ones: up to %d)%s"
              % (kind, L, len(rows), "--gen --selfloops" if src[0] == "--gen" else os.path.basename(src[1]), n1vr, nsl,
                 ", %d other generalized tadpoles" % ngt if kind == "h" else "", van, max([r[3] for r in rows if r[4] == 0] or [0]),
                 "" if not bad else "; UNEXPECTED: " + "; ".join(bad[:5])))
        ok = ok and not bad and rows and (st == 0 or kind == "c")
        tot["graphs"] += len(rows); tot["1VR"] += n1vr; tot["selfloop"] += nsl; tot["gtadpole"] += ngt; tot["vanish"] += van
    for L in range(2, 6):
        lst = os.path.join(C.LISTS, "gamma2_L%d_with_selfloops.txt" % L)
        st, so, se = C.run([exe, "info", "--kind", "h", "--loops", str(L), "--list", lst, "--drop-selfloops"])
        dropped = {m.group(1) for m in re.finditer(r"^dropped (\S+)\s+structural: self-loop", so, re.M)}
        with open(lst) as fh:
            sl = {f[1] for f in (x.rstrip("\n").split("\t") for x in fh) if len(f) > 4 and f[4] == "selfloop=1"}
        okd = st == 0 and dropped == sl
        print("info --drop-selfloops, h L=%d: %d dropped, %d self-loop graphs in the list  %s" % (L, len(dropped), len(sl), "OK" if okd else "MISMATCH"))
        ok = ok and okd
    C.verdict("I2", ok, "%d graphs (b 2-5, h 2-5, c 2-7): all %d 1VR, %d self-loop and %d further generalized-tadpole (h) graphs "
              "vanish, and only they (%d)" % (tot["graphs"], tot["1VR"], tot["selfloop"], tot["gtadpole"], tot["vanish"]))


if __name__ == "__main__":
    main()
