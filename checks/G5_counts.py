#!/usr/bin/env python3
# G5_counts.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check G5: the graph and integrand counts of the paper's Table tab:graphcounts follow from the lists, and the numbers
# of classes in the ledger equal the integrand counts of the orders that have data.
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
"""G5 -- the counts of graphs and integrands in the paper's Table tab:graphcounts, and the ledger's classes.

Claim (paper, App. B, Table tab:graphcounts; Sec. II B): per loop order, the numbers of 1PI graphs without self-loops
|G4(L)|, |G2(L)|, of one-vertex-reducible (1VR) graphs among them, of two-point graphs with both legs on one vertex
(generalized tadpoles), and of integrands that survive the pruning: B = |G4| - 1VR, C = |G2| - 1VR (1VR graphs vanish
for C after the pruning), H = |G2| - |1VR or generalized tadpole|; and the ledger holds exactly that many classes for
every order with data.

Inputs: integrator/data/lists/gamma4_L*.txt, gamma2_L*.txt (Nickel lists; gamma4_L10.txt.xz is xz-compressed),
paper/sections/appendices.tex (the table rows), ledger/orders.json (classes per order).
What runs: every list is read; a graph is 1VR when its internal graph has a cut vertex (articulation point), decided
here from the Nickel index itself, since the six- and seven-loop lists (Phi4tools' files) carry no 1VR flag; on the
lists that carry one the test is checked against the flag on every graph. A generalized tadpole is a two-point graph
with a vertex block that starts with 'ee' (both external legs on that vertex). The derived counts are compared with the table row of the same L, and with the
ledger's class counts where an order has contributions. L = 1 of the two-point table is a self-loop graph (the
tadpole with two insertions) and is not checked against a self-loop-free list.
PASS: every table entry that a list covers equals the derived count, and every ledger order with data has as many
classes as the table's integrand count.
"""
import json, lzma, os, re, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLAIM = "the counts of Table tab:graphcounts follow from the lists, and the ledger's classes equal the integrand counts"
QUICK = False   # the cut-vertex test over the 1.6 million ten-loop four-point graphs takes several minutes


def edges(nick):
    """Vertices and internal lines of a Nickel index: block i lists, after its 'e's, the vertices joined to i."""
    blocks = nick.split("|")[:-1]
    E = []
    for i, b in enumerate(blocks):
        for ch in b:
            if ch != "e":
                E.append((i, int(ch, 36)))   # vertex labels 0-9 then a, b, ... (never beyond 9 here)
    return len(blocks), E


def is_1vr(nick):
    """One-vertex-reducible: the internal graph has a cut vertex (articulation point); self-loops are ignored."""
    V, E = edges(nick)
    adj = [set() for _ in range(V)]
    for u, v in E:
        if u != v:
            adj[u].add(v); adj[v].add(u)
    for cut in range(V):
        start = next((w for w in range(V) if w != cut), None)
        if start is None:
            return False
        seen = {start}; stack = [start]
        while stack:
            w = stack.pop()
            for x in adj[w]:
                if x != cut and x not in seen:
                    seen.add(x); stack.append(x)
        if len(seen) < V - 1:
            return True
    return False


def read_list(path):
    """[(nickel, 1VR)] of a Nickel list; the 1VR flag column when present (checked against the structural test on
    every graph), else the structural test."""
    opener = lzma.open if path.endswith(".xz") else open
    rows = []; mismatch = 0; flagged = False
    with opener(path, "rt") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"):
                continue
            f = s.split("\t")
            nick = next((x for x in f if "|" in x), None)
            if nick is None:
                continue
            flags = [x.strip() for x in f if x.strip().startswith("1VR=")]
            struct = is_1vr(nick)
            if flags:
                flagged = True
                if (flags[0] == "1VR=1") != struct:
                    mismatch += 1
            rows.append((nick, struct))
    if mismatch:
        raise SystemExit("FAIL G5: %s: the 1VR flag disagrees with the cut-vertex test on %d graphs" % (os.path.basename(path), mismatch))
    return rows


def gentad(nick):
    return any(b.startswith("ee") for b in nick.split("|"))


def table_rows(tex):
    s = open(tex).read()
    i = s.index("\\label{tab:graphcounts}")
    j = s.index("\\end{tabular}", i)
    out = {}
    for line in s[i:j].splitlines():
        line = line.strip()
        m = re.match(r"^(\d+)\s*&(.*)\\\\$", line)
        if not m:
            continue
        cells = [re.sub(r"\\footnotemark\[\d+\]|\\,|\s", "", c) for c in m.group(2).split("&")]
        vals = []
        for c in cells:
            vals.append(int(c) if c.isdigit() else None)
        out[int(m.group(1))] = vals   # [G4, 1VR4, B, G2, 1VR2, onevertex, H, C, selfloops]
    return out


def main():
    lists = os.path.join(ROOT, "integrator", "data", "lists")
    tab = table_rows(os.path.join(ROOT, "paper", "sections", "appendices.tex"))
    ledger = json.load(open(os.path.join(os.environ.get("REBAR_LEDGER") or os.path.join(ROOT, "ledger"), "orders.json")))["orders"]
    bad = []; checked = 0
    for L, row in sorted(tab.items()):
        g4 = next((os.path.join(lists, n) for n in ("gamma4_L%d.txt" % L, "gamma4_L%d.txt.xz" % L) if os.path.exists(os.path.join(lists, n))), None)
        g2 = next((os.path.join(lists, n) for n in ("gamma2_L%d.txt" % L, "gamma2_L%d.txt.xz" % L) if os.path.exists(os.path.join(lists, n))), None)
        derived = {}
        if g4:
            r = read_list(g4); n1 = sum(v for _, v in r)
            derived.update({"G4": len(r), "1VR4": n1, "B": len(r) - n1})
        if g2 and L >= 2:
            r = read_list(g2); n1 = sum(v for _, v in r); gt = sum(gentad(n) for n, _ in r)
            nh = sum(1 for n, v in r if not v and not gentad(n))
            derived.update({"G2": len(r), "1VR2": n1, "onevertex": gt, "H": nh, "C": len(r) - n1})
        names = ["G4", "1VR4", "B", "G2", "1VR2", "onevertex", "H", "C"]
        for k, name in enumerate(names):
            if name in derived and row[k] is not None:
                checked += 1
                if derived[name] != row[k]:
                    bad.append("L=%d %s: table %d, list %d" % (L, name, row[k], derived[name]))
        line = "L=%2d " % L + "  ".join("%s=%s" % (n, derived[n]) for n in names if n in derived)
        for kind, name in (("b", "B"), ("h", "H"), ("c", "C")):
            o = ledger.get("%s%d" % (kind, L))
            ncl = (o or {}).get("nclasses") or (o or {}).get("classes")
            if o and ncl and (o.get("contributions") or o.get("ncontrib")):
                checked += 1
                exp = derived.get(name, row[names.index(name)])
                line += "  ledger %s%d=%d" % (kind, L, ncl)
                if exp is not None and ncl != exp:
                    bad.append("ledger %s%d has %d classes, expected %s" % (kind, L, ncl, exp))
        print(line)
    if bad:
        print("FAIL G5: " + "; ".join(bad)); return 1
    print("PASS G5: %d table entries and ledger class counts agree with the lists (L = 2..10)" % checked); return 0


if __name__ == "__main__":
    sys.exit(main())
