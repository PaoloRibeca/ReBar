#!/usr/bin/env python3
# G1_lists.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check G1: every graph list holds distinct, connected, 1PI graphs in canonical Nickel form, with the stated counts.
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
"""G1 -- every graph list holds distinct, connected, 1PI graphs in canonical form, with the stated counts.

Claim (paper, Sec. II B 1, Table tab:generation, App. B; graphs/README.md): the lists of 1PI Gamma(2) and Gamma(4)
graphs have the counts of thesis Table 3.1 through eight loops and the new ones beyond (Gamma(4): 197022, 1593254 at
9, 10 loops; Gamma(2) without self-loops: 9184, 59614; with self-loops, used only for checks: 63774, 463933); two
independent canonicalisers, one in the generator's string format and one on Nickel indices, agree on every graph, and
all graphs of every list are distinct.

Inputs: graphs/lists/* (the generator's GAU strings, through ten loops) and integrator/data/lists/* (the Nickel
lists the integrator reads, through nine loops).
What runs:
  (a) checks/tools/graphscan (compiled here) on every list: parse, vertex degrees, connectedness, one-particle
      irreducibility, loops, legs and self-loops of every graph, the canonical Nickel index of every graph (branch
      and bound), the number of distinct input strings and of distinct canonical forms;
  (b) the integrator's own graph code on every Nickel list with at most 9 vertices (integrator/tools/lists/listcheck,
      compiled with src/graph.cpp: its parser and degree check, is_1pi, an exact branch-and-bound canonicaliser, and
      src/graph.cpp's brute-force canonical form over all V! labellings for V <= 7);
  (c) graphs/tools/gaugraph.py's canonical Nickel index (Python) of every graph with L <= 7 and of every 997th
      graph beyond, against graphscan's;
  (d) the GAU lists, converted, against the Nickel lists as sets (the self-loop-free part where the GAU list has
      self-loops).
PASS: no parse error, every graph connected and 1PI with the list's loops and legs, no self-loops except in the lists
with them; the counts as stated; as many distinct strings and distinct canonical Nickel indices as graphs in every
list (so the generator's canonical strings and the Nickel indices are in bijection); every Nickel list in canonical
form for (a) and (b); (c) identical; (d) equal sets.
"""
import os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "every graph list holds distinct, connected, 1PI graphs in canonical Nickel form with the stated counts (Sec. II B 1, Table II, App. B)"
QUICK = False
COUNTS = {"4": dict(zip(range(1, 11), (1, 2, 8, 27, 129, 660, 3986, 26540, 197022, 1593254))),
          "2": dict(zip(range(2, 11), (1, 2, 6, 19, 75, 317, 1622, 9184, 59614))),
          "2s": dict(zip(range(1, 11), (1, 2, 5, 17, 68, 313, 1632, 9685, 63774, 463933)))}


def kind_of(name):
    """(L, E, family) of a list file: family '4', '2' (no self-loops) or '2s' (self-loops included)"""
    m = re.match(r"^(\d+)-loop-([24])(-ci)?$", name)
    if m:
        L, E = int(m.group(1)), m.group(2)
        return L, int(E), "4" if E == "4" else ("2" if m.group(3) else "2s")
    m = re.match(r"^gamma([24])_L(\d+)(_with_selfloops)?\.txt(\.xz)?$", name)
    if m:
        return int(m.group(2)), int(m.group(1)), "4" if m.group(1) == "4" else ("2s" if m.group(3) else "2")
    return None


def has_selfloop(nk):
    """a Nickel index lists a self-loop at vertex i as label i in part i"""
    return any((chr(48 + i) if i < 10 else chr(55 + i)) in p for i, p in enumerate(nk.split("|")[:-1]))


def gau_nickel(s):
    return C.gaugraph().parse(s).nickel()


def main():
    ok = True
    lists = []
    for d in (C.GAU, C.LISTS):
        for f in sorted(os.listdir(d)):
            k = kind_of(f)
            if k:
                lists.append((os.path.join(d, f), f) + k)
    lists.sort(key=lambda x: (x[0].startswith(C.LISTS), x[3], x[4], x[2]))
    total = 0
    print("(a) graphscan: graphs, distinct strings / canonical Nickel indices, canonical as listed (Nickel lists), loops, legs, self-loops")
    for path, name, L, E, fam in lists:
        r = C.scan(path)
        n = int(r["graphs"])
        total += n
        nick = r["nickel_inputs"] != "0"
        sl = C.histogram(r["selfloops"])
        good = (r["errors"] == "0" and r["disconnected"] == "0" and r["not_1pi"] == "0" and r["aut_overflow"] == "0"
                and C.histogram(r["loops"]) == {L: n} and C.histogram(r["legs"]) == {E: n}
                and (fam == "2s" or sl == {0: n}) and int(r["distinct_inputs"]) == n and int(r["distinct_nickel"]) == n
                and (not nick or int(r["canonical_as_listed"]) == n) and COUNTS[fam].get(L, n) == n)
        print("    %-30s %8d graphs (stated %s), distinct %s / %s%s; loops %s legs %s self-loops %s  %s"
              % (C.rel(path), n, COUNTS[fam].get(L, "-"), r["distinct_inputs"], r["distinct_nickel"],
                 ", canonical %s" % r["canonical_as_listed"] if nick else "", r["loops"], r["legs"], r["selfloops"], "OK" if good else "FAIL"))
        ok = ok and good
    # (b) the integrator's own graph code
    nb, bad_b = 0, []
    for path, name, L, E, fam in lists:
        if not path.startswith(C.LISTS):
            continue
        V = L + 1 if E == 4 else L
        if V > 9:
            print("(b) %s: %d vertices, not run (3^V pairings per graph in listcheck: hours)" % (name, V))
            continue
        text = C.listcheck_rows(path, V)
        rows = [l.split("\t") for l in text.splitlines() if not l.startswith("#")]
        status = re.search(r"#status (\d+)", text).group(1)
        canon = sum(1 for f in rows if f[9] == "1")
        brute = [f[10] for f in rows if f[10] != "-"]
        pi = sum(1 for f in rows if f[6] == "1")
        good = status == "0" and canon == len(rows) == pi and all(x == "1" for x in brute) and len({f[1] for f in rows}) == len(rows)
        nb += len(rows)
        if not good:
            bad_b.append(name)
        print("(b) listcheck %-30s %6d graphs: canonical (branch and bound) %d, brute force over V! labellings %s, 1PI %d  %s"
              % (name, len(rows), canon, "%d/%d" % (brute.count("1"), len(brute)) if brute else "not run (V = %d)" % V, pi, "OK" if good else "FAIL"))
    ok = ok and not bad_b
    # (c) gaugraph.py (Python) against graphscan (C++)
    nc, bad_c = 0, []
    for path, name, L, E, fam in lists:
        if path.startswith(C.LISTS):
            continue
        with open(path) as fh:
            strings = [x.strip() for x in fh if x.strip()]
        step = 1 if L <= 7 else 997
        idx = list(range(0, len(strings), step))
        py = C.pmap(gau_nickel, [strings[i] for i in idx])
        cc = C.nickel_out(path)
        diff = [i for i, x in zip(idx, py) if cc[i] != x]
        nc += len(idx)
        if diff:
            bad_c.append("%s: %d differ (first: line %d)" % (name, len(diff), diff[0] + 1))
    print("(c) gaugraph.py (Python) vs graphscan (C++) canonical Nickel indices: %d graphs (all with L <= 7, every 997th beyond), %d lists differ%s"
          % (nc, len(bad_c), ": " + "; ".join(bad_c) if bad_c else ""))
    ok = ok and not bad_c
    # (d) GAU lists against the Nickel lists
    nick_lists = {(L, E, fam): path for path, name, L, E, fam in lists if path.startswith(C.LISTS)}
    nd = 0
    for path, name, L, E, fam in lists:
        if path.startswith(C.LISTS):
            continue
        conv = C.nickel_out(path)
        for f2 in ([fam] if fam != "2s" else ["2s", "2"]):
            other = nick_lists.get((L, E, f2))
            if not other:
                continue
            mine = set(conv) if f2 == fam else {x for x in conv if not has_selfloop(x)}
            theirs = set(C.nickel_out(other))
            same = mine == theirs
            nd += 1
            ok = ok and same
            print("(d) %-22s%s = %-32s as sets: %s (%d graphs)" % (name, " (no self-loops)" if f2 != fam else "", os.path.basename(other),
                                                                   "EQUAL" if same else "DIFFERENT", len(mine)))
    C.verdict("G1", ok, "%d lists, %d graphs through ten loops: all connected, 1PI, distinct, counts as stated; Nickel lists canonical "
              "(graphscan and listcheck, %d graphs); gaugraph = graphscan on %d graphs; %d GAU/Nickel list pairs equal as sets"
              % (len(lists), total, nb, nc, nd))


if __name__ == "__main__":
    main()
