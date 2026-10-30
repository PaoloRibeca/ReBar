#!/usr/bin/env python3
# G3_phi4tools.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check G3: our graph lists equal Phi4tools' as sets, Gamma(4) at L <= 7 and Gamma(2) at L <= 8.
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
"""G3 -- our lists equal Phi4tools' as sets at L <= 7 (Gamma4) and L <= 8 (Gamma2).

Claim (paper, Sec. II B 3 "Independent lists"): converted to Nickel indices, the generator's lists
(graphs/lists) equal those of Phi4tools (Sberveglieri and Spada, generated with feyngen) as sets, for Gamma(4)
at L <= 7 and Gamma(2) at L <= 8.

Inputs: graphs/lists/<L>-loop-4 (L = 1-7) and <L>-loop-2 (L = 2-8, of which the graphs without self-loops);
the Phi4tools lists paclet/Resources/INTEGRANDS/int4pt0<L+1>.txt and int2pt0<L>.txt of
github.com/g-spada/Phi4tools at tag v1.0.2 = commit abd3b8b85ff66e0fb51dfb1e72f773ccfc33b43b, kept verbatim in
integrator/data/lists/ (gamma4_L6, L7 = int4pt07, 08; gamma2_L6, L7 = int2pt06, 07) and results/checks/phi4tools/
(the others), with the SHA-256 recorded below.
What runs: the SHA-256 of every Phi4tools file; every graph of our lists converted to its canonical Nickel index by
graphs/tools/gaugraph.py (the self-loop graphs of the <L>-loop-2 lists left out); the two sets compared.  With
--fetch, every Phi4tools file is also downloaded again at the pinned commit and compared with the stored copy.
PASS: every SHA-256 matches, no list has duplicates, and the sets are equal at every order.
"""
import os, sys, urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "our lists equal Phi4tools' as sets at L<=7 (Gamma4) and L<=8 (Gamma2) (Sec. II B 3)"
QUICK = True
URL = "https://raw.githubusercontent.com/g-spada/Phi4tools/abd3b8b85ff66e0fb51dfb1e72f773ccfc33b43b/paclet/Resources/INTEGRANDS/"
P4T = os.path.join(C.RESULTS, "phi4tools")
# (E, L): (Phi4tools file name, stored copy, SHA-256)
FILES = {(4, 1): ("int4pt02.txt", P4T, "b17d72f1f51058f0ac1fe18d1596b33633baa79f316a3cbfe1abe7ad77515866"),
         (4, 2): ("int4pt03.txt", P4T, "2599f7bb479922afb7d8fc9aaad733f053a47ce294b17712b46e7dffe4163152"),
         (4, 3): ("int4pt04.txt", P4T, "32aa168332153ee7d38f7e68e1999e9034f8e08b3b0f938f34460415059173d1"),
         (4, 4): ("int4pt05.txt", P4T, "fbc01c3246d70bec3ff319a301cc348277f8b3bdd9e119feb306771c5a10c8c3"),
         (4, 5): ("int4pt06.txt", P4T, "f0c52a9409669f7bc922882cc2da086f4946842059297c23804180fb2409dde5"),
         (4, 6): ("int4pt07.txt", os.path.join(C.LISTS, "gamma4_L6.txt"), "d25b13e67036793d37ffa673bda0b15689cb1162e2f332b71db88ceb70811ec9"),
         (4, 7): ("int4pt08.txt", os.path.join(C.LISTS, "gamma4_L7.txt"), "276defc71766b04eb64dbcb71e45812673151dbf090454fbf1caf71a4c265079"),
         (2, 2): ("int2pt02.txt", P4T, "436e9bd3705f4fa639d50837059c53990467a94c32aa13826d90dbb65dba0257"),
         (2, 3): ("int2pt03.txt", P4T, "98d5497fd2ee250b27e498835cd529b9181e8a053026458acaba15b1b01b5b9b"),
         (2, 4): ("int2pt04.txt", P4T, "3dc77f60cf033e9493a63b409e98e281b91a81da23c2ce5ed89f52d726893263"),
         (2, 5): ("int2pt05.txt", P4T, "ff55f652562a87a5aba9a4795a07d9372fb5229e9a7cc753459af6ea9b6063a6"),
         (2, 6): ("int2pt06.txt", os.path.join(C.LISTS, "gamma2_L6.txt"), "3469009760bc98d9981b73761d241cbf3b6b6a41599386568539469e45d8ad11"),
         (2, 7): ("int2pt07.txt", os.path.join(C.LISTS, "gamma2_L7.txt"), "a6585c8e272c28bd2e1a1e5aeb5305630b15665d23ce4163cc0eb3112e2e5340"),
         (2, 8): ("int2pt08.txt", P4T, "2b13d3c2dbe3389e0d908367835601fbc7505ff06a42d93682176a8006c190f6")}


def path_of(E, L):
    name, where, sha = FILES[(E, L)]
    return os.path.join(where, name) if os.path.isdir(where) else where


def nickel_of(s):
    """canonical Nickel index of a GAU string (None for a graph with self-loops), by gaugraph.py"""
    g = C.gaugraph().parse(s)
    return None if g.selfloops() else g.nickel()


def phi4tools_set(path):
    out = []
    with open(path) as fh:
        for line in fh:
            f = [x for x in line.rstrip("\n").split("\t") if "|" in x]
            if f and not line.startswith("#"):
                out.append(f[0].strip())
    return out


def main(argv):
    fetch = "--fetch" in argv
    ok = True
    for (E, L), (name, where, sha) in sorted(FILES.items()):
        path = path_of(E, L)
        h = C.sha256(path)
        ref = phi4tools_set(path)
        fetched = ""
        if fetch:
            try:
                data = urllib.request.urlopen(URL + name, timeout=60).read()
                fetched = "; fetched again: %s" % ("identical" if data == open(path, "rb").read() else "DIFFERENT")
                ok = ok and "identical" in fetched
            except OSError as e:
                fetched = "; fetch failed (%s)" % e
                ok = False
        with open(os.path.join(C.GAU, "%d-loop-%d" % (L, E))) as fh:
            ours = [x for x in C.pmap(nickel_of, [l.strip() for l in fh if l.strip()]) if x is not None]
        same = set(ours) == set(ref)
        good = h == sha and same and len(set(ours)) == len(ours) and len(set(ref)) == len(ref)
        ok = ok and good
        print("Gamma(%d) L=%d: graphs/lists/%d-loop-%d -> %4d Nickel indices (%d distinct); Phi4tools %s (%s, SHA-256 %s): "
              "%4d (%d distinct); %s%s" % (E, L, L, E, len(ours), len(set(ours)), name, C.rel(path), "ok" if h == sha else "DIFFERENT",
                                           len(ref), len(set(ref)), "EQUAL SETS" if same else "DIFFERENT (%d only ours, %d only Phi4tools)"
                                           % (len(set(ours) - set(ref)), len(set(ref) - set(ours))), fetched))
    C.verdict("G3", ok, "equal sets: Gamma(4) L = 1-7 (%s graphs), Gamma(2) L = 2-8 (%s); every Phi4tools file matches its SHA-256"
              % (", ".join(str(len(phi4tools_set(path_of(4, L)))) for L in range(1, 8)),
                 ", ".join(str(len(phi4tools_set(path_of(2, L)))) for L in range(2, 9))))

if __name__ == "__main__":
    main(sys.argv[1:])
