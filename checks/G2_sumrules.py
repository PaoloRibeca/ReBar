#!/usr/bin/env python3
# G2_sumrules.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check G2: the zero-dimensional sum rules hold exactly, as polynomials in N, for every graph list through ten loops.
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
"""G2 -- the zero-dimensional sum rules hold exactly, as polynomials in N, for every list through ten loops.

Claim (paper, Sec. II B 3 "Zero-dimensional sum rules", App. C, Table tab:sumrules): in zero dimensions every integral equals
one, so that sum_G S_G T_G(N) over a complete list is the coefficient of g^L in the vertex function of the zero-
dimensional O(N) model (bare mass shifted so that every self-loop cancels, for the lists without self-loops); every
list passes it with the full O(N) polynomial through ten loops, and the rows of Table tab:sumrules (Gamma(4) at L = 1-10,
Gamma(2) at L = 2-10) are these polynomials; at N = 1 they give Sigma_4,9 = 2245562203/4608 and Sigma_4,10 =
4057808535/1024 for the 197022 nine-loop and 1593254 ten-loop four-point graphs, and Sigma_2,10 = 506798407/41472
for the 59614 ten-loop two-point graphs.

Inputs: integrator/data/lists/* (Nickel lists; gamma4_L10.txt.xz is xz-compressed), graphs/lists/* (the
generator's lists, through ten loops); integrator/tools/lists/zerodim_on.py (the exact zero-dimensional series as
polynomials in N, derived from the moments, here through g^11).
What runs:
  (a) the integrator's own weights on every Nickel list with at most 9 vertices (all but gamma4_L9, gamma4_L10 and
      gamma2_L10, whose 3^V pairings per graph it would take hours to enumerate): integrator/tools/lists/listcheck,
      the integrator's S_G and the integer counts of the 3^V vertex pairings of T_G(N); then
      integrator/tools/lists/verify_lists.py sumrule, exactly, which also sums the integrator's double-precision weights;
  (b) the full O(N) polynomial, exactly (rational arithmetic), on every list of both directories through ten loops:
      S_G = E!/|Aut G| from checks/tools/graphscan's |Aut G| (a C++ companion of gaugraph.py) and T_G(N) from its
      pairing counts (graphscan --on, a transfer matrix over the vertices instead of the 3^V enumeration), which equal
      listcheck's graph by graph on every list of (a); and, apart, the sum of S_G alone against the series at N = 1;
  (c) the rows of Table tab:sumrules as printed (TABLE: numerator, denominator D, value at N = 1), against the series and (b);
  (d) the house checkers of graphs/tools on the generator's lists through seven loops: gamma4_n1.py (Gamma(4) at
      N = 1 against the 2003 gamma4ZeroI), sumrule_check.py (Gamma(2) with self-loops, and the Gamma(2,1) O(N)
      polynomials of thesis p. 107 at 5 and 6 loops).
PASS: every comparison exact (EQUAL); the three nine- and ten-loop values at N = 1 as quoted.
"""
import io, os, re, sys, tempfile, contextlib
from fractions import Fraction
from math import factorial

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the zero-dimensional sum rules hold exactly, as full O(N) polynomials, for every list through ten loops, and they are the rows of Table tab:sumrules (Sec. II B 3, App. C)"
QUICK = False
QUOTED = {("c4", 9): Fraction(2245562203, 4608), ("c4", 10): Fraction(4057808535, 1024), ("c2", 10): Fraction(506798407, 41472)}
# Table tab:sumrules as printed: (family, L) -> (numerator, D, value at N = 1)
TABLE = {
    ("c4", 1): ("8+N", 6, "3/2"),
    ("c4", 2): ("108+26N+N^2", 36, "15/4"),
    ("c4", 3): ("2048+724N+62N^2+N^3", 216, "105/8"),
    ("c4", 4): ("47648+21632N+2928N^2+124N^3+N^4", 1296, "893/16"),
    ("c4", 5): ("1297664+702864N+127392N^2+9044N^3+220N^4+N^5", 7776, "8795/32"),
    ("c4", 6): ("40362304+24950944N+5549104N^2+546408N^3+23404N^4+358N^5+N^6", 46656, "97987/64"),
    ("c4", 7): ("1410726400+966932160N+250483648N^2+31004768N^3+1898240N^4+53312N^5+546N^6+N^7", 279936, "10951025/1152"),
    ("c4", 8): ("54747705344+40784744192N+11897634304N^2+1746112576N^3+137234272N^4+5654416N^5+110208N^6+792N^7+N^8", 1679616,
                "49985915/768"),
    ("c4", 9): ("2336870225920+1865178040064N+598669644160N^2+100305560704N^3+9494392864N^4+511500848N^5+14961264N^6+211032N^7"
                "+1104N^8+N^9", 10077696, "2245562203/4608"),
    ("c4", 10): ("108856221154304+92103242480128N+31990963159296N^2+5962312144640N^3+652186499392N^4+42906530592N^5"
                 "+1667820720N^6+36012864N^7+379788N^8+1490N^9+N^{10}", 60466176, "4057808535/1024"),
    ("c2", 2): ("2+N", 18, "1/6"),
    ("c2", 3): ("10+7N+N^2", 54, "1/3"),
    ("c2", 4): ("280+232N+52N^2+3N^3", 648, "7/8"),
    ("c2", 5): ("2552+2384N+688N^2+71N^3+2N^4", 1944, "211/72"),
    ("c2", 6): ("113888+116608N+39952N^2+5680N^3+320N^4+5N^5", 23328, "3413/288"),
    ("c2", 7): ("1498912+1650832N+641936N^2+112904N^3+9260N^4+316N^5+3N^6", 69984, "48323/864"),
    ("c2", 8): ("90828928+106190272N+45557472N^2+9353504N^3+982000N^4+51060N^5+1134N^6+7N^7", 839808, "3123017/10368"),
    ("c2", 9): ("778012480+956252384N+444020448N^2+102678120N^3+12884816N^4+882730N^5+31028N^6+473N^7+2N^8", 1259712,
                "9443467/5184"),
    ("c2", 10): ("118963074560+152597681664N+75647848448N^2+19237783296N^3+2767156512N^4+231239968N^5+10977808N^6"
                 "+273456N^7+2982N^8+9N^9", 30233088, "506798407/41472"),
}


def family(name):
    m = re.match(r"^(\d+)-loop-([24])(-ci)?$", name)
    if m:
        return int(m.group(1)), "c4" if m.group(2) == "4" else ("c2" if m.group(3) else "c2B")
    m = re.match(r"^gamma([24])_L(\d+)(_with_selfloops)?\.txt(\.xz)?$", name)
    if m:
        return int(m.group(2)), "c4" if m.group(1) == "4" else ("c2B" if m.group(3) else "c2")
    return None


def table_poly(num, D):
    """'a+bN+cN^2+...' / D -> [Fraction coefficients of N^0, N^1, ...]"""
    c = {}
    for t in num.split("+"):
        m = re.match(r"^(\d*)(N(\^\{?(\d+)\}?)?)?$", t)
        k = int(m.group(4) or 1) if m.group(2) else 0
        c[k] = c.get(k, 0) + (int(m.group(1)) if m.group(1) else 1)
    return [Fraction(c.get(k, 0), D) for k in range(max(c) + 1)]


def main():
    Z = C.zerodim(12)
    zo = C.load("zerodim_on", os.path.join(C.ITOOLS, "lists", "zerodim_on.py"))
    ok = True
    # (a) full O(N) polynomials on the Nickel lists, with the integrator's weights
    vl = C.load("verify_lists", os.path.join(C.ITOOLS, "lists", "verify_lists.py"))
    vl.zerodim_on.series = lambda K: Z          # the cached exact series (a superset of every order asked)
    na, bad_a, lc = 0, [], {}
    print("(a) full O(N) polynomials with the integrator's weights (listcheck + verify_lists.py sumrule):")
    for f in sorted(os.listdir(C.LISTS), key=lambda n: (n[:6], len(n), n)):
        k = family(f)
        if not k:
            continue
        L, fam = k
        E = 4 if fam == "c4" else 2
        V = L + 1 if E == 4 else L
        if V > 9:
            print("    %s: %d vertices, 3^%d pairings per graph, not enumerated; the polynomial in (b)" % (f, V, V))
            continue
        rows = [l for l in C.listcheck_rows(os.path.join(C.LISTS, f), V).splitlines() if not l.startswith("#")]
        lc[f] = [l.split("\t")[13] for l in rows]
        with tempfile.NamedTemporaryFile("w", suffix=".check", delete=False) as tf:
            tf.write("".join(l + "\n" for l in rows))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            st = vl.cmd_sumrule(fam, L, os.path.join(C.LISTS, f), tf.name)
        os.remove(tf.name)
        out = buf.getvalue()
        eq = re.search(r"zero-dimensional c_L\(N\) = (.*) -> (EQUAL|DIFFERENT)", out)
        dev = re.search(r"max relative deviation (\S+)", out)
        print("    %-30s %-3s L=%d: %s c_L(N) = %s; double-precision weights within %s" % (f, fam, L, eq.group(2) if eq else "?", eq.group(1) if eq else "?",
                                                                                     dev.group(1) if dev else "?"))
        na += 1
        if st != 0 or not eq or eq.group(2) != "EQUAL":
            bad_a.append(f)
            print("      " + out.replace("\n", "\n      "))
    ok = ok and not bad_a
    # (b) full O(N) polynomials and N = 1 sums, exactly, on every list of both directories
    nb, bad_b, quoted, polys, npg, ngr = 0, [], {}, {}, 0, 0
    print("(b) full O(N) polynomials and N = 1 sums, exact (graphscan |Aut G| and pairing counts):")
    for d in (C.GAU, C.LISTS):
        for f in sorted(os.listdir(d), key=lambda n: (len(n), n)):
            k = family(f)
            if not k:
                continue
            L, fam = k
            path = os.path.join(d, f)
            r = C.scan(path)
            poly = []
            for E, V, a, n, cs in r["on"]:
                w = Fraction(factorial(E), a * 3 ** V)
                poly = zo.padd(poly, [w * c for c in cs])
            counted = sum(n for E, V, a, n, cs in r["on"])
            tot = sum(Fraction(factorial(E) * n, a) for E, a, n in r["aut"])
            sane = r["errors"] == "0" and r["aut_overflow"] == "0" and r.get("on_failed") == "0" and counted == int(r["graphs"])
            eq_poly, eq_n1 = poly == Z[fam][L], tot == sum(Z[fam][L])
            same = None
            if f in lc:                  # graph by graph against listcheck's enumeration of the 3^V pairings
                same = C.on_rows(path) == lc[f]
                npg += 1
                ngr += len(lc[f])
            polys.setdefault((fam, L), []).append(poly)
            if (fam, L) in QUOTED:
                quoted[(fam, L)] = quoted.get((fam, L), True) and tot == QUOTED[(fam, L)] and sum(poly) == QUOTED[(fam, L)]
            nb += 1
            if not (sane and eq_poly and eq_n1 and same is not False):
                bad_b.append(f)
            print("    %-36s %-3s L=%2d: %8s graphs, sum S_G T_G(N) = %s %s; N = 1: sum S_G = %s %s%s"
                  % (C.rel(path), fam, L, r["graphs"], zo.fmt(poly), "EQUAL" if eq_poly else "DIFFERENT", tot, "EQUAL" if eq_n1 else "DIFFERENT",
                     "" if same is None else "; pairing counts = listcheck's on every graph" if same else "; pairing counts DIFFER from listcheck's"))
            if not sane:
                print("      errors %s, |Aut| overflows %s, pairing counts failed %s, counted %d of %s graphs"
                      % (r["errors"], r["aut_overflow"], r.get("on_failed"), counted, r["graphs"]))
    ok = ok and not bad_b and len(quoted) == 3 and all(quoted.values())
    print("    pairing counts equal to listcheck's 3^V enumeration graph by graph: %d lists, %d graphs" % (npg, ngr))
    print("    quoted in the paper: %s" % ", ".join("%s(%d) = %s %s" % (fam, L, v, "reproduced" if quoted.get((fam, L)) else "NOT reproduced")
                                               for (fam, L), v in sorted(QUOTED.items())))
    # (c) the rows of Table tab:sumrules
    bad_c = []
    for (fam, L), (num, D, n1) in sorted(TABLE.items(), key=lambda x: (x[0][0] != "c4", x[0][1])):
        p = table_poly(num, D)
        good = (p == Z[fam][L] and Fraction(n1) == sum(p) and bool(polys.get((fam, L))) and all(q == p for q in polys[(fam, L)]))
        if not good:
            bad_c.append("%s L=%d" % (fam, L))
    print("(c) Table tab:sumrules, %d rows (Gamma(4) L = 1-10, Gamma(2) L = 2-10): numerator/D and N = 1 value EQUAL to the series and to every "
          "list of (b) for %d%s" % (len(TABLE), len(TABLE) - len(bad_c), "" if not bad_c else "; DIFFERENT: " + ", ".join(bad_c)))
    ok = ok and not bad_c
    # (d) the house checkers through seven loops
    tools = os.path.join(C.GRAPHS, "tools")
    st, so, se = C.python(os.path.join(tools, "gamma4_n1.py"), *[os.path.join(C.GAU, "%d-loop-4" % L) for L in range(1, 8)])
    n4 = so.count("-> EQUAL")
    print("(d) gamma4_n1.py, 1-7-loop-4: %d/7 EQUAL" % n4)
    with tempfile.TemporaryDirectory() as td:
        for L in range(1, 8):
            os.symlink(os.path.join(C.GAU, "%d-loop-2" % L), os.path.join(td, "%d-loop-2" % L))
        st2, so2, se2 = C.python(os.path.join(tools, "sumrule_check.py"), td, "gau", 6)
    n2 = len(re.findall(r"ratio 1( |$)", so2, re.M))
    nth = so2.count("-> EQUAL")
    print("(d) sumrule_check.py, 1-7-loop-2: sum S_G = exact coefficient (ratio 1) for %d/7; Gamma(2,1) O(N) polynomial = thesis p. 107 at %d/2 orders" % (n2, nth))
    for line in so2.splitlines():
        if "thesis" in line:
            print("    " + line.split(" | ", 1)[1])
    ok = ok and st == 0 and st2 == 0 and n4 == 7 and n2 == 7 and nth == 2
    C.verdict("G2", ok, "exact O(N) polynomials on %d lists through ten loops (graphscan; its pairing counts = listcheck's on %d lists, "
              "whose integrator weights give them on %d Nickel lists); Table tab:sumrules's %d rows EQUAL; at N = 1 Sigma_4,9 = 2245562203/4608, "
              "Sigma_4,10 = 4057808535/1024, Sigma_2,10 = 506798407/41472; graphs/tools checkers to 7 loops %s"
              % (nb, npg, na, len(TABLE), "EQUAL" if n4 == 7 and n2 == 7 and nth == 2 else "NOT all EQUAL"))


if __name__ == "__main__":
    main()
