#!/usr/bin/env python3
# checks.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Checks of the series that do not need a reference value: the large-order ratio test of the beta coefficients
# (successive ratios against the asymptotic form k! (-a)^k k^b), and a hook that runs the zero-dimensional sum rules
# of graphs/tools on the graph list of every order of the ledger.
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
"""checks.py -- large-order ratio test and zero-dimensional sum rules.

usage: checks.py [--coefficients FILE] [--json OUT] [--quiet]     large-order test (default input generated/coefficients.json)
       checks.py --sumrules [ORDER ...]                            run the graphs/tools checkers on the orders' lists

Large order.  beta_k (coefficient of ubar^k) ~ c (-a)^k k! k^b for large k, so that
    R_k = |beta_(k+1) / beta_k| / (a (k + b + 1))  ->  1 + O(1/k^2)
(the convention of the paper's Table largeorder), with a = 0.14777422 * 9/(N+8) (the instanton action does not depend
on N in g, and ubar = (N+8) u/6; a = 0.14777422 at N = 1) and b = 3 + N/2: the instanton action A = 113.3835 of
d = 3 and a = 48 pi/((N+8) A) = 0.147774232 * 9/(N+8), Zinn-Justin, Quantum Field Theory and Critical Phenomena,
3rd ed., Eq. (28.73) (check S3 compares
them with the paper's).  The error of R_k combines the errors of the two coefficients in quadrature (they share
inputs; the correlation is ignored).

Sum rules.  The zero-dimensional checks live in graphs/tools: gamma4_n1.py (4-point lists, N = 1), sumrule_check.py
(2-point lists with self-loops, with the Gamma(2,1) O(N) rule at 5 and 6 loops) and zerodim_tf.py (self-loop-free
2-point lists).  They run on the GAU-format lists graphs/lists/<L>-loop-<2|4> (9 loops: 9-loop-2-ci), from which
the integrator's Nickel lists are made; 'make -C graphs check' runs them with the regeneration.  8 loops take minutes.
"""
import json, math, os, re, subprocess, sys, tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebar_ledger as RL  # noqa: E402

A1 = 0.14777422          # a at N = 1: 48 pi/(9 A) with A = 113.3835 (Zinn-Justin, 3rd ed., Eq. (28.73): 0.147774232)
GRAPHS = os.path.join(RL.ROOT, "graphs")


def a_of(N):
    return A1 * 9.0 / (N + 8)


def b_of(N):
    return 3 + N / 2.0


def ratio_test(coef, N=1):
    """rows (k, |beta_(k+1)/beta_k|, err, asymptote, R_k, err) for every k with both coefficients"""
    beta = coef["coefficients"]["beta"]
    rows = []
    for k in range(2, max(int(n) for n in beta)):
        if str(k + 1) not in beta:
            continue
        v1, e1 = beta[str(k + 1)]["N"][str(N)]
        v0, e0 = beta[str(k)]["N"][str(N)]
        r = abs(v1 / v0)
        er = r * math.sqrt((e1 / v1) ** 2 + (e0 / v0) ** 2)
        asym = a_of(N) * (k + b_of(N) + 1)
        rows.append({"k": k, "raw": r, "raw_err": er, "asym": asym, "R": r / asym, "R_err": er / asym,
                     "provisional": beta[str(k + 1)].get("provisional", False)})
    return rows


SUMRULE_TOOLS = {"4": "gamma4_n1.py", "2": "sumrule_check.py", "2-ci": "zerodim_tf.py"}


def sumrule_list(kind, L):
    """(GAU list path, checker) of an order, or None"""
    gau = os.path.join(GRAPHS, "lists")
    if kind == "b":
        p = os.path.join(gau, "%d-loop-4" % L)
        return (p, "4") if os.path.exists(p) else None
    p = os.path.join(gau, "%d-loop-2" % L)
    if os.path.exists(p):
        return p, "2"
    p = os.path.join(gau, "%d-loop-2-ci" % L)
    return (p, "2-ci") if os.path.exists(p) else None


def run_sumrules(names):
    orders = RL.load_orders(RL.LEDGER)["orders"]
    names = names or [o for o, e in orders.items() if e.get("contributions")]
    lists, bad = {}, 0
    for o in names:
        kind, L = RL.parse_order(o)
        s = sumrule_list(kind, L)
        if not s:
            print("%-4s no GAU list in graphs/lists" % o)
            continue
        lists.setdefault(s, []).append(o)     # h and c share the 2-point list
    for (path, typ), os_ in sorted(lists.items()):
        tool = os.path.join(GRAPHS, "tools", SUMRULE_TOOLS[typ])
        if typ == "2":                          # sumrule_check.py reads every L-loop-2 of a directory
            with tempfile.TemporaryDirectory() as d:
                os.symlink(path, os.path.join(d, os.path.basename(path)))
                r = subprocess.run([sys.executable, "-I", tool, d, "list", "6"], stdout=subprocess.PIPE, universal_newlines=True)
        else:
            r = subprocess.run([sys.executable, "-I", tool, path], stdout=subprocess.PIPE, universal_newlines=True)
        if typ == "2":                          # as graphs/scripts/check.sh: 'ratio 1', and EQUAL where the thesis has the rule
            ok = re.search(r"ratio 1( |$)", r.stdout) is not None and ("thesis" not in r.stdout or "-> EQUAL" in r.stdout)
        else:
            ok = "-> EQUAL" in r.stdout
        ok = ok and r.returncode == 0
        bad += not ok
        print("%-8s %-14s %s  %s" % ("/".join(os_), os.path.basename(path), "PASS" if ok else "FAIL", r.stdout.strip()))
    return 1 if bad else 0


def main(argv):
    if len(argv) > 1 and argv[1] == "--sumrules":
        return run_sumrules(argv[2:])
    cf, js, quiet = os.path.join(RL.HERE, "generated", "coefficients.json"), None, False
    it = iter(argv[1:])
    for a in it:
        if a == "--coefficients":
            cf = next(it)
        elif a == "--json":
            js = next(it)
        elif a == "--quiet":
            quiet = True
        else:
            print(__doc__)
            return 2
    coef = RL.load_json(cf)
    out = {"a_N1": A1, "a": "0.14777422 * 9/(N+8)", "b": "3 + N/2",
           "source": "A = 113.3835, a = 48 pi/((N+8) A) = 0.147774232 * 9/(N+8), b = 3 + N/2: Zinn-Justin, 3rd ed., "
           "Eq. (28.73)", "N": {}}
    for N in coef["N"]:
        out["N"][str(N)] = ratio_test(coef, N)
    if js:
        RL.write_if_changed(js, RL.dump_json(out))
    if not quiet:
        print("Large-order test of beta (a = %s, b = %s; %s)" % (out["a"], out["b"], out["source"]))
        for N, rows in out["N"].items():
            print("  N=%s  " % N + "  ".join("k=%d %.4f/%.4f=%s" % (r["k"], r["raw"], r["asym"], RL.fmt_ve(r["R"], r["R_err"]) if r["R_err"] else "%.4f" % r["R"]) for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
