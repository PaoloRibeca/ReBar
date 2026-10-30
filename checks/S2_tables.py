#!/usr/bin/env python3
# S2_tables.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check S2: every number of the paper's generated tables is regenerated from the ledger, byte for byte.
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
"""S2 -- every number in the paper's generated tables is regenerated from the ledger.

Claim (paper, Appendix "Reproducibility"; series/README.md "The paper's numbers"): the macros and table bodies that
the paper inputs from paper/generated/ (values.tex, newsums.tex, validation.tex, largeorder.tex, largeorder-fit.tex, costhist.tex, exponents.tex, newpoly.tex) are written by
series/ from the ledger and nothing else, so that regenerating them reproduces them exactly.

Inputs: the ledger (orders.json and the per-class statistics, read through series/rebar_ledger.py),
integrator/tools/refs.py; series/coefficients.py, checks.py, tables.py, largeorder.py, costfig.py, resum.py, polynomials.py; paper/generated/*.tex
(as distributed); paper/sections/*.tex and paper/main.tex.
What runs: the chain of `make -C series` (coefficients.py --json, checks.py --json, tables.py, largeorder.py
--tex, costfig.py --tex, resum.py --tex, polynomials.py --tex), into a temporary directory, without touching the repository.
PASS: the regenerated .tex files are byte-identical to paper/generated/, and every one of them is \\input by
the paper.  Informational: series/generated/*.json and the snapshot results/coefficients/<date>/ against the
regenerated JSON (numbers to 1e-12 relative; the propagated errors are sums whose order follows a set's iteration
order, so their last bits can change between runs).
"""
import glob, json, os, re, shutil, sys, tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "every number in the paper's generated tables is regenerated from the ledger (values.tex, newsums.tex, validation.tex, largeorder.tex, largeorder-fit.tex, costhist.tex, exponents.tex, newpoly.tex)"
QUICK = True
TEX = ("values.tex", "newsums.tex", "validation.tex", "largeorder.tex", "largeorder-fit.tex", "costhist.tex", "exponents.tex", "newpoly.tex")


def numbers(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numbers(v, path + "/" + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numbers(v, "%s[%d]" % (path, i))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)
    else:
        yield path, x


def jsondiff(a, b):
    """(max relative difference of the numbers, number of non-numeric or structural differences)"""
    na, nb = dict(numbers(a)), dict(numbers(b))
    worst, other = 0.0, 0
    for k in set(na) | set(nb):
        x, y = na.get(k), nb.get(k)
        if isinstance(x, float) and isinstance(y, float):
            if x != y:
                worst = max(worst, abs(x - y) / max(abs(x), abs(y)))
        elif x != y:
            other += 1
    return worst, other


def main():
    d = tempfile.mkdtemp(prefix="rebar-S2-")
    try:
        steps = [(os.path.join(C.SERIES, "coefficients.py"), "--json", os.path.join(d, "coefficients.json"), "--quiet"),
                 (os.path.join(C.SERIES, "checks.py"), "--coefficients", os.path.join(d, "coefficients.json"), "--json", os.path.join(d, "checks.json"), "--quiet"),
                 (os.path.join(C.SERIES, "tables.py"), "--coefficients", os.path.join(d, "coefficients.json"), "--checks", os.path.join(d, "checks.json"), "--out", d),
                 (os.path.join(C.SERIES, "largeorder.py"), "--N", "0,1,2,3,4", "--tex", os.path.join(d, "largeorder.tex"),
                  "--fit-tex", os.path.join(d, "largeorder-fit.tex")),
                 (os.path.join(C.SERIES, "costfig.py"), "--tex", os.path.join(d, "costhist.tex")),
                 (os.path.join(C.SERIES, "resum.py"), "--json", os.path.join(d, "coefficients.json"), "--tex", os.path.join(d, "exponents.tex")),
                 (os.path.join(C.SERIES, "polynomials.py"), "--tex", os.path.join(d, "newpoly.tex"), "--json", os.path.join(d, "polynomials.json"), "--quiet")]
        for s in steps:
            st, so, se = C.python(*s)
            print("  %s: exit status %d" % (" ".join([os.path.basename(s[0])] + [os.path.basename(x) if os.sep in x else x for x in s[1:]]), st))
            if st:
                print(se)
                C.verdict("S2", False, "%s failed" % os.path.basename(s[0]))
        sources = " ".join(open(p).read() for p in glob.glob(os.path.join(C.PAPER, "sections", "*.tex")) + [os.path.join(C.PAPER, "main.tex")])
        same, used = [], []
        for t in TEX:
            new, old = open(os.path.join(d, t), "rb").read(), open(os.path.join(C.PAPER, "generated", t), "rb").read()
            inp = re.search(r"\\input\{generated/%s(\.tex)?\}" % re.escape(t[:-4]), sources) is not None
            nnum = len(re.findall(rb"-?\d+\.\d+", new))
            print("  paper/generated/%-15s %6d bytes, %4d decimal numbers: %s; \\input by the paper: %s"
                  % (t, len(old), nnum, "byte-identical" if new == old else "DIFFERENT", "yes" if inp else "NO"))
            same.append(new == old)
            used.append(inp)
        ok = all(same) and all(used)
        for name in ("coefficients.json", "checks.json", "polynomials.json"):
            new = json.load(open(os.path.join(d, name)))
            for ref in [os.path.join(C.SERIES, "generated", name)] + sorted(glob.glob(os.path.join(C.ROOT, "results", "coefficients", "*", name))):
                if os.path.exists(ref):
                    w, o = jsondiff(new, json.load(open(ref)))
                    print("  (informational) %-45s vs regenerated: max relative difference %.1e, %d other differences" % (C.rel(ref), w, o))
    finally:
        shutil.rmtree(d)
    C.verdict("S2", ok, "%d of %d generated files byte-identical after regeneration from the ledger, %d of %d \\input by the paper"
              % (sum(same), len(TEX), sum(used), len(TEX)))


if __name__ == "__main__":
    main()
