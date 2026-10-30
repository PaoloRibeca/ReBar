#!/usr/bin/env python3
# N7_replicas.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N7: the replica ensembles at six and seven loops scatter as their quoted errors say (Sec. III A, "Error bars").
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
"""N7 -- the replica ensembles at six and seven loops scatter as their quoted errors say.

Claim (paper, Sec. III A, results.tex, the paragraph "Error bars"): eight independent replicas of four to eight minutes
of sampling on one GPU of each of B6, H6, C6, B7, H7 and C7, with medians of 1e7 to 1e9 samples per class, scatter as their quoted
errors say: for the order sums chi^2 = 31.6 for 42 degrees of freedom (p = 0.88); class by class, over the 29 666
degrees of freedom of the classes sampled in every replica, chi^2/dof = 1.003 (p = 0.36); the classes that carry half
of the variance of each order give chi^2/dof between 0.93 and 1.20.

Inputs: results/checks/replicas/sums.tsv (order sum and error of each of the 48 replicas of 18 October 2026) and
classes.tsv (per replica and class at N = 1: n, contribution and its error), extracted from the 48 run files, which
are kept whole in the ledger's run archive; paper/sections/results.tex.
What runs, per order: (a) the chi^2 of the eight order sums about their inverse-variance mean (7 dof), summed over
the six orders; (b) for every class sampled in all eight replicas, the chi^2 of its eight contributions about their
inverse-variance mean (7 dof), summed; (c) the classes sorted by the variance of their pooled contribution, largest
first, until they hold half of the order's variance: their chi^2/dof; (d) the median samples per class per replica.
PASS: every number quoted in the paragraph reproduced to its printed digits.
"""
import math, os, re, statistics, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the replica ensembles at six and seven loops scatter as their quoted errors say (order sums and class by class)"
QUICK = True
RUN = os.path.join(C.RESULTS, "replicas")
ORDERS = ("b6", "h6", "c6", "b7", "h7", "c7")


def gammq(a, x):
    """upper regularized incomplete gamma Q(a, x)"""
    if x <= 0:
        return 1.0
    lg = math.lgamma(a)
    if x < a + 1:
        s = t = 1.0 / a; ap = a
        for _ in range(100000):
            ap += 1; t *= x / ap; s += t
            if abs(t) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - lg)
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 100000):
        an = -i * (i - a); b += 2
        d = an * d + b; d = d if abs(d) > 1e-300 else 1e-300
        c = b + an / c; c = c if abs(c) > 1e-300 else 1e-300
        d = 1 / d; e = d * c; h *= e
        if abs(e - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - lg) * h


def scatter(vals, errs):
    w = [1 / e ** 2 for e in errs]
    m = sum(a * b for a, b in zip(w, vals)) / sum(w)
    return sum((a - m) ** 2 * b for a, b in zip(vals, w)), 1 / sum(w)


def table(name):
    with open(os.path.join(RUN, name)) as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, l.rstrip("\n").split("\t"))) for l in f if l.strip()]


def main():
    if not os.path.exists(os.path.join(RUN, "classes.tsv")):
        C.verdict("N7", False, "missing %s" % RUN)
    sums, classes = table("sums.tsv"), table("classes.tsv")
    cs = cd = ks = kd = 0.0
    half, med = {}, {}
    for o in ORDERS:
        s = [r for r in sums if r["order"] == o]
        c2, _ = scatter([float(r["sum"]) for r in s], [float(r["err"]) for r in s])
        cs += c2; cd += len(s) - 1
        reps = sorted({r["replica"] for r in s})
        per = {}
        for r in classes:
            if r["order"] == o:
                per.setdefault(r["nickel"], {})[r["replica"]] = (float(r["contrib"]), float(r["contrib_err"]), int(r["n"]))
        rows = []
        for nk, d in per.items():
            if len(d) < len(reps) or min(x[1] for x in d.values()) <= 0:
                continue
            c2, var = scatter([d[i][0] for i in reps], [d[i][1] for i in reps])
            rows.append((var, c2, len(reps) - 1))
        ks += sum(x[1] for x in rows); kd += sum(x[2] for x in rows)
        rows.sort(reverse=True)
        V = sum(x[0] for x in rows); acc = h2 = hd = 0.0
        for var, c2, dof in rows:
            acc += var; h2 += c2; hd += dof
            if acc >= 0.5 * V:
                break
        half[o] = h2 / hd
        med[o] = statistics.median(int(r["n"]) for r in classes if r["order"] == o)
        print("  %-3s %d replicas: order sums chi2 %.2f / %d; %d classes in every replica; half-variance chi2/dof %.3f; "
              "median samples per class %.2g" % (o, len(s), scatter([float(r["sum"]) for r in s], [float(r["err"]) for r in s])[0],
                                                 len(s) - 1, len(rows), half[o], med[o]))
    got = {"sums_chi2": "%.1f" % cs, "sums_dof": "%d" % cd, "sums_p": "%.2f" % gammq(cd / 2, cs / 2),
           "class_dof": "%d" % kd, "class_ratio": "%.3f" % (ks / kd), "class_p": "%.2f" % gammq(kd / 2, ks / 2),
           "half_lo": "%.2f" % min(half.values()), "half_hi": "%.2f" % max(half.values()),
           "med_lo": "%d" % math.floor(math.log10(min(med.values()))), "med_hi": "%d" % math.ceil(math.log10(max(med.values())))}
    print("(a) order sums: chi2 %s for %s dof, p = %s" % (got["sums_chi2"], got["sums_dof"], got["sums_p"]))
    print("(b) classes in every replica: chi2/dof %s over %s dof, p = %s" % (got["class_ratio"], got["class_dof"], got["class_p"]))
    print("(c) half-variance classes: chi2/dof between %s and %s" % (got["half_lo"], got["half_hi"]))
    print("(d) median samples per class: 10^%s to 10^%s" % (got["med_lo"], got["med_hi"]))
    with open(os.path.join(C.PAPER, "sections", "results.tex")) as fh:
        text = " ".join(fh.read().split())
    pat = (r"medians of \$10\^\{(\d+)\}\$ to \$10\^\{(\d+)\}\$ samples per class.*?for the order sums \$\\chi\^2=([\d.]+)\$ for "
           r"(\d+) degrees of freedom \(\$p=([\d.]+)\$\), and class by class, over the ([\d\\,]+) degrees of freedom of the "
           r"classes sampled in every replica, \$\\chi\^2/\{\\rm dof\}=([\d.]+)\$ \(\$p=([\d.]+)\$\); the classes that carry "
           r"half of the variance of each order, the heavy-tailed ones among them, give \$\\chi\^2/\{\\rm dof\}\$ between "
           r"([\d.]+) and ([\d.]+)(?:~\\check\{N7\})?\.")
    m = re.search(pat, text)
    if not m:
        C.verdict("N7", False, "the replica paragraph of results.tex was not found in the expected wording")
    quoted = dict(zip(("med_lo", "med_hi", "sums_chi2", "sums_dof", "sums_p", "class_dof", "class_ratio", "class_p",
                       "half_lo", "half_hi"), [x.replace("\\,", "") for x in m.groups()]))
    bad = ["%s: quoted %s, recomputed %s" % (k, quoted[k], got[k]) for k in quoted if quoted[k] != got[k]]
    print("    text (results.tex): %d numbers quoted, %d differ" % (len(quoted), len(bad)))
    for b in bad:
        print("    " + b)
    C.verdict("N7", not bad, "48 replicas: order sums chi2 %s/%s (p %s), classes chi2/dof %s over %s dof (p %s), half-variance "
              "classes %s-%s, as quoted" % (got["sums_chi2"], got["sums_dof"], got["sums_p"], got["class_ratio"], got["class_dof"],
                                           got["class_p"], got["half_lo"], got["half_hi"]))


if __name__ == "__main__":
    main()
