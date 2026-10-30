#!/usr/bin/env python3
# N8_fourloop.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N8: the round-off example of H4, its class with a sunset against the other three over its ten runs, gives the
# numbers quoted (Sec. III A, "Round-off in the subtractions").
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
"""N8 -- the round-off example of H4: the class with a sunset subgraph against the other three classes.

Claim (paper, Sec. III A, results.tex, the paragraph "Round-off in the subtractions"): among the four classes of H4,
the only one with a two-point part, a sunset, has a single-sample round-off scale of up to 6e-4 of its per-sample
standard deviation, against about 1e-13 in the other three (the ten runs of H4 of 14 October 2026: the two
first runs, of the validation and production stages, and eight replicas).

Inputs: results/checks/fourloop/fourloop-sums.tsv (per run of 14 October 2026: role, seed, order sum and error,
samples, reference, node) and fourloop-classes.tsv (per run and class at N = 1: n, contribution and its error, kurtosis,
largest weight over the mean, largest and rms round-off scale over the per-sample standard deviation), extracted from
the 18 run files, which are kept whole in the ledger's run archive; paper/sections/results.tex.
What runs: (b) for H4, every class over the ten runs: the inverse-variance mean of its contribution, the chi^2 about
it (9 dof), the largest single-sample round-off scale; the class with a sunset subgraph (two vertices joined by three
lines), from its Nickel index; (c) when ledger-archive/ is present, every field of the two tables against the archived
run JSON of its contribution, the role of every run (V, P, replica k) against the output path it was written to, and
the node of every run (a label in the table) against the runs': one node per label.
PASS: the numbers of the sentence reproduced to their printed digits, every class of H4 consistent over the ten runs
(p > 1e-3), and, when the archive is present, the tables identical to the runs.
"""
import math, os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the round-off example of H4: the class with a sunset against the other three (ten runs of H4)"
QUICK = True
RUN = os.path.join(C.RESULTS, "fourloop")
WORDS = {3: "three", 4: "four", 8: "eight", 9: "nine", 10: "ten"}


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


def word(n):
    return WORDS.get(n, str(n))


def pool(vals, errs):
    """inverse-variance mean, its error, and the chi^2 of the values about it"""
    w = [1 / e ** 2 for e in errs]
    m = sum(a * b for a, b in zip(w, vals)) / sum(w)
    return m, sum(w) ** -0.5, sum((a - m) ** 2 * b for a, b in zip(vals, w))


def table(name):
    with open(os.path.join(RUN, name)) as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, l.rstrip("\n").split("\t"))) for l in f if l.strip()]


def fmt_err(v, e, nd=2):
    """v(e) with nd digits of error, the value rounded to the same place (an error that rounds up to 10^nd loses one)"""
    k = int(math.floor(math.log10(e))) - nd + 1
    if int(round(e / 10.0 ** k)) >= 10 ** nd:
        k += 1
    return "%.*f(%d)" % (max(0, -k), v, int(round(e / 10.0 ** k)))


def like(q, x):
    """x printed as the quoted decimal q is: same number of decimals, explicit sign if q has one"""
    dec = len(q.split(".")[1]) if "." in q else 0
    return ("%+.*f" if q[0] in "+-" else "%.*f") % (dec, x)


def sci_like(mant, exp, x):
    """x as m x 10^e with as many significant digits as the quoted mantissa (none quoted: one digit, mantissa 1)"""
    m, e = ("%.*e" % (len((mant or "1").replace(".", "")) - 1, x)).split("e")
    return m, int(e)


def sunset(nickel):
    """True when two vertices of the graph are joined by three lines (Nickel index: the neighbours of every vertex)"""
    lines = {}
    for v, part in enumerate(nickel.split("|")):
        for ch in part:
            if ch != "e":
                lines[(v, int(ch, 36))] = lines.get((v, int(ch, 36)), 0) + 1
    return max(lines.values()) >= 3


def role_of(R, order):
    """V, P or replica k from the output path the run was written to (run_all.sh: val_, prod_, replicas_<order>/r<k>)"""
    p = R.get("args", {}).get("out", "").replace("\\", "/").split("/")
    if p[-1] == "val_%s.json" % order:
        return ("V", "-")
    if p[-1] == "prod_%s.json" % order:
        return ("P", "-")
    if len(p) > 1 and p[-2] == "replicas_%s" % order and re.match(r"r\d+\.json$", p[-1]):
        return ("replica", p[-1][1:-5])
    return (None, None)


def against_archive(sums, classes):
    """(runs compared, list of differences); runs compared is 0 when the archive is absent"""
    RL = C.series_module("rebar_ledger")
    man = RL.load_orders(C.LEDGER)["orders"]
    done, bad, nodes = 0, [], {}
    for s in sums:
        o, cid = s["order"], s["id"]
        recs = [r for r in man[o].get("contributions", []) if r["id"] == cid]
        if not recs or str(recs[0].get("seed", s["seed"])) != s["seed"]:
            bad.append("%s: not a contribution of %s with seed %s in the manifest" % (cid, o, s["seed"]))
            continue
        try:
            R = RL.load_contribution(C.LEDGER, o, cid)
        except Exception:
            continue
        done += 1
        od = R["order"]
        mine = {"seed": str(R["seed"]), "sum": repr(od["sum"]), "err": repr(od["err"]), "samples": str(od["samples"]),
                "core_s": repr(od["core_s"]), "ref": repr(od["ref"]), "ref_err": repr(od["ref_err"]), "utc": R["utc"]}
        nodes.setdefault(s["host"], set()).add(str(R.get("host", "")).split(".")[0])
        bad += ["%s: %s %s in the table, %s in the run" % (cid, k, s[k], v) for k, v in mine.items() if s[k] != v]
        if role_of(R, o) != (s["role"], s["replica"]) or not R.get("final") or R.get("N") != 1:
            bad.append("%s: role %s %s in the table, %s in the run (final %s, N %s)" % (cid, s["role"], s["replica"],
                                                                                       role_of(R, o), R.get("final"), R.get("N")))
        rows = {c["nickel"]: c for c in classes if (c["order"], c["role"], c["replica"]) == (o, s["role"], s["replica"])}
        if set(rows) != {c["nickel"] for c in R["classes"]}:
            bad.append("%s: the table and the run have different classes" % cid)
            continue
        for c in R["classes"]:
            t = rows[c["nickel"]]
            want = {"seed": str(R["seed"]), "n": str(c["n"])}
            want.update((k, repr(c[k])) for k in ("contrib", "contrib_err", "kurtosis", "maxw_over_absmean",
                                                  "roundoff_max_over_sd", "roundoff_rms_over_sd"))
            bad += ["%s %s: %s %s in the table, %s in the run" % (cid, c["nickel"], k, t[k], v) for k, v in want.items() if t[k] != v]
    # the table names the node of a run by a label (gpu-node-N): one node of the runs per label, one label per node
    bad += ["node %s of the table: %d different nodes in the runs" % (k, len(v))
            for k, v in sorted(nodes.items()) if len(v) != 1]
    labels = {}
    for k, v in nodes.items():
        for h in v:
            labels.setdefault(h, set()).add(k)
    bad += ["one node of the runs under the labels %s of the table" % ", ".join(sorted(v))
            for v in labels.values() if len(v) > 1]
    return done, bad


class Text:
    """the paper's text, whitespace-normalised and without \\check{..} citations; find() records what is missing"""

    def __init__(self, path):
        with open(path) as fh:
            raw = " ".join(fh.read().split())
        a, b = raw.find("\\paragraph{Replica ensembles.}"), (raw.find("and in every run.") if "and in every run." in raw else raw.find("and in each of the ten runs."))
        self.cites = re.findall(r"\\check\{([A-Z]\d+)\}", raw[a:b]) if 0 <= a < b else []   # those of the three passages
        self.text = re.sub(r"~?\\check\{[A-Z]\d+\}", "", raw)
        self.missing = []

    def find(self, name, pat, within=None):
        m = re.search(pat, self.text if within is None else within)
        if not m:
            self.missing.append(name)
        return m


def main():
    for f in ("fourloop-sums.tsv", "fourloop-classes.tsv"):
        if not os.path.exists(os.path.join(RUN, f)):
            C.verdict("N8", False, "missing %s" % os.path.join(RUN, f))
    sums, classes = table("fourloop-sums.tsv"), table("fourloop-classes.tsv")

    # ---- (b) H4, the ten runs class by class
    ten = [r for r in sums if r["order"] == "h4"]
    roles = [(r["role"], r["replica"]) for r in ten]
    ref, ref_err = float(ten[0]["ref"]), float(ten[0]["ref_err"])
    hc = {}
    for r in classes:
        if r["order"] == "h4":
            hc.setdefault(r["nickel"], []).append(r)
    order_nk = [r["nickel"] for r in classes if r["order"] == "h4" and r["role"] == "V"]
    tot = tote2 = 0.0
    cls = {}
    for nk in order_nk:
        x = hc[nk]
        m, me, c2 = pool([float(r["contrib"]) for r in x], [float(r["contrib_err"]) for r in x])
        cls[nk] = dict(runs=len(x), m=m, e=me, chi2=c2, p=gammq((len(x) - 1) / 2.0, c2 / 2.0),
                       rmax=max(float(r["roundoff_max_over_sd"]) for r in x), rmin=min(float(r["roundoff_max_over_sd"]) for r in x),
                       rrms=max(float(r["roundoff_rms_over_sd"]) for r in x), kurt=[float(r["kurtosis"]) for r in x],
                       maxw=[float(r["maxw_over_absmean"]) for r in x], sunset=sunset(nk))
        tot += m; tote2 += me * me
        k = cls[nk]
        print("(b) h4 %-16s %2d runs: %s x 1e-4, chi2 %.2f / %d (p %.3f); kurtosis %.2f-%.2f, largest weight %.1f-%.1f x mean; "
              "round-off largest %.2g-%.2g, rms <= %.2g; sunset %s" % (
                  nk, k["runs"], fmt_err(m * 1e4, me * 1e4), c2, len(x) - 1, k["p"], min(k["kurt"]), max(k["kurt"]),
                  min(k["maxw"]), max(k["maxw"]), k["rmin"], k["rmax"], k["rrms"], "yes" if k["sunset"] else "no"))
    tote = math.sqrt(tote2)
    hdev = (tot - ref) / math.hypot(tote, ref_err)
    print("    H4 over the ten runs (%s): %s = %.10g +- %.3g, %+.3f sigma from the reference %.10g (relative uncertainty %.1e)" % (
        ", ".join("%s%s" % (a, "" if b == "-" else b) for a, b in roles), fmt_err(tot, tote), tot, tote, hdev, ref, ref_err / ref))
    allk = [x for k in cls.values() for x in k["kurt"]]
    allw = [x for k in cls.values() for x in k["maxw"]]
    sun = [nk for nk, k in cls.items() if k["sunset"]]
    others = max(k["rmax"] for nk, k in cls.items() if not k["sunset"])
    rrms = max(k["rrms"] for k in cls.values())
    print("    over the classes and the ten runs: kurtosis %.3f-%.3f, largest weight %.3f-%.3f times the class mean, "
          "rms round-off at most %.3g" % (min(allk), max(allk), min(allw), max(allw), rrms))
    for nk, k in cls.items():
        print("      %-16s largest weight per run: %s" % (nk, " ".join("%.1f" % w for w in k["maxw"])))
    print("    sunset class(es): %s, largest round-off %.3g-%.3g; the other classes at most %.3g" % (
        ", ".join(sun), cls[sun[0]]["rmin"] if sun else 0, cls[sun[0]]["rmax"] if sun else 0, others))

    # ---- the paper
    T = Text(os.path.join(C.PAPER, "sections", "results.tex"))
    cmp = []   # (name, quoted, recomputed)
    num = r"\$(?:([\d.]+)\\times)?10\^\{(-\d+)\}\$"
    m = T.find("round-off example", r"Among the (\w+) classes of \$H_4\$, for instance, the only one with a two-point part, a "
                                    r"sunset, has a single-sample round-off scale of up to " + num + r" of its per-sample standard "
                                    r"deviation, against about " + num + r" in the other (\w+)\.")
    if m:
        cmp.append(("text: the classes of H4", m.group(1), word(len(cls))))
        cmp.append(("text: exactly one class with a sunset", "yes", "yes" if len(sun) == 1 else "no (%d)" % len(sun)))
        if len(sun) == 1:
            sm, se = sci_like(m.group(2), int(m.group(3)), cls[sun[0]]["rmax"])
            cmp.append(("text: round-off, sunset class, highest", "%s\\times10^{%s}" % (m.group(2) or "1", m.group(3)),
                        "%s\\times10^{%d}" % (sm, se)))
        cmp.append(("text: round-off, the other classes, order of magnitude", "10^{%s}" % m.group(5),
                    "10^{%d}" % int(math.floor(math.log10(others)))))
        cmp.append(("text: the other classes", m.group(6), word(len(cls) - len(sun))))

    # consistency of the classes, and the archive
    incons = ["%s chi2 %.1f/%d p %.2g" % (nk, k["chi2"], k["runs"] - 1, k["p"]) for nk, k in cls.items() if k["p"] <= 1e-3]
    done, abad = against_archive(sums, classes)
    if done:
        print("(c) %d of the %d runs compared with the ledger's run archive, field by field: %d difference(s)" % (done, len(sums), len(abad)))
        for b in abad[:20]:
            print("    " + b)
    else:
        print("(c) ledger-archive/ absent: the tables were not compared with the archived runs%s" % (
            " (%d manifest problem(s))" % len(abad) if abad else ""))
        for b in abad[:20]:
            print("    " + b)

    bad = [c for c in cmp if c[1] != c[2]]
    print("    text and table (results.tex): %d numbers quoted, %d differ" % (len(cmp), len(bad)))
    for name, q, r in cmp:
        print("    %-58s quoted %-16s recomputed %-16s%s" % (name, q, r, "" if q == r else "  DIFFERS"))
    for x in T.missing:
        print("    not found in results.tex in the expected wording: %s" % x)
    print("    the passage cites: %s" % (", ".join("\\check{%s}" % x for x in T.cites) or "no check"))
    ok = not bad and not T.missing and not incons and not abad
    C.verdict("N8", ok, ("H4 over ten runs: the sunset class up to %.2g of its per-sample sd, the other %d classes at most %.2g; "
                         "%d of %d quoted numbers reproduced%s%s%s%s") % (
        cls[sun[0]]["rmax"] if sun else float("nan"), len(cls) - len(sun), others, len(cmp) - len(bad), len(cmp),
        "" if not bad else " (differ: %s)" % "; ".join("%s quoted %s, recomputed %s" % c for c in bad),
        "" if not T.missing else "; not found: %s" % ", ".join(T.missing),
        "" if not incons else "; inconsistent classes: %s" % ", ".join(incons),
        "; tables = archived runs (%d)" % done if done and not abad else ("; %d difference(s) from the archive" % len(abad) if abad else "")))



if __name__ == "__main__":
    main()
