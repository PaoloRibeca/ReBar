#!/usr/bin/env python3
# tables.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Writes the paper's generated fragments from the ledger: paper/generated/values.tex (the LaTeX macros the text
# uses: per-order values, errors, class counts, contributions and samples; coefficients at N = 1 and N = 0..4;
# large-order ratios; the ledger date), newsums.tex (the table of new and improved per-order sums) and
# validation.tex (orders 2-7 against exact values and Phi4tools).
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
"""tables.py -- paper fragments from the ledger.

usage: tables.py [--coefficients FILE] [--checks FILE] [--out DIR]
       (defaults: generated/coefficients.json, generated/checks.json, ../paper/generated)

Macro contract (every name is always defined; "---" when the value is absent).  Per order X in B7 B8 B9 B10 H7 H8 H9
H10 C7 C8 C9 C10 (digits spelled out: \\valBeight, \\valHten, \\valCnine): \\valX value(error), \\relX relative error,
\\nclX classes, \\ncontribX contributions, \\samplesX samples.  Coefficients at N = 1 (ubar normalisation):
\\etaEight \\etaNine \\etaTen \\betaNine \\betaTen \\betaEleven \\nuEight \\nuNine \\nuTen \\etaPhiEight \\etaPhiNine \\etaPhiTen, and the same at N = 0..4 with the
suffix Nzero .. Nfour (\\betaNineNzero); the relative error at N = 1 with the suffix Rel (\\betaNineRel); eta also in units of 1e-3 with the suffix K (\\etaNineNzeroK), nu in
units of 1e-2 with the suffix C (\\nuNineNzeroC).  Large order: \\ratioEight \\ratioNine (|beta_(k+1)/beta_k| / a(k+b+1) at
N = 1, k = 8, 9), \\rawratioEight \\rawratioNine (|beta_(k+1)/beta_k|).  \\ledgerDate: date of the latest run in the
ledger.  Negative and scientific values carry their own $...$ (the macros are used in text mode)."""
import math
import os, os, sys, time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebar_ledger as RL  # noqa: E402

WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
         9: "nine", 10: "ten", 11: "eleven"}
ORDERS = ["b7", "b8", "b9", "b10", "h7", "h8", "h9", "h10", "c7", "c8", "c9", "c10"]
COEFS = [("eta", 8, "etaEight"), ("eta", 9, "etaNine"), ("beta", 9, "betaNine"), ("beta", 10, "betaTen"),
         ("nu", 8, "nuEight"), ("nu", 9, "nuNine"), ("eta_phi2", 8, "etaPhiEight"), ("eta_phi2", 9, "etaPhiNine"),
         ("eta", 10, "etaTen"), ("nu", 10, "nuTen"), ("eta_phi2", 10, "etaPhiTen"), ("beta", 11, "betaEleven")]
SUM = {"b": "B", "h": "H", "c": "C"}
GREEK = {"b": r"$\beta$", "h": r"$\eta$", "c": r"$\nu$"}
ABSENT = "---"


def cap(w):
    return w[0].upper() + w[1:]


def math_if(s):
    """wrap a number in $...$ when it needs math mode (sign, \\times)"""
    return "$%s$" % s if s.startswith("-") or "\\times" in s else s


def prov(s, p):
    return s   # no provisional marker is printed


def sci(x, nd=2):
    t = RL.fmt_sci(x, nd)
    return ABSENT if t is None else "%s\\times10^{%d}" % t


def thousands(n):
    s = "%d" % n
    return s if len(s) < 5 else ",".join([s[max(0, i - 3):i] for i in range(len(s), 0, -3)][::-1]).replace(",", "\\,")


def is_prov(e):
    return any(v.startswith("0.3.") for v in e.get("versions", []))


def latest_run_date(orders):
    utc = [c["utc"] for e in orders["orders"].values() for c in e.get("contributions", []) if c.get("utc")]
    if not utc:
        return ABSENT
    t = time.strptime(max(utc)[:10], "%Y-%m-%d")
    return "%d %s %d" % (t.tm_mday, time.strftime("%B", t), t.tm_year)


def values_tex(orders, coef, chk):
    L = ["% generated/values.tex -- numbers of the paper, as LaTeX macros. WRITTEN BY series/tables.py",
         "% (make -C ../series) from the ledger (ledger/orders.json) and series/generated/*.json: do not edit by hand.",
         "%"] + ["% " + x for x in __doc__.split("\n\n")[2].replace("\\\\", "\\").split("\n")] + [
         "%", "\\newcommand{\\ledgerDate}{%s}" % latest_run_date(orders), "%"]
    for o in ORDERS:
        kind, Lp = RL.parse_order(o)
        name = SUM[kind] + WORDS[Lp]
        e = orders["orders"].get(o, {})
        has = bool(e.get("contributions"))
        p = is_prov(e)
        val = prov(math_if(RL.fmt_ve(e["sum"], e["err"])), p) if has else ABSENT
        rel = prov("$%s$" % sci(e["rel_err"]), p) if has else ABSENT
        smp = prov("$%s$" % sci(e["samples"]), p) if has else ABSENT
        L += ["\\newcommand{\\val%s}{%s}" % (name, val), "\\newcommand{\\rel%s}{%s}" % (name, rel),
              "\\newcommand{\\ncl%s}{%s}" % (name, thousands(e["classes"]) if has else ABSENT),
              "\\newcommand{\\ncontrib%s}{%d}" % (name, len(e.get("contributions", []))),
              "\\newcommand{\\samples%s}{%s}" % (name, smp)]
        if o in ("b10", "h10", "c10"):
            L.append("%")
    for f, n, name in COEFS:
        d = coef["coefficients"].get(f, {}).get(str(n))
        src = "%% %s: %s" % (name, "absent" if not d else "ledger inputs " + (", ".join(d["ledger_inputs"]) or "none"))
        L.append(src)
        # relative uncertainty at N = 1, one digit, for the precision column of Table tab:sota: \\betaNineRel
        rel = d["N"].get("1") if d else None
        L.append("\\newcommand{\\%sRel}{%s}" % (name, "$%s$" % sci(rel[1] / abs(rel[0]), 1) if rel and rel[0] else ABSENT))
        for N, suffix in [(1, "")] + [(N, "N" + WORDS[N]) for N in range(5)]:
            if d and str(N) in d["N"]:
                v, er = d["N"][str(N)]
                s = prov(math_if(fmt_err(v, er)), d.get("provisional"))
            else:
                s = ABSENT
            L.append("\\newcommand{\\%s%s}{%s}" % (name, suffix, s))
            # the narrow tables quote eta in units of 1e-3 (tab:newcoeffeta: \\etaEightNoneK) and nu in units of
            # 1e-2 (tab:newcoeffnu: \\nuEightNoneC)
            for fs, scale, tag in (("eta", 1e3, "K"), ("nu", 1e2, "C")):
                if f == fs:
                    sk = prov(math_if(fmt_err(scale * v, scale * er)), d.get("provisional")) if s != ABSENT else ABSENT
                    L.append("\\newcommand{\\%s%s%s}{%s}" % (name, suffix, tag, sk))
    L += text_macros(orders, coef)
    L.append("%% Large order (a = %s, b = %s, %s)" % (chk["a"], chk["b"], chk["source"]))
    rows = {r["k"]: r for r in chk["N"].get("1", [])}
    for k in (8, 9):
        r = rows.get(k)
        L.append("\\newcommand{\\ratio%s}{%s}" % (cap(WORDS[k]), prov("%.3f" % r["R"], r["provisional"]) if r else ABSENT))
        L.append("\\newcommand{\\rawratio%s}{%s}" % (cap(WORDS[k]), prov("%.3f" % r["raw"], r["provisional"]) if r else ABSENT))
    return "\n".join(L) + "\n"


def contribution_note(o, e):
    """note on the scatter of the complete contributions of an order (>= 2 of them), or None"""
    full = [c for c in e.get("contributions", []) if c.get("full")]
    ol = e.get("order_level")
    if len(full) < 2 or not ol:
        return None
    k, p = len(full), is_prov(e)
    sums = ", ".join(prov(math_if(RL.fmt_ve(c["sum"], c["err"])), p) for c in full)
    birge = math.sqrt(ol["chi2"] / ol["dof"])
    mean = sum(c["sum"] for c in full) / k
    sd = math.sqrt(sum((c["sum"] - mean) ** 2 for c in full) / (k - 1))
    kind, L = RL.parse_order(o)
    if k == 2:
        apart = abs(full[0]["sum"] - full[1]["sum"]) / math.sqrt(full[0]["err"] ** 2 + full[1]["err"] ** 2)
        what = "they differ by %s standard deviations" % prov("$%.1f$" % apart, p)
    else:
        what = "their scatter gives %s for %d degrees of freedom" % (prov("$\\chi^2=%.1f$" % ol["chi2"], p), ol["dof"])
    return ("$%s_{%d}$: %d contributions (independent seeds), %s; %s. The error from their scatter, $s/\\sqrt{%d}$ "
            "with $s$ their standard deviation, %s, is %s times the quoted one."
            % (SUM[kind], L, k, sums, what, k, prov("$%s$" % sci(sd / math.sqrt(k)), p),
               prov("$%.1f$" % (sd / math.sqrt(k) / e["err"]), p)))


def scatter_cells(e):
    """(delta X_s/|X|, chi^2/dof) from the order sums of the independent seeds that cover the whole order (>= 2), or
    (ABSENT, ABSENT): the relative error of the pool rescaled by their scatter, (delta X/|X|) sqrt(chi^2/dof) (the
    Birge ratio; comparable with the pooled error even when the seeds differ in precision), and their chi^2 about the
    inverse-variance mean"""
    full = [c for c in e.get("contributions", []) if c.get("full")]
    ol = e.get("order_level")
    if len(full) < 2 or not ol or not ol.get("dof"):
        return ABSENT, ABSENT
    return "$%s$" % sci(e["rel_err"] * math.sqrt(ol["chi2"] / ol["dof"])), "%.1f/%d" % (ol["chi2"], ol["dof"])


def ref_cell(kind, L, ref):
    exact = {("b", 2): "$1/2$", ("h", 2): "$1/81$", ("c", 2): "$1/12$"}
    if (kind, L) in exact:
        return exact[(kind, L)]
    if not ref:
        return ABSENT
    v, er = ref
    return math_if("%.10g" % v if er <= 1e-9 * abs(v) else RL.fmt_ve(v, er))



def text_macros(orders, coef):
    """numbers quoted in the running text, generated so that they follow the ledger: the ratios of successive
    coefficients of beta at N = 1, eta_8 and nu_8 against Phi4tools, the precision at six and seven loops against the
    sums derived from Phi4tools, the four-loop sums against the references, the range of the relative errors of the new coefficients, and the heaviest tails at
    seven loops"""
    import coefficients as CO
    L = ["% Running text"]
    c = coef["coefficients"]
    at1 = lambda f, k: c.get(f, {}).get(str(k), {}).get("N", {}).get("1")
    ks = [k for k in range(5, 12) if at1("beta", k) and at1("beta", k + 1)]
    r = ["$%.2f$" % abs(at1("beta", k + 1)[0] / at1("beta", k)[0]) for k in ks]
    L.append("\\newcommand{\\betaratios}{%s}" % (", ".join(r[:-1]) + " and " + r[-1] if len(r) > 1 else (r[0] if r else ABSENT)))
    L.append("\\newcommand{\\betaratiokmin}{%d}" % ks[0] if ks else "\\newcommand{\\betaratiokmin}{---}")
    L.append("\\newcommand{\\betaratiokmax}{%d}" % ks[-1] if ks else "\\newcommand{\\betaratiokmax}{---}")
    ph = CO.phi4tools(1)
    for f, k, name in (("eta", 8, "etaEight"), ("nu", 8, "nuEight")):
        ours, theirs = at1(f, k), ph[f][k]
        if ours:
            L.append("\\newcommand{\\%sPull}{$%+.1f$}" % (name, (ours[0] - theirs[0]) / theirs[1]))
            L.append("\\newcommand{\\%sGain}{%d}" % (name, int(float("%.2g" % (theirs[1] / ours[1])))))
    refs = RL.references(1)
    for o in ("b6", "h6", "c6", "b7", "h7", "c7"):
        kind, Lp = RL.parse_order(o); e = orders["orders"].get(o, {}); ref = refs.get((kind, Lp))
        if e.get("contributions") and ref and ref[1] > 0:
            g = (ref[1] / abs(ref[0])) / e["rel_err"]
            L.append("\\newcommand{\\prec%s%s}{%s}" % (SUM[kind], WORDS[Lp], "%.2g" % g))
            L.append("\\newcommand{\\precinv%s%s}{%s}" % (SUM[kind], WORDS[Lp], "%.2g" % (1 / g)))
    # the four-loop sums of the ledger against the references: their largest relative error, their joint relative
    # deviation, and the 95% bounds |mean| + 1.96 sigma on a bias common to the three and on that of H_4 alone
    dev = {}
    for o in ("b4", "h4", "c4"):
        kind, Lp = RL.parse_order(o); e = orders["orders"].get(o, {}); ref = refs.get((kind, Lp))
        if e.get("contributions") and ref:
            dev[o] = ((e["sum"] - ref[0]) / abs(ref[0]), math.hypot(e["err"], ref[1]) / abs(ref[0]), e["rel_err"])
    if len(dev) == 3:
        w = {o: v[1] ** -2 for o, v in dev.items()}
        m = sum(dev[o][0] * w[o] for o in dev) / sum(w.values()); sd = sum(w.values()) ** -0.5
        x = math.floor(math.log10(sd))
        L.append("\\newcommand{\\fourrelmax}{$%s$}" % sci(max(v[2] for v in dev.values()), 2))
        L.append("\\newcommand{\\fourjointdev}{$(%+.1f\\pm%.1f)\\times10^{%d}$}" % (m / 10 ** x, sd / 10 ** x, x))
        L.append("\\newcommand{\\fourbiasbound}{$%s$}" % sci(abs(m) + 1.96 * sd, 2))
        L.append("\\newcommand{\\hfourbiasbound}{$%s$}" % sci(abs(dev["h4"][0]) + 1.96 * dev["h4"][1], 2))
    else:
        for name in ("fourrelmax", "fourjointdev", "fourbiasbound", "hfourbiasbound"):
            L.append("\\newcommand{\\%s}{%s}" % (name, ABSENT))
    rels = []
    for f, n, name in COEFS:
        v = at1(f, n)
        if v and v[0] and not (f in ("eta", "nu") and n == 8):
            rels.append(v[1] / abs(v[0]))
    L.append("\\newcommand{\\newrelmin}{%s}" % ("$%s$" % sci(min(rels), 1) if rels else ABSENT))
    L.append("\\newcommand{\\newrelmax}{%s}" % ("$%s$" % sci(max(rels), 1) if rels else ABSENT))
    L += class_macros()
    # the pooled pulls of Table tab:validation at three to seven loops (N = 1), as check N4 recomputes them
    zs = [orders["orders"][RL.order_name(k, l)]["pull"] for k in "bhc" for l in range(3, 8)
          if orders["orders"].get(RL.order_name(k, l), {}).get("pull") is not None]
    if zs:
        chi2 = sum(z * z for z in zs)
        L += ["\\newcommand{\\valncmp}{%d}" % len(zs), "\\newcommand{\\valzmax}{%.2f}" % max(abs(z) for z in zs),
              "\\newcommand{\\valchisq}{%.1f}" % chi2, "\\newcommand{\\valpval}{%.2f}" % RL.chi2_sf(chi2, len(zs))]
    # the same comparisons with the classes recombined at N = 0, 2, 3 and 4 (check N4 prints them): the range of their
    # chi^2 and the largest |z|
    chis, zmaxN = [], 0.0
    for N in (0, 2, 3, 4):
        refsN, zN = RL.references(N), []
        for k in "bhc":
            for l in range(3, 8):
                sN = coef.get("sums", {}).get(RL.order_name(k, l), {}).get("N", {}).get(str(N)); r = refsN.get((k, l))
                if sN and r:
                    zN.append((sN[0] - r[0]) / math.hypot(sN[1], r[1]))
        if zN:
            chis.append(sum(z * z for z in zN)); zmaxN = max(zmaxN, max(abs(z) for z in zN))
    if chis:
        L += ["\\newcommand{\\valNchisqmin}{%.1f}" % min(chis), "\\newcommand{\\valNchisqmax}{%.1f}" % max(chis),
              "\\newcommand{\\valNzmax}{%.2f}" % zmaxN]
    else:
        L += ["\\newcommand{\\%s}{%s}" % (n, ABSENT) for n in ("valNchisqmin", "valNchisqmax", "valNzmax")]
    # every run merged into the ledger against the pool of the runs before it (rebar-merge.py records the per-class
    # chi^2 and the deviation of the order sum on the common classes): a test of the error bars that needs no reference
    # and grows with every run
    tested = [(o, c["check_detail"]) for o, e in orders["orders"].items() for c in e.get("contributions", [])
              if (c.get("check_detail") or {}).get("dof")]
    if tested:
        chi = sum(d["chi2"] for _, d in tested); dof = sum(d["dof"] for _, d in tested); n = len(tested)
        zr = [(d["chi2"] - d["dof"]) / math.sqrt(2 * d["dof"]) for _, d in tested]
        mz = sum(zr) / n; sz = math.sqrt(sum((z - mz) ** 2 for z in zr) / n)
        ds = [d["dsum_sigma"] for _, d in tested if d.get("dsum_sigma") is not None]
        chs = sum(x * x for x in ds); Ls = [orders["orders"][o]["L"] for o, _ in tested]
        L += ["\\newcommand{\\mergeruns}{%s}" % thousands(n), "\\newcommand{\\mergeorders}{%d}" % len({o for o, _ in tested}),
              "\\newcommand{\\mergeLmin}{%d}" % min(Ls), "\\newcommand{\\mergeLmax}{%d}" % max(Ls),
              "\\newcommand{\\mergechisq}{%s}" % thousands(int(round(chi))), "\\newcommand{\\mergedof}{%s}" % thousands(dof),
              "\\newcommand{\\mergeratio}{%.3f}" % (chi / dof), "\\newcommand{\\mergezmean}{$%+.2f$}" % mz,
              "\\newcommand{\\mergezsd}{%.2f}" % sz, "\\newcommand{\\mergesumn}{%s}" % thousands(len(ds)),
              "\\newcommand{\\mergesumchisq}{%.1f}" % chs, "\\newcommand{\\mergesump}{%.2f}" % RL.chi2_sf(chs, len(ds))]
    else:
        L += ["\\newcommand{\\%s}{%s}" % (n, ABSENT) for n in ("mergeruns", "mergeorders", "mergeLmin", "mergeLmax",
              "mergechisq", "mergedof", "mergeratio", "mergezmean", "mergezsd", "mergesumn", "mergesumchisq", "mergesump")]
    # the lower-order sums that the default policy takes from refs.py (Phi4tools), and the largest change of a
    # coefficient of order >= 8 when every sum comes from the ledger instead
    fromrefs = [RL.parse_order(o) for o, src in coef.get("sources", {}).items() if src == "refs"]
    parts = []
    for kind in "bhc":
        ls = sorted(l for k, l in fromrefs if k == kind)
        runs = []
        for l in ls:
            if runs and l == runs[-1][1] + 1:
                runs[-1][1] = l
            else:
                runs.append([l, l])
        for a, b in runs:
            parts.append("$%s_{%d}$" % (SUM[kind], a) if a == b else
                         ("$%s_{%d}$, $%s_{%d}$" if b == a + 1 else "$%s_{%d}$--$%s_{%d}$") % (SUM[kind], a, SUM[kind], b))
    L.append("\\newcommand{\\refinputs}{%s}" % ((", ".join(parts[:-1]) + " and " + parts[-1]) if len(parts) > 1 else
                                                 (parts[0] if parts else "none")))
    lo = coef.get("ledger_only")
    L.append("\\newcommand{\\ledgeronlyzmax}{%s}" % ("%.2g" % lo["max_shift_in_sigma"] if lo else ABSENT))
    return L


def class_macros():
    """one pass over the pooled classes of the orders at four to ten loops: the heaviest tails at seven loops and
    beyond (kurtosis, largest weight over the mean), the round-off indicators (largest single-sample scale at seven
    loops, rms scale through seven and at eight to ten loops), and the coherent round-off bound of every order: the
    shift of the order sum if every sample of every class moved by its rms round-off scale, in units of its error"""
    name = lambda o: "$%s_{%d}$" % (SUM[o[0]], int(o[1:]))
    two = lambda x: ("%.2g" % x) if 1e-3 <= x < 100 else "$%s$" % sci(x, 2)
    tails = {"kurt7": (0.0, ""), "wmax7": (0.0, ""), "kurt8": (0.0, ""), "wmax8": (0.0, "")}
    rmax7 = {}; rms_low = 0.0; rms_high = (0.0, ""); n_rms_high = 0; coh = (0.0, "")
    fields = ("kurtosis", "maxw_over_absmean", "roundoff_max_over_sd", "roundoff_rms_over_sd", "weight_poly", "pref",
              "sd_w", "contrib_err")
    for kind in "bhc":
        for l in range(4, 11):
            o = RL.order_name(kind, l)
            shift = 0.0; e2 = 0.0; cnt = 0; mx = 0.0; seen = False
            for cl in RL.iter_classes(RL.LEDGER, o, fields=fields):
                seen = True
                fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)
                ku, wm = cl.get("kurtosis"), cl.get("maxw_over_absmean")
                rx, rr = cl.get("roundoff_max_over_sd") or 0.0, cl.get("roundoff_rms_over_sd") or 0.0
                tag = "7" if l == 7 else ("8" if l >= 8 else None)
                if tag:
                    if fin(ku) and ku > tails["kurt" + tag][0]:
                        tails["kurt" + tag] = (ku, o)
                    if fin(wm) and wm > tails["wmax" + tag][0]:
                        tails["wmax" + tag] = (wm, o)
                if l == 7 and fin(rx):
                    cnt += rx > 1e-3; mx = max(mx, rx)
                if fin(rr):
                    if l <= 7:
                        rms_low = max(rms_low, rr)
                    else:
                        n_rms_high += rr > 1e-3
                        if rr > rms_high[0]:
                            rms_high = (rr, o)
                    shift += abs(sum(cl["weight_poly"]) * cl["pref"]) * rr * cl["sd_w"]
                if fin(cl.get("contrib_err")):
                    e2 += cl["contrib_err"] ** 2
            if not seen:
                continue
            if l == 7:
                rmax7[kind] = (cnt, mx)
            if e2 > 0 and shift / math.sqrt(e2) > coh[0]:
                coh = (shift / math.sqrt(e2), o)
    L = []
    for key, tag, word in (("kurt", "7", "seven"), ("wmax", "7", "seven"), ("kurt", "8", "high"), ("wmax", "8", "high")):
        v, o = tails[key + tag]
        L.append("\\newcommand{\\%smax%s}{%s}" % (key, word, "$%s$" % sci(v, 2) if o else ABSENT))
        L.append("\\newcommand{\\%smax%sorder}{%s}" % (key, word, name(o) if o else ABSENT))
    for kind in "bhc":
        c, m = rmax7.get(kind, (0, 0.0))
        L.append("\\newcommand{\\rmaxcount%sseven}{%d}" % (SUM[kind], c))
        L.append("\\newcommand{\\rmaxmax%sseven}{%s}" % (SUM[kind], two(m) if m else ABSENT))
    L.append("\\newcommand{\\rmsmaxlow}{%s}" % (two(rms_low) if rms_low else ABSENT))
    L.append("\\newcommand{\\rmscounthigh}{%d}" % n_rms_high)
    L.append("\\newcommand{\\rmsmaxhigh}{%s}" % (two(rms_high[0]) if rms_high[1] else ABSENT))
    L.append("\\newcommand{\\rmsmaxhighorder}{%s}" % (name(rms_high[1]) if rms_high[1] else ABSENT))
    L.append("\\newcommand{\\rcohmax}{%s}" % (two(coh[0]) if coh[1] else ABSENT))
    L.append("\\newcommand{\\rcohmaxorder}{%s}" % (name(coh[1]) if coh[1] else ABSENT))
    return L

def fmt_err(v, er):
    """v(er) with two digits of the error when the printed error starts with 1 (0.002910571(13), not 0.00291057(1)),
    else one: a leading digit from 1 to 1.95 keeps two digits, one from 1.95 to 9.5 rounds to one, and one from 9.5 up
    rounds up to the next decade and prints as (10) (2.98470(10), not 2.9847(1))"""
    if not er or er <= 0 or not math.isfinite(er):
        return RL.fmt_ve(v, er, 1)
    dec = math.floor(math.log10(er)); lead = er / 10 ** dec
    if lead >= 9.5:
        return RL.fmt_ve(v, 10.0 ** (dec + 1), 2)
    return RL.fmt_ve(v, er, 2 if lead < 1.95 else 1)


def tf_cell(e):
    """the integrand evaluations of all the runs of an order, in Tf (1 Tf = 10^12 function evaluations), two digits"""
    n = e.get("samples")
    if not n:
        return ABSENT
    x = n / 1e12
    if x >= 100:
        return "%d" % int(float("%.2g" % x))
    return "%.2g" % x


def newsums_tex(orders):
    rows, notes = [], []
    for o in ORDERS:
        kind, L = RL.parse_order(o)
        e = orders["orders"].get(o)
        if not e or (not e.get("contributions") and not e.get("list")):
            continue
        has = bool(e.get("contributions"))
        p = is_prov(e)
        f, sh = RL.ENTERS[kind]
        val = prov(math_if(RL.fmt_ve(e["sum"], e["err"])), p) if has else ABSENT
        sc, chi = scatter_cells(e) if has else (ABSENT, ABSENT)
        ref = RL.references(1).get((kind, L))
        rows.append("$%s_{%d}$ & %s, $\\ubar^{%d}$ & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            SUM[kind], L, GREEK[kind], L + sh, thousands(e["graphs"]) if e.get("graphs") else ABSENT,
            thousands(e["classes"]) if has else ABSENT, ref_cell(kind, L, ref[:2] if ref else None), val,
            prov("$%s$" % sci(e["rel_err"]), p) if has else ABSENT, tf_cell(e) if has else ABSENT, sc, chi))
    out = ["% generated/newsums.tex -- body of Table tab:newsums (new and improved per-order sums at N=1).",
           "% WRITTEN BY series/tables.py (make -C ../series) from ledger/orders.json: do not edit by hand.",
           "% Previous: values derived from Phi4tools through the Callan-Symanzik relations (integrator/tools/refs.py).",
           "\\begin{ruledtabular}", "\\begin{tabular}{llrrllllll}",
           "Sum & Enters & Graphs & Classes & Previous & This work & $\\delta X/|X|$ & Cost [Tf] & $\\delta X_{\\rm s}/|X|$ & $\\chi^2/{\\rm dof}$ \\\\",
           "\\hline"]
    out += rows + ["\\end{tabular}", "\\end{ruledtabular}"]
    return "\n".join(out) + "\n"


def validation_tex(orders):
    out = ["% generated/validation.tex -- body of Table tab:validation (per-order sums at N=1 against the references).",
           "% WRITTEN BY series/tables.py (make -C ../series) from ledger/orders.json: do not edit by hand.",
           "% This work: all contributions of the ledger pooled class by class. Reference: exact (L = 2) or derived from",
           "% Phi4tools through the Callan-Symanzik relations (integrator/tools/refs.py); z = pull.",
           "\\begin{ruledtabular}", "\\begin{tabular}{lrlrlllr}",
           "Sum & $L$ & Reference & Classes & This work & $\\delta X/|X|$ & Cost [Tf] & $z$ \\\\"]
    for kind in "bhc":
        out.append("\\hline")
        first = True
        for L in range(2, 8):
            e = orders["orders"].get(RL.order_name(kind, L))
            if not e or not e.get("contributions"):
                continue
            p = is_prov(e)
            ref = e.get("ref")
            z = prov("$%+.2f$" % e["pull"], p) if e.get("pull") is not None else ABSENT
            out.append("%s & %d & %s & %s & %s & %s & %s & %s \\\\" % (
                "$%s_L$" % SUM[kind] if first else "", L, ref_cell(kind, L, ref), thousands(e["classes"]),
                prov(math_if(RL.fmt_ve(e["sum"], e["err"])), p), prov("$%s$" % sci(e["rel_err"]), p),
                tf_cell(e), z))
            first = False
    out += ["\\end{tabular}", "\\end{ruledtabular}"]
    return "\n".join(out) + "\n"


def main(argv):
    gen = os.path.join(RL.HERE, "generated")
    cf, ck, outd = os.path.join(gen, "coefficients.json"), os.path.join(gen, "checks.json"), os.path.join(RL.ROOT, "paper", "generated")
    it = iter(argv[1:])
    for a in it:
        if a == "--coefficients":
            cf = next(it)
        elif a == "--checks":
            ck = next(it)
        elif a == "--out":
            outd = next(it)
        else:
            print(__doc__)
            return 2
    orders = RL.load_orders(RL.LEDGER)
    coef, chk = RL.load_json(cf), RL.load_json(ck)
    for name, text in (("values.tex", values_tex(orders, coef, chk)), ("newsums.tex", newsums_tex(orders)),
                       ("validation.tex", validation_tex(orders))):
        p = os.path.join(outd, name)
        print("%s %s" % ("wrote" if RL.write_if_changed(p, text) else "unchanged", os.path.relpath(p, RL.ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
