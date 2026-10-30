#!/usr/bin/env python3
"""
    costfig.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    The distribution of the Neyman cost over the graphs of an order, for the paper's Fig. fig:costhist.

    With sigma_i the standard deviation per sample of graph i's weighted contribution and tau_i its device time per
    sample, the device time that brings the order sum S_L to a relative uncertainty eps under Neyman allocation is
        T(eps) = (sum_i kappa_i)^2 / eps^2,    kappa_i = sigma_i sqrt(tau_i) / |S_L|,
    and graph i receives the share kappa_i / sum_j kappa_j of it.  This program histograms log10 kappa_i (bins of a
    quarter of a decade) for every order with data and writes the pgfplots bodies the paper draws, together with the
    macros it quotes (number of graphs, median kappa, sum of kappa, T(1e-5) in GPU hours, and the fraction of the
    graphs that carry 90 % of the cost).  sigma_i = |w_i(1)| pref_i sd_w_i from the pooled classes, tau_i the
    measured core nanoseconds per sample on the H200 (the ledger's core_ns_per_sample: the n-weighted mean over the
    runs on that device; a format-1 ledger, the median over its contributions).

    It also writes, for Fig. fig:cost, the CPU hours of the class build of every order (orders.json build_s: the
    single-thread build time of every class, counted once, from the first run that sampled it; a format-1 ledger, the
    median over its contributions) next to the measured GPU hours T(1e-5) of the complete orders: the pooled relative
    error of the order times the square root of the device time of all its contributions (orders.json core_s),
    scaled to 1e-5.  The same measured costs fill the body of Table tab:cost (\\costtablerows), this work's rows of Table
    tab:effort (\\effortrows) and the numbers the text quotes (\\costTm{order}, \\costgrowth{kind},
    \\costratios{kind}, \\costgrowthrange, \\costTsevenmin/max, \\costmatchbsix, \\costkappadev).

    usage: costfig.py [--ledger DIR] [--tex FILE] [--min-loops L]
    Default output: ../paper/generated/costhist.tex.

    This program was designed and developed by the author(s),
    with the assistance of the following AI tool(s):
      2026 Claude (Anthropic).
    The final logic and implementation were reviewed and verified in
    their entirety by the author(s).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rebar_ledger as RL  # noqa: E402
BIN = 0.25            # decades
KINDS = (("b", "B"), ("h", "H"), ("c", "C"))


def median(xs):
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


_MEMO = {}


def _memo(key, compute):
    if key not in _MEMO:
        _MEMO[key] = compute()
    return _MEMO[key]


def _entry(ledger, order):
    return _memo((ledger, None), lambda: RL.load_orders(ledger))["orders"].get(order) or {}


def _per_run(ledger, order, field):
    """{nickel: [the field of every contribution that has it]}: the runs themselves (format 1: the ledger's files)"""
    out = {}
    for _, c in RL.iter_order_contributions(ledger, order):
        if c.get(field) is not None:
            out.setdefault(c["nickel"], []).append(c[field])
    return out


def order_kappas(ledger, order):
    """[(nickel, kappa)] of an order, or None when it has no classes or no timing."""
    return _memo((ledger, order, "kappas"), lambda: _order_kappas(ledger, order))


def _order_kappas(ledger, order):
    e = _entry(ledger, order)
    if not e.get("contributions"):
        return None
    cls = RL.load_classes(ledger, order, fields=("nickel", "weight_poly", "pref", "sd_w", "core_ns_per_sample"))
    if any("core_ns_per_sample" in c for c in cls):    # the timing register (format 3)
        tau = {c["nickel"]: c["core_ns_per_sample"] for c in cls}
    else:                                              # format 1: the median over the contributions
        tau = {nk: median([t for t in ts if t > 0]) for nk, ts in _per_run(ledger, order, "core_ns_per_sample").items()
               if any(t > 0 for t in ts)}
    out = []
    for c in cls:
        t = tau.get(c["nickel"])
        if not t or t <= 0 or (c.get("sd_w") or 0) <= 0:
            continue
        sigma = abs(sum(c["weight_poly"])) * c["pref"] * c["sd_w"]      # N = 1
        out.append((c["nickel"], sigma * math.sqrt(t * 1e-9) / abs(e["sum"])))
    return out if out else None


def order_cpu_hours(ledger, order):
    """The single-thread build time of the order's classes, in hours: orders.json build_s (every class counted once,
    from the first run that sampled it); a format-1 ledger, the sum over the classes of the median over the
    contributions."""
    e = _entry(ledger, order)
    if "build_s" in e:
        return e["build_s"] / 3600.0 if e["build_s"] else None
    b = _memo((ledger, order, "build_s"), lambda: _per_run(ledger, order, "build_s"))
    return sum(median(v) for v in b.values()) / 3600.0 if b else None


def order_measured(ledger, order, entry):
    """(relative error, device seconds, rho_tau, T(1e-5) in hours) of a complete order: its pooled relative error and
    the device time of all its contributions (orders.json core_s, summed over GPUs); None if the order is
    incomplete."""
    if not entry or not entry.get("complete") or not entry.get("rel_err"):
        return None
    if "core_s" in entry:
        dev = entry["core_s"] or 0.0
    else:       # format 1: the order blocks of the runs
        dev = sum((RL.load_contribution_header(ledger, order, c["id"]).get("order") or {}).get("core_s") or 0.0
                  for c in entry.get("contributions", []))
    if dev <= 0:
        return None
    rel = entry["rel_err"]; rho = rel * math.sqrt(dev)
    return rel, dev, rho, rho * rho / 1e-10 / 3600


def sig2(x):
    """two significant digits, trailing zero kept (0.046, 0.20, 2.0, 16; three digits from 100 to 999; 1\\,100 and up
    rounded to two digits with a thin space)"""
    if abs(x) >= 1000:
        return "{:,}".format(int(float("%.2g" % x))).replace(",", "\\,")
    d = max(0, 1 - int(math.floor(math.log10(abs(x)))))
    return "%.*f" % (d, x)


def sig3(x):
    """three significant digits, trailing zeros kept (1.30, 12.9, 102)"""
    d = max(0, 2 - int(math.floor(math.log10(abs(x)))))
    return "%.*f" % (d, x)


WORDS = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def andlist(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def histogram(kappas):
    """[(left edge, count)] over quarter-decade bins from the smallest to the largest log10 kappa."""
    lk = [math.log10(k) for k in kappas]
    lo, hi = math.floor(min(lk) / BIN) * BIN, math.ceil(max(lk) / BIN) * BIN
    nb = int(round((hi - lo) / BIN))
    counts = [0] * nb
    for x in lk:
        counts[min(int((x - lo) / BIN), nb - 1)] += 1
    return [(lo + i * BIN, counts[i]) for i in range(nb)]


def main(argv):
    ledger = RL.LEDGER; tex = os.path.join(ROOT, "paper", "generated", "costhist.tex"); minL = 6
    it = iter(argv)
    for x in it:
        if x == "--ledger": ledger = next(it)
        elif x == "--tex": tex = next(it)
        elif x == "--min-loops": minL = int(next(it))
        else: raise SystemExit(__doc__)
    orders = RL.load_orders(ledger)["orders"]
    L = ["% generated/costhist.tex -- Fig. fig:costhist: histograms of log10 kappa_i, the per-graph Neyman cost weights.",
         "% WRITTEN BY series/costfig.py (make -C ../series): do not edit by hand.",
         "% \\costhist<B|C|H>: the \\addplot bodies of one panel (step histograms, per cent of the order's graphs per quarter decade, colours colL<L> defined by the paper, legend entries included);",
         "% \\costN{order}, \\costmedian{order}, \\costsum{order}, \\costT{order}, \\costninety{order}: the numbers quoted in the text;",
         "% \\costcurves: the \\addplot bodies of Fig. fig:cost (CPU hours of the class build, GPU hours of the sampling at 1e-5, per order), \\costcpu{order}.", ""]
    nums = {}
    print("order  graphs  median kappa  sum kappa [sqrt(s)]  T(1e-5) [GPU h]  graphs for 90%% of the cost")
    for kind, K in KINDS:
        body = []
        for order in sorted((o for o in orders if o[0] == kind), key=lambda o: int(o[1:])):
            Ln = int(order[1:])
            if Ln < minL:
                continue
            kap = order_kappas(ledger, order)
            if not kap:
                continue
            ks = [k for _, k in kap]; n = len(ks); total = sum(ks)
            srt = sorted(ks, reverse=True); acc = 0; n90 = 0
            for k in srt:
                acc += k; n90 += 1
                if acc >= 0.9 * total:
                    break
            T = total ** 2 / 1e-10 / 3600
            nums[order] = (n, median(ks), total, T, n90 / n)
            print("%-5s %7d  %12.3e  %19.3e  %15.3g  %5.2f" % (order, n, median(ks), total, T, n90 / n))
            h = histogram(ks)
            pts = " ".join("(%.2f,%.4f)" % (x, 100.0 * c / n) for x, c in h) + " (%.2f,0)" % (h[-1][0] + BIN)
            body.append("\\addplot[const plot, thick, mark=none, color=colL%d] coordinates {%s};" % (Ln, pts))
            body.append("\\addlegendentry{$L=%d$ (%s classes)}" % (Ln, "{:,}".format(n).replace(",", "\\,")))
        L.append("\\newcommand*{\\costhist%s}{%%" % K)
        L += ["  " + b for b in body]
        L.append("}")
    # Fig. fig:cost: per order, the CPU hours of the class build and the GPU hours of the sampling at 1e-5
    curves = []; cpu = {}
    styles = {"b": ("blue", "*", "o"), "h": ("red", "square*", "square"), "c": ("brown!60!black", "triangle*", "triangle")}
    print("order  CPU build [core h]  GPU sampling T(1e-5), measured [H200 h]")
    for kind, K in KINDS:
        gpu_pts = []; cpu_pts = []
        for order in sorted((o for o in orders if o[0] == kind), key=lambda o: int(o[1:])):
            Ln = int(order[1:])
            if Ln < 4:
                continue
            kap = order_kappas(ledger, order); hrs = order_cpu_hours(ledger, order)
            meas = order_measured(ledger, order, orders.get(order))
            if meas:
                gpu_pts.append((Ln, meas[3]))
            if hrs:
                cpu_pts.append((Ln, hrs)); cpu[order] = hrs
            print("%-5s %18s  %18s" % (order, "%.4g" % hrs if hrs else "-", "%.4g" % meas[3] if meas else "-"))
        col, mk, mko = styles[kind]
        curves.append("\\addplot[color=%s, thick, mark=%s] coordinates {%s};" % (col, mk, " ".join("(%d,%.4g)" % p for p in gpu_pts)))
        curves.append("\\addlegendentry{$%s_L$, GPU}" % K)
        curves.append("\\addplot[color=%s, thick, dashed, mark=%s] coordinates {%s};" % (col, mko, " ".join("(%d,%.4g)" % p for p in cpu_pts)))
        curves.append("\\addlegendentry{$%s_L$, CPU}" % K)
    L.append("\\newcommand*{\\costcurves}{%")
    L += ["  " + c for c in curves]
    L.append("}")
    L.append("\\newcommand*{\\costcpu}[1]{\\csname costcpu@#1\\endcsname}")
    for order, hrs in sorted(cpu.items()):
        L.append("\\expandafter\\def\\csname costcpu@%s\\endcsname{%s}" % (order, "%.3g" % hrs))
    # the same build times in core-seconds, in total and per class (thread-seconds per graph)
    L.append("\\newcommand*{\\costcpus}[1]{\\csname costcpus@#1\\endcsname}")
    L.append("\\newcommand*{\\buildpc}[1]{\\csname buildpc@#1\\endcsname}")
    for order, hrs in sorted(cpu.items()):
        sec = hrs * 3600.0; ncl = (orders.get(order) or {}).get("classes")
        L.append("\\expandafter\\def\\csname costcpus@%s\\endcsname{%s}" % (order, "{:,}".format(int(float("%.2g" % sec))).replace(",", "\\,")))
        if ncl:
            pc = sec / ncl
            L.append("\\expandafter\\def\\csname buildpc@%s\\endcsname{%s}" % (order, "%d" % round(pc) if pc >= 10 else "%.2g" % pc))
    # Table tab:cost (body rows), Table tab:effort (this work's rows) and the numbers the text quotes, all measured
    meas = {o: order_measured(ledger, o, orders[o]) for o in orders}
    meas = {o: m for o, m in meas.items() if m}
    rows = []
    for kind, K in KINDS:
        first = True
        for order in sorted((o for o in meas if o[0] == kind and int(o[1:]) >= 4), key=lambda o: int(o[1:])):
            rel, dev, rho, T = meas[order]
            # the hardware-independent counterpart: integrand evaluations (samples) of all the runs of the order
            n = orders[order].get("samples") or 0
            rn = rel * math.sqrt(n) if n else float("nan")
            rows.append("%s & %d & %.2f & %s & %s & %s & %s \\\\" % ("$%s_L$" % K if first else "     ", int(order[1:]), rel / 1e-5,
                        sig3(rho / 1e-4), sig2(T), sig3(rn), sig2(rn * rn / 1e-10 / 1e12)))
            first = False
    L.append("\\newcommand*{\\costtablerows}{%")
    L += ["  " + r for r in rows]
    L.append("}")
    Ls = [Ln for Ln in range(6, 13) if any(k + str(Ln) in meas for k, _ in KINDS)]
    quant, times = [], []
    for Ln in Ls:   # one line per loop order: the sums and their classes; GPU hours at 1e-5 and the CPU hours of the build
        os_ = [k + str(Ln) for k, _ in KINDS if k + str(Ln) in meas]
        cpu_h = sum(order_cpu_hours(ledger, o) or 0 for o in os_)
        quant.append("%s: %s classes" % (", ".join("$%s_{%d}$" % (o[0].upper(), Ln) for o in os_),
                                          "{:,}".format(sum(orders[o]["classes"] for o in os_)).replace(",", "\\,")))
        times.append(", ".join(sig2(meas[o][3]) for o in os_) + " GPU h" + ("; %s CPU h" % sig2(cpu_h) if cpu_h >= 1 else ""))
    stack = lambda xs: "\\begin{tabular}[t]{@{}l@{}}" + " \\\\ ".join(xs) + "\\end{tabular}"
    eff = ["This work, %s to %s loops & %s & NVIDIA H200 GPUs, Intel Xeon Platinum 8568Y+ CPUs & %s & "
           "$10^{-5}$ relative, per order sum \\\\" % (WORDS[Ls[0]], WORDS[Ls[-1]], stack(quant), stack(times))] if Ls else []
    L.append("\\newcommand*{\\effortrows}{%")
    L += ["  " + r for r in eff]
    L.append("}")
    L.append("\\newcommand*{\\costTm}[1]{\\csname costTm@#1\\endcsname}")
    for o, m in sorted(meas.items()):
        L.append("\\expandafter\\def\\csname costTm@%s\\endcsname{%s}" % (o, sig2(m[3])))
    growth = {}
    for kind, K in KINDS:
        Ls = sorted(int(o[1:]) for o in meas if o[0] == kind and int(o[1:]) >= 4)
        if 7 in Ls and Ls[-1] > 7:
            growth[kind] = (meas["%s%d" % (kind, Ls[-1])][3] / meas["%s7" % kind][3]) ** (1.0 / (Ls[-1] - 7))
            L.append("\\expandafter\\def\\csname costgrowth@%s\\endcsname{%.1f}" % (kind, growth[kind]))
        r2 = [meas["%s%d" % (kind, l + 1)][2] ** 2 / meas["%s%d" % (kind, l)][2] ** 2 for l in Ls[:-1] if l + 1 in Ls]
        L.append("\\expandafter\\def\\csname costratios@%s\\endcsname{%s}" % (kind, andlist(["%.2g" % r for r in r2])))
        L.append("\\expandafter\\def\\csname costratiorange@%s\\endcsname{%s to %s}" % (kind, WORDS[Ls[0]], WORDS[Ls[-1]]))
    L.append("\\newcommand*{\\costgrowth}[1]{\\csname costgrowth@#1\\endcsname}")
    L.append("\\newcommand*{\\costratios}[1]{\\csname costratios@#1\\endcsname}")
    L.append("\\newcommand*{\\costratiorange}[1]{\\csname costratiorange@#1\\endcsname}")
    g = sorted(growth.values())
    L.append("\\newcommand*{\\costgrowthrange}{%.1f--%.1f}" % (g[0], g[-1]))
    # the same growth counted in integrand evaluations, which do not depend on the hardware
    ev = lambda o: (meas[o][0] ** 2 * (orders[o].get("samples") or 0)) / 1e-10
    ge = []
    for kind, K in KINDS:
        Ls = sorted(int(o[1:]) for o in meas if o[0] == kind and int(o[1:]) >= 4)
        if 7 in Ls and Ls[-1] > 7 and ev("%s7" % kind) > 0:
            ge.append((ev("%s%d" % (kind, Ls[-1])) / ev("%s7" % kind)) ** (1.0 / (Ls[-1] - 7)))
    ge.sort()
    L.append("\\newcommand*{\\evalgrowthrange}{%.1f--%.1f}" % (ge[0], ge[-1]) if ge else "\\newcommand*{\\evalgrowthrange}{---}")
    seven = sorted(meas[o][3] for o in meas if o[1:] == "7")
    L.append("\\newcommand*{\\costTsevenmin}{%s}\\newcommand*{\\costTsevenmax}{%s}" % (sig2(seven[0]), sig2(seven[-1])))
    ref = orders.get("b6", {}).get("ref")
    if ref and "b6" in meas:
        tgt = ref[1] / abs(ref[0]); Tm = meas["b6"][3] * (1e-5 / tgt) ** 2
        L.append("\\newcommand*{\\costrefbsix}{%.2g}\\newcommand*{\\costmatchbsix}{%s}\\newcommand*{\\costmatchbsixfour}{%s}" % (tgt / 1e-6, sig2(Tm), sig2(Tm / 4)))
    dev = []
    for o, m in meas.items():
        if int(o[1:]) >= 6:
            kap = order_kappas(ledger, o)
            if kap:
                dev.append(abs((sum(k for _, k in kap) ** 2 / 1e-10 / 3600) / m[3] - 1))
    L.append("\\newcommand*{\\costkappadev}{%d}" % round(100 * max(dev)))
    # the numbers the captions of Fig. 1 and Fig. 2 quote: the growth per loop of the build (from seven loops on), and of
    # the graphs, of the sum of the Neyman weights and the shift of their median (from six loops on)
    cg = sorted(cpu["%s%d" % (k, l + 1)] / cpu["%s%d" % (k, l)] for k, _ in KINDS for l in range(7, 12)
                if cpu.get("%s%d" % (k, l)) and cpu.get("%s%d" % (k, l + 1)))
    g2 = lambda x: "%d" % int(float("%.2g" % x)) if x >= 10 else "%.2g" % x
    L.append("\\newcommand*{\\cpugrowthrange}{%s}" % ("%s to %s" % (g2(cg[0]), g2(cg[-1])) if cg else "---"))
    ks, sh, gg = [], [], []
    for k, _ in KINDS:
        for l in range(6, 12):
            a, b = order_kappas(ledger, "%s%d" % (k, l)), order_kappas(ledger, "%s%d" % (k, l + 1))
            if not a or not b:
                continue
            ka = [x for _, x in a if x > 0]; kb = [x for _, x in b if x > 0]
            ks.append(sum(kb) / sum(ka)); gg.append(len(kb) / len(ka))
            sh.append(median([math.log10(x) for x in ka]) - median([math.log10(x) for x in kb]))
    ks.sort(); gg.sort()
    L.append("\\newcommand*{\\kappagrowthrange}{%s}" % ("%.2g to %.2g" % (ks[0], ks[-1]) if ks else "---"))
    L.append("\\newcommand*{\\graphgrowthrange}{%s}" % ("%.2g to %.2g" % (gg[0], gg[-1]) if gg else "---"))
    L.append("\\newcommand*{\\kappashift}{%s}" % ("%.1f" % (sum(sh) / len(sh)) if sh else "---"))
    # what the tenth order (B_9, H_10, C_10) actually took: device hours of all its runs, CPU hours of its class builds
    tenth = [o for o in ("b9", "h10", "c10") if o in meas]
    if tenth:
        gpu = sum(meas[o][1] for o in tenth) / 3600.0; cpu = sum(order_cpu_hours(ledger, o) or 0 for o in tenth)
        r2 = lambda x: "{:,}".format(int(float("%.2g" % x))).replace(",", "\\,")
        L.append("\\newcommand*{\\costtenthgpu}{%s}\\newcommand*{\\costtenthcpu}{%s}" % (r2(gpu), r2(cpu)))
        # the same in node-hours: 32 cores per CPU node (the class builds), eight H200 per GPU node
        L.append("\\newcommand*{\\costtenthnodecpu}{%s}\\newcommand*{\\costtenthnodegpu}{%s}" % (r2(cpu / 32.0), r2(gpu / 8.0)))
    print("measured T(1e-5) [GPU h]: " + "  ".join("%s %s" % (o, sig2(meas[o][3])) for o in sorted(meas, key=lambda o: (o[0], int(o[1:])))))
    print("growth per loop from L = 7:", {k: round(v, 2) for k, v in growth.items()}, " kappa vs measured, max deviation %.0f%%" % (100 * max(dev)))
    L.append("\\newcommand*{\\costN}[1]{\\csname costN@#1\\endcsname}")
    L.append("\\newcommand*{\\costmedian}[1]{\\csname costmedian@#1\\endcsname}")
    L.append("\\newcommand*{\\costsum}[1]{\\csname costsum@#1\\endcsname}")
    L.append("\\newcommand*{\\costT}[1]{\\csname costT@#1\\endcsname}")
    L.append("\\newcommand*{\\costninety}[1]{\\csname costninety@#1\\endcsname}")
    f90s = sorted(round(100 * v[4]) for v in nums.values())
    L.append("\\newcommand*{\\costninetyrange}{%s}" % ("%d--%d" % (f90s[0], f90s[-1]) if f90s else "---"))
    for order, (n, med, total, T, f90) in sorted(nums.items()):
        L.append("\\expandafter\\def\\csname costN@%s\\endcsname{%d}" % (order, n))
        L.append("\\expandafter\\def\\csname costmedian@%s\\endcsname{%s}" % (order, texnum(med)))
        L.append("\\expandafter\\def\\csname costsum@%s\\endcsname{%s}" % (order, texnum(total)))
        L.append("\\expandafter\\def\\csname costT@%s\\endcsname{%s}" % (order, "%.3g" % T))
        L.append("\\expandafter\\def\\csname costninety@%s\\endcsname{%d}" % (order, round(100 * f90)))
    L.append("")
    with open(tex, "w") as f:
        f.write("\n".join(L))
    print("wrote", os.path.relpath(tex, ROOT))


def texnum(x):
    """1.23 \\times 10^{-4} (math mode content)"""
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    return "%.2f \\times 10^{%d}" % (m, e)


if __name__ == "__main__":
    main(sys.argv[1:])
