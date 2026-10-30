#!/usr/bin/env python3
# S3_largeorder.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check S3: the large-order ratios of Table tab:largeorder follow from the coefficients and the stated constants.
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
"""S3 -- the large-order ratios of Table tab:largeorder follow from the coefficients and the stated constants.

Claim (paper, Sec. II E "Large order", Sec. III C "Large-order and large-N tests" and Table tab:largeorder): every coefficient F_k of ubar^k in beta, eta and
eta_phi2 is divided by its leading large-order estimate c_F (-a)^k Gamma(k + b_F + 1), with the instanton action
A = 113.3835 of d = 3, a = 48 pi/(9 A) = 0.147774232 at N = 1 (a = 48 pi/((N+8) A) in general), b = 3 + N/2 for beta and
eta_phi2 and 2 + N/2 for eta, and the amplitudes c_F (Zinn-Justin's book, 3rd ed., Eq. (28.73)); at N = 1 the text quotes
the ratios quoted for beta at k = 7 onwards, 0.81-0.86 for eta_phi2 from k = 6 on, and 0.54-0.65 for eta over the
same orders.

Inputs: series/largeorder.py (the estimate, after R. Guida's ribeta.asymptotics.m of 2003-04), the coefficients
of series/coefficients.py (from the ledger and refs.py), paper/generated/largeorder.tex, paper/sections/results.tex.
What runs:
  (a) the constants: 48 pi/(9 A) with largeorder.py's A against the quoted a = 0.147774232 (and series/checks.py's
      a = 0.14777422, to 1e-7);
  (b) the k dependence of largeorder.asymptotic(k, F, N) against c_F (-a_N)^k Gamma(k + b_F + 1) with the quoted
      a_N and b_F (c_F fitted at k = 3), for k = 3..12 and N = 0..4: the ratio must be constant;
  (c) the ratios F_k / estimate at N = 1, k = 3..9, recomputed from the coefficients, against every cell of the
      table (three decimals), and the numbers quoted in the text.
PASS: (a) to the quoted digits; (b) constant to 1e-12; (c) every cell and every quoted number reproduced.
"""
import math, os, re, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the large-order ratios of Table (tab:largeorder) follow from the coefficients and the stated constants (Secs. II E, III C)"
QUICK = True
B = {"beta": lambda N: 3 + N / 2.0, "eta": lambda N: 2 + N / 2.0, "eta_phi2": lambda N: 3 + N / 2.0}


def main():
    LO = C.series_module("largeorder")
    CK = C.series_module("checks")
    ok = True
    # (a)
    a1 = 48 * math.pi / (9 * LO.AA)
    oka = abs(a1 - 0.147774232) < 1e-9 and abs(LO.AA - 113.3835) < 5e-5 and abs(a1 - CK.A1) < 1e-7
    print("(a) A = %.9f (quoted 113.3835); a = 48 pi/(9 A) = %.8f (quoted 0.147774)  %s" % (LO.AA, a1, "OK" if oka else "MISMATCH"))
    if CK.A1 != round(a1, 8):
        print("    NOTE: series/checks.py (ratio test of values.tex) uses a = %.8f, %.1e below 48 pi/(9 A) = %.8f (rounding; it "
              "moves its ratios by %.0e relative)" % (CK.A1, a1 - CK.A1, a1, (a1 - CK.A1) / a1))
    ok = ok and oka
    # (b)
    worst = 0.0
    for f in B:
        for N in range(5):
            aN = 48 * math.pi / ((N + 8) * LO.AA)
            form = lambda k: (-aN) ** k * math.gamma(k + B[f](N) + 1)
            c3 = LO.asymptotic(3, f, N) / form(3)
            for k in range(3, 13):
                worst = max(worst, abs(LO.asymptotic(k, f, N) / form(k) / c3 - 1))
    okb = worst < 1e-12
    print("(b) largeorder.asymptotic(k) / [c (-a_N)^k Gamma(k + b + 1)], b = 3 + N/2 (beta, eta_phi2), 2 + N/2 (eta): "
          "constant in k = 3..12 for N = 0..4 to %.1e  %s" % (worst, "OK" if okb else "MISMATCH"))
    ok = ok and okb
    # (c)
    CO = C.series_module("coefficients")
    res = CO.compute()
    got = {}
    for f in B:
        for k, v, e, est, r in LO.ratios(res, f, 1):
            got[(f, k)] = "%.3f" % r
    with open(os.path.join(C.PAPER, "generated", "largeorder.tex")) as fh:
        tex = fh.read()
    cells, bad = 0, []
    for line in tex.splitlines():
        m = re.match(r"^(\d+) & (.*) \\\\$", line)
        if not m:
            continue
        k = int(m.group(1))
        for f, c in zip(("beta", "eta", "eta_phi2"), m.group(2).split(" & ")):
            v = re.sub(r"\\provisional\{(.*)\}", r"\1", c.strip())
            if v == "---":
                continue
            cells += 1
            if got.get((f, k)) != v:
                bad.append("%s k=%d: table %s, recomputed %s" % (f, k, v, got.get((f, k))))
    print("(c) N = 1, k = 3-9: %d cells of paper/generated/largeorder.tex, %d differ from the ratios recomputed from the coefficients"
          % (cells, len(bad)))
    for k in sorted({k for _, k in got}):
        print("      k=%d  beta %s  eta %s  eta_phi2 %s" % (k, got.get(("beta", k)), got.get(("eta", k)), got.get(("eta_phi2", k))))
    for x in bad:
        print("    " + x)
    with open(os.path.join(C.PAPER, "sections", "results.tex")) as fh:   # Sec. III C, "Large-order and large-N tests"
        text = " ".join(fh.read().split())
    m = re.search(r"the ratio coefficient/estimate is ((?:\$[\d.]+\$,? )+)and \$([\d.]+)\$ for \$\\beta\$ at \$k=(\d+)\$ to \$(\d+)\$", text)
    quoted = (re.findall(r"[\d.]+", m.group(1)) + [m.group(2)]) if m else None
    ks = list(range(int(m.group(3)), int(m.group(4)) + 1)) if m else []
    beta_ok = bool(quoted) and len(quoted) == len(ks) and quoted == [got.get(("beta", k)) for k in ks]
    m2 = re.search(r"for \$\\eta_\{\\phi\^2\}\$ the ratio lies between \$([\d.]+)\$ and \$([\d.]+)\$ from \$k=(\d+)\$ on", text)
    rng = None
    if m2:
        lo, hi, k0 = float(m2.group(1)), float(m2.group(2)), int(m2.group(3))
        vals = [round(float(got[("eta_phi2", k)]), 2) for k in range(k0, 20) if ("eta_phi2", k) in got]
        rng = vals and min(vals) >= lo and max(vals) <= hi
    m3 = re.search(r"and for \$\\eta\$, whose asymptotic regime sets in late, between \$([\d.]+)\$ and \$([\d.]+)\$", text)
    rng_eta = None
    if m2 and m3:      # the same range of k as for eta_phi2
        lo, hi, k0 = float(m3.group(1)), float(m3.group(2)), int(m2.group(3))
        vals = [round(float(got[("eta", k)]), 2) for k in range(k0, 20) if ("eta", k) in got]
        rng_eta = vals and min(vals) >= lo and max(vals) <= hi
    print("    text (results.tex): eta 'between %s and %s' from the same k: %s" % (m3.group(1) if m3 else "?",
          m3.group(2) if m3 else "?", "reproduced" if rng_eta else "NOT reproduced"))
    print("    text (results.tex): beta at k = %s quoted %s: %s; eta_phi2 'between %s and %s from k=%s on': %s"
          % ("-".join(map(str, ks[::max(1, len(ks) - 1)])), quoted, "reproduced" if beta_ok else "NOT reproduced", m2.group(1) if m2 else "?", m2.group(2) if m2 else "?",
             m2.group(3) if m2 else "?", "reproduced" if rng else "NOT reproduced"))
    ok = ok and not bad and cells > 0 and beta_ok and bool(rng) and bool(rng_eta)
    C.verdict("S3", ok, "constants a, b as stated; %d table cells and the quoted ratios (beta %s at k = %s) recomputed from the coefficients"
              % (cells, ", ".join(quoted or ["?"]), "%d-%d" % (ks[0], ks[-1]) if ks else "?"))


if __name__ == "__main__":
    main()
