#!/usr/bin/env python3
# I1_pointwise.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check I1: the forest expansion of rebar agrees pointwise with an independent 100-digit evaluation of the
# forest formula (checks/reference/rebar_mp.py, mpmath), on a stored sample of points (results/checks/reference/).
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
"""I1 -- pointwise agreement of the forest expansion with an independent 100-digit evaluation.

Claim (paper, Sec. II B 3 "Integrands" and II E): the expansion of every forest term into products of atoms, with
the cancelling combinations removed, Eq. (integrand), agrees pointwise with an independent evaluation of the forest
formula in 100-digit arithmetic.

Inputs: results/checks/reference/sample.tsv: points x (log-coordinates, exact doubles) for b, h, c classes at 3-7 loops
(all classes at 3-5 loops and h at 6, every k-th beyond, one-vertex-reducible and self-loop graphs included), each with
the value of the projective integrand f(x) prod x_e (sum x)^(-pw) and the sum of the absolute values of the forest
terms, computed by checks/reference/rebar_mp.py at 100 digits (written by `I1_pointwise.py --make-sample`, with mpmath).
Base points have x_e in [0.05, 1]; face points have a random proper subset of the lines scaled by 1e-3.

What runs:
  (a) `rebar_cpu eval` (the production expansion, evaluated in long double) at every stored point, against the stored
      100-digit values: at base points |f_rebar - f_mp| <= 1e-12 |f_mp|; at face points, where the forest terms
      cancel, |f_rebar - f_mp| <= 1e-16 sum|terms| or <= 1e-12 |f_mp|; a vanishing graph (|f_mp| < 1e-60 sum|terms|)
      must give |f_rebar| <= 1e-16 sum|terms| (rebar gives exactly 0 when the pruning leaves no term).
  (b) with mpmath (in this interpreter, or in REBAR_MPMATH_PYTHON), rebar_mp.py recomputes the stored values of the
      classes with L <= REBAR_I1_MAXLOOPS (default 3), which must agree with the stored values to their 50 digits
      (|difference| <= 1e-45 |f| + 1e-80 sum|terms|): the stored values are what the independent code gives.
PASS: (a) and (b) hold.  Without mpmath (b) cannot run: SKIP after (a) has passed (exit 2).
"""
import math, os, random, sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the forest expansion agrees pointwise with an independent 100-digit evaluation (Sec. II B 3, II E)"
QUICK = True
SAMPLE = os.path.join(C.RESULTS, "reference", "sample.tsv")
MP = os.path.join(C.HERE, "reference", "rebar_mp.py")
# the sample: (kind, L, list, every k-th graph, point types)
PLAN = [("b", 3, "gamma4_L3.txt", 1, ("base", "face")), ("b", 4, "gamma4_L4.txt", 1, ("base", "face")),
        ("b", 5, "gamma4_L5.txt", 1, ("base",)), ("b", 6, "gamma4_L6.txt", 10, ("base",)), ("b", 7, "gamma4_L7.txt", 400, ("base",)),
        ("h", 3, "gamma2_L3.txt", 1, ("base", "face")), ("h", 4, "gamma2_L4_with_selfloops.txt", 1, ("base",)),
        ("h", 4, "gamma2_L4.txt", 1, ("face",)), ("h", 5, "gamma2_L5.txt", 1, ("base",)), ("h", 6, "gamma2_L6.txt", 1, ("base",)),
        ("h", 7, "gamma2_L7.txt", 32, ("base",)),
        ("c", 3, "gamma2_L3_with_selfloops.txt", 1, ("base", "face")), ("c", 4, "gamma2_L4_with_selfloops.txt", 1, ("base",)),
        ("c", 5, "gamma2_L5.txt", 1, ("base",)), ("c", 6, "gamma2_L6.txt", 3, ("base",)), ("c", 7, "gamma2_L7.txt", 64, ("base",))]


def read_list(path):
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            out.append(next("".join(t.split()) for t in line.rstrip("\n").split("\t") if "|" in t))
    return out


def make_point(kind, nk, typ):
    nlines = sum(1 for part in nk.split("|") for ch in part if ch != "e")
    rng = random.Random("%s|%s|%s" % (kind, nk, typ))
    lx = [rng.uniform(math.log(0.05), 0.0) for _ in range(nlines)]
    if typ == "face":
        S = rng.randrange(1, (1 << nlines) - 1)
        lx = [v + (math.log(1e-3) if (S >> e) & 1 else 0.0) for e, v in enumerate(lx)]
    return " ".join("%.17g" % v for v in lx)


def mp_eval(job):
    kind, L, nk, typ, lx = job
    import mpmath as mp
    R = C.load("rebar_mp", MP)
    r = R.Rbar(*R.from_nickel(nk), kind)
    v, s = r.value(R.point(lx))
    return "\t".join([kind, str(L), nk, typ, lx, mp.nstr(v, 50), mp.nstr(s, 10), str(r.nforests)])


def make_sample():
    jobs = []
    for kind, L, lst, step, types in PLAN:
        for i, nk in enumerate(read_list(os.path.join(C.LISTS, lst))):
            if i % step == 0:
                jobs += [(kind, L, nk, t, make_point(kind, nk, t)) for t in types]
    print("evaluating %d points in 100-digit arithmetic on %d processes ..." % (len(jobs), C.JOBS))
    sys.stdout.flush()
    rows = C.pmap(mp_eval, jobs)
    os.makedirs(os.path.dirname(SAMPLE), exist_ok=True)
    with open(SAMPLE, "w") as fh:
        fh.write("# I1 sample: written by checks/I1_pointwise.py --make-sample with checks/reference/rebar_mp.py (100 digits)\n")
        fh.write("# kind\tL\tnickel\tpoint\tlog x_e (exact doubles)\tf(x) prod x (sum x)^-pw\tsum |forest terms|\tforests\n")
        fh.write("".join(r + "\n" for r in rows))
    print("wrote %s (%d rows)" % (C.rel(SAMPLE), len(rows)))


def mp_python():
    try:
        import mpmath  # noqa: F401
        return sys.executable
    except ImportError:
        p = os.environ.get("REBAR_MPMATH_PYTHON")
        if p and C.run([p, "-c", "import mpmath"])[0] == 0:
            return p
    return None


def main(argv):
    if argv[:1] == ["--make-sample"]:
        if not mp_python() == sys.executable:
            sys.exit("--make-sample needs mpmath in this interpreter")
        make_sample()
        return
    rows = []
    with open(SAMPLE) as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                f = line.rstrip("\n").split("\t")
                rows.append({"kind": f[0], "L": int(f[1]), "nickel": f[2], "type": f[3], "lx": f[4], "f": float(f[5]),
                             "sabs": float(f[6])})
    print("stored sample %s: %d points, %d classes (b, h, c at 3-7 loops), values from checks/reference/rebar_mp.py at 100 digits"
          % (C.rel(SAMPLE), len(rows), len({(r["kind"], r["nickel"]) for r in rows})))
    exe = C.cpu_binary()
    if not exe:
        C.verdict("I1", True, "the stored sample is there but nothing was compared", skip="the CPU binary (make -C integrator cpu VARIANTS=e16 failed)")
    # (a) rebar's expansion at the stored points
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["nickel"]), []).append(r)
    worst = {"base": 0.0, "face": 0.0, "vanish": 0.0}
    nbad, nvan, nrows = 0, 0, {}
    for (kind, nk), rs in sorted(groups.items()):
        st, so, se = C.run([exe, "eval", "--kind", kind, "--nickel", nk], stdin="".join(r["lx"] + "\n" for r in rs))
        out = so.split("\n")
        if st or len([x for x in out if x.strip()]) != len(rs):
            print("  %s %s: rebar_cpu eval failed: %s" % (kind, nk, se.strip()))
            nbad += 1
            continue
        for r, o in zip(rs, out):
            v = float(o.split()[0])
            d = abs(v - r["f"])
            if r["f"] == 0 or abs(r["f"]) < 1e-60 * r["sabs"]:
                nvan += 1
                x = d / r["sabs"] if r["sabs"] else d
                worst["vanish"] = max(worst["vanish"], x)
                ok = x <= 1e-16
            elif r["type"] == "base":
                x = d / abs(r["f"])
                worst["base"] = max(worst["base"], x)
                ok = x <= 1e-12
            else:
                worst["face"] = max(worst["face"], min(d / r["sabs"], d / abs(r["f"])))
                ok = d <= 1e-16 * r["sabs"] or d <= 1e-12 * abs(r["f"])
            nrows[r["L"]] = nrows.get(r["L"], 0) + 1
            if not ok:
                nbad += 1
                print("  MISMATCH %s %s %s: rebar %.17g, 100 digits %.17g (sum|terms| %.3g)" % (kind, nk, r["type"], v, r["f"], r["sabs"]))
    print("(a) rebar_cpu eval (long double) vs 100 digits at %d points (%s): largest relative difference %.2e at base "
          "points, %.2e at face points (the smaller of |difference|/|f| and |difference|/sum|terms|); %d points of vanishing graphs, largest |f_rebar - f_mp|/sum|terms| %.2e; %d mismatches"
          % (len(rows), ", ".join("L=%d: %d" % kv for kv in sorted(nrows.items())), worst["base"], worst["face"], nvan, worst["vanish"], nbad))
    summary = "%d points of %d classes at 3-7 loops: rebar vs 100 digits within %.1e (base points), the %d points of vanishing graphs vanish" % (
        len(rows), len(groups), worst["base"], nvan)
    if nbad:
        C.verdict("I1", False, summary + "; %d mismatches" % nbad)
    # (b) the stored values are those of the independent code
    py = mp_python()
    if not py:
        C.verdict("I1", True, summary + "; stored values not recomputed", skip="mpmath (pip install mpmath, or REBAR_MPMATH_PYTHON=<python with mpmath>) for the live 100-digit evaluation")
    maxl = os.environ.get("REBAR_I1_MAXLOOPS", "3")
    st, so, se = C.run([py, "-I", "-B", MP, "verify", SAMPLE, "--max-loops", maxl])
    print("(b) %s%s" % (so.strip(), (" [" + se.strip() + "]") if se.strip() else ""))
    C.verdict("I1", st == 0, summary + "; the independent code reproduces the stored values (L <= %s) to their 50 digits" % maxl if st == 0 else
              summary + "; the independent code does NOT reproduce the stored values")


if __name__ == "__main__":
    main(sys.argv[1:])
