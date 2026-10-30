#!/usr/bin/env python3
# tests.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# The checks behind 'make -C series check': the Callan-Symanzik algebra against Phi4tools (round trip), the
# reproduction of the published eight-loop numbers from the ledger, the pooling formula against independent
# combinations and, exactly, against the integrator's own accumulators, and the merge rules and their idempotency on
# a scratch ledger.
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
"""tests.py -- series/ self-tests (make check).

usage: tests.py [--bin integrator/bin/rebar_cpu]   (the exact accumulator test needs the CPU binary; skipped without)
"""
import fnmatch, glob, json, math, os, re, shutil, struct, subprocess, sys, tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebar_ledger as RL  # noqa: E402
import coefficients as CO  # noqa: E402

FAILS = []


def check(ok, what):
    print("%s  %s" % ("PASS" if ok else "FAIL", what))
    if not ok:
        FAILS.append(what)


def tool(ledger, *opts):
    r = subprocess.run([sys.executable, "-I", os.path.join(RL.HERE, "rebar-merge.py"), "--ledger", ledger,
                        "--archive", ledger + "-archive"] + list(opts),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def merge(ledger, path, *opts):
    return tool(ledger, *(list(opts) + [path]))


def contribs(order, pattern):
    """the run files of the ledger's contributions whose id matches (format 1: in the ledger; format 3: archived)"""
    e = RL.load_orders(RL.LEDGER)["orders"][order]
    pat = re.sub(r"\.json$", "", pattern)
    return sorted(RL.contribution_file(RL.LEDGER, order, c["id"]) for c in e["contributions"] if fnmatch.fnmatch(c["id"], pat))


def stem(f):
    return re.sub(r"\.json(\.xz)?$", "", os.path.basename(f))


def snapshot(d):
    return {os.path.relpath(p, d): RL.sha256_file(p) for x in ("*.json", "*.tsv")
            for p in glob.glob(os.path.join(d, "**", x), recursive=True)}


def cleanup(d):
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(d + "-archive", ignore_errors=True)


def test_roundtrip():
    sys.path.insert(0, RL.TOOLS)
    import refs
    res = CO.compute(policy="refs")
    worst = 0.0
    for N in range(5):
        pt = {"beta": refs.beta_coeffs(N), "eta": refs.eta_coeffs(N), "nu": refs.nu_coeffs(N)}
        for f, c in pt.items():
            for n in range(1, len(c)):
                v = c[n][0]
                w = res["coefficients"][f][str(n)]["N"][str(N)][0]
                worst = max(worst, abs(w - v) / abs(v) if v else abs(w))
    check(worst < 1e-12, "Phi4tools -> B, H, C (refs.py) -> CS relations -> Phi4tools: beta, eta, nu to ubar^8 at "
          "N = 0..4, max rel. diff %.1e" % worst)


def test_reproduction():
    # the two replicas of 14-15 Oct (five H200) combined by inverse variance, as recorded by campaign
    # 2026-10-15-h200x5-night; later contributions to the same orders do not change this check
    import glob
    o = RL.load_orders(RL.LEDGER)["orders"]
    for name, val, err in (("b8", 30.4698252, 1.38e-4), ("h8", 0.0375077814, 3.79e-7)):
        runs = [RL.load_contribution_header(RL.LEDGER, name, stem(f)) for f in contribs(name, "20261014-h200x5-%s-r*-seed*" % name)]
        w = [1 / r["order"]["err"] ** 2 for r in runs]
        comb = sum(wi * r["order"]["sum"] for wi, r in zip(w, runs)) / sum(w); cerr = sum(w) ** -0.5
        check(len(runs) == 2 and abs(comb - val) < 1e-7 * abs(val) and abs(cerr - err) < 0.01 * err,
              "%s: the two night replicas combined by inverse variance: %.10g +- %.3g (recorded: %.10g +- %.3g)" % (name.upper(), comb, cerr, val, err))
        e = o[name]
        check(abs(e["sum"] - comb) < 3 * cerr,
              "%s pooled class by class over %d contribution(s): %s, %+.2f sigma from the night combination"
              % (name.upper(), len(e.get("contributions", [])), RL.fmt_ve(e["sum"], e["err"]), (e["sum"] - comb) / cerr))
    res = CO.compute()
    e8 = res["coefficients"]["eta"]["8"]["N"]["1"]
    b9 = res["coefficients"]["beta"]["9"]["N"]["1"]
    check(abs(e8[0] - 0.00291057) < 1e-7, "eta at ubar^8, N = 1: %s (night campaign: 0.00291057(3))" % RL.fmt_ve(e8[0], e8[1], 1))
    check(abs(b9[0] + 2.36642) < 2e-5, "beta at ubar^9, N = 1: %s (night campaign: -2.36642(1))" % RL.fmt_ve(b9[0], b9[1], 1))
    e9 = res["coefficients"]["eta"]["9"]["N"]["1"]; n9 = res["coefficients"]["nu"]["9"]["N"]["1"]
    check(abs(e9[0] + 0.0051034) < 2e-6, "eta at ubar^9, N = 1: %s (weekend run of 15-16 Oct: -0.0051034(1))" % RL.fmt_ve(e9[0], e9[1], 1))
    check(abs(n9[0] - 0.061330) < 1e-5, "nu at ubar^9, N = 1: %s (weekend run of 15-16 Oct: 0.061330(2))" % RL.fmt_ve(n9[0], n9[1], 1))


def scratch(files):
    d = tempfile.mkdtemp(prefix="rebar-ledger-test-")
    out = [merge(d, f, "--id", stem(f)) for f in files]
    return d, out


def test_pooling():
    # the 8 b4 replicas: their order sums combined by inverse variance: 0.9671903(47)
    reps = [f for f in contribs("b4", "*-r[1-8]-seed*.json")]
    d, out = scratch(reps)
    e = RL.load_orders(d)["orders"]["b4"]
    check(len(reps) == 8 and abs(e["sum"] - 0.9671903) < 0.3 * 4.7e-6 and abs(e["err"] - 4.7e-6) < 0.05e-6 * 2,
          "B4, 8 replicas pooled by class: %s (inverse-variance mean of the sums: 0.9671903(47))" % RL.fmt_ve(e["sum"], e["err"]))
    cleanup(d)
    # the ten H4 runs (V, P, 8 replicas) class by class (check N8) (contributions in units of 1e-4)
    ten = contribs("h4", "*-val-*.json") + contribs("h4", "*-prod-*.json") + contribs("h4", "*-r[1-8]-seed*.json")
    d, out = scratch(ten)
    cl = {r["nickel"]: r for r in RL.load_classes(d, "h4")}
    table = {"e112|e3|333||": (-2.115588, 28e-6), "e112|23|33|e|": (31.731246, 99e-6), "e112|33|e33||": (-1.343263, 63e-6),
             "e123|e23|33||": (18.718462, 80e-6)}
    worst = max(abs(cl[k]["contrib"] * 1e4 - v) / e for k, (v, e) in table.items())
    errs = max(abs(cl[k]["contrib_err"] * 1e4 - e) / e for k, (v, e) in table.items())
    s = RL.load_orders(d)["orders"]["h4"]
    check(len(ten) == 10 and worst < 0.02 and errs < 0.02 and abs(s["sum"] * 1e4 - 46.99086) < 0.1 * 14e-5,
          "H4, ten runs pooled by class (largest deviation %.3f sigma, errors within %.1f%%; "
          "sum %s)" % (worst, 100 * errs, RL.fmt_ve(s["sum"] * 1e4, s["err"] * 1e4)))
    cleanup(d)


def test_merge_rules():
    reps = contribs("b4", "*-r[1-2]-seed*.json")
    d, out = scratch(reps)
    check(all(r == 0 for r, _ in out), "scratch ledger: two b4 replicas merged")
    before = snapshot(d)
    again = [merge(d, f, "--id", stem(f)) for f in reps]
    check(all(r == 0 and "nothing to do" in t for r, t in again) and snapshot(d) == before,
          "re-merging the same contributions is a no-op (every file unchanged)")
    r, t = merge(d, reps[0], "--id", "same-seed-other-id")
    check(r == 1 and "seed 5001 already merged" in t and snapshot(d) == before, "the same seed under another id is refused")
    R = RL.read_contribution(reps[0])
    td = tempfile.mkdtemp(prefix="rebar-tampered-")
    tmp = os.path.join(td, "tampered.json")
    R["seed"] = "999999"
    c = max(R["classes"], key=lambda c: abs(c["contrib"]))
    c["mean_w"] += 10 * c["sd_w"] / math.sqrt(c["n"])          # a 10-sigma shift of one class
    json.dump(R, open(tmp, "w"))
    r, t = merge(d, tmp)
    check(r == 1 and "inconsistent with the ledger" in t and c["nickel"] in t and snapshot(d) == before,
          "a contribution with one class shifted by 10 sigma is refused, and the class is named")
    R = RL.read_contribution(reps[0])
    R["seed"] = "999998"
    R["classes"][0]["nickel"] = "ee12|e34|e34|e4|e|"            # not a b4 graph of the list
    json.dump(R, open(tmp, "w"))
    r, t = merge(d, tmp)
    check(r == 1 and "not in gamma4_L4.txt" in t, "a class outside the order's list is refused")
    R = RL.read_contribution(reps[0])
    R["seed"], R["omega_scale"] = "999997", 1.0
    json.dump(R, open(tmp, "w"))
    r, t = merge(d, tmp)
    check(r == 1 and "omega_scale" in t, "a non-production density is refused")
    R = RL.read_contribution(reps[0])
    R["seed"], R["program"] = "999996", "rebar-cpu"
    json.dump(R, open(tmp, "w"))
    r, t = merge(d, tmp)
    check(r == 1 and "program" in t, "an unknown program name is refused")
    check(all(RL.PROGRAM_RE.match(p) for p in ("rebar_cpu", "rebar_gpu-e18", "rbar_cpu-e20", "rbar_gpu-v030")) and
          not RL.PROGRAM_RE.match("rebar"), "the program check accepts the current and the recorded program names")
    shutil.rmtree(td)
    rb = tool(d, "--rebuild")
    check(rb[0] == 0 and snapshot(d) == before, "--rebuild reproduces orders.json byte for byte from the stats")
    rc = tool(d, "--check")
    check(rc[0] == 0 and "every invariant holds" in rc[1], "--check: checksums, samples and coverage of the manifest, statistics")
    rp = tool(d, "--replay")
    check(rp[0] == 0 and "bit for bit" in rp[1], "--replay rebuilds the ledger from the archived runs bit for bit")
    cleanup(d)


def unhex(h):
    """RbAcc (12 doubles) from the checkpoint's hex words"""
    f = [struct.unpack(">d", bytes.fromhex(h[16 * i:16 * i + 16]))[0] for i in range(12)]
    return dict(zip(("n", "s", "sc", "q", "qc", "q3", "q4", "sabs2", "maxw", "nbad", "ncanc", "maxsabs"), f))


def test_exact(binary):
    if not binary or not os.access(binary, os.X_OK):
        print("SKIP  exact pooling against the integrator's accumulators (no CPU binary; make -C integrator cpu VARIANTS=e16)")
        return
    d = tempfile.mkdtemp(prefix="rebar-exact-")
    env = dict(os.environ, OMP_NUM_THREADS="2")
    runs = []
    for seed in ("101", "102"):
        out = os.path.join(d, "b3_%s.json" % seed)
        subprocess.run([binary, "run", "--kind", "b", "--loops", "3", "--list", RL.list_path("b", 3), "--drop-1vr",
                        "--pilot", "1024", "--samples", "200000", "--seed", seed, "--threads", "2", "--out", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, check=True)
        runs.append(RL.load_json(out))
    worst = 0.0
    for nk in [c["nickel"] for c in runs[0]["classes"] if c.get("n")]:
        tot = {"n": 0.0, "S": 0.0, "Q": 0.0}
        parts = []
        for R in runs:
            c = next(c for c in R["classes"] if c["nickel"] == nk)
            parts.append(RL.run_moments(c))
            row = next(r for r in R["checkpoint"]["classes"] if r[0] == nk)
            for h in (row[9], row[10]):                       # acc and part
                a = unhex(h)
                tot["n"] += a["n"]; tot["S"] += a["s"] + a["sc"]; tot["Q"] += a["q"] + a["qc"]
        n, m, M2, _ = RL.pool_moments(parts)
        m_x = tot["S"] / tot["n"]
        sd_x = math.sqrt((tot["Q"] - tot["S"] ** 2 / tot["n"]) / (tot["n"] - 1))
        worst = max(worst, abs(m - m_x) / abs(m_x), abs(math.sqrt(M2 / (n - 1)) - sd_x) / sd_x, abs(n - tot["n"]))
    shutil.rmtree(d)
    check(worst < 1e-12, "pooling from (n, mean_w, sd_w) = pooling of the exact accumulators (sum w, sum w^2) of two "
          "CPU runs, every class of b3: max rel. diff %.1e" % worst)


def main(argv):
    binary = argv[2] if len(argv) > 2 and argv[1] == "--bin" else os.path.join(RL.ROOT, "integrator", "bin", "rebar_cpu")
    test_roundtrip()
    test_reproduction()
    test_pooling()
    test_merge_rules()
    test_exact(binary)
    print("tests: %s" % ("PASSED" if not FAILS else "%d FAILED" % len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
