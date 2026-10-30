#!/usr/bin/env python3
# N2_golden.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check N2: the GPUs reproduce the CPU's samples to round-off (golden runs on identical random numbers).
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
"""N2 -- the GPUs reproduce the CPU's samples to round-off (golden runs).

Claim (paper, Sec. II C and III A): with the same Philox keys and counters the GPU and the CPU draw the same points
in the same Hepp sectors, so that per class the sample counts agree exactly and the means differ only by the
round-off of the per-sample weights (device exp/log and FMA contraction): within 100 units of the double-precision
round-off of the per-sample terms and within 1e-4 of the statistical error (integrator/tools/golden_check.py),
on one GPU, on each GPU alone and with the work split over all of them.

Inputs: results/checks/golden-cpu/<kind>_L<L>.json (CPU references: samples 0..65535 of every class, seed 2026);
results/checks/golden-gpu/2026-10-14-h200x4-validation/ (rebar 0.3.0 on 4 H200: b 3-6, h 3-6, c 3-5, int4 4-5, b 5 on
each GPU alone and with forced splits) and 2026-10-15-h200x8-weekend/ (rebar 0.4.0 on 8 H200, both limit variants),
copied from the campaigns' result tarballs (whose complete records are kept outside the public repository).
What runs:
  (a) golden_check.py --require-backend cuda on every stored GPU output (the numbers came from the GPUs: no host
      fallback, no dead device, every device's start-up self-test passed), with --min-devices 2 for the forced splits;
  (b) this machine's CPU: `rebar_cpu run --gen --pilot 65536 --samples 0 --seed 2026` for b 3-4, h 3-4, c 3-4 and
      golden_check.py against the golden files (round-off level expected: another libm);
  (c) with a GPU (integrator/bin/rebar_gpu and a usable device): the same golden runs on the GPU, --require-backend cuda.
PASS: (a), (b) and (c) pass.  Without a GPU, (c) cannot run: SKIP after (a) and (b) have passed (exit 2).
"""
import glob, json, os, re, shutil, sys, tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "the GPUs reproduce the CPU's samples to round-off (golden runs; Sec. II C, III A)"
QUICK = True
GOLDEN = os.path.join(C.RESULTS, "golden-cpu")
STORED = os.path.join(C.RESULTS, "golden-gpu")
LIVE = [("b", 3), ("b", 4), ("h", 3), ("h", 4), ("c", 3), ("c", 4)]
CHECK = os.path.join(C.ITOOLS, "golden_check.py")


def summary_line(text):
    m = re.search(r"(\d+) classes, (\d+) failing.*?worst (\S+) ulp of the term scale, (\S+) sigma", text)
    return m.groups() if m else None


def main():
    ok = True
    # (a) stored GPU outputs
    stored = sorted(glob.glob(os.path.join(STORED, "*", "golden_*.json")))
    worst_ulp, worst_sig, nclasses, ndev = 0.0, 0.0, 0, set()
    for p in stored:
        name = os.path.basename(p)
        m = re.match(r"golden_([a-z0-9]+)_L(\d+)", name)
        ref = os.path.join(GOLDEN, "%s_L%s.json" % (m.group(1), m.group(2)))
        opts = ["--require-backend", "cuda"] + (["--min-devices", "2"] if "split" in name else [])
        st, so, se = C.python(CHECK, ref, p, *opts)
        R = json.load(open(p))
        devs = (R.get("backend_info") or {}).get("devices") or []
        st_ok = all((d.get("selftest") or {}).get("ok") and (d.get("selftest") or {}).get("build_signature_match") for d in devs)
        # a GPU: a device of one campaign (the two ran on different nodes; the records keep no GPU UUID)
        for d in devs:
            ndev.add((os.path.basename(os.path.dirname(p)), d.get("label")))
        g = summary_line(so)
        if g:
            nclasses += int(g[0])
            worst_ulp, worst_sig = max(worst_ulp, float(g[2])), max(worst_sig, float(g[3]))
        good = st == 0 and st_ok
        ok = ok and good
        print("(a) %s/%-26s %s %s on %d device(s), self-tests %s: %s" % (os.path.basename(os.path.dirname(p))[:10], name, R.get("program"),
              R.get("version"), sum(1 for d in devs if (d.get("samples") or 0) > 0), "passed" if st_ok else "NOT passed",
              (so.strip().splitlines() or ["?"])[-1].split(": ", 1)[-1] if st == 0 else so.strip()))
    print("(a) %d stored GPU runs (%d classes, %d distinct GPUs): worst %.3g ulp of the term scale, %.3g sigma" % (len(stored), nclasses, len(ndev), worst_ulp, worst_sig))
    if not stored:
        ok = False
    # (b) this CPU
    exe = C.cpu_binary()
    nb = 0
    if exe:
        d = tempfile.mkdtemp(prefix="rebar-N2-")
        try:
            for kind, L in LIVE:
                out = os.path.join(d, "%s_L%d.json" % (kind, L))
                st, so, se = C.run([exe, "run", "--kind", kind, "--loops", str(L), "--gen", "--pilot", "65536", "--samples", "0",
                                    "--seed", "2026", "--threads", C.THREADS, "--out", out])
                st2, so2, se2 = C.python(CHECK, os.path.join(GOLDEN, "%s_L%d.json" % (kind, L)), out)
                good = st == 0 and st2 == 0
                nb += good
                ok = ok and good
                print("(b) this CPU, %s L=%d: %s" % (kind, L, (so2.strip().splitlines() or ["?"])[-1].split(": ", 1)[-1]))
        finally:
            shutil.rmtree(d)
    else:
        print("(b) no CPU binary: not run")
        ok = False
    summary = "%d stored GPU golden runs (rebar 0.3.0 on 4 H200, 0.4.0 on 8 H200; %d GPUs) PASS golden_check --require-backend cuda " \
              "(worst %.2g ulp, %.1e sigma); this CPU %d/%d golden configurations" % (len(stored), len(ndev), worst_ulp, worst_sig, nb, len(LIVE))
    if not ok:
        C.verdict("N2", False, summary)
    # (c) a GPU here?
    gpu = os.path.join(C.INTEGRATOR, "bin", "rebar_gpu")
    usable = False
    if os.access(gpu, os.X_OK):
        st, so, se = C.run([gpu, "devices"], timeout=600)
        try:
            usable = (json.loads(so).get("devices_ok") or 0) > 0
        except ValueError:
            usable = False
    if not usable:
        C.verdict("N2", True, summary, skip="a GPU (integrator/bin/rebar_gpu and a usable CUDA device) for the live part")
    d = tempfile.mkdtemp(prefix="rebar-N2-gpu-")
    try:
        for kind, L in LIVE:
            out = os.path.join(d, "%s_L%d.json" % (kind, L))
            st, so, se = C.run([gpu, "run", "--backend", "cuda", "--kind", kind, "--loops", str(L), "--gen", "--pilot", "65536",
                                "--samples", "0", "--seed", "2026", "--out", out], timeout=1800)
            st2, so2, se2 = C.python(CHECK, os.path.join(GOLDEN, "%s_L%d.json" % (kind, L)), out, "--require-backend", "cuda")
            print("(c) this GPU, %s L=%d: %s" % (kind, L, (so2.strip().splitlines() or ["?"])[-1]))
            ok = ok and st == 0 and st2 == 0
    finally:
        shutil.rmtree(d)
    C.verdict("N2", ok, summary + "; this GPU: %s" % ("all golden configurations PASS" if ok else "FAILURES"))


if __name__ == "__main__":
    main()
