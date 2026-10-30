#!/usr/bin/env python3
"""PASS / WARN / FAIL verdicts for a run_all.sh results directory (Python 3 stdlib only).

Usage: validate.py RESULTS_DIR            (prints one line per test and a final summary line)

Tests (each only if its files exist):
  selftest_*.json    every device passed its start-up self-test (FP64 exp/log <= 2 ulp, sqrt/fma exact, Philox
                     known answers, build signature, 4096 samples per class identical sector / |dw| <= 1e-11 Sum|terms|,
                     one chunk through the production kernel vs the host)
  golden_*.json      GPU vs CPU on identical Philox counters (tools/golden_check.py --require-backend cuda):
                     n, nbad identical, means within 100 round-off units (and 1e-4 sigma); split runs must use >= 2 GPUs
  val_*.json         order sums vs references (exact B2 = 1/2, H2 = 1/81, C2 = 1/12; Phi4tools at 3-5 loops):
                     |pull| <= 3 PASS, <= 4 WARN, else FAIL; the run must be final, on the GPU(s), without host fallback,
                     and with nbad = 0 (else WARN)
  replicas_*/        8 independent seeds: pulls ~ N(0,1) (chi2 test, p > 1e-3) and per-class scatter chi2 (p > 1e-3)
  prod_*.json        whole-order production runs vs references where they exist (as val_*)
  logs/*.cmd         every step that ran must have written its --out JSON (FAIL if not; WARN for throughput/scaling)
  devices.json       FAIL if no CUDA device passed the probe (the GPU stages were skipped)
  SUMMARY.txt        WARN if steps were SKIPPED because the global budget was exhausted
Throughput, tuning and scaling runs are informational (tools/report.py tabulates them).
"""
import glob, json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQ = os.environ.get("REBAR_EXPECT_BACKEND", "cuda")   # "emu" only in the test mode of run_all.sh


def gammq(a, x):
    """regularised upper incomplete gamma Q(a, x) (Numerical Recipes)"""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 1000):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x, k):
    return gammq(k / 2.0, x / 2.0) if k > 0 else float("nan")


def load(f):
    try:
        return json.load(open(f))
    except Exception:
        return None


def origin_problems(R):
    p = []
    if R.get("backend") != REQ:
        p.append(f"backend {R.get('backend')}")
    bi = R.get("backend_info") or {}
    if bi.get("fallback_units"):
        p.append(f"{bi['fallback_units']} host-fallback units")
    dead = [d.get("label") for d in bi.get("devices", []) if not d.get("alive", True)]
    if dead:
        p.append("dead: " + ",".join(map(str, dead)))
    return p


def pull_verdict(R, name):
    o = R.get("order", {})
    if "ref" not in o:
        return ("INFO", f"{name}: sum {o.get('sum')} +- {o.get('err')} (no reference)")
    pull = o.get("pull")
    if pull is None:   # zero error on both sides (e.g. C1): exact agreement required
        pull = 0.0 if o.get("rel_dev") == 0 else float("inf")
    v = "PASS" if abs(pull) <= 3 else ("WARN" if abs(pull) <= 4 else "FAIL")
    notes = origin_problems(R)
    if not R.get("final"):
        notes.append("not final (interrupted / timed out)")
    if o.get("nbad", 0):
        notes.append(f"nbad {o['nbad']:.0f}")
        if v == "PASS":
            v = "WARN"
    if origin_problems(R) and v == "PASS":
        v = "WARN"
    val = o["phi4tools_S_L"] if "phi4tools_S_L" in o else o["sum"]
    err = o["phi4tools_S_L_err"] if "phi4tools_S_L_err" in o else o["err"]
    return (v, f"{name}: {R['kind']} L={R['L']} {R.get('precision', 'fp64')}  {val:.10g} +- {err:.3g}  ref {o['ref']:.10g}  pull {pull:+.2f}  "
               f"rel.dev {o.get('rel_dev', float('nan')):+.2e}  samples {o.get('samples', 0):.3g}  wall {o.get('wall_s', 0):.0f}s"
               + (f"  [{'; '.join(notes)}]" if notes else ""))


def evaluate(d):
    rows = []
    # ---- host checks (CPU build on this machine: emulation unit test, golden counters on this CPU)
    for f in sorted(glob.glob(os.path.join(d, "logs", "*emu_*.out"))):
        name = os.path.basename(f)[:-4]
        txt = open(f).read().strip().splitlines()
        last = txt[-1] if txt else "no output"
        rows.append(("PASS" if last.startswith("emulation test passed") else "FAIL", f"{name}: {last}"))
    for f in sorted(glob.glob(os.path.join(d, "cpu_golden_*.json"))):
        name = os.path.basename(f)[:-5]
        gf = os.path.join(HERE, "..", "..", "results", "checks", "golden-cpu", name[len("cpu_golden_"):] + ".json")
        if not os.path.exists(gf):
            continue
        p = subprocess.run([sys.executable, os.path.join(HERE, "golden_check.py"), gf, f], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "no output"
        rows.append(("PASS" if p.returncode == 0 else "FAIL", f"{name} (this CPU vs the golden CPU file; round-off level expected): " + last.split(": ", 1)[-1]))
    # ---- self-tests
    for f in sorted(glob.glob(os.path.join(d, "selftest_*.json"))):
        R = load(f)
        name = os.path.basename(f)[:-5]
        if not R:
            rows.append(("FAIL", f"{name}: no JSON")); continue
        devs = (R.get("backend_info") or {}).get("devices", [])
        bad = [x.get("label") for x in devs if not (x.get("selftest") or {}).get("ok", False)]
        if not devs:
            rows.append(("FAIL", f"{name}: no device")); continue
        st = [x.get("selftest") or {} for x in devs]
        ue = max(s.get("exp_max_ulp", -1) for s in st); ul = max(s.get("log_max_ulp", -1) for s in st)
        dw = max((c.get("max_abs_dw_over_sum_abs_terms", 0) for s in st for c in s.get("classes", [])), default=float("nan"))
        rows.append(("FAIL" if bad else "PASS", f"{name}: {len(devs) - len(bad)}/{len(devs)} devices passed (exp {ue} ulp, log {ul} ulp, max |dw|/Sum|terms| {dw:.2e})"
                     + (f"  FAILED: {bad}" if bad else "")))
    # ---- golden
    gdir = os.path.join(HERE, "..", "..", "results", "checks", "golden-cpu")
    ndev = 0
    D = load(os.path.join(d, "devices.json"))
    if D:
        ndev = D.get("devices_ok", 0)
    for f in sorted(glob.glob(os.path.join(d, "golden_*.json"))):
        name = os.path.basename(f)[:-5]
        cfg = name[len("golden_"):]
        parts = cfg.split("_")
        base = "_".join(parts[:2])
        gf = os.path.join(gdir, base + ".json")
        if not os.path.exists(gf):
            continue
        args = [sys.executable, os.path.join(HERE, "golden_check.py"), gf, f, "--require-backend", REQ]
        if "split" in cfg and ndev > 1:
            args += ["--min-devices", "2"]
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "no output"
        last = last.split(": ", 1)[-1] if " vs golden " in last else last
        rows.append(("PASS" if p.returncode == 0 else "FAIL", f"{name}: {last}"))
    # ---- order sums vs references
    for f in sorted(glob.glob(os.path.join(d, "val_*.json"))) + sorted(glob.glob(os.path.join(d, "prod_*.json"))):
        R = load(f)
        name = os.path.basename(f)[:-5]
        if not R:
            rows.append(("FAIL", f"{name}: no JSON")); continue
        v, msg = pull_verdict(R, name)
        if "fp32" in name and v in ("WARN", "FAIL"):
            v = "INFO"; msg += "  (FP32 evaluation: informational)"
        rows.append((v, msg))
    # ---- replica ensembles
    for rd in sorted(glob.glob(os.path.join(d, "replicas_*"))):
        Rs = [load(f) for f in sorted(glob.glob(os.path.join(rd, "r*.json")))]
        Rs = [R for R in Rs if R and R.get("final") and "pull" in R.get("order", {})]
        name = os.path.basename(rd)
        if len(Rs) < 3:
            rows.append(("WARN", f"{name}: only {len(Rs)} complete replicas")); continue
        pulls = [R["order"]["pull"] or 0.0 for R in Rs]
        n = len(pulls)
        chi2 = sum(p * p for p in pulls)
        mean = sum(pulls) / n
        sd = math.sqrt(sum((p - mean) ** 2 for p in pulls) / (n - 1))
        cls = {}
        for R in Rs:
            for c in R["classes"]:
                if "n" in c and c.get("contrib_err", 0) > 0:
                    cls.setdefault(c["nickel"], []).append((c["contrib"], c["contrib_err"]))
        c2 = 0.0; dof = 0
        for v in cls.values():
            if len(v) < 3:
                continue
            w = [1 / e ** 2 for _, e in v]
            m = sum(wi * x for wi, (x, _) in zip(w, v)) / sum(w)
            c2 += sum(wi * (x - m) ** 2 for wi, (x, _) in zip(w, v)); dof += len(v) - 1
        p1, p2 = chi2_sf(chi2, n), chi2_sf(c2, dof)
        v = "PASS" if p1 > 1e-3 and (dof == 0 or p2 > 1e-3) else "FAIL"
        bad = [o for R in Rs for o in origin_problems(R)]
        if bad and v == "PASS":
            v = "WARN"
        rows.append((v, f"{name}: {n} replicas, mean pull {mean:+.2f}, sd {sd:.2f}, chi2/n {chi2 / n:.2f} (p={p1:.3g}); per-class chi2/dof {c2 / max(dof, 1):.2f} "
                        f"(dof {dof}, p={p2:.3g})" + (f"  [{'; '.join(sorted(set(bad)))}]" if bad else "")))
    # ---- O(N) recombination
    f = os.path.join(d, "val_b5_on.txt")
    if os.path.exists(f):
        lines = open(f).read().strip().splitlines()
        if lines and lines[-1].startswith("max |pull|"):
            mp = float(lines[-1].split()[2])
            rows.append(("PASS" if mp <= 3 else ("WARN" if mp <= 4 else "FAIL"), "O(N) recombination of val_b5 at N=0..3: " + lines[-1]))
    rows += missing_results(d)
    # ---- steps that never started because the global budget was used up: the run is incomplete
    sf = os.path.join(d, "SUMMARY.txt")
    if os.path.exists(sf):
        sk = re.findall(r"\[([^\]\s]+)\] SKIPPED", open(sf).read())
        if sk:
            rows.append(("WARN", f"{len(sk)} step(s) SKIPPED for lack of time ({', '.join(sk[:10])}{', ...' if len(sk) > 10 else ''}): "
                                 "the run is incomplete; rerun with a larger --budget"))
    # ---- no GPU at all: the GPU stages were skipped, so the verdict must not read "ALL PASSED"
    if REQ == "cuda" and D is None:
        rows.append(("FAIL", "devices: no devices.json, i.e. no GPU binary (see logs/build_gpu.err and config.mk): the GPU stages did not run"))
    elif REQ == "cuda" and not D.get("devices_ok", 0):
        rows.append(("FAIL", f"devices: no usable CUDA device ({D.get('device_count', 0)} present, {D.get('devices_ok', 0)} passed the probe"
                             f"{'; ' + D['error'] if D.get('error') else ''}): the GPU stages did not run (see logs/devices.*)"))
    return rows


def missing_results(d):
    """steps that ran (logs/NAME.cmd exists) but left no JSON behind (crash, kill, a device that could not be opened):
    a missing result is a failure, not a silence.  The .cmd files hold absolute paths of the machine that ran them,
    so the JSON is also looked for relative to the results directory (tarball extracted elsewhere)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "logs", "*.cmd"))):
        name = os.path.basename(f)[:-4]
        toks = open(f).read().split()
        for i in range(len(toks) - 1):
            if toks[i] != "--out" or not toks[i + 1].endswith(".json"):
                continue
            o = toks[i + 1].strip("'\"")
            cand = [o, os.path.join(d, os.path.basename(o)), os.path.join(d, os.path.basename(os.path.dirname(o)), os.path.basename(o))]
            if any(os.path.exists(c) for c in cand):
                continue
            info = name.startswith(("tput_", "scale_"))
            rows.append(("WARN" if info else "FAIL", f"{name}: no JSON written ({os.path.basename(o)}): the step crashed, was killed or "
                                                      f"could not open a device; exit status in SUMMARY.txt, details in logs/{name}.err"))
    return rows


def main():
    if len(sys.argv) != 2:
        print(__doc__); return 2
    rows = evaluate(sys.argv[1])
    cnt = {}
    for v, m in rows:
        cnt[v] = cnt.get(v, 0) + 1
        print(f"{v:4s}  {m}")
    print("VALIDATION: " + ", ".join(f"{cnt.get(k, 0)} {k}" for k in ("PASS", "WARN", "FAIL", "INFO")) +
          ("  -> NO RESULTS" if not rows else "  -> FAILURES" if cnt.get("FAIL") else "  -> WARNINGS (see the WARN lines)" if cnt.get("WARN")
           else "  -> ALL PASSED"))
    return 1 if cnt.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
