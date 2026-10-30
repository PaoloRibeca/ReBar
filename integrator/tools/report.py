#!/usr/bin/env python3
"""Markdown report of a run_all.sh results directory: environment, build, validation verdicts, throughput per
kernel / loop order / precision, tuning variants, multi-GPU scaling, production runs.  Python 3 stdlib only.

Usage: report.py RESULTS_DIR > REPORT.md
"""
import glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True   # leave no tools/__pycache__ behind
import validate  # noqa: E402


def load(f):
    try:
        return json.load(open(f))
    except Exception:
        return None


def dev_rate(R):
    """aggregate device throughput (samples per kernel-second, summed over devices) and wall-clock throughput of the
    main phase (samples of the launches after the first quarter of the run / their time span)"""
    bi = R.get("backend_info") or {}
    devs = bi.get("devices", [])
    agg = sum(d.get("samples_per_s", 0) for d in devs)
    log = bi.get("launch_log", [])
    wall = float("nan")
    if len(log) >= 4:
        t0 = min(r[1] for r in log); t1 = max(r[1] + r[2] for r in log)
        late = [r for r in log if r[1] >= t0 + 0.25 * (t1 - t0)]
        if late:
            a = min(r[1] for r in late); b = max(r[1] + r[2] for r in late)
            if b > a:
                wall = sum(r[3] for r in late) / (b - a)
    fill = [d.get("mean_fill", 0) for d in devs if d.get("launches")]
    return agg, wall, (sum(fill) / len(fill) if fill else float("nan")), len([d for d in devs if d.get("samples", 0) > 0])


def ns_stats(R):
    v = [x for x in (R.get("backend_info") or {}).get("class_device_ns_per_sample", []) if isinstance(x, (int, float)) and x == x]
    if not v:
        return float("nan"), float("nan")
    return sum(v) / len(v), max(v)


def main():
    d = sys.argv[1]
    print(f"# rebar GPU results: {os.path.basename(os.path.abspath(d))}\n")
    D = load(os.path.join(d, "devices.json"))
    if D:
        print(f"Driver {D.get('driver_version')}, runtime {D.get('runtime_version')}, {D.get('device_count')} device(s), {D.get('devices_ok')} usable; "
              f"program {D.get('program')} {D.get('version')}\n")
        for x in D.get("devices", []):
            p = x.get("properties", {})
            print(f"- {p.get('name')} (cc {p.get('compute_capability')}, {p.get('sms')} SMs, {p.get('global_mem_bytes', 0) / 2**30:.0f} GiB, "
                  f"FP32:FP64 {p.get('single_to_double_perf_ratio')}, ECC {p.get('ecc')}): kernel {p.get('kernel_regs')} regs, "
                  f"{p.get('kernel_local_bytes')} B local, {p.get('blocks_per_sm')} blocks/SM, grid {p.get('grid')}; signature match {x.get('build_signature_match')}")
        print()
    cfg = os.path.join(d, "config.mk")
    if os.path.exists(cfg):
        print("Build configuration (build/config.mk):\n\n```\n" + open(cfg).read().strip() + "\n```\n")
    # validation
    rows = validate.evaluate(d)
    if rows:
        print("## Validation\n\n| verdict | test |\n|---|---|")
        for v, m in rows:
            print(f"| {v} | {m.replace('|', '/')} |")
        print()
    # throughput
    T = []
    for f in sorted(glob.glob(os.path.join(d, "tput_*.json"))):
        R = load(f)
        if not R:
            continue
        agg, wall, fill, nd = dev_rate(R)
        mn, mx = ns_stats(R)
        T.append((os.path.basename(f)[5:-5], R, agg, wall, fill, nd, mn, mx))
    if T:
        print("## Throughput (equal allocation over the classes of an order)\n")
        print("| run | kind | L | precision | classes | GPUs | device samples/s (sum) | wall samples/s (main phase) | mean fill | mean / max ns per sample per class (device) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for name, R, agg, wall, fill, nd, mn, mx in T:
            print(f"| {name} | {R['kind']} | {R['L']} | {R.get('precision')} | {len(R['classes'])} | {nd} | {agg:.3g} | {wall:.3g} | {fill:.2f} | {mn:.3g} / {mx:.3g} |")
        print()
    # scaling
    S = []
    for f in glob.glob(os.path.join(d, "scale_b5_n*.json")):
        R = load(f)
        if R:
            n = int(re.search(r"_n(\d+)\.json$", f).group(1))
            S.append((n,) + dev_rate(R))
    if S:
        S.sort()
        base = S[0][2] if S[0][2] == S[0][2] else S[0][1]
        print("## Multi-GPU scaling (b, 5 loops)\n\n| GPUs | device samples/s | wall samples/s | efficiency (wall) |\n|---|---|---|---|")
        for n, agg, wall, fill, nd in S:
            print(f"| {n} | {agg:.3g} | {wall:.3g} | {wall / (n * base) if base else float('nan'):.2f} |")
        print()
    # production
    P = [(os.path.basename(f)[5:-5], load(f)) for f in sorted(glob.glob(os.path.join(d, "prod_*.json")))]
    P = [(n, R) for n, R in P if R]
    if P:
        print("## Production-style runs (whole orders, Neyman allocation by device time)\n")
        print("| run | value | error | ref | pull | rho (Neyman) | samples | wall s | classes | nbad | final |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for n, R in P:
            o = R["order"]
            print(f"| {n} | {o['sum']:.10g} | {o['err']:.3g} | {o.get('ref', '')} | {o.get('pull', float('nan')):+.2f} | {o.get('rho_neyman_per_sample', 0):.3g} | "
                  f"{o.get('samples', 0):.3g} | {o.get('wall_s', 0):.0f} | {len(R['classes'])} | {o.get('nbad', 0):.0f} | {R.get('final')} |")
        print()
    print("Raw data: every run's JSON (classes[], order, backend_info with per-device self-tests, launch_log, ns per sample "
          "per class), logs/*.{cmd,out,err}, environment.txt, nvidia-smi-q.txt, SUMMARY.txt.")


if __name__ == "__main__":
    main()
