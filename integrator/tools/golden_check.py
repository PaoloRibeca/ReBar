#!/usr/bin/env python3
"""Check a run against a golden reference run with the same seed and sample ranges (typically: a GPU run of
`run --pilot 65536 --samples 0 --seed 2026` against ../results/checks/golden-cpu/K_LL.json from the CPU).

Usage: golden_check.py GOLDEN.json RUN.json [--max-ulp C] [--max-sigma S] [--require-backend cuda]
                       [--min-devices K]

Same Philox keys and counters => same Hepp sectors and the same points up to the last bits, so per class
  n, nbad                       must be identical,
  mean_w                        may differ only by round-off of the per-sample weights.  Reported as
      ulp  = |dmean| / (eps * rms_i sum_k |term_k|)   (round-off units of the term scale; eps = 2^-52), and
      sig  = |dmean| / (sd_w / sqrt(n))               (units of the statistical error),
  sd_w, maxw_over_absmean       relative differences (informational).
PRIMARY criterion: ulp <= C (default 100).  SECONDARY: sig <= S (default 1e-4; one missing or wrong sample moves
the mean by ~1/sqrt(n) = 4e-3 sigma at n = 65536, so this still catches any real defect).  The output says which
criterion fired.  Calibration: moving every exp/log result by up to +-2 ulp on the CPU
(bin/rebar_cpu_perturb, a proxy for CUDA's libm) gives ulp <= 0.81 and sig <= 2.9e-8 over b, h, c, int4 at 4-5
loops; FMA contraction on the device is not emulated and adds round-off of the same order.

--require-backend cuda: FAIL (exit 3) unless the numbers really came from the GPU(s): backend == "cuda" (not
"cuda->cpu"), backend_info.fallback_units == 0 and every device alive.  --min-devices K: FAIL unless at least K
devices processed samples (multi-GPU split).  The per-device sample split is printed.
Exit status: 0 PASS, 1 numbers differ, 2 configuration mismatch, 3 not (all) computed on the required backend.
"""
import json, math, sys

EPS = 2.0 ** -52


def main():
    args, opts = [], {}
    it = iter(sys.argv[1:])
    for x in it:
        if x.startswith("--"):
            opts[x[2:]] = next(it)
        else:
            args.append(x)
    if len(args) != 2:
        print(__doc__)
        return 2
    G, R = (json.load(open(f)) for f in args)
    cmax = float(opts.get("max-ulp", 100))
    smax = float(opts.get("max-sigma", 1e-4))
    for k in ("kind", "L", "seed", "omega_scale", "streams"):
        if G.get(k, "legacy" if k == "streams" else None) != R.get(k, "legacy" if k == "streams" else None):
            print(f"configuration differs: {k} = {G.get(k)} vs {R.get(k)}  -> FAIL")
            return 2
    cg, cr = G["classes"], R["classes"]
    if [c["nickel"] for c in cg] != [c["nickel"] for c in cr]:
        print("class lists differ  -> FAIL")
        return 2
    # where did the numbers come from?
    be = R.get("backend", "?")
    bi = R.get("backend_info") or {}
    devs = bi.get("devices") or []
    split = ", ".join(f"{d.get('label', '?')}: {d.get('samples', 0):.4g}{'' if d.get('alive', True) else ' [DEAD: ' + str(d.get('why_dead', '')) + ']'}"
                      for d in devs)
    fb = bi.get("fallback_units", 0) or 0
    active = sum(1 for d in devs if (d.get("samples") or 0) > 0)
    origin = f"backend {be}, devices with samples {active}/{len(devs)}, host-fallback units {fb}" + (f" [{split}]" if split else "")
    origin_bad = []
    req = opts.get("require-backend")
    if req:
        if be != req:
            origin_bad.append(f"backend is {be!r}, not {req!r}")
        if fb:
            origin_bad.append(f"{fb} unit(s) were computed on the host (fallback)")
        dead = [d.get("label", "?") for d in devs if not d.get("alive", True)]
        if dead:
            origin_bad.append("dead device(s): " + ", ".join(dead))
    if "min-devices" in opts and active < int(opts["min-devices"]):
        origin_bad.append(f"only {active} device(s) processed samples (required {opts['min-devices']})")
    bad = 0
    fired = {"n/nbad": 0, "ulp": 0, "sigma": 0}
    worst = {"ulp": 0.0, "sig": 0.0, "dsd": 0.0, "dmax": 0.0}
    for g, r in zip(cg, cr):
        if "n" not in g or "n" not in r:
            continue
        msg = []
        if g["n"] != r["n"] or g["nbad"] != r["nbad"]:
            msg.append(f"n/nbad {g['n']}/{g['nbad']} vs {r['n']}/{r['nbad']}")
            fired["n/nbad"] += 1
        n, sd = g["n"], g["sd_w"]
        rms_sabs = g["roundoff_rms_over_sd"] * sd / 2.2e-16   # the JSON stores 2.2e-16*rms(sum|terms|)/sd
        dm = abs(g["mean_w"] - r["mean_w"])
        ulp = dm / (EPS * rms_sabs) if rms_sabs > 0 else (0.0 if dm == 0 else math.inf)
        sig = dm / (sd / math.sqrt(n)) if sd > 0 else (0.0 if dm == 0 else math.inf)
        dsd = abs(g["sd_w"] - r["sd_w"]) / sd if sd > 0 else 0.0
        dmax = abs(g["maxw_over_absmean"] - r["maxw_over_absmean"]) / max(abs(g["maxw_over_absmean"]), 1e-300)
        worst["ulp"] = max(worst["ulp"], ulp); worst["sig"] = max(worst["sig"], sig)
        worst["dsd"] = max(worst["dsd"], dsd); worst["dmax"] = max(worst["dmax"], dmax)
        if ulp > cmax:
            fired["ulp"] += 1
            msg.append(f"mean_w {g['mean_w']!r} vs {r['mean_w']!r}: {ulp:.3g} round-off units > {cmax:g} (primary criterion)")
        elif sig > smax:
            fired["sigma"] += 1
            msg.append(f"mean_w {g['mean_w']!r} vs {r['mean_w']!r}: {sig:.3g} sigma > {smax:g} (secondary criterion)")
        if msg:
            bad += 1
            print(f"  {g['nickel']:30s} " + "; ".join(msg))
    o = abs(G["order"]["sum"] - R["order"]["sum"]) / G["order"]["err"] if G["order"]["err"] > 0 else 0.0
    verdict = "PASS" if not bad and not origin_bad else "FAIL"
    crit = ", ".join(f"{k}: {v}" for k, v in fired.items() if v)
    print(f"  origin: {origin}")
    for x in origin_bad:
        print(f"  NOT A GPU RESULT: {x}")
    print(f"{args[1]} vs golden {args[0]}: {len(cg)} classes, {bad} failing{' (' + crit + ')' if crit else ''}; worst {worst['ulp']:.3g} ulp of the term "
          f"scale, {worst['sig']:.3g} sigma; max rel d(sd_w) {worst['dsd']:.2e}, d(maxw) {worst['dmax']:.2e}; order sum differs by {o:.3g} sigma"
          f"{'; ' + '; '.join(origin_bad) if origin_bad else ''}  -> {verdict}")
    if bad:
        return 1
    return 3 if origin_bad else 0


if __name__ == "__main__":
    sys.exit(main())
