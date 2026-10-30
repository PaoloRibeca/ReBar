#!/usr/bin/env python3
"""Compare two rebar run JSONs (e.g. CPU vs GPU, CPU vs emulator, or a run vs ../results/checks/golden-cpu/).

Usage: jsoncmp.py A.json B.json [--tol REL] [--keys k1,k2,...] [--checkpoint] [--quiet]

Compares every per-class numeric field and every 'order' field that does not depend on timing or on the
backend.  Prints, per key, how many classes are bitwise identical and the largest relative difference.
Exit status 0 if every compared value agrees within --tol (default 0 = bitwise), 1 otherwise, 2 on a
structural mismatch (different class lists).

Timing / backend-dependent keys are ignored: build_s, build_expand_s, build_tables_s, core_ns_per_sample, core_s, wall_s, rho_time_sqrt_core_s,
mean_core_ns_per_sample_neyman_time, utc, host, program, version, build, threads, args, backend*, cost_unit, final; also image_bytes
and host_bytes, which depend on the limit variant (the atom_loc table has RB_MAXE columns), not on the numbers.
--checkpoint also compares the "checkpoint" rows of the two runs (rebar >= 0.4.0): per class the committed sample
counter, the partial-task counter, the plan (base, nalloc) and the accumulator bit patterns (hex) must be identical,
and so must the plan header (pilot_done, round_done, rounds, npilot, alloc, total, complete).  Costs and sessions are
not compared.  This is the test of an interrupted-and-resumed run against an uninterrupted one.
"""
import json, math, sys

IGNORE = {"build_s", "build_expand_s", "build_tables_s", "image_bytes", "host_bytes", "core_ns_per_sample", "core_s", "wall_s", "rho_time_sqrt_core_s",
          "mean_core_ns_per_sample_neyman_time", "utc", "host", "program", "version", "build", "threads", "args",
          "backend", "backend_info", "backend_note", "cost_unit", "final"}


def reldiff(a, b):
    if a == b:
        return 0.0
    if a is None or b is None:
        return math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        s = max(abs(a), abs(b))
        return abs(a - b) / s if s > 0 else 0.0
    return math.inf


def main():
    args, opts = [], {}
    it = iter(sys.argv[1:])
    for x in it:
        if x.startswith("--"):
            k = x[2:]
            opts[k] = next(it) if k in ("tol", "keys") else True
        else:
            args.append(x)
    if len(args) != 2:
        print(__doc__)
        return 2
    A, B = (json.load(open(f)) for f in args)
    tol = float(opts.get("tol", 0))
    only = set(opts["keys"].split(",")) if "keys" in opts else None
    ca, cb = A["classes"], B["classes"]
    if [c["nickel"] for c in ca] != [c["nickel"] for c in cb]:
        print("class lists differ")
        return 2
    stats = {}
    for x, y in zip(ca, cb):
        for k, v in x.items():
            if k in IGNORE or (only and k not in only):
                continue
            if isinstance(v, (int, float)) or v is None:
                d = reldiff(v, y.get(k))
                s = stats.setdefault("classes." + k, [0, 0, 0.0])
                s[0] += 1
                s[1] += d == 0
                s[2] = max(s[2], d)
    for k, v in A["order"].items():
        if k in IGNORE or (only and k not in only):
            continue
        if isinstance(v, (int, float)) or v is None:
            d = reldiff(v, B["order"].get(k))
            stats["order." + k] = [1, int(d == 0), d]
    if opts.get("checkpoint"):
        ka, kb = A.get("checkpoint"), B.get("checkpoint")
        if not ka or not kb:
            print("checkpoint: missing in " + ("both" if not ka and not kb else args[0] if not ka else args[1]))
            stats["checkpoint.present"] = [1, 0, math.inf]
        else:
            hdr = ("pilot_done", "round_done", "rounds", "npilot", "alloc", "total", "complete")
            dh = sum(ka.get(k) != kb.get(k) for k in hdr)
            stats["checkpoint.header"] = [len(hdr), len(hdr) - dh, math.inf if dh else 0.0]
            ra, rb = ka["classes"], kb["classes"]
            names = ("nickel", "salt", "ncomp", "natoms", "next", "part_done", "base", "nalloc", "cost_s", "acc", "part")
            if len(ra) != len(rb):
                stats["checkpoint.rows"] = [1, 0, math.inf]
            else:
                for j, k in enumerate(names):
                    if k == "cost_s":
                        continue
                    d = sum(x[j] != y[j] for x, y in zip(ra, rb))
                    stats["checkpoint." + k] = [len(ra), len(ra) - d, math.inf if d else 0.0]
    bad = 0
    for k in sorted(stats):
        n, same, mx = stats[k]
        ok = mx <= tol
        bad += not ok
        if not opts.get("quiet") or not ok:
            print(f"{k:42s} {same:4d}/{n:<4d} bitwise   max rel diff {mx:.3e}  {'ok' if ok else 'DIFFERENT'}")
    print(f"{args[0]} vs {args[1]}: {'AGREE' if not bad else 'DIFFER'} (tol {tol:g}; {bad} key(s) beyond tolerance)")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
