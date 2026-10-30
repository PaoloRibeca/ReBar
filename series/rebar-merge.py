#!/usr/bin/env python3
# rebar-merge.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Validates contributions (the JSON of one rebar_cpu / rebar_gpu run, version >= 0.3.0, read streaming) against the
# ledger and folds them in: archives each run as received (ledger-archive/<order>/<id>.json.xz), folds its per-class
# registers into the order's stats (Welford registers merged pairwise), adds the exact weights of new classes,
# records the seed and a manifest entry, and recomputes the order's statistics, chunk by chunk of list positions.
# Refuses, with a report, a contribution that fails a rule of doc/LEDGER.md unless --force is given. Re-merging the
# same file is a no-op. --check verifies the ledger's invariants, --replay rebuilds it from the archive.
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
"""rebar-merge.py -- validate runs and fold them into the ledger (doc/LEDGER.md, format 3).

usage: rebar-merge.py [--ledger DIR] [--archive DIR] [--id ID | --label LABEL] [--force] [--dry-run] [--note TEXT]
                      RUN.json[.xz] [RUN.json ...]
       rebar-merge.py [--ledger DIR] --rebuild [ORDER ...]       recompute orders.json's statistics from the stats
       rebar-merge.py [--ledger DIR] --check [ORDER ...]         verify the ledger's invariants
       rebar-merge.py [--ledger DIR] [--archive DIR] --replay    rebuild the ledger from the archived runs, compare

A run is the JSON of a rebar_cpu / rebar_gpu run (plain or .xz, read streaming, any size).  Several files are merged as
one batch: each is validated against the ledger's state and the files of the batch accepted before it, chunk by
chunk of list positions (memory bounded by a chunk); every accepted run is archived as received
(<archive>/<order>/<id>.json.xz, default <ledger>/../ledger-archive; never deleted) and checked to decompress to its
SHA-256 before the ledger is touched; then its per-class registers are folded into the order's stats and the
order's statistics recomputed.  With several files, --id is a template: {n} is the last number in the file's name,
{stem} its name without .json, {i} its position in the batch, e.g. --id 20261017-h200x1-b9-shard{n:02d}.

The id defaults to <date>-<devices>-<order>-<label>-seed<seed> (date of the run's utc, devices e.g. h200x5 or cpu2,
label 'run'); a file already at ledger/<order>/runs/<id>.json (where rebar-refine.sh writes) keeps its name.  Rules
(a failure refuses the contribution unless --force; warnings do not):
  1. known program and version (>= 0.3.0), kind b/h/c, production settings (fp64, salted-v1 streams, omega_scale 0.7,
     omega_shift2 2), no emulated devices, no --dedup;
  2. every class is in the order's list (integrator/data/lists; without a list, in the classes the ledger already has),
     no duplicates; its salt is FNV-1a(kind|L|nickel), its prefactor Gamma(pw) pi^(-L/2) and the order's, its 1VR
     flag the list's, its weight polynomial S c_k / 3^m exactly (integers c_k) and the ledger's, S its value at N = 1;
     a complete contribution has the same class set as the earlier complete ones;
  3. the seed is unused for this (kind, L) in seeds.json (or reserved for this id by rebar-refine.sh);
  4. known kernel build signature, and on GPUs a passed start-up self-test on every device that produced samples;
  5. classes with non-finite samples (nbad > 0) are excluded from the pool and listed;
  6. chi^2 of the per-class values against the pooled ledger (same classes): p > 1e-6, and the largest |z| with
     Bonferroni-corrected p > 1e-6; the report names the worst classes (all of it goes into the manifest).
The same id merged again: identical file (the same SHA-256) -> nothing to do; anything else -> refused (the state is
folded: a run cannot be taken out or replaced)."""
import heapq, math, os, re, shutil, sys, tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebar_ledger as RL  # noqa: E402

P_MIN = 1e-6
PREF_TOL = 1e-12


def devtag(R):
    bi = R.get("backend_info") or {}
    if R.get("backend") == "cuda":
        devs = [d for d in bi.get("devices", []) if d.get("samples", 0) > 0] or bi.get("devices", [])
        name = (devs[0].get("properties", {}).get("name") or devs[0].get("label", "gpu")) if devs else "gpu"
        model = re.sub(r"[^a-z0-9]", "", name.lower().replace("nvidia", "").split()[-1] if name.split() else "gpu")
        return "%sx%d" % (model or "gpu", len(devs))
    if R.get("backend") == "cpu":
        return "cpu%s" % (R.get("threads") or "")
    return str(R.get("backend") or "unknown")


def manifest_args(a, kept=None):
    """The run's arguments as the manifest records them: an absolute path (it names the directories of the machine the
    run was made on) is reduced to its file name, or, with kept (--replay: the manifest's own arguments), taken from
    kept when that has the same file name, so that a manifest whose paths were shortened otherwise still replays."""
    if not isinstance(a, dict):
        return a
    kept = kept if isinstance(kept, dict) else {}
    out = {}
    for k, v in a.items():
        if isinstance(v, str) and v.startswith("/"):
            b = os.path.basename(v)
            w = kept.get(k)
            v = w if isinstance(w, str) and os.path.basename(w) == b else b
        out[k] = v
    return out


def make_id(R, label):
    date = (R.get("utc") or RL.utc_now())[:10].replace("-", "")
    cid = "%s-%s-%s-%s-seed%s" % (date, devtag(R), RL.order_name(R["kind"], R["L"]), label, R.get("seed"))
    return re.sub(r"[^A-Za-z0-9._-]", "_", cid)


def vtuple(v):
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except ValueError:
        return (0,)


def run_rules(R):
    """Rules 1 and 4, from the run-level fields: (fails of rule 1, fails of rule 4, warnings, notes)."""
    f1, f4, warns, info = [], [], [], []
    if not RL.PROGRAM_RE.match(R.get("program") or ""):
        f1.append("program %r is not rebar_cpu / rebar_gpu (nor rbar_cpu / rbar_gpu, the name recorded before 2026-10-27), "
                  "with an optional limit-variant suffix such as -e18" % R.get("program"))
    if vtuple(R.get("version")) < (0, 3, 0):
        f1.append("version %s < 0.3.0 (no weight polynomials)" % R.get("version"))
    elif R.get("version") not in RL.KNOWN_VERSIONS:
        f1.append("version %s is not among the known good %s" % (R.get("version"), ", ".join(RL.KNOWN_VERSIONS)))
    for k, v in RL.PRODUCTION.items():
        if R.get(k) != v:
            f1.append("%s = %r, production value %r (class integrals do not depend on it, but only production "
                      "settings are pooled)" % (k, R.get(k), v))
    if R.get("backend") not in ("cuda", "cpu"):
        f1.append("backend %r (emulated or unknown devices)" % R.get("backend"))
    if (R.get("args") or {}).get("dedup"):
        f1.append("--dedup run: merged Whitney twins are different classes")
    if R.get("build_signature") not in RL.KNOWN_SIGNATURES:
        f4.append("build signature %s is not known good" % R.get("build_signature"))
    bi = R.get("backend_info") or {}
    if R.get("backend") == "cuda":
        for d in bi.get("devices", []):
            if d.get("samples", 0) > 0:
                st = d.get("selftest") or {}
                if not st.get("ok") or not st.get("build_signature_match"):
                    f4.append("device %s produced samples without a passed self-test" % d.get("label"))
        if bi.get("fallback_samples"):
            warns.append("%g samples were computed on the host after a device failure" % bi["fallback_samples"])
    else:
        info.append("cpu backend: no device self-test (the CPU golden checks of integrator/ apply)")
    return f1, f4, warns, info


class Consistency:
    """Rule 6, accumulated class by class: z = (v - V)/sqrt(e^2 + E^2) of the classes in common with the pool."""
    def __init__(self):
        self.chi2, self.k, self.seq, self.top = 0, 0, 0, []
        self.s_c = self.s_p = self.v_c = self.v_p = 0.0

    def add(self, c, p):
        v, e = c["pref"] * c["mean_w"], c["pref"] * c["sd_w"] / math.sqrt(c["n"])
        d = math.sqrt(e * e + p["error"] ** 2)
        if d > 0:
            z = (v - p["value"]) / d
            self.chi2 += z ** 2
            self.k += 1
            self.seq += 1
            heapq.heappush(self.top, (abs(z), -self.seq, (z, c["nickel"], v, e, p["value"], p["error"])))
            if len(self.top) > 5:
                heapq.heappop(self.top)
        w = RL.poly(c["weight_poly"], 1)
        self.s_c += w * v
        self.v_c += (w * e) ** 2
        self.s_p += w * p["value"]
        self.v_p += (w * p["error"]) ** 2

    def result(self):
        """(report lines, failed, detail for the manifest)"""
        if not self.k:
            return (["consistency: no class in common with the ledger yet (first contribution)"], False,
                    {"chi2": 0.0, "dof": 0})
        k, chi2 = self.k, self.chi2
        p = RL.chi2_sf(chi2, k)
        rows = [t[2] for t in sorted(self.top, key=lambda t: (-t[0], -t[1]))]
        pz = min(1.0, k * RL.norm_sf2(rows[0][0]))
        vv = self.v_c + self.v_p
        ds = (self.s_c - self.s_p) / math.sqrt(vv) if vv > 0 else 0
        lines = ["consistency: chi2 %.1f / %d classes (chi2/dof %.3f, p = %.3g); largest |z| %.2f (Bonferroni p = %.3g); "
                 "order sum on the common classes %+.2f sigma from the ledger's"
                 % (chi2, k, chi2 / k, p, abs(rows[0][0]), pz, ds)]
        for z, nk, v, e, pv, pe in rows:
            lines.append("    z %+6.2f  %-32s %s  ledger %s" % (z, nk, RL.fmt_ve(v, e), RL.fmt_ve(pv, pe)))
        detail = {"chi2": chi2, "dof": k, "p": p, "maxz": abs(rows[0][0]), "p_bonferroni": pz, "dsum_sigma": ds,
                  "worst": [[round(z, 3), nk] for z, nk, _, _, _, _ in rows]}
        return lines, (p < P_MIN or pz < P_MIN), detail


def consistency(R, pool, excluded):
    """chi^2 of a run-like contribution's classes against an in-memory pool {nickel: record}; (lines, failed)."""
    acc = Consistency()
    for c in R["classes"]:
        p = pool.get(c["nickel"])
        if not c.get("n") or not p or c["nickel"] in excluded:
            continue
        acc.add(c, p)
    return acc.result()[:2]


# ------------------------------------------------------------------------------------------------- merging
class Item:
    def __init__(self, path, cid):
        self.path, self.cid, self.X, self.out = path, cid, None, []

    def say(self, s):
        self.out.append(s)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def run_summary(h):
    """The per-run statistics of the manifest entry: the integrator's order block (its scalars, as written), the time
    box (requested seconds, device and wall seconds used, the wall fraction of the box), the checkpoint's sessions,
    extensions and rounds, per device its name, self-test, samples, kernel seconds and rate, and the run's own rho_tau
    (relative error times the square root of its device seconds, as costfig.py defines it)."""
    o = h.get("order") or {}
    out = {"order": {k: v for k, v in o.items() if isinstance(v, (int, float, str, bool)) or v is None}}
    sec = _num((h.get("args") or {}).get("seconds"))
    wall, core = _num(o.get("wall_s")), _num(o.get("core_s"))
    out["time_box"] = {"seconds": sec, "wall_s": wall, "core_s": core,
                       "wall_fraction": wall / sec if wall is not None and sec else None}
    ck = h.get("checkpoint") if isinstance(h.get("checkpoint"), dict) else None
    if ck:
        out["checkpoint"] = {k: ck.get(k) for k in ("complete", "pilot_done", "round_done", "rounds", "extensions")}
        out["checkpoint"]["sessions"] = [{k: x.get(k) for k in ("utc", "version", "wall_s", "samples", "round_from",
                                          "round_to", "extended")} for x in ck.get("sessions", [])]
    devs = []
    for d in (h.get("backend_info") or {}).get("devices", []):
        st = d.get("selftest") or {}
        devs.append({"name": (d.get("properties") or {}).get("name") or d.get("label"), "alive": d.get("alive"),
                     "selftest": bool(st.get("ok") and st.get("build_signature_match")) if st else None,
                     "samples": d.get("samples"), "kernel_s": d.get("kernel_s"), "samples_per_s": d.get("samples_per_s")})
    if devs:
        out["device_list"] = devs
    sm, er = _num(o.get("sum")), _num(o.get("err"))
    out["rho_tau"] = abs(er / sm) * math.sqrt(core) if sm and er is not None and core else None
    return out


def is_reference(h):
    """a run on the reference device only (every GPU that produced samples is a RL.REF_DEVICE)"""
    devs = [d for d in (h.get("backend_info") or {}).get("devices", []) if d.get("samples", 0) > 0]
    return h.get("backend") == "cuda" and bool(devs) and all((d.get("properties") or {}).get("name") == RL.REF_DEVICE for d in devs)


def _close(a, b, tol):
    return abs(a - b) <= tol * abs(b)


def check_order(ctx, items, refused, write):
    """Rules 2 and 6 chunk by chunk, every item against the state and the items before it in the batch, folded in
    unless refused; with write, the new stats and weights chunks (pending in ctx.txn)."""
    kind, L, C = ctx.kind, ctx.L, ctx.C
    known = set(ctx.extra) if not ctx.lp else None
    pref = ctx.entry.get("pref")
    for it in items:
        it.mism, it.unknown, it.salt, it.pref, it.frac, it.vr, it.covered = [], [], [], [], [], [], 0
        it.cons, it.excl = Consistency(), set(it.X.excluded)
        it.pooled = it.psamples = it.approx = it.flagged = 0
        it.pw, it.folded, it.others, it.exclpos = None, [], [], []
        it.build, it.build_first = 0.0, 0.0
        it.ref = is_reference(it.X.header)
    first_pref = None
    for k in sorted(set(k for it in items for k in it.X.spool.chunks())):
        st, wt = ctx.read("stats", k), ctx.read("weights", k)
        nm = ctx.names(k)
        lo = k * C
        state = {}
        for it in items:
            X = it.X
            if k not in X.spool.count:
                continue
            cl, ck, dr = X.rows(k)
            cov = {p for p in dr if p < ctx.graphs}
            others = set(dr)
            for pos in sorted(cl):
                c = cl[pos]
                nk, flag = nm[pos - lo] if pos - lo < len(nm) else (None, None)
                if nk != c["nickel"]:
                    raise RL.LedgerError("%s: class %s at position %d is %s in the order (two nickels with the same "
                                         "string hash: run the merge again)" % (it.path, c["nickel"], pos, nk))
                if isinstance(c["build_s"], (int, float)):
                    it.build += c["build_s"]
                if not c["n"]:
                    continue
                if pos < ctx.graphs:
                    cov.add(pos)
                if flag is not None and bool(c["is1vr"]) != flag:
                    it.vr.append(nk)
                if c["salt"] != RL.class_salt(kind, L, nk):
                    it.salt.append(nk)
                ref = pref if pref is not None else first_pref
                pw_ok = isinstance(c["pw"], (int, float)) and _close(c["pref"], RL.class_pref(c["pw"], L), PREF_TOL)
                if not pw_ok or (ref is not None and not _close(c["pref"], ref, PREF_TOL)) or (it.pw is not None and c["pw"] != it.pw):
                    it.pref.append(nk)
                it.pw = c["pw"] if it.pw is None else it.pw
                s = state.get(pos)
                if s is None:
                    s = state[pos] = [RL.stats_regs(st[pos - lo]), wt[pos - lo] or None, None]
                if s[1]:
                    if s[2] is None:
                        s[2] = RL.weight_floats(s[1])
                    wl, Sl = s[2]
                    top = max(abs(x) for x in wl)
                    if not (c["S"] == Sl and len(c["weight_poly"]) == len(wl)
                            and all(abs(a - b) <= 1e-12 * max(1e-300, top) for a, b in zip(c["weight_poly"], wl))):
                        it.mism.append(nk)
                    wnew = None
                else:
                    wnew = RL.weight_fractions(c["weight_poly"], c["S"], L)
                    if wnew is None:
                        it.frac.append(nk)
                if s[0] is not None and nk not in it.excl:
                    R = s[0]
                    sd = math.sqrt(R[RL.M2] / (R[RL.N_] - 1)) if R[RL.N_] > 1 else float("nan")
                    pp = ref if ref is not None else c["pref"]
                    it.cons.add(c, {"value": pp * R[RL.MEAN], "error": pp * sd / math.sqrt(R[RL.N_])})
                elif s[0] is None and known is not None and known and nk not in known:
                    it.unknown.append(nk)
                if nk in it.excl:
                    others.add(pos)
                    it.exclpos.append(pos)
                if it in refused or nk in it.excl:
                    continue
                regs, exact, flagged = RL.run_registers(c, ck.get(pos))
                if s[0] is None and isinstance(c["build_s"], (int, float)):
                    it.build_first += c["build_s"]       # the order's build time counts a class once: its first run
                s[0] = RL.fold(s[0], regs, it.ref)
                it.folded.append(pos)
                if wnew is not None:
                    s[1], s[2] = wnew, None
                if first_pref is None:
                    first_pref = c["pref"]
                it.pooled += 1
                it.psamples += c["n"]
                it.approx += not exact
                it.flagged += flagged
            it.covered += len(cov)
            it.others.extend(sorted(others))
        if write and state:
            changed_s = changed_w = False
            for pos, (R, w, _) in state.items():
                line = RL.stats_line(R)
                if line != st[pos - lo]:
                    st[pos - lo], changed_s = line, True
                if (w or "") != wt[pos - lo]:
                    wt[pos - lo], changed_w = w or "", True
            if changed_s:
                ctx.write("stats", k, st)
            if changed_w:
                ctx.write("weights", k, wt)
    return first_pref


def decide(items, ctx, seeds, order, force, dry, known_empty):
    """The verdict of every item in batch order (on copies of the seeds and the order's state)."""
    sd = dict(seeds.get("orders", {}).get(order, {}))
    exp = ctx.entry.get("classes_expected")
    out = []
    for i, it in enumerate(items):
        X, h = it.X, it.X.header
        f1, f4, warns, info = run_rules(h)
        fails = f1 + f4
        if X.twins:
            fails.append("classes with merged Whitney twins (a --dedup run)")
        if X.dup:
            fails.append("duplicate classes in the contribution")
        if ctx.lp:
            if X.unlisted:
                fails.append("%d classes not in %s: %s" % (len(X.unlisted), os.path.basename(ctx.lp), ", ".join(X.unlisted[:5])))
            full = it.covered >= ctx.graphs > 0
        else:
            full = bool((h.get("args") or {}).get("gen"))
            if known_empty and i == 0:
                warns.append("no list for %s: this contribution defines the order's classes" % order)
            elif it.unknown:
                fails.append("%d classes not among the order's (no list; ledger classes): %s" % (len(it.unknown), ", ".join(it.unknown[:5])))
        for lst, what in ((it.mism, "differ from the ledger's in S or weight polynomial"),
                          (it.salt, "have a salt other than FNV-1a(kind|L|nickel)"),
                          (it.pref, "have a prefactor other than Gamma(pw) pi^(-L/2) or the order's"),
                          (it.frac, "have a weight polynomial that is not S c_k / 3^m with integers c_k"),
                          (it.vr, "have a 1VR flag other than the list's")):
            if lst:
                fails.append("%d classes %s: %s" % (len(lst), what, ", ".join(lst[:5])))
        if full and exp is not None and X.nsampled != exp:
            fails.append("complete contribution with %d classes, the earlier complete ones have %d" % (X.nsampled, exp))
        s = str(h.get("seed"))
        used = sd.get(s)
        if used and used.get("id") != it.cid:
            fails.append("seed %s already %s for %s (%s): same seed = same samples" % (s, used.get("state"), order,
                         used.get("id") or used.get("note")))
        if X.excluded:
            warns.append("%d classes with non-finite samples excluded: %s" % (len(X.excluded), ", ".join(X.excluded[:10])))
        if it.flagged:
            warns.append("%d classes whose checkpoint accumulators disagree with the JSON (n, kurtosis or round-off "
                         "beyond 1e-6): folded from the accumulators" % it.flagged)
        if not h.get("final", True) or h.get("complete") is False:
            info.append("checkpoint / time-boxed run (final %s, complete %s): its counters are consistent" % (h.get("final"), h.get("complete")))
        lines, bad, detail = it.cons.result()
        if bad:
            fails.append("per-class values inconsistent with the ledger (p < %g)" % P_MIN)
        ok = not fails or force
        out.append((it, ok, fails, warns, info, lines, full, detail))
        if ok:
            sd[s] = {"state": "dry-run" if dry else "merged", "id": it.cid}
            if full and exp is None:
                exp = X.names
    return out


def merge_order(ctx, items, seeds, order, force, dry, note, archive, overrides, txn):
    """Validate the items of one order and fold the accepted ones in; returns (status, number accepted)."""
    try:
        return _merge_order(ctx, items, seeds, order, force, dry, note, archive, overrides, txn)
    except BaseException:
        if ctx.txn is not txn:
            ctx.txn.abort()
        raise


def _merge_order(ctx, items, seeds, order, force, dry, note, archive, overrides, txn):
    known_empty = not ctx.lp and not ctx.extra
    refused = set()
    while True:
        ctx.txn = RL.Txn()
        first_pref = check_order(ctx, items, refused, write=not dry)
        res = decide(items, ctx, seeds, order, force, dry, known_empty)
        new = next((r[0] for r in res if not r[1] and r[0] not in refused), None)
        if new is None or new is items[-1]:
            break
        ctx.txn.abort()
        refused.add(new)          # a refused file must not be in the pool of the files after it: check them again
    status, accepted, entry = 0, [], ctx.entry
    for it, ok, fails, warns, info, lines, full, detail in res:
        X = it.X
        for x in info:
            it.say("  note: " + x)
        for x in lines:
            it.say("  " + x)
        for x in warns:
            it.say("  WARNING: " + x)
        for x in fails:
            it.say("  FAIL: " + x)
        if not ok:
            it.say("refused (%d failure(s)); --force merges it anyway" % len(fails))
            status = 1
            continue
        h = X.header
        if dry:
            it.say("dry run: would merge%s" % (" (forced)" if fails else ""))
            seeds["orders"].setdefault(order, {})[str(h["seed"])] = {"state": "dry-run", "id": it.cid}
            continue
        accepted.append((it, fails, warns, lines, full, detail))
    if dry or not accepted:
        ctx.txn.abort()
        return status, 0
    for it, *_ in accepted:       # the archive first: a run is in the ledger only once it is kept
        RL.archive_run(it.path, RL.archive_path(ctx.ledger, order, it.cid, archive), it.X.sha)
    for it, fails, warns, lines, full, detail in accepted:
        X, h = it.X, it.X.header
        bi = h.get("backend_info") or {}
        devs = [d for d in bi.get("devices", []) if d.get("samples", 0) > 0]
        ov = overrides.get(it.cid, {})
        rec = {"id": it.cid, "seed": str(h["seed"]), "program": h.get("program"), "version": h.get("version"),
               "build_signature": h.get("build_signature"), "utc": h.get("utc"),
               "backend": h.get("backend"), "devices": devtag(h), "final": h.get("final"), "complete": h.get("complete"),
               "args": manifest_args(h.get("args"), ov.get("args")), "classes": X.names, "full": full, "excluded": X.excluded,
               "samples": X.samples, "sum": X.s1, "err": X.e1,
               "wall_s": (h.get("order") or {}).get("wall_s"), "core_s": (h.get("order") or {}).get("core_s"),
               "selftest": all((d.get("selftest") or {}).get("ok") and (d.get("selftest") or {}).get("build_signature_match")
                               for d in devs) if h.get("backend") == "cuda" else None,
               "file_sha256": X.sha, "merged_utc": ov.get("merged_utc") or RL.utc_now(),
               "check": lines[0], "check_detail": detail, "forced": fails if fails else None,
               "warnings": warns or None, "note": ov.get("note", note),
               "classes_pooled": it.pooled, "samples_pooled": it.psamples,
               "coverage": RL.coverage(it.folded, sorted(set(it.others)), ctx.graphs, ctx.survivors()),
               "excluded_positions": it.exclpos, "reference_device": it.ref,
               "build_s": it.build, "build_s_first": it.build_first, "cpu": h.get("cpu") or h.get("host_cpu"),
               "registers": "exact" if not it.approx else "approximate" if it.approx == it.pooled else "partly approximate",
               "diagnostics": X.diag}
        rec.update(run_summary(h))
        if it.flagged:
            rec["register_flags"] = it.flagged
        if bi.get("devices_total") is not None:
            rec["devices_total"] = bi.get("devices_total")
        entry["contributions"] = [c for c in entry["contributions"] if c["id"] != it.cid] + [rec]
        if it.approx:
            entry["approximate"] = sorted(set(entry.get("approximate", [])) | {it.cid})
        if full and entry.get("classes_expected") is None:
            entry["classes_expected"] = X.names
        seeds["orders"].setdefault(order, {})[str(h["seed"])] = {"state": "merged", "id": it.cid}
        it.say("accepted%s: %s (%d classes, %.3g samples, sum %s)" % (" (FORCED)" if fails else "", it.cid, X.names,
               rec["samples"], RL.fmt_ve(rec["sum"], rec["err"])))
    if entry.get("pref") is None and first_pref is not None:
        entry["pref"], entry["pw"] = first_pref, accepted[0][0].pw
    entry["rows"] = ctx.C
    if ctx.extra:
        entry["extra"] = ctx.extra
    txn.pending.update(ctx.txn.pending)
    ctx.txn = txn
    RL.order_statistics(ctx, pending=True)
    return status, len(accepted)


def merge(ledger, path, cid=None, label="run", force=False, dry=False, note=None, archive=None):
    return merge_many(ledger, [(path, cid)], label, force, dry, note, archive)


def merge_many(ledger, items, label="run", force=False, dry=False, note=None, archive=None, overrides=None, quiet=False):
    """Validate and fold in several runs; the files of one order are checked chunk by chunk together."""
    say = (lambda s: None) if quiet else print
    orders = RL.ensure_orders(RL.load_orders(ledger))
    if orders.get("format", RL.FORMAT1) != RL.FORMAT3:
        if any(e.get("contributions") for e in orders["orders"].values()):
            say("refused: %s is not a format-3 ledger (doc/LEDGER.md)" % ledger)
            return 2
        orders["format"] = RL.FORMAT3
    seeds = RL.load_seeds(ledger)
    txn, ctxs, touched, status, groups, staged = RL.Txn(), {}, [], 0, {}, []

    def get_order(kind, L):
        o = RL.order_name(kind, L)
        if o not in ctxs:
            ctxs[o] = RL._Order(ledger, o, orders["orders"].setdefault(o, RL.default_entry(kind, L)), txn)
        return ctxs[o]

    try:
        for i, (path, cid) in enumerate(items):
            it = Item(path, cid)
            staged.append(it)
            if len(items) > 10 and not quiet:
                sys.stderr.write("rebar-merge: reading %d/%d %s\n" % (i + 1, len(items), path))
                sys.stderr.flush()
            st = stage(it, ledger, label, get_order)
            if st is not None:
                status = max(status, st)
                continue
            entry = ctxs[it.X.order].entry
            old = next((c for c in entry["contributions"] if c["id"] == it.cid), None)
            twin = next((x for x in groups.get(it.X.order, []) if x.cid == it.cid), None)
            if (old and old["file_sha256"] == it.X.sha) or (twin and twin.X.sha == it.X.sha):
                it.say("already merged (identical file): nothing to do")
                continue
            if old or twin:
                it.say("refused: id %s is already merged (format 3 folds runs irreversibly: a merged run cannot be "
                       "replaced; a continued checkpoint repeats its earlier samples)" % it.cid)
                status = max(status, 1)
                continue
            groups.setdefault(it.X.order, []).append(it)
        for o, its in groups.items():
            st, nacc = merge_order(ctxs[o], its, seeds, o, force, dry, note, archive, overrides or {}, txn)
            status = max(status, st)
            if nacc:
                touched.append(o)
        for it in staged:
            for line in it.out:
                say(line)
        if dry or not touched:
            txn.abort()
            return status
        txn.commit()
        RL.save_orders(ledger, orders)
        RL.save_seeds(ledger, seeds)
    except BaseException:
        txn.abort()
        raise
    finally:
        for it in staged:
            if it.X is not None:
                it.X.close()
    for order in touched:
        e = orders["orders"][order]
        say("merged: %s = %s (rel. %.2e), %d classes, %d contribution(s), %.3g samples, status %s"
            % (order.upper(), RL.fmt_ve(e["sum"], e["err"]), e["rel_err"], e["classes"],
               len(e["contributions"]), e["samples"], e["status"]))
    return status


def stage(it, ledger, label, get_order):
    """Read one file (streaming); returns None, or a status when it cannot be merged at all."""
    path = it.path
    if not os.path.isfile(path):
        it.say("refused: no such file: %s" % path)
        return 2
    X = it.X = RL.Run(path)
    try:
        X.read(get_order)
    except (ValueError, RL.LedgerError) as e:
        it.say("refused: %s is not a run JSON (%s)" % (path, e))
        return 2
    R = X.header
    for k in ("kind", "L", "seed"):
        if k not in R:
            it.say("refused: %s has no %r: not a run JSON" % (path, k))
            return 2
    if "classes" not in X.keys:
        it.say("refused: %s has no 'classes': not a run JSON" % path)
        return 2
    if R["kind"] not in RL.KINDS or X.spool is None:
        it.say("refused: kind %r is not b, h or c (or L %r is not a number of loops)" % (R["kind"], R["L"]))
        return 2
    order = X.order
    if not it.cid:
        ap = os.path.dirname(os.path.abspath(path))
        stem = re.sub(r"\.json(\.xz)?$", "", os.path.basename(path))
        it.cid = stem if ap == os.path.abspath(os.path.join(ledger, order, "runs")) else make_id(R, label)
    it.say("contribution %s (%s %s, %s, seed %s, %d classes) -> %s" % (it.cid, R.get("program"), R.get("version"),
           R.get("utc"), R.get("seed"), X.ncl, order))
    return None


# ------------------------------------------------------------------------------------------------- rebuild, check, replay
def rebuild(ledger, names):
    """orders.json's statistics recomputed from the stats (and, for format 1, from the contributions)"""
    orders = RL.ensure_orders(RL.load_orders(ledger))
    for o in names or [o for o, e in orders["orders"].items() if e.get("contributions")]:
        RL.rebuild_order(ledger, o, orders)
        print("rebuilt %s" % o)
    RL.save_orders(ledger, orders)
    return 0


def check(ledger, names):
    """The invariants of a format-3 ledger: every file's SHA-256 as recorded; per order the sum of n over the classes
    equal to the samples folded in by its contributions, the sum of k (derived from their coverage records) to the
    classes they folded; the statistics of orders.json equal to those recomputed from the stats and the manifest;
    every class with stats has its weights."""
    import json
    orders = RL.load_orders(ledger)
    bad, n = [], 0
    for o, e in orders["orders"].items():
        if names and o not in names or not e.get("contributions"):
            continue
        e2 = json.loads(json.dumps(e))
        ctx = RL._Order(ledger, o, e2)
        try:
            tot = RL.order_statistics(ctx, pending=False, verify=True)
            for k in ctx.chunks():
                for line, w in zip(ctx.read("stats", k), ctx.read("weights", k)):
                    if line and not w:
                        bad.append("%s: a class with stats and no weights in chunk %d" % (o, k))
                        break
        except RL.LedgerError as x:
            bad.append(str(x))
            continue
        recs = e["contributions"]
        if tot["samples"] != sum(c.get("samples_pooled", 0) for c in recs):
            bad.append("%s: sum of n over the classes %d, samples folded by the contributions %d" % (o, tot["samples"], sum(c.get("samples_pooled", 0) for c in recs)))
        if tot["k"] != sum(c.get("classes_pooled", 0) for c in recs):
            bad.append("%s: sum of k (from the coverage records) %d, classes folded by the contributions %d" % (o, tot["k"], sum(c.get("classes_pooled", 0) for c in recs)))
        for key in set(e) | set(e2):
            if json.dumps(e.get(key), sort_keys=True) != json.dumps(e2.get(key), sort_keys=True):
                bad.append("%s: %s in orders.json differs from the stats (%r, recomputed %r)" % (o, key, e.get(key), e2.get(key)))
        n += 1
        print("checked %-4s %7d classes, %3d contribution(s): %s" % (o, tot["classes"], len(recs), "ok" if not bad else "PROBLEMS"))
    for x in bad:
        print("  " + x)
    print("check: %d orders, %s" % (n, "every invariant holds" if not bad else "%d problem(s)" % len(bad)))
    return 1 if bad else 0


POLICY_KEYS = ("kind", "L", "list", "list_sha256", "graphs", "target_rel", "closed", "rows")


def replay(ledger, archive=None, out=None, keep=False):
    """Rebuild the ledger from the archived runs, in manifest order (each order one batch: the state does not depend
    on the batches), into out (default: a temporary directory), and compare it with the ledger: every stats and
    weights file and orders.json, byte for byte."""
    import json
    orders = RL.load_orders(ledger)
    out = out or tempfile.mkdtemp(prefix="rebar-replay-")
    os.makedirs(out, exist_ok=True)
    sk = {"format": RL.FORMAT3, "orders": {}}
    items, ov = [], {}
    for o, e in orders["orders"].items():
        sk["orders"][o] = {k: e[k] for k in POLICY_KEYS if k in e}
        sk["orders"][o]["contributions"] = []
        sk["orders"][o]["status"] = RL.order_status(sk["orders"][o])
        for c in e.get("contributions", []):
            items.append((RL.archive_path(ledger, o, c["id"], archive), c["id"]))
            ov[c["id"]] = {"merged_utc": c.get("merged_utc"), "note": c.get("note"), "args": c.get("args")}
    RL.write_if_changed(os.path.join(out, "orders.json"), RL.dump_json(sk))
    shutil.copyfile(os.path.join(ledger, "seeds.json"), os.path.join(out, "seeds.json"))
    missing = [p for p, _ in items if not os.path.exists(p)]
    if missing:
        print("replay: %d archived runs missing, e.g. %s" % (len(missing), missing[0]))
        return 2
    st = merge_many(out, items, force=True, archive=RL.archive_dir(ledger, archive), overrides=ov, quiet=True)
    diffs = []
    for d, _, fs in os.walk(ledger):
        rel = os.path.relpath(d, ledger)
        for f in fs:
            if f == "orders.json" and rel == "." or re.match(r"^(stats|weights)-\d+\.tsv$", f):
                a, b = os.path.join(d, f), os.path.join(out, rel, f)
                if not os.path.exists(b) or RL.sha256_file(a) != RL.sha256_file(b):
                    diffs.append(os.path.join(rel, f))
    for d, _, fs in os.walk(out):
        rel = os.path.relpath(d, out)
        for f in fs:
            if re.match(r"^(stats|weights)-\d+\.tsv$", f) and not os.path.exists(os.path.join(ledger, rel, f)):
                diffs.append(os.path.join(rel, f) + " (only in the replay)")
    print("replay of %d archived runs into %s: %s" % (len(items), out, "identical to the ledger, bit for bit (stats, weights, orders.json)"
          if not diffs else "%d file(s) differ: %s" % (len(diffs), ", ".join(diffs[:10]))))
    if not keep and not diffs:
        shutil.rmtree(out, ignore_errors=True)
    return 1 if diffs or st > 1 else 0


def main(argv):
    ledger, cid, label, force, dry, note, rb, files = RL.LEDGER, None, "run", False, False, None, False, []
    archive, mode = None, None
    it = iter(argv[1:])
    for a in it:
        if a == "--ledger":
            ledger = next(it)
        elif a == "--id":
            cid = next(it)
        elif a == "--label":
            label = next(it)
        elif a == "--note":
            note = next(it)
        elif a == "--force":
            force = True
        elif a == "--dry-run":
            dry = True
        elif a == "--rebuild":
            rb = True
        elif a in ("--check", "--replay"):
            mode = a[2:]
        elif a == "--archive":
            archive = next(it)
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            files.append(a)
    if rb:
        return rebuild(ledger, files)
    if mode == "check":
        return check(ledger, set(files))
    if mode == "replay":
        return replay(ledger, archive, files[0] if files else None, keep=bool(files))
    if not files:
        print(__doc__)
        return 2
    items = []
    for i, f in enumerate(files):
        stem = re.sub(r"\.json(\.xz)?$", "", os.path.basename(f))
        nums = re.findall(r"\d+", stem)
        if cid and len(files) > 1:
            if "{" not in cid:
                print("refused: with several files --id must be a template ({n}, {stem}, {i})")
                return 2
            items.append((f, cid.format(n=int(nums[-1]) if nums else i, stem=stem, i=i)))
        else:
            items.append((f, cid))
    ids = [c for _, c in items if c]
    if len(ids) != len(set(ids)):
        print("refused: the --id template gives the same id to several files")
        return 2
    return merge_many(ledger, items, label, force, dry, note, archive)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
