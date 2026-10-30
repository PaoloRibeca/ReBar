#!/usr/bin/env python3
# rebar_ledger.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# The ledger's shared code: paths, graph lists, the per-class statistics of a run and their exact pooling, the
# folded-register storage of format 3 (exact weights, per-class registers merged pairwise, plain TSV chunks) and the
# archive of the runs, the reading API of the consumers (iter_classes, load_classes, iter_contribution_classes,
# load_contribution), seeds, references and number formatting; used by rebar-merge.py,
# coefficients.py, checks.py, tables.py and rebar-refine.sh (through the small command-line
# interface at the end: pick, pending, reserve, status).
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
"""Shared code of the ReBar ledger (doc/LEDGER.md).  Python >= 3.6, standard library only.

Pooling.  A class seen by contributions i (independent seeds) with n_i samples, weight mean m_i and weight standard
deviation s_i (the integrator's s_i^2 = (Q_i - n_i m_i^2)/(n_i - 1), Q_i = sum of w^2) pools exactly as one sample:
    n = sum n_i,   m = sum n_i m_i / n,   Q = sum Q_i = sum [(n_i - 1) s_i^2 + n_i m_i^2],
    s^2 = (Q - n m^2)/(n - 1) = [sum (n_i - 1) s_i^2 + sum n_i (m_i - m)^2] / (n - 1),
the last form being the one evaluated (no cancellation).  The class integral is pref * m with error pref * s / sqrt(n),
its contribution at any N is weight_poly(N) times that.  Kurtosis: the n-weighted mean of the per-run fourth central
moments kappa_i mu2_i^2 over the pooled mu2^2 (the third moments are not in the JSON; the spread of the run means
changes it by O(1/n)).  nbad adds; max|w|/|mean| and the round-off indicators keep their maximum.

Storage (doc/LEDGER.md).  Format 1: <order>/classes.json and the run JSONs under <order>/contributions/.  Format 3
(orders.json "format": "rebar-ledger-3"): per order the weight polynomials as exact fractions (weights-*.tsv) and the
folded registers of every class (stats-*.tsv: Welford's n, mean, m2 merged pairwise by Chan, Golub and LeVeque,
m3, m4 by Pebay's update, extremes, sums, k), plain TSV, one row per list position; the runs are archived outside
git (ledger-archive/<order>/<id>.json.xz) and replay to the same state.  Reading (either format):
    iter_classes(ledger, order, fields=None)          pooled classes, the fields of format 1's classes.json
    load_classes(ledger, order, fields=None)          the same as a list
    iter_contribution_classes(ledger, order, cid)     per class of one run (format 3: from the archive)
    iter_order_contributions(ledger, order)           (id, class) for every run of an order (format 3: the archive)
    load_contribution(ledger, order, cid)             a run as a dict; read_contribution(path) from a file
    load_contribution_header(ledger, order, cid)      its run-level fields (order block with core_s, sum, err, ...)
"""
import array, bisect, hashlib, json, lzma, marshal, math, os, re, shutil, struct, sys, tempfile, time
from fractions import Fraction

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LEDGER = os.environ.get("REBAR_LEDGER") or os.path.join(ROOT, "ledger")     # REBAR_LEDGER: another ledger (tests)
LISTS = os.path.join(ROOT, "integrator", "data", "lists")
TOOLS = os.path.join(ROOT, "integrator", "tools")
KINDS = ("b", "h", "c")
KNOWN_VERSIONS = ("0.3.0", "0.4.0", "0.4.1", "0.4.2", "0.4.3")
# kernel build signatures of the limit variants (the signature covers RB_MAXE, RB_MAXLOOP, RB_MAXATOMS and the kernel):
#   e16 (16 lines, 768 atoms); e18 (18 lines) with 2048, 4096 and 8192 atoms; e20 (20 lines, 16384 atoms)
import re as _re
PROGRAM_RE = _re.compile(r"^(rebar|rbar)_(cpu|gpu)(-e\d+)?(-v\d+)?$")   # rebar_gpu, rebar_gpu-e18, rebar_cpu-e20, build tags;
                                                                     # rbar_* is the name the runs before 2026-10-27 record
KNOWN_SIGNATURES = ("ca352d68edca3efe", "f1d51e636e9875c9", "f56a68de64161b7e", "2c95533b955221d6", "81c3bb8b3e25914c")
PRODUCTION = {"omega_scale": 0.7, "omega_shift2": 2, "precision": "fp64", "streams": "salted-v1"}
FIRST_FRESH_SEED = 100000                       # fresh seeds are above every seed used so far, and above this
# Orders the ledger knows, with their policy: target relative error, or None; "closed" = validation order, never chosen
# by rebar-refine.sh unless named.  L <= 6: validation (exact values or more precise references exist).
POLICY = {"b7": 5e-6, "b8": 1e-5, "b9": None, "h7": 2e-5, "h8": 5e-5, "h9": 1e-4, "h10": None,
          "c7": 2e-5, "c8": 5e-5, "c9": 1e-4, "c10": None}
ENTERS = {"b": ("beta", 1), "h": ("eta", 0), "c": ("nu", 0)}   # B_L enters beta at ubar^(L+1), H_L / C_L at ubar^L


def order_name(kind, L):
    return "%s%d" % (kind, L)


def parse_order(name):
    m = re.match(r"^([bhc])(\d+)$", name or "")
    if not m:
        raise ValueError("not an order name (b8, h9, c7, ...): %r" % name)
    return m.group(1), int(m.group(2))


def list_name(kind, L):
    return ("gamma4_L%d.txt" if kind == "b" else "gamma2_L%d.txt") % L


def list_path(kind, L):
    """The order's list in integrator/data/lists, plain or xz-compressed (the ten-loop four-point list), or None."""
    p = os.path.join(LISTS, list_name(kind, L))
    for q in (p, p + ".xz"):
        if os.path.exists(q):
            return q
    return None


def open_list(path):
    """Text handle on a plain or xz-compressed list."""
    if path.endswith(".xz"):
        import lzma
        return lzma.open(path, "rt")
    return open(path)


def expected_classes(kind, L):
    """Number of non-vanishing classes of an order from its list: graphs that are not one-vertex-reducible (they
    vanish for b and h by rule and for c after the algebraic pruning) and, for h, not generalized tadpoles (both legs
    on one vertex).  Needs the list's 1VR flag column (lists from eight loops up); None without it."""
    lp = list_path(kind, L)
    if not lp or L <= 7:
        return None
    n = 0; flagged = False
    with open_list(lp) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            nick = next((x.strip() for x in f if "|" in x), None)
            if nick is None:
                continue
            flags = [x.strip() for x in f if x.strip().startswith("1VR=")]
            if not flags:
                return None
            flagged = True
            if flags[0] == "1VR=1":
                continue
            if kind == "h" and any(b.startswith("ee") for b in nick.split("|")):
                continue
            n += 1
    return n if flagged else None


def iter_list(path):
    """Nickel indices of a graph list, in file order (first tab-separated field containing '|')."""
    with open_list(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.split("\t", 2)          # the nickel is the first or second field of the lists; else look further
            if "|" in f[0]:
                yield f[0].strip()
            elif len(f) > 1 and "|" in f[1]:
                yield f[1].strip()
            else:
                for x in line.rstrip("\n").split("\t"):
                    if "|" in x:
                        yield x.strip()
                        break


def read_list(path):
    return list(iter_list(path))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path, default=None):
    if default is not None and not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def _key(k):
    """key order of the JSON files: seeds numerically, orders by kind then loops (b2 ... b10), the rest alphabetically"""
    if re.match(r"^\d+$", k):
        return (0, int(k), "")
    m = re.match(r"^([bhc])(\d+)$", k)
    return (1, "bhc".index(m.group(1)) * 100 + int(m.group(2)), "") if m else (2, 0, k)


def _clean(x):
    """NaN and infinities are not JSON (written as null); dictionaries in the order of _key."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(x[k]) for k in sorted(x, key=_key)}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def dump_json(obj, rows_key=None):
    """Deterministic JSON text; with rows_key, the list under that key is written one element per line."""
    obj = _clean(obj)
    if rows_key is None or rows_key not in obj:
        return json.dumps(obj, indent=1) + "\n"
    rows = obj.pop(rows_key)
    head = json.dumps(obj, indent=1)
    body = ",\n".join("  " + json.dumps(r) for r in rows)
    return head[:-2] + ',\n "%s": [\n%s\n ]\n}\n' % (rows_key, body) if obj else '{\n "%s": [\n%s\n ]\n}\n' % (rows_key, body)


def write_if_changed(path, text):
    """Atomic write; returns False (and leaves the file alone) when the content is unchanged."""
    try:
        with open(path) as fh:
            if fh.read() == text:
                return False
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    return True


def poly(c, N):
    return sum(x * N ** k for k, x in enumerate(c))


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------- statistics
def run_moments(c):
    """(n, m, M2, M4) of one class of one run: M2 = sum (w - m)^2 = (n - 1) s^2, M4 = n * mu4 (fourth central moment)."""
    n, m, s = float(c["n"]), c["mean_w"], c["sd_w"]
    M2 = (n - 1) * s * s
    k = c.get("kurtosis")
    mu2 = M2 / n
    M4 = n * k * mu2 * mu2 if isinstance(k, (int, float)) and math.isfinite(k) else float("nan")
    return n, m, M2, M4


def pool_moments(parts):
    n = sum(p[0] for p in parts)
    m = sum(p[0] * p[1] for p in parts) / n
    M2 = sum(p[2] for p in parts) + sum(p[0] * (p[1] - m) ** 2 for p in parts)
    M4 = sum(p[3] for p in parts)
    return n, m, M2, M4


def chi2_sf(x, k):
    """P(chi^2_k > x): regularised upper incomplete gamma Q(k/2, x/2) (series / continued fraction)."""
    if k <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    a, y = 0.5 * k, 0.5 * x
    lg = math.lgamma(a)
    if y < a + 1:
        s = t = 1.0 / a
        ap = a
        for _ in range(100000):
            ap += 1
            t *= y / ap
            s += t
            if abs(t) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-y + a * math.log(y) - lg))
    b = y + 1 - a
    c, d = 1e300, 1 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        e = d * c
        h *= e
        if abs(e - 1) < 1e-15:
            break
    return math.exp(-y + a * math.log(y) - lg) * h


def norm_sf2(z):
    """two-sided normal tail probability"""
    return math.erfc(abs(z) / math.sqrt(2))


# ------------------------------------------------------------------------------------------------- ledger files
def ledger_paths(ledger):
    return {"orders": os.path.join(ledger, "orders.json"), "seeds": os.path.join(ledger, "seeds.json")}


def contrib_path(ledger, order, cid):
    return os.path.join(ledger, order, "contributions", cid + ".json")


def load_orders(ledger):
    """orders.json; a ledger without one is a new ledger of format 3"""
    return load_json(ledger_paths(ledger)["orders"], {"format": FORMAT3, "orders": {}})


SEEDS_NOTE = ("seeds per order: merged = in the ledger, running = reserved by rebar-refine.sh, excluded = runs left "
              "out on purpose (ledger/README.md); a contribution must use a seed absent here for its order")


def load_seeds(ledger):
    s = load_json(ledger_paths(ledger)["seeds"], {"format": "rebar-seeds-1", "orders": {}})
    s["note"] = SEEDS_NOTE
    return s


def save_seeds(ledger, seeds):
    return write_if_changed(ledger_paths(ledger)["seeds"], dump_json(seeds))


def fresh_seed(seeds):
    used = [int(s) for o in seeds.get("orders", {}).values() for s in o if re.match(r"^\d+$", s)]
    return str(max(used + [FIRST_FRESH_SEED]) + 1)


_REFS = {}


def references(N=1):
    """{(kind, L): (value, err, source)} at this N: exact values for L <= 2, refs.py (Phi4tools via the CS relations)."""
    if N in _REFS:
        return _REFS[N]
    sys.path.insert(0, TOOLS)
    import refs                       # integrator/tools/refs.py
    sys.path.pop(0)
    r = refs.refs(float(N))
    out = {}
    for key, kind in (("B", "b"), ("H", "h"), ("C", "c")):
        for L, (v, e) in r[key].items():
            out[(kind, int(L))] = (v, e, "Phi4tools (refs.py)")
    for (kind, L), v in exact_low(N).items():
        out[(kind, L)] = (v, 0.0, "exact")
    _REFS[N] = out
    return out


def exact_low(N):
    return {("b", 1): (N + 8) / 12.0, ("b", 2): (5 * N + 22) / 54.0, ("h", 2): (N + 2) / 243.0,
            ("c", 1): (N + 2) / 12.0, ("c", 2): (N + 2) / 36.0}


def default_entry(kind, L):
    o = order_name(kind, L)
    lp = list_path(kind, L)
    return {"kind": kind, "L": L, "list": os.path.basename(lp) if lp else None,
            "list_sha256": sha256_file(lp) if lp else None, "graphs": sum(1 for _ in iter_list(lp)) if lp else None,
            "target_rel": POLICY.get(o), "closed": L <= 6, "contributions": []}


def order_status(e):
    if not e.get("contributions"):
        return "planned"
    if e.get("closed"):
        return "closed"
    t = e.get("target_rel")
    if t is not None and e.get("rel_err") is not None and e["rel_err"] <= t:
        return "target"
    return "open"


def class_meta(c):
    m = {k: c.get(k) for k in ("nickel", "S", "weight_poly", "salt", "pref", "is1vr")}
    if c.get("merged_twins"):
        m["merged_twins"] = c["merged_twins"]
    return m


def contribution_classes(path, excluded=()):
    """{nickel: (meta, moments, extras)} of the sampled classes of one run file (format 1)."""
    R = read_contribution(path)
    out = {}
    for c in R.get("classes", []):
        if "n" not in c or c["nickel"] in excluded or not c["n"]:
            continue
        out[c["nickel"]] = (class_meta(c), run_moments(c), _extras(c, c["pref"]))
    return R, out


def _extras(c, pref):
    return {"maxw_over_absmean": c.get("maxw_over_absmean"), "roundoff_max_over_sd": c.get("roundoff_max_over_sd"),
            "roundoff_rms_over_sd": c.get("roundoff_rms_over_sd"), "nbad": c.get("nbad", 0),
            "value": pref * c["mean_w"], "error": pref * c["sd_w"] / math.sqrt(c["n"])}


def _pool_class(meta, parts, exs, ids):
    """Pooled record of one class from its per-run moments and extras (in manifest order)."""
    n, m, M2, M4 = pool_moments(parts)
    pref = meta["pref"]
    sd = math.sqrt(M2 / (n - 1)) if n > 1 else float("nan")
    mu2 = M2 / n
    val, err = pref * m, pref * sd / math.sqrt(n)
    r = dict(meta)
    r.update({"n": int(n) if float(n).is_integer() else n, "mean_w": m, "sd_w": sd, "value": val, "error": err,
              "kurtosis": (M4 / n) / (mu2 * mu2) if mu2 > 0 and math.isfinite(M4) else None,
              "nbad": sum(x["nbad"] or 0 for x in exs), "contributions": ids, "ncontrib": len(ids)})
    for k in ("maxw_over_absmean", "roundoff_max_over_sd", "roundoff_rms_over_sd"):
        v = [x[k] for x in exs if isinstance(x[k], (int, float)) and math.isfinite(x[k])]
        r[k] = max(v) if v else None
    if len(exs) > 1:   # scatter of the contributions about the pooled value
        r["chi2"] = sum((x["value"] - val) ** 2 / x["error"] ** 2 for x in exs if x["error"] > 0)
        r["chi2_dof"] = len(exs) - 1
    r["contrib"] = poly(r["weight_poly"], 1) * val
    r["contrib_err"] = abs(poly(r["weight_poly"], 1)) * err
    return r


def pool_order(ledger, order, entry, skip_id=None, check_sha=True):
    """Pool the contributions of an order (all of those in its manifest except skip_id), in memory.
    Returns ({nickel: pooled record}, [ids used], problems).  Format 3: the current state (iter_classes), skip_id
    unsupported."""
    if _format_of(ledger) == FORMAT3:     # the folded state: no leave-one-out (replay the archive for that)
        if skip_id is not None:
            raise LedgerError("format 3 holds the folded state only: a pool without %s needs a replay of the archive" % skip_id)
        try:
            return ({r["nickel"]: r for r in iter_classes(ledger, order, verify=check_sha)},
                    [c["id"] for c in entry.get("contributions", [])], [])
        except LedgerError as e:
            return {}, [], [str(e)]
    acc, ids, problems = {}, [], []
    for rec in entry.get("contributions", []):
        cid = rec["id"]
        if cid == skip_id:
            continue
        p = contrib_path(ledger, order, cid)
        if not os.path.exists(p):
            problems.append("%s: file missing" % cid)
            continue
        if check_sha and sha256_file(p) != rec["file_sha256"]:
            problems.append("%s: file changed since it was merged (re-merge it)" % cid)
            continue
        _, cls = contribution_classes(p, set(rec.get("excluded", [])))
        ids.append(cid)
        for nk, (meta, mom, ex) in cls.items():
            a = acc.setdefault(nk, {"meta": meta, "parts": [], "ex": [], "ids": []})
            a["parts"].append(mom)
            a["ex"].append(ex)
            a["ids"].append(cid)
    pooled = {nk: _pool_class(a["meta"], a["parts"], a["ex"], a["ids"]) for nk, a in acc.items()}
    return pooled, ids, problems


def order_sums(pooled, N=1):
    s = v = 0.0
    for r in pooled.values():
        w = poly(r["weight_poly"], N)
        s += w * r["value"]
        v += (w * r["error"]) ** 2
    return s, math.sqrt(v)


def sum_polynomial(pooled):
    """Coefficients of the order sum as a polynomial in N, with the error of each coefficient (they are correlated)."""
    deg = max((len(r["weight_poly"]) for r in pooled.values()), default=0)
    c, e = [0.0] * deg, [0.0] * deg
    for r in pooled.values():
        for k, w in enumerate(r["weight_poly"]):
            c[k] += w * r["value"]
            e[k] += (w * r["error"]) ** 2
    return c, [math.sqrt(x) for x in e]


def rebuild_order(ledger, order, orders, write=True, txn=None):
    """Recompute the statistics of orders.json for one order: format 1 from its contributions (and classes.json),
    format 3 from its stats and weights (checked against their SHA-256)."""
    if orders.get("format", FORMAT1) == FORMAT3:
        order_statistics(_Order(ledger, order, orders["orders"][order], txn))
        return orders["orders"][order]
    entry = orders["orders"][order]
    pooled, ids, problems = pool_order(ledger, order, entry)
    if problems:
        raise RuntimeError("order %s: %s" % (order, "; ".join(problems)))
    kind, L = entry["kind"], entry["L"]
    lp = list_path(kind, L)
    names = read_list(lp) if lp else []
    rank = {nk: i for i, nk in enumerate(names)}
    if lp and not entry.get("graphs"):   # an entry made before its list was in the repository
        entry.update({"list": os.path.basename(lp), "list_sha256": sha256_file(lp), "graphs": len(names)})
    rows = sorted(pooled.values(), key=lambda r: (rank.get(r["nickel"], 1 << 30), r["nickel"]))
    index = {cid: i for i, cid in enumerate(ids)}
    for r in rows:   # contribution ids by their index in the header list (b8: 21689 classes)
        r["from"] = [index[cid] for cid in r.pop("contributions")]
    if pooled:
        s, e = order_sums(pooled, 1)
        cp, ce = sum_polynomial(pooled)
        _order_stats(entry, len(pooled), s, e, cp, ce, sum(r["n"] for r in pooled.values()),
                     sum(r["nbad"] for r in pooled.values()),
                     max((r["roundoff_max_over_sd"] or 0) for r in pooled.values()))
    entry["status"] = order_status(entry)
    if write:
        cl = {"format": "rebar-classes-1", "order": order, "kind": kind, "L": L,
              "note": "pooled per-class statistics (series/rebar_ledger.py): value = pref * mean_w (the integral "
                      "without its weight), error = pref * sd_w / sqrt(n), contrib = weight_poly(1) * value, at any N "
                      "weight_poly(N) * value; from = indices into contributions",
              "contributions": ids, "classes": rows}
        write_if_changed(os.path.join(ledger, order, "classes.json"), dump_json(cl, "classes"))
    return entry


def _order_stats(entry, nclasses, s, e, cp, ce, samples, nbad, maxro):
    """The order's statistics in orders.json from its class sums (both formats)."""
    kind, L = entry["kind"], entry["L"]
    entry.update({"sum": s, "err": e, "rel_err": abs(e / s) if s else None, "classes": nclasses,
                  "samples": samples, "sum_poly": cp, "sum_poly_err": ce, "nbad": nbad,
                  "max_roundoff_max_over_sd": maxro,
                  "versions": sorted({c["version"] for c in entry["contributions"]})})
    exp = entry.get("classes_expected")
    if exp is None:   # no complete contribution yet (e.g. an order computed in shards): expect the list's classes
        exp = expected_classes(kind, L)
        if exp is not None:
            entry["classes_expected"] = exp
    entry["complete"] = exp is not None and nclasses == exp
    full = [c for c in entry["contributions"] if c.get("full")]
    if full:   # cross-check: inverse-variance mean of the order sums of the complete contributions, and their scatter
        w = [1 / c["err"] ** 2 for c in full]
        mu = sum(wi * c["sum"] for wi, c in zip(w, full)) / sum(w)
        entry["order_level"] = {"sum": mu, "err": 1 / math.sqrt(sum(w)), "n": len(full),
                                "chi2": sum((c["sum"] - mu) ** 2 / c["err"] ** 2 for c in full), "dof": len(full) - 1}
    ref = references(1).get((kind, L))
    if ref:
        entry["ref"], entry["ref_source"] = [ref[0], ref[1]], ref[2]
        entry["pull"] = (s - ref[0]) / math.sqrt(e * e + ref[1] ** 2)


def save_orders(ledger, orders):
    stamps = [c["merged_utc"] for e in orders["orders"].values() for c in e.get("contributions", [])]
    orders["updated"] = max(stamps) if stamps else None
    if orders.get("format") != FORMAT3:
        return write_if_changed(ledger_paths(ledger)["orders"], dump_json(orders))
    # format 3: every manifest entry on one line (compact JSON, keys sorted): a merge adds lines
    o = _clean(orders)
    recs = {}
    for name, e in o["orders"].items():
        for i, c in enumerate(e.get("contributions", [])):
            tok = "@@rec-%s-%d@@" % (name, i)
            recs['"%s"' % tok] = json.dumps(c, sort_keys=True, separators=(",", ":"))
            e["contributions"][i] = tok
    text = json.dumps(o, indent=1) + "\n"
    text = re.sub(r'"@@rec-[^"]+@@"', lambda m: recs[m.group(0)], text)
    return write_if_changed(ledger_paths(ledger)["orders"], text)


def ensure_orders(orders):
    """Add the entries of every order with a list or a policy (planned orders included); keep existing entries."""
    names = set(POLICY)
    for kind in KINDS:
        for L in range(1, 11):
            if list_path(kind, L):
                names.add(order_name(kind, L))
    for o in names:
        if o not in orders["orders"]:
            kind, L = parse_order(o)
            orders["orders"][o] = default_entry(kind, L)
            orders["orders"][o]["status"] = order_status(orders["orders"][o])
    return orders


# ------------------------------------------------------------------------------------------------- format 3: the folded registers
# Per order (doc/LEDGER.md): weights-PPPPPPP.tsv (the weight polynomials as exact fractions, written once) and
# stats-PPPPPPP.tsv (per class the folded registers of every run merged so far: Welford's n, mean, m2 merged pairwise
# by Chan, Golub and LeVeque, higher central moments by Pebay's update, extremes, sums, the number of runs), one row
# per list position from position PPPPPPP on, empty where there is no class (yet).  Nickel, salt, prefactor and 1VR
# flag follow from the list and the order and are recomputed, not stored.  The runs themselves are archived as
# received (<ledger>/../ledger-archive/<order>/<id>.json.xz) and replay to the same state bit for bit.
FORMAT1, FORMAT3 = "rebar-ledger-1", "rebar-ledger-3"
ROWS = 131072                  # list positions per weights / stats file (about 18 MB of stats at B10)
RB_EPS = 2.2e-16               # the integrator's unit of round-off: roundoff_* = RB_EPS * sum|terms| / sd
ARCHIVE_PRESET = 1             # xz preset of the archive (4x on run JSONs, about 10 MB/s)
STATS_COLS = ("n", "mean", "m2", "m3", "m4", "wmaxabs", "wmin", "wmax", "nbad", "rmax", "r2", "tau_ref")
N_, MEAN, M2, M3, M4, WMAXABS, WMIN, WMAX, NBAD, RMAX, R2, TAU = range(len(STATS_COLS))
MOMENTS = {2: M2, 3: M3, 4: M4}          # central moment registers by order (columns m5, m6, ... extend it)
DIGITS = {M3: 6, M4: 6, WMAXABS: 6, WMIN: 6, WMAX: 6, RMAX: 4, R2: 4, TAU: 3}   # significant digits kept
REF_DEVICE = "NVIDIA H200"     # tau_ref: core ns per sample of a class on this device (runs on others leave it)
SHARD_RULE = "survivor j of the list (its lines without 1VR=1) in shard j mod n (b9-split.py)"
OMIT = (("backend_info", "class_device_ns_per_sample"), ("backend_info", "launch_log"))


class LedgerError(Exception):
    pass


def _format_of(ledger):
    return load_orders(ledger).get("format", FORMAT1)


# ------------------------------------------------------------------------------------------------- registers
def _sig(x, d):
    return x if x is None or not math.isfinite(x) else float("%.*g" % (d, x))


def _round_regs(R):
    for i, d in DIGITS.items():
        if R[i] is not None:
            R[i] = _sig(R[i], d)
    return R


def stats_line(R):
    """a stats row: integers as such, floats as their shortest round-trip repr (the approximate registers are already
    rounded), nbad empty when 0"""
    if R is None:
        return ""
    out = []
    for i, x in enumerate(R):
        if x is None or (i == NBAD and not x):
            out.append("")
        else:
            out.append(str(x) if type(x) is int else repr(x))
    return "\t".join(out)


def stats_regs(line):
    if not line:
        return None
    f = line.split("\t")
    R = [float(x) if x else None for x in f]
    R[N_] = int(f[N_])
    R[NBAD] = int(f[NBAD]) if f[NBAD] else 0
    return R


def _comb(p, k):
    return math.factorial(p) // (math.factorial(k) * math.factorial(p - k))


def pebay(p, ma, mb, na, nb, d):
    """The p-th central moment sum of the union of two sets from theirs (ma[q], mb[q] for 2 <= q <= p; m_1 = 0),
    d = mean_b - mean_a: P. Pebay, SAND2008-6212 (2008), eq. (2.7)."""
    n = na + nb
    s = ma[p] + mb[p]
    for k in range(1, p - 1):
        s += _comb(p, k) * d ** k * ((-nb / n) ** k * ma[p - k] + (na / n) ** k * mb[p - k])
    return s + (na * nb * d / n) ** p * (1.0 / nb ** (p - 1) - (-1.0 / na) ** (p - 1))


def _opt(f, a, b):
    return b if a is None else a if b is None else f(a, b)


def fold(A, Bv, ref=True):
    """The state A of a class (None: no run yet) with the registers Bv of one more run.  n, mean, m2 (Welford's
    registers) by the pairwise update of Chan, Golub and LeVeque: n = na + nb, d = mb - ma, mean = ma + d nb / n,
    m2 = m2a + m2b + d^2 na nb / n; m3, m4 by Pebay's; max|w|, max round-off and min/max w as extremes; nbad and r2
    summed; tau_ref the n-weighted mean over the runs on the reference device (ref; the weight of the earlier value is
    the class's n).  The approximate registers are rounded to their stored digits after every fold, so that the state
    does not depend on how the runs were batched."""
    if A is None:
        R = list(Bv)
        if not ref:
            R[TAU] = None
        return _round_regs(R)
    na, nb = A[N_], Bv[N_]
    n = na + nb
    d = Bv[MEAN] - A[MEAN]
    R = [None] * len(STATS_COLS)
    R[N_] = n
    R[MEAN] = A[MEAN] + d * nb / n
    R[M2] = A[M2] + Bv[M2] + d * d * na * nb / n
    ma = {p: A[i] for p, i in MOMENTS.items()}
    mb = {p: Bv[i] for p, i in MOMENTS.items()}
    for p, i in MOMENTS.items():
        if p > 2:
            R[i] = None if any(m[q] is None for m in (ma, mb) for q in range(2, p + 1)) else pebay(p, ma, mb, na, nb, d)
    for i in (WMAXABS, WMAX, RMAX):
        R[i] = _opt(max, A[i], Bv[i])
    R[WMIN] = _opt(min, A[WMIN], Bv[WMIN])
    R[R2] = _opt(lambda x, y: x + y, A[R2], Bv[R2])
    R[NBAD] = (A[NBAD] or 0) + (Bv[NBAD] or 0)
    R[TAU] = A[TAU]
    if ref and Bv[TAU] is not None:
        R[TAU] = Bv[TAU] if A[TAU] is None else (na * A[TAU] + nb * Bv[TAU]) / n
    return _round_regs(R)


def run_registers(c, ck):
    """(registers, exact, flagged) of one class of a run.  n, mean_w and m2 = (n - 1) sd_w^2 from the JSON (as the pooling
    always used them); m3, m4, max|w|, round-off from the checkpoint's exact accumulators ck = (n, sum w, sum w^2,
    sum w^3, sum w^4, sum (sum|terms|)^2, max|w|, nbad, max sum|terms|): m3 = q3 - 3 m q + 2 n m^3, m4 = q4 - 4 m q3
    + 6 m^2 q - 3 n m^4; without them (0.3.0 runs) from the JSON's kurtosis and ratios, m3 = 0 (exact False).
    flagged: the accumulators disagree with the JSON (n, or kurtosis / round-off ratios beyond 1e-6)."""
    n, m, sd = c["n"], c["mean_w"], c["sd_w"]
    m2 = (n - 1) * sd * sd
    R = [None] * len(STATS_COLS)
    R[N_], R[MEAN], R[M2], R[NBAD] = n, m, m2, c.get("nbad") or 0
    ct = c.get("core_ns_per_sample")
    R[TAU] = float(ct) if isinstance(ct, (int, float)) and ct > 0 else None
    flagged = False
    if ck is not None and ck[0] == n:
        nn, s, q, q3, q4, sabs2, maxw, nbad, maxsabs = ck
        R[M3] = q3 - 3 * m * q + 2 * nn * m ** 3
        R[M4] = q4 - 4 * m * q3 + 6 * m * m * q - 3 * nn * m ** 4
        R[WMAXABS], R[RMAX], R[R2] = maxw, RB_EPS * maxsabs, RB_EPS * RB_EPS * sabs2
        kj, rj = c.get("kurtosis"), c.get("roundoff_max_over_sd")
        if m2 > 0 and isinstance(kj, float) and kj and abs(n * R[M4] / (m2 * m2) - kj) > 1e-6 * abs(kj):
            flagged = True
        if sd and isinstance(rj, float) and rj and abs(R[RMAX] / sd - rj) > 1e-6 * rj:
            flagged = True
        return R, True, flagged
    k, mu2 = c.get("kurtosis"), m2 / n
    R[M3] = 0.0
    R[M4] = n * k * mu2 * mu2 if isinstance(k, (int, float)) and math.isfinite(k) else None
    x = c.get("maxw_over_absmean")
    R[WMAXABS] = x * abs(m) if isinstance(x, (int, float)) else None
    x = c.get("roundoff_max_over_sd")
    R[RMAX] = x * sd if isinstance(x, (int, float)) and math.isfinite(sd) else None
    x = c.get("roundoff_rms_over_sd")
    R[R2] = n * (x * sd) ** 2 if isinstance(x, (int, float)) and math.isfinite(sd) else None
    return R, False, ck is not None


# ------------------------------------------------------------------------------------------------- class constants
def class_salt(kind, L, nickel):
    """the integrator's per-class salt: FNV-1a 64 of "kind|L|nickel" (integrator/src/main.cpp class_salt)"""
    h = 1469598103934665603
    for b in ("%s|%d|%s" % (kind, L, nickel)).encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def class_pref(pw, L):
    """Gamma(pw) pi^(-L/2) (integrator/src/build.cpp)"""
    return math.gamma(pw) * math.pi ** (-0.5 * L)


def weight_fractions(wp, S, L):
    """The weight polynomial as exact fractions: S c_k / 3^M with integers c_k summing to 3^M (the O(N) factor is a
    polynomial with coefficients in Z/3^m, equal to 1 at N = 1), each within 1e-12 of the integrator's double, and
    S = float(sum).  Returns the text ("8/729,17/729,...") or None."""
    try:
        Sf = Fraction(S).limit_denominator(1 << 40)
    except (TypeError, ValueError):
        return None
    if not S or float(Sf) != S or not wp:
        return None
    M = L + 4
    D = 3 ** M
    c = [int(round(x / S * D)) for x in wp]
    if sum(c) != D or any(abs(x / S * D - ci) > 1e-3 for x, ci in zip(wp, c)):
        return None
    fr = [Sf * Fraction(ci, D) for ci in c]
    top = max(abs(x) for x in wp)
    if float(sum(fr)) != S or any(abs(float(f) - x) > 1e-12 * top for f, x in zip(fr, wp)):
        return None
    return ",".join(str(f) for f in fr)


def weight_floats(text):
    """(weight polynomial as floats, correctly rounded; S = its value at N = 1, exactly) from a weights row"""
    fl, num, den = [], 0, 1
    for t in text.split(","):
        p, _, q = t.partition("/")
        p, q = int(p), int(q) if q else 1
        fl.append(p / q)
        num, den = num * q + p * den, den * q
    return fl, num / den


# ------------------------------------------------------------------------------------------------- files
def sha256_any(path):
    """SHA-256 of a file's content, decompressed if it ends in .xz"""
    h = hashlib.sha256()
    with (lzma.open(path, "rb") if path.endswith(".xz") else open(path, "rb")) as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Txn:
    """The files of one merge: written next to their final names (.new) and moved into place together by commit(),
    just before orders.json is saved; an interrupted merge leaves the ledger as it was."""
    def __init__(self):
        self.pending = {}

    def path(self, p):
        """where to read p from: its pending version, if any"""
        return self.pending.get(p) or p

    def write(self, p, text):
        """returns the SHA-256 of text; a file whose content is unchanged is left alone"""
        data = text.encode()
        sha = hashlib.sha256(data).hexdigest()
        if os.path.exists(p) and sha256_file(p) == sha:
            t = self.pending.pop(p, None)
            if t and os.path.exists(t):
                os.remove(t)
            return sha
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".new", "wb") as fh:
            fh.write(data)
        self.pending[p] = p + ".new"
        return sha

    def commit(self):
        for p, t in self.pending.items():
            os.replace(t, p)
        self.pending = {}

    def abort(self):
        for t in self.pending.values():
            if os.path.exists(t):
                os.remove(t)
        self.pending = {}


class _Spool:
    """Records (marshal) bucketed by chunk in a temporary directory, read back one chunk at a time."""
    def __init__(self, chunk):
        self.dir = tempfile.mkdtemp(prefix="rebar-spool-")
        self.chunk, self.fh, self.count = chunk, {}, {}

    def add(self, pos, rec):
        k = pos // self.chunk
        f = self.fh.get(k)
        if f is None:
            f = self.fh[k] = open(os.path.join(self.dir, str(k)), "ab")
        marshal.dump((pos,) + rec, f)
        self.count[k] = self.count.get(k, 0) + 1

    def seal(self):
        for f in self.fh.values():
            f.close()
        self.fh = {}

    def chunks(self):
        return sorted(self.count)

    def read(self, k):
        self.seal()
        out = []
        if k in self.count:
            with open(os.path.join(self.dir, str(k)), "rb") as fh:
                for _ in range(self.count[k]):
                    out.append(marshal.load(fh))
        return out

    def close(self):
        self.seal()
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------------------------------------------- streaming JSON
_WS = re.compile(r"[ \t\n\r]*")


class _JsonStream:
    """Incremental reader of one JSON document: values are decoded by json's raw_decode on a sliding buffer, so that
    selected arrays can be read element by element."""
    def __init__(self, fh):
        self.fh, self.buf, self.i, self.eof = fh, "", 0, False
        self.dec = json.JSONDecoder()
        self.omitted, self.keys = [], []

    def _more(self):
        if self.eof:
            return False
        if self.i > (1 << 20) and self.i > len(self.buf) // 2:
            self.buf, self.i = self.buf[self.i:], 0
        s = self.fh.read(max(1 << 20, len(self.buf) - self.i))
        if not s:
            self.eof = True
            return False
        self.buf += s
        return True

    def peek(self):
        while True:
            self.i = _WS.match(self.buf, self.i).end()
            if self.i < len(self.buf):
                return self.buf[self.i]
            if not self._more():
                raise LedgerError("unexpected end of the JSON document")

    def take(self, ch):
        if self.peek() != ch:
            raise LedgerError("malformed JSON: %r expected at %r" % (ch, self.buf[self.i:self.i + 40]))
        self.i += 1

    def value(self):
        self.peek()
        while True:
            try:
                v, end = self.dec.raw_decode(self.buf, self.i)
                if end < len(self.buf) or self.eof:
                    self.i = end
                    return v
            except ValueError:
                if self.eof:
                    raise
            self._more()

    def items(self):
        """elements of the array that starts here"""
        self.take("[")
        if self.peek() == "]":
            self.i += 1
            return
        while True:
            yield self.value()
            c = self.peek()
            self.i += 1
            if c == "]":
                return
            if c != ",":
                raise LedgerError("malformed JSON array")


_STREAM = {("classes",): "class", ("dropped",): "dropped", ("checkpoint", "classes"): "ckpt"}


def _walk(st, path):
    """Generator over an object: yields (kind, element) for the elements of the per-class arrays of _STREAM, skips
    those of OMIT, returns the dict of the other fields."""
    st.take("{")
    out = {}
    if st.peek() == "}":
        st.i += 1
        return out
    while True:
        key = st.value()
        st.take(":")
        sub = path + (key,)
        if not path:
            st.keys.append(key)
        if sub in _STREAM and st.peek() == "[":
            for v in st.items():
                yield _STREAM[sub], v
        elif sub in OMIT and st.peek() == "[":
            for _ in st.items():
                pass
            st.omitted.append(".".join(sub))
        elif sub in (("checkpoint",), ("backend_info",)) and st.peek() == "{":
            out[key] = yield from _walk(st, sub)
        else:
            out[key] = st.value()
        c = st.peek()
        st.i += 1
        if c == "}":
            return out
        if c != ",":
            raise LedgerError("malformed JSON object")


def stream_run(path):
    """Events of a run JSON (plain or .xz) read with bounded memory: ("class", c), ("dropped", d), ("ckpt", checkpoint
    row), then ("keys", its top-level keys) and ("header", the other fields)."""
    with (lzma.open(path, "rt", encoding="utf-8") if path.endswith(".xz") else open(path, encoding="utf-8")) as fh:
        st = _JsonStream(fh)
        h = yield from _walk(st, ())
    yield "keys", st.keys
    yield "header", h


def _unhex(h):
    return struct.unpack(">12d", bytes.fromhex(h))


def _ckpt_regs(row, cols):
    """(n, sum w, sum w^2, sum w^3, sum w^4, sum (sum|terms|)^2, max|w|, nbad, max sum|terms|) of a checkpoint row:
    the accumulators acc + part (sums added, maxima kept)"""
    a, b = _unhex(row[cols["acc"]]), _unhex(row[cols["part"]])
    return (a[0] + b[0], (a[1] + a[2]) + (b[1] + b[2]), (a[3] + a[4]) + (b[3] + b[4]), a[5] + b[5], a[6] + b[6],
            a[7] + b[7], max(a[8], b[8]), a[9] + b[9], max(a[11], b[11]))


# ------------------------------------------------------------------------------------------------- lists
def iter_list_flags(path):
    """(nickel, 1VR flag or None) of a graph list, in file order"""
    with open_list(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            nk = next((x.strip() for x in f if "|" in x), None)
            if nk is None:
                continue
            fl = next((x.strip() for x in f if x.strip().startswith("1VR=")), None)
            yield nk, (fl == "1VR=1") if fl else None


class _ListIndex:
    """nickel -> list position with 12 bytes per graph (sorted 64-bit string hashes; the nickel at the position found is
    checked against the list by the callers); also the positions of the 1VR graphs (vr), from the same pass."""
    def __init__(self, path):
        hs, self.vr = array.array("q"), array.array("l")
        for i, (nk, f) in enumerate(iter_list_flags(path)):
            hs.append(hash(nk))
            if f:
                self.vr.append(i)
        order = sorted(range(len(hs)), key=hs.__getitem__)
        self.h = array.array("q", (hs[i] for i in order))
        self.p = array.array("l", order)
        del hs, order
        self.n = len(self.h)
        if any(self.h[i] == self.h[i + 1] for i in range(self.n - 1)):
            raise LedgerError("%s: two graphs with the same string hash (run again: hashes are per process)" % path)

    def get(self, nickel):
        i = bisect.bisect_left(self.h, hash(nickel))
        return self.p[i] if i < self.n and self.h[i] == hash(nickel) else None


class _ListChunks:
    """The list's (nickel, 1VR flag) one chunk at a time, read sequentially (chunks asked for in increasing order)."""
    def __init__(self, path, chunk):
        self.path, self.chunk, self.it, self.k, self.pos, self.last = path, chunk, None, None, 0, None

    def get(self, k):
        if self.k == k:
            return self.last
        if self.it is None or k < self.k:
            self.it, self.pos = iter_list_flags(self.path), 0
        lo, hi, out = k * self.chunk, (k + 1) * self.chunk, []
        while self.pos < hi:
            x = next(self.it, None)
            if x is None:
                break
            if self.pos >= lo:
                out.append(x)
            self.pos += 1
        self.k, self.last = k, out
        return out


class _Survivors:
    """List positions <-> survivor ranks of the shard rule (the list's lines without 1VR=1; every line if it has no
    flags): rank(p) = p minus the 1VR graphs before p."""
    def __init__(self, vr):
        self.vr = vr
        self.isvr = set(self.vr) if len(self.vr) < 1 << 22 else None

    def rank(self, p):
        return p - bisect.bisect_left(self.vr, p)

    def survivor(self, p):
        return p not in self.isvr if self.isvr is not None else (bisect.bisect_left(self.vr, p) == len(self.vr) or self.vr[bisect.bisect_left(self.vr, p)] != p)


def _ranges(pos):
    out = []
    for p in pos:
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return out


def coverage(folded, others, graphs, surv):
    """The classes a run folded in, as the manifest records them: "all" (every graph of the list is folded, dropped or
    excluded), {"shard": i, "of": n, "rule": SHARD_RULE} when folded plus dropped and excluded are exactly the
    survivors j = i mod n, else {"ranges": [[first, last], ...]} of the folded positions.  folded, others (dropped
    or excluded positions): sorted lists."""
    if graphs and len(folded) + len(others) == graphs:
        return "all"
    if surv is not None and len(folded) > 1 and all(surv.survivor(p) for p in folded):
        rk = [surv.rank(p) for p in folded]
        n = 0
        for a, b in zip(rk, rk[1:]):
            n = math.gcd(n, b - a)
        if n > 1:
            i = rk[0] % n
            tot = graphs - len(surv.vr)
            mine = [p for p in others if surv.survivor(p) and surv.rank(p) % n == i]
            if len(folded) + len(mine) == len(range(i, tot, n)):
                return {"shard": i, "of": n, "rule": SHARD_RULE}
    return {"ranges": _ranges(folded)}


def _counts_in_chunk(ctx, recs, k, surv):
    """k (the contributions that sampled it) of every position of chunk k, from the manifest's coverage records"""
    lo, hi = k * ctx.C, min((k + 1) * ctx.C, ctx.npos())
    cnt = [0] * (hi - lo)
    alln, shards = 0, {}
    for c in recs:
        cv = c.get("coverage")
        if cv == "all":
            alln += 1
        elif isinstance(cv, dict) and "shard" in cv:
            shards.setdefault(cv["of"], {}).setdefault(cv["shard"], 0)
            shards[cv["of"]][cv["shard"]] += 1
        elif isinstance(cv, dict):
            for a, b in cv.get("ranges", []):
                for p in range(max(a, lo), min(b + 1, hi)):
                    cnt[p - lo] += 1
            continue                     # ranges hold the folded classes only
        for p in c.get("excluded_positions", []):
            if lo <= p < hi:
                cnt[p - lo] -= 1
    if alln or shards:
        for p in range(lo, hi):
            x = alln
            if shards and surv is not None and surv.survivor(p):
                r = surv.rank(p)
                for n, res in shards.items():
                    x += res.get(r % n, 0)
            cnt[p - lo] += x
    return cnt


# ------------------------------------------------------------------------------------------------- one order
class _Order:
    """An order of a format-3 ledger: its weights and stats files (rows per file C), list positions (the list's, then
    the classes outside it, entry["extra"]), the constants recomputed from the order."""
    def __init__(self, ledger, order, entry, txn=None):
        self.ledger, self.order, self.entry = ledger, order, entry
        self.txn = txn or Txn()
        self.kind, self.L = entry["kind"], entry["L"]
        self.C = entry.get("rows") or ROWS
        self.lp = list_path(self.kind, self.L)
        self.graphs = (entry.get("graphs") or 0) if self.lp else 0
        self.dir = os.path.join(ledger, order)
        self.extra = list(entry.get("extra", []))
        self._xpos = {nk: self.graphs + i for i, nk in enumerate(self.extra)}
        self._index = self._names = self._surv = None

    def index(self):
        if self._index is None:
            self._index = _ListIndex(self.lp)
        return self._index

    def survivors(self):
        if self._surv is None and self.lp:
            self._surv = _Survivors(self.index().vr)
        return self._surv

    def npos(self):
        return self.graphs + len(self.extra)

    def first(self, k):
        return k * self.C

    def fname(self, what, k):
        return "%s-%07d.tsv" % (what, k * self.C)

    def records(self, what):
        return {f["first"] // self.C: f for f in (self.entry.get("files") or {}).get(what, [])}

    def chunks(self):
        return sorted(set(self.records("stats")) | set(self.records("weights")))

    def read(self, what, k, pending=False, verify=True):
        """the rows of chunk k (a list of strings, one per position; empty strings where nothing is stored)"""
        n = max(0, min(self.C, self.npos() - k * self.C))
        f = self.records(what).get(k)
        if f is None:
            return [""] * n
        p = os.path.join(self.dir, f["file"])
        if pending:
            p = self.txn.path(p)
        with open(p, "rb") as fh:
            data = fh.read()
        if verify and not pending and hashlib.sha256(data).hexdigest() != f["sha256"]:
            raise LedgerError("%s: %s changed since it was written (SHA-256 differs from orders.json)" % (self.order, f["file"]))
        rows = data.decode().split("\n")[1:-1]
        return rows + [""] * (n - len(rows))

    def write(self, what, k, rows):
        cols = STATS_COLS if what == "stats" else ("weight_poly",)
        while rows and not rows[-1] and len(rows) > 0 and k * self.C + len(rows) > self.npos():
            rows.pop()
        text = "\t".join(cols) + "\n" + "".join(r + "\n" for r in rows)
        name = self.fname(what, k)
        sha = self.txn.write(os.path.join(self.dir, name), text)
        files = self.entry.setdefault("files", {}).setdefault(what, [])
        files[:] = [f for f in files if f["first"] != k * self.C] + [{"first": k * self.C, "file": name, "rows": len(rows), "sha256": sha}]
        files.sort(key=lambda f: f["first"])

    def list_names(self):
        if self._names is None:
            self._names = _ListChunks(self.lp, self.C) if self.lp else None
        return self._names

    def names(self, k):
        """[(nickel, 1VR flag)] of the positions of chunk k (the list's, then the extra ones)"""
        lo, hi = k * self.C, (k + 1) * self.C
        out = list(self.list_names().get(k)) if self.lp and lo < self.graphs else []
        for p in range(max(lo, self.graphs), min(hi, self.npos())):
            out.append((self.extra[p - self.graphs], None))
        return out

    def position(self, nickel, add=True):
        """(list position or None, position after the list or None): a graph of the list, or one outside it (a new
        position if add)"""
        if self.lp:
            p = self.index().get(nickel)
            if p is not None and p < self.graphs:
                return p, None
        if nickel not in self._xpos and add:
            self._xpos[nickel] = self.graphs + len(self.extra)
            self.extra.append(nickel)
        return None, self._xpos.get(nickel)


# ------------------------------------------------------------------------------------------------- reading runs
class Run:
    """A run file (JSON, plain or .xz) read once with bounded memory, its classes, dropped graphs and checkpoint rows
    spooled by chunk of list positions.  Run-level numbers of the merge rules: ncl (class rows), samples, s1 and e1
    (its order sum and error), excluded (nbad > 0), unlisted (sampled classes not in the list), dup (sampled classes
    given twice), nsampled (sampled rows), names (distinct sampled classes), diagnostics (its largest round-off,
    kurtosis and max|w|/|mean|, the classes with nbad)."""
    def __init__(self, path):
        self.path = path
        self.sha, self.size = sha256_any(path), os.path.getsize(path)
        self.header, self.spool, self.keys = None, None, []

    def read(self, get_order):
        """parse the file; get_order(kind, L) -> _Order of the destination ledger"""
        tmp = tempfile.NamedTemporaryFile("wb", prefix="rebar-raw-", delete=False)
        self.raw = tmp.name
        cols = None
        try:
            n = 0
            for ev, obj in stream_run(self.path):
                if ev == "class":
                    g = obj.get
                    marshal.dump(("c", n, g("nickel"), g("S"), g("weight_poly"), g("salt"), g("pref"), g("pw"), g("is1vr"),
                                  g("n"), g("mean_w"), g("sd_w"), g("kurtosis"), g("nbad"), g("maxw_over_absmean"),
                                  g("roundoff_max_over_sd"), g("roundoff_rms_over_sd"), g("core_ns_per_sample"),
                                  g("build_s"), bool(g("merged_twins"))), tmp)
                    n += 1
                elif ev == "dropped":
                    marshal.dump(("d", obj.get("nickel"), obj.get("reason")), tmp)
                elif ev == "ckpt":
                    marshal.dump(("k", obj), tmp)
                elif ev == "keys":
                    self.keys = obj
                else:
                    self.header = obj
        finally:
            tmp.close()
        h = self.header
        ck = h.get("checkpoint") if isinstance(h.get("checkpoint"), dict) else {}
        if ck.get("class_columns"):
            cols = {c.split(" (")[0].strip(): i for i, c in enumerate(re.split(r",\s*(?![^()]*\))", ck["class_columns"]))}
            cols = cols if "acc" in cols and "part" in cols else None
        if h.get("kind") not in KINDS or not isinstance(h.get("L"), int):
            os.remove(self.raw)
            return self
        self.kind, self.L = h["kind"], h["L"]
        self.order = order_name(self.kind, self.L)
        ctx = get_order(self.kind, self.L)
        self.spool = _Spool(ctx.C)
        self.ncl = self.samples = self.nsampled = 0
        self.nickels = set()
        self.s1 = v1 = 0.0
        self.excluded, self.unlisted, self.dropped_unlisted, self.dup, self.twins = [], [], [], False, False
        self.sorted, last = True, -1
        self.diag = {"roundoff_max_over_sd": None, "kurtosis": None, "maxw_over_absmean": None, "nbad_classes": 0,
                     "nbad_positions": []}
        seen, seenx = bytearray(ctx.graphs), set()
        with open(self.raw, "rb") as fh:
            while True:
                try:
                    r = marshal.load(fh)
                except EOFError:
                    break
                if r[0] == "k":
                    if cols is None:
                        continue
                    lpos, xpos = ctx.position(r[1][cols["nickel"]], add=False)
                    p = lpos if lpos is not None else xpos
                    if p is not None:
                        self.spool.add(p, ("k", r[1][cols["nickel"]], _ckpt_regs(r[1], cols)))
                    continue
                if r[0] == "d":
                    lpos, _ = ctx.position(r[1], add=False)
                    if lpos is None:
                        self.dropped_unlisted.append({"nickel": r[1], "reason": r[2]})
                    else:
                        self.spool.add(lpos, ("d", r[1], r[2]))
                    continue
                nk, n = r[2], r[9]
                lpos, xpos = ctx.position(nk)
                pos = lpos if lpos is not None else xpos
                self.spool.add(pos, r)
                self.ncl += 1
                if pos < last:
                    self.sorted = False
                last = pos
                self.samples += n or 0
                if r[19]:
                    self.twins = True
                if not n:
                    continue
                self.nsampled += 1
                if not ctx.lp:
                    self.nickels.add(nk)
                if lpos is None:
                    self.unlisted.append(nk)
                    if nk in seenx:
                        self.dup = True
                    seenx.add(nk)
                else:
                    if seen[lpos]:
                        self.dup = True
                    seen[lpos] = 1
                for key, x in (("roundoff_max_over_sd", r[15]), ("kurtosis", r[12]), ("maxw_over_absmean", r[14])):
                    if isinstance(x, (int, float)) and math.isfinite(x) and (self.diag[key] is None or x > self.diag[key]):
                        self.diag[key] = x
                if r[13]:
                    self.excluded.append(nk)
                    self.diag["nbad_classes"] += 1
                    if len(self.diag["nbad_positions"]) < 20:
                        self.diag["nbad_positions"].append(pos)
                    continue
                w, pref = poly(r[4], 1), r[6]
                self.s1 += w * pref * r[10]
                v1 += (w * pref * r[11]) ** 2 / n
        self.e1 = math.sqrt(v1)
        self.excluded.sort()
        self.names = sum(seen) + len(seenx)
        self.spool.seal()
        os.remove(self.raw)
        return self

    def rows(self, k):
        """({pos: class}, {pos: checkpoint registers}, [dropped positions]) of chunk k (the last of duplicates kept)"""
        cl, ck, dr = {}, {}, []
        for r in self.spool.read(k):
            if r[1] == "c":
                cl[r[0]] = {"pos": r[0], "row": r[2], "nickel": r[3], "S": r[4], "weight_poly": r[5], "salt": r[6],
                            "pref": r[7], "pw": r[8], "is1vr": r[9], "n": r[10], "mean_w": r[11], "sd_w": r[12],
                            "kurtosis": r[13], "nbad": r[14], "maxw_over_absmean": r[15], "roundoff_max_over_sd": r[16],
                            "roundoff_rms_over_sd": r[17], "core_ns_per_sample": r[18], "build_s": r[19]}
            elif r[1] == "k":
                ck[r[0]] = r[3]
            else:
                dr.append(r[0])
        return cl, ck, dr

    def close(self):
        if self.spool:
            self.spool.close()
        if getattr(self, "raw", None) and os.path.exists(self.raw):
            os.remove(self.raw)


# ------------------------------------------------------------------------------------------------- the archive
def archive_dir(ledger, archive=None):
    """the archive of the full runs: <ledger>/../ledger-archive unless given"""
    return archive or os.path.join(os.path.dirname(os.path.abspath(ledger)), "ledger-archive")


def archive_path(ledger, order, cid, archive=None):
    return os.path.join(archive_dir(ledger, archive), order, cid + ".json.xz")


def archive_run(path, dest, sha):
    """Keep a run as received, xz-compressed, at dest (never overwritten: an existing copy must hold the same run),
    and check that it decompresses to the SHA-256 sha."""
    if os.path.exists(dest):
        if sha256_any(dest) != sha:
            raise LedgerError("%s exists and holds another run" % dest)
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest[:-3] + ".tmp%d.xz" % os.getpid()
    if path.endswith(".xz"):
        shutil.copyfile(path, tmp)
    else:
        z = lzma.LZMACompressor(preset=ARCHIVE_PRESET)
        with open(path, "rb") as src, open(tmp, "wb") as out:
            for b in iter(lambda: src.read(1 << 22), b""):
                out.write(z.compress(b))
            out.write(z.flush())
    if sha256_any(tmp) != sha:
        os.remove(tmp)
        raise LedgerError("%s: the archive copy does not decompress to the run's SHA-256" % dest)
    os.replace(tmp, dest)
    return True


# ------------------------------------------------------------------------------------------------- order statistics
def order_statistics(ctx, pending=True, verify=True):
    """Recompute the order's statistics in orders.json from its weights and stats (classes in list order): sum, err,
    sum_poly, classes, samples, nbad, max round-off; build_s and core_s from the manifest; returns the totals used by
    check (classes, sum of n, sum of k)."""
    e = ctx.entry
    s = v = 0.0
    deg, cp, ce = 0, [], []
    nclasses = samples = nbad = ksum = 0
    maxro = None
    pref = e.get("pref")
    surv = ctx.survivors()
    recs = e.get("contributions", [])
    for k in ctx.chunks():
        st = ctx.read("stats", k, pending, verify)
        wt = ctx.read("weights", k, pending, verify)
        kc = _counts_in_chunk(ctx, recs, k, surv)
        memo = {}
        for line, wl, kk in zip(st, wt, kc):
            if not line:
                continue
            ksum += kk
            R = stats_regs(line)
            if wl not in memo:
                memo[wl] = weight_floats(wl)
            wp = memo[wl][0]
            n = R[N_]
            sd = math.sqrt(R[M2] / (n - 1)) if n > 1 else float("nan")
            val, err = pref * R[MEAN], pref * sd / math.sqrt(n)
            w1 = poly(wp, 1)
            s += w1 * val
            v += (w1 * err) ** 2
            if len(wp) > deg:
                cp += [0.0] * (len(wp) - deg)
                ce += [0.0] * (len(wp) - deg)
                deg = len(wp)
            for i, w in enumerate(wp):
                cp[i] += w * val
                ce[i] += (w * err) ** 2
            nclasses += 1
            samples += n
            nbad += R[NBAD] or 0
            x = R[RMAX] / sd if R[RMAX] is not None and sd and math.isfinite(sd) else 0
            if maxro is None or x > maxro:
                maxro = x
    if nclasses:
        _order_stats(e, nclasses, s, math.sqrt(v), cp, [math.sqrt(x) for x in ce], samples, nbad, maxro)
    e["build_s"] = sum(c.get("build_s_first") or 0.0 for c in recs)
    e["reference_device"] = REF_DEVICE
    e["core_s"] = sum(c.get("core_s") or 0.0 for c in recs)
    sc = [c["check_detail"] for c in e.get("contributions", []) if (c.get("check_detail") or {}).get("dof")]
    e["scatter"] = {"chi2": sum(x["chi2"] for x in sc), "dof": sum(x["dof"] for x in sc)}
    e["status"] = order_status(e)
    return {"classes": nclasses, "samples": samples, "k": ksum}


# ------------------------------------------------------------------------------------------------- reading (API)
def _entry(ledger, order):
    orders = load_orders(ledger)
    if order not in orders["orders"]:
        raise LedgerError("no order %s in %s" % (order, ledger))
    return orders.get("format", FORMAT1), orders["orders"][order]


def _finite(r):
    """non-finite numbers as None (format 1 wrote them as null)"""
    for k, v in r.items():
        if isinstance(v, float) and not math.isfinite(v):
            r[k] = None
    return r


def iter_classes(ledger, order, fields=None, verify=False):
    """Pooled classes of an order in list order, as dicts: nickel (from the list), S, weight_poly (the exact fractions
    as floats), salt (recomputed), pref, is1vr (the list's flag), n, mean_w, sd_w, value (= pref mean_w), error
    (= pref sd_w / sqrt(n)), contrib and contrib_err (weight_poly(1) times value, error), kurtosis (= n m4 / m2^2),
    nbad, maxw_over_absmean (= max|w| / |mean|), roundoff_max_over_sd, roundoff_rms_over_sd, core_ns_per_sample
    (= tau_ref, on the reference device), ncontrib (k: the contributions that sampled it, from their coverage in the
    manifest); non-finite numbers as None.
    fields: keep only these keys (and compute only what they need).  A format-1 ledger: its classes.json."""
    fmt, e = _entry(ledger, order)
    keep = set(fields) if fields else None
    if fmt != FORMAT3:
        p = os.path.join(ledger, order, "classes.json")
        if not os.path.exists(p):
            return
        with open(p) as fh:      # one class per line (dump_json)
            inside = False
            for line in fh:
                if not inside:
                    inside = line.startswith(' "classes": [')
                    continue
                s = line.strip().rstrip(",")
                if s.startswith("{"):
                    r = json.loads(s)
                    yield {k: v for k, v in r.items() if k in keep} if keep else r
        return
    ctx = _Order(ledger, order, e)
    want = lambda *ks: keep is None or any(k in keep for k in ks)
    pref, kind, L = e.get("pref"), e["kind"], e["L"]
    surv = ctx.survivors() if want("ncontrib") else None
    for k in ctx.chunks():
        st, wt = ctx.read("stats", k, verify=verify), ctx.read("weights", k, verify=verify)
        nm = ctx.names(k) if want("nickel", "salt", "is1vr") else None
        kc = _counts_in_chunk(ctx, e.get("contributions", []), k, surv) if want("ncontrib") else None
        memo = {}
        for i, (line, wl) in enumerate(zip(st, wt)):
            if not line:
                continue
            R = stats_regs(line)
            if wl not in memo:
                memo[wl] = weight_floats(wl)
            wp, S = memo[wl]
            n, m = R[N_], R[MEAN]
            sd = math.sqrt(R[M2] / (n - 1)) if n > 1 else float("nan")
            val, err = pref * m, pref * sd / math.sqrt(n)
            w1 = poly(wp, 1)
            r = {"n": n, "mean_w": m, "sd_w": sd, "value": val, "error": err, "contrib": w1 * val,
                 "contrib_err": abs(w1) * err, "weight_poly": list(wp), "S": S, "pref": pref, "nbad": R[NBAD] or 0,
                 "ncontrib": kc[i] if kc is not None else None, "core_ns_per_sample": R[TAU],
                 "kurtosis": n * R[M4] / (R[M2] * R[M2]) if R[M4] is not None and R[M2] > 0 else None,
                 "maxw_over_absmean": R[WMAXABS] / abs(m) if R[WMAXABS] is not None and m else None,
                 "roundoff_max_over_sd": R[RMAX] / sd if R[RMAX] is not None and sd and math.isfinite(sd) else None,
                 "roundoff_rms_over_sd": math.sqrt(R[R2] / n) / sd if R[R2] is not None and sd and math.isfinite(sd) else None}
            if nm is not None:
                nk, fl = nm[i]
                r["nickel"], r["is1vr"] = nk, bool(fl)
                if want("salt"):
                    r["salt"] = class_salt(kind, L, nk)
            yield _finite({x: v for x, v in r.items() if x in keep} if keep else r)


def load_classes(ledger, order, fields=None):
    """iter_classes as a list (memory grows with the order: restrict fields for large ones)"""
    return list(iter_classes(ledger, order, fields))


def _rec(e, cid):
    rec = next((c for c in e.get("contributions", []) if c["id"] == cid), None)
    if rec is None:
        raise LedgerError("no contribution %s in the manifest" % cid)
    return rec


def contribution_file(ledger, order, cid, archive=None):
    """the run file of a contribution: format 1, ledger/<order>/contributions/<id>.json; format 3, its archived copy"""
    fmt, e = _entry(ledger, order)
    if fmt != FORMAT3:
        return contrib_path(ledger, order, cid)
    _rec(e, cid)
    p = archive_path(ledger, order, cid, archive)
    if not os.path.exists(p):
        raise LedgerError("format 3 keeps no per-run data in the ledger: contribution %s is read from the archive, "
                          "and %s is missing (merge it with its run, or give --archive / archive=)" % (cid, p))
    return p


def read_contribution(path):
    """a run file (JSON, plain or .xz) as a dict"""
    if path.endswith(".xz"):
        with lzma.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return load_json(path)


def iter_contribution_classes(ledger, order, cid, archive=None):
    """Per class of one contribution, the run JSON's own dicts (every field): format 1 from the ledger, format 3 from
    the archived run (LedgerError when the archive does not have it); read streaming."""
    for ev, c in stream_run(contribution_file(ledger, order, cid, archive)):
        if ev == "class":
            yield c


def iter_order_contributions(ledger, order, archive=None):
    """(id, class) for every class of every contribution of an order, contribution by contribution"""
    _, e = _entry(ledger, order)
    for rec in e.get("contributions", []):
        for c in iter_contribution_classes(ledger, order, rec["id"], archive):
            yield rec["id"], c


def load_contribution(ledger, order, cid, archive=None):
    """a contribution's run JSON as a dict (format 3: the archived run; memory grows with its classes)"""
    return read_contribution(contribution_file(ledger, order, cid, archive))


def load_contribution_header(ledger, order, cid, archive=None):
    """the run-level fields of a contribution's run (without classes, dropped graphs and checkpoint rows)"""
    h = None
    for ev, x in stream_run(contribution_file(ledger, order, cid, archive)):
        if ev == "header":
            h = x
    return h


# ------------------------------------------------------------------------------------------------- formatting
def fmt_ve(v, e, nd=2):
    """30.4698481, 1.37e-4 -> '30.46985(14)'"""
    if e is None or not math.isfinite(e) or e <= 0:
        return "%.10g" % v
    d = nd - 1 - int(math.floor(math.log10(e)))
    u = int(round(e * 10 ** d))
    if u >= 10 ** nd:            # 9.96e-5 -> 10(0) at nd = 2: one digit less
        d -= 1
        u = int(round(e * 10 ** d))
    if d <= 0:
        return "%.0f(%d)" % (round(v, d), round(e))
    return "%.*f(%d)" % (d, v, u)


def fmt_sci(x, nd=2):
    """1.23e13 -> ('1.2', 13)"""
    if x is None or x == 0 or not math.isfinite(x):
        return None
    ex = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** ex
    if round(abs(m), nd - 1) >= 10:
        m /= 10
        ex += 1
    return "%.*f" % (nd - 1, m), ex


# ------------------------------------------------------------------------------------------------- command line
def _pick(ledger, name=None):
    orders = ensure_orders(load_orders(ledger))
    if name:
        kind, L = parse_order(name)
        e = orders["orders"].get(name) or default_entry(kind, L)
    else:
        cand = [(e.get("rel_err") or 0, o) for o, e in orders["orders"].items()
                if order_status(e) == "open" and e.get("list")]
        if not cand:
            sys.exit("pick: no open order with data (name one: rebar-refine.sh --order h9 ...)")
        name = max(cand)[1]
        e = orders["orders"][name]
    if not list_path(e["kind"], e["L"]):
        sys.exit("pick: order %s has no list in integrator/data/lists" % name)
    print(name, e["kind"], e["L"], list_path(e["kind"], e["L"]))


def _pending(ledger, name=None):
    seeds = load_seeds(ledger)
    for o, d in sorted(seeds["orders"].items()):
        if name and o != name:
            continue
        for s, r in sorted(d.items()):
            if r.get("state") == "running":
                print(o, s, r["id"])
                return


def _reserve(ledger, order, label, devtag):
    seeds = load_seeds(ledger)
    s = fresh_seed(seeds)
    cid = "%s-%s-%s-%s-seed%s" % (time.strftime("%Y%m%d", time.gmtime()), devtag, order, label, s)
    seeds["orders"].setdefault(order, {})[s] = {"state": "running", "id": cid, "utc": utc_now()}
    save_seeds(ledger, seeds)
    print(cid, s)


def _status(ledger):
    orders = load_orders(ledger)
    print("%-5s %-8s %-22s %-10s %8s %10s %5s %s" % ("order", "status", "sum (N=1)", "rel.err", "classes", "samples", "runs", "target"))
    for o, e in orders["orders"].items():
        s = fmt_ve(e["sum"], e["err"]) if e.get("sum") is not None else "---"
        print("%-5s %-8s %-22s %-10s %8s %10s %5d %s" % (o, e.get("status"), s, "%.2e" % e["rel_err"] if e.get("rel_err") else "---",
              e.get("classes", "---"), "%.3g" % e["samples"] if e.get("samples") else "---", len(e.get("contributions", [])),
              e.get("target_rel")))


def main(argv):
    ledger = LEDGER
    if len(argv) > 2 and argv[1] == "--ledger":
        ledger, argv = argv[2], argv[:1] + argv[3:]
    cmds = {"pick": (_pick, 0, 1), "pending": (_pending, 0, 1), "reserve": (_reserve, 3, 3), "status": (_status, 0, 0)}
    if len(argv) < 2 or argv[1] not in cmds or not cmds[argv[1]][1] <= len(argv) - 2 <= cmds[argv[1]][2]:
        sys.exit("usage: rebar_ledger.py [--ledger DIR] pick [ORDER] | pending [ORDER] | reserve ORDER LABEL DEVTAG | status")
    cmds[argv[1]][0](ledger, *argv[2:])


if __name__ == "__main__":
    main(sys.argv)
