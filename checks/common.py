#!/usr/bin/env python3
# common.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Shared code of the reproducibility checks (checks/<id>_<slug>.py): repository paths, loading of the
# repository's own Python tools, the CPU binary of the integrator (built on demand) and the compiled helpers
# (checks/tools/graphscan.cpp, integrator/tools/lists/listcheck.cpp), a cache of their outputs keyed by the
# SHA-256 of every input, and the final verdict line of a check.
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
"""Shared code of the checks.  Python >= 3.6, standard library only.

Every check prints its numbers and ends with exactly one verdict line, written by verdict():
    PASS <id>: <summary>     exit status 0
    FAIL <id>: <summary>     exit status 1
    SKIP <id>: needs <what>  exit status 2 (the parts that could run were run; their numbers are printed)
Binaries run with OMP_NUM_THREADS=2 (override with REBAR_THREADS); Python work is split over REBAR_JOBS processes
(default 2)."""
import hashlib, importlib.util, multiprocessing, os, subprocess, sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUILD = os.path.join(HERE, "build")
CACHE = os.path.join(BUILD, "cache")
RESULTS = os.path.join(ROOT, "results", "checks")
GRAPHS = os.path.join(ROOT, "graphs")
GAU = os.path.join(GRAPHS, "lists")
INTEGRATOR = os.path.join(ROOT, "integrator")
LISTS = os.path.join(INTEGRATOR, "data", "lists")
ITOOLS = os.path.join(INTEGRATOR, "tools")
SERIES = os.path.join(ROOT, "series")
LEDGER = os.environ.get("REBAR_LEDGER") or os.path.join(ROOT, "ledger")   # REBAR_LEDGER: another ledger, as series/
PAPER = os.path.join(ROOT, "paper")
CPU = os.path.join(INTEGRATOR, "bin", "rebar_cpu")
THREADS = os.environ.get("REBAR_THREADS", "2")
JOBS = int(os.environ.get("REBAR_JOBS", "2"))
ENV = dict(os.environ, OMP_NUM_THREADS=THREADS)


def load(name, path):
    """Import a module from a file of the repository (the tools are scripts, not packages)."""
    if name in sys.modules:
        return sys.modules[name]
    d = os.path.dirname(path)
    sys.path.insert(0, d)            # their own sibling imports (rebar_ledger, zerodim_on, sumrules ...)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(d)
    return m


def gaugraph():
    return load("gaugraph", os.path.join(GRAPHS, "tools", "gaugraph.py"))


def series_module(name):
    return load(name, os.path.join(SERIES, name + ".py"))


def run(cmd, stdin=None, timeout=None, cwd=None):
    """Run a command (list), return (status, stdout, stderr); binaries get OMP_NUM_THREADS."""
    p = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       timeout=timeout, env=ENV, cwd=cwd)
    return p.returncode, p.stdout, p.stderr


def python(script, *args, timeout=None):
    """Run one of the repository's Python tools with this interpreter, isolated (-I) as the house Makefiles do."""
    return run([sys.executable, "-I", "-B", script] + [str(a) for a in args], timeout=timeout)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cpu_binary():
    """integrator/bin/rebar_cpu, built when absent (make -C integrator cpu VARIANTS=e16, about 3 minutes); None if it
    cannot be built (the build log is then in checks/build/integrator-build.log)."""
    if os.access(CPU, os.X_OK):
        return CPU
    os.makedirs(BUILD, exist_ok=True)
    log = os.path.join(BUILD, "integrator-build.log")
    print("building %s (make -C integrator cpu VARIANTS=e16; log %s) ..." % (os.path.relpath(CPU, ROOT), os.path.relpath(log, ROOT)))
    sys.stdout.flush()
    with open(log, "w") as fh:
        subprocess.run(["make", "-C", INTEGRATOR, "cpu", "VARIANTS=e16", "-j", THREADS], stdout=fh, stderr=subprocess.STDOUT)
    return CPU if os.access(CPU, os.X_OK) else None


def _compile(out, sources, extra=()):
    """Compile a helper into checks/build unless an up-to-date binary is there; returns its path or None."""
    os.makedirs(BUILD, exist_ok=True)
    if os.access(out, os.X_OK) and all(os.path.getmtime(out) >= os.path.getmtime(s) for s in sources):
        return out
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, "-O2", "-std=c++17"] + list(extra) + ["-o", out] + list(sources)
    st, so, se = run(cmd)
    if st:
        print("cannot build %s: %s\n%s" % (os.path.basename(out), " ".join(cmd), se))
        return None
    return out


def graphscan():
    return _compile(os.path.join(BUILD, "graphscan"), [os.path.join(HERE, "tools", "graphscan.cpp")], ["-fopenmp"])


def listcheck():
    src = os.path.join(INTEGRATOR, "src")
    return _compile(os.path.join(BUILD, "listcheck"), [os.path.join(ITOOLS, "lists", "listcheck.cpp"), os.path.join(src, "graph.cpp")],
                    ["-fopenmp", "-I" + src])


def cached(tag, inputs, compute):
    """Text output of compute() (a function returning a string), stored in checks/build/cache under the SHA-256 of
    the input files: an unchanged input is not recomputed.  `make clean` empties the cache."""
    h = hashlib.sha256(tag.encode())
    for p in inputs:
        h.update(sha256(p).encode())
    path = os.path.join(CACHE, "%s-%s.txt" % (tag, h.hexdigest()[:16]))
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read()
    text = compute()
    os.makedirs(CACHE, exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)
    return text


def _scan_key(path, tag="graphscan"):
    h = hashlib.sha256(tag.encode())
    for p in (path, os.path.join(HERE, "tools", "graphscan.cpp")):
        h.update(sha256(p).encode())
    base = os.path.join(CACHE, "%s-%s-%s" % (tag, os.path.basename(path), h.hexdigest()[:16]))
    return base + ".txt", base + ".nickel"


def _run_graphscan(path, opts):
    """graphscan OPTS PATH -> stdout; an xz-compressed list (.xz) is first decompressed to a file in checks/build"""
    exe = graphscan()
    if not exe:
        raise RuntimeError("graphscan could not be built (needs g++ with C++17 and OpenMP)")
    os.makedirs(CACHE, exist_ok=True)
    plain = None
    try:
        if path.endswith(".xz"):
            import lzma, shutil, tempfile
            fd, plain = tempfile.mkstemp(suffix=".txt", dir=BUILD)
            with lzma.open(path, "rb") as src, os.fdopen(fd, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
        st, so, se = run([exe] + list(opts) + [plain or path])
    finally:
        if plain:
            os.remove(plain)
    if st:
        raise RuntimeError("graphscan %s: %s" % (path, se))
    return so


def _graphscan(path):
    """run graphscan --on once on a list (summary, with the summed pairing counts of the O(N) factors, and canonical
    Nickel indices), cached in checks/build/cache"""
    summary, nickel = _scan_key(path)
    if not (os.path.exists(summary) and os.path.exists(nickel)):
        so = _run_graphscan(path, ["--on", "--nickel-out", nickel + ".tmp"])
        os.replace(nickel + ".tmp", nickel)
        with open(summary + ".tmp", "w") as fh:
            fh.write(so)
        os.replace(summary + ".tmp", summary)
    return summary, nickel


def scan(path):
    """graphscan summary of a list (cached): {key: value string}, with 'aut' -> [(E, |Aut|, graphs)] and
    'on' -> [(E, V, |Aut|, graphs, [pairing counts by closed loops, summed over those graphs])]."""
    out = {"aut": [], "on": []}
    with open(_graphscan(path)[0]) as fh:
        for line in fh:
            k, _, v = line.rstrip("\n").partition(" ")
            if k == "aut":
                E, a, n = v.split()
                out["aut"].append((int(E), int(a), int(n)))
            elif k == "on":
                E, V, a, n, c = v.split()
                out["on"].append((int(E), int(V), int(a), int(n), [int(x) for x in c.split(",")]))
            else:
                out[k] = v
    return out


def on_rows(path):
    """graphscan's pairing counts of every graph of a list, in list order, as listcheck prints them ('c0,c1,...';
    cached; meant for the lists that listcheck counts too)"""
    rows, _ = _scan_key(path, "graphscan-on")
    if not os.path.exists(rows):
        _run_graphscan(path, ["--on-out", rows + ".tmp"])
        os.replace(rows + ".tmp", rows)
    with open(rows) as fh:
        return fh.read().split()


def nickel_out(path):
    """canonical Nickel index of every graph of a list, in list order, by graphscan (cached)"""
    with open(_graphscan(path)[1]) as fh:
        return fh.read().split()


def listcheck_rows(path, V):
    """integrator/tools/lists/listcheck output for a Nickel list (cached): the integrator's own graph code
    (src/graph.cpp) on every graph, with its brute-force canonical check for V <= 7 and its own on_factor for
    V <= 8 (cost: 3^V pairings per graph; about 4 min for gamma4_L8 on 2 threads, not run for V = 10)."""
    exe = listcheck()
    if not exe:
        raise RuntimeError("listcheck could not be built (needs g++ with C++17 and OpenMP)")
    opts = ["--brute", "all" if V <= 7 else "none", "--proto-on", "all" if V <= 8 else "none"]
    srcs = [os.path.join(ITOOLS, "lists", "listcheck.cpp"), os.path.join(INTEGRATOR, "src", "graph.cpp"),
            os.path.join(INTEGRATOR, "src", "graph.h")]

    def compute():
        st, so, se = run([exe, path] + opts)
        return so + "".join("#stderr " + x + "\n" for x in se.splitlines()) + "#status %d\n" % st
    return cached("listcheck-" + os.path.basename(path) + "-" + "".join(opts), [path] + srcs, compute)


def zerodim(K=12):
    """exact zero-dimensional coefficients (integrator/tools/lists/zerodim_on.py, cached):
    {'c4' | 'c2' | 'c2B' | 'c4B': {L: [Fraction coefficients of N^0, N^1, ...]}}"""
    from fractions import Fraction
    src = os.path.join(ITOOLS, "lists", "zerodim_on.py")

    def compute():
        z = load("zerodim_on", src).series(K)
        return "".join("%s %d %s\n" % (fam, L, " ".join(str(c) for c in cs)) for fam in sorted(z) for L, cs in sorted(z[fam].items()))
    out = {}
    for line in cached("zerodim-%d" % K, [src], compute).splitlines():
        f = line.split()
        out.setdefault(f[0], {})[int(f[1])] = [Fraction(x) for x in f[2:]]
    return out


def histogram(text):
    """'4:27 5:3' -> {4: 27, 5: 3}"""
    return {int(a): int(b) for a, b in (x.split(":") for x in text.split())}


def pmap(fn, items, jobs=None):
    """map over REBAR_JOBS processes (fn must be a module-level function)"""
    jobs = JOBS if jobs is None else jobs
    items = list(items)
    if jobs <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    with multiprocessing.Pool(jobs) as pool:
        return pool.map(fn, items, chunksize=1)


def rel(path):
    return os.path.relpath(path, ROOT)


def verdict(cid, ok, summary, skip=None):
    """Print the final line and exit: ok True -> PASS (0), False -> FAIL (1); skip='what' -> SKIP (2) if ok."""
    sys.stdout.flush()
    if not ok:
        print("FAIL %s: %s" % (cid, summary))
        sys.exit(1)
    if skip:
        print("SKIP %s: needs %s (%s)" % (cid, skip, summary))
        sys.exit(2)
    print("PASS %s: %s" % (cid, summary))
    sys.exit(0)
