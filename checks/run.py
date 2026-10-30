#!/usr/bin/env python3
# run.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Runs the reproducibility checks (all, the quick ones, or those named), times them, and writes checks/REPORT.md:
# one row per check with its claim, verdict, numbers and time.
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
"""run.py -- run the checks and write REPORT.md.

usage: run.py [--quick] [--list] [--report FILE] [ID ...]
  ID ...     the checks to run (G1 ... S5; default: all)
  --quick    only the checks marked QUICK (each about a minute or less once the CPU binary exists)
  --list     print the checks (id, quick, claim) and exit
  --report   where to write the report (default checks/REPORT.md)

Every check runs as `python3 -I checks/<id>_<slug>.py`; its output goes to the console and to checks/build/logs/<id>.log;
its verdict is its last line (PASS / FAIL / SKIP <id>: ...) and its exit status (0 / 1 / 2). REPORT.md is updated
in place: the rows of the checks run now are replaced, the others are kept with the date of their run.
Exit status 1 if any check run now FAILs (a SKIP is not a failure), else 0."""
import ast, datetime, glob, os, platform, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORDER = "GINS"


def checks():
    """{id: (path, claim, quick)} from the module-level constants CLAIM and QUICK of each script (not executed)."""
    out = {}
    for p in glob.glob(os.path.join(HERE, "[A-Z][0-9]_*.py")):
        cid = os.path.basename(p).split("_")[0]
        claim, quick = "", False
        with open(p) as fh:
            tree = ast.parse(fh.read(), p)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "CLAIM":
                    claim = ast.literal_eval(node.value)
                elif node.targets[0].id == "QUICK":
                    quick = ast.literal_eval(node.value)
        out[cid] = (p, claim, quick)
    return dict(sorted(out.items(), key=lambda kv: (ORDER.index(kv[0][0]) if kv[0][0] in ORDER else 9, int(kv[0][1:]))))


def run_one(cid, path):
    os.makedirs(os.path.join(HERE, "build", "logs"), exist_ok=True)
    log = os.path.join(HERE, "build", "logs", cid + ".log")
    t0 = time.time()
    last = ""
    with open(log, "w") as fh:
        p = subprocess.Popen([sys.executable, "-I", "-B", path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, cwd=ROOT)
        for line in p.stdout:
            sys.stdout.write("  " + line)
            sys.stdout.flush()
            fh.write(line)
            if line.strip():
                last = line.strip()
        p.wait()
    dt = time.time() - t0
    m = re.match(r"^(PASS|FAIL|SKIP) %s: (.*)$" % cid, last)
    expect = {"PASS": 0, "FAIL": 1, "SKIP": 2}
    if not m:
        return "FAIL", "no verdict line (exit status %d; log checks/build/logs/%s.log)" % (p.returncode, cid), dt
    if expect[m.group(1)] != p.returncode:
        return "FAIL", "verdict %s but exit status %d: %s" % (m.group(1), p.returncode, m.group(2)), dt
    return m.group(1), m.group(2), dt


def read_report(path):
    """rows {id: [claim, status, numbers, seconds]} and their dates, from an earlier REPORT.md"""
    rows, dates = {}, {}
    if not os.path.exists(path):
        return rows, dates
    with open(path) as fh:
        text = fh.read()
    for line in text.splitlines():
        m = re.match(r"^\| ([A-Z]\d+) \| (.*) \| (PASS|FAIL|SKIP) \| (.*) \| ([0-9.]+) \|$", line)
        if m:
            rows[m.group(1)] = [m.group(2), m.group(3), m.group(4), m.group(5)]
    m = re.search(r"<!-- rows: (.*) -->", text)
    if m:
        dates = dict(x.split("=", 1) for x in m.group(1).split())
    return rows, dates


def cell(s):
    """a table cell: '|' escaped once (rows kept from an earlier report are already escaped), newlines flattened"""
    return re.sub(r"(?<!\\)\|", r"\\|", s).replace("\n", " ")


def commit():
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           universal_newlines=True)
        if r.returncode == 0:
            d = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               universal_newlines=True).stdout.strip()
            return r.stdout.strip() + (" (with uncommitted changes)" if d else "")
    except OSError:
        pass
    return "unknown"


def write_report(path, allc, rows, dates, now_ids, argv):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
    L = ["# Reproducibility checks: report", "",
         "Written by `checks/run.py%s` on %s (Python %s, commit %s). Each claim is the paper's; the docstring of "
         "`checks/<id>_*.py` gives its inputs, what runs and what PASS means, and `checks/README.md` how to run them. SKIP: the "
         "part that needs the missing resource did not run here; the parts that could run passed, and the stored outputs of "
         "the run that had it were checked (see the numbers)." % ((" " + " ".join(argv)) if argv else "", now, platform.python_version(),
                                                                  commit()),
         "", "| id | claim | status | numbers | seconds |", "|---|---|---|---|---|"]
    # (a variable that holds a path is named without its value: no local directory goes into the report)
    env = " ".join(k if "/" in os.environ[k] else "%s=%s" % (k, os.environ[k]) for k in sorted(os.environ) if k.startswith("REBAR_"))
    if env:
        L[2] += " Set: `%s`." % env
    for cid, (p, claim, quick) in allc.items():
        if cid in rows:
            r = rows[cid]
            L.append("| %s | %s | %s | %s | %s |" % (cid, cell(claim or r[0]), r[1], cell(r[2]), r[3]))
        else:
            L.append("| %s | %s | -- | not run yet | -- |" % (cid, cell(claim)))
    n = {s: sum(1 for c in allc if c in rows and rows[c][1] == s) for s in ("PASS", "FAIL", "SKIP")}
    L += ["", "%d PASS, %d FAIL, %d SKIP of %d checks." % (n["PASS"], n["FAIL"], n["SKIP"], len(allc))]
    old = [c for c in allc if c in rows and c not in now_ids]
    if old:
        L.append("Rows kept from earlier runs: " + ", ".join("%s (%s)" % (c, dates.get(c, "?")) for c in old) + ".")
    L += ["", "<!-- rows: %s -->" % " ".join("%s=%s" % (c, dates[c]) for c in allc if c in dates)]
    with open(path + ".tmp", "w") as fh:
        fh.write("\n".join(L) + "\n")
    os.replace(path + ".tmp", path)


def main(argv):
    quick, listing, report, ids = False, False, os.path.join(HERE, "REPORT.md"), []
    it = iter(argv)
    for a in it:
        if a == "--quick":
            quick = True
        elif a == "--list":
            listing = True
        elif a == "--report":
            report = next(it)
        elif re.match(r"^[A-Z]\d+$", a):
            ids.append(a)
        else:
            print(__doc__)
            return 2
    allc = checks()
    if listing:
        for cid, (p, claim, q) in allc.items():
            print("%-3s %-5s %s" % (cid, "quick" if q else "", claim))
        return 0
    bad = [c for c in ids if c not in allc]
    if bad:
        print("unknown check(s): %s (known: %s)" % (", ".join(bad), " ".join(allc)))
        return 2
    todo = [c for c in allc if (c in ids if ids else (allc[c][2] or not quick))]
    rows, dates = read_report(report)
    now_ids = []
    t0 = time.time()
    summary = []
    for cid in todo:
        print("== %s: %s" % (cid, allc[cid][1]))
        sys.stdout.flush()
        st, msg, dt = run_one(cid, allc[cid][0])
        rows[cid] = [allc[cid][1], st, msg, "%.0f" % dt]
        now_ids.append(cid)
        dates[cid] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
        summary.append((cid, st, dt))
        print("-> %s %s (%.0f s)\n" % (st, cid, dt))
        write_report(report, allc, rows, dates, now_ids, argv)
    print("summary: " + "  ".join("%s %s" % (c, s) for c, s, _ in summary) + "   (%.0f s; report %s)" % (time.time() - t0, os.path.relpath(report, ROOT)))
    return 1 if any(s == "FAIL" for _, s, _ in summary) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
