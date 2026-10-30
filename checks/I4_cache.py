#!/usr/bin/env python3
# I4_cache.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Check I4: a run that loads its classes from the class cache is bitwise identical to one that builds them.
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
"""I4 -- a run with the class cache is bitwise identical to one without.

Claim (paper, Sec. II B 2 "Constructing the integrands"): the built classes of a list are written once to a cache
file by a CPU-only job (`rebar build --class-cache FILE`) and loaded by every later run; a run with and without the
cache gives bitwise the same results.

Inputs: integrator/data/lists/gamma4_L5.txt (b), gamma2_L5.txt (h), gamma2_L4.txt (c).
What runs, for each order: `rebar_cpu build --class-cache F`; `rebar_cpu run ... --class-cache F` (it must report that
every class came from F) and the same run without the cache (same seed 4242, pilot 1024, 1e5 samples, 2 threads);
integrator/tools/jsoncmp.py A B --checkpoint with tolerance 0: every per-class and order number of the two JSONs
(timings excluded) and the checkpoint rows (sample counters, plan, accumulator bit patterns).
PASS: for every order the cached run used only cached classes and jsoncmp reports AGREE.
"""
import os, re, shutil, sys, tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLAIM = "a run with the class cache is bitwise identical to one without (Sec. II B 2)"
QUICK = True
ORDERS = [("b", 5, "gamma4_L5.txt", ["--drop-1vr"]), ("h", 5, "gamma2_L5.txt", ["--drop-1vr"]), ("c", 4, "gamma2_L4.txt", [])]


def main():
    exe = C.cpu_binary()
    if not exe:
        C.verdict("I4", True, "nothing run", skip="the CPU binary (make -C integrator cpu VARIANTS=e16 failed)")
    d = tempfile.mkdtemp(prefix="rebar-I4-")
    ok, cells = True, []
    try:
        for kind, L, lst, extra in ORDERS:
            base = ["--kind", kind, "--loops", str(L), "--list", os.path.join(C.LISTS, lst)] + extra
            cache = os.path.join(d, "%s%d.cache" % (kind, L))
            st, so, se = C.run([exe, "build"] + base + ["--class-cache", cache])
            print("%s%d %s" % (kind, L, (so.strip() or se.strip()).replace(cache, os.path.basename(cache))))
            runs = {}
            for tag, opt in (("cached", ["--class-cache", cache]), ("built", [])):
                out = os.path.join(d, "%s%d_%s.json" % (kind, L, tag))
                st2, so2, se2 = C.run([exe, "run"] + base + opt + ["--pilot", "1024", "--samples", "100000", "--seed", "4242",
                                                                   "--threads", C.THREADS, "--out", out])
                runs[tag] = (st2, out, se2)
            m = re.search(r"class cache: all (\d+) classes from", runs["cached"][2])
            st3, so3, se3 = C.python(os.path.join(C.ITOOLS, "jsoncmp.py"), runs["cached"][1], runs["built"][1], "--checkpoint")
            keys = [x for x in so3.splitlines() if "bitwise" in x]
            good = st == 0 and runs["cached"][0] == 0 and runs["built"][0] == 0 and m is not None and st3 == 0 and "AGREE" in so3
            print("   run with the cache (%s classes from the file) vs run without: %s; %d keys compared, %s"
                  % (m.group(1) if m else "NOT", (so3.strip().splitlines()[-1] if so3.strip() else se3.strip()).replace(d + os.sep, ""), len(keys),
                     "all bitwise identical" if st3 == 0 else "DIFFERENCES"))
            for x in keys:
                if "DIFFERENT" in x:
                    print("     " + x)
            ok = ok and good
            cells.append("%s%d %s" % (kind, L, "AGREE" if good else "DIFFER"))
    finally:
        shutil.rmtree(d)
    C.verdict("I4", ok, "cached vs built classes, bitwise (every number and the checkpoint accumulators): " + ", ".join(cells))


if __name__ == "__main__":
    main()
