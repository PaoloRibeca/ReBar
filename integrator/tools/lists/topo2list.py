#!/usr/bin/env python3
"""Convert a compact list of the generator 'topologies' (GAU strings, one graph per line: graphs/lists) into
a rebar graph list, independently of the generator's own -N output.

usage: python3 -I topo2list.py [--gaugraph PATH] [--keep-selfloops] RAW > LIST

Each string is decoded and given its canonical Nickel index by gaugraph.py (graphs/tools; GAU syntax documented
there).  The output has the format of `rebar_cpu gen` (src/main.cpp cmd_gen), sorted in the same Nickel order
('|' < 'e' < '0' < '1' < ...):
    index <TAB> Nickel index <TAB> S = E!/|Aut G| (%.12g) <TAB> 1VR=0|1 <TAB> selfloop=0|1
S comes from gaugraph's GAU weight (globalSymmetryFactor), the flags from gaugraph's predicates; listcheck
recomputes all of them with the integrator's own code, and make_lists.sh compares this output with the generator's.
Graphs with self-loops are dropped unless --keep-selfloops; the script refuses a list with a duplicate Nickel index,
a graph that is not 1PI, or one with a typed line.
"""
import importlib.util
import os
import sys

sys.dont_write_bytecode = True   # write no __pycache__ next to gaugraph.py
DEFAULT_GG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'graphs', 'tools', 'gaugraph.py'))


def nk_key(s):
    return [0 if c == '|' else 1 if c == 'e' else 2 + (ord(c) - 48 if c <= '9' else ord(c) - 55) for c in s]


def main(argv):
    gg_path, keep, raw = DEFAULT_GG, False, None
    i = 0
    while i < len(argv):
        if argv[i] == '--gaugraph': gg_path = argv[i + 1]; i += 2
        elif argv[i] == '--keep-selfloops': keep = True; i += 1
        elif raw is None: raw = argv[i]; i += 1
        else: raise SystemExit(__doc__)
    if raw is None: raise SystemExit(__doc__)
    spec = importlib.util.spec_from_file_location('gaugraph', gg_path)
    gg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gg)
    rows, seen, nraw, nsl = [], set(), 0, 0
    for line in open(raw):
        s = line.strip()
        if not s: continue
        nraw += 1
        g = gg.parse(s)
        if not g.is_plain(): raise SystemExit('typed line or effective vertex in %r' % s)
        if not g.is_1pi(): raise SystemExit('not 1PI: %r' % s)
        sl = g.selfloops() > 0
        if sl: nsl += 1
        if sl and not keep: continue
        nk = g.nickel()
        if nk in seen: raise SystemExit('duplicate Nickel index %s (from %r)' % (nk, s))
        seen.add(nk)
        rows.append((nk_key(nk), nk, float(g.weight()), int(g.is_1vr()), int(sl)))
    rows.sort()
    out = sys.stdout
    for k, (_key, nk, S, vr, sl) in enumerate(rows, 1):
        out.write('%d\t%s\t%.12g\t1VR=%d\tselfloop=%d\n' % (k, nk, S, vr, sl))
    sys.stderr.write('%s: %d raw strings, %d with self-loops, %d written\n' % (raw, nraw, nsl, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
