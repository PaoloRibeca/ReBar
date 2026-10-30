#!/usr/bin/env python3
"""Checks of a rebar graph list against the output of listcheck (the integrator's own graph code) and the exact
zero-dimensional O(N) sum rules of zerodim_on.py.  Standard library only.

usage:
  verify_lists.py sumrule FAMILY L LIST CHECK      FAMILY: c4 (Gamma4, self-loop-free), c2 (Gamma2, self-loop-free),
                                                   c2B (Gamma2, self-loops included); CHECK = listcheck output for LIST
  verify_lists.py sets A B                         Nickel sets of two lists (first field containing '|')
  verify_lists.py sample LIST CHECK STEP           every STEP-th graph: |Aut| and O(N) factor recomputed by
                                                   gaugraph.py (GAU's automorphismGroupDim + on_factor), vs CHECK
Options: --gaugraph PATH (default ../../../graphs/tools/gaugraph.py from this directory).

sumrule checks, for every graph: the S column of LIST (gaugraph's GAU weight, %.12g) equals the integrator's
sym_factor printed the same way, and the 1VR/selfloop flags equal the integrator's predicates; then
  exact:  sum_G E!/|Aut G| * on_counts_k / 3^V   (integers from the integrator's aut_order and on_factor)
  double: sum_G S_G * on_k                       (the doubles the integrator itself uses, summed in list order)
against c_L(N) of zerodim_on.py, coefficient by coefficient (exact equality; max relative deviation of the double sum).
"""
import importlib.util
import os
import sys
from fractions import Fraction
from math import factorial

sys.dont_write_bytecode = True   # leave no tools/lists/__pycache__ behind
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zerodim_on  # noqa: E402

GG = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'graphs', 'tools', 'gaugraph.py'))


def nickels(path):
    out = []
    for line in open(path):
        if not line.strip() or line.startswith('#'): continue
        nk = next((''.join(t.split()) for t in line.rstrip('\n').split('\t') if '|' in t), None)
        if nk: out.append((nk, line.rstrip('\n').split('\t')))
    return out


def read_check(path):
    rows = []
    for line in open(path):
        f = line.rstrip('\n').split('\t')
        rows.append({'nk': f[1], 'V': int(f[2]), 'E': int(f[3]), 'L': int(f[4]), 'legs': f[5], '1pi': f[6] == '1', '1vr': f[7] == '1',
                     'sl': f[8] == '1', 'canon': f[9] == '1', 'brute': f[10], 'aut': int(f[11]), 'S': float(f[12]),
                     'counts': [int(x) for x in f[13].split(',')], 'on': [float(x) for x in f[14].split(',')], 'proto_on': f[15]})
    return rows


def cmd_sumrule(family, L, lst, chk):
    entries = nickels(lst); rows = read_check(chk)
    ok = True; msgs = []
    if len(entries) != len(rows) or any(e[0] != r['nk'] for e, r in zip(entries, rows)):
        print('FAIL: list and check file differ'); return 1
    # columns of the list (gen format) vs the integrator's own values
    nS = nflag = ncol = 0
    for (nk, f), r in zip(entries, rows):
        if len(f) >= 5 and f[3].startswith('1VR='):
            ncol += 1
            if f[2] != '%.12g' % r['S']: nS += 1
            if f[3] != '1VR=%d' % r['1vr'] or f[4] != 'selfloop=%d' % r['sl']: nflag += 1
    if nS or nflag: ok = False
    if ncol:
        msgs.append('S column vs integrator sym_factor: %d of %d differ; 1VR/selfloop flags vs integrator: %d differ' % (nS, ncol, nflag))
    else:
        msgs.append('no S / 1VR / selfloop columns (Phi4tools format): weights only from the integrator')
    # structure
    E = 4 if family == 'c4' else 2
    Vexp = L + 1 if family == 'c4' else L
    bad = [r['nk'] for r in rows if not (r['1pi'] and r['canon'] and r['E'] == E and r['L'] == L and r['V'] == Vexp and r['brute'] != '0'
                                         and r['proto_on'] != '0')]
    if family in ('c4', 'c2'): bad += [r['nk'] for r in rows if r['sl']]
    nk_set = set(r['nk'] for r in rows)
    if bad or len(nk_set) != len(rows): ok = False
    msgs.append('%d graphs, %d distinct; all 1PI, canonical, E = %d, L = %d, V = %d%s: %s; brute-force canonical checked %d; '
                'integrator on_factor bit-identical %d/%d' % (len(rows), len(nk_set), E, L, Vexp, ', no self-loops' if family != 'c2B' else '',
                'yes' if not bad else 'NO (%d)' % len(bad), sum(r['brute'] == '1' for r in rows), sum(r['proto_on'] == '1' for r in rows),
                sum(r['proto_on'] != '-' for r in rows)))
    # sums
    exact = []
    dsum = []
    for r in rows:
        S = Fraction(factorial(r['E']), r['aut'])
        if float(S) != r['S']: ok = False; msgs.append('sym_factor not the correctly rounded E!/|Aut| for %s' % r['nk'])
        p3 = 3 ** r['V']
        if sum(r['counts']) != p3: ok = False; msgs.append('pairing counts do not add up to 3^V for %s' % r['nk'])
        for k, c in enumerate(r['counts']):
            while len(exact) <= k: exact.append(Fraction(0))
            exact[k] += S * Fraction(c, p3)
        for k, c in enumerate(r['on']):
            while len(dsum) <= k: dsum.append(0.0)
            dsum[k] += r['S'] * c
    exact = zerodim_on.ptrim(exact)
    ref = zerodim_on.series(L + 2)[family][L]
    eq = exact == ref
    ok = ok and eq
    rel = max(abs(d - float(x)) / abs(float(x)) for d, x in zip(dsum, ref)) if len(dsum) == len(ref) else float('inf')
    if rel > 1e-12: ok = False
    msgs.append('sum_G S_G T_G(N) = %s' % zerodim_on.fmt(exact))
    msgs.append('zero-dimensional c_L(N) = %s -> %s' % (zerodim_on.fmt(ref), 'EQUAL' if eq else 'DIFFERENT'))
    msgs.append('at N = 1: %s = %.15g; double-precision sum (integrator weights) max relative deviation %.2e' % (sum(exact), float(sum(exact)), rel))
    print('%s [%s, L = %d]' % (os.path.basename(lst), family, L))
    for m in msgs: print('  ' + m)
    print('  -> %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def cmd_sets(a, b):
    A = [e[0] for e in nickels(a)]; B = [e[0] for e in nickels(b)]
    sa, sb = set(A), set(B)
    same = sa == sb and len(A) == len(sa) and len(B) == len(sb)
    print('%s (%d, %d distinct) vs %s (%d, %d distinct): %s%s' % (a, len(A), len(sa), b, len(B), len(sb), 'IDENTICAL SETS' if same else 'DIFFERENT',
                                                              '' if same else ' (only in first %d, only in second %d)' % (len(sa - sb), len(sb - sa))))
    return 0 if same else 1


def cmd_sample(lst, chk, step, gg_path):
    spec = importlib.util.spec_from_file_location('gaugraph', gg_path)
    gg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gg)
    rows = read_check(chk); n = bad = 0
    for i in range(0, len(rows), step):
        r = rows[i]; g = gg.from_nickel(r['nk'])
        # gaugraph: vertex automorphisms (colours = legs) x line multiplicities! x self-loops (2^k k!) x legs!
        aut_v = g.automorphisms()
        mult = 1
        for a in range(g.V):
            for b in range(a, g.V):
                k = g.nlines(a, b)
                mult *= (2 ** k * factorial(k)) if a == b else factorial(k)
        legs = g.legs()
        for x in legs: mult *= factorial(x)
        on = g.on_factor()
        on_counts = [int(on.get(k, 0) * 3 ** g.V) for k in range(max(on) + 1)]
        w = g.weight()
        okr = aut_v * mult == r['aut'] and on_counts == r['counts'] and w == Fraction(factorial(sum(legs)), r['aut']) and g.nickel() == r['nk']
        n += 1; bad += not okr
        if not okr: print('  MISMATCH %s: gaugraph aut %d counts %s weight %s; integrator aut %d counts %s' % (r['nk'], aut_v * mult, on_counts, w, r['aut'], r['counts']))
    print('%s: %d graphs (every %d-th): gaugraph |Aut|, O(N) pairing counts, GAU weight and Nickel index vs the integrator: %s'
          % (os.path.basename(lst), n, step, 'NONE CHECKED' if n == 0 else 'all EQUAL' if not bad else '%d DIFFER' % bad))
    return 0 if n and not bad else 1


def main(argv):
    gg_path = GG
    if '--gaugraph' in argv:
        i = argv.index('--gaugraph'); gg_path = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    if not argv: raise SystemExit(__doc__)
    if argv[0] == 'sumrule' and len(argv) == 5: return cmd_sumrule(argv[1], int(argv[2]), argv[3], argv[4])
    if argv[0] == 'sets' and len(argv) == 3: return cmd_sets(argv[1], argv[2])
    if argv[0] == 'sample' and len(argv) == 4: return cmd_sample(argv[1], argv[2], int(argv[3]), gg_path)
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
