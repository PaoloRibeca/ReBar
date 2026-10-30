#!/usr/bin/env python3
# bruteforce.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# An independent check of the generator on a small list of vertex valences: enumerates every labelled graph
# whose vertices have at most their valence in line ends and self-loops (the rest are external legs), groups
# the graphs by isomorphism class (the minimum over the vertex permutations of valences, self-loops, legs and
# upper triangle) and prints, for every number of legs E, the number of graphs and the sum of S_G = E!/|Aut G|
# over all graphs, over the connected ones and over the one-particle-irreducible ones. The three sums are also
# computed from the zero-dimensional theory with one coupling per valence, Z(J) = int dx exp(-x^2/2 +
# sum_d g_d x^d/d! + J x) as a multivariate series: all graphs from exp(-J^2/2) Z (the factor removes the
# propagator joining two legs), connected from W = log Z, 1PI from the Legendre transform of W; a MISMATCH
# between the enumeration and the series is flagged. With -b the generator is run on every list (-t, every E,
# connected and 1PI) and its -w output (count and sum per E) is compared with the enumeration.
#   python3 -I bruteforce.py [-b TOPOLOGIES] DEGREES...
# e.g. "3,3,4" or "4,4,6"; a second takes three vertices, two minutes five (fewer with small valences).
# Standard library only. Exit status 1 on any mismatch.
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

import itertools
import re
import subprocess
import sys
from fractions import Fraction
from math import factorial


def graphs(val):
    """Every labelled graph with deg_i <= val_i: the line multiplicities (dict over pairs) and self-loops."""
    V = len(val)
    pairs = [(i, j) for i in range(V) for j in range(i + 1, V)]

    def rec(k, m, deg):
        if k == len(pairs):
            ranges = [range((val[i] - deg[i]) // 2 + 1) for i in range(V)]
            for t in itertools.product(*ranges):
                yield dict(m), t
            return
        i, j = pairs[k]
        for x in range(min(val[i] - deg[i], val[j] - deg[j]) + 1):
            m[(i, j)] = x
            deg[i] += x
            deg[j] += x
            yield from rec(k + 1, m, deg)
            deg[i] -= x
            deg[j] -= x
        del m[(i, j)]
    yield from rec(0, {}, [0] * V)


def matrix(V, m):
    A = [[0] * V for _ in range(V)]
    for (i, j), x in m.items():
        A[i][j] = A[j][i] = x
    return A


def components(V, A, skip=None):
    """Connected components through the lines, one line skip = (i, j) removed."""
    seen = [False] * V
    n = 0
    for s in range(V):
        if seen[s]:
            continue
        n += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in range(V):
                k = A[u][v]
                if skip and (u, v) in (skip, skip[::-1]):
                    k -= 1
                if k and not seen[v]:
                    seen[v] = True
                    stack.append(v)
    return n


def opi(V, A):
    for i in range(V):
        for j in range(i + 1, V):
            if A[i][j] == 1 and components(V, A, (i, j)) > components(V, A):
                return False
    return True


def enumerate_classes(val):
    """Per E: [all, sum S, connected, sum S, 1PI, sum S] over the isomorphism classes."""
    V = len(val)
    classes = {}
    for m, t in graphs(val):
        A = matrix(V, m)
        deg = [sum(A[i]) + 2 * t[i] for i in range(V)]
        legs = [val[i] - deg[i] for i in range(V)]
        forms = []
        for p in itertools.permutations(range(V)):
            forms.append((tuple(val[p[q]] for q in range(V)), tuple(t[p[q]] for q in range(V)),
                          tuple(legs[p[q]] for q in range(V)),
                          tuple(A[p[q]][p[r]] for q in range(V) for r in range(q + 1, V))))
        best = min(forms)
        if best in classes:
            continue
        aut = forms.count(best)
        for x in m.values():
            aut *= factorial(x)
        for i in range(V):
            aut *= 2 ** t[i] * factorial(t[i]) * factorial(legs[i])
        E = sum(legs)
        classes[best] = (E, Fraction(factorial(E), aut), components(V, A) == 1, opi(V, A))
    out = {}
    for E, S, conn, pi in classes.values():
        d = out.setdefault(E, [0, Fraction(0), 0, Fraction(0), 0, Fraction(0)])
        d[0] += 1
        d[1] += S
        if conn:
            d[2] += 1
            d[3] += S
        if conn and pi:
            d[4] += 1
            d[5] += S
    return out


def series(val):
    """Per E: E! times the coefficient of prod_d g_d^(n_d) J^E (or phi^E) in exp(-J^2/2) Z, W and Gamma-tilde.
    A series is a dict {(n_d..., e): Fraction}, truncated to n_d <= N_d and e <= sum of the valences."""
    ds = sorted(set(val))
    N = [val.count(d) for d in ds]
    Emax = sum(val)
    nv = len(ds)

    def ok(k):
        return all(k[i] <= N[i] for i in range(nv)) and k[nv] <= Emax

    def add(a, b, c=1):
        r = dict(a)
        for k, v in b.items():
            r[k] = r.get(k, 0) + c * v
        return {k: v for k, v in r.items() if v != 0}

    def mul(a, b):
        r = {}
        for ka, va in a.items():
            for kb, vb in b.items():
                k = tuple(x + y for x, y in zip(ka, kb))
                if ok(k):
                    r[k] = r.get(k, 0) + va * vb
        return {k: v for k, v in r.items() if v != 0}

    one = {(0,) * (nv + 1): Fraction(1)}

    def moment(n):
        return Fraction(0) if n % 2 else Fraction(factorial(n) // (2 ** (n // 2) * factorial(n // 2)))
    # P(x) = sum_d g_d x^d/d! + J x, by power of x
    P = {}
    for i, d in enumerate(ds):
        k = [0] * (nv + 1)
        k[i] = 1
        P.setdefault(d, {})[tuple(k)] = Fraction(1, factorial(d))
    k = [0] * (nv + 1)
    k[nv] = 1
    P.setdefault(1, {})[tuple(k)] = Fraction(1)
    # Z = sum_k <P^k>/k!
    Z = dict(one)
    Pk = {0: dict(one)}
    kmax = sum(N) + Emax
    for kk in range(1, kmax + 1):
        new = {}
        for pa, ca in Pk.items():
            for pb, cb in P.items():
                prod = mul(ca, cb)
                if prod:
                    new[pa + pb] = add(new.get(pa + pb, {}), prod)
        Pk = new
        if not Pk:
            break
        term = {}
        for pw, c in Pk.items():
            mo = moment(pw)
            if mo:
                term = add(term, c, mo)
        Z = add(Z, term, Fraction(1, factorial(kk)))
    # W = log Z
    z = add(Z, one, -1)
    W = {}
    zn = dict(one)
    for n in range(1, kmax + 2):
        zn = mul(zn, z)
        if not zn:
            break
        W = add(W, zn, Fraction((-1) ** (n + 1), n))

    # The Legendre transform: phi = W'(J), J(phi) by iteration, Gamma(phi) = J phi - W(J)
    def dJ(a):
        r = {}
        for k, v in a.items():
            if k[nv]:
                kk = k[:nv] + (k[nv] - 1,)
                r[kk] = r.get(kk, 0) + v * k[nv]
        return r

    def subst(a, Jphi):
        r = {}
        powers = {0: dict(one)}
        for k, v in a.items():
            e = k[nv]
            if e not in powers:
                for q in range(max(powers) + 1, e + 1):
                    powers[q] = mul(powers[q - 1], Jphi)
            r = add(r, mul({k[:nv] + (0,): v}, powers[e]))
        return r
    phi = {(0,) * nv + (1,): Fraction(1)}
    J = dict(phi)
    for _ in range(kmax + 2):
        Wp = subst(dJ(W), J)
        Jn = add(phi, add(Wp, J, -1), -1)
        if Jn == J:
            break
        J = Jn
    Gam = add(mul(J, phi), subst(W, J), -1)
    Gt = add({(0,) * nv + (2,): Fraction(1, 2)}, Gam, -1)
    # exp(-J^2/2) Z: every graph with legs on vertices
    expo = dict(one)
    term = dict(one)
    J2 = {(0,) * nv + (2,): Fraction(-1, 2)}
    for n in range(1, Emax // 2 + 2):
        term = mul(term, J2)
        if not term:
            break
        expo = add(expo, term, Fraction(1, factorial(n)))
    ZJ = mul(expo, Z)
    Nk = tuple(N)

    def coef(a, E):
        return a.get(Nk + (E,), Fraction(0)) * factorial(E)
    return {E: (coef(ZJ, E), coef(W, E), coef(Gt, E)) for E in range(Emax + 1)}


def generator(binary, degrees, opi_only):
    """The generator's -w output on its own list: {E: (count, sum)}."""
    args = [binary, '-t', '-w', '-o', '/dev/null'] + (['-i'] if opi_only else []) + [degrees]
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode:
        sys.exit('%s failed on %s:\n%s' % (binary, degrees, r.stderr))
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r'(\d+) graphs with (\d+) legs: sum of E!/\|Aut\| = ([0-9/]+)', line)
        if m:
            out[int(m.group(2))] = (int(m.group(1)), Fraction(m.group(3)))
    return out


def main():
    args = sys.argv[1:]
    binary = None
    if args[:1] == ['-b']:
        binary = args[1]
        args = args[2:]
    if not args:
        sys.exit('usage: bruteforce.py [-b TOPOLOGIES] DEGREES...')
    bad = 0
    for arg in args:
        val = [int(x) for x in arg.split(',')]
        bf = enumerate_classes(val)
        gf = series(val)
        gen = (generator(binary, arg, False), generator(binary, arg, True)) if binary else None
        print('{%s}:  E   all (sum S_G)   connected (sum S_G)   1PI (sum S_G)%s'
              % (arg, '   generator: connected, 1PI' if binary else ''))
        for E in sorted(bf):
            a = bf[E]
            g = gf[E]
            flag = ''
            if (a[1], a[3], a[5]) != g:
                flag += '  MISMATCH with the series (%s %s %s)' % g
            line = '  %2d   %3d (%s)   %3d (%s)   %3d (%s)' % (E, a[0], a[1], a[2], a[3], a[4], a[5])
            if gen:
                c = gen[0].get(E, (0, Fraction(0)))
                p = gen[1].get(E, (0, Fraction(0)))
                line += '   %3d (%s), %3d (%s)' % (c[0], c[1], p[0], p[1])
                if c != (a[2], a[3]) or p != (a[4], a[5]):
                    flag += '  MISMATCH with the generator'
            if flag:
                bad += 1
            print(line + flag)
        if gen:
            extra = (set(gen[0]) | set(gen[1])) - set(bf)
            if extra:
                bad += 1
                print('  MISMATCH: the generator has graphs with E in %s' % sorted(extra))
    print('bruteforce: %s' % ('OK' if not bad else '%d MISMATCHES' % bad))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
