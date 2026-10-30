#!/usr/bin/env python3
# sumrules.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Nickel-index parser, automorphism counting and O(N) tensor factors, used by the checkers. Run on its own
# it checks thesis p.107 (section 6.4.3.1) sum rules sum_G S_G T_G(N) against Nickel-index graph lists,
# S_G = E!/|Aut G|, T_G(N) = 3^-V sum over vertex pairings of N^(closed loops), and counts the
# one-vertex-reducible (cut-vertex) graphs.
#
# usage: sumrules.py FILE...  (one Nickel index per line, or tab-separated fields containing one)
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
import sys, itertools
from fractions import Fraction
from math import factorial
from collections import Counter

def parse(nk):
    parts = nk.split('|')
    # trailing empty part after final '|'
    if parts and parts[-1] == '':
        parts = parts[:-1]
    V = len(parts)
    ext = [0]*V
    edges = []  # list of (u,v), u<=v
    for u, s in enumerate(parts):
        for ch in s:
            if ch == 'e':
                ext[u] += 1
            else:
                v = int(ch, 36)
                edges.append((u, v))
    return V, ext, edges

def aut_order(V, ext, edges):
    m = Counter()
    for (u, v) in edges:
        m[(min(u, v), max(u, v))] += 1
    deg = [ext[u] for u in range(V)]
    for (u, v) in edges:
        deg[u] += 1; deg[v] += 1
    # vertex-permutation automorphisms by backtracking
    adj = [[m.get((min(a, b), max(a, b)), 0) for b in range(V)] for a in range(V)]
    sig = [(ext[u], adj[u][u], tuple(sorted(adj[u]))) for u in range(V)]
    count = 0
    perm = [-1]*V
    used = [False]*V
    def bt(i):
        nonlocal count
        if i == V:
            count += 1; return
        for c in range(V):
            if used[c] or sig[c] != sig[i]:
                continue
            ok = True
            for j in range(i):
                if adj[i][j] != adj[c][perm[j]]:
                    ok = False; break
            if ok:
                used[c] = True; perm[i] = c
                bt(i+1)
                used[c] = False
        perm[i] = -1
    bt(0)
    f = count
    for (u, v), k in m.items():
        if u == v:
            f *= (2**k) * factorial(k)
        else:
            f *= factorial(k)
    for u in range(V):
        f *= factorial(ext[u])
    return f

def on_factor(V, ext, edges):
    # half-edges: per vertex list of half-edge ids; internal edge e has halves 2e, 2e+1; externals separate ids
    halves = [[] for _ in range(V)]
    nE = len(edges)
    for e, (u, v) in enumerate(edges):
        halves[u].append(2*e); halves[v].append(2*e+1)
    nid = 2*nE
    for u in range(V):
        for _ in range(ext[u]):
            halves[u].append(nid); nid += 1
    NH = nid
    pairings = []
    for u in range(V):
        h = halves[u]
        assert len(h) == 4, (u, h)
        a, b, c, d = h
        pairings.append([((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))])
    poly = Counter()
    # partner via internal edge: 2e <-> 2e+1 ; externals have no edge partner
    epart = {}
    for e in range(nE):
        epart[2*e] = 2*e+1; epart[2*e+1] = 2*e
    for choice in itertools.product(range(3), repeat=V):
        vpart = {}
        for u in range(V):
            (x, y), (z, w) = pairings[u][choice[u]]
            vpart[x] = y; vpart[y] = x; vpart[z] = w; vpart[w] = z
        seen = set(); loops = 0
        # open strands start at external halves
        for s in range(2*nE, NH):
            if s in seen: continue
            cur = s
            while True:
                seen.add(cur)
                nxt = vpart[cur]; seen.add(nxt)
                if nxt in epart:
                    cur = epart[nxt]
                    if cur in seen: break
                else:
                    break
        for s in range(2*nE):
            if s in seen: continue
            loops += 1
            cur = s
            while cur not in seen:
                seen.add(cur)
                nxt = vpart[cur]; seen.add(nxt)
                cur = epart[nxt]
        poly[loops] += 1
    tot = 3**V
    return {k: Fraction(c, tot) for k, c in poly.items()}

def is_1vr(V, edges):
    adj = [set() for _ in range(V)]
    for (u, v) in edges:
        if u != v:
            adj[u].add(v); adj[v].add(u)
    for cut in range(V):
        rest = [x for x in range(V) if x != cut]
        if not rest: continue
        seen = {rest[0]}; st = [rest[0]]
        while st:
            x = st.pop()
            for y in adj[x]:
                if y != cut and y not in seen:
                    seen.add(y); st.append(y)
        if len(seen) < len(rest):
            return True
    return False

def run(path):
    tot = Counter(); n = 0; n1vr = 0; tot_non1vr = Counter()
    with open(path) as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            nk = next((x for x in f if '|' in x), None)
            if nk is None: continue
            V, ext, edges = parse(nk)
            E = sum(ext)
            S = Fraction(factorial(E), aut_order(V, ext, edges))
            T = on_factor(V, ext, edges)
            r = is_1vr(V, edges)
            n += 1; n1vr += r
            for k, c in T.items():
                tot[k] += S*c
                if not r: tot_non1vr[k] += S*c
    return n, n1vr, tot, tot_non1vr

def fmt(p):
    ks = sorted(p)
    from math import gcd
    lcm = lambda a, b: a*b//gcd(a, b)
    den = 1
    for k in ks: den = lcm(den, p[k].denominator)
    return '(' + ' + '.join(f'{p[k]*den}N^{k}' for k in ks) + f')/{den}'

if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit('usage: sumrules.py FILE...  (files with one Nickel index per line, or tab-separated fields)')
    for path in sys.argv[1:]:
        n, n1, tot, tn = run(path)
        print(path, 'graphs', n, '1VR', n1, 'non-1VR', n-n1)
        print('  sum S*T(N) =', fmt(tot), ' at N=1:', float(sum(tot.values())))
        sys.stdout.flush()
