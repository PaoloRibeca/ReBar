#!/usr/bin/env python3
# rebar_mp.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Independent multiprecision evaluation of the R-bar (Callan-Symanzik route) integrands of rebar, for the pointwise
# check I1: the forest formula evaluated term by term in 100-digit arithmetic (mpmath), with no code in common with
# integrator/src/build.cpp.
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
"""Independent R-bar (CS-route) integrand in multiprecision, for pointwise checks (needs mpmath).

The method (it shares no code with integrator/src/build.cpp):
  * renormalisation parts: proper line subsets S, (V(S),S) connected and bridgeless, legs = sum_v (4 - deg_S v)
    in {2, 4}; degree t0 (4 legs, no dot), t2 (2 legs, no dot; t0 if both legs on one vertex), t0 (2 legs, 1 dot)
  * forests: pairwise nested or vertex-disjoint
  * each forest node rho: reduced graph = lines of rho minus lines of its maximal children; t0 child contracted to
    one vertex; t2 child replaced by an effective massless line between its terminals with parameter
    eps = W_child/U_child, kept to first order (U of a 2-terminal insertion is exactly linear in eps)
  * Kirchhoff polynomials by the matrix-tree theorem: U = prod_e x_e * det(reduced Laplacian with conductances 1/x_e)
    (self-loops contribute x_e); W(a|b) = U of the graph with a and b identified
  * projective integrand returned as f(x) * prod_e x_e * (sum x)^(-pw), the normalisation of rebar's eval_point()
    (`rebar_cpu eval`), without the prefactor Gamma(L/2+1) pi^(-L/2).
Lines are numbered as rebar numbers them: in the order of the Nickel index (vertex i, then each label j of its part).

usage: rebar_mp.py eval KIND NICKEL < POINTS     one point per line (the l log-coordinates log x_e); prints the
                                                value (50 digits), sum |forest terms| and the number of forests
       rebar_mp.py verify SAMPLE [--max-loops L] [--digits D]
                                                recompute the rows of a stored sample (checks/I1) with at most L loops
                                                and compare with the stored values; exit status 1 on a mismatch
As a module: R = Rbar(V, ext, edges, kind); R.value(lx) -> (value, sum |terms|), lx a list of log x (mpmath numbers).
"""
import itertools, sys
import mpmath as mp

mp.mp.dps = 100


def det(M):
    n = len(M)
    if n == 0:
        return mp.mpf(1)
    return mp.det(mp.matrix(M))


class Rbar:
    def __init__(self, V, ext, edges, kind):
        self.V, self.ext, self.edges, self.kind = V, list(ext), [tuple(e) for e in edges], kind
        self.l = l = len(edges)
        self.L = l - V + 1
        self.deriv = kind in ('h', 'bare2d', 'int2d')
        self.ndot = {'b': 1, 'h': 1, 'c': 2}.get(kind, 0)
        self.pw = l + self.ndot - 1.5 * self.L + (1 if self.deriv else 0)
        self.rbar = kind in ('b', 'h', 'c')
        self.extv = [v for v in range(V) for _ in range(ext[v])]
        # candidate parts
        self.cand = []
        for r in range(1, l):
            for S in itertools.combinations(range(l), r):
                S = frozenset(S)
                vs = frozenset(x for i in S for x in edges[i])
                if not self._conn(vs, S) or not self._bridgeless(vs, S):
                    continue
                deg = {v: 0 for v in vs}
                for i in S:
                    a, b = edges[i]
                    deg[a] += 1; deg[b] += 1
                legs = sum(4 - deg[v] for v in vs)
                if legs not in (2, 4):
                    continue
                term = sorted(v for v in vs for _ in range(4 - deg[v]))
                if kind in ('int4', 'int2d') and not (legs == 2 and len(S) == 3 and len(vs) == 2):
                    continue
                if not (self.rbar or kind in ('int4', 'int2d')):
                    continue
                self.cand.append(dict(S=S, V=vs, legs=legs, term=term))
        if self.ndot == 1:
            self.dots = [((e,), 1) for e in range(l)]
        elif self.ndot == 2:
            self.dots = [((e, f), 1 if e == f else 2) for e in range(l) for f in range(e, l)]
        else:
            self.dots = [((), 1)]
        self.nforests = 0

    def _conn(self, vs, S):
        vs = set(vs)
        if not vs:
            return True
        adj = {v: [] for v in vs}
        for i in S:
            a, b = self.edges[i]
            adj[a].append(b); adj[b].append(a)
        st = [next(iter(vs))]; seen = {st[0]}
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); st.append(y)
        return seen == vs

    def _bridgeless(self, vs, S):
        for i in S:
            if self.edges[i][0] == self.edges[i][1]:
                continue
            if not self._conn(vs, S - {i}):
                return False
        return True

    # ---------- Kirchhoff polynomial of a multigraph given as (vertex classes, list of (u, v, x)) ----------
    @staticmethod
    def kirchhoff(nv, lines):
        """nv vertices 0..nv-1, lines (u, v, x) with x an mpf (effective lines included).  U = sum_T prod_{e not in T} x_e."""
        prod_all = mp.mpf(1)
        Lap = [[mp.mpf(0)] * nv for _ in range(nv)]
        for (u, v, x) in lines:
            prod_all *= x
            if u == v:
                continue
            c = 1 / x
            Lap[u][u] += c; Lap[v][v] += c; Lap[u][v] -= c; Lap[v][u] -= c
        # connectivity
        par = list(range(nv))
        def f(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for (u, v, x) in lines:
            par[f(u)] = f(v)
        if len({f(a) for a in range(nv)}) != 1:
            return mp.mpf(0)
        M = [row[1:] for row in Lap[1:]]
        return prod_all * det(M)

    def reduced_poly(self, rho_S, rho_V, children, split, x, ident=None):
        """U of node rho's reduced graph: children (list of part dicts with 'mode' 0/2) contracted, except those in
        `split` (t2 children) which are deleted with their terminals kept apart.  ident: pair of vertices to identify."""
        rep = {v: v for v in rho_V}
        def find(a):
            while rep[a] != a:
                a = rep[a]
            return a
        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                rep[a] = b
        kept = set(rho_S)
        dropped_vertices = set()
        for ci, c in enumerate(children):
            kept -= c['S']
            if ci in split:
                ta, tb = sorted(set(c['term']))
                # internal vertices of the child disappear (no line of rho outside the child touches them)
                for v in c['V']:
                    if v not in (ta, tb):
                        dropped_vertices.add(v)
            else:
                vs = sorted(c['V'])
                for v in vs[1:]:
                    union(vs[0], v)
        if ident is not None:
            a, b = ident
            if find(a) == find(b):
                return mp.mpf(0)
            union(a, b)
        classes = sorted({find(v) for v in rho_V if v not in dropped_vertices})
        idx = {c: i for i, c in enumerate(classes)}
        lines = []
        for i in kept:
            a, b = self.edges[i]
            assert a not in dropped_vertices and b not in dropped_vertices
            lines.append((idx[find(a)], idx[find(b)], x[i]))
        return self.kirchhoff(len(classes), lines)

    # ---------- jets: dict frozenset(jet vars) -> mpf ----------
    @staticmethod
    def jmul(A, B):
        r = {}
        for ka, va in A.items():
            for kb, vb in B.items():
                if ka & kb:
                    continue
                k = ka | kb
                r[k] = r.get(k, 0) + va * vb
        return r

    @staticmethod
    def jpow(A, p):
        a0 = A[frozenset()]
        q = {k: v / a0 for k, v in A.items() if k}
        res = {frozenset(): mp.mpf(1)}
        qk = {frozenset(): mp.mpf(1)}
        coef = mp.mpf(1)
        nvars = len(frozenset().union(*A.keys())) if A else 0
        for j in range(1, nvars + 1):
            qk = Rbar.jmul(qk, q)
            if not qk:
                break
            coef *= (mp.mpf(p) - j + 1) / j
            for k, v in qk.items():
                res[k] = res.get(k, 0) + coef * v
        return {k: v * a0 ** p for k, v in res.items()}

    def node_jets(self, node, children, kids_eps, x, terminals):
        """U and W jets of a node; kids_eps[ci] = jet of eps for split-able children (t2 with distinct terminals)."""
        t2 = [ci for ci, c in enumerate(children) if c['deg'] == 2 and len(set(c['term'])) == 2]
        U, W = {}, {}
        for r in range(len(t2) + 1):
            for sp in itertools.combinations(t2, r):
                pre = {frozenset(): mp.mpf(1)}
                for ci in sp:
                    pre = self.jmul(pre, kids_eps[ci])
                if not pre:
                    continue
                u = self.reduced_poly(node['S'], node['V'], children, set(sp), x)
                for k, v in pre.items():
                    U[k] = U.get(k, 0) + v * u
                if terminals is not None:
                    # terminals mapped: a child's internal vertex never is a terminal of its parent; a contracted child
                    # containing a terminal is handled by the union-find inside reduced_poly
                    w = self.reduced_poly(node['S'], node['V'], children, set(sp), x, ident=terminals)
                    for k, v in pre.items():
                        W[k] = W.get(k, 0) + v * w
        return U, W

    def forest_value(self, parts, F, x):
        nodes = [parts[i] for i in F]
        root = dict(S=frozenset(range(self.l)), V=frozenset(range(self.V)), term=self.extv, deg=None)
        allnodes = nodes + [root]
        n = len(allnodes)
        parent = {}
        for a in range(n - 1):
            best = n - 1
            for b in range(n - 1):
                if b != a and allnodes[a]['S'] < allnodes[b]['S'] and len(allnodes[b]['S']) < len(allnodes[best]['S']):
                    best = b
            parent[a] = best
        order = sorted(range(n - 1), key=lambda a: len(allnodes[a]['S'])) + [n - 1]
        eps = {}
        total = {frozenset(): mp.mpf((-1) ** len(F))}
        for a in order:
            node = allnodes[a]
            ch = [b for b in range(n - 1) if parent[b] == a]
            children = [allnodes[b] for b in ch]
            kids_eps = {ci: eps[b] for ci, b in enumerate(ch) if b in eps}
            is_root = (a == n - 1)
            need_w = (self.deriv if is_root else (node['deg'] == 2 and len(set(node['term'])) == 2))
            terms = None
            if need_w:
                t = sorted(set(node['term']))
                terms = (t[0], t[1]) if len(t) == 2 else None
            U, W = self.node_jets(node, children, kids_eps, x, terms if need_w else None)
            if not U or U.get(frozenset(), 0) == 0:
                return {}
            fac = self.jpow(U, -1.5)
            if is_root and self.deriv:
                if terms is None:
                    return {}
                fac = self.jmul(fac, self.jmul(W, self.jpow(U, -1)))
            if not is_root and need_w:
                e = self.jmul(W, self.jpow(U, -1))
                bit = frozenset([a])
                eps[a] = {k | bit: v for k, v in e.items()}
            total = self.jmul(total, fac)
        return total

    def value(self, lx, per_dot=False):
        lx = [mp.mpf(v) for v in lx]
        x = [mp.e ** v for v in lx]
        acc = mp.mpf(0)
        sabs = mp.mpf(0)
        self.nforests = 0
        for (dts, mult) in self.dots:
            parts = []
            for c in self.cand:
                k = sum(1 for d in dts if d in c['S'])
                if self.kind in ('int4', 'int2d'):
                    deg = 0
                elif c['legs'] == 4 and k == 0:
                    deg = 0
                elif c['legs'] == 2 and k == 0:
                    deg = 2
                elif c['legs'] == 2 and k == 1:
                    deg = 0
                else:
                    continue
                d = dict(c); d['deg'] = deg
                parts.append(d)
            np_ = len(parts)
            def compat(p, q):
                return p['S'] <= q['S'] or q['S'] <= p['S'] or not (p['V'] & q['V'])
            forests = []
            def rec(i, cur):
                if i == np_:
                    forests.append(list(cur)); return
                rec(i + 1, cur)
                if all(compat(parts[i], parts[j]) for j in cur):
                    cur.append(i); rec(i + 1, cur); cur.pop()
            rec(0, [])
            xd = mp.mpf(mult)
            for d in dts:
                xd *= x[d]
            for F in forests:
                self.nforests += 1
                t = self.forest_value(parts, F, x)
                for v in t.values():
                    acc += xd * v
                    sabs += abs(xd * v)
        sx = sum(x)
        meas = mp.e ** (sum(lx)) * sx ** (-self.pw)
        return acc * meas, sabs * meas


def from_nickel(nk):
    """(V, ext, edges) of a Nickel index, lines in the order rebar's parse_nickel gives them"""
    parts = nk.strip().split('|')
    if parts and parts[-1] == '':
        parts.pop()
    V = len(parts)
    ext, edges = [0] * V, []
    for i, p in enumerate(parts):
        for ch in p:
            if ch == 'e':
                ext[i] += 1
            else:
                j = int(ch, 36)
                if not 0 <= j < V:
                    raise ValueError('bad Nickel index %r' % nk)
                edges.append((i, j))
    return V, ext, edges


def point(text):
    """log-coordinates as exact binary doubles: the same numbers rebar reads (no decimal conversion noise)"""
    return [mp.mpf(float(z)) for z in text.split()]


def read_sample(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            f = line.rstrip('\n').split('\t')
            rows.append({'kind': f[0], 'L': int(f[1]), 'nickel': f[2], 'type': f[3], 'lx': f[4], 'value': f[5],
                         'sabs': f[6], 'forests': int(f[7])})
    return rows


def verify(path, max_loops, digits):
    """recompute the stored rows with at most max_loops loops; |difference| <= 10^-digits |stored| + 1e-80 sum|terms|"""
    worst, n, bad = 0.0, 0, 0
    cache = {}
    for r in read_sample(path):
        if r['L'] > max_loops:
            continue
        key = (r['kind'], r['nickel'])
        if key not in cache:
            cache[key] = Rbar(*from_nickel(r['nickel']), r['kind'])
        v, s = cache[key].value(point(r['lx']))
        stored = mp.mpf(r['value'])
        d = abs(v - stored)
        worst = max(worst, float(d / abs(stored)) if stored else 0.0)
        n += 1
        if d > mp.mpf(10) ** -digits * abs(stored) + mp.mpf(10) ** -80 * s:
            bad += 1
            print('MISMATCH %s %s %s: %s, stored %s' % (r['kind'], r['nickel'], r['type'], mp.nstr(v, 40), r['value']))
    print('verify: %d stored values with L <= %d recomputed in %d-digit arithmetic, %d differ by more than 1e-%d relative '
          '(+ 1e-80 sum|terms|); largest relative difference %.1e' % (n, max_loops, mp.mp.dps, bad, digits, worst))
    return 1 if bad or not n else 0


def main(argv):
    if len(argv) >= 3 and argv[0] == 'eval':
        kind, nk = argv[1], argv[2]
        R = Rbar(*from_nickel(nk), kind)
        for line in sys.stdin:
            if len(line.split()) != len(R.edges):
                continue
            v, s = R.value(point(line))
            print(mp.nstr(v, 50), mp.nstr(s, 10), R.nforests)
        return 0
    if len(argv) >= 2 and argv[0] == 'verify':
        max_loops, digits = 99, 45
        it = iter(argv[2:])
        for a in it:
            if a == '--max-loops':
                max_loops = int(next(it))
            elif a == '--digits':
                digits = int(next(it))
        return verify(argv[1], max_loops, digits)
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
