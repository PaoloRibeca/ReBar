#!/usr/bin/env python3
"""Exact zero-dimensional O(N) phi^4 sum rules for 1PI graph lists (standard library only).

Zero-dimensional O(N) model: Z = int d^N x exp(-m0^2 x^2/2 - g (x^2)^2/24), with x^2 = sum_i x_i^2.  Then
    G = <x_1^2> = <r^2>/N,   A = <r^4>/(N(N+2))   (<x_i x_j x_k x_l> = A (d_ij d_kl + d_ik d_jl + d_il d_jk)),
    Gamma2 = 1/G,            Gamma4 = -3 (A - G^2)/G^4     (Gamma^(4)_ijkl = Gamma4 * S_ijkl, tree level g),
and, by the radial integral, <r^(2k)> = I(k)/I(0) with
    I(k) = sum_n (-g/24)^n/n! * m0^(-2(k+2n)) * P_(k+2n)(N),   P_j(N) = N (N+2) ... (N+2j-2).
Every coefficient of g is a polynomial in N.

Schemes (unit mass):
  bare (B)  m0^2 = 1.  Gamma2 = 1 + sum_{L>=1} (-1)^(L+1) c2B_L g^L,  Gamma4 = sum_{L>=0} (-1)^L c4B_L g^(L+1)
            c_L = sum over ALL 1PI graphs with L loops, self-loops included, of S_G T_G(N).
  tadpole-free (I)  m0^2 = 1 - g (N+2)/6: the mass shift cancels the bare one-loop tadpole, so every graph with
            a self-loop cancels against a counterterm graph, and
            Gamma2 = 1 + sum_{L>=2} (-1)^(L+1) c2_L g^L,  Gamma4 = sum_{L>=0} (-1)^L c4_L g^(L+1)
            with c_L = sum over the SELF-LOOP-FREE 1PI graphs (one-vertex-reducible ones included) of S_G T_G(N).
Here S_G = E!/|Aut G| and T_G(N) = 3^-V sum over the vertex pairings of N^(closed index loops): the weights of
src/graph.cpp (sym_factor, on_factor).  The thesis's 2003 output soloSumRules (not distributed) holds the same series
(gamma2ZeroB/I to g^8, gamma4ZeroB/I to g^9, mmB$5 = bare mass, gI = intermediate coupling); this script derives them
independently, compares them with such a file (--res FILE), and extends them.

usage: zerodim_on.py [--order K] [--res FILE]     prints the c_L(N) polynomials and the comparison with FILE
As a module: series(K) -> {'c2B','c4B','c2','c4'}: dict L -> list of Fractions (coefficients of N^0, N^1, ...).
"""
import re
import sys
from fractions import Fraction
from math import factorial

# ---- polynomials in N: lists of Fractions, index = power
def padd(a, b):
    r = [Fraction(0)] * max(len(a), len(b))
    for i, x in enumerate(a): r[i] += x
    for i, x in enumerate(b): r[i] += x
    return ptrim(r)
def pmul(a, b):
    if not a or not b: return []
    r = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b): r[i + j] += x * y
    return ptrim(r)
def pscale(a, c): return ptrim([x * c for x in a])
def ptrim(a):
    a = list(a)
    while a and a[-1] == 0: a.pop()
    return a
def pdivexact(a, b):
    """a / b for polynomials, exact division required."""
    a = ptrim(a); b = ptrim(b); q = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    while len(a) >= len(b) and a:
        c = a[-1] / b[-1]; k = len(a) - len(b); q[k] = c
        a = ptrim(padd(a, pscale([Fraction(0)] * k + b, -c)))
    if a: raise ValueError('inexact polynomial division')
    return ptrim(q)

# ---- power series in g with polynomial coefficients, truncated at order K (exclusive)
def sadd(a, b): return [padd(x, y) for x, y in zip(a, b)]
def smul(a, b, K):
    r = [[] for _ in range(K)]
    for i in range(K):
        if not a[i]: continue
        for j in range(K - i):
            if b[j]: r[i + j] = padd(r[i + j], pmul(a[i], b[j]))
    return r
def sinv(a, K):
    """1/a; a[0] must be a nonzero constant polynomial."""
    if len(a[0]) != 1: raise ValueError('leading coefficient must be a constant')
    a0 = a[0][0]; q = [[] for _ in range(K)]; q[0] = [1 / a0]
    for n in range(1, K):
        s = []
        for i in range(n):
            if q[i] and a[n - i]: s = padd(s, pmul(q[i], a[n - i]))
        q[n] = pscale(s, -1 / a0)
    return q
def spow(a, k, K):
    r = [[Fraction(1)]] + [[] for _ in range(K - 1)]
    b = a if k >= 0 else sinv(a, K)
    for _ in range(abs(k)): r = smul(r, b, K)
    return r

def P(j):
    """N (N+2) ... (N+2j-2) as a polynomial in N."""
    r = [Fraction(1)]
    for i in range(j): r = pmul(r, [Fraction(2 * i), Fraction(1)])
    return r

def moments(K, m2):
    """I(0), I(1), I(2) as series in g (m2 = m0^2 as a series)."""
    out = []
    for k in range(3):
        I = [[] for _ in range(K)]
        for n in range(K):
            mpow = spow(m2, -(k + 2 * n), K)   # m0^(-2(k+2n))
            c = Fraction((-1) ** n, 24 ** n * factorial(n))
            Pj = pscale(P(k + 2 * n), c)
            for i in range(K - n):
                if mpow[i]: I[n + i] = padd(I[n + i], pmul(Pj, mpow[i]))
        out.append(I)
    return out

def gammas(K, tadpole_free):
    m2 = [[Fraction(1)]] + [[] for _ in range(K - 1)]
    if tadpole_free: m2[1] = [Fraction(-2, 6), Fraction(-1, 6)]   # -(N+2)/6
    I0, I1, I2 = moments(K, m2)
    inv0 = sinv(I0, K)
    r2 = smul(I1, inv0, K); r4 = smul(I2, inv0, K)
    G = [pdivexact(x, [Fraction(0), Fraction(1)]) if x else [] for x in r2]            # / N
    A = [pdivexact(x, [Fraction(0), Fraction(2), Fraction(1)]) if x else [] for x in r4]  # / (N(N+2))
    g2 = sinv(G, K)
    G2 = smul(G, G, K)
    num = [pscale(x, -3) for x in sadd(A, [pscale(y, -1) for y in G2])]
    g4 = smul(num, spow(G, -4, K), K)
    return g2, g4

def series(K):
    """Sum-rule coefficients c_L(N) (see module docstring); Gamma2 to g^(K-1), Gamma4 to g^(K-1)."""
    out = {}
    for tag, tf in (('B', False), ('I', True)):
        g2, g4 = gammas(K, tf)
        c2 = {L: pscale(g2[L], (-1) ** (L + 1)) for L in range(1, K)}
        c4 = {L: pscale(g4[L + 1], (-1) ** L) for L in range(0, K - 1)}
        assert g2[0] == [1] and g4[0] == [] and g4[1] == [1]
        if tf: assert not g2[1]
        out['c2' + ('B' if tag == 'B' else '')] = c2
        out['c4' + ('B' if tag == 'B' else '')] = c4
    return out

def fmt(p):
    """polynomial -> '(a0 + a1 N + ...)/den'"""
    if not p: return '0'
    den = 1
    for x in p:
        d = x.denominator; a, b = den, d
        while b: a, b = b, a % b
        den = den * d // a
    terms = []
    for k, x in enumerate(p):
        c = x * den
        if c == 0: continue
        terms.append('%d%s' % (c, '' if k == 0 else ' N' if k == 1 else ' N^%d' % k))
    return '(' + ' + '.join(terms) + ')/%d' % den

# ---- reader for the 2003 Mathematica output (+ - * / ^, parentheses, integers, symbols)
def parse_res(path):
    text = open(path).read()
    out = {}
    for m in re.finditer(r'RES\["ZeroDim",\s*"(\w+)"\]\s*=\s*(.*?)(?=RES\[|\Z)', text, re.S):
        out[m.group(1)] = _Parser(m.group(2)).parse()
    return out

class _Parser:
    """Expression -> {(power of the coupling, power of n): Fraction}; the bare mass mmB$5 is set to 1."""
    def __init__(self, s):
        self.t = re.findall(r'\d+|[A-Za-z][A-Za-z0-9$]*|[-+*/^()]', s); self.i = 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def take(self): self.i += 1; return self.t[self.i - 1]
    def parse(self):
        v = self.expr()
        if self.peek() is not None: raise ValueError('trailing tokens %r' % self.t[self.i:])
        return v
    def expr(self):
        v = self.term()
        while self.peek() in ('+', '-'):
            op = self.take(); w = self.term()
            v = _padd2(v, w if op == '+' else {k: -x for k, x in w.items()})
        return v
    def term(self):
        v = self.factor()
        while self.peek() in ('*', '/'):
            op = self.take(); w = self.factor()
            if op == '*': v = _pmul2(v, w)
            else:
                if set(w) != {(0, 0)}: raise ValueError('division by a non-constant')
                v = {k: x / w[(0, 0)] for k, x in v.items()}
        return v
    def factor(self):
        if self.peek() == '-': self.take(); return {k: -x for k, x in self.factor().items()}
        if self.peek() == '+': self.take(); return self.factor()
        v = self.atom()
        if self.peek() == '^':
            self.take(); e = int(self.take()); r = {(0, 0): Fraction(1)}
            for _ in range(e): r = _pmul2(r, v)
            v = r
        return v
    def atom(self):
        t = self.take()
        if t == '(':
            v = self.expr()
            if self.take() != ')': raise ValueError('missing )')
            return v
        if t.isdigit(): return {(0, 0): Fraction(int(t))}
        if t in ('gI', 'lambdaB$5'): return {(1, 0): Fraction(1)}
        if t == 'n': return {(0, 1): Fraction(1)}
        if t == 'mmB$5': return {(0, 0): Fraction(1)}
        raise ValueError('unknown symbol %r' % t)

def _padd2(a, b):
    r = dict(a)
    for k, x in b.items(): r[k] = r.get(k, 0) + x
    return {k: x for k, x in r.items() if x}
def _pmul2(a, b):
    r = {}
    for (i, j), x in a.items():
        for (k, l), y in b.items(): r[(i + k, j + l)] = r.get((i + k, j + l), 0) + x * y
    return {k: x for k, x in r.items() if x}

def res_coeff(expr, power):
    """coefficient of coupling^power as a polynomial in n"""
    d = {j: x for (i, j), x in expr.items() if i == power}
    return ptrim([d.get(j, Fraction(0)) for j in range(max(d) + 1)]) if d else []

def compare_res(path, S):
    """Compare the derived Gamma2/Gamma4 series with the 2003 file, coefficient by coefficient."""
    res = parse_res(path); lines = []; ok = True
    plan = (('gamma2ZeroB', 'c2B', lambda L: L, lambda L: (-1) ** (L + 1), 1),
            ('gamma2ZeroI', 'c2', lambda L: L, lambda L: (-1) ** (L + 1), 1),
            ('gamma4ZeroB', 'c4B', lambda L: L + 1, lambda L: (-1) ** L, 0),
            ('gamma4ZeroI', 'c4', lambda L: L + 1, lambda L: (-1) ** L, 0))
    for name, key, gp, sgn, L0 in plan:
        if name not in res: lines.append('%s: not in file' % name); ok = False; continue
        e = res[name]; top = max(i for i, _ in e)
        Ls = [L for L in sorted(S[key]) if L >= L0 and gp(L) <= top]
        bad = [L for L in Ls if pscale(res_coeff(e, gp(L)), sgn(L)) != S[key][L]]
        const_ok = res_coeff(e, 0) == ([Fraction(1)] if name.startswith('gamma2') else [])
        ok = ok and not bad and const_ok
        lines.append('%s: coefficients of g^%d..g^%d (L = %d..%d) %s%s' % (name, gp(Ls[0]), gp(Ls[-1]), Ls[0], Ls[-1],
                     'all EQUAL to the derived series' if not bad else 'DIFFERENT at L = %s' % bad,
                     '' if const_ok else '; constant term DIFFERENT'))
    return ok, lines

def main(argv):
    K = 11; res = None
    i = 0
    while i < len(argv):
        if argv[i] == '--order': K = int(argv[i + 1]) + 1; i += 2
        elif argv[i] == '--res': res = argv[i + 1]; i += 2
        else: raise SystemExit(__doc__)
    S = series(K)
    names = {'c2B': 'Gamma2, all 1PI graphs (self-loops included), bare',
             'c2': 'Gamma2, self-loop-free 1PI graphs (tadpole-free scheme)',
             'c4B': 'Gamma4, all 1PI graphs (self-loops included), bare',
             'c4': 'Gamma4, self-loop-free 1PI graphs (tadpole-free scheme)'}
    for key in ('c2', 'c4', 'c2B', 'c4B'):
        print('# %s: sum_G S_G T_G(N) at L loops' % names[key])
        for L in sorted(S[key]):
            p = S[key][L]
            print('%s\tL=%d\t%s\tN=1: %s' % (key, L, fmt(p), sum(p)))
    if res:
        ok, lines = compare_res(res, S)
        print('# comparison with %s' % res)
        for l in lines: print(l)
        print('-> %s' % ('ALL EQUAL' if ok else 'DIFFERENCES'))
        return 0 if ok else 1
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
