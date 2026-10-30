#!/usr/bin/env python3
# zerodim_tf.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Zero-dimensional N=1 check of self-loop-free 1PI 2-point lists at any loop order.
# Exact 0-d: G(m0^2, lam) = <x^2> for exp(-m0^2 x^2/2 - lam x^4/24) = g(lam/m0^4)/m0^2.
# Removing every bare self-loop is the mass shift of the intermediate scheme restricted to the bare one-loop tadpole,
# M^2 = m0^2 + lam/(2 M^2) (N=1; thesis eq. (2.8) tadpole term), so with M=1:
#   m0^2 = 1 - lam/2,  G = g(lam/m0^4)/m0^2,  Sigma_tf(lam) = 1 - 1/G  (sum over self-loop-free 1PI graphs).
# Compare |[lam^L] Sigma_tf| with sum_G S_G, S_G = E!/|Aut G|, over a GAU-format list (self-loop-free rows only).
#
# usage: zerodim_tf.py L-loop-2...  (GAU-format 2-point lists, named <L>-loop-2[-ci])
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
import sys, importlib.util, os
from fractions import Fraction
from math import factorial
K = 11
def dfact(n):
    r = 1
    while n > 1: r *= n; n -= 2
    return r
def mul(a, b): return [sum(a[i]*b[n-i] for i in range(n+1)) for n in range(K)]
def inv(a):
    q = []
    for n in range(K): q.append(((1 if n == 0 else 0) - sum(q[i]*a[n-i] for i in range(n))) / a[0])
    return q
def powr(a, k):  # a^k, integer k (may be negative)
    r = [Fraction(1)] + [Fraction(0)]*(K-1); b = a if k >= 0 else inv(a)
    for _ in range(abs(k)): r = mul(r, b)
    return r
Z = [Fraction((-1)**k * dfact(4*k-1), 24**k * factorial(k)) for k in range(K)]
X = [Fraction((-1)**k * dfact(4*k+1), 24**k * factorial(k)) for k in range(K)]
g = mul(X, inv(Z))   # <x^2> at m0=1 as series in u = lam
G = [Fraction(1)] + [Fraction(0)]*(K-1)
lam = [Fraction(0), Fraction(1)] + [Fraction(0)]*(K-2)
for it in range(K+1):
    m2 = [Fraction(1) - (x/2) for x in [Fraction(0)] + [0]*(K-1)]
    m2 = [Fraction(1), Fraction(-1, 2)] + [Fraction(0)]*(K-2)
    acc = [Fraction(0)]*K
    for k in range(K):
        term = mul(powr(lam, k), powr(m2, -2*k-1)); acc = [acc[n] + g[k]*term[n] for n in range(K)]
    G = acc
sig = [(1 if n == 0 else 0) - x for n, x in enumerate(inv(G))]
spec = importlib.util.spec_from_file_location('gaugraph', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaugraph.py'))
gg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gg)
spec2 = importlib.util.spec_from_file_location('sumrules', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sumrules.py'))
sr = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(sr)
usage = 'usage: zerodim_tf.py L-loop-2...  (GAU-format 2-point lists, named <L>-loop-2[-ci])'
if len(sys.argv) < 2: sys.exit(usage)
for p in sys.argv[1:]:
    L = int(os.path.basename(p).split('-')[0]); tot = Fraction(0); n = 0
    with open(p) as fh:
        for l in fh:
            if not l.strip(): continue
            V, ext, edges = sr.parse(gg.parse(l.strip()).nickel())
            if any(u == v for (u, v) in edges): continue
            tot += Fraction(factorial(sum(ext)), sr.aut_order(V, ext, edges)); n += 1
    print('%s: %d self-loop-free graphs, sum S_G = %s, |0-d tadpole-free coefficient| = %s -> %s' % (p.split('/')[-1], n, tot, abs(sig[L]), 'EQUAL' if tot == abs(sig[L]) else 'DIFFERENT'))
    sys.stdout.flush()
