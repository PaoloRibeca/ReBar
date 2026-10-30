#!/usr/bin/env python3
# sumrule_check.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# (1) zero-dimensional check of completeness: sum_G S_G over a whole 2-point list (self-loops included)
#     vs the exact coefficient of the 0-d 1PI 2-point function; (2) Gamma(2,1) O(N) sum rule with GAU's
#     phi^2 marking rule (no self-loop: any line; one self-loop: that line; >=2: none), vs thesis p. 107.
#
# usage: sumrule_check.py DIR TAG Lmax_ON  (DIR holds GAU-format L-loop-2 files)
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
from collections import Counter
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gg = load('gaugraph', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaugraph.py'))
sr = load('sumrules', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sumrules.py'))
usage = 'usage: sumrule_check.py DIR TAG Lmax_ON  (DIR holds GAU-format L-loop-2 files)'
if len(sys.argv) < 4: sys.exit(usage)
d, tag, lmax_on = sys.argv[1], sys.argv[2], int(sys.argv[3])
# exact 0-d: <x^2> with weight exp(-x^2/2 - g x^4/24); Gamma2 = 1/<x^2> = 1 - Sigma
K = 12  # through L = 11
def dfact(n):  # (n)!! for odd n, (-1)!!=1
    r = 1
    while n > 1: r *= n; n -= 2
    return r
Z = [Fraction((-1)**k * dfact(4*k-1), 24**k * factorial(k)) for k in range(K)]
X = [Fraction((-1)**k * dfact(4*k+1), 24**k * factorial(k)) for k in range(K)]
def div(a, b):
    q = []
    for n in range(K):
        s = a[n] - sum(q[i]*b[n-i] for i in range(n)); q.append(s / b[0])
    return q
G2 = div(X, Z); Gam = div([Fraction(1)] + [Fraction(0)]*(K-1), G2)  # 1/<x^2>
thesis21 = {5: '(100640N^0 + 97872N^1 + 30536N^2 + 3676N^3 + 150N^4 + 1N^5)/7776',
            6: '(2706816N^0 + 2861760N^1 + 1037888N^2 + 162848N^3 + 11016N^4 + 262N^5 + 1N^6)/46656'}
for L in range(1, K):
    p = os.path.join(d, '%d-loop-2' % L)
    if not os.path.exists(p): continue
    with open(p) as fh:
        nks = [gg.parse(l.strip()).nickel() for l in fh if l.strip()]
    tot = Fraction(0); on = Counter(); n21 = 0
    for nk in nks:
        V, ext, edges = sr.parse(nk)
        S = Fraction(factorial(sum(ext)), sr.aut_order(V, ext, edges))
        tot += S
        if L <= lmax_on:
            s = sum(1 for (u, v) in edges if u == v)
            marks = len(edges) if s == 0 else (1 if s == 1 else 0)
            if marks:
                n21 += 1
                for k, c in sr.on_factor(V, ext, edges).items(): on[k] += S*c*marks
    exact = abs(Gam[L])  # coefficient of g^L in 1/<x^2>
    msg = '%s L=%d graphs %d: sum S_G = %s, |0-d coefficient| = %s, ratio %s' % (tag, L, len(nks), tot, exact, tot/exact)
    if L <= lmax_on:
        f = sr.fmt(on); msg += ' | Gamma(2,1): %d contributing graphs, sum S*T*marks = %s' % (n21, f)
        if L in thesis21: msg += ' | thesis p.107 g^%d: %s -> %s' % (L, thesis21[L], 'EQUAL' if f == thesis21[L] else 'DIFFERENT')
    print(msg); sys.stdout.flush()
