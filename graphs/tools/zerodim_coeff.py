#!/usr/bin/env python3
# zerodim_coeff.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# The exact zero-dimensional sum rules of the graph lists: for every loop order L the absolute value of the
# coefficient of g^L in the 1PI two-point function 1/<x^2> and in the 1PI four-point function -W_4/<x^2>^4 of the
# zero-dimensional phi^4 model with action x^2/2 + g x^4/24, which the sum of E!/|Aut G| over the 1PI graphs with
# E legs and L loops (self-loops included; topologies -w) must equal (g^L has L loops with two legs, L - 1 with four).
#   python3 -I zerodim_coeff.py [Lmax (default 12)]
# Standard library only.  The two-point coefficients are those of tools/sumrule_check.py.
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

import sys
from fractions import Fraction
from math import factorial

Lmax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
K = Lmax + 2   # series length: g^(L+1) is needed for four legs


def dfact(n):   # n!! for odd n, (-1)!! = 1
    r = 1
    while n > 1:
        r *= n
        n -= 2
    return r


def moments(p):   # the series in g of the integral of x^p exp(-x^2/2 - g x^4/24), up to normalisation
    return [Fraction((-1)**k * dfact(4*k + p - 1), 24**k * factorial(k)) for k in range(K)]


def mul(a, b):
    return [sum(a[i] * b[n - i] for i in range(n + 1)) for n in range(K)]


def div(a, b):
    q = []
    for n in range(K):
        q.append((a[n] - sum(q[i] * b[n - i] for i in range(n))) / b[0])
    return q


Z, X2, X4 = moments(0), moments(2), moments(4)
W2 = div(X2, Z)                                   # <x^2>
W4 = [a - 3 * b for a, b in zip(div(X4, Z), mul(W2, W2))]   # <x^4> - 3 <x^2>^2
one = [Fraction(1)] + [Fraction(0)] * (K - 1)
G2 = div(one, W2)                                 # 1PI two-point function
G4 = [-x for x in div(W4, mul(mul(W2, W2), mul(W2, W2)))]   # 1PI four-point function
for L in range(1, Lmax + 1):   # L loops: g^L for two legs (V = L), g^(L+1) for four (V = L + 1)
    print('L=%d\tE=2: %s\tE=4: %s' % (L, abs(G2[L]), abs(G4[L + 1])))
