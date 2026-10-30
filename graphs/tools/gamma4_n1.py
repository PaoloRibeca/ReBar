#!/usr/bin/env python3
# gamma4_n1.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Checks 4-point lists: the sum of S_G = 4!/|Aut G| over each L-loop list equals the 2003
# zero-dimensional gamma4ZeroI coefficient at N = 1 (the thesis's 2003 output), transcribed below.
#
# usage: gamma4_n1.py L-loop-4...  (GAU-format 4-point lists, named <L>-loop-4)
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
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
gg = load('gaugraph', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaugraph.py'))
sr = load('sumrules', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sumrules.py'))
# the 2003 gamma4ZeroI, |coefficient of gI^(L+1)| at n=1
c = {1: [8, 1, 6], 2: [108, 26, 1, 36], 3: [2048, 724, 62, 1, 216], 4: [47648, 21632, 2928, 124, 1, 1296],
     5: [1297664, 702864, 127392, 9044, 220, 1, 7776], 6: [40362304, 24950944, 5549104, 546408, 23404, 358, 1, 46656],
     7: [1410726400, 966932160, 250483648, 31004768, 1898240, 53312, 546, 1, 279936],
     8: [54747705344, 40784744192, 11897634304, 1746112576, 137234272, 5654416, 110208, 792, 1, 1679616]}
usage = 'usage: gamma4_n1.py L-loop-4...  (GAU-format 4-point lists, named <L>-loop-4)'
if len(sys.argv) < 2: sys.exit(usage)
for p in sys.argv[1:]:
    L = int(os.path.basename(p).split('-')[0]); tot = Fraction(0); n = 0
    with open(p) as fh:
        for l in fh:
            if not l.strip(): continue
            V, ext, edges = sr.parse(gg.parse(l.strip()).nickel()); n += 1
            tot += Fraction(factorial(sum(ext)), sr.aut_order(V, ext, edges))
    ref = Fraction(sum(c[L][:-1]), c[L][-1])
    print('%s: %d graphs, sum S_G = %s, 2003 zero-dim gamma4ZeroI(n=1) = %s -> %s' % (os.path.basename(p), n, tot, ref, 'EQUAL' if tot == ref else 'DIFFERENT')); sys.stdout.flush()
