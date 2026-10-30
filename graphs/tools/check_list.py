#!/usr/bin/env python3
# check_list.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Prints, for each GAU-format list, the number of graphs and of distinct canonical forms, whether all
# are 1PI, and the distributions of loops, self-loops and legs.
#
# usage: check_list.py LIST...
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
from collections import Counter
spec = importlib.util.spec_from_file_location('gaugraph', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaugraph.py'))
gg = importlib.util.module_from_spec(spec); spec.loader.exec_module(gg)
usage = 'usage: check_list.py LIST...  (GAU-format list files)'
if len(sys.argv) < 2: sys.exit(usage)
for p in sys.argv[1:]:
    with open(p) as fh:
        gs = [gg.parse(l.strip()) for l in fh if l.strip()]
    nk = [g.nickel() for g in gs]
    print('%s: %d graphs, %d distinct, all 1PI %s, loops %s, self-loops %s, legs %s' % (p.split('/')[-1], len(gs), len(set(nk)), all(g.is_1pi() for g in gs),
          dict(Counter(g.loops() for g in gs)), dict(Counter(g.selfloops() for g in gs)), dict(Counter(tuple(sorted(x for x in g.legs() if x)) for g in gs))))
