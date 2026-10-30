#!/usr/bin/env python3
# gau_order.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Puts a two-point list of compact strings in any order back into the order of the stored lists, i.e. byte for byte
# what GAU's gamma2 wrote with the 2003 generator and what topologies -G writes: the degree lists in GAU's order
# (graphs/scripts/gau_lists.sh 2 L: "2,4,..." first, the graphs with a vertex carrying both legs, then "3,3,4,..."),
# and within each the reverse of the 2003 generator's
# std::set<topology> order (graphs/topologies/generators.hpp, operator< of topology and of smartPermutation):
# self-loop counts per vertex lexicographically ascending, then the upper triangle of the canonical line matrix
# lexicographically descending. Checked on every two-point list of graphs/lists (sorted, then reordered:
# identical to the stored list).
#
# usage: gau_order.py LIST > ORDERED      (Python >= 3.6, standard library only)
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
import re, sys

TOKEN = re.compile(r"\s*(?:\.\s*(\d+)|(\d+)(?:\s*\^\s*(\d+))?)")   # '.k' | 'v' | 'v^k', blanks between integers


def rle(text):
    """'2.7 1^2.3' -> [2, 0 x 7, 1, 1, 0 x 3] (topologies' run-length code: '.k' k zeros, 'v^k' k copies of v)"""
    out, pos, text = [], 0, text.rstrip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError("bad run-length code %r" % text)
        pos = m.end()
        if m.group(1):
            out += [0] * int(m.group(1))
        else:
            out += [int(m.group(2))] * int(m.group(3) or 1)
    return out


def decode(s):
    """compact string -> (V, self-loops per vertex, upper triangle row by row)"""
    m = re.match(r"^(\d+)\((?:\[([^\]]*)\])?([^)]*)\)$", s)
    if not m:
        raise ValueError("not a topologies compact string: %r" % s)
    V = int(m.group(1))
    tad = rle(m.group(2)) if m.group(2) is not None else [0] * V
    tri = rle(m.group(3)) if m.group(3) else []
    if len(tad) != V or len(tri) != V * (V - 1) // 2:
        raise ValueError("inconsistent lengths in %r" % s)
    return V, tad, tri


def degrees(V, tad, tri):
    d = [2 * t for t in tad]
    k = 0
    for i in range(V):
        for j in range(i + 1, V):
            d[i] += tri[k]; d[j] += tri[k]; k += 1
    return d


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: gau_order.py LIST > ORDERED")
    with open(sys.argv[1]) as fh:
        strings = [l.strip() for l in fh if l.strip()]
    groups = ([], [])
    for s in strings:
        V, tad, tri = decode(s)
        two = V == 1 or 2 in degrees(V, tad, tri)
        # descending tri, ascending tad: negate the triangle
        groups[0 if two else 1].append(((V, tad, [-x for x in tri]), s))
    for g in groups:
        g.sort()
        sys.stdout.write("".join(s + "\n" for _, s in g))


if __name__ == "__main__":
    main()
