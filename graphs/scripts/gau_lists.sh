#!/bin/bash
# gau_lists.sh -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Prints the vertex-degree lists for which GAU's diagramList(ext, L) calls topologies, in GAU's order
# (GAU/stuff.C:643-700 diagramList, PARIS/library/partition.C strictPartition; gamma4's initial ++list4
# skips the disconnected "0,4,..." list, stuff.C:4706-4708). A degree d < 4 marks a vertex with 4-d legs.
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
#
# usage: gau_lists.sh 2|4 L
[ $# = 2 ] && { [ "$1" = 2 ] || [ "$1" = 4 ]; } || { echo 'usage: gau_lists.sh 2|4 L' >&2; exit 2; }
ext=$1; L=$2
fours() { local n=$1 s=""; for ((i=0;i<n;i++)); do s="$s,4"; done; echo "$s"; }
if [ "$ext" = 2 ]; then
  V=$L
  echo "2$(fours $((V-1)))"
  if [ $V -ge 2 ]; then echo "3,3$(fours $((V-2)))"; fi
else
  V=$((L+1))
  echo "1,3$(fours $((V-2)))"
  echo "2,2$(fours $((V-2)))"
  if [ $V -ge 3 ]; then echo "2,3,3$(fours $((V-3)))"; fi
  if [ $V -ge 4 ]; then echo "3,3,3,3$(fours $((V-4)))"; fi
fi
exit 0
