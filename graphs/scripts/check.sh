#!/bin/bash
# check.sh -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Checks <topologies binary> against the stored lists and the exact sum rules. It regenerates the phi^4 lists
# up to <V2> vertices (two-point) and <V4> vertices (four-point) into <dir>/gen, with the integrator's lists,
# and compares every list that ../lists holds with it as a set of graphs (the canonical keys of -r); it
# regenerates them with -G up to <VG> vertices into <dir>/G and compares the self-loop-free ones byte for byte
# (the stored lists are in the labelling and the order of the 2003 generator, which -G reproduces for the
# graphs without self-loops); every complete list (<L>-loop-2, <L>-loop-4-cit) must satisfy its
# zero-dimensional sum rule (-w, against the exact coefficients the program carries); the integrator's lists
# are compared with ../../integrator/data/lists, byte for byte where the format is ours and by Nickel index
# where they are Phi4tools' files. Then the Python checks of ../tools run on the lists through 7 loops
# (sumrule_check.py: 2-point sum rules and the Gamma(2,1) O(N) sum rule; gamma4_n1.py: 4-point sum rules at
# N = 1; zerodim_tf.py: self-loop-free 2-point lists), and tools/bruteforce.py compares the general mode with
# a brute-force enumeration on small degree lists.
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
# usage: check.sh <topologies binary> <dir> <V2> <V4> [threads (default 2)] [VG (default 8)]
set -euo pipefail
usage() { echo 'usage: check.sh <topologies binary> <dir> <V2> <V4> [threads (default 2)] [VG (default 8)]'; }
[ $# -ge 4 ] && [ -x "$1" ] || { usage >&2; exit 2; }
here=$(cd "$(dirname "$0")/.." && pwd); bin=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); out=$2
V2=$3; V4=$4; threads=${5:-2}; VG=${6:-8}; py=${PYTHON:-python3}; gau=$here/lists
ilists=$here/../integrator/data/lists; fail=0
export OMP_NUM_THREADS=$threads
keys() { "$bin" -r < "$1" | LC_ALL=C sort; }
report() { printf '%-16s %8s graphs  %s\n' "$1" "$2" "$3"; }
rm -rf "$out"; mkdir -p "$out/gen" "$out/G"
echo "# phi^4 lists: -E 2 -V $V2 and -E 4 -V $V4, $threads threads"
t0=$SECONDS
"$bin" -E 2 -V "$V2" -N -w -o "$out/gen" > "$out/gen/log-2"
"$bin" -E 4 -V "$V4" -N -w -o "$out/gen" > "$out/gen/log-4"
echo "# generated in $((SECONDS - t0)) s; comparison with lists as sets of graphs, sum rules"
for f in "$out"/gen/*-loop-*; do
  n=$(basename "$f"); r=""
  if [ -f "$gau/$n" ]; then
    if cmp -s <(keys "$f") <(keys "$gau/$n"); then r="same set as the stored list"
    else r="DIFFERENT from the stored list ($(grep -c . "$gau/$n") graphs)"; fail=1; fi
  fi
  case "$n" in
    *-loop-2|*-loop-4-cit)
      w=$("$bin" -w < "$f")
      if [[ "$w" == *"the zero-dimensional coefficient"* ]]; then r="$r${r:+, }sum rule EXACT"
      else r="$r${r:+, }sum rule WRONG: $w"; fail=1; fi;;
  esac
  report "$n" "$(grep -c . "$f")" "$r"
done
echo "# -G up to $VG vertices: the self-loop-free lists byte for byte"
"$bin" -E 2 -V $((V2 < VG ? V2 : VG)) -G -o "$out/G" > "$out/G/log-2"
"$bin" -E 4 -V $((V4 < VG ? V4 : VG)) -G -o "$out/G" > "$out/G/log-4"
for f in "$out"/G/*-loop-4 "$out"/G/*-loop-2-ci; do
  n=$(basename "$f"); [ -f "$gau/$n" ] || continue
  if cmp -s "$f" "$gau/$n"; then report "$n" "$(grep -c . "$f")" "byte for byte the stored list"
  else report "$n" "$(grep -c . "$f")" "DIFFERENT from the stored list byte for byte"; fail=1; fi
done
echo "# the integrator's lists against integrator/data/lists"
for f in "$out"/gen/gamma*.txt; do
  n=$(basename "$f"); [ -f "$ilists/$n" ] || continue
  case "$n" in
    gamma4_L6.txt|gamma4_L7.txt|gamma2_L6.txt|gamma2_L7.txt) # Phi4tools' own files: the Nickel column
      if cmp -s <(cut -f2 "$f" | LC_ALL=C sort) <(cut -f2 "$ilists/$n" | LC_ALL=C sort); then r="same Nickel indices as the stored list"
      else r="DIFFERENT Nickel indices from the stored list"; fail=1; fi;;
    *)
      if cmp -s "$f" "$ilists/$n"; then r="byte for byte the stored list"
      else r="DIFFERENT from the stored list byte for byte"; fail=1; fi;;
  esac
  report "$n" "$(grep -c . "$f")" "$r"
done
echo "# Python checks on the lists through 7 loops"
mkdir -p "$out/py"; two=(); four=(); ci=()
for L in 1 2 3 4 5 6 7; do
  [ -f "$out/gen/$L-loop-2" ] && { cp "$out/gen/$L-loop-2" "$out/py/"; two+=("$out/py/$L-loop-2"); }
  [ -f "$out/gen/$L-loop-2-ci" ] && { cp "$out/gen/$L-loop-2-ci" "$out/py/$L-loop-2-ci"; ci+=("$out/py/$L-loop-2-ci"); }
  [ -f "$out/gen/$L-loop-4" ] && { cp "$out/gen/$L-loop-4" "$out/py/"; four+=("$out/py/$L-loop-4"); }
done
log=$out/zerodim.txt; : > "$log"
if [ ${#two[@]} -gt 0 ]; then
  "$py" -I "$here/tools/sumrule_check.py" "$out/py" regenerated 6 | tee -a "$log" || { echo "check: sumrule_check.py FAILED"; exit 1; }
fi
if [ ${#four[@]} -gt 0 ]; then
  "$py" -I "$here/tools/gamma4_n1.py" "${four[@]}" | tee -a "$log" || { echo "check: gamma4_n1.py FAILED"; exit 1; }
fi
if [ ${#ci[@]} -gt 0 ]; then
  "$py" -I "$here/tools/zerodim_tf.py" "${ci[@]}" | tee -a "$log" || { echo "check: zerodim_tf.py FAILED"; exit 1; }
fi
if grep -q 'DIFFERENT' "$log" || grep 'ratio' "$log" | grep -qv 'ratio 1 \|ratio 1$'; then
  echo "check: a zero-dimensional check FAILED (see $log)"; fail=1
fi
echo "# the general mode against the brute force"
"$py" -I "$here/tools/bruteforce.py" -b "$bin" 3,3 4,4 3,3,4 3,4,5 6,6 4,4,6 3,3,3,3 4,4,4 4,4,4,4 | tee "$out/bruteforce.txt" \
  || { echo "check: bruteforce.py FAILED"; fail=1; }
[ $fail = 0 ] && echo "check: PASSED" || { echo "check: FAILED"; exit 1; }
