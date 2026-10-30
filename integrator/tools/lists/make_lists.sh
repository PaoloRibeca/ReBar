#!/bin/bash
# Regenerate the graph lists of data/lists with the generator 'topologies' (../../../graphs): run its phi^4 mode
# with -N, which writes the integrator's lists itself, then convert its raw compact output independently with
# topo2list.py (gaugraph.py's Nickel canonicaliser and weights) and compare the two byte for byte.
#
# usage: tools/lists/make_lists.sh WORKDIR OUTDIR [LISTS...]
#   LISTS  e.g. 4:8 (Gamma4, 8 loops), 2:9 (Gamma2, 9 loops), 2t:8 (Gamma2 with self-loops, 8 loops);
#          default: 4:8 2:8 2:9
#   TOPOLOGIES  the generator binary (default <repository>/graphs/bin/topologies, built by 'make -C graphs')
#   GAUGRAPH    gaugraph.py (default <repository>/graphs/tools/gaugraph.py)
#   THREADS     OpenMP threads of the generator (default 2)
# Output: OUTDIR/gamma4_L<L>.txt, gamma2_L<L>.txt, gamma2_L<L>_with_selfloops.txt in the format of 'rebar_cpu gen';
# the generator's own output stays in WORKDIR/<E>: the compact lists <L>-loop-<E>[-ci|-cit] and the integrator's
# lists of every order up to the one asked.  Then check them with tools/lists/verify_all.sh OUTDIR CHECKDIR.
# Cost: Gamma4 L = 8 about 25 s to generate (one core of a 2018 workstation) and 250 s to convert with topo2list.py;
# Gamma2 L = 9 about 10 s + 105 s.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../../.." && pwd)
wd=${1:?usage: make_lists.sh WORKDIR OUTDIR [LISTS...]}; od=${2:?usage: make_lists.sh WORKDIR OUTDIR [LISTS...]}; shift 2
bin=${TOPOLOGIES:-$root/graphs/bin/topologies}
gg=${GAUGRAPH:-$root/graphs/tools/gaugraph.py}
[ -x "$bin" ] || { echo "topologies not found at $bin (set TOPOLOGIES, or make -C graphs)" >&2; exit 1; }
[ -f "$gg" ] || { echo "gaugraph.py not found at $gg (set GAUGRAPH)" >&2; exit 1; }
export OMP_NUM_THREADS=${THREADS:-2}
lists=("$@"); [ ${#lists[@]} -eq 0 ] && lists=(4:8 2:8 2:9)
mkdir -p "$od"
# one run of the generator per number of legs, up to the largest order asked
V2=0; V4=0
for x in "${lists[@]}"; do
  kind=${x%%:*}; L=${x##*:}
  case $kind in
    4) [ $((L + 1)) -gt $V4 ] && V4=$((L + 1)) ;;
    2|2t) [ $L -gt $V2 ] && V2=$L ;;
    *) echo "bad list $x" >&2; exit 1 ;;
  esac
done
[ $V4 -gt 0 ] && { mkdir -p "$wd/4"; "$bin" -E 4 -V $V4 -N -o "$wd/4" > "$wd/4/log"; }
[ $V2 -gt 0 ] && { mkdir -p "$wd/2"; "$bin" -E 2 -V $V2 -N -o "$wd/2" > "$wd/2/log"; }
pids=()
for x in "${lists[@]}"; do
  kind=${x%%:*}; L=${x##*:}
  case $kind in
    4)  raw=$wd/4/$L-loop-4;    gen=$wd/4/gamma4_L$L.txt;                out=$od/gamma4_L$L.txt;                keep= ;;
    2)  raw=$wd/2/$L-loop-2-ci; gen=$wd/2/gamma2_L$L.txt;                out=$od/gamma2_L$L.txt;                keep= ;;
    2t) raw=$wd/2/$L-loop-2;    gen=$wd/2/gamma2_L${L}_with_selfloops.txt; out=$od/gamma2_L${L}_with_selfloops.txt; keep=--keep-selfloops ;;
  esac
  sha256sum "$raw"
  cp "$gen" "$out"
  ( python3 -I "$here/topo2list.py" --gaugraph "$gg" $keep "$raw" > "$out.topo2list" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
fail=0
for x in "${lists[@]}"; do
  kind=${x%%:*}; L=${x##*:}
  case $kind in 4) f=$od/gamma4_L$L.txt ;; 2) f=$od/gamma2_L$L.txt ;; 2t) f=$od/gamma2_L${L}_with_selfloops.txt ;; esac
  if cmp -s "$f" "$f.topo2list"; then r="= topo2list.py"; rm -f "$f.topo2list"
  else r="DIFFERS from topo2list.py ($f.topo2list kept)"; fail=1; fi
  printf '%s\t%s graphs\t%s\n' "$(sha256sum "$f")" "$(wc -l < "$f")" "$r"
done
exit $fail
