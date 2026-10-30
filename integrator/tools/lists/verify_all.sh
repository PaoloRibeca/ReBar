#!/bin/bash
# Verify graph lists with the integrator's own graph code (listcheck) and the exact zero-dimensional O(N) sum rules.
#
# usage: tools/lists/verify_all.sh LISTDIR OUTDIR [NAME...]
#   NAME: list basenames without .txt (default: every gamma4_L*, gamma2_L* file of LISTDIR)
#   LISTCHECK  listcheck binary (default: OUTDIR/listcheck, built from tools/lists/listcheck.cpp + src/graph.cpp)
#   GAUGRAPH   gaugraph.py for the sampled cross-check (default ../../../graphs/tools/gaugraph.py); skipped if absent
#   BRUTE9     brute-force canonical check of every BRUTE9-th graph with 9 vertices (default 53; 2.5 s per graph)
#   BRUTE8     the same for 8 vertices (default 1 = all; 0.3 s per graph); 7 or fewer: all
#   PROTO_ON9  the integrator's on_factor on every PROTO_ON9-th graph with 9 vertices (default 1 = all; 8 or fewer: all)
#   SAMPLE     gaugraph cross-check of every SAMPLE-th graph with >= 8 vertices (default 17 at 8, 53 at 9 vertices)
# Writes OUTDIR/NAME.check (one row per graph, see listcheck.cpp) and prints a log; exit status 1 on any failure.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); proto=$(cd "$here/../.." && pwd)
ld=$1; od=$2; shift 2
mkdir -p "$od"
lc=${LISTCHECK:-$od/listcheck}
gg=${GAUGRAPH:-$proto/../graphs/tools/gaugraph.py}
if [ ! -x "$lc" ]; then
  ${CXX:-g++} -O2 -std=c++17 -fopenmp -I"$proto/src" -o "$lc" "$here/listcheck.cpp" "$proto/src/graph.cpp" || exit 1
fi
names=("$@")
if [ ${#names[@]} -eq 0 ]; then for f in "$ld"/gamma4_L*.txt "$ld"/gamma2_L*.txt; do names+=("$(basename "$f" .txt)"); done; fi
fail=0
for n in "${names[@]}"; do
  f=$ld/$n.txt
  L=$(echo "$n" | sed -E 's/^gamma[24]_L([0-9]+).*/\1/')
  case $n in
    gamma4_*) fam=c4; V=$((L + 1)) ;;
    gamma2_*_with_selfloops) fam=c2B; V=$L ;;
    gamma2_*) fam=c2; V=$L ;;
  esac
  br=all; po=all; [ $V -eq 8 ] && br=${BRUTE8:-1}; [ $V -ge 9 ] && { br=${BRUTE9:-53}; po=${PROTO_ON9:-1}; }
  echo "== $n ($fam, L = $L, V = $V; brute-force canonical check: $br; integrator on_factor: $po)"
  t0=$(date +%s)
  "$lc" "$f" --brute $br --proto-on $po > "$od/$n.check" 2> "$od/$n.check.log" || fail=1
  sed 's/^/  /' "$od/$n.check.log"
  python3 -I "$here/verify_lists.py" sumrule $fam $L "$f" "$od/$n.check" | sed 's/^/  /' || fail=1
  if [ $V -ge 8 ] && [ -f "$gg" ]; then
    step=${SAMPLE:-$([ $V -ge 9 ] && echo 53 || echo 17)}
    python3 -I "$here/verify_lists.py" --gaugraph "$gg" sample "$f" "$od/$n.check" $step | sed 's/^/  /' || fail=1
  fi
  echo "  ($(( $(date +%s) - t0 )) s)"
done
echo "-> $([ $fail = 0 ] && echo 'ALL PASS' || echo FAILURES)"
exit $fail
