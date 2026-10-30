#!/bin/bash
# rebar-refine.sh -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>
#
# Adds statistics to the ledger: picks an order (named, or the open one with the largest relative error), reserves
# a fresh seed in ledger/seeds.json, runs the integrator on that order's whole list for a time budget, writing the
# run JSON to ledger/<order>/runs/<id>.json (outside git), and merges it with rebar-merge.py, which folds it into
# the ledger and keeps it in the archive (ledger-archive/<order>/<id>.json.xz). An interrupted run stays reserved
# and is continued with the integrator's --resume the next time this is called.
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
set -o pipefail
usage() {
  cat <<'EOF'
usage: rebar-refine.sh --bin BIN --seconds T [--order ORDER] [--label LABEL] [--pilot N] [--round-s S]
                       [--threads N] [--ledger DIR] [--dry-run] [-- EXTRA INTEGRATOR OPTIONS]
  BIN       integrator/bin/rebar_gpu or rebar_cpu, version >= 0.4.0 (make -C integrator gpu | cpu)
  T         time budget of the run in seconds (the run checkpoints every --round-s, default 900)
  ORDER     b8, h9, ...; default: the open order with the largest relative error (rebar_ledger.py pick)
  LABEL     part of the contribution id, default 'refine'
The run: BIN run --kind K --loops L --list integrator/data/lists/<list> [--drop-1vr for b, h] --pilot 16384
--alloc time --seconds T --seed S --round-s 900 --out ledger/<order>/runs/<id>.json, then rebar-merge.py (which
archives the run; runs/ is ignored by git).  The integrator runs in the repository's top directory, with the paths
above relative to it (paths among the EXTRA options are taken relative to it too).
If the integrator does not end normally (killed, SIGTERM: exit status != 0), the seed stays reserved ("running" in
seeds.json) and the next call continues that run with BIN run --resume FILE --seconds T before anything else.
EOF
}
die() { echo "rebar-refine: $*" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd); root=$(dirname "$here"); py=${PYTHON:-python3}
ledger=$root/ledger; bin=; secs=; order=; label=refine; pilot=16384; rounds=900; threads=; dry=0; extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --bin) bin=$2; shift;;
    --seconds) secs=$2; shift;;
    --order) order=$2; shift;;
    --label) label=$2; shift;;
    --pilot) pilot=$2; shift;;
    --round-s) rounds=$2; shift;;
    --threads) threads=$2; shift;;
    --ledger) ledger=$2; shift;;
    --dry-run) dry=1;;
    -h|--help) usage; exit 0;;
    --) shift; extra=("$@"); break;;
    *) usage >&2; exit 2;;
  esac
  shift
done
[ -n "$bin" ] && [ -n "$secs" ] || { usage >&2; exit 2; }
[ -x "$bin" ] || die "no integrator binary '$bin' (build it: make -C integrator gpu, or make -C integrator cpu)"
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
# the integrator runs in the top directory of the repository with paths relative to it, so that its run record (args,
# checkpoint sessions) holds no absolute path
rel() { realpath -m --relative-to="$root" "$1" 2> /dev/null || echo "$1"; }
lib=("$py" -I "$here/rebar_ledger.py" --ledger "$ledger")
devtag() {  # h200x5 from the probe of rebar_gpu, cpu<threads> for rebar_cpu
  "$bin" devices 2>/dev/null | "$py" -I -c '
import json, re, sys
try:
    D = json.load(sys.stdin)
except Exception:
    D = {}
ok = [d for d in D.get("devices", []) if d.get("opened", True) and not d.get("error")]
n = D.get("devices_ok") or 0
if n and ok:
    name = (ok[0].get("properties") or {}).get("name") or "gpu"
    print("%sx%d" % (re.sub(r"[^a-z0-9]", "", name.lower().replace("nvidia", "").split()[-1]) or "gpu", n))
else:
    print("cpu%s" % sys.argv[1])' "${threads:-${OMP_NUM_THREADS:-$(nproc)}}"
}
# 1. an interrupted run is continued first
read -r porder pseed pid <<< "$("${lib[@]}" pending $order)"
if [ -n "$pid" ]; then
  order=$porder; seed=$pseed; id=$pid; state="continuing the interrupted run"
else
  read -r order kind L list <<< "$("${lib[@]}" pick $order)" || exit 2
  [ -n "$list" ] || exit 2
  if [ $dry = 1 ]; then
    id="(dry run: a fresh id)"; seed="(fresh)"
  else
    read -r id seed <<< "$("${lib[@]}" reserve "$order" "$label" "$(devtag)")" || die "could not reserve a seed"
  fi
  state="new run"
fi
kind=${order:0:1}; L=${order:1}
read -r _ _ _ list <<< "$("${lib[@]}" pick "$order")"
out=$ledger/$order/runs/$id.json
complete=
[ -f "$out" ] && complete=$("$py" -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("complete"))' "$out")
topt=(); [ -n "$threads" ] && topt=(--threads "$threads")
if [ -f "$out" ] && [ "$complete" = "True" ]; then
  cmd=()                                   # the run finished but was not merged: merge only
elif [ -f "$out" ]; then
  cmd=("$bin" run --resume "$(rel "$out")" --seconds "$secs" "${topt[@]}" "${extra[@]}")
else
  cmd=("$bin" run --kind "$kind" --loops "$L" --list "$(rel "$list")")
  [ "$kind" != c ] && cmd+=(--drop-1vr)
  cmd+=(--pilot "$pilot" --alloc time --seconds "$secs" --seed "$seed" --round-s "$rounds" --out "$(rel "$out")" "${topt[@]}" "${extra[@]}")
fi
echo "rebar-refine: $order ($state), seed $seed, id $id"
[ ${#cmd[@]} -gt 0 ] && echo "  ${cmd[*]}"
[ $dry = 1 ] && { echo "  (dry run: nothing reserved, run or merged)"; exit 0; }
if [ ${#cmd[@]} -gt 0 ]; then
  mkdir -p "$(dirname "$out")"
  (cd "$root" && "${cmd[@]}"); st=$?
  if [ $st -ne 0 ]; then
    echo "rebar-refine: the integrator exited with status $st; $id stays reserved: run rebar-refine.sh again to continue it" >&2
    exit $st
  fi
fi
"$py" -I "$here/rebar-merge.py" --ledger "$ledger" "$out"; st=$?
[ $st -eq 0 ] || echo "rebar-refine: $id was not merged; it stays reserved (inspect it; rebar-merge.py --force, or delete the file and its seeds.json entry)" >&2
exit $st
