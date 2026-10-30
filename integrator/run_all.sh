#!/bin/bash
# run_all.sh -- one-shot build, self-test, validation, benchmark and production run of rebar on a CUDA machine.
# Everything is time-boxed; at the end (or on any failure / interruption) ONE results tarball is written:
# results-<UTC date>.tar.gz.  Nothing that names the machine or the user's directories goes into the results (paths
# are relative to this script's directory).
#
# Usage:   ./run_all.sh [--budget MIN] [--out DIR] [--stages LIST] [--with-7loops | --no-7loops] [--jobs N]
#   --budget MIN   global wall-clock budget in minutes (default 60); stages get shares of it, each with a hard timeout
#   --stages LIST  comma-separated subset of: env,build,cpu,gpu,validate,tuning,throughput,scaling,production,loops7
#                  (default: all; loops7 only with --with-7loops or a budget >= 150 min)
#   --out DIR      results directory (default results-<UTC date>)
# Environment:  NVCC, HOSTCXX, CUDA_ARCH (default 90), CUDA_VISIBLE_DEVICES (which GPUs to use; default all).
# Needs: bash, make, a C++17 compiler with OpenMP, a CUDA 12.x toolkit, coreutils; python3 (stdlib only) for the
# reports (without it the raw logs/JSON are still collected).  No root, no network, no scheduler.
set -u
set -m   # job control: every step runs in its own process group, so a timeout can stop it with all its children
cd "$(dirname "$0")"
ROOT=$(pwd)
T0=$(date +%s)
BUDGET_MIN=60
OUT=""
STAGES="env,build,cpu,gpu,validate,tuning,throughput,scaling,production,loops7"
WITH7=auto
JOBS=$(nproc 2>/dev/null || echo 8)
while [ $# -gt 0 ]; do
  case $1 in
    --budget) BUDGET_MIN=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    --with-7loops) WITH7=yes; shift ;;
    --no-7loops) WITH7=no; shift ;;
    --jobs) JOBS=$2; shift 2 ;;
    -h|--help) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
[ -n "$OUT" ] || OUT="results-$STAMP"
mkdir -p "$OUT"
# relative to ROOT (the working directory from here on), so that no absolute path reaches a run record or a log
OUT=$(cd "$OUT" && pwd)
case $OUT in
  "$ROOT") OUT=. ;;
  "$ROOT"/*) OUT=${OUT#"$ROOT"/} ;;
  *) OUT=$(realpath --relative-to="$ROOT" "$OUT" 2> /dev/null || echo "$OUT") ;;
esac
TARBALL="$(basename "$(cd "$OUT" && pwd)").tar.gz"
BUDGET=$((BUDGET_MIN * 60))
T_END=$((T0 + BUDGET))
S="$OUT/SUMMARY.txt"
: > "$S"
PY=$(command -v python3 || true)
CUDA_ARCH=${CUDA_ARCH:-90}
export CUDA_ARCH
[ "$WITH7" = auto ] && { [ "$BUDGET_MIN" -ge 150 ] && WITH7=yes || WITH7=no; }

log() { echo "[$(date -u +%H:%M:%S) +$(( $(date +%s) - T0 ))s] $*" | tee -a "$S"; }
now() { date +%s; }
left() { echo $(( T_END - $(now) )); }
has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
# nvidia-smi -q, only the sections that matter here (ECC, clocks, temperatures, power), without UUIDs, serial and part
# numbers or PCI bus ids (each GPU block is headed by its bus id: replaced by its index)
smi_q() {
  nvidia-smi -q -d ECC,PERFORMANCE,CLOCK,TEMPERATURE,POWER 2>&1 | awk '/^GPU [0-9A-Fa-f]+:/ { print "GPU " n++; next }
    /UUID|Serial|Part Number|Bus Id|Board Id|PCI/ { next } { print }'
}

# ---- always leave a tarball, whatever happens -------------------------------------------------------------
FINISHED=0
WATCHDOG=""
finish() {
  [ "$FINISHED" = 1 ] && return
  FINISHED=1
  [ -n "$WATCHDOG" ] && kill -- -"$WATCHDOG" 2> /dev/null   # the global watchdog must not outlive the script
  log "packaging (exit reason: ${1:-normal})"
  # verdicts of whatever has run so far (also after an interruption or a global timeout)
  [ -n "$PY" ] && [ -f tools/validate.py ] && timeout -k 10 120 "$PY" tools/validate.py "$OUT" > "$OUT/VALIDATION.txt" 2>&1
  if [ -n "$PY" ] && [ -f tools/report.py ]; then
    timeout -k 10 120 "$PY" tools/report.py "$OUT" > "$OUT/REPORT.md" 2> "$OUT/report.err" || echo "report.py failed (see report.err)" >> "$S"
  fi
  cp -f "$S" "$OUT/SUMMARY.txt" 2>/dev/null
  tar czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null
  echo "==> wrote $TARBALL ($(du -h "$TARBALL" 2>/dev/null | cut -f1))"
}
CUR=""
on_signal() {
  if [ -n "$CUR" ]; then kill -TERM -- -"$CUR" 2> /dev/null; sleep 5; kill -KILL -- -"$CUR" 2> /dev/null; fi
  finish interrupted
  exit 130
}
trap on_signal INT TERM
trap 'finish' EXIT
# global watchdog: if something wedges beyond the budget (+10 min), terminate this script (the trap packages)
( sleep $((BUDGET + 600)); kill -TERM $$ 2>/dev/null ) > /dev/null 2>&1 &
WATCHDOG=$!
disown $WATCHDOG 2> /dev/null

# run_step NAME SECONDS CMD...: run CMD with stdout/stderr to $OUT/logs/NAME.{out,err}; SIGTERM at the limit (rebar
# then finishes its launches and writes its JSON), SIGKILL 60 s later; never waits on a process that does not die
# (e.g. stuck in the driver): the script goes on.  Returns the command's exit status (124 on timeout).
mkdir -p "$OUT/logs"
run_step() {
  local name=$1 lim=$2; shift 2
  local l; l=$(left)
  [ "$lim" -gt $((l - 60)) ] && lim=$((l - 60))
  if [ "$lim" -lt 10 ]; then log "[$name] SKIPPED (global budget exhausted)"; return 125; fi
  local t0; t0=$(now)
  echo "+ $*" > "$OUT/logs/$name.cmd"
  "$@" > "$OUT/logs/$name.out" 2> "$OUT/logs/$name.err" &
  local pid=$! rc=0 killed=0
  CUR=$pid
  while kill -0 $pid 2> /dev/null; do
    if [ $(( $(now) - t0 )) -ge "$lim" ]; then
      if [ $killed = 0 ]; then kill -TERM -- -$pid 2> /dev/null; killed=1; lim=$((lim + 60));
      else kill -KILL -- -$pid 2> /dev/null; killed=2; sleep 2; break; fi
    fi
    sleep 1
  done
  if [ $killed = 2 ] && kill -0 $pid 2> /dev/null; then rc=137; log "[$name] process $pid does not die (stuck in the driver?); continuing without it";
  else wait $pid; rc=$?; [ $killed -ge 1 ] && [ $rc = 0 ] && rc=124; fi
  CUR=""
  log "[$name] exit $rc ($(( $(now) - t0 )) s)$([ $killed -ge 1 ] && echo ' TIMEOUT')"
  return $rc
}
# Time of the variable-length stages: what is left of the budget (minus 2 min for packaging) is divided among the
# stages still to come in proportion to their weights, so an overrun early on shrinks every later stage instead of
# eliminating the last ones.  take STAGE sets SHARE (seconds) and removes the stage's weight.
declare -A W=([validate]=20 [tuning]=10 [throughput]=12 [scaling]=5 [production]=25 [loops7]=30)
REMW=0
take() { local l=$(( $(left) - 120 )); [ $l -lt 0 ] && l=0; SHARE=$(( l * W[$1] / (REMW > 0 ? REMW : 1) )); REMW=$((REMW - W[$1])); }

log "rebar run_all.sh: budget $BUDGET_MIN min, stages $STAGES, 7 loops: $WITH7, results in $OUT"
B=bin/rebar_cpu
G=bin/rebar_gpu
GOLD=../results/checks/golden-cpu
LISTS=data/lists

# ============================================================================================ 0. environment
if has_stage env; then
  log "== stage env"
  {
    echo "### date"; date -u; echo; echo "### uname"; uname -srmo; echo; echo "### os-release"; cat /etc/os-release 2> /dev/null
    echo; echo "### lscpu"; lscpu 2> /dev/null | head -30; echo; echo "### memory"; free -g 2> /dev/null
    # GPU models, memory, clocks and driver; no UUID, serial or PCI bus id
    echo; echo "### nvidia-smi"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,persistence_mode,clocks.max.sm,clocks.max.mem,power.limit,temperature.gpu,utilization.gpu --format=csv 2>&1
    echo; echo "### nvidia-smi -L"; nvidia-smi -L 2>&1 | sed -E 's/ \((UUID: )?[A-Z]+-[0-9a-fA-F-]+\)//'
    echo; echo "### driver"; sed -E 's/ ?\([^()]*@[^()]*\)//' /proc/driver/nvidia/version 2>&1
    echo; echo "### nvidia-smi topo -m"; nvidia-smi topo -m 2>&1 | head -40
    echo; echo "### CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  CUDA_ARCH=$CUDA_ARCH  NVCC=${NVCC:-<auto>}  HOSTCXX=${HOSTCXX:-<auto>}"
    echo; echo "### nvcc"; ${NVCC:-nvcc} --version 2>&1; ls -d /usr/local/cuda* 2> /dev/null
    echo; echo "### compilers"; for c in g++ g++-14 g++-13 g++-12 g++-11 g++-10 g++-9 g++-8 g++-7 clang++; do command -v $c > /dev/null && echo "$c: $($c --version | head -1)"; done
    echo; echo "### python3"; ${PY:-false} --version 2>&1
    echo; echo "### sources (sha256)"; find include src cuda tools data "$GOLD" -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.cu' -o -name '*.py' -o -name '*.sh' -o -name '*.txt' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum
    sha256sum Makefile run_all.sh
  } > "$OUT/environment.txt" 2>&1
  smi_q > "$OUT/nvidia-smi-q.txt" 2>&1
  log "environment recorded ($(nvidia-smi -L 2> /dev/null | wc -l) GPU(s) listed by nvidia-smi)"
fi

# ================================================================================================= 1. build
if has_stage build; then
  log "== stage build"
  make distclean > /dev/null 2>&1   # never reuse objects or a configuration from another machine
  run_step build_config 300 make config
  cp -f build/config.mk "$OUT/config.mk" 2> /dev/null
  log "  toolkit: $(grep -E '^(NVCC|CUDA_VERSION|HOSTCXX|NVCC_AUTO|DETECT_NOTE) ' build/config.mk 2> /dev/null | tr '\n' ' ')"
  run_step build_cpu 900 make -j"$JOBS" cpu
  run_step build_gpu 1200 make -j"$JOBS" gpu
  make -n gpu > "$OUT/make_n_gpu.txt" 2>&1
  run_step cuda_resources 600 make -s cuda-resources
  [ -x $B ] || log "NO CPU BINARY: see logs/build_cpu.err"
  [ -x $G ] || log "NO GPU BINARY: see logs/build_gpu.err and config.mk (set NVCC=... HOSTCXX=... and rerun)"
fi
NDEV=0; NDEVC=0
if [ -x $G ]; then
  TDEV=$(now)
  run_step devices 300 $G devices --out "$OUT/devices.json"
  TDEV=$(( $(now) - TDEV ))
  if [ -n "$PY" ]; then
    NDEV=$("$PY" -c "import json;d=json.load(open('$OUT/devices.json'));print(d.get('devices_ok',0))" 2> /dev/null || echo 0)
    NDEVC=$("$PY" -c "import json;d=json.load(open('$OUT/devices.json'));print(d.get('device_count',0))" 2> /dev/null || echo 0)
  else
    NDEVC=$(nvidia-smi -L 2> /dev/null | wc -l); NDEV=$NDEVC
  fi
  log "CUDA devices: $NDEVC present, $NDEV passed the probe (Philox on the device, build signature)"
  if [ "$NDEV" = 0 ] && [ -n "$PY" ]; then
    E=$("$PY" -c "import json;d=json.load(open('$OUT/devices.json'));print(d.get('error') or '; '.join(e for e in (x.get('error') or x.get('fptest_error') for x in d.get('devices',[])) if e))" 2> /dev/null)
    [ -n "$E" ] && log "  CUDA reports: $E"
  fi
  # every GPU step is a new process that initialises CUDA on all visible GPUs: if that is slow (e.g. persistence mode
  # off), every step pays it and the later stages get less time
  [ "$NDEVC" -gt 0 ] && [ "$TDEV" -ge 10 ] && log "  WARNING: CUDA start-up took $TDEV s (GPU persistence mode off? see environment.txt); every GPU step pays it, so the later stages get less time: consider --budget 90 or more"
fi
# REBAR_TEST_EMU=1: test mode of this script on a machine without GPU: every "GPU" step runs on 2 EMULATED devices
# (the device program on the host), so that all stages, time boxes and reports can be exercised.  Not for production runs.
BK=cuda; EMUOPT=""
if [ -n "${REBAR_TEST_EMU:-}" ]; then BK=emu; EMUOPT="--emu-devices 2 --emu-blocks 4"; NDEV=2; NDEVC=2; log "TEST MODE: GPU stages on 2 emulated devices"; fi
export REBAR_EXPECT_BACKEND=$BK
GPUOK=0; [ -x $G ] && [ "$NDEV" -gt 0 ] && GPUOK=1
GPUANY=0; [ -x $G ] && [ "$NDEVC" -gt 0 ] && GPUANY=1
[ $GPUANY = 0 ] && log "NO USABLE GPU: the GPU stages are skipped (see logs/devices.* and logs/build_gpu.*)"
GRUN="$G run --backend $BK $EMUOPT"   # GPU runs never fall back to the CPU silently
for st in validate tuning throughput production; do has_stage $st && REMW=$((REMW + W[$st])); done
has_stage scaling && [ "$NDEV" -gt 1 ] && REMW=$((REMW + W[scaling]))
has_stage loops7 && [ "$WITH7" = yes ] && [ -f $LISTS/gamma4_L7.txt ] && REMW=$((REMW + W[loops7]))

# ============================================================================================ 2. CPU checks
if has_stage cpu && [ -x $B ]; then
  log "== stage cpu (host build: Philox known answers, GPU emulation test, golden configurations on this CPU)"
  run_step cpu_philox 60 $B check --what philox
  run_step cpu_emu_b3 600 $B check --what emu --kind b --loops 3
  run_step cpu_emu_h4 600 $B check --what emu --kind h --loops 4 --n 20000
  # the same counters as the golden files, on THIS machine's CPU (another glibc / -march: round-off level only)
  for g in b_L4 c_L3 h_L4; do
    k=${g%_L*}; L=${g#*_L}
    run_step cpu_golden_$g 600 $B run --kind $k --loops $L --gen --pilot 65536 --samples 0 --seed 2026 --out "$OUT/cpu_golden_$g.json" &&
      [ -n "$PY" ] && log "  cpu_golden_$g: $("$PY" tools/golden_check.py $GOLD/$g.json "$OUT/cpu_golden_$g.json" | tail -1 | sed 's#^.* vs golden [^:]*: ##')"
  done
  [ -x $G ] && run_step gpubin_emu_b3 600 $G check --what emu --kind b --loops 3
  # v0.4.0: the e18 limit variant reproduces the golden counters bitwise on the CPU
  if [ -x $B-e18 ]; then
    if run_step cpu_golden_b_L4_e18 600 env REBAR_NO_REEXEC=1 $B-e18 run --kind b --loops 4 --gen --pilot 65536 --samples 0 --seed 2026 --out "$OUT/cpu_golden_b_L4_e18.json"; then
      [ -n "$PY" ] && [ -s "$OUT/cpu_golden_b_L4.json" ] && log "  cpu_golden_b_L4_e18 vs e16 (bitwise): $("$PY" tools/jsoncmp.py "$OUT/cpu_golden_b_L4.json" "$OUT/cpu_golden_b_L4_e18.json" --quiet | tail -1 | sed 's#.*: ##')"
    fi
  fi
fi

# ====================================================================== 3. GPU self-tests and golden checks
golden() {   # golden NAME GOLDENFILE MINDEV EXTRA...: GPU run on the same counters as the CPU golden file
  local name=$1 gf=$2 mind=$3; shift 3
  local b; b=$(basename "$gf" .json); local k=${b%_L*} L=${b#*_L}
  # graph source as recorded in the golden file (--gen, or --list data/lists/... [--drop-1vr])
  local src="--gen"
  [ -n "$PY" ] && src=$("$PY" -c "import json;a=json.load(open('$gf'))['args'];print(('--list '+a['list']) if 'list' in a else '--gen', '--drop-1vr' if 'drop-1vr' in a else '')" 2> /dev/null || echo --gen)
  if run_step "$name" 900 $GRUN --kind $k --loops $L $src --pilot 65536 --samples 0 --seed 2026 "$@" --out "$OUT/$name.json"; then
    [ -n "$PY" ] && log "  $name: $("$PY" tools/golden_check.py "$gf" "$OUT/$name.json" --require-backend $BK --min-devices "$mind" | tail -1 | sed 's#^.* vs golden [^:]*: ##')"
  fi
}
if has_stage gpu && [ $GPUANY = 1 ]; then
  log "== stage gpu (device self-tests; GPU vs CPU on identical Philox counters)"
  for a in "b 4" "h 4" "c 4" "int4 4" "b 5"; do
    set -- $a
    run_step selftest_$1$2 600 $G selftest --backend $BK $EMUOPT --kind $1 --loops $2 --selftest-classes 5 --out "$OUT/selftest_$1$2.json"
  done
  run_step selftest_int44_fp32 600 $G selftest --backend $BK $EMUOPT --kind int4 --loops 4 --fp32 --selftest-classes 5 --out "$OUT/selftest_int44_fp32.json"
  for f in $GOLD/*.json; do golden golden_$(basename $f .json) "$f" 1; done
  # the e18 limit variant (needed by h 9, c 9): same golden numbers, its own build signature checked on the device
  if [ -x $G-e18 ]; then
    GRUN16=$GRUN; GRUN="env REBAR_NO_REEXEC=1 $G-e18 run --backend $BK $EMUOPT"
    for g in b_L5 h_L5 c_L4; do golden golden_${g}_e18 $GOLD/$g.json 1; done
    GRUN=$GRUN16
  fi
  # force the work to be split into many launches over all GPUs (requeueing / merging across devices)
  M2=$(( NDEV > 1 ? 2 : 1 ))
  golden golden_b_L5_split32 $GOLD/b_L5.json $M2 --max-launch-chunks 32
  golden golden_int4_L5_split8 $GOLD/int4_L5.json $M2 --max-launch-chunks 8 --launch-s 0.01
  # every GPU on its own, including any that failed the probe (a single bad device must be identifiable)
  for ((d = 0; d < NDEVC && d < 16; d++)); do golden golden_b_L5_dev$d $GOLD/b_L5.json 1 --devices $d; done
fi

# Within a stage: stage_begin SECONDS TOTAL_WEIGHT NSTEPS; then tnext W sets T = seconds for the next step (weight W):
# what is left of the stage, minus ~8 s fixed cost per remaining step, in proportion to the remaining weight (>= 15 s)
stage_begin() { STAGE_END=$(( $(now) + $1 )); SW=$2; SN=$3; }
tnext() { local r=$(( STAGE_END - $(now) - 8 * SN )); T=$(( r * $1 / (SW > 0 ? SW : 1) )); SW=$((SW - $1)); SN=$((SN - 1)); [ $T -lt 15 ] && T=15; }

# ========================================================================== 4. validation against references
# Exact low orders at very high statistics (sharp test of device numerics), 3-5 loop orders vs Phi4tools, replica
# ensembles (are the error bars right?), O(N) recombination, FP32 bare kernel.  PASS/FAIL by tools/validate.py.
if has_stage validate && [ $GPUOK = 1 ]; then
  log "== stage validate"
  # weights: 12 main runs (8 or 4), 16 replicas (1), FP32/FP64 pair (4 each): 72 + 16 + 8 = 96, 30 steps
  take validate; stage_begin $SHARE 96 30
  for a in "b 2 8" "h 2 8" "c 2 8" "b 3 4" "h 3 4" "c 3 4" "b 4 4" "h 4 4" "c 4 4" "b 5 8" "h 5 8" "c 5 8"; do
    set -- $a
    tnext $3
    run_step val_$1$2 $((T + 600)) $GRUN --kind $1 --loops $2 --gen --pilot 65536 --seconds $T --seed $((900 + ${#1} * 10 + $2)) --out "$OUT/val_$1$2.json"
  done
  # replica ensembles: 8 independent seeds each (pulls should be N(0,1); per-class scatter chi2/dof ~ 1)
  for a in "b 4" "h 4"; do
    set -- $a
    mkdir -p "$OUT/replicas_$1$2"
    for r in $(seq 1 8); do
      tnext 1
      run_step rep_$1$2_$r $((T + 300)) $GRUN --kind $1 --loops $2 --gen --pilot 16384 --seconds $T --seed $((5000 + r)) --out "$OUT/replicas_$1$2/r$r.json"
    done
  done
  # FP32 evaluation (bare kinds only): int4 (t0-subtracted sunsets) in FP32 and FP64 with the same seed
  tnext 4
  run_step val_int45_fp32 $((T + 600)) $GRUN --kind int4 --loops 5 --gen --pilot 65536 --seconds $T --seed 951 --fp32 --out "$OUT/val_int45_fp32.json"
  tnext 4
  run_step val_int45_fp64 $((T + 600)) $GRUN --kind int4 --loops 5 --gen --pilot 65536 --seconds $T --seed 951 --out "$OUT/val_int45_fp64.json"
  [ -n "$PY" ] && "$PY" tools/recombine_on.py "$OUT/val_b5.json" > "$OUT/val_b5_on.txt" 2>&1
  [ -n "$PY" ] && { "$PY" tools/validate.py "$OUT" > "$OUT/VALIDATION.txt" 2>&1; log "  $(tail -1 "$OUT/VALIDATION.txt")"; }
fi

# ===================================================================== 5. tuning variants (occupancy, FMA)
if has_stage tuning && [ $GPUOK = 1 ]; then
  # the default build is MINBLOCKS=8 FMAD=false since v0.4.0 (measured on the H200: +30 % and the same speed); the
  # variants are the v0.3.0 build (MINBLOCKS=1, fmad=true), MINBLOCKS=6, and fmad=true at MINBLOCKS=8
  log "== stage tuning (variants of the device build: v030 = RB_MINBLOCKS 1 + fmad, mb6 = RB_MINBLOCKS 6, fmad = fmad at 8)"
  take tuning; TS=$SHARE; per=$(( (TS - 3 * 90 - 9 * 8) / 6 )); [ $per -lt 15 ] && per=15
  for v in "v030 1 true" "mb6 6 false" "fmad 8 true"; do
    set -- $v
    if run_step build_$1 900 make -j"$JOBS" gpu TAG=$1 MINBLOCKS=$2 FMAD=$3 VARIANTS=e16; then
      make -s cuda-resources MINBLOCKS=$2 FMAD=$3 VARIANTS=e16 > "$OUT/cuda_resources_$1.txt" 2>&1
      GV=bin/rebar_gpu-$1
      if run_step golden_b_L5_$1 600 $GV run --backend $BK $EMUOPT --kind b --loops 5 --gen --pilot 65536 --samples 0 --seed 2026 --out "$OUT/golden_b_L5_$1.json"; then
        [ -n "$PY" ] && log "  golden_b_L5_$1: $("$PY" tools/golden_check.py $GOLD/b_L5.json "$OUT/golden_b_L5_$1.json" --require-backend $BK | tail -1 | sed 's#^.* vs golden [^:]*: ##')"
      fi
      for a in "b 5" "c 5"; do
        set -- $1 $a
        run_step tput_$2$3_$1 $((per + 300)) $GV run --backend $BK $EMUOPT --kind $2 --loops $3 --gen --alloc equal --pilot 8192 --seconds $per --seed 77 --out "$OUT/tput_$2$3_$1.json"
      done
    fi
  done
fi

# ============================================================= 6. throughput per kernel and per loop order
if has_stage throughput && [ $GPUOK = 1 ]; then
  log "== stage throughput (equal allocation over the classes of an order; device samples/s)"
  take throughput; stage_begin $SHARE 19 19
  tput() {   # tput NAME KIND L LISTARGS... : one time-boxed equal-allocation run
    local name=$1 k=$2 L=$3; shift 3
    tnext 1
    run_step tput_$name $((T + 600)) $GRUN --kind $k --loops $L "$@" --alloc equal --pilot 8192 --seconds $T --seed 78 --out "$OUT/tput_$name.json"
  }
  for L in 4 5; do for k in b h c int4; do tput $k$L $k $L --gen; done; done
  tput b6 b 6 --list $LISTS/gamma4_L6.txt --drop-1vr
  tput h6 h 6 --list $LISTS/gamma2_L6.txt --drop-1vr
  tput c6 c 6 --list $LISTS/gamma2_L6.txt
  tput int46 int4 6 --list $LISTS/gamma4_L6.txt
  # FP64 vs FP32 evaluation (FP64 accumulation) for the bare kernels
  for L in 4 5; do
    tput bare4$L bare4 $L --gen --skip-divergent
    tput bare4${L}_fp32 bare4 $L --gen --skip-divergent --fp32
    tput int4${L}_fp32 int4 $L --gen --fp32
  done
  # long launches at 6 loops (chunk time vs --launch-s)
  tput c6_launch2 c 6 --list $LISTS/gamma2_L6.txt --launch-s 2
fi

# ========================================================================================= 7. multi-GPU scaling
if has_stage scaling && [ $GPUOK = 1 ] && [ "$NDEV" -gt 1 ]; then
  log "== stage scaling (b at 5 loops on 1, 2, 4, ... GPUs)"
  nsub=0; n=1; while :; do nsub=$((nsub + 1)); [ $n -ge $NDEV ] && break; n=$((n * 2)); [ $n -gt $NDEV ] && n=$NDEV; done
  take scaling; stage_begin $SHARE $nsub $nsub
  n=1
  while :; do
    devs=$(seq -s, 0 $((n - 1)))
    tnext 1
    run_step scale_b5_n$n $((T + 300)) $GRUN --kind b --loops 5 --gen --alloc equal --pilot 8192 --seconds $T --seed 79 --devices $devs --out "$OUT/scale_b5_n$n.json"
    [ $n -ge $NDEV ] && break
    n=$((n * 2)); [ $n -gt $NDEV ] && n=$NDEV
  done
fi

# ================================================== 8. production-style runs: whole orders, Neyman allocation
if has_stage production && [ $GPUOK = 1 ]; then
  log "== stage production (b, h, c at 4, 5, 6 loops; pilot, Neyman allocation by measured device time, rounds)"
  take production; stage_begin $SHARE 20 9   # time weights: 6 loops get most of the time
  for a in "b 4 1" "h 4 1" "c 4 1" "b 5 2" "h 5 2" "c 5 2" "b 6 4" "h 6 3" "c 6 4"; do
    set -- $a
    tnext $3
    case $2 in
      6) if [ $1 = b ]; then src="--list $LISTS/gamma4_L6.txt --drop-1vr"; elif [ $1 = h ]; then src="--list $LISTS/gamma2_L6.txt --drop-1vr"; else src="--list $LISTS/gamma2_L6.txt"; fi ;;
      *) src="--gen" ;;
    esac
    run_step prod_$1$2 $((T + 600)) $GRUN --kind $1 --loops $2 $src --pilot 32768 --seconds $T --alloc time --seed $((700 + $2)) --out "$OUT/prod_$1$2.json"
  done
fi

# =================================================================================== 9. optional: 7 loops
if has_stage loops7 && [ $GPUOK = 1 ] && [ "$WITH7" = yes ] && [ -f $LISTS/gamma4_L7.txt ]; then
  log "== stage loops7 (b, h, c at 7 loops; the host builds the classes first: minutes)"
  take loops7; stage_begin $SHARE 4 3
  tnext 1; run_step prod_h7 $((T + 1800)) $GRUN --kind h --loops 7 --list $LISTS/gamma2_L7.txt --drop-1vr --pilot 16384 --seconds $T --alloc time --seed 707 --out "$OUT/prod_h7.json"
  tnext 2; run_step prod_b7 $((T + 3600)) $GRUN --kind b --loops 7 --list $LISTS/gamma4_L7.txt --drop-1vr --pilot 16384 --seconds $T --alloc time --seed 707 --out "$OUT/prod_b7.json"
  tnext 1; run_step prod_c7 $((T + 3600)) $GRUN --kind c --loops 7 --list $LISTS/gamma2_L7.txt --pilot 16384 --seconds $T --alloc time --seed 707 --out "$OUT/prod_c7.json"
fi

# ============================================================================================ 10. wrap up
smi_q > "$OUT/nvidia-smi-after.txt" 2>&1
dmesg 2> /dev/null | grep -i "xid\|nvrm" | tail -50 | sed -E 's/PCI:[0-9a-fA-F:.]+/PCI:xxxx/g; s/ ?\([^()]*@[^()]*\)//' > "$OUT/dmesg_xid.txt" 2>&1
[ -n "$PY" ] && [ -f tools/validate.py ] && "$PY" tools/validate.py "$OUT" > "$OUT/VALIDATION.txt" 2>&1
[ -f "$OUT/VALIDATION.txt" ] && log "$(tail -1 "$OUT/VALIDATION.txt")"
kill -- -$WATCHDOG 2> /dev/null
log "done in $(( $(now) - T0 )) s"
exit 0
