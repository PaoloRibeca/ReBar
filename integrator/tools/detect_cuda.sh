#!/bin/bash
# detect_cuda.sh -- find a CUDA toolkit (nvcc), its version, and a host C++ compiler that this nvcc accepts;
# print the result as Makefile assignments (the Makefile writes it to build/config.mk).
#
# Environment overrides (all optional):
#   NVCC       nvcc to use              (default: nvcc in PATH, then $CUDA_HOME, $CUDA_PATH, /usr/local/cuda, ...)
#   HOSTCXX    host compiler for nvcc   (default: the first of g++, g++-14 ... g++-7, the conda compiler next to nvcc,
#                                        clang++ that nvcc accepts)
#   CUDA_ARCH  SASS targets             (default: 90, i.e. H100/H200)
# Never fails: without a usable toolkit it prints NVCC := (empty) and the reason.
set -u
log() { echo "# $*"; echo "detect_cuda: $*" >&2; }
TMP=$(mktemp -d 2>/dev/null || mktemp -d -t rbdetect)
trap 'rm -rf "$TMP"' EXIT

# ---- nvcc
cand=()
[ -n "${NVCC:-}" ] && cand+=("$NVCC")
command -v nvcc > /dev/null 2>&1 && cand+=("$(command -v nvcc)")
for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}" "${CUDA_ROOT:-}" /usr/local/cuda; do [ -n "$d" ] && cand+=("$d/bin/nvcc"); done
for d in $(ls -d /usr/local/cuda-1[2-9]* /usr/local/cuda-1[2-9]*.* 2>/dev/null | sort -rV) \
         $(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/1[2-9]* /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda 2>/dev/null | sort -rV); do
  cand+=("$d/bin/nvcc")
done
NV=""
for c in "${cand[@]}"; do
  if [ -x "$c" ] && "$c" --version > /dev/null 2>&1; then NV=$c; break; fi
done
if [ -z "$NV" ]; then
  log "no working nvcc found (tried: ${cand[*]:-none}); GPU build disabled (set NVCC=/path/to/nvcc)"
  echo "NVCC :="
  echo "CUDA_VERSION :="
  echo "DETECT_NOTE := no nvcc found"
  exit 0
fi
VER=$("$NV" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
MAJ=${VER%%.*}; MIN=${VER#*.}
log "nvcc: $NV (CUDA $VER)"
ARCH=${CUDA_ARCH:-90}
if [ -z "$VER" ] || [ "$MAJ" -lt 11 ] || { [ "$MAJ" -eq 11 ] && [ "$MIN" -lt 8 ]; }; then
  log "CUDA $VER is too old for sm_90 (needs >= 11.8; this code is tested with 12.4); GPU build disabled"
  echo "NVCC :="
  echo "CUDA_VERSION := $VER"
  echo "DETECT_NOTE := CUDA $VER too old (need >= 11.8, preferably 12.x)"
  exit 0
fi
[ "$MAJ" -lt 12 ] && log "WARNING: CUDA $VER < 12: untested (developed with 12.4)"

# ---- host compiler: the first one this nvcc accepts AND that compiles C++17 with OpenMP
cat > "$TMP/probe.cu" <<'EOF'
#include <cuda_runtime.h>
#include <atomic>
#include <map>
#include <memory>
#include <thread>
#include <vector>
template <class R> __global__ void k(R *x) { x[threadIdx.x] = (R)1; }
int main() { std::vector<double> v(4); std::atomic<bool> b{false}; double *d = nullptr; k<double><<<1, 1>>>(d); k<float><<<1, 1>>>((float *)d); return (int)v.size() + b; }
EOF
cat > "$TMP/omp.cpp" <<'EOF'
#include <omp.h>
#include <vector>
#include <map>
int main() { std::vector<int> v(8); _Pragma("omp parallel for") for (int i = 0; i < 8; ++i) v[i] = omp_get_thread_num(); return 0; }
EOF
hc=()
[ -n "${HOSTCXX:-}" ] && hc+=("$HOSTCXX")
hc+=(g++ g++-14 g++-13 g++-12 g++-11 g++-10 g++-9 g++-8 g++-7)
# conda / pixi toolkits ship their own host compiler next to nvcc
for x in "$(dirname "$NV")/x86_64-conda-linux-gnu-g++" "$(dirname "$NV")/g++"; do [ -x "$x" ] && hc+=("$x"); done
hc+=(clang++)
HC=""; EXTRA=""
for c in "${hc[@]}"; do
  command -v "$c" > /dev/null 2>&1 || continue
  "$c" -std=c++17 -fopenmp "$TMP/omp.cpp" -o "$TMP/omp" > /dev/null 2>&1 || { log "host compiler $c: no C++17/OpenMP, skipped"; continue; }
  if "$NV" -ccbin "$c" -std=c++17 -arch=sm_$(echo "$ARCH" | awk '{print $NF}') -c "$TMP/probe.cu" -o "$TMP/probe.o" > "$TMP/err.txt" 2>&1; then
    HC=$c; break
  fi
  log "host compiler $c rejected by nvcc: $(grep -m1 -i 'error' "$TMP/err.txt" | sed 's/^.*[Ee]rror: *//' | cut -c1-160)"
done
if [ -z "$HC" ]; then
  # last resort: the first C++17/OpenMP compiler with -allow-unsupported-compiler (may still fail in the front end)
  for c in "${hc[@]}"; do
    command -v "$c" > /dev/null 2>&1 || continue
    "$c" -std=c++17 -fopenmp "$TMP/omp.cpp" -o "$TMP/omp" > /dev/null 2>&1 || continue
    if "$NV" -ccbin "$c" -allow-unsupported-compiler -std=c++17 -arch=sm_$(echo "$ARCH" | awk '{print $NF}') -c "$TMP/probe.cu" -o "$TMP/probe.o" > "$TMP/err.txt" 2>&1; then
      HC=$c; EXTRA=-allow-unsupported-compiler
      log "WARNING: using $c with -allow-unsupported-compiler (install a gcc that CUDA $VER supports if the build fails)"
      break
    fi
  done
fi
if [ -z "$HC" ]; then
  log "no host compiler accepted by $NV (CUDA $VER needs e.g. gcc <= 12 for 12.0-12.3, <= 13 for 12.4-12.6, <= 14 for 12.8+); set HOSTCXX=..."
  echo "NVCC :="
  echo "CUDA_VERSION := $VER"
  echo "DETECT_NOTE := no host compiler accepted by nvcc $VER"
  exit 0
fi
log "host compiler for nvcc: $HC ($("$HC" --version | head -1)) ${EXTRA}"
echo "NVCC := $NV"
echo "CUDA_VERSION := $VER"
echo "HOSTCXX := $HC"
echo "NVCC_AUTO := $EXTRA"
echo "DETECT_NOTE := ok"
