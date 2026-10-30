# Makefile -- rebar: zero-momentum R-bar integrands of 3-D O(N) phi^4, CPU and CUDA builds from one source.
#
#   make                 bin/rebar_cpu{,-e18} and, if a CUDA toolkit is found, bin/rebar_gpu{,-e18} (otherwise CPU only)
#   make cpu             bin/rebar_cpu and bin/rebar_cpu-e18 (C++17 + OpenMP; backends cpu and emu)
#   make gpu             bin/rebar_gpu and bin/rebar_gpu-e18 (CUDA; default --backend auto); fails without nvcc
#   make config          (re)detect nvcc, the CUDA version and a host compiler nvcc accepts -> build/config.mk
#   make info            print the configuration that will be used
#   make gpu-strict      as gpu, with host warnings and all nvcc warnings as errors
#   make cuda-resources  ptxas registers / stack / spills of the kernels for each CUDA_ARCH and limit variant
#   make asan | perturb  sanitizer build; build with exp/log moved by <= 2 ulp (calibrates CPU-vs-GPU tolerances)
#   make clean
#
# Limit variants (compile-time sizes of the per-thread arrays; the driver picks the binary it needs by itself):
#   e16  (bin/rebar_cpu, bin/rebar_gpu)          RB_MAXE 16, RB_MAXLOOP 9, RB_MAXATOMS 768: every order up to 8-loop
#        Gamma4 (b 8) and 8-loop Gamma2 (h 8, c 8); the v0.3.0 limits, fastest kernel
#   e18  (bin/rebar_cpu-e18, bin/rebar_gpu-e18)  RB_MAXE 18, RB_MAXLOOP 10, RB_MAXATOMS 4096: 9-loop Gamma2 (17 lines;
#        h needs atoms of L+1 = 10 loops) and 9-loop Gamma4 (18 lines).  An e16 binary that is given a list it cannot
#        hold re-executes itself as <its own path>-e18 (same arguments), so `bin/rebar_gpu run ...` works for every order.
#   Numbers do not depend on the variant (the limits only size arrays): e16 and e18 CPU runs are bitwise identical.
#
# Variables (command line or environment):
#   CUDA_ARCH   SASS targets (default 90 = H100/H200; e.g. "80 90" adds A100). PTX of the last one is embedded too.
#   NVCC        path to nvcc               (default: detected: PATH, $CUDA_HOME, /usr/local/cuda, /usr/local/cuda-12.*)
#   HOSTCXX     host compiler for nvcc and for the .cpp files of the GPU build (default: detected, the first of
#               g++, g++-14 ... g++-7 that nvcc accepts; CUDA 12.0-12.3 need gcc <= 12, 12.4-12.6 gcc <= 13)
#   VARIANTS    limit variants built by cpu / gpu (default "e16 e18"; VARIANTS=e16 builds only the 16-line one;
#               e20 = 20 lines, 11 loops, 16384 atoms: 10-loop Gamma2 and Gamma4, built with VARIANTS=e20;
#               e22 = 22 lines, 12 loops, 32768 atoms: 11-loop Gamma2, built with VARIANTS=e22)
#   MINBLOCKS   __launch_bounds__ minimum blocks per SM of the kernel (register cap; default 8 since v0.4.0:
#               +30 % on the H200 at 5 loops; v0.3.0 had 1).  DEFS=-DRB_MINBLOCKS=k overrides it as well.
#   FMAD        nvcc --fmad (FMA contraction on the device; default false since v0.4.0: same speed on the H200 and
#               ~5x closer to the CPU; v0.3.0 had true).  NVCC_EXTRA=-fmad=true overrides it as well.
#               The v0.3.0 device build is:  make gpu TAG=v030 MINBLOCKS=1 FMAD=true  -> bin/rebar_gpu-v030{,-e18}
#   DEFS        -D options for EVERY compile (host and device), e.g. DEFS=-DRB_MINBLOCKS=6.  Compile-time limits
#               (RB_MAXE, RB_MAXLOOP, ...) must be identical on host and device: pass them here, never only to nvcc,
#               and only with VARIANTS=e16 (the e18 variant sets its own).
#   NVCC_EXTRA  extra nvcc-only flags; -D options found here are forwarded to DEFS, -fmad=X to FMAD
#   TAG         build variant: objects in build/gpu-TAG, binaries bin/rebar_gpu-TAG{,-e18} (e.g. TAG=mb6 MINBLOCKS=6)
#   CXX, CXXFLAGS  CPU build compiler / flags (default g++, -O3 -march=native); LDFLAGS extra link options (-static)
SHELL := /bin/bash
CXX      ?= g++
CXXFLAGS ?= -O3 -march=native
CUDA_ARCH ?= 90
DEFS     ?=
NVCC_EXTRA ?=
TAG      ?=
VARIANTS ?= e16 e18
MINBLOCKS ?= 8
FMAD     ?= false
LDFLAGS  ?=
# The version is releases/CURRENT's (the source's own when the file is absent), stamped with the build of
# tools/build-info (empty outside a git checkout); both are passed to every compile as -D options.
RELEASE   := $(shell sed -n 1p ../releases/CURRENT 2>/dev/null)
BUILDINFO := $(shell bash ../tools/build-info --root .. 2>/dev/null)
VERDEFS   := $(if $(RELEASE),-DRB_VERSION='"$(RELEASE)"',) $(if $(BUILDINFO),-DRB_BUILD_INFO='"$(BUILDINFO)"',)
override DEFS += $(filter -D%,$(NVCC_EXTRA))
FMAD_X   := $(patsubst -fmad=%,%,$(filter -fmad=%,$(NVCC_EXTRA)))
FMAD_EFF := $(if $(FMAD_X),$(lastword $(FMAD_X)),$(FMAD))
NVCC_OWN := $(filter-out -D% -fmad=%,$(NVCC_EXTRA))
MBDEF    := $(if $(filter -DRB_MINBLOCKS=%,$(DEFS)),,-DRB_MINBLOCKS=$(MINBLOCKS))
# limit variants: extra -D options and binary / object-directory suffixes
VS_e16   :=
VS_e18   := -e18
VS_e20   := -e20
VS_e22   := -e22
VLIM_e16  :=
VLIM_e18  := -DRB_MAXE=18 -DRB_MAXLOOP=10 -DRB_MAXATOMS=4096
VLIM_e20  := -DRB_MAXE=20 -DRB_MAXLOOP=11 -DRB_MAXATOMS=16384
VLIM_e22  := -DRB_MAXE=22 -DRB_MAXLOOP=12 -DRB_MAXATOMS=32768
VDEFS_e16 :=
VDEFS_e18 := $(VLIM_e18) -DRB_VARIANT='"e18"'
VDEFS_e20 := $(VLIM_e20) -DRB_VARIANT='"e20"'
VDEFS_e22 := $(VLIM_e22) -DRB_VARIANT='"e22"'
# -ffp-contract=off: no a*b+c -> fma contraction on the host, so that CPU results do not depend on -march (GCC
# contracts by default even in ISO C++ mode).  The device has --fmad=$(FMAD) (default false, see above).
FPFLAGS   = -ffp-contract=off
WARN      = -Wall -Wextra -Wno-unused-parameter
CPUFLAGS  = $(CXXFLAGS) $(FPFLAGS) -std=c++17 -fopenmp $(WARN) -Iinclude -Isrc $(DEFS) $(VERDEFS)
SRC  = src/graph.cpp src/build.cpp src/cache.cpp src/main.cpp src/backend.cpp src/device_emu.cpp
HDR  = $(wildcard include/rbar/*.h) $(wildcard src/*.h)

# ---- CUDA configuration (detected once, cached in build/config.mk; 'make config' redoes it)
CUDA_GOALS := all gpu gpu-strict cuda-resources info config
ifeq ($(MAKECMDGOALS),)
NEED_CUDA := 1
else
NEED_CUDA := $(filter $(CUDA_GOALS),$(MAKECMDGOALS))
endif
ifneq ($(NEED_CUDA),)
ifeq ($(filter config,$(MAKECMDGOALS)),)
-include build/config.mk
endif
endif

all: cpu
	@if [ -n "$(NVCC)" ]; then $(MAKE) --no-print-directory gpu; \
	 else echo "NOTE: no usable CUDA toolkit ($(DETECT_NOTE)); built the CPU binaries only. Run 'make config NVCC=/path/to/nvcc [HOSTCXX=g++-12]', then 'make gpu'."; fi

build/config.mk: tools/detect_cuda.sh
	@mkdir -p build
	@NVCC="$(NVCC)" HOSTCXX="$(HOSTCXX)" CUDA_ARCH="$(CUDA_ARCH)" bash tools/detect_cuda.sh > $@.tmp && mv $@.tmp $@

config:
	@mkdir -p build
	@NVCC="$(NVCC)" HOSTCXX="$(HOSTCXX)" CUDA_ARCH="$(CUDA_ARCH)" bash tools/detect_cuda.sh > build/config.mk.tmp && mv build/config.mk.tmp build/config.mk
	@cat build/config.mk

info:
	@echo "CPU:  CXX=$(CXX) ($$($(CXX) --version 2>/dev/null | head -1))  CXXFLAGS=$(CXXFLAGS) $(FPFLAGS)"
	@echo "CUDA: NVCC=$(NVCC)  CUDA_VERSION=$(CUDA_VERSION)  HOSTCXX=$(HOSTCXX)  CUDA_ARCH=$(CUDA_ARCH)  NVCC_AUTO=$(NVCC_AUTO)  ($(DETECT_NOTE))"
	@echo "DEFS=$(DEFS)  NVCC_EXTRA=$(NVCC_OWN)  TAG=$(TAG)  VARIANTS=$(VARIANTS)  MINBLOCKS=$(if $(MBDEF),$(MINBLOCKS),(DEFS))  FMAD=$(FMAD_EFF)"

# ------------------------------------------------------------------------------------------------- CPU
cpu: $(foreach v,$(VARIANTS),bin/rebar_cpu$(VS_$(v)))

define CPU_VARIANT
CPU_OBJ_$(1) = $$(patsubst src/%.cpp,build/cpu$$(VS_$(1))/%.o,$$(SRC) src/device_cuda_stub.cpp)
bin/rebar_cpu$$(VS_$(1)): $$(CPU_OBJ_$(1))
	@mkdir -p bin
	$$(CXX) $$(CPUFLAGS) -o $$@ $$(CPU_OBJ_$(1)) -fopenmp $$(LDFLAGS)
build/cpu$$(VS_$(1))/%.o: src/%.cpp $$(HDR) Makefile
	@mkdir -p build/cpu$$(VS_$(1))
	$$(CXX) $$(CPUFLAGS) -DRB_PROGRAM='"rebar_cpu$$(VS_$(1))"' $$(VDEFS_$(1)) -c $$< -o $$@
endef
$(foreach v,e16 e18 e20 e22,$(eval $(call CPU_VARIANT,$(v))))

bin/rebar_cpu_asan: $(SRC) src/device_cuda_stub.cpp $(HDR)
	@mkdir -p bin
	$(CXX) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined $(FPFLAGS) -std=c++17 -fopenmp $(WARN) -Iinclude -Isrc $(DEFS) -o $@ $(SRC) src/device_cuda_stub.cpp -fopenmp
asan: bin/rebar_cpu_asan

bin/rebar_cpu_perturb: $(SRC) src/device_cuda_stub.cpp $(HDR)
	@mkdir -p bin
	$(CXX) $(CPUFLAGS) -DRB_LIBM_PERTURB=2 -DRB_PROGRAM='"rebar_cpu_perturb"' -o $@ $(SRC) src/device_cuda_stub.cpp -fopenmp
perturb: bin/rebar_cpu_perturb

# ------------------------------------------------------------------------------------------------ CUDA
HOSTFLAGS  ?= -O3
SUFFIX      = $(if $(TAG),-$(TAG),)
GENCODE     = $(foreach a,$(CUDA_ARCH),-gencode arch=compute_$(a),code=sm_$(a)) \
              -gencode arch=compute_$(lastword $(CUDA_ARCH)),code=compute_$(lastword $(CUDA_ARCH))
FMADDEF     = -DRB_FMAD=$(if $(filter false 0,$(FMAD_EFF)),0,1)
GPUDEFS_COMMON = -DRB_DEFAULT_BACKEND='"auto"' $(MBDEF) $(FMADDEF) $(DEFS) $(VERDEFS)
NVBASE      = -O3 -std=c++17 -lineinfo -ccbin $(HOSTCXX) $(NVCC_AUTO) -fmad=$(FMAD_EFF) -Iinclude -Isrc -Xcompiler $(FPFLAGS) $(NVCC_OWN)
comma := ,
empty :=
space := $(empty) $(empty)
GPU_STRICT ?=
ifeq ($(GPU_STRICT),1)
  HOSTWARN  = $(WARN) -Werror
  NVWARN    = --Werror all-warnings -Xcompiler $(subst $(space),$(comma),$(WARN) -Werror)
else
  HOSTWARN  = $(WARN)
  NVWARN    = -Xcompiler $(subst $(space),$(comma),$(WARN))
endif

gpu: $(foreach v,$(VARIANTS),bin/rebar_gpu$(SUFFIX)$(VS_$(v)))

# 'config' must run in its own invocation: when it is among the goals, build/config.mk is not read (make config gpu
# would see no HOSTCXX), hence the second test
check-nvcc:
	@if [ -z "$(NVCC)" ]; then echo "error: no usable nvcc ($(DETECT_NOTE)); run 'make config NVCC=/path/to/nvcc [HOSTCXX=g++-12]', then 'make gpu'"; exit 1; fi
	@if [ -z "$(HOSTCXX)" ]; then echo "error: no host compiler for $(NVCC) configured; run 'make config' first (in its own make invocation, with HOSTCXX=... if needed), then 'make gpu'"; exit 1; fi

gpu-strict:
	rm -rf $(foreach v,$(VARIANTS),build/gpu$(SUFFIX)$(VS_$(v)) bin/rebar_gpu$(SUFFIX)$(VS_$(v)))
	$(MAKE) gpu GPU_STRICT=1

define GPU_VARIANT
GPUDIR_$(1)  = build/gpu$$(SUFFIX)$$(VS_$(1))
GPUBIN_$(1)  = bin/rebar_gpu$$(SUFFIX)$$(VS_$(1))
GPUDEFS_$(1) = -DRB_PROGRAM='"rebar_gpu$$(SUFFIX)$$(VS_$(1))"' $$(GPUDEFS_COMMON) $$(VDEFS_$(1))
NVFLAGS_$(1) = $$(NVBASE) $$(GENCODE) $$(GPUDEFS_$(1))
GPU_OBJ_$(1) = $$(addprefix $$(GPUDIR_$(1))/,graph.o build.o cache.o main.o backend.o device_emu.o device_cuda.o)
$$(GPUBIN_$(1)): $$(GPU_OBJ_$(1)) | check-nvcc
	@mkdir -p bin
	$$(NVCC) $$(NVFLAGS_$(1)) -Xcompiler -fopenmp -o $$@ $$(GPU_OBJ_$(1))
$$(GPUDIR_$(1))/%.o: src/%.cpp $$(HDR) Makefile | check-nvcc
	@mkdir -p $$(GPUDIR_$(1))
	$$(HOSTCXX) $$(HOSTFLAGS) $$(FPFLAGS) -std=c++17 -fopenmp $$(HOSTWARN) -Iinclude -Isrc $$(GPUDEFS_$(1)) -c $$< -o $$@
# the emulator is compiled by nvcc (host pass of CUDA source): same front end / preprocessing as the kernel
$$(GPUDIR_$(1))/device_emu.o: src/device_emu.cpp $$(HDR) Makefile | check-nvcc
	@mkdir -p $$(GPUDIR_$(1))
	$$(NVCC) $$(NVFLAGS_$(1)) $$(NVWARN) -Xcompiler -fopenmp -x cu -c $$< -o $$@
$$(GPUDIR_$(1))/device_cuda.o: cuda/device_cuda.cu $$(HDR) Makefile | check-nvcc
	@mkdir -p $$(GPUDIR_$(1))
	$$(NVCC) $$(NVFLAGS_$(1)) $$(NVWARN) -Xcompiler -fopenmp -c $$< -o $$@
endef
$(foreach v,e16 e18 e20 e22,$(eval $(call GPU_VARIANT,$(v))))

cuda-resources: check-nvcc
	@for a in $(CUDA_ARCH); do for v in $(VARIANTS); do \
	  case $$v in e18) vd="$(VLIM_e18)" ;; e20) vd="$(VLIM_e20)" ;; e22) vd="$(VLIM_e22)" ;; *) vd="" ;; esac; \
	  echo "== sm_$$a $$v (MINBLOCKS $(if $(MBDEF),$(MINBLOCKS),from DEFS), fmad $(FMAD_EFF))"; \
	  $(NVCC) -O3 -std=c++17 -ccbin $(HOSTCXX) $(NVCC_AUTO) -fmad=$(FMAD_EFF) -Iinclude -Isrc $(MBDEF) $(DEFS) $$vd $(NVCC_OWN) -arch=sm_$$a -Xptxas -v,-warn-spills -c cuda/device_cuda.cu -o /dev/null 2>&1 \
	    | grep -E "Compiling|registers|stack|spill" ; done; done

clean-gpu:
	rm -rf build/gpu build/gpu-* bin/rebar_gpu bin/rebar_gpu-*

clean:
	rm -rf bin build/cpu build/cpu-* build/gpu build/gpu-*

distclean: clean
	rm -rf build

.PHONY: all cpu gpu gpu-strict asan perturb cuda-resources clean clean-gpu distclean config info check-nvcc
