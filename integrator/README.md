# integrator — R̄ (Callan–Symanzik route) integrands of 3-D O(N) φ⁴ on NVIDIA GPUs

`rebar` computes, graph by graph and order by order, the zero-momentum $\bar R$-subtracted integrals whose
per-order sums give the RG functions $\beta$, $\eta$, $\eta_{\phi^2}$ and $\nu$ of the 3-D $O(N)$ $\phi^4$ model in the massive zero-momentum scheme.
Every graph is one parametric integrand, sampled by tropical (Hepp-sector) Monte Carlo, on CPUs (OpenMP) or on any
number of NVIDIA GPUs from the same source code. It was validated on a CPU against exact values and the Phi4tools
series, and on GPUs against the CPU on identical random numbers (checks N2 and N4 of
[`../checks/`](../checks/README.md)); every run of the ledger ([`../ledger/`](../ledger/)) was made with it, from
version 0.3.0 to 0.4.3. `run_all.sh` builds it, checks it against the CPU and benchmarks it on a CUDA machine in one
time-boxed session, and collects the results in one tarball.

## Quick start

    cd integrator
    ./run_all.sh                       # ~60 min: build, self-tests, validation, benchmarks, production runs

It names the results tarball at the end: `results-<UTC date>.tar.gz` (a few MB). It runs unattended for
the whole budget: on a remote machine start it in `tmux` / `screen`, or as `nohup ./run_all.sh > run_all.log 2>&1 &`.
Ctrl-C (or `kill`) stops it within seconds and still writes the tarball. If the copy lost the execute bit, use
`bash run_all.sh`.

Variants:

    NVCC=/usr/local/cuda-12.4/bin/nvcc ./run_all.sh        # if nvcc is not found automatically
    HOSTCXX=g++-12 ./run_all.sh                             # if the default g++ is too new for nvcc
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_all.sh               # use only some GPUs (default: all)
    ./run_all.sh --budget 180 --with-7loops                 # 3 hours, including whole 7-loop orders of b, h, c
    ./run_all.sh --budget 30 --stages env,build,cpu,gpu,validate   # self-tests and validation only

Since v0.4.0 `make` builds two limit variants of each binary, `rebar_gpu` (16 lines: every order up to 8 loops, the
fastest kernel) and `rebar_gpu-e18` (18 lines: 9-loop $\Gamma^{(2)}$, i.e. h 9 and c 9); `bin/rebar_gpu run ...` re-executes the
e18 binary by itself when the graphs need it, so both must be present. A `run` can be interrupted (time limit, SIGTERM,
SIGKILL) and continued exactly with `bin/rebar_gpu run --resume FILE.json --seconds T`, repeated until the JSON says
`"complete": true` (same bits as an uninterrupted run; no sample is counted twice or drawn twice): for a SLURM queue, give
each job `--seconds` a little below its time limit and resubmit with `--resume`. Details in `TECHNICAL.md` (Usage).
The tenth order needs the e20 variant (20 lines), and eleven-loop $\Gamma^{(2)}$ the e22 variant (22 lines): `make VARIANTS=e20`
builds `bin/rebar_cpu-e20` and `bin/rebar_gpu-e20`, which are called by name (the automatic re-execution goes only to e18).

Requirements: Linux, NVIDIA driver, CUDA toolkit 12.x (11.8 is the minimum for sm_90; developed with 12.4), a host C++
compiler with C++17 and OpenMP that this nvcc accepts (CUDA 12.0–12.3: gcc ≤ 12; 12.4–12.6: gcc ≤ 13; 12.8+: gcc ≤ 14;
the script finds one among g++, g++-14 … g++-7), GNU make, coreutils, python3 ≥ 3.6 (standard library only; without
it the raw data are still collected, only the verdicts and report are missing). No root, no network, no scheduler, no
Python packages. Disk: < 100 MB. The default target is sm_90 (H100/H200); `CUDA_ARCH="80 90"` adds A100.

### Which GPUs

Any NVIDIA GPU that CUDA 12 supports can run the integrator; AMD and Intel GPUs cannot, since the device code is CUDA.
Three things decide whether a given card works and how fast:

- **Build target.** The default build carries machine code for compute capability 9.0 (H100, H200) and PTX for the
  same capability. Newer GPUs run that PTX through the driver's just-in-time compiler, so a Blackwell card works with
  the default; older ones do not, and need `CUDA_ARCH` set to their capability at build time (80 for A100, 70 for V100,
  86 or 89 for GeForce RTX 30 and 40 series; several values are allowed, e.g. `CUDA_ARCH="80 90"`). A binary without
  code for the card fails with `kernel not loadable on this device` (Troubleshooting).
- **Portability.** The kernel uses plain IEEE double arithmetic without fused multiply-add, Philox counters and a
  per-thread stack; nothing in it is specific to one architecture, and the few `__CUDA_ARCH__` switches only separate
  host from device code. Every device reproduces the CPU to round-off on identical samples (check N2 of `../checks/`).
  The occupancy defaults (a register cap that fits eight blocks per multiprocessor) were tuned on the H200 and are
  reasonable on any recent part.
- **Speed.** Everything runs in double precision, so the FP64 rate of the card decides. Data-centre GPUs (P100, V100,
  A100, H100, H200) run doubles at half their single-precision rate; GeForce cards at 1/32 or 1/64 of it, so a GeForce
  RTX 4090 is roughly 25 times slower than an H200 per card: correct, and usable for low orders and tests, but not for
  production.

## What it computes

$d = 3$, unit renormalised mass $m_M = 1$, zero external momenta, Nickel integral normalisation (one $d^3k/\pi^2$ per loop),
$u = g/8\pi = 6\bar g/(N+8)$ ($\bar g$ = Phi4tools $\tilde u$, Nickel's normalised coupling $\hat g = (N+8)g/48\pi$ of thesis §2.1.3.3,
p. 30). The thesis/PARIS and `int4` integrals instead have $m_I = 1$ (thesis §2.3, p. 34). The Callan–Symanzik equation (the
thesis's RG equation of §2.2, p. 31, in the massive scheme) gives

$$
\eta = \frac{2h}{1+h},\qquad \beta_u = -u + 2\eta u + (2-\eta)\,\tilde b,\qquad \eta_{\phi^2} = -(2-\eta)\,c,\qquad \frac{1}{\nu} = (2-\eta)(1-c)
$$

with $\tilde b = m\,\Gamma_R^{(4,1)}(0)/8\pi$, $h = -m^2\,\partial\Gamma_R^{(2,1)}/\partial p^2\big|_0$, $c = m^2\,\Gamma_R^{(2,2)}(0)$, and

$$
\tilde b = \sum_L (-1)^{L+1} B_L\,u^{L+1},\qquad h = \sum_L (-1)^L H_L\,u^L,\qquad c = \sum_L (-1)^{L+1} C_L\,u^L
$$

$$
\begin{aligned}
B_L &= \sum_{G \in \Gamma^{(4)}(L)} S_G \sum_e \Gamma(L/2+1)\,\pi^{-L/2}\int_\Delta x_e\,\bar R_e\bigl[U^{-3/2}\bigr]\\
H_L &= \sum_{G \in \Gamma^{(2)}(L)} S_G \sum_e \Gamma(L/2+1)\,\pi^{-L/2}\int_\Delta x_e\,\bar R_e\bigl[W\,U^{-5/2}\bigr] && \text{(}W = U\text{ with the external vertices identified)}\\
C_L &= \sum_{G \in \Gamma^{(2)}(L)} S_G \sum_{(e,e')} \Gamma(L/2+1)\,\pi^{-L/2}\int_\Delta x_e x_{e'}\,\bar R_{ee'}\bigl[U^{-3/2}\bigr] && \text{(ordered pairs, }e = e'\text{ allowed)}
\end{aligned}
$$

(`--kind b`, `h`, `c`). $S_G = E!/\lvert\mathrm{Aut}\,G\rvert \times$ the $O(N)$ tensor factor (1 at $N = 1$; every class also carries its $O(N)$
polynomial, so one $N = 1$ run gives every $N$: `tools/recombine_on.py`). These are the thesis's weights and $O(N)$ factors
(thesis §3.3, pp. 49–54); summed over the full tadpole-free lists, one-vertex-reducible graphs included, they reproduce its
zero-dimensional sum rules (thesis §6.4.3.1, p. 107) through 7 loops (check G2 of [`../checks/`](../checks/README.md)). $\bar R$: zero-momentum
Zimmermann forests of 1PI subgraphs, pairwise nested or vertex-disjoint; 4-point parts without a dot $t^0$, 2-point parts without a dot $t^2$ (mass
and field), 2-point parts with one dot $t^0$. A $\phi^2$-insertion ("dot") on line $e$ is a factor $x_e$; all forest terms share
the simplex and the prefactor. Each graph's forest terms are expanded into products of Kirchhoff polynomials of
2-connected blocks and identical products are summed exactly, so all exactly cancelling forest pairs disappear
algebraically (one-vertex-reducible and tadpole graphs come out identically zero). Validated values ($N = 1$):
$B_1 = 3/4$, $B_2 = 1/2$, $H_2 = 1/81$, $C_1 = 1/4$, $C_2 = 1/12$; higher orders against Phi4tools (Sberveglieri & Spada,
arXiv:2311.12576) through the relations above (`src/refs.h`, `tools/refs.py`). Also available, for comparison:
`--kind int4` (Phi4tools' $\Gamma^{(4)}$ in the intermediate scheme; order sum $S_L = (2^L/3)\sum_G S_G I_G$, $S_5 = 400.1741$),
`bare4`, `bare2d`, `int2d`, `int2` (`int4`, `int2d` and `int2` are the inputs of route B).

Sampling: per graph a forest-aware tropical density (exponent per Hepp face = minimum over the surviving forest terms,
$+1$ on $t^2$-protected faces, $\times 0.7$), Borinsky's recursion, Philox4x32-10 counter-based random numbers with a key per
graph (seed, kind, L, Nickel index), FP64 evaluation in log space, FP64 compensated accumulation; per order a pilot,
then Neyman allocation over the graphs (by samples, or by measured device time with `--seconds`), then rounds that
stop before the time box. Results: JSON with every graph's integral, error and diagnostics, and the order sum, its
error, the reference and the pull.

## What run_all.sh does (default budget 60 min)

| stage | what | ≈ time |
|---|---|---|
| env | `environment.txt` (OS, CPU, memory, `nvidia-smi`, `-q`, topology, driver, nvcc, compilers, sha256 of all sources) | < 1 min |
| build | `make config` (detects nvcc, CUDA version, a host compiler nvcc accepts), `make cpu`, `make gpu`, ptxas resources | 1–3 min |
| cpu | host build: Philox known answers, GPU-emulation unit test, the golden counters on this CPU | 2–3 min |
| gpu | per device: FP64 exp/log/sqrt/fma, Philox and build signature on the device, 4096 samples per class vs the host, one chunk through the production kernel; then **GPU vs CPU on identical random numbers** for 13 golden configurations (b, h, c, int4 at 3–6 loops), forced splits over all GPUs, b 5 on every GPU alone | 4–8 min |
| validate | $B_2$, $H_2$, $C_2$ (exact) at high statistics; $B$, $H$, $C$ at 3, 4, 5 loops vs Phi4tools; 8-seed replica ensembles of b4, h4; $O(N)$ recombination; FP32 vs FP64 bare kernel | ≈ 13 min |
| tuning | rebuilds the v0.3.0 device build (`RB_MINBLOCKS=1`, `-fmad=true`), `RB_MINBLOCKS=6`, and `-fmad=true` at 8, against the v0.4.0 default (`RB_MINBLOCKS=8`, `-fmad=false`); golden b 5 and throughput b 5, c 5 for each | ≈ 7 min |
| throughput | samples/s per kernel (b, h, c, int4) and loop order (4, 5, 6), FP64 and FP32-evaluation for the bare kernels; c 6 with long launches | ≈ 8 min |
| scaling | b 5 on 1, 2, 4, … GPUs | ≈ 4 min |
| production | whole 4-, 5-, 6-loop orders of b, h, c, Neyman allocation by device time, against Phi4tools (6 loops: $B_6$, $H_6$, $C_6$ references exist) | ≈ 16 min |
| loops7 | only with `--with-7loops` or `--budget ≥ 150`: whole 7-loop orders of b (3150 classes), h, c | rest |

The times of validate … production are not fixed: what is left of the budget after the fixed stages is divided among
them by weight (validate 20, tuning 10, throughput 12, scaling 5, production 25, loops7 30), so a slow build or a slow
GPU shrinks every later stage instead of dropping the last ones; `--budget 120` simply gives every run twice as
long. Every step has a hard limit (SIGTERM, then SIGKILL; a process stuck in the driver is abandoned and the script
goes on), stages that no longer fit are logged as SKIPPED, and the tarball is written in every case, including
Ctrl-C. Progress is printed and logged in `SUMMARY.txt`. A test mode, `REBAR_TEST_EMU=1`, runs every GPU step on two
emulated devices on a machine without a GPU; it exercises the script, not the hardware, and is not for the GPU machine.

## The results tarball

`results-<date>.tar.gz` contains `SUMMARY.txt` (the console log), `VALIDATION.txt` (verdicts), `REPORT.md`
(tables), `environment.txt`, `nvidia-smi-q.txt`, `devices.json`, every run's JSON and every step's command line and
stdout/stderr (`logs/`). It describes the hardware and software (CPU and GPU models, memory, clocks, driver, CUDA and
compiler versions) but not the machine: no host name, user, job id, GPU UUID or serial, PCI bus id or absolute path
is recorded (paths are relative to the script's directory).

## How to read the verdicts (`VALIDATION.txt`, first table of `REPORT.md`)

| line | PASS means | if it FAILs |
|---|---|---|
| `logs/cpu_emu_*`, `gpubin_emu_*` | the GPU program, run on the host by the emulator, reproduces the CPU bit for bit (multi-device, faults, eviction, time box) | a build problem on this machine |
| `cpu_golden_*` | this machine's CPU reproduces the golden files within round-off (not bitwise: another glibc) | a compiler or libm problem |
| `selftest_*` | every GPU passed its start-up test (exp/log $\mathord{\le}\;2$ ulp, sqrt/fma exact, Philox, build signature, 4096 samples per class: same Hepp sector, $\lvert\Delta w\rvert \le 10^{-11}\sum\lvert\text{terms}\rvert$, one chunk of the production kernel) | that GPU is excluded from all runs; the reason is in the JSON |
| `golden_*` | GPU and CPU agree on identical random numbers: same sample counts, means within 100 units of double-precision round-off of the per-sample terms and $10^{-4}\,\sigma$; the numbers really came from the GPU(s) (no host fallback, no dead device; split runs used ≥ 2 GPUs) | the essential test: device code, compiler or hardware give different numbers |
| `val_*`, `prod_*` | the order sum agrees with the reference: $\lvert\text{pull}\rvert \le 3$ (WARN $\mathord{\le}\;4$, FAIL beyond), on the GPU(s), no non-finite weights | with $\sim 25$ such tests one WARN can occur by chance ($p \approx 0.003$ each); a FAIL, or several WARN, is a real problem |
| `replicas_*` | 8 independent seeds: pulls consistent with $N(0,1)$ and the per-graph scatter with the quoted errors $(\chi^2\; p > 10^{-3})$ | error bars are not trustworthy |
| FP32 lines | informational: FP32 evaluation is a benchmark of the bare kernel, not a production option | — |
| `devices: no usable CUDA device` / `no devices.json` | (FAIL) no GPU passed the probe, or no GPU binary was built: every GPU stage was skipped, only the host checks above ran | see Troubleshooting; the CUDA error is also printed on the console (`CUDA reports: …`) |
| `<step>: no JSON written` | (FAIL; WARN for throughput / scaling) a step ran but left no result: it crashed, was killed at its time limit or could not open a device | `logs/<step>.err`; the exit status is in `SUMMARY.txt` |
| `step(s) SKIPPED for lack of time` | (WARN) the global budget ran out before these steps could start: the run is incomplete | rerun with a larger `--budget` (the console warns early if CUDA start-up is slow) |

The last line of `VALIDATION.txt` (also printed at the end) sums up: `-> ALL PASSED`, `-> WARNINGS (see the WARN
lines)` or `-> FAILURES`.

Reference values used ($N = 1$, `src/refs.h`; Phi4tools' $O(N)$ series through the CS relations, errors propagated from
the quoted Phi4tools errors; the JSON's `order.ref`, `order.pull`):

| $L$ | $B_L$ (`b`) | $H_L$ (`h`) | $C_L$ (`c`) |
|---|---|---|---|
| 1 | 3/4 | – | 1/4 |
| 2 | 1/2 | 1/81 | 1/12 |
| 3 | 0.5986037059 | −0.001542750058 | 0.07785997192 |
| 4 | 0.9671926870 | 0.004699104918 | 0.1006770275 |
| 5 | 1.8965254(5) | 0.002443913(10) | 0.17055192(8) |
| 6 | 4.3006836(49) | 0.00802056(9) | 0.3459107(9) |
| 7 | 10.917594(45) | 0.0145496(8) | – |

`int4`: Phi4tools $\Gamma^{(4)}$ order sums $S_4 = 75.2084920642$, $S_5 = 400.1741436(18)$. Pull $\mathord{=}\;(\text{value} - \text{ref})/\sqrt{\text{err}^2 + \text{ref}_{\text{err}}^2}$.

Throughput, tuning and scaling are informational (tables in `REPORT.md`): device samples per second per kernel, loop
order and precision, the occupancy variants, the multi-GPU efficiency, and per-launch fill factors.

## Troubleshooting

- **`no usable nvcc`** (`config.mk`, `logs/build_config.err`): nvcc is not in PATH, `$CUDA_HOME`, `/usr/local/cuda`
  or `/usr/local/cuda-12.*`. Run `NVCC=/path/to/nvcc ./run_all.sh`. Check by hand with `make config && make info`.
- **Host compiler rejected** (`unsupported GNU version`): install a supported gcc (e.g. `g++-12`) and pass
  `HOSTCXX=g++-12`. `-allow-unsupported-compiler` is tried only as a last resort and usually fails later.
- **CUDA version**: < 11.8 cannot build sm_90 (the build is disabled with a message); 11.8 is untested; 12.x is expected.
- **`cudaErrorInsufficientDriver`** (`devices.json`, console line `CUDA reports: …`): the toolkit is newer than the
  driver — or there is no NVIDIA driver at all (no `libcuda.so`: this is what a machine without a GPU reports). Use a
  toolkit whose version is ≤ the "CUDA Version" printed by `nvidia-smi` (the CUDA runtime is linked statically, so
  only the driver matters at run time).
- **`kernel not loadable on this device`**: the binary has no code for this GPU; set `CUDA_ARCH` (90 = H100/H200,
  80 = A100, `"80 90"` both). PTX of the last architecture is embedded, so newer GPUs JIT-compile it.
- **No / slow FP64**: everything is FP64 (H200: FP32:FP64 = 2:1, full-rate FP64). On GPUs with crippled FP64
  (ratio 32 or 64, `single_to_double_perf_ratio` in `devices.json`) the results are correct but $\sim 30\mathord{\times}$ slower.
  `--fp32` exists only for the bare kinds and is not for production.
- **A GPU fails its self-test or dies mid-run**: it is removed, its work goes to the others (or the host); the JSON
  records why (`backend_info.devices[].why_dead`). `auto` never silently falls back to the CPU when GPUs exist.
- **Slow start-up** (console: `WARNING: CUDA start-up took … s`): every step is a new process that initialises CUDA
  on all visible GPUs; with GPU persistence mode off (`Persistence Mode` in `nvidia-smi-q.txt`) this can take seconds
  per GPU, and with ~100 GPU steps it eats into the later stages. Give a larger `--budget` (or ask the administrator
  for `nvidia-smi -pm 1`).
- **A hung GPU**: every wait on the device has a watchdog (≥ 300 s, scaled with the expected launch time); the
  process then exits without CUDA teardown, the script continues; `dmesg_xid.txt` has driver errors if readable.
- **No GPU found at all** (console `NO USABLE GPU`, `VALIDATION.txt`: `FAIL devices: …`): only the host stages ran
  (≈ 5 min, a 20 KB tarball whose host checks should all PASS). Check `nvidia-smi`, the driver and
  `CUDA_VISIBLE_DEVICES`; the CUDA error is in `devices.json` and on the console.
- **Shared machine / MIG**: other jobs only lower the throughput; with MIG, select instances via
  `CUDA_VISIBLE_DEVICES`. If the GPUs are in Exclusive-Process compute mode (`nvidia-smi -q | grep 'Compute Mode'`), a
  GPU held by another process should fail to open: it is excluded (`why_dead` in the JSON) and the `golden_*` lines
  then FAIL with "dead device"; leave busy GPUs out with `CUDA_VISIBLE_DEVICES`.
- **Running things by hand**: `make` (CPU + GPU), `bin/rebar_gpu devices`, `bin/rebar_gpu selftest --backend cuda
  --kind b --loops 5`, `bin/rebar_gpu run --backend cuda --kind b --loops 6 --list data/lists/gamma4_L6.txt --drop-1vr
  --seconds 3600 --out b6.json` (see `TECHNICAL.md`).

## Validation

The checks of [`../checks/`](../checks/README.md) verify the integrator: the forest expansion pointwise against an
independent 100-digit evaluation (I1), the vanishing graphs (I2), Whitney twins (I3), the class cache (I4), the GPUs
against the CPU on identical random numbers (N2), the order sums at two to seven loops against exact values and
Phi4tools (N3, N4), and the scatter of the replica ensembles at six and seven loops (N7). Version 0.4.0 was also
validated as follows:
- $B_1$ to $B_5$, $H_2$ to $H_5$, $C_1$ to $C_5$ at $N = 1$ against exact values and Phi4tools (all within $1.9\sigma$); $O(N)$ factors at $N = 0, 2, 3$;
  the exact sunset-in-ring value; the bare 4- and 5-loop orders $S_4 = 75.2085$, $S_5 = 400.1741$; CPU runs at
  6 loops: $B_6 = 4.3012(33)$ ($+0.15\sigma$), $H_6 = 0.008015(11)$ ($-0.48\sigma$), $C_6 = 0.3467(34)$ ($+0.24\sigma$; from the `run_all.sh` test mode on emulated devices).
- The integrand: pointwise against an independent Python implementation and an independent 50-digit implementation
  (3–6 loops); exact pruning; vanishing of one-vertex-reducible and tadpole graphs; Whitney twins.
- Error bars: Gaussian pulls over 340 replica runs at 2–4 loops and over 400 replicas at 3–4 loops
  (mean pulls within $\pm 0.1$, sd 0.91–1.06); kurtosis-based error of the error in every JSON.
- The $\bar R$ route needs $32\mathord{\times}$ / $60\mathord{\times}$ fewer samples than the bare route for $B_4$ / $B_5$ ($14\mathord{\times}$ / $22\mathord{\times}$ at
  equal instruction count), $24\mathord{\times}$ / $76\mathord{\times}$ for $H_4$ / $H_5$.
- The GPU code path through the emulator: bit-for-bit agreement with the CPU for any number of devices, blocks,
  launch sizes, lane order, device failures (also a device reading a zeroed class view), evictions, time boxes.
- Builds: nvcc 12.4 for sm_80/sm_90, host gcc 7.5 and 12.4, warning-free with `-Werror`; ptxas: 110 registers,
  1104 B stack, 0 spills for the FP64 kernel.

Per-graph $\bar R$ values have no external reference (only the order sums do). The softening $s = 0.7$ and the $+1$
face shift were tuned at 4–5 loops.

Files: `run_all.sh`; `Makefile` + `tools/detect_cuda.sh`; `include/rbar/` (single-source kernel), `src/` (host:
graphs, forests, driver, backends), `cuda/device_cuda.cu`; `tools/` (validate, report, golden check, recombination,
…); `data/lists/` (graph lists with provenance); `../results/checks/golden-cpu/` (CPU references on fixed counters);
`TECHNICAL.md` (algorithm, options, JSON).
