# rebar — zero-momentum R̄ (CS-route) integrands for 3-D O(N) φ⁴, single source for CPU and GPU

`rebar` is the integrator of ReBar: the Callan–Symanzik route to the RG functions of the 3-D $O(N)$ $\phi^4$ model in the
massive zero-momentum scheme (Parisi; Nickel–Meiron–Baker; PARIS; Phi4tools), with every graph integrated
as one BPHZ-subtracted ($\bar R$) parametric integrand by tropical (Hepp-sector) Monte Carlo.

## Physics conventions

$d = 3$, unit renormalised mass $m_M = 1$, zero external momenta, Nickel integral normalisation (one $d^3k/\pi^2$ per loop).
$u = g/8\pi = 6\bar g/(N+8)$ ($\bar g$ = Phi4tools $\tilde u$). The thesis's "Nickel's normalization" (thesis §2.1.3.3, p. 30) is instead
the coupling, $\hat g = (N+8)g/48\pi = \bar g$; `int4`, `int2d` and the thesis/PARIS integrals have $m_I = 1$ (thesis §2.3, p. 34).
With the CS equation:

$$
\eta = \frac{2h}{1+h},\qquad \beta_u = -u + 2\eta u + (2-\eta)\,\tilde b,\qquad \eta_{\phi^2} = -(2-\eta)\,c,\qquad \frac{1}{\nu} = (2-\eta)(1-c)
$$

$$
\tilde b = \sum_L (-1)^{L+1} B_L\,u^{L+1},\qquad h = \sum_L (-1)^L H_L\,u^L,\qquad c = \sum_L (-1)^{L+1} C_L\,u^L
$$

$$
\begin{aligned}
B_L &= \sum_{G \in \Gamma^{(4)}(L)} S_G \sum_e \Gamma(L/2+1)\,\pi^{-L/2}\int_\Delta x_e\,\bar R_e\bigl[U^{-3/2}\bigr]\\
H_L &= \sum_{G \in \Gamma^{(2)}(L)} S_G \sum_e \Gamma(L/2+1)\,\pi^{-L/2}\int_\Delta x_e\,\bar R_e\bigl[W\,U^{-5/2}\bigr] && \text{(}W = U\text{ with the two external vertices identified)}\\
C_L &= \sum_{G \in \Gamma^{(2)}(L)} S_G \sum_{(e,e')} \Gamma(L/2+1)\,\pi^{-L/2}\int_\Delta x_e x_{e'}\,\bar R_{ee'}\bigl[U^{-3/2}\bigr] && \text{(ordered pairs, }e = e'\text{ allowed)}
\end{aligned}
$$

$S_G = E!/\lvert\mathrm{Aut}\,G\rvert \times$ ($O(N)$ tensor factor; 1 at $N = 1$). $S_G$ equals GAU's `globalSymmetryFactor` on all 251 GAU
graphs to 5 loops, and the $O(N)$ factor (a sum over the $3^V$ pairings at the vertices) equals GAU's symbolic polynomials on all 47
graphs of GAU's four-loop tables (`../results/checks/gau/beta.specs`, check G4 of `../checks/`). $\bar R$: zero-momentum Zimmermann forests of 1PI subgraphs,
elements pairwise nested or **vertex-disjoint** (Lagrangian counterterms); subtraction degrees:
4-point without dot $t^0$; 2-point without dot $t^2$ (mass + field); 2-point with one dot $t^0$; otherwise none.
(The three types implement the four renormalisation conditions of the massive scheme, eq. (2.6) of thesis §2.1.3.1, p. 29:
$t^2$ on 2-point parts without a dot carries both the mass and the field condition.)
All forest terms share the simplex and the prefactor; $t^0$ contracts the subgraph ($U \to U_\gamma U_{G/\gamma}$); $t^2$ keeps the
first order in $\varepsilon_\gamma = W_\gamma/U_\gamma$ of the subgraph replaced by an effective massless line $(U = U_{\text{contracted}} + \varepsilon\,U_{\text{split}})$.

`rebar_cpu run` prints/stores the raw per-order sums (`order.sum` = $B_L$, $H_L$, $C_L$) and the signed series
coefficients (`order.series_coeff`). Reference values ($N = 0\ldots3$) come from the Phi4tools $O(N)$ series
(arXiv:2311.12576 app. D) via the relations above (`tools/refs.py`, `src/refs.h`).

Other kinds, for validation: `bare4` ($I_G = \Gamma(L/2)\,\pi^{-L/2}\int U^{-3/2}$), `int4` (Phi4tools/PARIS intermediate scheme:
sunsets $t^0$-subtracted; Phi4tools order sum $S_L = (2^L/3)\sum_G S_G I_G$), `bare2d`, `int2d` ($\Gamma^{(2)\prime}$, sunset-subtracted), `int2`
($\Gamma_I^{(2)}(0)$, sunset-subtracted, from three loops; with `int4` and `int2d` the inputs of route B).
(Intermediate scheme as in thesis §2.1.2, p. 28, and eq. (2.8), p. 29. Thesis §6.3.2.1, p. 102, gives an exact identity between
$\Gamma_I^{(2)}(0)$ and $\Gamma_I^{(2,1)}(0;0)$, whose integrals are $\Gamma^{(4)}$-type ones at $p = 0$, thesis §3.2.3.2, p. 47.)

## Algorithm

1. **Graphs**: Nickel lists (`--list FILE`, any tab-separated file with the Nickel index in some column, e.g. the
   Phi4tools `int4ptNN.txt`) or the built-in generator (`--gen`; reproduces the Phi4tools lists string for string
   at 4–5 loops $\Gamma^{(4)}$ and 6 loops $\Gamma^{(2)}$). The historical GAU lists used by PARIS ($\Gamma^{(4)}$ to 7 loops, $\Gamma^{(2)}$ to 8,
   with PARIS labels, weights and 1VR flags) are tabulated by `../graphs/tools/gaugraph.py lists` (`gau_lists.tsv`, from
   `../graphs/lists/`) and equal Phi4tools as sets (for $\Gamma^{(2)}$, their self-loop-free rows). To use them,
   select rows (e.g. `grep -P '^7-loop-4\t'`) and pass them with
   `--list`; the Nickel index is the first field containing `|`. PARIS graph numbers `N<k>` follow GAU's list order and are
   not Nickel indices (the thesis's generator, thesis §3.2.1, pp. 40–44, uses a lexicographically minimal adjacency matrix
   as its canonical form). Thesis Table 3.1, p. 44, gives the counts through 8 loops: 1622 $\Gamma^{(2)}$, 26540 $\Gamma^{(4)}$.
   `data/lists/` holds them (`gamma2_L8.txt`, `gamma4_L8.txt`), the 9-loop $\Gamma^{(2)}$ list (`gamma2_L9.txt`, 9184 graphs)
   and the lists of the higher orders (`gamma4_L9.txt`, `gamma2_L10.txt`, `gamma4_L10.txt.xz`), made with the
   generator `topologies` of `../graphs` (`tools/lists/`, provenance and checks in `data/lists/README.md`). The 9-loop
   $\Gamma^{(2)}$ graphs have 17 lines, one more than `RB_MAXE` = 16, and h needs atoms of $L + 1 = 10$ loops (`RB_MAXLOOP` = 9).
2. **Forest expansion (host)**: renormalisation parts, Lagrangian forests, $t^2$ jets; every forest term expanded into
   products of Kirchhoff polynomials of 2-connected blocks of minors ("atoms"), with exact dyadic coefficients;
   identical products summed → all exact cancellations removed algebraically (one-vertex-reducible and tadpole
   graphs come out with no terms and are dropped).
3. **Density**: $\omega(S) = \lvert S\rvert +{}$ min over surviving terms of their tropical degree on $S$ (exact: nullities of each atom
   restricted to $S$, plus dots in $S$); $+1$ on faces where that minimum is $\mathord{\le}\;0$ ($t^2$-protected faces; checked);
   then softened, $\omega \to s\cdot\omega$ with $s = 0.7$ by default (`--omega-scale`): halves the per-sample sd for both $\bar R$ and
   bare integrands at 4–5 loops. Borinsky's recursion $J(S) = \sum_{e\in S} J(S\setminus e)/\omega(S\setminus e)$ gives the
   sampler and the normalisation $J(E)$.
4. **Kernel (single source, `include/rbar/`)**: Philox4x32-10 → (sector, point) → atoms by Kruskal + rescaled
   loop matrix + $LDL^T$ → grouped terms (Neumaier) → weight $w$ with $E[w] = \int_\Delta f$. FP64; FP64 accumulation with
   compensated sums.
5. **Driver**: per class (= graph, or Whitney-twin class with `--dedup`) independent Philox streams; pilot →
   Neyman allocation (`--alloc samples` $\mathord{\propto}\;\lvert S_G\rvert\,\text{sd}$, or `--alloc time` $\mathord{\propto}\;\lvert S_G\rvert\,\text{sd}/\sqrt{\text{cost}}$ with `--seconds`) → main
   phase in checkpointed rounds that stop before the time box; JSON written after every round.

## Build

    make                          # bin/rebar_cpu{,-e18} and, if a CUDA toolkit is found, bin/rebar_gpu{,-e18}
    make cpu                      # CPU only: bin/rebar_cpu and bin/rebar_cpu-e18 (C++17, OpenMP); backends cpu (default) and emu
    make gpu                      # CUDA: bin/rebar_gpu and bin/rebar_gpu-e18 (default --backend auto: all usable GPUs)
    make config && make info      # (re)detect nvcc, CUDA version and a host compiler nvcc accepts (build/config.mk)
    make gpu CUDA_ARCH="80 90"    # SASS for A100 and H100/H200 (+ PTX of the last for newer GPUs); default "90"
    make config NVCC=/usr/local/cuda-12.4/bin/nvcc HOSTCXX=g++-12 && make gpu   # explicit toolkit / host compiler
    make gpu TAG=v030 MINBLOCKS=1 FMAD=true    # the v0.3.0 device build, as bin/rebar_gpu-v030{,-e18} (comparison)
    make gpu TAG=mb6 MINBLOCKS=6 VARIANTS=e16  # another register cap, 16-line variant only: bin/rebar_gpu-mb6
    make gpu-strict               # host -Werror and nvcc --Werror all-warnings
    make cuda-resources           # ptxas registers / stack / spills of every kernel, both limit variants
    make asan | perturb           # sanitizer build; build with exp/log moved by <= 2 ulp (calibrates CPU-vs-GPU tolerances)

**Limit variants (v0.4.0).** The compile-time limits of `include/rbar/config.h` (`RB_MAXE` lines, `RB_MAXLOOP` loops of an
atom, `RB_MAXV` vertices of an atom, `RB_MAXATOMS` atoms per class) size per-thread arrays of the kernel, so they cost
registers and stack. Two variants are built from the same sources: **e16** (`bin/rebar_cpu`, `bin/rebar_gpu`: the v0.3.0
limits 16 / 9 / 12 / 768, the fastest kernel; holds every order up to 8-loop $\Gamma^{(4)}$ and $\Gamma^{(2)}$) and **e18**
(`bin/rebar_cpu-e18`, `bin/rebar_gpu-e18`: 18 / 10 / 12 / 4096, needed by 9-loop $\Gamma^{(2)}$, whose graphs have 17 lines and, for h,
atoms of 10 loops, and by 9-loop $\Gamma^{(4)}$). The driver chooses by itself: every command first checks what its graphs need
(lines, atom loops = $L + 1$ for the derivative kinds h, `bare2d`, `int2d`, vertices); an e16 binary that cannot hold them
re-executes `<its own path>-e18` with the same arguments (`REBAR_NO_REEXEC=1` disables this, and then it is an error), and so
does the builder if a class turns out to need more atoms than `RB_MAXATOMS`. The numbers do not depend on the variant (the
limits only size arrays): e16 and e18 CPU runs are bitwise identical, and the golden regression
is run through both. `version` and the run JSON (`limits`) say which variant produced a result; the build signature differs
between the variants (it includes the limits), so a mismatched host / device pair is still caught by the device self-test.
Two more variants are built only on request: **e20** (`make VARIANTS=e20`: 20 / 11 / 12 / 16384, 10-loop
$\Gamma^{(2)}$ and $\Gamma^{(4)}$) and **e22** (`make VARIANTS=e22`: 22 / 12 / 12 / 32768, 11-loop
$\Gamma^{(2)}$). They are called by name: the automatic re-execution goes only from e16 to e18.

`tools/detect_cuda.sh` looks for nvcc in `$NVCC`, `PATH`, `$CUDA_HOME`, `$CUDA_PATH`, `/usr/local/cuda`, `/usr/local/cuda-12.*`
and the NVIDIA HPC SDK, requires CUDA ≥ 11.8 (sm_90; developed with 12.4), and picks the first host compiler among
`$HOSTCXX`, g++, g++-14 … g++-7, the conda compiler next to nvcc, clang++ that nvcc accepts and that compiles C++17 with
OpenMP (`-allow-unsupported-compiler` only as a last resort). **Compile-time limits must be the same on host and
device**: pass `-D` options through `DEFS` (the Makefile also forwards any `-D` found in `NVCC_EXTRA`); every device
checks a build signature of all such constants and struct layouts in its start-up test. The header functions have
internal linkage, so every object file runs its own copy, compiled with its own compiler. Host code is compiled with
`-ffp-contract=off` (no FMA contraction, results independent of `-march`); since v0.4.0 the device build is
`--fmad=false` as well (`FMAD=true` restores the v0.3.0 contraction: measured on the H200 at the same speed, and FMA-free is
$\sim 5\times$ closer to the CPU), and its kernel is compiled with `__launch_bounds__(128, MINBLOCKS)`, `MINBLOCKS=8` by default
(register cap for 8 resident blocks per SM: +30 % throughput at 5 loops on the H200 over the v0.3.0 value 1; 6 gives +20 %).
`RB_MINBLOCKS` and `RB_FMAD` are reported in `backend_info` (`min_blocks_per_sm_bound`, `fmad`). The CUDA runtime is linked statically (only the driver, `libcuda.so`, is
needed at run time).

## Backends (`--backend`)

| backend | what runs | cost unit in the JSON |
|---|---|---|
| `cpu` | OpenMP on the host; one thread = one block of the canonical batch algorithm | core-s (thread CPU time) |
| `cuda` | every visible GPU (`--devices 0,2` to select), one host thread per GPU, dynamic queue of 65 536-sample units | device-s (cudaEvent time of each launch × its fill factor, apportioned to classes by their block cycles) |
| `emu` | the same multi-device scheduler driving **emulated** GPUs: the device program runs on the host with the device's data layout (packed relocated class images, interleaved scratch, persistent blocks, atomic chunk counter) | device-s |
| `auto` | `cuda` if at least one GPU passes its start-up self-test; `cpu` only if there is no GPU at all (if GPUs exist but all fail, it stops unless `--allow-cpu-fallback`) | |

One algorithm for all backends (`include/rbar/batch.h`): a chunk of 4096 consecutive samples of a class is evaluated
by 128 lanes × 32 samples (FP64, Neumaier-compensated per-lane accumulators in registers), reduced by a fixed binary
tree over the lanes, and chunks are merged in order within units of 16 chunks, units in order. Per-class results are
therefore **bitwise identical** for `cpu` and `emu`, any thread count, any number of (emulated) devices and blocks, any
launch size and any order of execution; a real GPU differs only by device-libm / FMA round-off (`tools/golden_check.py`).

**Random numbers.** Philox4x32-10 with a per-class key `rb_class_key(seed, salt)`, salt = FNV-1a 64 of
`kind|L|Nickel index`, counter = (sample index, kind << 26, call). A class's random numbers depend only on (seed, kind,
L, graph), not on its position in the list or on how an order is split into runs: runs at different loop orders or
of different sub-lists are independent even with the same `--seed` (`check --what streams`: $\lvert\rho\rvert \le 0.005$ between
index-matched classes of b4 and b5, against up to 0.35 with the legacy scheme). Replicas of the same order must use
different `--seed` values. `--streams legacy` reproduces the v0.1–0.2 scheme (key = seed, stream = kind << 26 | index).

GPU start-up diagnostics (per device, in `backend_info.devices[].selftest`; a failing device is not used): compute
capability ≥ 6.0, kernel loadable for the device's architecture, occupancy, stack limit; **build signature** (limits and
struct layouts) equal on device and host; FP64 `exp`/`log` ($\mathord{\le}\;2$ ulp from the host), `sqrt` and `fma` (exact), Philox
known-answer vectors on the device; 4096 samples of up to 3 classes evaluated on the device and on the host (same Hepp
sector for every sample, $\lvert\Delta w\rvert \le 10^{-11}\sum\lvert\text{terms}\rvert$); one full chunk through the production kernel vs the host reduction.
During the run: all copies on the device's own stream followed by a watchdog-guarded synchronisation, class images
and views verified by read-back after upload, launches refused for non-resident classes, the kernel refuses invalid
class views (`rb_view_ok`: such a chunk returns n = NaN), every unit's sample count checked, every wait on the device
guarded by a watchdog (`--hang-s`, at least 300 s, and 20 × the expected duration of a launch). A failing device is
removed and its work requeued on the others (on the host if none is left; recorded as `fallback_units`).

**Launch sizing.** A device's first launch is one wave of its grid (calibration); later launches hold up to `--launch-s`
(0.5 s) of work, but never fewer than two full waves while work remains, and never more than fits before the deadline.
Device cost per sample is normalised by the launch's fill factor ($\text{chunks}/(\text{waves}\times\text{grid})$), so partial waves do not
inflate it (in a simulation of the scheduler: 100 % utilisation for any chunk time). The launch log records the fill factor
and the expected duration of every launch.

**FP32** (`--fp32`, bare kinds `bare4`, `int4`, `bare2d`, `int2d` only): evaluation in single precision (atoms, terms),
point generation, density and accumulation in FP64. For the unsubtracted `bare4` the per-sample error is $\sim 10^{-6}$ of the
sd; for `int4` ($t^0$-subtracted sunsets) it reaches $10^{-2}$ of the sd per sample and biases the mean by up to $\sim 10^{-4}$ sd
per sample (`check --what fp32`), i.e. FP32 is a benchmark option, not for production. $\bar R$ kinds refuse it.

Options: `--devices`, `--launch-s` (default 0.5 s), `--max-launch-chunks`, `--mem-frac` (device memory usable for class
images, default 0.7), `--hang-s`, `--selftest-classes`, `--selftest-tol`, `--no-selftest`, `--allow-cpu-fallback`,
`--fp32`; emulator: `--emu-devices`, `--emu-blocks`, `--emu-reverse`, `--emu-mem`, `--emu-fail-device`,
`--emu-fail-after`, `--emu-fail-mode 0|1` (1: the launch reads a zeroed class view). SIGINT / SIGTERM: current
launches finish, the JSON is finalised, exit status 130; a second signal exits at once.

## Usage

    bin/rebar_cpu gen   --legs 4 --loops 5                               # list graphs (practical up to 5 loops Γ4, 6 loops Γ2)
    bin/rebar_cpu info  --kind b --loops 5 --gen [--dedup]               # combinatorics per class
    bin/rebar_cpu run   --kind b --loops 5 --gen --pilot 50000 --samples 1e8 --seed 1 --out b5.json
    bin/rebar_cpu run   --kind h --loops 5 --gen --seconds 1800 --out h5.json      # time-boxed, Neyman by time
    bin/rebar_cpu run   --kind b --loops 6 --list data/lists/gamma4_L6.txt --drop-1vr --seconds 3600 --out b6.json
    bin/rebar_cpu run   --kind c --loops 6 --list data/lists/gamma2_L6.txt --seconds 3600 --out c6.json   # --drop-1vr is ignored for c
    bin/rebar_gpu run   --kind h --loops 9 --list data/lists/gamma2_L9.txt --drop-1vr --seconds 36000 --out h9.json  # re-executes rebar_gpu-e18
    bin/rebar_gpu run   --resume h9.json --seconds 36000                  # continue an interrupted run (same computation)
    bin/rebar_gpu run   --resume h9.json --extend --seconds 7200          # add a time budget to a complete run (new plan)
    bin/rebar_cpu bench --kind b --loops 5 --gen                         # single-thread ns/sample per class
    bin/rebar_cpu check --what philox|streams|vanish|prune|twins|faces|fp|fp32|tail|topw --kind b --loops 4 --gen
    bin/rebar_cpu check --what emu --kind b --loops 4                   # GPU-emulation unit test (bitwise vs cpu)
    # GPU machine
    ./run_all.sh                                                        # everything, time-boxed, one tarball (see README.md)
    REBAR_TEST_EMU=1 ./run_all.sh --budget 35                           # test mode WITHOUT a GPU: GPU steps on 2 emulated devices
    bin/rebar_gpu devices                                                # driver / runtime / device properties (JSON)
    bin/rebar_gpu selftest --backend cuda --kind b --loops 5             # start-up diagnostics only
    bin/rebar_gpu run --backend cuda --kind b --loops 7 --list data/lists/gamma4_L7.txt --drop-1vr --seconds 36000 --out b7.json
    python3 tools/golden_check.py ../results/checks/golden-cpu/b_L5.json gpu_b_L5.json --require-backend cuda
    python3 tools/jsoncmp.py a.json b.json                              # bitwise comparison of two runs (non-timing keys)
    python3 tools/recombine_on.py b5.json --N 0,1,2,3                   # O(N) recombination of one run

Options: `--N` ($O(N)$ weight; default 1), `--threads`, `--rounds` (default 8; raised automatically so that a round holds
$\mathord{\le}\;2\times10^6$ units), `--noprune`, `--omega-scale` (density softening, default 0.7; honoured by every command, `check` included),
`--omega-shift` ($2\cdot\text{shift}$ on $t^2$-protected faces, default 2), `--selfloops` (generator), `--drop-1vr` (b/h only; see below),
`--drop-selfloops` (b/h/c, $L \ge 2$), `--skip-divergent` (bare4/bare2d: drop sunset-divergent graphs), `--dedup` (merge
Whitney twins after a pointwise check; the merged class carries the summed weight polynomial), `--streams`,
`--allow-negative-faces` (see below). `eval --kind K --nickel NK < points` prints $f(x)\cdot\prod x$.

**Checkpoint and resume (v0.4.0).** `run` writes its JSON after the pilot, after the plan, and after every round; the file
holds a `checkpoint` object with everything a later `run --resume FILE` needs to continue the run *exactly*: the plan (pilot
size, allocation rule, planned rounds, per-class targets `base`, `nalloc`), the rounds done, and per class the committed
sample counter `next` (the first Philox sample index not yet merged), the accumulator as hex bit patterns (exact), and the
*prefix accumulator* `part` of a task cut short mid-way with its unit count `part_done`. The multi-device scheduler evaluates
a task in units of `RB_UNIT` = 65536 samples aligned at the task's first sample and merges them in order; a stop (deadline,
SIGTERM, or the test hooks) leaves every task with a prefix of its units evaluated. The resumed run continues that task from
`next + part_done`, merging the remaining units into `part`, so the merge order is the one of an uninterrupted run and the
bits are the same: *(resume of a split run) == (uninterrupted run)* for the CPU backend, the emulated multi-device backend,
and across backends (checked on the CPU and on the emulated multi-device backend; on a real GPU the equality holds up to the device's own round-off,
as for any GPU result). Samples that a killed session had evaluated but not checkpointed are recomputed from the same Philox
counters, never drawn twice and never skipped. A resume inherits every option that defines the computation (kind, list,
seed, N, pilot, density, dedup, …; giving one with another value is an error) but not the *session* options (`--backend`,
`--devices`, `--seconds`, `--threads`, `--out`, launch sizing, self-test options): each session gives its own, or gets the
defaults; without `--out` the resumed run overwrites the checkpoint file. A complete run is not touched (`--resume` says so
and exits 0); `--extend --samples N | --seconds T` adds a new plan on top of a complete run (Neyman weights from the current
statistics, top-up targets; interrupted tasks are folded in first), recorded as an extension. `--round-s` (default 1800 s)
bounds the planned length of a round in a time-boxed run, i.e. the checkpoint interval; with `--samples` use `--rounds`.
The JSON is replaced atomically (temporary file, `fsync`, `rename`), so a kill never leaves a half-written checkpoint, and a
failed write (disk full) keeps the previous one. For a SLURM job: run with `--seconds` a little below the time limit, and
resubmit with `run --resume FILE --seconds ...` until `complete` is true (the `sessions` array of the checkpoint lists every
session, its UTC start, version, backend, wall time and samples; no host name: an older file's `host` is read and
dropped). Test hooks (never for production): `--test-stop-after-units K` sets the
stop flag, as SIGTERM does, after K units (multi-device backends) or tasks (cpu); `--test-kill-after-units K` sends SIGKILL.

**One-vertex-reducible graphs and c.** One-vertex-reducible $\Gamma^{(2)}$ graphs vanish for c too (c 5: all 3; c 6: 62 = 75 − 13 classes, the 13 dropped being exactly the 1VR graphs; c 7: 265 = 317 − 52),
as zero-momentum factorisation at a cut vertex predicts (thesis §3.2.3.3, pp. 47–48). `--drop-1vr` is in any case ignored for c
(`src/main.cpp:668`), so for c they are built and then dropped as "vanishes identically after algebraic pruning".

**Faces with negative pre-shift exponent.** The builder refuses a class with a face where the minimum over the forest
terms of $2\omega$ is negative (only nested $t^2$ subtractions could produce it): there the FP64 evaluation of the separately
exponentiated $t^2$ jets would give round-off errors with infinite variance. No such face exists for b, h, c up to
7 loops (all 3150 + 209 + 265 classes at 7 loops checked: minimum 0).

## Output (JSON)

Identical schema for every backend. Top level: `streams`, `precision`, `build_signature`, `limits` (variant and the four
compile-time limits), `complete` (every planned round done; `final` only says the writing session has ended), `checkpoint`
(see Usage; `format` `rbar-checkpoint-1`), `warnings` (non-finite
weights excluded from the sums, large single-sample round-off, backend notes), `backend`, `backend_note`, `cost_unit`,
`backend_info` (devices with properties, self-test results, launches, kernel seconds, samples/s, fill factors, longest
chunk, uploads, evictions; per-class device ns per sample at full occupancy; the launch log).

`classes[]`: Nickel index, $S$, $O(N)$ polynomial, `weight_poly` ($S\cdot\mathrm{on\_poly}$, summed over merged twins: use it to
recombine at other $N$), `salt`, `philox_key`, combinatorics (parts, forests, raw/pruned terms, atoms, shifted faces,
`omega2_pre_min`, `nneg_faces`), `value ± error` (integral without $S_G$), `contrib` ($\mathord{=}\;S_G(N) \times \text{value}$), per-sample
relative sd, $\max\lvert w\rvert/\lvert\text{mean}\rvert$, `nbad`, deep-cancellation count, `roundoff_rms_over_sd` and `roundoff_max_over_sd`
(worst single sample), core-ns per sample (device-ns with `cuda` / `emu`), kurtosis, `sd_relerr`.
`order`: sum ± err, `err_relerr`, signed series coefficient, `rho_neyman_per_sample`, `rho_time_sqrt_core_s`,
$\sum\lvert\text{contrib}\rvert/\lvert\text{sum}\rvert$, `nbad`, `classes_with_nbad`, maximal round-off ratios, reference and pull.
At the default density ($s = 0.7$) single-run error bars are Gaussian (340 replica runs at 2–4 loops; at six and seven
loops, check N7 of `../checks/`); the fat tails seen at $s = 1$ do not occur there. Judge agreement over several seeds anyway.

## Versions

Every run of the ledger was made with a version from 0.3.0 to 0.4.3 (`../ledger/`, where each run records its version
and build signature; checks N2 and N4 of `../checks/`). v0.3.0 was validated on a CPU against exact values and the
Phi4tools series ($B_1$ to $B_5$, $H_2$ to $H_5$ and $C_1$ to $C_5$ at $N = 1$, and the $O(N)$ factors at $N = 0$ to $3$)
and on GPUs through the golden runs, which compare the GPU with the CPU on identical samples (check N2). v0.4.0 added
the device-build defaults `MINBLOCKS=8` and `--fmad=false`, the e16 / e18 limit variants with automatic selection,
per-class build cost and size diagnostics, checkpoint / resume with bitwise exactness (on the CPU; on a GPU up to the
device's own round-off) and the eight- and nine-loop graph lists; v0.4.1 added the class cache, and v0.4.3 the `int2`
kind.
