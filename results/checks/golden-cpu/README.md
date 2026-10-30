# Golden references for CPU/GPU consistency

The files are the output of `rebar_cpu run --kind K --loops L <graphs> --pilot 65536 --samples 0 --seed 2026` for
b, h at 3-5 loops, c at 3-5 loops, int4 at 4-5 loops (`--gen`), b and h at 6 loops (`--list data/lists/gamma{4,2}_L6.txt
--drop-1vr`), default density `--omega-scale 0.7`: exactly samples 0...65535 of every class.

Files in this directory: **version 0.3.0**, stream scheme `salted-v1`: Philox key of a class =
`rb_class_key(2026, salt)`, salt = FNV-1a 64 of `kind|L|Nickel index` (both written per class in the JSON as `salt`
and `philox_key`), counter = (sample, kind << 26, call); canonical batch reduction of `integrator/include/rbar/batch.h` (lanes
of 32 samples, block tree over 128 lanes, chunks of 4096 merged in order within units of 16 chunks, units in order).

`v0.2.0-cuda/`: the files of version 0.2.0 (legacy streams: key = seed, stream = (kind << 26) | position in the list).
v0.3.0 run with `--streams legacy` reproduces them **bitwise** in every key of `integrator/tools/jsoncmp.py` (b 3-5, h 3-4, c 3,
int4 4 checked), i.e. version 0.3.0 changed the numbers only through the new keys.

**Checking a GPU run** (same command with `--backend cuda`; `integrator/run_all.sh` does it for every file here):

    python3 integrator/tools/golden_check.py results/checks/golden-cpu/b_L5.json gpu_b_L5.json --require-backend cuda

`n` and `nbad` must agree exactly (same Philox counters, same Hepp sectors: the sector choice uses only correctly
rounded + * / on bit-identical uniforms and tables). `mean_w` may differ by round-off of the per-sample weights only
(device `exp`/`log`, FMA contraction). The checker reports that difference in round-off units of the per-sample term
scale ($\lvert\Delta\,\text{mean}\rvert / (2^{-52} \cdot \operatorname{rms}\sum\lvert\text{terms}\rvert)$, primary criterion: $\mathord{\le}\;100$) and in units of the statistical error (secondary:
$\mathord{\le}\;10^{-4}\,\sigma$; one wrong or missing sample moves a mean by $\sim 4\times10^{-3}\,\sigma$). Calibration: moving every host `exp`/`log` result by
up to $\pm 2$ ulp (`make perturb`, a proxy for CUDA's libm) moves the means by $\mathord{\le}\;0.81$ units and $\mathord{\le}\;2.9\times10^{-8}\,\sigma$ over b, h, c,
int4 at 4–5 loops. `--require-backend cuda` also fails (exit 3) when any number came
from the host (fallback) or a device died; `--min-devices 2` when the work was not split over at least two GPUs.

A CPU run on another machine (other glibc, other `-march`) agrees with these files at round-off level, not bitwise
(host code is compiled with `-ffp-contract=off`, but libm differs): use the same checker without `--require-backend`.
