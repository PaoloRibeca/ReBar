# series — from the ledger to the RG functions

The tools that keep the [ledger](../ledger/README.md) (the pooled per-class Monte Carlo statistics) and derive from it
the per-order sums $B_L$, $H_L$, $C_L$, the coefficients of $\beta$, $\eta$, $\eta_{\phi^2}$ and $\nu$, the checks and the paper's
numbers. Design and rules: [`../doc/LEDGER.md`](../doc/LEDGER.md). Python ≥ 3.6, standard library only; bash for the
driver.

    make -C series              # generated/coefficients.{json,txt}, generated/checks.{json,txt}, ../paper/generated/*.tex
    make -C series check        # self-tests (about 20 s; the eight-loop reproduction reads the run archive)
    make -C series status       # the orders of the ledger
    python3 series/rebar-merge.py RUN.json                       # fold a run into the ledger
    series/rebar-refine.sh --bin integrator/bin/rebar_gpu --seconds 36000 [--order b8]   # a new run, then merge

| File | What it does |
|---|---|
| `rebar_ledger.py` | shared code: paths, lists, the per-class statistics of a run and their pooling, the folded-register storage of the ledger (format 3: exact weights, per-class registers merged pairwise, plain TSV), the archive of the runs, the reading API (`iter_classes`, `load_classes`, `iter_contribution_classes`, `load_contribution`), seeds, references, number formatting; a small CLI used by the driver (`pick`, `pending`, `reserve`, `status`) |
| `rebar-merge.py` | validates runs against the rules of `doc/LEDGER.md`, reports (per-class $\chi^2$ against the pool, worst classes, the order sum's deviation), archives them and folds them in, chunk by chunk; `--check` verifies the ledger's invariants, `--replay` rebuilds it from the archive, `--rebuild` recomputes the order statistics |
| `rebar-refine.sh` | picks an order and a fresh seed, runs the integrator on the order's whole list for a time budget, merges; continues an interrupted run |
| `coefficients.py` | per-order sums as polynomials in $N$ and at $N = 0$–4; the Callan–Symanzik relations; $\beta$, $\eta$, $\eta_{\phi^2}$, $\nu$ in $\bar u$ at $N = 0$–4 with errors, and their numerator polynomials; comparison with Phi4tools |
| `checks.py` | large-order ratio test of $\beta$; `--sumrules` runs the zero-dimensional sum rules of `graphs/tools` on the orders' lists |
| `tables.py` | writes `../paper/generated/values.tex` (macros), `newsums.tex`, `validation.tex` |
| `largeorder.py` | the large-order test: every coefficient against its leading large-order estimate, the subleading fit and the prediction of each new coefficient from the lower orders (`largeorder.tex`, `largeorder-fit.tex`) |
| `costfig.py` | the measured costs: Table tab:cost, this work's row of the effort table, Figs. 1 and 2 and the cost numbers of the text (`costhist.tex`) |
| `routeb.py` | route B: the RG functions from the intermediate-scheme vertex functions through the $Z$ factors, against Phi4tools and route A (`routes.tex`) |
| `polynomials.py` | the numerator polynomials $p_{X,k}(N)$ by exact algebra on polynomials in $N$, their full covariance from the per-class integrals, and the large-$N$ test (`newpoly.tex`, `generated/polynomials.json`, `.tsv`) |
| `resum.py` | the resummation: Borel–Leroy transform with conformal mapping, the scan of its parameters, the statistical errors by resampling, the benchmarks with their sources (`exponents.tex`) |
| `tests.py` | `make check` |
| `generated/` | the derived files (`make`) |

## Running the integrator for the ledger

Build it once (`integrator/README.md`, `TECHNICAL.md`): `make -C integrator cpu` (OpenMP; `VARIANTS=e16` builds only
the 16-line variant, enough to 8 loops, in a few minutes) or `make -C integrator` for `bin/rebar_gpu` (CUDA 12.x; it
re-executes `bin/rebar_gpu-e18` by itself for the 9-loop orders, so keep both; the 10-loop orders need the e20
variant, built with `VARIANTS=e20`). Version 0.4.0 or later:
`rebar-refine.sh` relies on `--round-s` and `--resume`.

`rebar-refine.sh --bin BIN --seconds T [--order ORDER]` then

1. continues a run of an earlier call that did not end normally (a seed in `seeds.json` with state `running`), with
   `BIN run --resume FILE --seconds T`;
2. otherwise picks the order (the one named, else the `open` order with the largest relative error), reserves a seed
   above every seed used so far, and runs
   `BIN run --kind K --loops L --list integrator/data/lists/<list> [--drop-1vr] --pilot 16384 --alloc time --seconds T --seed S --round-s 900 --out ledger/<order>/runs/<id>.json`
   (`--drop-1vr` for b and h; for c the one-vertex-reducible graphs are built and dropped as vanishing);
3. merges the result with `rebar-merge.py` (a run that ended at its time box with its plan incomplete is a valid
   contribution: its counters are consistent).

The integrator checkpoints every `--round-s` seconds and writes its JSON atomically, so a killed run loses at most one
round; calling `rebar-refine.sh` again continues it with the same seed and the same counters. A merged contribution can
be extended later (`BIN run --resume FILE --extend --seconds T`, then `rebar-merge.py FILE`: the same id with more samples
replaces the old version). Options: `--label`, `--pilot`, `--round-s`, `--threads`, `--ledger`, `--dry-run`, and
`-- EXTRA` for further integrator options (`--devices 0,1`, …). The allocation within a run is the integrator's own
Neyman rule by measured cost; it does not use the pooled errors of the ledger (see `doc/LEDGER.md`).

## From sums to coefficients

Conventions of `integrator/README.md` ("What it computes"): $u = g/8\pi$, $\bar u = (N+8)u/6$,
$\tilde b = \sum_L (-1)^{L+1} B_L u^{L+1}$, $h = \sum_L (-1)^L H_L u^L$, $c = \sum_L (-1)^{L+1} C_L u^L$,

$$
\eta = \frac{2h}{1+h},\qquad \beta_u = -u + 2\eta u + (2-\eta)\,\tilde b,\qquad \eta_{\phi^2} = -(2-\eta)\,c,\qquad \frac1\nu = (2-\eta)(1-c),
$$

and in $\bar u$ (Phi4tools' $\tilde u$): $\beta_n = k^{1-n}[u^n]\beta_u$, $\eta_n = k^{-n}[u^n]\eta$, likewise $\nu_n$, $\eta_{\phi^2,n}$, with
$k = (N+8)/6$. The numerators $p_{\beta,n}(N) = (N+8)^{n-1}\beta_n$, $p_{\eta,n} = (N+8)^n\eta_n$, … are polynomials, found by exact
interpolation at integer $N$ (one extra point checks the degree).

Each per-order sum comes from one source, fixed at $N = 1$: the exact low orders ($B_1$, $B_2$, $H_2$, $C_1$, $C_2$), Phi4tools
through `integrator/tools/refs.py`, or the ledger. The default (`--source best`) takes the smallest error; at present
that is the ledger for $B_7$ to $B_{10}$, $H_5$ to $H_{10}$, $C_7$ to $C_{10}$ and Phi4tools or exact values for every other
order. `--source ledger` prefers the ledger wherever it has the order, `--source refs` uses no Monte Carlo at all: fed
with refs.py's sums the Callan–Symanzik algebra returns Phi4tools' $\beta$, $\eta$, $\nu$ to $6\times10^{-16}$ (`make check`).

Errors are propagated to first order from every input's error and added in quadrature, the inputs being independent
(different orders and kinds are different integrals, sampled with different seeds). Caveats: refs.py's inputs come
from the same Phi4tools coefficients and are treated as independent; the coefficients of different orders and
functions, and the values at different $N$, share inputs and are correlated — combining them (a resummation, a fit)
needs their covariance. `polynomials.py` gives it: the numerator polynomials with the full covariance of all their
coefficients, propagated from the per-class integrals (`generated/polynomials.json`).

At $N = 1$ (the ledger of 19 October 2026): $\eta_8 = 0.002910571(13)$ against Phi4tools' $0.0029093(16)$; new,
$\beta_9 = -2.3664203(59)$, $\beta_{10} = 4.781152(47)$, $\beta_{11} = -10.38718(12)$, $\eta_9 = -0.005103393(51)$,
$\eta_{10} = 0.00995358(14)$.

## Checks

Large order: $\beta_k \sim c\,(-a)^k\,k!\,k^b$, so $R_k = \lvert\beta_{k+1}/\beta_k\rvert / [a(k+b+1)] \to 1$; with $a = 0.14777422 \cdot 9/(N+8)$ and
$b = 3 + N/2$ (the instanton action $A = 113.3835$ of $d = 3$ and $a = 48\pi/[(N+8)A]$, Zinn-Justin, *Quantum Field Theory
and Critical Phenomena*, 3rd ed., Eq. (28.73): $a(d = 3) = 0.147774232 \cdot 9/(N+8)$; check S3), $R_k$ at $N = 1$ is 0.750, 0.854,
0.937, 0.975, 0.998 for $k = 3$–7 and 1.008, 1.013, 1.014 for $k = 8$–10 (the new $\beta_9$ to $\beta_{11}$). The
zero-dimensional sum rules are those of `graphs/tools` (`make -C graphs check` runs them with the list regeneration);
`checks.py --sumrules [ORDER…]` runs
them on the GAU lists of the ledger's orders.

## The paper's numbers

`tables.py` writes `../paper/generated/values.tex` with one macro per number (contract in its header: `\valBeight`,
`\relHeight`, `\nclCseven`, `\ncontribBeight`, `\samplesHeight`, …, `\etaEight`, `\betaNine`, `\nuEight`, … with
`Nzero`–`Nfour` variants, `\ratioEight`, `\rawratioEight`, `\ledgerDate`; every name is defined, `---` when absent), and
the table bodies `newsums.tex` (new and improved sums, with the scatter of the contributions of every order that has
at least two covering the whole order) and `validation.tex` (orders 2–7 against the references). The other
generated files of the paper come from `largeorder.py`, `costfig.py`, `routeb.py`, `polynomials.py` and `resum.py`
(the table above).
