# results — coefficient snapshots and the reference data of the checks

What ReBar's computing campaigns produced, other than the per-class statistics. Those are in the
[ledger](../ledger/README.md): every run folded class by class into `ledger/<order>/stats-*.tsv`, with a record of the
run in `ledger/orders.json`, and every run's JSON kept as received in the run archive (`ledger-archive/`, outside git)
([`../doc/LEDGER.md`](../doc/LEDGER.md)). Everything
here is provisional, as the numbers of the paper are (see the status in [`../README.md`](../README.md)).

    results/
      coefficients/<ledger date>/      a snapshot of what `make -C series` derives from the ledger
      checks/                          reference data and inputs of the reproducibility checks

## The campaigns

The records of the campaigns are kept outside the public repository: one directory per campaign, named by its start
date (UTC), its devices and a label, with its report, logs and environment record, and a README that says what ran,
on what, and where its per-class data went. The ids of the ledger contributions start with the same date and devices,
the run file of every contribution is in the ledger's run archive, and what the checks read of the campaigns is in
[`checks/`](checks/README.md). A node of the cluster is named by a label (`gpu-node-N`, `cpu-node-N`).

| Campaign | When, on what | Orders | Ledger contributions |
|---|---|---|---|
| `2026-10-14-h200x4-validation` | 14 October 2026, rebar 0.3.0, 4 × H200 (`gpu-node-2`), 2 h 44 min | self-tests and golden runs; validation at 2–5 loops; replicas of $B_4$, $H_4$; throughput, tuning, scaling; complete orders at 4–7 loops | `20261014-h200x4-…` |
| `2026-10-15-h200x5-night` | night of 14–15 October 2026, rebar 0.3.0, 5 × H200 (`gpu-node-2`), 13 h | two replicas each of $B_8$ and $H_8$; the four-loop calibration diagnostic | `20261014-h200x5-…` |
| `2026-10-15-h200x8-weekend` | 15–16 October 2026, rebar 0.4.0, 8 × H200 (`gpu-node-3`), 20.7 h | $H_9$, $C_9$, $C_8$, $B_8$, $H_8$ to their target errors | `20261015-h200x8-weekend-…` |
| `2026-10-16-h200x8-refine` | 16–17 October 2026, rebar 0.4.0, 8 × H200 (`gpu-node-3`), 31.7 h | $H_9$ and $C_9$ extended to $10^{-5}$; $B_7$, $H_7$, $C_7$ recomputed to $3\times10^{-6}$ and $5\times10^{-6}$ | `20261015-h200x8-weekend-h9`, `-c9` (replaced); `20261016-h200x8-refine-…` |
| `2026-10-17-h200x4-c10` | 17 October 2026, rebar 0.4.1 (e20), 24 shards on 4 × H200 each, one hour per shard | $C_{10}$, the first tenth-order sum, 53 397 classes | `20261017-h200x4-c10-shard00` … `shard23` |
| `2026-10-17-h200x1-b9` | 17 October 2026, rebar 0.4.2 (e20), 24 shards on one H200 each, 40 minutes per shard | $B_9$, the coefficient of $\bar u^{10}$ in $\beta$, 166 047 classes | `20261017-h200x1-b9-shard00` … `shard23` |
| `2026-10-17-h200x4-h10` | 17–18 October 2026, rebar 0.4.2 (e20), 16 shards on 4 × H200 each, one hour per shard | $H_{10}$, the coefficient of $\bar u^{10}$ in $\eta$, 46 102 classes | `20261017-h200x4-h10-shard00` … `shard15` |
| `2026-10-17-h200x1-b10` | 17–18 October 2026, rebar 0.4.2 (e20), 480 shards on one H200 each, 4–7 minutes per shard | $B_{10}$, the coefficient of $\bar u^{11}$ in $\beta$, 1 377 678 classes | `20261017-h200x1-b10-shard000` … `shard479` |
| `2026-10-18-h200x1-replicas` | 18 October 2026, rebar 0.4.3 (e16), 48 replicas on one H200 each, 4–8 minutes per replica | eight independent replicas of each of $B_6$, $H_6$, $C_6$, $B_7$, $H_7$, $C_7$: the scatter test of the quoted errors | `20261018-h200x1-<order>-replica1` … `replica8` |
| `2026-10-18-h200x1-routeb` | 18 October 2026, rebar 0.4.3 (e16, e18), six runs on one H200 each, 15–33 minutes per run | route B: the intermediate-scheme $\Gamma^{(4)}$, $\Gamma^{(2)}$ and $\partial_{p^2}\Gamma^{(2)}$ at eight and nine vertices | not in the ledger (the runs: [`checks/routeb/runs/`](checks/README.md)) |
| `2026-10-18-h200x1-sixloop` | 18 October 2026, rebar 0.4.3 (e16), 24 replicas on one H200 each, 27–45 minutes per replica | $B_6$, $H_6$, $C_6$: eight independent replicas of each | `20261018-h200x1-<order>-refine1` … `refine8` |
| `2026-10-18-h200x1-sixloop2` | 18 October 2026, rebar 0.4.3 (e16), 100 runs on one H200 each, 27–44 minutes per run | $B_6$, $C_6$: fifty independent runs of each | `20261018-h200x1-<order>-batch2-r01` … `r50` |
| `2026-10-18-h200x1-fiveloop` | 18 October 2026, rebar 0.4.3 (e16), 24 runs on one H200 each, 38–55 minutes per run | $B_5$, $H_5$, $C_5$: eight independent runs of each | `20261018-h200x1-<order>-refine1` … `refine8` |
| `2026-10-18-h200x1-fiveloop2` | 18 October 2026, rebar 0.4.3 (e16), 100 runs on one H200 each, 38–54 minutes per run | $B_5$, $C_5$: fifty independent runs of each | `20261018-h200x1-<order>-batch2-r01` … `r50` |
| `2026-10-19-h200x1-lowloop` | 19 October 2026, rebar 0.4.3 (e16), 49 runs on one H200 each, 55 minutes per run | $B_3$, $H_3$, $C_3$, $B_4$, $H_4$, $C_4$: 3 to 22 independent runs of each, to below $10^{-6}$ | `20261019-h200x1-<order>-lowloop-r01` … |
| `2026-10-19-h200x1-routeb10` | 19 October 2026, rebar 0.4.3 (e18, e20), 32 shards on one H200 each, one hour per shard | route B at ten vertices: the intermediate-scheme $\Gamma^{(4)}$ at nine loops, $\Gamma^{(2)}$ and $\partial_{p^2}\Gamma^{(2)}$ at ten, for the tenth order | not in the ledger (the reduced runs: [`checks/routeb10/runs/`](checks/README.md); whole shards in `ledger-archive/routeb/`) |
| `2026-10-21-cpu-gentime` | 21 October 2026, the generator on one Xeon 8568Y+ node, idle | the graph lists of Table II regenerated and timed, seven to ten loops; every list identical to the stored one | not in the ledger (no integrals) |
| `2026-10-22-cpu-build11` | 22 October 2026, `rebar_cpu info` on one Xeon 9242 node | the class builds of samples of the eleven-loop $\Gamma^{(2)}$ list (int2, int2d, e22) and of the ten-loop $\Gamma^{(4)}$ list (int4, e20): the build cost of route B at the eleventh order | not in the ledger (no samples) |
| `2026-10-22-cpu-growtime` | 22 October 2026, the vertex-insertion generator (`grow`, now `topologies`) on one Xeon 9242 node | the lists of seven to ten loops regenerated and timed on one core; the $\Gamma^{(2)}$ lists to twelve loops and the $\Gamma^{(4)}$ list at eleven on 96 threads, with their sum rules | not in the ledger (no integrals) |

## `coefficients/`: snapshots of the derived series

`make -C series` regenerates, from the ledger, the route-B runs of [`checks/`](checks/README.md) and
`integrator/tools/refs.py`, the files of `series/generated/` and `paper/generated/`
([`../series/README.md`](../series/README.md)). A snapshot copies them under the date of the ledger they come from, so
that the numbers of a version of the paper can be found again after the ledger has moved on:

| File | Content |
|---|---|
| `coefficients.json` | per order, the source (exact, Phi4tools or ledger), the sum at $N = 0$–4 and as a polynomial in $N$; the coefficients of $\beta$, $\eta$, $\eta_{\phi^2}$ and $\nu$ in $\bar u$ at $N = 0$–4 with their errors, and their numerator polynomials |
| `checks.json` | the large-order ratio test of $\beta$ |
| `values.tex`, `newsums.tex`, `validation.tex`, `largeorder.tex` | the paper's generated macros and table bodies |
| `polynomials.json`, `polynomials.tsv`, `polynomials.txt` | the numerator polynomials $p_{X,k}(N)$ with the covariance of all their coefficients, and the report of their checks (`series/polynomials.py`) |

`2026-10-16/` is the ledger updated on 2026-10-16 at 10:16 UTC (the merge of the weekend run; `\ledgerDate` = 16
October 2026). The snapshots are copies, not sources: when the ledger changes, the outputs of `make -C series` go into a
new directory. Check S2 of [`../checks/`](../checks/) verifies that the paper's tables are what the ledger gives.

`2026-10-18/` holds the polynomials and their covariance from the ledger of 18 October 2026 (after the second six-loop
merge), as cited in the paper's Appendix on data and code.

## `checks/`: reference data of the checks

The stored inputs and outputs that the reproducibility checks of [`../checks/`](../checks/) compare against when a
live recomputation is not possible (for example the sample of points evaluated in 100-digit arithmetic for check I1),
and the inputs they take from the campaigns: the route-B runs, the tables of the replica ensembles, the golden GPU
runs ([`checks/README.md`](checks/README.md)). `make -C checks` runs the suite (see "Reproducing the checks" in
[`../README.md`](../README.md)).

## Credits and licence

The credits are in [`../README.md`](../README.md). The data are under CC BY 4.0
([`../LICENSE-DATA.md`](../LICENSE-DATA.md)).
