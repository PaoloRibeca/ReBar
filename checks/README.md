# checks — every claim of the paper as a script a reviewer can run

Each check is one script, `checks/<id>_<slug>.py`, for one claim of the paper, which cites it as "[check G2]". Its
docstring states the claim (section, table or equation), the inputs (paths in this repository), what runs, and what
PASS means. It prints its numbers and ends with one line:

    PASS <id>: <numbers>          exit status 0
    FAIL <id>: <numbers>          exit status 1
    SKIP <id>: needs <what> (...) exit status 2: what needs the missing resource did not run; the parts that could
                                  run did, and the stored outputs of the run that had it were checked instead

The table below lists the claims and the tools behind them; the latest results are in [`REPORT.md`](REPORT.md).

## Running them

    make -C checks              # all of them -> checks/REPORT.md (about 40 minutes the first time on a 4-core workstation, 30 after)
    make -C checks quick        # the quick ones (each well under a minute once the CPU binary exists)
    make -C checks G2           # one of them (its row of REPORT.md is updated, the others are kept)
    make -C checks list         # ids, quick or not, claims
    python3 checks/run.py G2 N4 # the same, directly
    python3 checks/G2_sumrules.py   # one check, without the report
    make -C checks clean        # remove checks/build (compiled helpers, cache, logs); REPORT.md stays

Requirements: Python ≥ 3.6 (standard library only), GNU make, and g++ with C++17 and OpenMP. The first check that
needs the integrator builds `integrator/bin/rebar_cpu` if it is absent (`make -C integrator cpu VARIANTS=e16`, one to
three minutes); the helpers of `checks/tools/` and `integrator/tools/lists/listcheck.cpp` are compiled into
`checks/build/` on first use. Outputs of the slow helpers (graph scans, `listcheck`, the zero-dimensional series)
are cached in `checks/build/cache/` under the SHA-256 of their inputs, so a changed list is always re-checked.
Every binary runs with `OMP_NUM_THREADS=2` and Python work uses two processes.

| Variable | Default | Meaning |
|---|---|---|
| `REBAR_THREADS` | 2 | OpenMP threads of every binary |
| `REBAR_JOBS` | 2 | Python worker processes (G1, G3, I1) |
| `REBAR_MPMATH_PYTHON` | — | an interpreter that has mpmath, for the live part of I1 when this one has not (`pip install mpmath` is the other way) |
| `REBAR_I1_MAXLOOPS` | 3 | I1 recomputes the stored 100-digit values of the classes up to this loop order |
| `PYTHON`, `CXX` | `python3`, `g++` | the interpreter (Makefile) and the C++ compiler |

## The checks

Times are those of a shared 4-core workstation (two cores used), with the cache filled; "first": with an
empty cache (the CPU binary already built).

| id | what it shows | needs | quick | time |
|---|---|---|---|---|
| G1 | every list of `graphs/lists` (through 11 loops, 1.6 million graphs at 10 loops) and `integrator/data/lists` holds distinct, connected, 1PI graphs with the stated counts; the Nickel lists are canonical (three implementations: `checks/tools/graphscan`, the integrator's `listcheck` with `src/graph.cpp`, `graphs/tools/gaugraph.py`); the generator's lists and the Nickel lists are the same sets | g++ | no | 35 s (first 10 min) |
| G2 | the zero-dimensional sum rules: the full $O(N)$ polynomial, exactly, on every list through 10 loops (`graphscan --on`, whose pairing counts equal those of the integrator's `listcheck` graph by graph on every Nickel list with at most 9 vertices, where `listcheck` gives the polynomial too), the rows of Table tab:sumrules as printed, $N=1$ from $S_G$ alone, and the `graphs/tools` checkers through 7 loops | g++ | no | 46 s (first 11 min, 1 min after G1) |
| G3 | our lists equal Phi4tools' as sets ($\Gamma^{(4)}$ $L\le7$, $\Gamma^{(2)}$ $L\le8$); the Phi4tools files are checked by SHA-256 (`--fetch` downloads them again) | — | yes | 18 s |
| G4 | the weights and $O(N)$ polynomials of `src/graph.cpp` equal GAU's (2003) on all 260 graphs with $L\le5$ and on GAU's own four-loop tables (`beta.specs`); `rebar_cpu gen` regenerates the short lists | CPU binary | yes | 17 s |
| G5 | the counts of the paper's Table of graphs and integrands per order (graphs, 1VR, generalized tadpoles, $B$/$H$/$C$ integrands, L = 2–10) follow from the Nickel lists, 1VR being decided by a cut-vertex test that is checked against the lists' flags, and the ledger holds that many classes for every order with data | `integrator/data/lists/`, `paper/sections/appendices.tex`, `ledger/orders.json` | no | 11 min |
| I1 | the integrator's forest expansion agrees pointwise with an independent 100-digit evaluation (482 stored points, 3–7 loops) | CPU binary; mpmath for the live recomputation (else SKIP) | yes | 15 s |
| I2 | one-vertex-reducible, self-loop and generalized-tadpole graphs have no terms after pruning (b, h to 5 loops, c to 7), and only they | CPU binary | no | 66 s |
| I3 | Whitney twins have pointwise equal integrands (b 4–6) | CPU binary | yes | 17 s |
| I4 | runs with and without the class cache are bitwise identical (b 5, h 5, c 4) | CPU binary | yes | 12 s |
| N1 | Philox4x32-10 reproduces Random123's known answers (binary, and an independent Python implementation) | CPU binary | yes | < 1 s |
| N2 | the GPUs reproduce the CPU's samples to round-off: 23 stored GPU golden runs (two campaigns, 12 H200) pass `golden_check.py --require-backend cuda`; this CPU reproduces the golden files; a live GPU run when there is a GPU (else SKIP) | CPU binary; a GPU for the live part | yes | 14 s |
| N3 | the ledger reproduces $B_2=1/2$, $H_2=1/81$, $C_2=1/12$ within errors, with the exact $N$ dependence | — | yes | < 1 s |
| N4 | the sums at 3–7 loops agree with Phi4tools' within errors (Table III), with the pull statistics | — | yes | 2 s |
| N5 | the ledger is what its runs give: its invariants hold (`rebar-merge.py --check`: the SHA-256 of every file, the samples and classes its runs folded, the statistics recomputed from the registers) and it is rebuilt bit for bit from the archived runs (`--replay`); every run passed the merge test against the pool of the runs before it (per-class $\chi^2$ and largest $\lvert z\rvert$, $p > 10^{-6}$), and the per-class scatter between seeds of every order is consistent | `ledger-archive/` for the replay (else SKIP) | no | 6 min |
| N6 | FP64 round-off does not bias the order sums: the stored paired FP64/FP80 evaluation of the 55 most exposed classes (`rebar_cpu check --what fp`, 2×10⁵ samples each) is re-analysed against the ledger, and four six-loop classes are recomputed live and must reproduce byte for byte | CPU binary; `results/checks/paired-precision/pairs.tsv`, `ledger/` | yes | 68 s |
| N7 | the replica ensembles at six and seven loops (eight replicas of each of B6, H6, C6, B7, H7, C7) scatter as their quoted errors say, order sums and class by class, with every number quoted in the paper | `results/checks/replicas/` | yes | 1 s |
| N8 | the round-off example of the paper: among the four classes of $H_4$, combined over its ten runs of October 2026, the one with a sunset reaches a single-sample round-off scale of 6×10⁻⁴ of its per-sample standard deviation, the others about 10⁻¹³; with `ledger-archive/`, the tables are compared with the archived runs | `results/checks/fourloop/` | yes | 3 s |
| S1 | the Callan–Symanzik relations map Phi4tools' sums back to its coefficients to $10^{-15}$, and the exact two-loop sums to the exact low-order coefficients | — | yes | < 1 s |
| S2 | the paper's generated tables are regenerated from the ledger byte for byte | — | yes | 21 s |
| S3 | the large-order ratios of Table tab:largeorder follow from the coefficients and the stated constants, as quoted in the text | — | yes | 10 s |
| S4 | route B reproduces Phi4tools' coefficients through the eighth order from its per-graph values; its runs at eight vertices agree with Phi4tools' sums, and its ninth- and tenth-order coefficients with route A's; Table tab:routes is regenerated byte for byte | `results/checks/routeb/runs/`, `results/checks/routeb10/runs/` | no | 6 min |
| S5 | the leading powers of $N$ of the order sums follow the large-$N$ limit and the $1/N$ expansion (computed in series/polynomials.py, where it reproduces Phi4tools' leading coefficients to $3\times10^{-10}$): no class has an $N^L$ term, the $N^{L-1}$ terms of all orders of the ledger agree with it; the polynomials $p_{X,k}$ have the predicted degrees and $[N^k]p_{\nu,k}=2^{-k-1}$ exactly; the text's large-$N$ numbers are regenerated | — | no | 3 min |

Without a GPU, N2 ends in SKIP after checking the stored GPU outputs; without mpmath, I1 ends in SKIP after comparing
the integrator with the stored 100-digit values; without `ledger-archive/` (kept outside git), N5 ends in SKIP after
checking the ledger's invariants and merge tests. All the others run on any Linux machine with the requirements above.

## Files

| Path | What it is |
|---|---|
| `run.py`, `Makefile` | the driver: runs the checks, times them, writes `REPORT.md` |
| `common.py` | shared code: paths, loading of the repository's tools, the CPU binary, the compiled helpers and their cache, the verdict line |
| `tools/graphscan.cpp` | a compiled scanner of graph lists (GAU strings or Nickel indices): structure, $\lvert\mathrm{Aut}\,G\rvert$, canonical Nickel index and the pairing counts of the $O(N)$ factor (a transfer matrix over the vertices) of every graph, for the 9- and 10-loop lists that the Python tools would take hours on; G1 checks it against `gaugraph.py`, G2 its pairing counts against `listcheck` |
| `reference/rebar_mp.py` | the independent multiprecision evaluator of the forest formula, used by I1 |
| `build/` | compiled helpers, cache, `logs/<id>.log` (made by the checks; `make clean`) |

The reference data the checks compare against (the I1 sample, the stored GPU runs, the Phi4tools lists that are not in
`integrator/data/lists`, GAU's four-loop tables) and the inputs they take from the campaigns (the route-B runs, the
tables of the replica ensembles and of the four-loop runs) are in [`../results/checks/`](../results/checks/README.md);
the complete records of the campaigns are kept outside the public repository.
