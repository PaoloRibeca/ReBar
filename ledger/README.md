# ledger — pooled Monte Carlo statistics of every graph class

Every number of ReBar's series comes from here. Each contribution is the JSON of one `rebar_cpu` / `rebar_gpu run`
(version ≥ 0.3.0) with an unused seed; the classes of an order pool exactly across contributions. The ledger holds,
per order, the weights of its classes as exact fractions and, per class, the folded registers of every run merged so
far (Welford's n, mean, m2 merged pairwise, higher moments, extremes), plain text whose size does not grow with the
number of runs, and a manifest entry per run; the runs themselves are archived as received outside git
(`../ledger-archive/`, to be deposited on Zenodo by batch) and replay to this ledger bit for bit. The design,
the format and the rules are in [`../doc/LEDGER.md`](../doc/LEDGER.md); the tools are in
[`../series/`](../series/README.md).

| Path | What it is |
|---|---|
| `orders.json` | per order (`b8`, `h9`, …): list and its SHA-256, graphs, classes, pooled sum at $N = 1$ with error, relative error, samples, the sum as a polynomial in $N$, reference and pull, status, target, prefactor, build and device hours, the scatter between seeds, the files of its tables, and the manifest of its contributions (per run: identity, devices and self-tests, time box, the integrator's order block, which classes it covered, the merge test, diagnostics, the SHA-256 of the run file) |
| `<order>/weights-*.tsv` | the weight polynomial of every class as exact fractions, one row per list position |
| `<order>/stats-*.tsv` | per class the folded registers: $n$, mean, $m_2$ (exact), $m_3$, $m_4$, max $\lvert w\rvert$, `nbad`, round-off, time per sample on the H200, one row per list position |
| `<order>/runs/` | (ignored by git) run JSONs written or continued by `rebar-refine.sh` |
| `../ledger-archive/<order>/<id>.json.xz` | (outside git) every merged run, byte for byte, xz-compressed |
| `seeds.json` | every seed per order: `merged` (in the ledger), `running` (reserved by `rebar-refine.sh`), `excluded` (runs deliberately left out) |

Status of an order: `open` (refined until its target relative error), `target` (target reached), `closed`
(validation order, $L \le 6$: exact values or more precise references exist; refined only when named), `planned`
(no data yet).

## Contributing

    python3 series/rebar-merge.py RUN.json            # validate, report, fold in (refuses with reasons; --force overrides)
    series/rebar-refine.sh --bin integrator/bin/rebar_gpu --seconds 36000 [--order h9]   # run a fresh seed and merge

A contribution can also be sent to the maintainer as its run JSON, or merged locally and sent as a pull request
together with the run file (which goes into the archive). Use a seed no `seeds.json` entry has for that order
(`rebar-refine.sh` takes one above every seed used so far). `rebar-merge.py --check` verifies the ledger's
invariants, `--replay` rebuilds it from the archive.

## Contents

Since 19 October 2026 the ledger holds 953 contributions to 27 orders, $B_2$ to $B_{10}$, $H_2$ to $H_{10}$ and
$C_2$ to $C_{10}$, from rebar 0.3.0 to 0.4.3 (the campaigns are listed in [`../results/README.md`](../results/README.md)).
The orders through six loops are `closed`; those of seven to nine loops have reached their target, except $B_9$;
$B_9$, $B_{10}$, $H_{10}$ and $C_{10}$ are `open`. `make -C series status` prints the state of every order, and the
paper's Tables tab:validation and tab:newsums give the sums ([`../paper/generated/`](../paper/generated/)).
