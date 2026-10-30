# Supplemental Material

This repository is the Supplemental Material of the paper *Ten-loop fixed-dimension renormalization-group
functions of the three-dimensional O(N) model from distributed GPU-accelerated parametric integration* (P. Ribeca;
[arXiv:2610.39307](https://arxiv.org/abs/2610.39307), [PDF](paper/main.pdf)). This file says where each item listed in the paper's appendix "Data and code
availability" is. Everything here is regenerated from the ledger by `make -C series`, and check S2 verifies that
the paper's numbers are what the ledger gives ([`checks/`](checks/README.md)).

## The numbers

| What | File | Format |
|---|---|---|
| The coefficients of $\beta$, $\eta$, $\eta_{\phi^2}$ and $\nu$ in $\bar u$ at $N = 0$–4, with their errors | [`series/generated/coefficients.json`](series/generated/coefficients.json) | JSON; a readable summary in `coefficients.txt` |
| The numerator polynomials $p_{X,k}(N)$ of the paper's Eq. (series-form), with the covariance of all their coefficients | [`series/generated/polynomials.json`](series/generated/polynomials.json), [`polynomials.tsv`](series/generated/polynomials.tsv) | JSON and TSV; dated copy in [`results/coefficients/2026-10-18/`](results/coefficients/2026-10-18/README.md), which describes the format |
| The per-order sums $B_L$, $H_L$, $C_L$ at $N = 1$ and as polynomials in $N$, with their errors and the record of every run | [`ledger/orders.json`](ledger/orders.json) | JSON ([`doc/LEDGER.md`](doc/LEDGER.md)) |
| The pooled statistics of every graph class | [`ledger/<kind><L>/`](ledger/README.md) | TSV: `stats-*.tsv` (registers), `weights-*.tsv` (exact weights) |
| The resummed exponents, with the parameter scans and the benchmarks with their sources | [`series/generated/resum.txt`](series/generated/resum.txt), [`paper/generated/exponents.tex`](paper/generated/exponents.tex) | text, LaTeX macros |
| Route B: the bare sums at eight and nine vertices and the comparison with route A | [`results/checks/routeb/runs/`](results/checks/README.md), [`paper/generated/routes.tex`](paper/generated/routes.tex) | the six runs (xz-compressed JSON); the comparison as LaTeX macros and the body of Table tab:routes |
| Route B at ten vertices, for the tenth order | [`results/checks/routeb10/runs/`](results/checks/README.md) | the three sums merged from their 32 shards (xz-compressed JSON) |
| The large-order test and the large-$N$ test | [`series/generated/largeorder.txt`](series/generated/largeorder.txt), [`series/generated/polynomials.txt`](series/generated/polynomials.txt) | text |
| Every table body and number of the paper | [`paper/generated/`](paper/generated/) | LaTeX macros, written by `make -C series` |

## The items of the paper's appendix

1. **Graph lists**: [`graphs/lists/`](graphs/README.md) (the generator's output, $\Gamma^{(4)}$ and $\Gamma^{(2)}$ to
   ten loops, with self-loops for the checks, and $\Gamma^{(2)}$ without self-loops at eleven) and
   [`integrator/data/lists/`](integrator/data/lists/README.md) (the Nickel lists the integrator reads, with weights and
   flags). The $O(N)$ polynomials of the classes are in the ledger (`weights-*.tsv`). The lists taken from Phi4tools
   carry its MIT licence notice (`LICENSE-Phi4tools.txt`).
2. **Per-class results**: [`ledger/`](ledger/README.md) (format 3: [`doc/LEDGER.md`](doc/LEDGER.md)). The output of
   every run as received (about 1 GB) is too large for the repository: it will be deposited on Zenodo with the release, and
   `series/rebar-merge.py --replay` rebuilds the ledger from it bit for bit.
3. **Run records**: [`results/checks/`](results/README.md) (the reference data and inputs of the checks, among them
   the route-B runs and the tables of the replica ensembles), [`results/coefficients/`](results/README.md) (dated
   snapshots of the derived series). The complete records of the campaigns (reports, logs, environment records) are
   kept outside the public repository; the run files of the ledger's contributions are deposited with its run archive.
4. **Series**: [`series/`](series/README.md): the tools that validate and pool the runs and derive everything above.
5. **Checks**: [`checks/`](checks/README.md): one script per claim of the paper, `make -C checks`; the last results
   in [`checks/REPORT.md`](checks/REPORT.md).
6. **Code**: [`integrator/`](integrator/README.md) (the integrator, C++17 and CUDA from one source, with its
   self-tests; its CPU golden references are in [`results/checks/golden-cpu/`](results/checks/README.md)) and
   [`graphs/`](graphs/README.md) (the graph generator and the list tools).
