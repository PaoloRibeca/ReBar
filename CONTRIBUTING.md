# Contributing to ReBar

## Computing time

The most useful contribution is GPU or CPU time. The flow is described in full in [`doc/LEDGER.md`](doc/LEDGER.md).

1. **Build.** Run `make -C integrator`. On a GPU machine, the devices must pass their start-up self-test, which every
   run performs; a contribution from a device that has not passed it is refused. The ten-loop orders need the
   20-line build, `make -C integrator VARIANTS=e20`, whose binaries (`bin/rebar_gpu-e20`, `bin/rebar_cpu-e20`) are given
   to `rebar-refine.sh --bin` by name.
2. **Refine.** Run `series/rebar-refine.sh` for as long as you like: a few hours are already useful.
   - It chooses the order to improve, or takes the one you name, and spends the time on the graphs whose pooled errors
     need it most.
   - It checkpoints every 15 minutes. An interrupted run is still a valid contribution, and it can be continued.
   - It takes a seed that the ledger has never used for that order. **Never reuse a seed, or copy someone else's.**
     The same seed draws the same samples, which adds nothing, and the merge refuses it.
3. **Send.** It writes the run to `ledger/<kind><L>/runs/<id>.json`, which git ignores, and merges it into your copy
   of the ledger. The ledger keeps, per class, the statistics merged over all the runs (`stats-*.tsv`, with the exact
   weights in `weights-*.tsv`), and a manifest entry per run in `orders.json`; the runs themselves go, as received,
   into the archive (`ledger-archive/`, outside git, to be deposited on Zenodo), from which `rebar-merge.py --replay`
   rebuilds the ledger bit for bit. Send the run file to the maintainer, or open a pull request with the ledger files
   the merge changed and the run file attached or linked. Do not edit the ledger's files by hand: the merge writes
   them. In the description, say what the machine was (GPU model and count) and how long the run took.
4. **Merge.** The maintainer runs `series/rebar-merge.py` on it. The merge checks:
   - the list checksum and the class set;
   - that the seed is unused;
   - the build signature and the self-test;
   - that there are no non-finite samples;
   - the consistency of every class with the pool.

   If all pass, it archives the run and folds it in, and the tables are regenerated. If one fails, you get the
   report.

## Reporting a problem

Open an issue with:

- the exact command line;
- the console output;
- the run's JSON, or the tarball that `integrator/run_all.sh` writes.

The tarballs keep no host name, GPU UUID, PCI bus id or directory of the machine, but they describe its hardware and
software: look at them before posting them publicly. A disagreement
with a published coefficient is also a problem worth reporting. Give both numbers and the source.

## Code

- **File header.** Every source file carries the house header, as in `graphs/tools/gaugraph.py`: name and copyright,
  the statement on AI assistance, and the GPL notice.
- **C++.** C++17, warning-free at `-Wall -Wextra` with g++ 7.5 to 16. The integrand and sampler stay single-source for
  CPU and CUDA (`integrator/include/rbar/`).
- **Python.** Python ≥ 3.6, standard library only. Shell tools are written in bash.
- **Language.** Names, comments and messages are in English.
- **Numbers.** A change that alters numbers must say so. The golden comparisons (`results/checks/golden-cpu/`) and
  `make -C graphs check` must still pass.

Pull requests are welcome, but the commits are made by the maintainer, who applies the change in the house style.

## Licence

By contributing you agree that your contribution is distributed under the repository's licences: code under the GPL
v3 or later; data (including ledger contributions) and text under CC BY 4.0.
