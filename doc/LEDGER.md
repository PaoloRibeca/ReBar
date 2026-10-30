# The ledger: pooled Monte Carlo statistics of every graph class, and how to add to them

ReBar's results are sums over graph classes of Monte Carlo estimates of forest-subtracted parametric integrals.
Every class has its own Philox key, derived from (seed, kind, loops, Nickel index), so runs with different seeds
sample a class independently and their results pool exactly. The *ledger* is the repository's record of those pooled
statistics: it is what the coefficient tables are computed from, and what `rebar-refine.sh` improves. The tools are in
`series/` (`series/README.md`); the data in `ledger/README.md`. A *contribution* is the
JSON that `rebar_cpu` / `rebar_gpu run` writes (version >= 0.3.0): program, version, build signature, kind, L, N,
seed, settings, backend and devices with their self-tests, per class nickel, S, weight polynomial, salt, prefactor,
n, mean_w, sd_w, kurtosis, max|w|/|mean|, round-off, nbad, timing, the order sum; v0.4 adds the checkpoint (the exact
accumulators). Checkpoints and time-boxed runs whose plan did not finish are valid contributions. Kinds: `b`
(four-point with one insertion -> B_L), `h` (two-point, one insertion, p^2 derivative -> H_L), `c` (two-point, two
insertions -> C_L); normalisations as in `integrator/README.md`. The id is `<date>-<devices>-<order>-<label>-seed<seed>`.

## Two tiers

The ledger is kept in two tiers. In git, the *folded registers* (format 3): per order the weight polynomials of its
classes as exact fractions, written once, and per class the merged statistics of every run so far, plus a manifest
entry per run. Their size depends on the number of classes, not on the number of runs: a second seed of an order
changes digits, not lines. Outside git, the *archive*: every merged run file as received, byte for byte,
xz-compressed (`ledger-archive/<order>/<id>.json.xz` next to the ledger, to be deposited on Zenodo by batch), with its
SHA-256 in the manifest. The ledger is a function of the archive: `rebar-merge.py --replay` rebuilds it from the
archived runs and checks it bit for bit, so that whatever the registers do not hold (per-run class values, build
times, accumulators, the checkpoints to `--resume` a seed) can always be recomputed. Nothing in the tools deletes
an archived run.

## Layout (format 3)

    ledger/
      README.md
      orders.json            "format": "rebar-ledger-3"; per order its list and SHA-256, the files of its weights and
                             stats with their SHA-256, the prefactor, the statistics (sum, error, sum polynomial in N,
                             classes, samples, reference and pull, status, target, the scatter between seeds, build
                             and device hours) and the manifest of its contributions
      seeds.json             every seed per (kind, L), with its state: merged, running (reserved), excluded
      <kind><L>/weights-PPPPPPP.tsv   weight polynomials, exact fractions; one row per list position from PPPPPPP on
      <kind><L>/stats-PPPPPPP.tsv     the per-class registers, one row per list position from PPPPPPP on
      <kind><L>/runs/        (not in git) run JSONs being written or continued by rebar-refine.sh
    ledger-archive/<kind><L>/<id>.json.xz   (not in git) every merged run as received

Both kinds of table are plain UTF-8 TSV (git compresses what it stores) with a header line of column names; after it,
line i holds list position PPPPPPP + i, an empty line where the position has no class (a vanishing, one-vertex-
reducible or dropped graph, or one no run has sampled yet). The positions are those of the order's list
(`integrator/data/lists`); classes outside it (forced merges) and every class of an order without a list follow it,
their nickels in `orders.json` (`extra`). Files hold 131072 positions (`rows`): at ten loops at most 12.5 MB of
stats and 7 MB of weights each, so every file stays well under 50 MB. Not stored, because they follow from the list and the
order and are recomputed (and checked against every incoming run): the nickel (the list's), the salt (FNV-1a 64 of
`kind|L|nickel`, `integrator/src/main.cpp`), the prefactor Gamma(pw) pi^(-L/2) (`build.cpp`; one per order, in
`orders.json` with pw), the 1VR flag (the list's).

**weights** (`weight_poly`): the weight polynomial S_G O(N) as exact fractions, comma-separated
(`13120/19683,3904/6561,...`); S = its value at N = 1. A row is written when a class first receives data and never
changed. The fractions are recovered from the integrator's doubles as S c_k / 3^m with integers c_k summing to 3^m
(the O(N) factor is a polynomial with coefficients in Z/3^m equal to 1 at N = 1), checked to agree with every double
within 1e-12 and float(sum) = S exactly. They are not bitwise the integrator's doubles: those carry round-off of up
to 3e-13 relative (94% of the 1.9 million coefficients of the ledger differ from the nearest double of their
fraction, by up to a hundred ulps and more). The ledger therefore serves the correctly rounded value of the exact weight; the order
sums move by up to 2.5e-13 relative (B9), the per-class statistics not at all.

**stats**, per class, merged over the runs that sampled it:

| column | register | merged | stored |
|---|---|---|---|
| `n`, `mean`, `m2` | Welford's: count, mean, sum of squared deviations (m2 = (n - 1) sd^2 of a run) | pairwise (Chan, Golub, LeVeque 1979) | exact (shortest round-trip repr) |
| `m3`, `m4` | central moment sums | pairwise (Pebay 2008, the general order-p update) | 6 significant digits |
| `wmaxabs`, `wmin`, `wmax` | max \|w\|, min w, max w | extremes (`wmin`, `wmax` empty until the integrator provides them) | 6 digits |
| `nbad` | non-finite samples | sum (empty if 0) | integer |
| `rmax`, `r2` | max over samples of eps sum\|terms\|, sum of its squares (eps = 2.2e-16) | max, sum | 4 digits |
| `tau_ref` | core ns per sample on the reference device (NVIDIA H200, `orders.json` `reference_device`) | n-weighted mean over the runs on it; others leave it | 3 digits |

On read: sd = sqrt(m2/(n - 1)), value = pref mean, error = pref sd / sqrt(n), kurtosis = n m4 / m2^2,
max|w|/|mean| = wmaxabs/|mean|, roundoff_max_over_sd = rmax/sd, roundoff_rms_over_sd = sqrt(r2/n)/sd,
core_ns_per_sample = tau_ref. A run's registers: n, mean_w and m2 = (n - 1) sd_w^2 from its JSON; m3, m4, max|w| and
the round-off sums from the exact accumulators of its checkpoint (acc + part: m3 = q3 - 3 m q + 2 n m^3,
m4 = q4 - 4 m q3 + 6 m^2 q - 3 n m^4), which reproduce the JSON's kurtosis to 1e-15; a run without them (0.3.0)
enters with m4 from its kurtosis, m3 = 0, and the extremes from its ratios, and is listed in `orders.json`
(`approximate`). The approximate registers are rounded to their digits after every fold, so that the state is the
same however the runs were batched (the replay depends on it). The state after a sequence of merges differs from a
one-shot pooling only at the ulp level (2.5e-16 relative on the means of the ledger, 3e-15 on one mean near 0).

**orders.json**, per order: the statistics (as before: sum, err, rel_err, sum_poly, sum_poly_err, classes,
classes_expected, complete, samples, nbad, max_roundoff_max_over_sd, versions, order_level, ref, pull, status),
`files` (weights and stats: first position, rows, SHA-256), `rows`, `pref`, `pw`, `reference_device`, `build_s`
(the single-thread build time of the classes, each counted once, from the first run that sampled it), `core_s`
(device seconds of all runs), `scatter` (the sums of chi^2 and dof of the merge tests: the order's per-class scatter
between seeds, check N5), `approximate`; and the manifest, one entry per run (2-4 kB): id, seed, program, version, build
signature, utc, backend, devices and `device_list` (per device: name, self-test, samples, kernel seconds, samples
per second), args (an absolute path reduced to its file name), final/complete, `order` (the integrator's order
block, every scalar as written: sum, err, rho, core_s, wall_s, nbad, round-off, ref and pull where present),
`time_box` (requested seconds, wall and device seconds used, the wall fraction of the box), `checkpoint` (rounds,
extensions, sessions: utc, version, wall seconds, samples, rounds), `rho_tau` (relative error times sqrt(device
seconds), costfig.py's), classes, `coverage` (which classes it sampled: `"all"`, a shard `{"shard": i, "of": n}`
by the rule of the shards of B9 (survivor j of the list's lines without 1VR=1 goes to shard j mod n), or the position
ranges), `excluded` and `excluded_positions` (nbad > 0), `classes_pooled`, `samples_pooled`, samples, its order
sum and error, `check` and `check_detail` (the merge test: chi^2, dof, p, largest |z|, Bonferroni p, the order
sum's deviation, the five worst classes), `diagnostics` (its largest round-off, kurtosis and max|w|/|mean|, the
classes with nbad), `registers` (exact or approximate), `build_s` (its classes' build time) and `build_s_first`
(the part counted for the order), `file_sha256` (the run file, which the archive holds), merged_utc, forced,
warnings, note (free text: it should not name machines or directories). No host name, GPU UUID or PCI bus id is
copied from the run. The number of runs that sampled a class (k) follows from the coverage entries.

### Reading the ledger (`series/rebar_ledger.py`)

    iter_classes(ledger, order, fields=None)        the classes in list order with the fields
                                                    nickel, S, weight_poly, salt, pref, is1vr, n, mean_w, sd_w, value,
                                                    error, contrib, contrib_err, kurtosis, nbad, maxw_over_absmean,
                                                    roundoff_max_over_sd, roundoff_rms_over_sd, core_ns_per_sample
                                                    (= tau_ref), ncontrib (= k); fields: only these (and only what they
                                                    need is computed); the scatter is in the manifest
    load_classes(ledger, order, fields=None)        the same as a list
    iter_contribution_classes(ledger, order, cid)   per class of one run, every field of its JSON: from the archive
    iter_order_contributions(ledger, order)         (id, class) for every run of the order: from the archive
    load_contribution(ledger, order, cid)           a run as a dict, from the archive; read_contribution(path)
    load_contribution_header(ledger, order, cid)    its run-level fields (order block, backend_info, ...)
    pool_order(ledger, order, entry)                the current classes {nickel: class}; leave-one-out (skip_id) needs
                                                    a replay

The per-run functions raise a clear error when the archive is not there (`archive=` or `--archive` to point at it).
`REBAR_LEDGER` in the environment replaces the default `ledger/`.

## Pooling

For a class with contributions i of n_i samples, weight mean m_i and weight standard deviation s_i, where the integrator
defines s_i^2 = (Q_i - n_i m_i^2)/(n_i - 1) with Q_i the sum of the squared weights, the pooled sample is

    n = sum n_i,   m = sum n_i m_i / n,   s^2 = [sum (n_i - 1) s_i^2 + sum n_i (m_i - m)^2] / (n - 1),

the statistics of one run holding all the samples (exact for the mean and the variance; checked to 2e-16 against the
integrator's own accumulators of two CPU runs, `make -C series check`). Format 3 keeps it as Welford's registers
(n, mean, m2 = (n - 1) s^2) and folds each run in with the pairwise update of Chan, Golub and LeVeque,

    n = na + nb,   d = mb - ma,   mean = ma + d nb / n,   m2 = m2a + m2b + d^2 na nb / n,

the same algebra, and the higher central moments with Pebay's (SAND2008-6212, eq. 2.7, implemented for any order p:
columns m5, m6, ... can be added when the runs provide them):

    m3 = m3a + m3b + d^3 na nb (na - nb)/n^2 + 3 d (na m2b - nb m2a)/n
    m4 = m4a + m4b + d^4 na nb (na^2 - na nb + nb^2)/n^3 + 6 d^2 (na^2 m2b + nb^2 m2a)/n^2 + 4 d (na m3b - nb m3a)/n

The class estimate is pref * mean with error pref * sd / sqrt(n); its contribution at any N is weight_poly(N) times
that. The order sum and its error follow from the classes (independent), in list order.

The class-pooled order sum differs from an inverse-variance combination of the contributions' order sums (a different
estimator of the same quantity, with weights estimated from the data): for the two b8 replicas 30.469848(137) against
30.469825(138), 0.17 sigma. `orders.json` keeps the latter (`order_level`, with the scatter chi^2 of the complete
contributions) as a cross-check; the ledger's value is the former.

The per-class scatter between seeds (check N5; the chi^2/dof of the paper's table of the new sums, tab:newsums, is
that of the order sums of the complete seeds, `order_level`) is the sum of the merge tests' chi^2 over their degrees
of freedom, the classes in common (rule 6: each run against the pool of the runs before it, a sequence of independent
tests); the merge test divides each run's deviation by the combined error of the run and of the pool, since the error
estimate of a short run of a heavy-tailed weight is itself uncertain.

## Merging a batch

    series/rebar-merge.py --id 20261017-h200x1-b9-shard{n:02d} b9-shard-*/results-b9/shard_*.json

With several files `--id` is a template (`{n}`: the last number in the file name; `{stem}`: the name without
`.json`; `{i}`: the position in the batch). The files may be plain or .xz. Each is read once, streaming (a run JSON of
any size), its classes and checkpoint rows spooled to disk by chunk of list positions; the rules are checked chunk by
chunk, every file against the state and the files of the batch accepted before it (folded in); a refused file is
left out of the pool of the files after it, which are checked again. The accepted runs are then archived (and each
copy checked to decompress to its SHA-256) before anything of the ledger changes; only the chunks that changed are
rewritten, every file next to its final name (`.new`) and moved into place just before `orders.json`: an
interrupted merge leaves the ledger as it was. Memory is bounded by a chunk plus 12 bytes per graph of the list (the
nickel index): a batch of 480 shards of a ten-loop order merges in about half an hour on one core, within 0.5 GB.
The stats take about 100 bytes per class and the weights about 60, and they do not grow with the runs (a second seed
changes digits, not lines); the manifest does, by one line of 2-4 kB per run in `orders.json`; the archive holds the
runs themselves.

A merged run cannot be replaced or taken out (the state is folded): the same id again is a no-op if the file is
identical (its SHA-256), refused otherwise; a checkpoint is merged once it is finished (rebar-refine.sh continues an
interrupted run before merging it).

## Validation of a contribution (`rebar-merge.py` refuses otherwise, unless `--force`)

1. Program `rebar_cpu` / `rebar_gpu`, a known good version (0.3.0, 0.4.0 to 0.4.3), kind b/h/c, the production settings (FP64,
   `salted-v1` streams, `omega_scale` 0.7, `omega_shift2` 2: class integrals do not depend on the density, but other
   densities are diagnostics, and s = 1 has heavy tails), no emulated devices, no `--dedup`.
2. Every class is in the order's list (`integrator/data/lists/gamma4_L<L>.txt` for b, `gamma2_L<L>.txt` for h and c;
   at L <= 5 these are byte for byte what `--gen` produces; for an order without a list, the classes the ledger
   already has, the first `--gen` contribution defining them), without duplicates; its salt is FNV-1a 64 of
   `kind|L|nickel`, its prefactor Gamma(pw) pi^(-L/2) and the order's, its 1VR flag the list's, its weight
   polynomial S c_k / 3^m exactly and the ledger's, S the value at N = 1. A contribution may cover a subset (a shard); a complete one (its
   classes and dropped graphs cover the list) must have as many classes as the earlier complete ones. An order is
   `complete` when its pooled classes are as many as expected: the number of classes of its complete contributions,
   or, for an order computed only in shards, the number of graphs of its list that are not one-vertex-reducible and,
   for h, not generalized tadpoles (lists from eight loops up carry the 1VR flag; the ten-loop four-point list is
   stored xz-compressed and read as such).
3. The seed is not in `seeds.json` for that (kind, L), in any state (same seed = same samples = not independent), unless
   it is reserved for this very id by `rebar-refine.sh`.
4. The kernel build signature is known good, and on GPUs every device that produced samples passed its start-up
   self-test (`backend_info.devices[].selftest.ok`, build signature match). CPU runs have no device self-test (the
   integrator's CPU golden checks apply).
5. Classes with non-finite samples (`nbad` > 0) are excluded from the pool and listed (a warning, not a refusal).
6. Per-class consistency with the pooled ledger (the same classes, the contribution itself left out): chi^2 of
   z = (v - V)/sqrt(e^2 + E^2) with p > 1e-6, and the largest |z| with p > 1e-6 after the Bonferroni count; the report
   names the five worst classes and gives the deviation of the order sum on the common classes.

The same id merged again: an identical file (the same SHA-256) is a no-op (nothing is rewritten); anything else is
refused (a folded run cannot be replaced). The outcome of every rule, the merge test and the run's diagnostics are in
its manifest entry.

## Orders, status and targets

Every order with a list or a policy has an entry (planned ones included). Status: `open` (refined until its target
relative error `target_rel`), `target` (reached), `closed` (validation orders, L <= 6, where exact values or more
precise references exist: refined only when named), `planned` (no data). Targets: b7 5e-6, b8 1e-5, h7 2e-5, h8 5e-5,
h9 1e-4, c7 2e-5, c8 5e-5, c9 1e-4; `series/rebar_ledger.py` (`POLICY`) holds them, `orders.json` keeps them across
merges.

## Refining

`series/rebar-refine.sh --bin BIN --seconds T [--order ORDER]` chooses the target (the order named, else the `open`
order with the largest relative error; planned orders only when named), reserves a fresh seed (above every seed in
`seeds.json`, and above 100000; state `running`), runs the integrator on that order's whole list with
`--pilot 16384 --alloc time --round-s 900` (checkpoint every 15 minutes), writing the run JSON to
`ledger/<order>/runs/<id>.json` (ignored by git), and merges it (it keeps its name as id; the merge archives it). A
run that does not end normally stays reserved and is continued with the integrator's `--resume` by the next call; the
samples are the same as in an uninterrupted run.

Allocation: within a run the integrator allocates by its own Neyman rule on measured cost and per-sample variance. Since
every contribution is allocated that way, the pool stays close to the optimum; the integrator does not take the
*pooled* errors into account (a top-up allocation, giving less to classes already precise), and a shard (a list of
the classes with the largest pooled error excess) is the way to target them.

A contribution is sent to the maintainer as its run JSON (it goes into the archive and the Zenodo deposit), or
merged locally and sent as a pull request together with the run file.

## Checking and replaying

    series/rebar-merge.py --check [ORDER ...]       the invariants: every file's SHA-256 as recorded; per order the sum
                                                    of n over the classes = the samples folded by its runs, the sum of k
                                                    (from the coverage entries) = the classes they folded; the statistics
                                                    of orders.json = those recomputed from the stats; weights for every
                                                    class with stats
    series/rebar-merge.py --rebuild [ORDER ...]     recompute orders.json's statistics from the stats (the derived tables)
    series/rebar-merge.py --replay [--archive DIR]  rebuild the whole ledger from the archived runs, in manifest order,
                                                    and compare every stats and weights file and orders.json, byte for byte

Replaying the archive reproduces the ledger bit for bit, and `--check` passes on it (check N5). The build hours of an
order count each class once, from the first run that sampled it; the time per sample of a class (tau_ref) is the
n-weighted mean over the runs on the reference device, so it moves when runs of different speed pool (the 0.4.0 runs
take about 20% less device time per sample than the 0.3.0 ones).

## Derived tables (`series/`)

From `orders.json` and the classes (`iter_classes`): the per-order sums B_L(N), H_L(N), C_L(N) as polynomials in N (exact O(N) weights,
pooled means) and at N = 0..4 with their errors; the coefficients of beta, eta, eta_phi2 and nu in ubar through the
Callan-Symanzik relations at N = 0..4, with errors propagated to first order and added in quadrature (inputs from
different orders are independent), and their numerator polynomials; the exact low orders and the Phi4tools-derived
sums of `integrator/tools/refs.py` where the ledger has no data or a larger error. The coefficients of different orders,
functions and N share inputs and are correlated; `series/polynomials.py` gives the covariance of all the coefficients
of the numerator polynomials, propagated from the per-class integrals. Then the large-order test and the
hook to the sum rules of `graphs/tools`, and the paper's fragments (`paper/generated/`). These are regenerated by
`make -C series`; `make -C series check` tests the algebra (Phi4tools round trip), the reproduction of the published
eight-loop numbers, the pooling and the merge rules.
