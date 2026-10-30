# graphs — generation and checking of the φ⁴ graphs

The first step of the series production: the lists of one-particle-irreducible (1PI) φ⁴ graphs contributing to
$\Gamma^{(2)}$ and $\Gamma^{(4)}$ at each loop order, produced by `topologies`, the generator of the Feynman graphs with a
given list of vertex valences, and checked against exact zero-dimensional sum rules. The program derives from the
generator written in 2000–2003 for GAU and the seven-loop computation, with three improvements: it grows the graphs
by vertex insertion instead of screening labelled pairings, it canonicalises them by colour refinement, and it takes
an arbitrary list of valences.

    make                  # bin/topologies
    make check            # the lists through 8 loops (two-point) and 7 loops (four-point) against lists, the sum rules,
                          # the integrator's lists, the Python checks, the general mode against a brute force (a few minutes)
    make check-full       # through 10 loops (two-point) and 9 loops (four-point): about ten minutes on two threads

## Contents

| Path | What it is |
|---|---|
| `topologies/topologies.cpp` | the generator: the growth by vertex insertion, the canonical form, the compact output, the read modes (keys, sum rules, integrator lists); one translation unit, which includes the headers below |
| `topologies/*.hpp` | the classes kept from the 2003 generator, used by `-G`: `handle.hpp` (reference-counted handle), `permutations.hpp` (permutations within colour classes), `partitions.hpp` (integer partition tables), `generators.hpp` (`topology`, its canonical form `getCanonical` and the compact output) |
| `scripts/gau_lists.sh` | the vertex-degree lists GAU's `diagramList` passed to the generator for a given number of legs and loops: the groups, in order, of the stored lists and of `-G` |
| `scripts/check.sh` | the `make check` driver: regenerated lists against `lists` as sets of graphs (`-r`) and byte for byte (`-G`), sum rules (`-w`), the integrator's lists, the Python checks, the brute force |
| `tools/gaugraph.py` | reader, writer and canonicaliser of GAU graph strings (the format of `lists`); conversion to Nickel indices; weights, $O(N)$ factors, predicates. Python ≥ 3.6, standard library only; `python3 -I tools/gaugraph.py selftest` |
| `tools/sumrules.py` | Nickel-index parser, automorphism counting, $O(N)$ tensor factors (used by the checkers) |
| `tools/sumrule_check.py` | 2-point lists: $\sum_G S_G$ equals the exact zero-dimensional 1PI coefficient; $\Gamma^{(2,1)}$ $O(N)$ sum rule, compared with thesis p. 107 at 5 and 6 loops |
| `tools/gamma4_n1.py` | 4-point lists: $\sum_G S_G$ equals the 2003 zero-dimensional `gamma4ZeroI` at $N = 1$ (the thesis's 2003 values, transcribed in the script) |
| `tools/zerodim_coeff.py` | the exact zero-dimensional coefficients of $1/\langle x^2\rangle$ and $-W_4/\langle x^2\rangle^4$ at every loop order, the sum rules of the complete (self-loops included) 2- and 4-point lists (the table `topologies -w` carries) |
| `tools/bruteforce.py` | an independent check of the generator on a small list of valences: every graph by brute force, grouped by isomorphism class, with the counts and the sums of $S_G$ per number of legs (all, connected, 1PI) against the zero-dimensional theory with one coupling per valence; `-b bin/topologies` compares the generator's output |
| `tools/zerodim_tf.py` | self-loop-free 2-point lists: $\sum_G S_G$ equals the zero-dimensional tadpole-free coefficient |
| `tools/check_list.py` | counts, distinct canonical forms, 1PI, loops, self-loops and legs of a list |
| `tools/gau_order.py` | puts a 2-point list in any order back into the order of the stored lists (that of GAU's `gamma2` and of `-G`), byte for byte |
| `lists/` | the reference lists (GAU string format, one graph per line), see below |

$S_G = E!/\lvert\mathrm{Aut}\,G\rvert$ throughout ($E$ = number of external legs).

## The generator

A vertex of valence $d$ with $s$ line ends and $t$ self-loops has $d - s - 2t$ external legs, so the legs are implicit
and the class generated for a list of valences is every connected graph with those valences and any number of legs.
Every such graph with $V$ vertices is obtained from one with $V - 1$ vertices by inserting a vertex $w$ on $a$ slots (a
line or a self-loop, split into two lines to $w$; the same slot several times if its multiplicity allows), $b$ legs
(moved to $w$) and $t$ self-loops at $w$, with $d - 2a - b - 2t$ legs left at $w$. The inverse step removes a leaf of a
spanning tree and re-pairs its half-lines; re-pairing them as far as possible leaves at most one leg, so the
insertions with $b \le 1$ generate every connected graph, and the program grows them by levels over the sub-multisets of
the valences, one hash set of canonical forms per sub-multiset, in parallel with OpenMP. When every valence is even
and the number of legs $E$ is 0, 2 or 4 (the φ⁴ lists), the class with $E$ legs is grown alone with the insertions
that preserve $E$ ($a + b + t = d/2$): a leaf whose legs exceed its line ends cannot be re-paired, and two such leaves
need $E \ge 6$. Each candidate is canonicalised by colour refinement on neighbour bitmasks with backtracking, and
the canonical form is the valence and self-loops of every vertex and the upper triangle of the adjacency matrix
packed into bits (348 bits at twelve vertices, six 64-bit words), which the hash set of a class keys on. The cost is
the number of graphs grown times a few hundred insertions, so it follows the size of the class, growing by a factor
of ten to twelve per loop for the φ⁴ orders on one core.

    bin/topologies [-l L] [-i] [-t] [-c] [-o FILE] [-j THREADS] [-N] [-w] DEGREES

grows the connected graphs with the valences `DEGREES` (comma-separated, 1 to 8, at most 12 entries, e.g. `3,3,4,4`
or `6,4,4`) and writes them in the compact format of `lists`, one graph per line, in the order of its canonical
keys: `-l L` keeps the graphs with $L$ loops, i.e. with $E = \sum d - 2(L + V - 1)$ legs (without it every $E$ goes to
the same list), `-i` the 1PI ones (no bridge), `-t` includes the graphs with self-loops (always grown, since they are
the parents of other graphs; written only on request), `-c` is accepted for compatibility (only connected graphs are
grown), `-o` writes to a file instead of standard output (the log then goes to standard error), `-N` writes the
integrator's list instead (index, Nickel index, $S_G$, the one-vertex-reducibility and self-loop flags, sorted by
Nickel index), `-w` prints the sum of $S_G$ over the graphs written, per number of legs, as an exact fraction. When
some valence differs from 4 the colour section of the compact format carries the legs as negative colours
(`colour -k` = a vertex with $k$ legs, the convention of GAU and `tools/gaugraph.py`), e.g. `2(<-2^2>1)` is the
one-line graph with two three-valent vertices and two legs at each; φ⁴ strings are unchanged. GAU's degree lists
(`scripts/gau_lists.sh`) count line ends instead, with $4 - d$ legs at an entry $d < 4$: such a list is the $E = 0$
class of the valence reading (`-l` at the largest loop number, and `-x` in the read modes below), and a φ⁴ order,
which GAU enumerated one leg pattern at a time, is one class here.

    bin/topologies -E 2|4 -V VMAX [-o DIR] [-j THREADS] [-N] [-w] [-a] [-G]

is the φ⁴ mode, order by order: for every $V \le$ `VMAX` with $L = V + 1 - E/2$ loops it writes `DIR/<L>-loop-2`
(1PI, self-loops included, the `-cit` lists) and `DIR/<L>-loop-2-ci`, or `DIR/<L>-loop-4` (no self-loops, the `-ci`
lists) and `DIR/<L>-loop-4-cit`, and prints the size of every class with its time. `-N` also writes the integrator's
lists, `DIR/gamma<E>_L<L>.txt` and `_with_selfloops.txt` (`../integrator/data/lists`), with its own Nickel
canonicaliser (the convention of `tools/gaugraph.py`, exact branch and bound) and $|\mathrm{Aut}\,G|$ counted during
the canonical search: all 23 stored integrator lists are reproduced, byte for byte where their format is ours and in
their Nickel column where they are Phi4tools' own files. `-w` prints the zero-dimensional sum rule of every order,
the exact sum of $S_G$ over the 1PI graphs with self-loops as a fraction, against the exact coefficients of
$1/\langle x^2\rangle$ and $-W_4/\langle x^2\rangle^4$ it carries (computed once by `tools/zerodim_coeff.py`). `-a`
also writes `DIR/<L>-loop-<E>-all`, every connected graph of the order, 1PI or not. `-G` relabels the output with
the `getCanonical` of the 2003 classes and writes it in the order of the stored lists (one GAU degree list at a time,
within each the self-loop vectors ascending and the upper triangle descending), after which the self-loop-free lists
(`<L>-loop-4`, `<L>-loop-2-ci`) are byte for byte the stored ones; the stored strings of graphs with self-loops carry
another labelling, so those lists are compared as sets. `-G` costs far more than the generation (`getCanonical`
enumerates the permutations within colour classes) and serves byte comparisons only.

Three read modes work on any list in the compact format (or on the plain integers of `gaugraph.py adj`):
`topologies -r < LIST` prints the canonical key of every graph in hexadecimal, so that two lists are compared as sets
of graphs with `sort` and `cmp`; `topologies -w < LIST` prints the sum rules of the list, per number of legs;
`topologies -N < LIST` prints the list as an integrator list. A list without colour section is read with the φ⁴
convention ($4 - s - 2t$ legs); with `-x` with no legs, the line-end reading of GAU's degree lists. `scripts/check.sh`
(`make check`) does all the comparisons with the generator and bash alone, then runs the Python checks of `tools/`
and `tools/bruteforce.py`.

The lists `1-10-loop-2`, `9-10-loop-2-ci` and `1-10-loop-4` are reproduced exactly, and the sum rules of the complete
lists hold through 11 loops ($\Gamma^{(2)}$) and 9 loops ($\Gamma^{(4)}$ with self-loops, 1 101 532 graphs). On a
general list of valences the program agrees with a brute-force enumeration and with the zero-dimensional theory with
one coupling per valence (`tools/bruteforce.py`) on every list tried, among them `3,3,4,4`, `6,4,4`, `3,4,5` and
`5,5,5,5`. The cost is about 0.2 ms per connected graph on one core of a 2018 laptop (1.8 GHz CPU):
the 10-loop $\Gamma^{(2)}$ order, 706 572 connected graphs, takes 140 s and 161 MB, the 9-loop $\Gamma^{(4)}$ order,
2 317 683 connected graphs, 348 s and 435 MB, and the cost per graph grows slowly with the order, with the number of
slots. (The popcount instruction matters: without `-mpopcnt` the compiler's library routine for the bit counts of
the refinement makes the program four times slower.)

### Provenance

The program derives from the generator Paolo Ribeca wrote in 2000–2003 for GAU and the seven-loop computation, the
one behind thesis §3.2.1 and Table 3.1, which enumerated labelled pairings of half-lines from partitions of the vertex
degrees and kept the canonical ones. It keeps that program's compact format, its graph classes and four of its
headers, which `-G` uses for the labelling of the stored lists; the self-loop count joined the vertex colouring of
`getCanonical` in 2026, without which two graphs differing only in where their self-loops sit could be taken for one.

## Reference lists (`lists`)

| File | Graphs | Content |
|---|---|---|
| `1-loop-2` … `10-loop-2` | 1, 2, 5, 17, 68, 313, 1632, 9685, 63774, 463933 | 1PI $\Gamma^{(2)}$ graphs, self-loops included (`-cit`) |
| `1-loop-4` … `10-loop-4` | 1, 2, 8, 27, 129, 660, 3986, 26540, 197022, 1593254 | 1PI $\Gamma^{(4)}$ graphs (`-ci`; no self-loops) |
| `9-loop-2-ci`, `10-loop-2-ci`, `11-loop-2-ci` | 9184, 59614, 428164 | 1PI $\Gamma^{(2)}$ graphs at 9, 10 and 11 loops without self-loops (`-ci`); see below for the 11-loop one |

Without self-loops the $\Gamma^{(2)}$ counts are 1, 2, 6, 19, 75, 317, 1622 at 2–8 loops (thesis Table 3.1).
Every list passes its zero-dimensional check: `sumrule_check.py` for `1-10-loop-2` (and the $\Gamma^{(2,1)}$ $O(N)$
polynomials of thesis p. 107), `gamma4_n1.py` for `1-8-loop-4`, `zerodim_tf.py` for `9-loop-2-ci`. The integrator
reads Nickel-index lists (`../integrator/data/lists/`), converted from these with `tools/gaugraph.py nickel`.

`9-loop-2` and `10-loop-2` (October 2026; SHA-256 `acf83c5a3c6d7f230bb3a97cfde5814929de55c039eac96762a929b9f09b4d65`
and `af4619a2e107d3fbf24af29b457078bb94aa0746cc7ff4faec6b928fabda49b1`) are in the labelling and the order of the
lower orders, that of GAU's `gamma2` (one degree list at a time, `scripts/gau_lists.sh`; within each the self-loop
vectors ascending and the upper triangle descending), which `-G` and `tools/gau_order.py` reproduce. Their graphs
without self-loops (no `[...]` section) are `9-loop-2-ci` and `10-loop-2-ci`, byte for byte and in the same order;
`checks/G2` also finds the full $O(N)$ sum rule exact on both.

On a cluster node (Intel Xeon Platinum 9242) the generator takes, on one core, 2.1 s,
22 s, 258 s and 3 158 s for the four-point orders at 7-10 loops and 0.1 s, 1.1 s, 9.8 s and 98 s for the two-point
ones (compact lists, Nickel lists and sum rules included); on 96 threads the eleven-loop two-point order takes 23 s,
the twelve-loop one 275 s (44 402 094 connected graphs, 31 534 160 1PI, 3 393 754 without self-loops, 11.7 GB) and
the eleven-loop four-point one 1 005 s (177 397 551 connected, 96 047 201 1PI, 13 975 511 without self-loops, 38 GB),
every sum rule exact.

`11-loop-2-ci` (October 2026; SHA-256 `e8722643acc5a1bfc7fb02587137b32727bce12d79c5e5e24c79a36fe500794a`) was made by
an earlier version of this program, which did not reorder the output: 5 375 249 connected two-point graphs with
eleven vertices, of which 3 677 401 are 1PI and 428 164 of those have no self-loops. The full list with self-loops,
`11-loop-2` (244 MB,
SHA-256 `2b69964fa66a2c40a5bcd073f97bc1540fc64072bd26b056c8083d60e8128741`), is kept outside git in the run archive
(`ledger-archive/lists/`, with the generator's log) and is regenerated by `topologies -E 2 -V 11`; its
zero-dimensional sum rule, $\sum_G S_G = 7001454875/20736$ (`topologies -w`), is exact, as is that of every lower
order. The lists are in the order in which that version wrote them, not in the order of the stored lower orders
(`sort` both before a byte comparison with `-G`).
