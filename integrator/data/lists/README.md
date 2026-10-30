# Graph lists

Any tab-separated file with the Nickel index in some column can be passed with `--list FILE` (the first field
containing `|` is taken; lines starting with `#` are ignored).  Loop orders 1-5 ($\Gamma^{(4)}$) and 2-6 ($\Gamma^{(2)}$) can also be
generated on the fly with `--gen` (the built-in generator reproduces the Phi4tools lists string for string at
$\Gamma^{(4)}$ 4-5 loops and $\Gamma^{(2)}$ 6 loops); at 6 loops $\Gamma^{(4)}$ and beyond the generator is too slow (> 15 min), use the lists below.

| file | content | graphs | source |
|---|---|---|---|
| `gamma4_L{1..5}.txt` | 1PI $\Gamma^{(4)}$ graphs, $L$ loops ($L+1$ vertices), no self-loops | 1, 2, 8, 27, 129 | `rebar_cpu gen --legs 4 --loops L` (columns: index, Nickel, $S = 4!/\lvert\operatorname{Aut}\rvert$ at $N = 1$, 1VR flag, self-loop flag) |
| `gamma2_L{2..5}.txt` | 1PI $\Gamma^{(2)}$ graphs, $L$ loops ($L$ vertices), no self-loops | 1, 2, 6, 19 | `rebar_cpu gen --legs 2 --loops L` |
| `gamma2_L{1..5}_with_selfloops.txt` | the same including tadpole (self-loop) graphs | 1, 2, 5, 17, 68 | `rebar_cpu gen ... --selfloops` (to check that they vanish). Complete: their weight sums reproduce the zero-dimensional $\Gamma^{(2)}$ series, while GAU's stored lists lack 1 and 8 of these graphs at 4 and 5 loops |
| `gamma4_L6.txt` | $\Gamma^{(4)}$, 6 loops (7 vertices), incl. one-vertex-reducible graphs | 660 | Phi4tools `int4pt07.txt` |
| `gamma2_L6.txt` | $\Gamma^{(2)}$, 6 loops (6 vertices) | 75 | Phi4tools `int2pt06.txt` (identical to `rebar_cpu gen --legs 2 --loops 6`) |
| `gamma4_L7.txt` | $\Gamma^{(4)}$, 7 loops (8 vertices) | 3986 | Phi4tools `int4pt08.txt` |
| `gamma2_L7.txt` | $\Gamma^{(2)}$, 7 loops (7 vertices) | 317 | Phi4tools `int2pt07.txt` |
| `gamma4_L8.txt` | $\Gamma^{(4)}$, 8 loops (9 vertices), incl. 4851 one-vertex-reducible graphs, no self-loops | 26540 | `topologies -N` (below; columns as `rebar_cpu gen`). New: no other list exists |
| `gamma2_L8.txt` | $\Gamma^{(2)}$, 8 loops (8 vertices), incl. 235 one-vertex-reducible graphs | 1622 | the same pipeline; Nickel indices, order and numbering identical to Phi4tools `int2pt08.txt` |
| `gamma2_L9.txt` | $\Gamma^{(2)}$, 9 loops (9 vertices), incl. 1146 one-vertex-reducible graphs | 9184 | the same pipeline. New: no other list exists. 17 lines per graph: needs `RB_MAXE` ≥ 17, the e18 variant (the e16 binary, `RB_MAXE` = 16, re-executes its e18 companion) |
| `gamma4_L9.txt` | $\Gamma^{(4)}$, 9 loops (10 vertices), incl. 30975 one-vertex-reducible graphs | 197022 | the same pipeline; the same set as `../../../graphs/lists/9-loop-4` (check G1). 18 lines per graph: e18 variant |
| `gamma2_L10.txt` | $\Gamma^{(2)}$, 10 loops (10 vertices), incl. 6217 one-vertex-reducible graphs | 59614 | the same pipeline; as `10-loop-2` there. 19 lines per graph: e20 variant |
| `gamma4_L10.txt.xz` | $\Gamma^{(4)}$, 10 loops (11 vertices), incl. 215576 one-vertex-reducible graphs (xz-compressed) | 1593254 | the same pipeline; as `10-loop-4` there. 20 lines per graph: e20 variant |

Phi4tools files: G. Sberveglieri, G. Spada, *Phi4tools*, arXiv:2311.12576, JHEP 05 (2024) 073;
<https://github.com/g-spada/Phi4tools>, tag v1.0.2 = commit `abd3b8b85ff66e0fb51dfb1e72f773ccfc33b43b`, directory
`paclet/Resources/INTEGRANDS/` (columns: index, Nickel index, number of sunset-type subgraphs, Phi4tools integrand
expression).  Fetched on 2026-10-13 from
`https://raw.githubusercontent.com/g-spada/Phi4tools/abd3b8b85ff66e0fb51dfb1e72f773ccfc33b43b/paclet/Resources/INTEGRANDS/<file>`;
files copied verbatim (renamed).  SHA-256:

    3469009760bc98d9981b73761d241cbf3b6b6a41599386568539469e45d8ad11  gamma2_L6.txt  (int2pt06.txt)
    a6585c8e272c28bd2e1a1e5aeb5305630b15665d23ce4163cc0eb3112e2e5340  gamma2_L7.txt  (int2pt07.txt)
    d25b13e67036793d37ffa673bda0b15689cb1162e2f332b71db88ceb70811ec9  gamma4_L6.txt  (int4pt07.txt)
    276defc71766b04eb64dbcb71e45812673151dbf090454fbf1caf71a4c265079  gamma4_L7.txt  (int4pt08.txt)

The Phi4tools lists contain the one-vertex-reducible graphs (158 and 836 in `gamma4_L6`/`L7`, 13 and 52 in `gamma2_L6`/`L7`)
and no self-loops. These graphs vanish identically for b, h and c: for b and h pass `--drop-1vr` (they are then not even
built); for c the option is ignored, and they are built and then dropped after algebraic pruning (`../../TECHNICAL.md`).
The counts are those of thesis Table 3.1, p. 44; `wc -l` gives one less for the four Phi4tools files, which have no final newline.
`int4pt09` (8-loop $\Gamma^{(4)}$) does not exist in Phi4tools v1.0.2; `int2pt08.txt` does, and `gamma2_L8.txt` was checked against it.

Every list holds the self-loop-free (tadpole-free) graphs, which is all that b, h and c need: graphs with a self-loop
vanish identically after algebraic pruning (zero-momentum mass subtraction), so no `*_with_selfloops` list is shipped beyond 5 loops. The 6-8-loop
ones were made and checked as below (not shipped) and can be remade with `../../tools/lists/make_lists.sh WORK OUT 2t:6 2t:7 2t:8`
(or `topologies -E 2 -V 8 -N`).

## 8- and 9-loop lists (v0.4.0) and beyond: provenance

**Generator.** `topologies` (`../../../graphs`, documented in `graphs/README.md`), the generator of the Feynman graphs
with a given list of vertex valences, which derives from the one P. Ribeca wrote in 2000-2003 for GAU and the seven-loop
computation (thesis §3.2.1 and Table 3.1) and grows the graphs by vertex insertion. Its φ⁴ mode writes, for every order
up to the one asked, the compact lists of `graphs/lists` and, with `-N`, these integrator lists: the canonical
Nickel index (exact branch and bound, Phi4tools / `src/graph.cpp` convention), $S = E!/\lvert\mathrm{Aut}\,G\rvert$ from the
automorphisms counted during the canonical search, the 1VR and self-loop flags, sorted in Nickel order. The 8- and
9-loop lists were first made in October 2026 with the 2003 generator and `gaugraph.py`'s converter (`topo2list.py`);
`topologies -N` reproduces them byte for byte, as it does every other list of this directory whose columns are ours,
and the Nickel column of the four Phi4tools files (`make -C graphs check`).

**Commands** (`../../tools/lists/make_lists.sh WORK OUT 4:8 2:8 2:9` runs them and compares with `topo2list.py`):

    make -C graphs                                        # graphs/bin/topologies
    graphs/bin/topologies -E 4 -V 9 -N -o WORK/4          # WORK/4/gamma4_L<L>.txt, L = 1-8, and the compact lists <L>-loop-4
    graphs/bin/topologies -E 2 -V 9 -N -o WORK/2          # WORK/2/gamma2_L<L>.txt and gamma2_L<L>_with_selfloops.txt, L = 1-9
    python3 -I tools/lists/topo2list.py WORK/4/8-loop-4 > gamma4_L8.txt   # the independent conversion, byte for byte the same

`topo2list.py` decodes each GAU string with `../../../graphs/tools/gaugraph.py`, gives it its canonical Nickel index
(exact branch and bound), $S$ from GAU's weight formula and the 1VR / self-loop flags, and sorts in Nickel order; it is
the second, independent route to the same file. Cost of the generator on one core of a cluster node's Intel Xeon
Platinum 9242 (compact lists, Nickel lists and sum rules of every order through the one asked): 22 s for `gamma4_L8`
(258 s for `gamma4_L9`, 3 158 s for `gamma4_L10`), 9.8 s for `gamma2_L9` (98 s for `gamma2_L10`); the conversion with
`topo2list.py` takes 250 s for `gamma4_L8` and 105 s for `gamma2_L9`.

Lists (SHA-256):

    44182684ab0bb1915fdfdb94130f9c80c47693d9403fecbfae21ec586cdcb0e4  gamma4_L8.txt
    de908c99a6ab243913c69dca0b7d18936f649a187d6fa1f880a70c3758ad1df8  gamma2_L8.txt
    8460d0687508c1f89a13ccb0dc351aa17f46ca4399c7518c42ad3bd01a180296  gamma2_L9.txt

## Verification (all lists)

Tools: `listcheck` (`../../tools/lists/listcheck.cpp`, compiled with the integrator's own `src/graph.cpp`: the parser and
degree-4 check that `--list` uses, `is_1pi`, `is_1vr`, `has_selfloop`, `aut_order`, `sym_factor`, `on_factor` and the
brute-force `nickel_canonical`, i.e. the functions `src/build.cpp:307-311` puts into every class), `zerodim_on.py` (exact
zero-dimensional series) and `verify_lists.py` / `verify_all.sh`.

| check | result |
|---|---|
| (a) counts | $\Gamma^{(4)}$ $L = 1$-8: 1, 2, 8, 27, 129, 660, 3986, 26540; $\Gamma^{(2)}$ $L = 2$-8: 1, 2, 6, 19, 75, 317, 1622: thesis Table 3.1, p. 44, entry for entry. $\Gamma^{(2)}$ $L = 9$: 9184 (no published count). With self-loops, $L = 1$-8: 1, 2, 5, 17, 68, 313, 1632, 9685 |
| (b) the pipeline reproduces the older lists | the generator's compact lists = the stored `graphs/lists` lists, byte for byte for the self-loop-free ones (`-G` labelling) and as sets for the others; converted, `gamma4_L1`-`L5`, `gamma2_L2`-`L5` and `gamma2_L{1..5}_with_selfloops` = the v0.3.0 files (`rebar_cpu gen`) byte for byte; `gamma4_L6`, `L7`, `gamma2_L6`, `L7` = the Phi4tools files as sets and in the same order; `gamma2_L8` = Phi4tools `int2pt08.txt` (SHA-256 `2b13d3c2dbe3389e0d908367835601fbc7505ff06a42d93682176a8006c190f6`) in index and Nickel column, line by line; the self-loop-free rows of the self-loop lists = the self-loop-free lists at $L = 2$-8 |
| (c) zero-dimensional sum rules, full $O(N)$ polynomials | for every list, $\sum_G S_G\,T_G(N)$ with the integrator's own $S_G$ and $T_G(N)$, summed exactly (integer pairing counts), equals the zero-dimensional coefficient $c_L(N)$ of `zerodim_on.py` as a polynomial: $\Gamma^{(4)}$ $L = 1$-8 and $\Gamma^{(2)}$ $L = 2$-9 (self-loop-free lists: tadpole-free scheme); also $\Gamma^{(4)}$ $L = 9$, 10 and $\Gamma^{(2)}$ $L = 10$, with the $O(N)$ factors counted by the transfer-matrix counter of `checks/tools/graphscan --on` (check G2, which also compares it graph by graph with the $3^V$ enumeration), $\Gamma^{(2)}$ with self-loops $L = 1$-8 (bare; $L = 6$-8: the unshipped lists). `zerodim_on.py` reproduces every coefficient of the 2003 zero-dimensional series (the thesis's `soloSumRules` output: `gamma2ZeroB`, `gamma2ZeroI` to $g^8$, `gamma4ZeroB`, `gamma4ZeroI` to $g^9$), so the 8-loop lists match the 2003 values; the 9-loop $\Gamma^{(2)}$ value is new (below). The integrator's double-precision weights give the same sums to $\mathord{\le}\;1.6\times10^{-13}$ relative |
| (d) structure | every graph: parsed by the integrator's reader with every vertex of degree 4; 1PI; $E = 4$ or 2 legs, $L$ loops, $L+1$ or $L$ vertices; no self-loop (except in the `_with_selfloops` lists); the Nickel index as listed is canonical for two independent exact canonicalisers (`gaugraph.py` in Python, `listcheck` in C++; both branch and bound), and `listcheck`'s equals the integrator's brute force over all $V!$ relabellings for every graph of every shipped list with $\mathord{\le}\;8$ vertices (to `gamma4_L7` and `gamma2_L8`) and for 126 + 126 graphs of `gamma4_L8`, `gamma2_L9`; all strings distinct, hence no two graphs isomorphic |
| weights and flags | in every list with the `rebar_cpu gen` columns, the $S$ column (for the new lists: the generator's automorphism count, and GAU's weight formula through `gaugraph.py`) = the integrator's `sym_factor` printed `%.12g` and the 1VR / self-loop flags = its `is_1vr` / `has_selfloop`, for every graph; the integrator's `on_factor` is bit-identical to the exact counts for every graph to 8 vertices and every 5th at 9; `gaugraph.py`'s own $\lvert\mathrm{Aut}\rvert$, $O(N)$ counts, GAU weight and Nickel index agree on samples (126 graphs of `gamma4_L8`, 30 of `gamma2_L9`, 25-96 of `gamma2_L8`, 60-235 of `gamma4_L7`) |
| the integrator reads them | `rebar_cpu info` on samples: 8-loop b, h, c classes build with the default limits, and their JSON `S`, `on_poly`, `weight_poly` equal `listcheck`'s. 9 loops: the e16 binary cannot hold them (17 lines) and re-executes its e18 companion, which builds the h and c classes (h needs RB_MAXLOOP 10: $W$ has $L + 1$ loops) |

The 9-loop $\Gamma^{(2)}$ sum rule (derived here, `zerodim_on.py`; the list reproduces it exactly):

$$
\sum_{G \in \Gamma^{(2)}(9)} S_G\,T_G(N) = \frac{778012480 + 956252384\,N + 444020448\,N^2 + 102678120\,N^3 + 12884816\,N^4 + 882730\,N^5 + 31028\,N^6 + 473\,N^7 + 2\,N^8}{1259712},
$$

$9443467/5184$ at $N = 1$ (also the value of the independent $N = 1$ formula of `../../../graphs/tools/zerodim_tf.py`, and of `topologies -w`); it
vanishes at $N = -2$, as every $\Gamma^{(2)}$ coefficient must (there the zero-dimensional propagator is free). The derivation:
$\langle r^{2k}\rangle = I(k)/I(0)$ with $I(k) = \sum_n (-g/24)^n/n!\; m_0^{-2(k+2n)} \prod_{i<k+2n}(N+2i)$, $G = \langle r^2\rangle/N$,
$A = \langle r^4\rangle/(N(N+2))$, $\Gamma^{(2)} = 1/G = 1 + \sum_L (-1)^{L+1} c_L g^L$, $\Gamma^{(4)} = -3(A - G^2)/G^4 = \sum_L (-1)^L c_L g^{L+1}$; bare
$m_0^2 = 1$, tadpole-free $m_0^2 = 1 - g(N+2)/6$ (the mass shift that cancels the one-loop tadpole, so that every self-loop graph
cancels against a counterterm graph and the self-loop-free graphs remain).
