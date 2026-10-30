# Paired FP64/FP80 evaluation (check N6)

`pairs.tsv` holds, for each of 55 classes, its list line and the output of

    rebar_cpu[-e18|-e20] check --what fp --kind K --loops L --list <that line> --n 200000

run on 17 October 2026 with version 0.4.2/0.4.3 (build signatures ca352d68edca3efe, f56a68de64161b7e, 81c3bb8b3e25914c,
the production ones): the same 2×10⁵ points of the class's sampling density, Philox seed 99, evaluated in double and in
x87 extended precision. The classes are the 43 with the largest round-off indicators of the orders from eight to ten
loops (every class with an rms indicator above 10⁻³ or a single-sample indicator above 1, as recorded in the ledger)
and the two largest of each order at six and seven loops as controls. Check N6 re-analyses these lines against the
ledger and recomputes the four six-loop controls live.
