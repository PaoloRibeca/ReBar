# results/coefficients/2026-10-18 — the polynomials p_X,k(N) and their covariance

The numerator polynomials of the RG functions in the form of the paper's Eq. (series-form),

    beta     = -ubar + ubar^2 + sum_{k>=3} p_beta,k(N)     ubar^k / (N+8)^(k-1)
    eta      =                  sum_{k>=2} p_eta,k(N)      ubar^k / (N+8)^k
    eta_phi2 =                  sum_{k>=1} p_eta_phi2,k(N) ubar^k / (N+8)^k
    nu       = 1/2            + sum_{k>=1} p_nu,k(N)       ubar^k / (N+8)^k

(ubar = (N+8) g/(48 pi)), with p_X,k(N) = sum_j a_j N^j, for every term the ledger gives (beta to k = 11, the others to
k = 10), with the covariance of all their coefficients. Tables tab:newpoly and tab:newpolynu of the paper print the new
and improved terms (beta_9-11, eta_8-10, eta_phi2,8-10, nu_8-10) from the same numbers.

Written by `series/polynomials.py` from the ledger of 18 October 2026 (the second six-loop refinement merged; latest
run 2026-10-18 10:05 UTC; `ledger/orders.json` SHA-256 `92f97b26…4006d06`) and `integrator/tools/refs.py`, with

    python3 -I series/polynomials.py --tex paper/generated/newpoly.tex \
        --json results/coefficients/2026-10-18/polynomials.json --tsv results/coefficients/2026-10-18/polynomials.tsv \
        > results/coefficients/2026-10-18/polynomials.txt

(`make -C series` writes the same files to `series/generated/`). Like the other snapshots, these are copies, not
sources: regenerate them when the ledger moves.

| File | Content |
|---|---|
| `polynomials.json` | everything below, as JSON (format `rebar-polynomials-1`) |
| `polynomials.tsv` | the coefficients as plain text: one row per coefficient (label, function, k, power of N+8, j, value, error) followed by its covariance with every coefficient, one column each, headed by their labels |
| `polynomials.txt` | the report: sources, verification, every polynomial with its errors, the large-N test |

## How they are computed

The Callan–Symanzik algebra of `series/coefficients.py` is applied to polynomials in N directly, in exact rational
arithmetic, instead of interpolating values at integer N (which is ill-conditioned). Each per-order sum enters as a
polynomial from the source `coefficients.py` uses (`"policy": "best"`): exact for B1, B2, H2, C1, C2; Phi4tools
through `refs.py` for B3–B6, H3–H5, C3–C6; the ledger for B7–B10, H6–H10, C7–C10. A ledger sum is
X_L(N) = sum_c w_c(N) v_c over its classes, with the exact weight fractions. Errors are propagated linearly:

    C = J_ledger S J_ledger^T + J_P J_P^T

with S block-diagonal over the ledger orders, each block sum_c w_cj w_cj' e_c^2 (the classes are independent, and so
are different orders), and J_P the response to the uncertain Phi4tools coefficients (beta_5–8, eta_4–8, nu_4–8),
each taken as an independent unit Gaussian times its quoted error polynomial. The Phi4tools part is at most 3e-4 of
the variance of any coefficient of the new terms. The Jacobian is exact to rounding (complex step, checked against
exact central differences).

Checks of the numbers (in `polynomials.txt` and `"verification"`): evaluated at N = 0–4 the polynomials reproduce
`coefficients.py` to 3e-14 relative (the difference is the rounding of its sums at fixed N); their errors from the
ledger agree with its finite-difference propagation to 1e-8; the same algebra in floating point agrees with the exact
one to 3e-14; with Phi4tools' sums alone the algebra returns Phi4tools' polynomials exactly. Check S5 tests the
large-N structure (below), check S2 the regeneration of the paper's tables.

## `polynomials.json`

| Key | Content |
|---|---|
| `format`, `description`, `policy`, `ledger_date` | `rebar-polynomials-1`; the formulas above; the source policy; the latest run in the ledger (UTC) |
| `sums` | per order (`b1` … `c10`): `source` (`exact`, `refs`, `ledger`); for ledger orders `poly` (coefficients of N^0, N^1, …), `cov` (their covariance), `classes`, `weight_polynomials` (distinct ones); for exact ones `poly` as fractions |
| `phi4tools_inputs` | the uncertain Phi4tools coefficients (`beta_5`, …, `nu_8`) and the numerators of their quoted error polynomials |
| `polynomials` | one entry per p_X,k, in the order of `index`: `name` (`beta_9`), `function`, `k`, `denominator` (`(N+8)^8`), `degree`, `first_index` (its first coefficient in `index`), `in_table` (printed in the paper), `coefficients` a_0 … a_degree, `errors`, `errors_ledger`, `errors_phi4tools` (the two parts, in quadrature), `exact` (the exact rational coefficients when none has an error) |
| `index` | the label of every coefficient of the global vector, `name:j` (`beta_9:7` is the coefficient of N^7 of p_beta,9) |
| `covariance` | the covariance matrix of that vector, one row per line |
| `verification` | the comparisons above, as numbers |
| `large_n` | the large-N test of check S5: `one_over_N`, the coefficients of ubar^k of X_1 in X = X_inf + X_1/N + O(1/N^2) for beta, eta, eta_phi2, nu; per ledger order (`sums`) the degree of its weight polynomials, the N^(L-1) coefficient of the sum, its error, the 1/N prediction and the pull; `chi2`, `orders`, `p`, `largest_pull`; `phi4tools_largest_relative_difference` (the 1/N expansion against the leading coefficients of Phi4tools' polynomials) |

To evaluate a function at some N with its error: for p = sum_j a_j N^j at that N, the variance is v^T C v with
v_j = N^j / (N+8)^power over the coefficients of that polynomial (the block of `covariance` from `first_index`).
Coefficients of different polynomials are correlated through shared inputs (e.g. B9 enters beta_10 and beta_11, H9
enters every function at order 9 and above): any joint use (a resummation, a fit) needs the cross-covariances, which
are in the same matrix. The matrix is rank-deficient (its rank is at most the number of independent inputs: 108
coefficients of the ledger sums and 14 Phi4tools errors, for 229 coefficients): use it as a covariance, not through its
inverse.

Exact coefficients have zero variance: all those of the terms that only exact values and Phi4tools coefficients
without quoted errors determine (beta_3, beta_4, eta_2, eta_3, nu_1–3, eta_phi2,1–3; Phi4tools' decimal constants enter
as the binary fractions of their doubles), and the leading coefficients that the limit N -> infinity fixes:
[N^k] p_nu,k = 2^-(k+1), and the absent powers (p_beta,k has degree k-2, p_eta,k and p_eta_phi2,k degree k-1).

## Credits and licence

The credits are in [`../../../README.md`](../../../README.md). The data are under CC BY 4.0
([`../../../LICENSE-DATA.md`](../../../LICENSE-DATA.md)).
