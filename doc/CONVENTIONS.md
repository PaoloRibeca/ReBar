# Conventions and normalisations

The conventions behind ReBar's integrals, per-order sums and series. The paper states the same, with references, in
Sec. II A and Appendix A ([`paper/sections/methods.tex`](../paper/sections/methods.tex),
[`paper/sections/appendices.tex`](../paper/sections/appendices.tex)); a change to one should be made to the other.

## Model and scheme

$d = 3$, Euclidean action for an $N$-component field

$$
S[\phi]=\int d^3x\,\Bigl[\tfrac12(\partial_\mu\phi)^2+\tfrac12 m_0^2\,\phi^2+\tfrac{\lambda_0}{4!}\,(\phi^2)^2\Bigr].
$$

$\Gamma^{(a,b)}$ are the one-particle-irreducible vertex functions with $a$ legs and $b$ insertions of $\tfrac12\phi^2$ at
zero momentum, with the $O(N)$ tensors factored out (for four legs $(\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})/3$).
Massive zero-momentum scheme (Parisi; Baker, Nickel, Green and Meiron): the renormalised mass $m$, the normalisations
of $\phi$ and $\phi^2$ and the dimensionless coupling $g$ are fixed by

$$
\Gamma^{(2)}_R(p)=m^2+p^2+O(p^4),\qquad \Gamma^{(4)}_R(0)=m\,g,\qquad \Gamma^{(2,1)}_R(0;0)=1 ,
$$

and $\partial\Gamma^{(a,b)}/\partial m_0^2=\Gamma^{(a,b+1)}$ at zero insertion momentum, so that, for example, $C_1>0$.

## Couplings: $u$ and $\bar u$

The integrals are expanded in $u = g/8\pi$, one power per loop. The RG functions are quoted in

$$
\bar u=\frac{N+8}{6}\,u=\frac{N+8}{48\pi}\,g ,\qquad
\beta(\bar u)\equiv m\frac{\partial\bar u}{\partial m}=\frac{N+8}{6}\,\beta_u ,\qquad \beta_u = m\frac{\partial u}{\partial m},
$$

the normalised coupling of Baker, Nickel and Meiron, which is Phi4tools' $\tilde u$ (and the "Nickel normalised"
coupling $\hat g$ of the 2003 thesis). With it $\beta(\bar u)=-\bar u+\bar u^2+O(\bar u^3)$ for every $N$.
$\eta_{\phi^2}$ is the anomalous dimension of $\phi^2$ ($\eta_2$ in the thesis), so that $1/\nu=2-\eta+\eta_{\phi^2}$, as in
Phi4tools.

## Normalisation of the integrals

* **Measure.** One $d^3k/\pi^2$ per loop (Nickel), that is $8\pi$ times $d^3k/(2\pi)^3$: each loop comes with one power
  of $u$, the one-loop bubble at zero momentum equals one, and the low-order sums are rational. Integrals with
  $d^3k/(2\pi)^3$ per loop are $(8\pi)^{-L}$ times ours. (In the thesis "Nickel's normalisation" refers to the coupling
  $\bar u$, not to the measure.)
* **Parametric form.** At zero external momenta and unit mass, for a graph $G$ with $\ell$ lines and $L$ loops,

  $$
  A_G=\frac{\Gamma(\ell-3L/2)}{\pi^{L/2}}\int_\Delta d\mu\;U_G^{-3/2},\qquad U_G=\sum_{T}\prod_{e\notin T}x_e ,
  $$

  with $T$ the spanning trees, $\Delta$ the simplex and $d\mu=\delta(1-\sum_e x_e)\prod_e dx_e$. An insertion of
  $\tfrac12\phi^2$ on line $e$ multiplies the integrand by $x_e$ and raises the argument of the Gamma function by one.
  The $p^2$ derivative of a two-point graph at $p=0$ is $-\Gamma(\ell-3L/2+1)\,\pi^{-L/2}\int_\Delta d\mu\,W_G\,U_G^{-5/2}$,
  where $W_G$ is the Kirchhoff polynomial of $G$ with its two external vertices identified. Since $\ell=2L$ for
  $\Gamma^{(4)}$ and $\ell=2L-1$ for $\Gamma^{(2)}$, every integral of a per-order sum carries the same prefactor
  $\kappa_L=\Gamma(L/2+1)\,\pi^{-L/2}$.
* **Masses.** The propagators of $B_L$, $H_L$ and $C_L$ have the renormalised mass, $m=1$. Those of Nickel, Meiron and
  Baker, of the thesis (PARIS, `int4`) and of Phi4tools, and the intermediate-scheme integrals used for comparison
  (`--kind int4`), have the mass of the intermediate scheme, $m_I=1$. Individual integrals differ by powers of
  $m^2/m_I^2$; the RG functions do not.
* **Phi4tools graph values.** Phi4tools' $V_G$ include its weight factor and $(16\pi)^L$ per $L$-loop graph, with the
  measure $d^3k/(2\pi)^3$. At $N=1$ the sum of its intermediate-scheme four-point values at $L$ loops equals
  $(2^L/3)\sum_G S_G I_G$, with $I_G$ in our normalisation and $m_I=1$ (for example 400.1741436 at $L=5$).

## Weights: $S_G$

$$
S_G=T_G(N)\,\frac{E!}{\lvert\mathrm{Aut}\,G\rvert},
$$

with $E$ the number of legs, $\mathrm{Aut}\,G$ the automorphism group at the level of half-edges with unlabelled legs,
and $T_G(N)$ the $O(N)$ factor: the vertex tensors contracted over the $3^V$ pairings at the $V$ vertices and projected
on the external tensor, normalised to $T_G(1)=1$ as Phi4tools' symmetry factors. $E!/\lvert\mathrm{Aut}\,G\rvert$ includes
the sum over the labellings of the legs, which is legitimate at zero external momenta. The integrals do not depend on
$N$: every class carries its polynomial $T_G(N)$, so one run gives every $N$. Whitney twins (2-isomorphic graphs) have
equal integrals; a merged class carries the sum of their weights.

## Callan–Symanzik relations

The renormalised functions obey, for $b\le1$ and at fixed bare parameters,

$$
\Bigl[m\,\partial_m+\beta\,\partial_g-\tfrac a2\,\eta-b\,(\eta_{\phi^2}-\eta)\Bigr]\Gamma^{(a,b)}_R=(2-\eta)\,m^2\,\Gamma^{(a,b+1)}_R ,
$$

with $\beta=m\,\partial g/\partial m$, $\eta=m\,\partial\ln Z/\partial m$, $\eta_{\phi^2}=m\,\partial\ln Z_{\phi^2}/\partial m$.
Applied at zero momenta to the normalisation conditions (the first through its $p^2$ derivative) it gives

$$
\eta=\frac{2h}{1+h},\qquad \beta_u=-u+2\eta\,u+(2-\eta)\,\tilde b,\qquad
\eta_{\phi^2}=-(2-\eta)\,c,\qquad \frac1\nu=(2-\eta)(1-c),
$$

$$
\tilde b=\frac{m\,\Gamma^{(4,1)}_R(0)}{8\pi},\qquad
h=-m^2\,\frac{\partial\Gamma^{(2,1)}_R}{\partial p^2}\bigg|_{p=0},\qquad
c=m^2\,\Gamma^{(2,2)}_R(0),
$$

$$
\tilde b=\sum_{L\ge1}(-1)^{L+1}B_L\,u^{L+1},\qquad h=\sum_{L\ge2}(-1)^L H_L\,u^L,\qquad c=\sum_{L\ge1}(-1)^{L+1}C_L\,u^L .
$$

The three functions are linear in the graphs and need no $Z$ factor, because the subtractions implement the
normalisation conditions graph by graph. Only products and quotients of series enter; no series is reversed.

## The three kinds of sums: `b`, `h`, `c`

$$
\begin{aligned}
B_L&=\kappa_L\sum_{G\in\mathcal G_4(L)}S_G\sum_{e}\int_\Delta d\mu\;x_e\,\bar R_{G,e}\bigl[U_G^{-3/2}\bigr],\\
H_L&=\kappa_L\sum_{G\in\mathcal G_2(L)}S_G\sum_{e}\int_\Delta d\mu\;x_e\,\bar R_{G,e}\bigl[W_G\,U_G^{-5/2}\bigr],\\
C_L&=\kappa_L\sum_{G\in\mathcal G_2(L)}S_G\sum_{(e,e')}\int_\Delta d\mu\;x_e\,x_{e'}\,\bar R_{G,ee'}\bigl[U_G^{-3/2}\bigr],
\end{aligned}
$$

where $\mathcal G_E(L)$ is the list of $L$-loop graphs with $E$ legs, $e$ runs over the lines and $(e,e')$ over the
ordered pairs of lines, $e=e'$ included.

| Kind (`--kind`, `ledger/`) | Sum | Graphs | Insertions | Enters |
|---|---|---|---|---|
| `b` | $B_L$ | $\Gamma^{(4)}$, $L$ loops | one, on any line | $\beta$ at $\bar u^{L+1}$ |
| `h` | $H_L$ | $\Gamma^{(2)}$, $L$ loops | one, and the $p^2$ derivative | $\eta$ at $\bar u^{L}$, $\beta$ at $\bar u^{L+1}$, $\nu$ at $\bar u^{L}$ |
| `c` | $C_L$ | $\Gamma^{(2)}$, $L$ loops | two, ordered pairs of lines | $\eta_{\phi^2}$ and $\nu$ at $\bar u^{L}$ |

$\bar R$ is the zero-momentum forest subtraction: Zimmermann forests of 1PI subgraphs, pairwise nested or
vertex-disjoint; four-point parts without an insertion get $t^0$, two-point parts without an insertion $t^2$ (mass and
field), two-point parts with one insertion $t^0$. One-vertex-reducible graphs, and for $H_L$ the generalised tadpoles,
give identically vanishing integrands and are dropped from the lists. The lowest sums are exact,

$$
B_1=\frac{N+8}{12},\qquad B_2=\frac{5N+22}{54},\qquad H_1=0,\qquad H_2=\frac{N+2}{243},\qquad C_1=\frac{N+2}{12},\qquad C_2=\frac{N+2}{36},
$$

that is $3/4$, $1/2$, $1/81$, $1/4$ and $1/12$ at $N=1$.

## The series: Phi4tools normalisation

$$
\begin{aligned}
\beta(\bar u)&=-\bar u+\bar u^2+\sum_{k\ge3}\frac{p_{\beta,k}(N)}{(N+8)^{k-1}}\,\bar u^{k},\\
\eta(\bar u)&=\sum_{k\ge2}\frac{p_{\eta,k}(N)}{(N+8)^{k}}\,\bar u^{k},\\
\nu(\bar u)&=\frac12+\sum_{k\ge1}\frac{p_{\nu,k}(N)}{(N+8)^{k}}\,\bar u^{k},
\end{aligned}
$$

with polynomials $p_{X,k}(N)$. Writing $[\bar u^k]F$ for the coefficient of $\bar u^k$ in $F$, at each order

$$
\begin{aligned}
[\bar u^{L+1}]\beta&=2(-1)^{L+1}\Bigl(\frac{6}{N+8}\Bigr)^{L}\bigl(B_L-2H_L\bigr)+\dots,\\
[\bar u^{L}]\eta&=2(-1)^{L}\Bigl(\frac{6}{N+8}\Bigr)^{L}H_L+\dots,\\
[\bar u^{L}]\eta_{\phi^2}&=2(-1)^{L}\Bigl(\frac{6}{N+8}\Bigr)^{L}C_L+\dots,\\
[\bar u^{L}]\nu&=\frac{(-1)^{L+1}}{2}\Bigl(\frac{6}{N+8}\Bigr)^{L}\bigl(C_L-H_L\bigr)+\dots,
\end{aligned}
$$

where the dots are products of lower-order sums. The errors of the new sums therefore enter linearly, with these
prefactors. The reference values of $B_L$, $H_L$, $C_L$ are obtained by solving these relations order by order for the
sums, starting from Phi4tools' series. The exact sums give

$$
\begin{aligned}
\beta(\bar u)&=-\bar u+\bar u^2-\frac{4(41N+190)}{27(N+8)^2}\,\bar u^3+O(\bar u^4),\\
\eta(\bar u)&=\frac{8(N+2)}{27(N+8)^2}\,\bar u^2+O(\bar u^3),\\
\eta_{\phi^2}(\bar u)&=-\frac{N+2}{N+8}\,\bar u+\frac{2(N+2)}{(N+8)^2}\,\bar u^2+O(\bar u^3),\\
\nu(\bar u)&=\frac12+\frac{N+2}{4(N+8)}\,\bar u+\frac{(N+2)(27N-38)}{216(N+8)^2}\,\bar u^2+O(\bar u^3),
\end{aligned}
$$

in agreement with Phi4tools: $p_{\beta,3}=-4(41N+190)/27$, $p_{\eta,2}=8(N+2)/27$, $p_{\nu,1}=(N+2)/4$,
$p_{\nu,2}=(N+2)(27N-38)/216$. At $N=1$ the coefficients are $-308/729$ ($\bar u^3$ in $\beta$), $8/729$ ($\bar u^2$ in
$\eta$), $-1/3$ and $2/27$ ($\eta_{\phi^2}$), $1/12$ and $-11/5832$ ($\nu$).

## References

* G. Sberveglieri and G. Spada, *Phi4tools: compilation of Feynman diagrams for Landau–Ginzburg–Wilson theories*,
  JHEP 05 (2024) 073, arXiv:2311.12576 (the series and graph lists called Phi4tools above).
* B. G. Nickel, D. I. Meiron and G. A. Baker, Jr., *Compilation of 2-pt. and 4-pt. graphs for continuous spin models*,
  University of Guelph report (1977, unpublished).
* G. A. Baker, Jr., B. G. Nickel, M. S. Green and D. I. Meiron, Phys. Rev. Lett. 36 (1976) 1351; G. A. Baker, Jr.,
  B. G. Nickel and D. I. Meiron, Phys. Rev. B 17 (1978) 1365; G. Parisi, J. Stat. Phys. 23 (1980) 49.
* P. Ribeca, *Precise evaluation of universal quantities in second-order phase transitions*, PhD thesis, Università
  di Genova and Université Paris-Sud (2003): the RG equation, the weights and $O(N)$ factors, the zero-dimensional
  sum rules.
