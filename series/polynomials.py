#!/usr/bin/env python3
"""
    polynomials.py -- (c) 2026 Paolo Ribeca, <paolo.ribeca@gmail.com>

    The numerator polynomials p_X,k(N) of the RG functions in Phi4tools' form (paper, Eq. series-form),
        beta = -ubar + ubar^2 + sum_k p_beta,k(N) ubar^k / (N+8)^(k-1),
        eta = sum_k p_eta,k(N) ubar^k / (N+8)^k,   eta_phi2 = sum_k p_eta_phi2,k(N) ubar^k / (N+8)^k,
        nu = 1/2 + sum_k p_nu,k(N) ubar^k / (N+8)^k,
    with the covariance of all their coefficients, computed on polynomials in N directly instead of by interpolation
    at N = 0, 1, 2, ... (coefficients.py's numerator(), which is ill-conditioned: an ulp in the values moves the small
    high powers by up to 1e-5 relative, and it leaves noise where a power is absent).

    Inputs: per order the source coefficients.py chooses (--source as there; best: exact for B1, B2, H2, C1, C2, else
    the smaller error at N = 1 of Phi4tools, through integrator/tools/refs.py, and of the ledger), as polynomials in N:
      ledger  X_L(N) = sum_c w_c(N) v_c over the pooled classes (rebar_ledger.iter_classes): the coefficient of N^j is
              the sum over the distinct weight polynomials W of W_j times the sum of the values of the classes that
              have it, the W_j exact fractions (rebar_ledger.weight_fractions, each equal to the ledger's double) and
              the sums of values correctly rounded (math.fsum), added in exact rational arithmetic; its covariance is
              sum_c w_cj w_cj' e_c^2 (the classes are independent), different orders independent;
      exact   rebar_ledger.exact_low with N a polynomial with rational coefficients;
      refs    refs.py's beta_coeffs, eta_coeffs, nu_coeffs and derive run with N a rational function of N (class RN,
              exact arithmetic on refs.py's binary constants): B_L, H_L, C_L as polynomials.  Their error model: each
              uncertain Phi4tools coefficient (beta_5..8, eta_4..8, nu_4..8) is an independent unit Gaussian times its
              quoted error polynomial, i.e. the errors of the powers of N of one published polynomial are added
              linearly, as refs.py does at fixed N, and different published coefficients are independent.
    Algebra: coefficients.py's own rg() run on series whose coefficients are polynomials (class Poly; N = -2 there,
    so that k = (N+8)/6 = 1 and rg returns the series in u, whose u^k coefficient times 6^(k-1) for beta, 6^k for the
    others, is p_X,k), in exact rational arithmetic for the values (one rounding at the end) and in complex floating
    point for the Jacobian (complex step: each input moved by i 1e-30, the derivative exact to rounding).
    Errors: C = J_ledger S J_ledger^T + J_Phi4tools J_Phi4tools^T, S block-diagonal over the ledger orders.

    Verification (printed): the values at N = 0..4 against coefficients.py's (its own procedure on the same inputs),
    the errors at N = 0..4 against its finite-difference propagation (the ledger part must agree; the Phi4tools part
    differs by the error model), the exact algebra against the same algebra in floating point, the complex-step
    Jacobian against exact central differences, the Phi4tools round trip in exact arithmetic (refs.py's sums give
    back refs.py's polynomials exactly), and coefficients.py's interpolation against these polynomials.

    Large N (check S5).  At N = infinity with ubar fixed the zero-momentum 1PI functions are bubble chains: with
    x = N g0/(48 pi m) (g0 the bare coupling, m the renormalized mass), ubar = x/(1+x), Z = 1, Gamma^(2,1)(0;0) =
    1/(1+x), so that beta = -x d ubar/dx = -ubar + ubar^2, eta = 0, eta_phi2 = -ubar, nu = 1/(2 - ubar): the leading
    coefficients of the polynomials vanish or are 2^-(k+1) (p_beta,k of degree k-2, p_eta,k and p_eta_phi2,k of degree
    k-1, [N^k] p_nu,k = 2^-(k+1)), equivalently [N^L] B_L = [N^L] C_L = 0 (L >= 2), [N^L] H_L = 0.  At order 1/N
    (large_n; m = 1, int_q = int d^3q/(2 pi)^3, Pi = arctan(q/2)/(4 pi q) the bubble, G = 1/(q^2+1), D = 1 + 8 pi x Pi):
        Z = 1 + z/N,  z = -(8 pi/3) int_q x (2q^2 - 6) G^3/D        (the sigma exchange, d/dp^2 at p = 0),
        ubar = (1 + 8/N) x/(1+x) [1 + (2z + S_b)/N],   S_b = -x z/(2(1+x)) + int_q B,
        Z_2 Z Gamma^(2,1)(0;0) = 1,   ln Z_2 = ln(1+x) + S_n/N,   S_n = x z/(2(1+x)) - z + int_q V,
        B = 8 pi x^2 A/((1+x) D) - 32 pi x G^2/D - 32 pi x G^2 (q^2+4+3x)^2/((1+x)(q^2+4)^2 D^2),
        V = -8 pi x^2 A/((1+x) D) + 16 pi x G^2/D - 32 pi x^2 G (q^2+4+3x)/((1+x)(q^2+4)^2 D^2),
        A = 8/(q^2+4)^2 - 3/((q^2+4)(q^2+1)),
    from Gamma^(4)(0) and Gamma^(2)(0) as derivatives of the effective potential at order 1/N,
    U_1 = 1/2 int_q ln[1 + (N g0/6) Pi_M(q) + (g0/3) phi^2 G_M(q)] at the leading gap mass M(phi), re-expressed
    through m (the power divergences cancel between the two: B and V decay as 1/q^4); then beta = -x d ubar/dx,
    eta = -x d ln Z/dx, eta_2 = -x d ln Z_2/dx, 1/nu = 2 + eta_2, eta_phi2 = eta_2 + eta (so that 1/nu = 2 - eta +
    eta_phi2), re-expanded in ubar at order 1/N.  The integrals are done by double-exponential quadrature, as power
    series in x at each node (precision about 1e-10 relative at order 8, 1e-7 at order 11).  These fix the two leading
    coefficients of every p_X,k, and through the inverse Callan-Symanzik relations the N^(L-1) coefficient of every
    sum; they reproduce the leading coefficients of Phi4tools' polynomials (beta, eta and nu through order 8) and are
    compared with the N^(L-1) coefficient of every order of the ledger, with its error (large_n_tests).

    usage: polynomials.py [--ledger DIR] [--source best|ledger|refs] [--tex FILE] [--json FILE] [--tsv FILE] [--quiet]
      --tex    paper/generated/newpoly.tex: the bodies of Tables tab:newpoly and tab:newpolynu (macros \\newpolybetaeta,
               \\newpolyphinu) and the numbers of the large-N test that the text quotes (\\largeN...)
      --json   the polynomials, their errors and their full covariance, the per-order sums and the large-N test
               (format: results/coefficients/2026-10-18/README.md)
      --tsv    the coefficients as plain text: one row per coefficient, with its covariance with every coefficient
    The report (stdout) gives the sources, the verification, every polynomial with its errors (from the ledger and
    from Phi4tools) and the large-N test.  Python >= 3.6, standard library only.

    This program was designed and developed by the author(s),
    with the assistance of the following AI tool(s):
      2026 Claude (Anthropic).
    The final logic and implementation were reviewed and verified in
    their entirety by the author(s).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import math
import os
import sys
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rebar_ledger as RL  # noqa: E402
import coefficients as CO  # noqa: E402

sys.path.insert(0, RL.TOOLS)
import refs as RF  # noqa: E402  (integrator/tools/refs.py)
sys.path.pop(0)

FUNCS = CO.FUNCS
NS = (0, 1, 2, 3, 4)
FIRST = {"beta": 3, "eta": 2, "eta_phi2": 1, "nu": 1}     # the first term of each sum in Eq. series-form
TABLE = [("beta", 9), ("beta", 10), ("beta", 11), ("eta", 8), ("eta", 9), ("eta", 10),
         ("eta_phi2", 8), ("eta_phi2", 9), ("eta_phi2", 10), ("nu", 8), ("nu", 9), ("nu", 10)]
STEP = 1e-30                                              # complex step
TEXF = {"beta": r"\beta", "eta": r"\eta", "eta_phi2": r"\eta_{\phi^2}", "nu": r"\nu"}


def pw(f, n):
    """the power of N+8 under p_f,n"""
    return n - 1 if f == "beta" else n


# ------------------------------------------------------------------------------------------------- polynomials
class Poly:
    """A polynomial in N, c[j] the coefficient of N^j (floats or complex numbers; PolyQ: fractions), with the
    arithmetic that coefficients.rg, refs.derive and rebar_ledger.exact_low apply to their series coefficients:
    + - * with polynomials and numbers, / by a number, truth value.  Trailing exact zeros are dropped."""
    __slots__ = ("c",)
    num = staticmethod(lambda x: x)

    def __init__(self, c):
        c = list(c)
        while c and not c[-1]:
            c.pop()
        self.c = c

    def _co(self, x):
        return x if isinstance(x, Poly) else type(self)([self.num(x)])

    def __add__(a, b):
        x, y = a.c, a._co(b).c
        if len(x) < len(y):
            x, y = y, x
        return type(a)([p + q for p, q in zip(x, y)] + x[len(y):])
    __radd__ = __add__

    def __neg__(a):
        return type(a)([-p for p in a.c])

    def __sub__(a, b):
        return a + (-a._co(b))

    def __rsub__(a, b):
        return a._co(b) + (-a)

    def __mul__(a, b):
        if not isinstance(b, Poly):
            b = a.num(b)
            return type(a)([p * b for p in a.c])
        x, y = a.c, b.c
        if not x or not y:
            return type(a)([])
        r = [x[0] * 0] * (len(x) + len(y) - 1)
        for i, p in enumerate(x):
            for j, q in enumerate(y):
                r[i + j] += p * q
        return type(a)(r)
    __rmul__ = __mul__

    def __truediv__(a, b):
        b = a.num(b)
        return type(a)([p / b for p in a.c])

    def __bool__(a):
        return bool(a.c)

    def __call__(a, N):
        s = a.c[-1] * 0 if a.c else 0
        for p in reversed(a.c):
            s = s * N + p
        return s

    def coef(a, deg):
        """the coefficients of N^0 .. N^deg (zeros above the stored ones)"""
        return a.c[:deg + 1] + [a.c[0] * 0 if a.c else 0] * (deg + 1 - len(a.c))


class PolyQ(Poly):
    """Poly with exact rational coefficients (numbers enter as Fraction(x): a double exactly)"""
    __slots__ = ()
    num = staticmethod(Fr)


NQ = PolyQ([0, 1])                                        # the variable N


def n8(k):
    """(N+8)^k as a PolyQ, k >= 0"""
    p = PolyQ([1])
    for _ in range(k):
        p = p * PolyQ([8, 1])
    return p


def as_poly(x, cls=PolyQ):
    return x if isinstance(x, Poly) else cls([cls.num(x)])


class RN:
    """p(N) (N+8)^(-m) with p a PolyQ not divisible by N+8 and m any integer: what refs.py builds from N with
    + - * / and integer powers (its divisions are by numbers and by powers of N+8), exactly"""
    __slots__ = ("p", "m")

    def __init__(self, p, m=0):
        p = as_poly(p)
        while p.c and p(-8) == 0:                         # divide out N + 8 (synthetic division)
            d = len(p.c) - 1
            q = [Fr(0)] * d
            q[d - 1] = p.c[d]
            for j in range(d - 1, 0, -1):
                q[j - 1] = p.c[j] - 8 * q[j]
            p, m = PolyQ(q), m - 1
        self.p, self.m = p, (m if p.c else 0)

    @staticmethod
    def _co(x):
        return x if isinstance(x, RN) else RN(PolyQ([Fr(x)]))

    def __add__(a, b):
        b = RN._co(b)
        m = max(a.m, b.m)
        return RN(a.p * n8(m - a.m) + b.p * n8(m - b.m), m)
    __radd__ = __add__

    def __neg__(a):
        return RN(-a.p, a.m)

    def __sub__(a, b):
        return a + (-RN._co(b))

    def __rsub__(a, b):
        return RN._co(b) + (-a)

    def __mul__(a, b):
        b = RN._co(b)
        return RN(a.p * b.p, a.m + b.m)
    __rmul__ = __mul__

    def __truediv__(a, b):
        b = RN._co(b)
        if len(b.p.c) != 1:
            raise ValueError("RN: division by a polynomial other than c (N+8)^k")
        return RN(a.p / b.p.c[0], a.m - b.m)

    def __rtruediv__(a, b):
        return RN._co(b) / a

    def __pow__(a, e):
        r = RN(1)
        for _ in range(e):
            r = r * a
        return r

    def times_n8(a, k):
        """this times (N+8)^k, as a PolyQ (it must be a polynomial)"""
        if k - a.m < 0:
            raise ValueError("RN: not a polynomial")
        return a.p * n8(k - a.m)


# ------------------------------------------------------------------------------------------------- inputs
def exact_inputs():
    """{(kind, L): PolyQ}: the exact low orders, rebar_ledger.exact_low with N a polynomial"""
    return {k: as_poly(v) for k, v in RL.exact_low(NQ).items()}


_REFS = []


def refs_inputs():
    """Phi4tools through refs.py, exactly: ({(kind, L): PolyQ}, [(label, sigma numerator, {(kind, L): PolyQ})]) --
    the sums B_L, H_L, C_L and, for every uncertain Phi4tools coefficient, its quoted error polynomial (the numerator
    over the power of N+8 of its term) and the derivative of the sums with respect to its unit Gaussian (a central
    difference in exact arithmetic with step 2^-40: refs.derive is polynomial in its inputs, so it is exact up to
    2^-80 relative)"""
    if _REFS:
        return _REFS[0]
    N = RN(NQ)
    cs = (RF.beta_coeffs(N), RF.eta_coeffs(N), RF.nu_coeffs(N))
    base = [[x[0] for x in c] for c in cs]

    def sums(vals):
        B, H, C = RF.derive(N, *vals)
        out = {}
        for kind, d in (("b", B), ("h", H), ("c", C)):
            for L, v in d.items():
                out[(kind, L)] = v.times_n8(0) if isinstance(v, RN) else as_poly(v)
        return out
    ref = sums(base)
    t = Fr(1, 2 ** 40)
    eps = []
    for w, (f, c) in enumerate(zip(("beta", "eta", "nu"), cs)):
        for i, (v, s) in enumerate(c):
            if not isinstance(s, RN) or not s.p.c:
                continue
            up, dn = [list(x) for x in base], [list(x) for x in base]
            up[w][i], dn[w][i] = v + t * s, v - t * s
            a, b = sums(up), sums(dn)
            eps.append(("%s_%d" % (f, i), s.times_n8(pw(f, i)), {k: (a[k] - b[k]) / (2 * t) for k in ref}))
    _REFS.append((ref, eps))
    return ref, eps


def ledger_sums(ledger, Ns=NS):
    """{(kind, L): order} for the ledger's orders with data, one pass over the classes of each: 'poly' the sum
    polynomial (exact fractions), 'cov' its covariance (floats), 'atN' {N: (value, error)} exactly as
    coefficients.at_N computes them (same operations, same order), 'classes', 'groups' (distinct weight
    polynomials), 'inexact' (groups whose weights could not be recovered as fractions; then their doubles), 'deg' (the
    largest degree of the classes' weight polynomials), 'versions'.  The exact weights are those of
    rebar_ledger.weight_fractions (S c_k / 3^(L+4), with the further condition that each equals the ledger's double)"""
    orders = RL.load_orders(ledger)
    out = {}
    for o in sorted(orders["orders"], key=RL._key):
        e = orders["orders"][o]
        if not e.get("contributions"):
            continue
        kind, L = e["kind"], e["L"]
        groups, acc, n = {}, {N: [0.0, 0.0] for N in Ns}, 0
        for r in RL.iter_classes(ledger, o, fields=("weight_poly", "value", "error", "S")):
            wp = r["weight_poly"]
            key = tuple(wp)
            g = groups.get(key)
            if g is None:
                g = groups[key] = (r["S"], [], [], [RL.poly(wp, N) for N in Ns])
            v, er = r["value"], r["error"]
            g[1].append(v)
            g[2].append(er * er)
            for N, w in zip(Ns, g[3]):
                a = acc[N]
                a[0] += w * v
                a[1] += (w * er) ** 2
            n += 1
        deg = max(len(k) for k in groups)
        D, SCALE = 3 ** (L + 4), 1100                     # weights S c_k / D; doubles n / 2^k, k <= 1074
        sfr, num, other, rows, inexact = {}, {}, [Fr(0)] * deg, [], 0
        for key in sorted(groups):
            Sg, vals, e2, _ = groups[key]
            if Sg not in sfr:                             # S as a fraction (rebar_ledger.weight_fractions)
                q = Fr(Sg).limit_denominator(1 << 40) if Sg else None
                sfr[Sg] = q if q is not None and float(q) == Sg else None
            q, c = sfr[Sg], None
            if q is not None:                             # its exact weights, each equal to the ledger's double
                c = [int(round(x / Sg * D)) for x in key]
                if sum(c) != D or any(abs(x / Sg * D - ci) > 1e-3 or q.numerator * ci / (q.denominator * D) != x
                                      for x, ci in zip(key, c)):
                    c = None
            nv, dv = math.fsum(vals).as_integer_ratio()
            if c is None:                                 # not recovered: the doubles
                inexact += 1
                for i, x in enumerate(key):
                    other[i] += Fr(x) * Fr(nv, dv)
            else:                                         # sum c_k nv/dv exactly, as an integer over 2^SCALE
                tot, m = num.setdefault(q, [0] * deg), nv << (SCALE - dv.bit_length() + 1)
                for i, ci in enumerate(c):
                    tot[i] += ci * m
            rows.append((list(key) + [0.0] * (deg - len(key)), math.fsum(e2)))
        S = list(other)
        for q, tot in num.items():
            for i in range(deg):
                S[i] += q * Fr(tot[i], D << SCALE)
        cov = [[0.0] * deg for _ in range(deg)]
        for j in range(deg):
            for jj in range(j, deg):
                cov[j][jj] = cov[jj][j] = math.fsum(w[j] * w[jj] * s for w, s in rows)
        out[(kind, L)] = {"order": o, "poly": PolyQ(S), "deg": deg - 1, "cov": cov,
                          "atN": {N: (acc[N][0], math.sqrt(acc[N][1])) for N in Ns}, "classes": n,
                          "groups": len(groups), "inexact": inexact, "versions": e.get("versions", [])}
    return out


def choose(led, policy):
    """coefficients.choose on the ledger's errors at N = 1 (its at_N of a single class of weight 1 returns them)"""
    fake = {k: ([{"weight_poly": [1.0], "value": 0.0, "error": d["atN"][1][1]}], d["versions"]) for k, d in led.items()}
    return CO.choose(fake if policy != "refs" else {}, policy)


# ------------------------------------------------------------------------------------------------- algebra
def numerators(vals, avail, cls):
    """{(f, n): Poly p_f,n} from {(kind, L): Poly}: coefficients.rg on the polynomial series at N = -2 (k = 1, the
    series in u), times 6^(power of N+8)"""
    out = CO.rg(vals, -2)
    res = {}
    for f in FUNCS:
        for n in range(1, avail[f] + 1):
            res[(f, n)] = as_poly(out[f][n], cls) * (6 ** pw(f, n))
    return res


def to_complex(p):
    return Poly([complex(float(x)) for x in p.c])


# ------------------------------------------------------------------------------------------------- the computation
def compute(ledger=RL.LEDGER, policy="best", led=None, check=True):
    """The polynomials of every term of Eq. series-form that the sources allow, their Jacobian and covariance, and
    (check) the verification against coefficients.py.  led: the result of ledger_sums (to reuse one pass)"""
    if led is None:
        led = ledger_sums(ledger) if policy != "refs" else {}
    src = choose(led, policy)
    avail = CO.available(set(src))
    ref, eps = refs_inputs()
    ex = exact_inputs()
    vals = {k: led[k]["poly"] if s == "ledger" else ex[k] if s == "exact" else ref[k] for k, s in src.items()}
    exact = numerators(vals, avail, PolyQ)
    terms = [(f, n) for f in FUNCS for n in range(FIRST[f], avail[f] + 1)]
    deg = {t: len(exact[t].c) - 1 for t in terms}
    index = [(f, n, j) for (f, n) in terms for j in range(deg[(f, n)] + 1)]
    nout = len(index)

    # floating point (complex) evaluation and the Jacobian by complex step
    base = {k: to_complex(v) for k, v in vals.items()}

    def run(v):
        r = numerators(v, avail, Poly)
        return [x for t in terms for x in r[t].coef(deg[t])]
    f0 = [x.real for x in run(base)]
    ledkeys = sorted(k for k, s in src.items() if s == "ledger")
    refkeys = sorted(k for k, s in src.items() if s == "refs")
    cols = []                                             # (kind of input, its name, its sigma, column)
    for key in ledkeys:
        for j in range(len(led[key]["cov"])):
            v = dict(base)
            v[key] = base[key] + Poly([0j] * j + [1j * STEP])
            cols.append(("ledger", (key, j), math.sqrt(led[key]["cov"][j][j]), [x.imag / STEP for x in run(v)]))
    if refkeys:
        for lab, sig, d in eps:
            v = dict(base)
            for k in refkeys:
                v[k] = base[k] + to_complex(d[k]) * (1j * STEP)
            cols.append(("phi4tools", lab, 1.0, [x.imag / STEP for x in run(v)]))

    # covariance: the ledger orders' blocks and the Phi4tools inputs
    Cl, Cp = [[0.0] * nout for _ in range(nout)], [[0.0] * nout for _ in range(nout)]
    for key in ledkeys:
        V = led[key]["cov"]
        J = [c[3] for c in cols if c[0] == "ledger" and c[1][0] == key]
        nz = [a for a in range(nout) if any(col[a] for col in J)]
        M = {a: [sum(V[j][jj] * J[jj][a] for jj in range(len(J))) for j in range(len(J))] for a in nz}
        for ia, a in enumerate(nz):
            Ma = M[a]
            for b in nz[ia:]:
                s = sum(Ma[j] * J[j][b] for j in range(len(J)))
                Cl[a][b] += s
                if b != a:
                    Cl[b][a] += s
    for kind, lab, sig, col in cols:
        if kind != "phi4tools":
            continue
        nz = [a for a in range(nout) if col[a]]
        for ia, a in enumerate(nz):
            for b in nz[ia:]:
                s = col[a] * col[b]
                Cp[a][b] += s
                if b != a:
                    Cp[b][a] += s
    C = [[x + y for x, y in zip(r, q)] for r, q in zip(Cl, Cp)]
    val = [float(exact[(f, n)].coef(deg[(f, n)])[j]) for f, n, j in index]
    err = [math.sqrt(max(0.0, C[a][a])) for a in range(nout)]
    res = {"policy": policy, "sources": src, "available": avail, "terms": terms, "deg": deg, "index": index,
           "exact": exact, "value": val, "error": err, "cov": C, "cov_ledger": Cl, "cov_phi4tools": Cp,
           "led": led, "eps": eps, "refkeys": refkeys, "cols": cols, "float": f0}
    res["verify"] = verify(res, vals, avail) if check else {"lines": []}
    res["largeN"] = large_n_tests(res) if check else None
    return res


def block(res, t):
    """the indices of the coefficients of term t in the global vector"""
    return [a for a, (f, n, j) in enumerate(res["index"]) if (f, n) == t]


def at_N(res, t, N, C=None):
    """value and error of the coefficient of ubar^n at N from the polynomial and the covariance"""
    f, n = t
    d = Fr(N + 8) ** pw(f, n)
    v = float(res["exact"][t](Fr(N)) / d)
    idx = block(res, t)
    w = [float(Fr(N) ** j / d) for j in range(len(idx))]
    C = res["cov"] if C is None else C
    var = sum(w[i] * w[k] * C[a][b] for i, a in enumerate(idx) for k, b in enumerate(idx))
    return v, math.sqrt(max(0.0, var))


def verify(res, vals, avail):
    """the comparisons listed in the header, as a dict of numbers (and the lines of the report)"""
    src, led, terms, index = res["sources"], res["led"], res["terms"], res["index"]
    out, lines = {}, []
    # (a) values and errors at N = 0..4 against coefficients.py's procedure on the same inputs
    wv = wa = wi = 0.0
    ratios = {"total": [], "ledger": [], "phi4tools": []}
    for N in NS:
        refN = RL.references(N)
        inp = {k: (led[k]["atN"][N] + ("ledger",)) if s == "ledger" else (refN[k][0], refN[k][1], s) for k, s in src.items()}
        b0, e0 = CO.propagate(inp, N)
        # the same inputs through the same algebra in exact arithmetic: where the differences come from
        u = CO.rg({k: PolyQ([Fr(v[0])]) for k, v in inp.items()}, -2)
        _, el = CO.propagate({k: (v, e if s == "ledger" else 0.0, s) for k, (v, e, s) in inp.items()}, N)
        _, ep = CO.propagate({k: (v, e if s != "ledger" else 0.0, s) for k, (v, e, s) in inp.items()}, N)
        for t in terms:
            f, n = t
            v, e = at_N(res, t, N)
            ref = b0[f][n]
            x = as_poly(u[f][n])
            xq = float((x.c[0] if x.c else 0) * Fr(6, N + 8) ** pw(f, n))
            if ref:
                wv = max(wv, abs(v - ref) / abs(ref))
                wa = max(wa, abs(xq - ref) / abs(ref))
                wi = max(wi, abs(v - xq) / abs(ref))
            for kind, ce, C in (("total", e0, res["cov"]), ("ledger", el, res["cov_ledger"]), ("phi4tools", ep, res["cov_phi4tools"])):
                mine = at_N(res, t, N, C)[1]
                if ce[f][n] > 0:
                    ratios[kind].append((mine / ce[f][n], t, N))
    out["values_vs_coefficients_py"] = [wv, wa, wi]
    lines.append("values at N = 0..4 against coefficients.py (its rg on its at-N inputs): max relative difference %.1e; "
                 "the same inputs in exact arithmetic differ from coefficients.py by up to %.1e (its floating point) and "
                 "from these polynomials by up to %.1e (its at-N sums)" % (wv, wa, wi))
    for kind in ("total", "ledger", "phi4tools"):
        r = ratios[kind]
        if r:
            lo, hi = min(r), max(r)
            out["errors_ratio_" + kind] = [lo[0], hi[0]]
            lines.append("errors at N = 0..4, %-9s part: this / coefficients.py between %.6f (%s_%d, N=%d) and %.6f (%s_%d, N=%d)"
                         % (kind, lo[0], lo[1][0], lo[1][1], lo[2], hi[0], hi[1][0], hi[1][1], hi[2]))
    # (b) the exact algebra against the same algebra in floating point
    wf = max(abs(x - y) / abs(y) for x, y in zip(res["float"], res["value"]) if y)
    out["float_vs_exact"] = wf
    lines.append("the algebra in floating point against exact rational arithmetic: max relative difference %.1e" % wf)
    # (c) the complex-step Jacobian against central differences in exact arithmetic, step h = the input's error
    #     rounded to a power of 2 (which also measures the nonlinearity over one standard deviation), in units of
    #     the outputs' errors
    def run_exact(v):
        r = numerators(v, avail, PolyQ)
        return [x for t in terms for x in r[t].coef(res["deg"][t])]
    wj = 0.0
    for kind, name, sig, col in res["cols"]:
        h = Fr(2) ** int(round(math.log(sig, 2)))
        up, dn = dict(vals), dict(vals)
        if kind == "ledger":
            key, j = name
            up[key] = vals[key] + PolyQ([0] * j + [h])
            dn[key] = vals[key] - PolyQ([0] * j + [h])
        else:
            d = next(x[2] for x in res["eps"] if x[0] == name)
            for k in res["refkeys"]:
                up[k], dn[k] = vals[k] + d[k] * h, vals[k] - d[k] * h
        fu, fd = run_exact(up), run_exact(dn)
        for a in range(len(index)):
            if res["error"][a] > 0:
                wj = max(wj, abs(float((fu[a] - fd[a]) / (2 * h)) - col[a]) * sig / res["error"][a])
    out["jacobian_vs_differences"] = wj
    lines.append("complex-step Jacobian against exact central differences (step: the input's error): largest "
                 "difference %.1e of an output's error" % wj)
    # (d) the Phi4tools round trip, exactly: refs.py's sums give back refs.py's polynomials
    ref, _ = refs_inputs()
    ex = exact_inputs()
    rsrc = CO.choose({}, "refs")
    rav = CO.available(set(rsrc))
    back = numerators({k: ex[k] if s == "exact" else ref[k] for k, s in rsrc.items()}, rav, PolyQ)
    Nr = RN(NQ)
    ph = {"beta": RF.beta_coeffs(Nr), "eta": RF.eta_coeffs(Nr), "nu": RF.nu_coeffs(Nr)}
    same = total = 0
    for f in ph:
        for n in range(1, len(ph[f])):
            v = ph[f][n][0]
            p = v.times_n8(pw(f, n)) if isinstance(v, RN) else as_poly(v) * n8(pw(f, n))
            total += 1
            same += (f, n) in back and back[(f, n)].c == p.c
    out["phi4tools_round_trip"] = [same, total]
    lines.append("Phi4tools round trip in exact arithmetic (refs.py's sums -> rg -> numerators): %d of %d polynomials "
                 "of refs.py reproduced exactly" % (same, total))
    # (e) coefficients.py's interpolation (numerator) of the correctly rounded values at N = 0..deg+1 against these
    #     polynomials, and its sensitivity to an ulp of those values
    worst = {}
    for t in TABLE:
        if t not in res["exact"]:
            continue
        f, n = t
        idx = block(res, t)
        cval = {N: float(res["exact"][t](Fr(N)) / Fr(N + 8) ** pw(f, n)) for N in range(12)}   # correctly rounded
        p, _ = CO.numerator(f, n, lambda N: cval[N])
        q, _ = CO.numerator(f, n, lambda N: cval[N] * (1 + (-1) ** N * 2.0 ** -52))
        dev = [abs(p[j] - res["value"][a]) / res["error"][a] if res["error"][a] else
               abs(p[j] - res["value"][a]) / abs(res["value"][a]) for j, a in enumerate(idx) if j < len(p)]
        ulp = [abs(p[j] - q[j]) / abs(p[j]) for j in range(len(idx)) if p[j]]
        extra = p[len(idx):]
        worst[t] = (max(dev), max(ulp), extra)
    out["interpolation"] = {"%s_%d" % t: list(v[:2]) + [v[2]] for t, v in worst.items()}
    if worst:
        lines.append("coefficients.py's interpolation (numerator) against these polynomials, for the terms of the table:")
        for t, (d, u, extra) in worst.items():
            lines.append("  %-11s largest deviation %.1e (of the error; of the value where exact), an ulp of its values "
                         "moves a coefficient by %.1e relative%s" % ("%s_%d" % t, d, u, "; its N^%d term %+.1e (absent here)"
                                                                       % (len(block(res, t)), extra[0]) if extra else ""))
    out["lines"] = lines
    return out


# ------------------------------------------------------------------------------------------------- large N
# The limit N -> infinity and the 1/N terms of the RG functions (header, "Large N"), and the tests of check S5:
# float power series in x or ubar, LK terms.
LK = 13                                  # the large-N series in x and ubar, through order 12
_PI = math.pi


def _S(c):
    c = list(c)[:LK]
    return c + [0.0] * (LK - len(c))


def _add(*a):
    return [sum(t) for t in zip(*a)]


def _sc(a, s):
    return [s * x for x in a]


def _mul(a, b):
    r = [0.0] * LK
    for i, x in enumerate(a):
        if x:
            for j in range(LK - i):
                r[i + j] += x * b[j]
    return r


def _inv(a):
    r = [0.0] * LK
    r[0] = 1 / a[0]
    for n in range(1, LK):
        r[n] = -sum(a[j] * r[n - j] for j in range(1, n + 1)) / a[0]
    return r


def _der(a):
    return [(i + 1) * a[i + 1] for i in range(LK - 1)] + [0.0]


def _compose(a, b):
    """a(b(u)) for b[0] = 0"""
    r, p = [0.0] * LK, _S([1.0])
    for n in range(LK):
        r = _add(r, _sc(p, a[n]))
        p = _mul(p, b)
    return r


_X, _ONE = _S([0, 1.0]), _S([1.0])
_INV1PX = _inv(_add(_ONE, _X))           # 1/(1+x)


def _nodes(h=1 / 96.0, tmax=4.6):
    """double-exponential (exp-sinh) quadrature on (0, inf): q = exp(pi/2 sinh t)"""
    out = []
    n = int(tmax / h)
    for i in range(-n, n + 1):
        t = i * h
        u = _PI / 2 * math.sinh(t)
        if abs(u) < 700:
            q = math.exp(u)
            out.append((q, h * q * _PI / 2 * math.cosh(t)))
    return out


def _integrands(q):
    """N z, B and V of the large-N section of the header at this q, as power series in x"""
    G = 1 / (q * q + 1)
    Pi = math.atan(q / 2) / (4 * _PI * q) if q > 1e-6 else 1 / (8 * _PI) - q * q / (96 * _PI)
    Di = _inv(_add(_ONE, _sc(_X, 8 * _PI * Pi)))
    Di2 = _mul(Di, Di)
    q4 = q * q + 4
    x2 = _mul(_X, _X)
    A = 8 / q4 ** 2 - 3 / (q4 * (q * q + 1))
    lin = _add(_S([q4]), _sc(_X, 3))      # q^2 + 4 + 3x
    x2D = _mul(_mul(x2, _INV1PX), Di)
    B = _add(_sc(x2D, 8 * _PI * A), _sc(_mul(_X, Di), -32 * _PI * G * G),
             _sc(_mul(_mul(_mul(_X, _mul(lin, lin)), _INV1PX), Di2), -32 * _PI * G * G / q4 ** 2))
    V = _add(_sc(x2D, -8 * _PI * A), _sc(_mul(_X, Di), 16 * _PI * G * G),
             _sc(_mul(_mul(_mul(x2, lin), _INV1PX), Di2), -32 * _PI * G / q4 ** 2))
    z = _sc(_mul(_X, Di), -(16 * _PI / 6) * (2 * q * q - 6) * G ** 3)
    return z, B, V


def large_n():
    """{f: [the coefficient of ubar^k of its 1/N term, k = 0..LK-1]} for beta, eta, eta_phi2, nu (f = f_inf + f_1/N)"""
    z, Bi, Vi = [0.0] * LK, [0.0] * LK, [0.0] * LK
    for q, w in _nodes():
        a, b, c = _integrands(q)
        f = w * q * q / (2 * _PI * _PI)
        z, Bi, Vi = _add(z, _sc(a, f)), _add(Bi, _sc(b, f)), _add(Vi, _sc(c, f))
    xh = _mul(_X, _sc(_INV1PX, 0.5))
    Sb = _add(_sc(_mul(xh, z), -1), Bi)
    Sn = _add(_mul(xh, z), _sc(z, -1), Vi)
    f0 = _mul(_X, _INV1PX)               # ubar at N = infinity
    f1 = _mul(f0, _add(_S([8.0]), _sc(z, 2), Sb))
    U = _S([0, 1.0])
    x0 = _mul(U, _inv(_add(_ONE, _sc(U, -1))))  # x(ubar) = ubar/(1-ubar)
    f0p = _mul(_INV1PX, _INV1PX)
    f0pp = _sc(_mul(f0p, _INV1PX), -2)
    x1 = _sc(_mul(_compose(f1, x0), _inv(_compose(f0p, x0))), -1)          # x = x0 + x1/N
    beta = _add(_sc(_mul(x1, _compose(_add(f0p, _mul(_X, f0pp)), x0)), -1), _sc(_mul(x0, _compose(_der(f1), x0)), -1))
    eta = _sc(_mul(x0, _compose(_der(z), x0)), -1)
    eta2 = _add(_mul(_compose(_sc(f0p, -1), x0), x1), _compose(_sc([0.0] + _der(Sn)[:LK - 1], -1), x0))   # eta_2 = -x d ln Z_2/dx
    tmu = _add(_S([2.0]), _sc(U, -1))
    nu = _sc(_mul(eta2, _mul(_inv(tmu), _inv(tmu))), -1)                 # nu = 1/(2 - ubar + eta2_1/N)
    return {"beta": beta, "eta": eta, "eta_phi2": _add(eta2, eta), "nu": nu}


def predicted(LN, f, k):
    """(degree, {power of N: coefficient}) of p_f,k from the large-N expansion: its two leading coefficients, exact
    (fractions) where N = infinity fixes them, else (floats) from the 1/N term"""
    if f == "beta":
        return (0, {0: Fr(-1)}) if k == 1 else (1, {1: Fr(1), 0: Fr(8)}) if k == 2 else \
            (k - 2, {k - 2: LN[f][k]})
    if f == "eta":
        return k - 1, {k - 1: LN[f][k]}
    if f == "eta_phi2":
        return (1, {1: Fr(-1), 0: -8 + LN[f][1]}) if k == 1 else (k - 1, {k - 1: LN[f][k]})
    return k, {k: Fr(1, 2 ** (k + 1)), k - 1: 8 * k * 2.0 ** -(k + 1) + LN[f][k]}


def compare(p, deg, known):
    """(degree as predicted and exact coefficients equal, [(power, value, predicted)] of the 1/N ones)"""
    ok = len(p.c) - 1 == deg
    c = p.coef(deg)
    rows = []
    for j, v in sorted(known.items(), reverse=True):
        if isinstance(v, Fr):
            ok = ok and c[j] == v
        else:
            rows.append((j, float(c[j]), v))
    return ok, rows


def predicted_sums(LN):
    """{(kind, L): (N^L coefficient, N^(L-1) coefficient)} of the order sums from the large-N series: the polynomials
    p_X,k truncated to their two leading coefficients (which alone determine those of the sums), in u, through the
    inverse Callan-Symanzik relations (exact arithmetic on the doubles)"""
    Q, Kc = PolyQ, CO.K
    def poly(known):
        p = Q([])
        for j, v in known.items():
            p = p + Q([0] * j + [Fr(v)])
        return p
    bu, eu, nu = [0] * Kc, [0] * Kc, [0] * Kc
    for k in range(1, Kc):
        bu[k] = poly(predicted(LN, "beta", k)[1]) / 6 ** (k - 1)
        eu[k] = poly(predicted(LN, "eta", k)[1]) / 6 ** k if k >= 2 else Q([])
        nu[k] = poly(predicted(LN, "nu", k)[1]) / 6 ** k
    nu[0] = Fr(1, 2)
    tme = [-x for x in eu]
    tme[0] += 2
    itme = CO.inv(tme)
    h = CO.mul(eu, itme)
    num = [bu[i] - (2 * eu[i - 1] if i >= 1 else 0) for i in range(Kc)]
    num[1] += 1
    bt = CO.mul(num, itme)
    omc = CO.inv(CO.mul(nu, tme))
    c = [-x for x in omc]
    c[0] += 1
    out = {}
    for kind, L, v in [("b", L, (-1) ** (L + 1) * bt[L + 1]) for L in range(1, Kc - 1)] + \
                      [("h", L, (-1) ** L * h[L]) for L in range(2, Kc)] + [("c", L, (-1) ** (L + 1) * c[L]) for L in range(1, Kc)]:
        p = as_poly(v)
        out[(kind, L)] = (p.coef(L)[L], p.coef(L)[L - 1])
    return out


def large_n_tests(res, LN=None):
    """The comparisons of check S5 (and of the macros of --tex): (a) Phi4tools' published polynomials, (b) the exact
    low orders, (c) every order of the ledger (degree, N^(L-1) coefficient), (d) the polynomials of res"""
    LN = LN or large_n()
    out = {"one_over_N": {f: LN[f][:CO.K] for f in LN}}
    N = RN(NQ)
    ph = {"beta": RF.beta_coeffs(N), "eta": RF.eta_coeffs(N), "nu": RF.nu_coeffs(N)}
    rows, worst, ok = [], 0.0, True
    for f, lo in (("beta", 3), ("eta", 2), ("nu", 1)):
        for k in range(lo, len(ph[f])):
            v = ph[f][k][0]
            p = v.times_n8(pw(f, k)) if isinstance(v, RN) else as_poly(v) * n8(pw(f, k))
            d, known = predicted(LN, f, k)
            e_ok, cmp = compare(p, d, known)
            ok = ok and e_ok
            for j, got, want in cmp:
                worst = max(worst, abs(got - want) / abs(got))
                rows.append((f, k, len(p.c) - 1, {i: str(x) for i, x in known.items() if isinstance(x, Fr)}, j, got, want))
    out["phi4tools"] = {"rows": rows, "worst": worst, "exact": ok}
    pred = predicted_sums(LN)
    worst = 0.0
    for key, p in sorted(exact_inputs().items()):
        L = key[1]
        q = p.coef(L)
        a, b = pred[key]
        worst = max(worst, abs(float(q[L] - a)), abs(float(q[L - 1] - b)) / abs(float(q[L - 1])))
    out["exact_sums"] = worst
    rows, chi2, zmax, degok = [], 0.0, 0.0, True
    for key in sorted(res["led"], key=lambda k: ("bhc".index(k[0]), k[1])):
        d, L = res["led"][key], key[1]
        top, err = float(d["poly"].coef(L - 1)[L - 1]), math.sqrt(d["cov"][L - 1][L - 1])
        want = float(pred[key][1])
        z = (top - want) / err
        chi2, zmax, degok = chi2 + z * z, max(zmax, abs(z)), degok and d["deg"] <= L - 1
        rows.append((d["order"], d["classes"], d["deg"], L, top, err, want, z))
    out["ledger"] = {"rows": rows, "chi2": chi2, "dof": len(rows), "p": RL.chi2_sf(chi2, len(rows)) if rows else None,
                     "zmax": zmax, "degrees": degok}
    rows, ok = [], True
    for t in res["terms"]:
        f, k = t
        idx = block(res, t)
        d, known = predicted(LN, f, k)
        this, cmp = compare(res["exact"][t], d, known)
        this = this and all(res["error"][idx[j]] == 0.0 for j, x in known.items() if isinstance(x, Fr))
        ok = ok and this
        for j, got, want in cmp:
            rows.append((name(t), res["deg"][t], {i: str(x) for i, x in known.items() if isinstance(x, Fr)}, j, got,
                         res["error"][idx[j]], want, this))
    out["polynomials"] = {"rows": rows, "exact": ok}
    return out


def large_n_macros(T):
    """the numbers of the text on the large-N test (methods.tex, item "Large N"), as (macro name, LaTeX)"""
    import tables as TB
    c = T["ledger"]
    if not c["rows"]:
        return []
    Ls = [r[3] for r in c["rows"]]
    return [("largeNorders", "%d" % c["dof"]), ("largeNLmin", "%d" % min(Ls)), ("largeNLmax", "%d" % max(Ls)),
            ("largeNchisq", "%.1f" % c["chi2"]), ("largeNpvalue", "%.2f" % c["p"]), ("largeNzmax", "%.1f" % c["zmax"]),
            ("largeNphi", "$%s$" % TB.sci(T["phi4tools"]["worst"], 1))]


# ------------------------------------------------------------------------------------------------- outputs
def name(t):
    return "%s_%d" % t


def exact_cell(q):
    """an exact coefficient: 2^m, p/q or a decimal"""
    if q == 0 or abs(q) == 1:
        return "$%d$" % q if q < 0 else "%d" % q
    a = abs(q)
    if a.numerator == 1 and a.denominator & (a.denominator - 1) == 0 or a.denominator == 1 and a.numerator & (a.numerator - 1) == 0:
        m = a.denominator.bit_length() - 1 if a.numerator == 1 else -(a.numerator.bit_length() - 1)
        return "$%s2^{%d}$" % ("-" if q < 0 else "", -m)
    if a.denominator < 10 ** 6:
        return "$%s%d/%d$" % ("-" if q < 0 else "", a.numerator, a.denominator)
    return "$%.10g$" % float(q)


def cell(v, e):
    """value(error) rounded as tables.fmt_err; as a mantissa times a power of 10 when |v| >= 10^5, |v| < 10^-3 or the
    error is 9.5 or more (fmt_err would round the value to tens or more)"""
    import tables as TB
    x = int(math.floor(math.log10(abs(v)))) if v else 0
    if -3 <= x < 5 and e < 9.5:
        return TB.math_if(TB.fmt_err(v, e))
    m = 10.0 ** x
    return TB.math_if("%s\\times10^{%d}" % (TB.fmt_err(v / m, e / m), x))


def tex_table(res, terms):
    idx = {t: block(res, t) for t in terms}
    rows = max(len(i) for i in idx.values())
    L = ["\\begin{ruledtabular}", "\\begin{tabular}{r%s}" % ("l" * len(terms)),
         "$j$ & " + " & ".join("$p_{%s,%d}$" % (TEXF[f], n) for f, n in terms) + " \\\\", "\\hline"]
    for j in range(rows):
        cells = []
        for t in terms:
            if j >= len(idx[t]):
                cells.append("")
                continue
            a = idx[t][j]
            cells.append(exact_cell(res["exact"][t].coef(res["deg"][t])[j]) if res["error"][a] == 0 else
                         cell(res["value"][a], res["error"][a]))
        L.append("%d & %s \\\\" % (j, " & ".join(cells)))
    return L + ["\\end{tabular}", "\\end{ruledtabular}"]


def write_tex(res, path):
    A = [t for t in TABLE if t[0] in ("beta", "eta")]
    B = [t for t in TABLE if t[0] in ("eta_phi2", "nu")]
    L = ["% generated/newpoly.tex -- the coefficients a_j of the numerator polynomials p_X,k(N) = sum_j a_j N^j",
         "% (Eq. series-form) of the new and improved terms, with their errors: \\newpolybetaeta (beta_9-11, eta_8-10)",
         "% and \\newpolyphinu (eta_phi2,8-10, nu_8-10), each the body of a table.  Exact coefficients (no Monte Carlo",
         "% input: the top coefficients of p_nu,k, 2^-(k+1)) are printed as such; powers beyond a polynomial's degree are",
         "% empty.  The coefficients are correlated: their covariance is in results/coefficients/<date>/polynomials.json.",
         "% WRITTEN BY series/polynomials.py --tex from the ledger and integrator/tools/refs.py: do not edit by hand.",
         "\\newcommand*{\\newpolybetaeta}{%"] + tex_table(res, A) + ["}", "\\newcommand*{\\newpolyphinu}{%"] + \
        tex_table(res, B) + ["}", "% The large-N test of the sums (methods.tex, item \"Large N\"; check S5): the orders of the ledger compared,",
                             "% their range of L, chi^2 and its probability, the largest |pull|, and the largest relative difference of the",
                             "% 1/N expansion from the leading coefficients of Phi4tools' polynomials."] + \
        ["\\newcommand*{\\%s}{%s}" % kv for kv in large_n_macros(res["largeN"])]
    return RL.write_if_changed(path, "\n".join(L) + "\n")


def json_obj(res):
    led, src = res["led"], res["sources"]
    polys = []
    for t in res["terms"]:
        idx = block(res, t)
        f, n = t
        cl, cp = res["cov_ledger"], res["cov_phi4tools"]
        polys.append({"name": name(t), "function": f, "k": n, "denominator": "(N+8)^%d" % pw(f, n),
                      "degree": res["deg"][t], "first_index": idx[0], "in_table": t in TABLE,
                      "coefficients": [res["value"][a] for a in idx], "errors": [res["error"][a] for a in idx],
                      "errors_ledger": [math.sqrt(max(0.0, cl[a][a])) for a in idx],
                      "errors_phi4tools": [math.sqrt(max(0.0, cp[a][a])) for a in idx],
                      "exact": [str(x) for x in res["exact"][t].coef(res["deg"][t])] if all(res["error"][a] == 0 for a in idx) else None})
    sums = {}
    for key in sorted(src, key=lambda k: ("bhc".index(k[0]), k[1])):
        s, o = src[key], RL.order_name(*key)
        d = {"source": s}
        if s == "ledger":
            e = led[key]
            d.update({"poly": [float(x) for x in e["poly"].coef(e["deg"])], "cov": e["cov"], "classes": e["classes"],
                      "weight_polynomials": e["groups"]})
        elif s == "exact":
            d["poly"] = [str(x) for x in exact_inputs()[key].c]
        sums[o] = d
    return {"format": "rebar-polynomials-1",
            "description": "numerator polynomials p_X,k(N) = sum_j a_j N^j of beta = -ubar + ubar^2 + sum_k p_beta,k ubar^k "
                           "/ (N+8)^(k-1), eta = sum_k p_eta,k ubar^k / (N+8)^k, eta_phi2 = sum_k p_eta_phi2,k ubar^k / "
                           "(N+8)^k, nu = 1/2 + sum_k p_nu,k ubar^k / (N+8)^k (ubar = (N+8) g / (48 pi)); coefficients, "
                           "errors and the covariance of all coefficients (linear propagation from the per-class "
                           "integrals of the ledger and the quoted errors of Phi4tools); written by series/polynomials.py",
            "policy": res["policy"], "ledger_date": latest_run(),
            "sums": sums,
            "phi4tools_inputs": [{"name": lab, "sigma_numerator": [float(x) for x in sig.c]} for lab, sig, _ in res["eps"]]
            if res["refkeys"] else [],
            "polynomials": polys,
            "index": ["%s:%d" % (name((f, n)), j) for f, n, j in res["index"]],
            "verification": {k: v for k, v in res["verify"].items() if k != "lines"},
            "large_n": large_n_json(res["largeN"]) if res.get("largeN") else None,
            "covariance": [[x for x in row] for row in res["cov"]]}


def large_n_json(T):
    c = T["ledger"]
    return {"one_over_N": T["one_over_N"],
            "phi4tools_largest_relative_difference": T["phi4tools"]["worst"],
            "sums": {o: {"degree": deg, "top": top, "error": err, "one_over_N": want, "pull": z}
                     for o, n, deg, L, top, err, want, z in c["rows"]},
            "chi2": c["chi2"], "orders": c["dof"], "p": c["p"], "largest_pull": c["zmax"],
            "degrees_as_predicted": c["degrees"] and T["polynomials"]["exact"]}


def latest_run():
    orders = RL.load_orders(RL.LEDGER)
    utc = [c["utc"] for e in orders["orders"].values() for c in e.get("contributions", []) if c.get("utc")]
    return max(utc) if utc else None


def write_json(res, path):
    return RL.write_if_changed(path, RL.dump_json(json_obj(res), rows_key="covariance"))


def write_tsv(res, path):
    lab = ["%s:%d" % (name((f, n)), j) for f, n, j in res["index"]]
    L = ["# polynomials.tsv -- written by series/polynomials.py (policy %s): one row per coefficient a_j of p_X,k(N) = sum_j a_j N^j"
         % res["policy"],
         "# (Eq. series-form); columns: label, function, k, power of N+8 under it, j, value, error, then the covariance with"
         " every coefficient (the column headed by its label)",
         "\t".join(["label", "function", "k", "denominator_power", "j", "value", "error"] + lab)]
    for a, (f, n, j) in enumerate(res["index"]):
        L.append("\t".join([lab[a], f, str(n), str(pw(f, n)), str(j), repr(res["value"][a]), repr(res["error"][a])] +
                           [repr(x) for x in res["cov"][a]]))
    return RL.write_if_changed(path, "\n".join(L) + "\n")


def report(res):
    led, src = res["led"], res["sources"]
    out = ["Sources (policy %s): %s" % (res["policy"], ", ".join("%s %s" % (RL.order_name(*k), s) for k, s in
                                                                sorted(src.items(), key=lambda kv: ("bhc".index(kv[0][0]), kv[0][1]))))]
    out.append("\nLedger orders: classes, distinct weight polynomials (exact fractions recovered for all but the number "
               "shown), degree in N of the sum")
    for key in sorted(led, key=lambda k: ("bhc".index(k[0]), k[1])):
        d = led[key]
        out.append("  %-4s %8d classes %6d weights (%d inexact) degree %d%s" % (d["order"], d["classes"], d["groups"], d["inexact"],
                                                                         d["deg"], "" if src.get(key) == "ledger" else "  [not used: %s]" % src.get(key)))
    out.append("\nVerification:")
    out += ["  " + x for x in res["verify"]["lines"]]
    out.append("\nPolynomials p_X,k(N) = sum_j a_j N^j: a_j(error) [error from the ledger / from Phi4tools]")
    for t in res["terms"]:
        idx = block(res, t)
        tag = "  (table)" if t in TABLE else ""
        out.append("  %s / %s, degree %d%s" % (name(t), "(N+8)^%d" % pw(*t), res["deg"][t], tag))
        for j, a in enumerate(idx):
            v, e = res["value"][a], res["error"][a]
            el, ep = math.sqrt(max(0.0, res["cov_ledger"][a][a])), math.sqrt(max(0.0, res["cov_phi4tools"][a][a]))
            q = res["exact"][t].coef(res["deg"][t])[j]
            out.append("    j=%-2d %s" % (j, ("%+.16e (%s)" % (v, "exactly %s" % q if q.denominator < 10 ** 6 else "no error"))
                                         if e == 0 else "%+.12e +- %.3e  [%.2e / %.2e]" % (v, e, el, ep)))
    T = res.get("largeN")
    if T:
        out.append("\nLarge N (check S5): the 1/N expansion against Phi4tools' leading coefficients (largest relative difference "
                   "%.1e, degrees and exact coefficients %s) and the exact low orders (%.1e); the ledger's orders, the degree of "
                   "their weight polynomials and their N^(L-1) coefficient:" % (T["phi4tools"]["worst"], "as predicted" if
                                                                               T["phi4tools"]["exact"] else "NOT AS PREDICTED", T["exact_sums"]))
        for o, n, deg, L, top, err, want, z in T["ledger"]["rows"]:
            out.append("  %-4s degree %d  N^%d %+.10e +- %.2e  1/N expansion %+.10e  pull %+.2f" % (o, deg, L - 1, top, err, want, z))
        c = T["ledger"]
        out.append("  chi^2 = %.2f for %d orders (p = %.3f), largest |pull| %.2f; every class of degree <= L-1: %s; the polynomials' "
                   "degrees and exact leading coefficients: %s" % (c["chi2"], c["dof"], c["p"], c["zmax"], c["degrees"],
                                                                   T["polynomials"]["exact"]))
        out.append("  1/N coefficients of ubar^k, k = 0..%d:" % (CO.K - 1))
        for f in FUNCS:
            out.append("    %-8s %s" % (f, " ".join("%+.12e" % x for x in T["one_over_N"][f])))
    out.append("\nAt N = 1 (from the polynomials and the covariance): value, error, and the share of the variance from Phi4tools")
    for t in TABLE:
        if t in res["exact"]:
            v, e = at_N(res, t, 1)
            ep = at_N(res, t, 1, res["cov_phi4tools"])[1]
            out.append("  %-11s %+.10e +- %.3e   Phi4tools %.1e of the variance" % (name(t), v, e, (ep / e) ** 2 if e else 0))
    return "\n".join(out)


def main(argv):
    ledger, policy, tex, js, tsv, quiet = RL.LEDGER, "best", None, None, None, False
    it = iter(argv)
    for a in it:
        if a == "--ledger":
            ledger = next(it)
        elif a == "--source":
            policy = next(it)
        elif a == "--tex":
            tex = next(it)
        elif a == "--json":
            js = next(it)
        elif a == "--tsv":
            tsv = next(it)
        elif a == "--quiet":
            quiet = True
        else:
            print(__doc__)
            return 2
    if policy not in ("best", "ledger", "refs"):
        print(__doc__)
        return 2
    res = compute(ledger, policy)
    for path, w in ((tex, write_tex), (js, write_json), (tsv, write_tsv)):
        if path:
            print("%s %s" % ("wrote" if w(res, path) else "unchanged", path), file=sys.stderr)
    if not quiet:
        print(report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
