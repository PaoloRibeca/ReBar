#!/usr/bin/env python3
"""Reference values of the CS-route per-order sums B_L, H_L, C_L (raw sums, positive-integrand convention)
from the Phi4tools O(N) series (Sberveglieri-Spada, arXiv:2311.12576, appendix D).

Conventions (see README.md):  u = g/(8 pi), ubar (= Phi4tools u~) = (N+8) u / 6.
  eta = 2h/(1+h)  <=>  h = eta/(2-eta)
  beta_u = -u + 2 eta u + (2-eta) btilde  <=>  btilde = (beta_u + u - 2 eta u)/(2-eta)
  1/nu = (2-eta)(1-c)  <=>  c = 1 - 1/(nu (2-eta))
  btilde = sum_L (-1)^(L+1) B_L u^(L+1),  h = sum_L (-1)^L H_L u^L,  c = sum_L (-1)^(L+1) C_L u^L.
Uncertainties: the quoted Phi4tools errors are propagated linearly (first order) by finite differences.
Usage: refs.py [N ...]  -> JSON on stdout."""
import json, sys
from fractions import Fraction as Fr

K = 9  # series length (powers 0..8)

def beta_coeffs(N, d=None):
    # beta_ubar(ubar) coefficients c[1..8] of ubar^n; entries (value, sigma)
    n8 = N + 8
    c = [(0, 0)] * K
    c[1] = (-1, 0); c[2] = (1, 0)
    c[3] = (-4 * (41 * N + 190) / 27 / n8 ** 2, 0)
    c[4] = ((1.348942760866478 * N**2 + 54.940377049302200 * N + 199.640417221105907) / n8 ** 3, 0)
    c[5] = (-(-0.155645907585201 * N**3 + 35.82020347182 * N**2 + 602.5212285602 * N + 1832.2067281779) / n8 ** 4,
            (7e-11 * N**2 + 6e-10 * N + 14e-10) / n8 ** 4)
    c[6] = ((0.051236212811530 * N**4 + 3.237874 * N**3 + 668.55456 * N**2 + 7819.5673 * N + 20770.183) / n8 ** 5,
            (11e-6 * N**3 + 24e-5 * N**2 + 20e-4 * N + 5e-3) / n8 ** 5)
    c[7] = (-(-0.023424226049759 * N**5 - 1.07182 * N**4 + 265.8411 * N**3 + 12669.295 * N**2 + 114181.79 * N + 271300.61) / n8 ** 6,
            (8e-5 * N**4 + 20e-4 * N**3 + 24e-3 * N**2 + 0.13 * N + 0.28) / n8 ** 6)
    c[8] = ((0.012640642324067 * N**6 + 0.5433 * N**5 - 14.386 * N**4 + 8828.74 * N**3 + 246972.5 * N**2 + 1840997 * N + 3981620) / n8 ** 7,
            (5e-4 * N**5 + 16e-3 * N**4 + 0.25 * N**3 + 2.0 * N**2 + 8 * N + 14) / n8 ** 7)
    return c

def eta_coeffs(N):
    n8 = N + 8
    e = [(0, 0)] * K
    e[2] = (8 * (N + 2) / 27 / n8 ** 2, 0)
    e[3] = (0.0246840009259343 * (N**2 + 10 * N + 16) / n8 ** 3, 0)
    e[4] = ((-0.004298563333341 * N**3 + 0.667985910868 * N**2 + 4.60922100685 * N + 6.51210986356) / n8 ** 4,
            (20e-12 * N**2 + 3e-11 * N + 18e-11) / n8 ** 4)
    e[5] = (-(0.006550923035200 * N**4 - 0.13245107140 * N**3 + 1.891116 * N**2 + 15.18794 * N + 21.64700) / n8 ** 5,
            (8e-11 * N**3 + 10e-6 * N**2 + 6e-5 * N + 9e-5) / n8 ** 5)
    e[6] = ((-0.005548920737435 * N**5 - 0.02039935040 * N**4 + 3.05407 * N**3 + 64.0777 * N**2 + 300.7218 * N + 369.714) / n8 ** 6,
            (31e-11 * N**4 + 7e-5 * N**3 + 8e-4 * N**2 + 34e-4 * N + 4e-3) / n8 ** 6)
    e[7] = (-(0.004390810855773 * N**6 + 0.0612032025 * N**5 - 1.2705 * N**4 + 35.311 * N**3 + 751.79 * N**2 + 3345.53 * N + 3988.40) / n8 ** 7,
            (13e-10 * N**5 + 4e-4 * N**4 + 7e-3 * N**3 + 5e-2 * N**2 + 0.18 * N + 0.21) / n8 ** 7)
    e[8] = ((-0.003473417276666 * N**7 - 0.070431737 * N**6 + 0.151 * N**5 + 11.6 * N**4 + 1111 * N**3 + 13674 * N**2 + 52140 * N + 58297) / n8 ** 8,
            (5e-9 * N**6 + 3e-3 * N**5 + 0.1 * N**4 + 1.5 * N**3 + 9 * N**2 + 27 * N + 30) / n8 ** 8)
    return e

def nu_coeffs(N):
    n8 = N + 8
    v = [(0, 0)] * K
    v[0] = (0.5, 0)
    v[1] = ((N + 2) / 4 / n8, 0)
    v[2] = ((27 * N**2 + 16 * N - 76) / 216 / n8 ** 2, 0)
    v[3] = ((N**3 / 16 + 0.357987483753229 * N**2 + 2.20259658610878 * N + 3.473243237204659) / n8 ** 3, 0)
    v[4] = (-(-N**4 / 32 - 0.399474877149036 * N**3 + 0.609039560891 * N**2 + 14.977065564467 * N + 24.82217386818) / n8 ** 4,
            (5e-12 * N**2 + 32e-12 * N + 4e-11) / n8 ** 4)
    v[5] = ((N**5 / 64 + 0.333425900294786 * N**4 + 0.7393030 * N**3 + 27.122009 * N**2 + 184.70641 * N + 262.00439) / n8 ** 5,
            (9e-7 * N**3 + 18e-6 * N**2 + 11e-5 * N + 15e-5) / n8 ** 5)
    v[6] = (-(-N**6 / 128 - 0.245952216148592 * N**5 - 1.875537 * N**4 + 0.87861 * N**3 + 417.7907 * N**2 + 2385.941 * N + 3130.387) / n8 ** 6,
            (5e-6 * N**4 + 19e-5 * N**3 + 19e-4 * N**2 + 7e-3 * N + 8e-3) / n8 ** 6)
    v[7] = ((N**7 / 256 + 0.170195139781328 * N**6 + 2.298000 * N**5 + 0.1341 * N**4 + 328.511 * N**3 + 7521.91 * N**2 + 35180.6 * N + 42962.6) / n8 ** 7,
            (25e-6 * N**5 + 20e-4 * N**4 + 29e-3 * N**3 + 0.19 * N**2 + 0.6 * N + 0.6) / n8 ** 7)
    v[8] = (-(-N**8 / 512 - 0.113811144429731 * N**7 - 2.23349 * N**6 - 9.232 * N**5 - 116.02 * N**4 + 9073.7 * N**3 + 139051 * N**2 + 567393 * N + 652860) / n8 ** 8,
            (12e-5 * N**6 + 14e-3 * N**5 + 0.24 * N**4 + 2.2 * N**3 + 11 * N**2 + 30 * N + 29) / n8 ** 8)
    return v

def mul(a, b):
    r = [0.0] * K
    for i in range(K):
        for j in range(K - i): r[i + j] += a[i] * b[j]
    return r

def inv(a):
    r = [0.0] * K; r[0] = 1 / a[0]
    for n in range(1, K): r[n] = -sum(a[j] * r[n - j] for j in range(1, n + 1)) / a[0]
    return r

def derive(N, cb, ce, cv):
    k = (N + 8) / 6.0                       # ubar = k u
    beta_u = [0.0] * K; eta = [0.0] * K; nu = [0.0] * K
    for i in range(K):
        beta_u[i] = cb[i] * k ** (i - 1) if i >= 1 else 0.0   # beta_u(u) = beta_ubar(k u)/k
        eta[i] = ce[i] * k ** i
        nu[i] = cv[i] * k ** i
    two_m_eta = [-x for x in eta]; two_m_eta[0] += 2
    u = [0.0] * K; u[1] = 1
    num = [beta_u[i] + u[i] - 2 * mul(eta, u)[i] for i in range(K)]
    bt = mul(num, inv(two_m_eta))
    h = mul(eta, inv(two_m_eta))
    one_m_c = inv(mul(nu, two_m_eta))
    c = [-x for x in one_m_c]; c[0] += 1
    B = {L: (-1) ** (L + 1) * bt[L + 1] for L in range(1, K - 1)}
    H = {L: (-1) ** L * h[L] for L in range(2, K)}
    C = {L: (-1) ** (L + 1) * c[L] for L in range(1, K)}
    return B, H, C

def refs(N):
    cb = beta_coeffs(N); ce = eta_coeffs(N); cv = nu_coeffs(N)
    v = lambda c: [x[0] for x in c]
    B, H, C = derive(N, v(cb), v(ce), v(cv))
    # first-order error propagation: perturb each uncertain input coefficient by its sigma
    errs = {k: {L: 0.0 for L in d} for k, d in (("B", B), ("H", H), ("C", C))}
    for which in range(3):
        base = [list(v(cb)), list(v(ce)), list(v(cv))]
        src = (cb, ce, cv)[which]
        for i in range(K):
            s = src[i][1]
            if s == 0: continue
            pert = [list(x) for x in base]; pert[which][i] += s
            B2, H2, C2 = derive(N, *pert)
            for key, d0, d1 in (("B", B, B2), ("H", H, H2), ("C", C, C2)):
                for L in d0: errs[key][L] += abs(d1[L] - d0[L])
    out = {}
    for key, d in (("B", B), ("H", H), ("C", C)):
        out[key] = {str(L): [d[L], errs[key][L]] for L in d}
    return out

if __name__ == "__main__":
    Ns = [float(x) for x in sys.argv[1:]] or [1.0]
    res = {}
    for N in Ns:
        r = refs(N)
        # exact low orders as a self-check
        exact = {"B1": (N + 8) / 12, "B2": (5 * N + 22) / 54, "H2": (N + 2) / 243, "C1": (N + 2) / 12, "C2": (N + 2) / 36}
        for k, x in exact.items():
            got = r[k[0]][k[1:]][0]
            assert abs(got - x) < 1e-12 * max(1, abs(x)), (N, k, got, x)
        res[str(int(N)) if N == int(N) else str(N)] = r
    json.dump({"source": "Phi4tools arXiv:2311.12576 appendix D, converted via the CS relations (tools/refs.py)",
               "convention": "B_L, H_L, C_L raw per-order sums; btilde=sum(-1)^(L+1)B_L u^(L+1), h=sum(-1)^L H_L u^L, c=sum(-1)^(L+1)C_L u^L; entries [value, propagated_sigma]",
               "N": res}, sys.stdout, indent=1)
