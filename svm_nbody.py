#!/usr/bin/env python3
r"""Correlated-Gaussian variational solve of an N-quark colour-singlet baryon, any N and Nc.

Generalizes the N in (2, 3) solver on Ben's branch (``svm_baryon.py``, whose matrix elements and
basis recipe this reuses) to arbitrary N, which is what a general-Nc baryon needs: a singlet of
Nc quarks has N_coord = Nc constituents and Nc(Nc-1)/2 pairs.

    H = sum_k p_k^2 / 2m + sum_{k<l} V(r_kl),    V(r) = -c_pair * VB / r
    VB    = alpha * CF / (Nc - 1),    CF = (Nc^2 - 1) / (2 Nc)
    c_pair = Nc - 1  for the N = 2 qqbar singlet  (so the pair coupling is alpha CF)
           = 1       for a pair inside the totally antisymmetric N = Nc singlet

so every pair inside the baryon sits at ``-alpha CF/(Nc-1) / r``, exactly the ``'OA'`` channel
the GFMC generator uses.  The reported radius is the one-body charge radius about the centre of
mass, matching ``gfmc_me_gevp.py`` and ``svm_baryon.py``'s ``<r^2>_ch``:

    <r^2>_ch = (1/N) sum_k <(r_k - X)^2>,     F(q)/F(0) = (1/N) sum_k <e^{i q.(r_k - X)}>.

**Pure Coulomb, so everything scales exactly**: E0 ~ alpha^2, <r^2> ~ 1/alpha^2 and F(q) is a
function of q/alpha.  Solve once at alpha = 1 and rescale to any coupling -- which is why the
choice of Nc trajectory (fixed alpha_s vs fixed 't Hooft coupling) does not need making here.

Coordinates are built numerically rather than tabulated, which is the whole generalization.  With
the standard Jacobi set x_k = (1/k) sum_{i<=k} r_i - r_{k+1} (k = 1..N-1) and X the centre of
mass, let ``M`` be the N x N matrix whose row 0 is X and whose rows 1.. are the x_k.  Then

    Lam    = M[1:] M[1:]^T / m          kinetic matrix, diagonal diag(1/mu_k), mu_k = k m/(k+1)
    Wk[i]  = inv(M)[i, 1:]              r_i - X = Wk[i] . x
    W[ij]  = Wk[i] - Wk[j]              r_ij    = W[ij] . x
    Rmat   = (1/N) sum_i Wk[i] Wk[i]^T  so <r^2>_ch = 3 tr(Rmat C^-1) per Gaussian pair

and the N = 2, 3 cases reproduce the hand-written blocks they replace (asserted in --self_test).

Spatial symmetrization is optional and only accelerates convergence: the Hamiltonian is
permutation symmetric, so its ground state is totally symmetric whatever basis is used -- and
the GFMC spatial trial is symmetric too, with the antisymmetry carried by colour.

Usage
-----
    svm_nbody.py --self_test
    svm_nbody.py --Nc 3 --N 2 --alpha 1                 # meson:  <r^2>_ch = 27/16 = 1.6875
    svm_nbody.py --Nc 3 --N 3 --alpha 1 --nbasis 600     # baryon: <r^2>_ch = 3.826
    svm_nbody.py --Nc 4 --N 4 --alpha 1 --nbasis 600 --qs 0.4,0.8,1.2
"""
import argparse
import itertools
import sys

import numpy as np
from scipy.linalg import eigh

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--self_test', action='store_true',
               help='check the generated coordinates against the hand-written N=2,3 blocks and '
                    'the closed-form Coulomb meson; no other arguments needed')
p.add_argument('--Nc', type=int, default=3)
p.add_argument('--N', type=int, default=None,
               help='number of constituents; defaults to Nc (the singlet baryon). N=2 is the '
                    'qqbar meson.')
p.add_argument('--alpha', type=float, default=1.0)
p.add_argument('--nbasis', type=int, default=400)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--bmin', type=float, default=0.05, help='in units of the basis scale')
p.add_argument('--bmax', type=float, default=6.0)
p.add_argument('--ascale', type=float, default=1.0,
               help='basis scale relative to the Coulomb pair Bohr radius')
p.add_argument('--scut', type=float, default=1e-10,
               help='drop overlap eigen-directions below scut x max (ghost-state guard)')
p.add_argument('--symmetrize', action='store_true',
               help='add every particle permutation of each width set. Multiplies the basis by '
                    'N!, so it is affordable at N=3 (x6) and marginal at N=4 (x24).')
p.add_argument('--qs', default='', help='comma list of |q| for F(q)/F(0)')
p.add_argument('--out', default='', help='optional csv for F(q)/F(0)')
args = p.parse_args()

m = 1.0
N = args.N if args.N is not None else args.Nc
assert N >= 2, 'N must be at least 2'


# ----------------------------------------------------------------------------------------------
# coordinates: everything below is generated from the Jacobi transformation
# ----------------------------------------------------------------------------------------------
def coordinates(N, m=1.0):
    """Lam, W (per pair), Wk (per particle), Rmat, and the pair index list."""
    M = np.zeros((N, N))
    M[0] = 1.0 / N                                  # centre of mass
    for k in range(1, N):                           # x_k = mean(r_0..r_{k-1}) - r_k
        M[k, :k] = 1.0 / k
        M[k, k] = -1.0
    Minv = np.linalg.inv(M)
    assert np.allclose(Minv[:, 0], 1.0), 'CM column of M^-1 must be all ones'
    Lam = M[1:] @ M[1:].T / m
    assert np.allclose(Lam, np.diag(np.diag(Lam))), 'Jacobi kinetic matrix must be diagonal'
    Wk = [Minv[i, 1:].copy() for i in range(N)]
    idx = [(i, j) for i in range(N) for j in range(i + 1, N)]
    W = [Wk[i] - Wk[j] for i, j in idx]
    Rmat = sum(np.outer(w, w) for w in Wk) / N
    return Lam, W, Wk, Rmat, idx


Lam, W, Wk, Rmat, PAIRS = coordinates(N, m)
nj = N - 1


def self_test():
    ok = True
    # N = 2: x = r1 - r2, mu = m/2, r_k - X = +- x/2, <r^2>_ch = |x|^2/4
    L, Wp, Wq, R, _ = coordinates(2)
    ok &= np.allclose(L, [[1 / (m / 2)]]) and np.allclose(Wp, [[1.0]])
    ok &= np.allclose(sorted(np.ravel(Wq)), [-0.5, 0.5]) and np.allclose(R, [[0.25]])
    print(f'  N=2 coordinates match the hand-written block: {ok}')
    # N = 3: mu = (m/2, 2m/3), Rmat = diag(1/2, 2/3)/3
    L, Wp, Wq, R, _ = coordinates(3)
    o3 = (np.allclose(np.diag(L), [1 / (m / 2), 1 / (2 * m / 3)])
          and np.allclose(R, np.diag([0.5, 2.0 / 3.0]) / 3.0))
    # the pair vectors are (1,0), (1/2,1), (-1/2,1) up to overall sign
    o3 &= sorted(tuple(np.round(np.abs(w), 10)) for w in Wp) == \
        sorted([(1.0, 0.0), (0.5, 1.0), (0.5, 1.0)])
    print(f'  N=3 coordinates match the hand-written block: {o3}')
    ok &= o3
    # mu_k = k m/(k+1) for every N
    for n in (2, 3, 4, 5, 6):
        L, *_ = coordinates(n)
        mu = 1.0 / np.diag(L)
        om = np.allclose(mu, [k * m / (k + 1) for k in range(1, n)])
        print(f'  N={n} reduced masses mu_k = k m/(k+1): {om}')
        ok &= om
    return ok


if args.self_test:
    print('coordinate generation:')
    good = self_test()
    print('\nclosed-form Coulomb meson (alpha = 1, Nc = 3): <r^2>_ch = 27/16, E0 = -4/9')
    print('  run:  svm_nbody.py --Nc 3 --N 2 --alpha 1   and compare')
    sys.exit(0 if good else 1)


# ----------------------------------------------------------------------------------------------
# potential strength
# ----------------------------------------------------------------------------------------------
CF = (args.Nc ** 2 - 1) / (2 * args.Nc)
VB = args.alpha * CF / (args.Nc - 1)
cpair = (args.Nc - 1) if N == 2 else 1.0
kappa = cpair * VB                       # pair coupling, V_pair = -kappa/r
aC = 1.0 / (0.5 * m * kappa)             # Coulomb pair Bohr radius
L0 = aC * args.ascale


def G(beta):
    """Pair-potential kernel: <exp(-r^2/(2 beta)) | 1/r> / overlap, exact for Coulomb."""
    return VB * np.sqrt(2.0 / (np.pi * beta))


# ----------------------------------------------------------------------------------------------
# basis and matrix elements (Suzuki-Varga, d = 3)
# ----------------------------------------------------------------------------------------------
rng = np.random.default_rng(args.seed)
npair = len(W)
b = L0 * np.exp(rng.uniform(np.log(args.bmin), np.log(args.bmax), size=(args.nbasis, npair)))
if args.symmetrize:
    # a particle permutation acts on the pair widths by permuting which pair is which
    perms = []
    for pm in itertools.permutations(range(N)):
        order = [PAIRS.index(tuple(sorted((pm[i], pm[j])))) for i, j in PAIRS]
        perms.append(order)
    b = np.concatenate([b[:, o] for o in perms])

Wm = np.array(W)
A = np.einsum('np,pi,pj->nij', 1.0 / b ** 2, Wm, Wm)

# Built in row blocks rather than as one (n, n, nj, nj) array.  C and its inverse are the memory
# bottleneck -- 8*nj^2 bytes per basis pair, so 2 GB at n = 4000, nj = 4 -- but every quantity
# needed from them is a scalar per pair, so a block only has to live long enough to be reduced.
# This is what makes N >= 5 reachable: the basis has to grow with the Jacobi dimension, and at
# Nc = 5 an unchunked n = 4000 does not fit.
nb = len(b)
BLK = max(1, int(4e7 // max(1, nb * nj * nj)))
S = np.empty((nb, nb))
T = np.empty((nb, nb))
V = np.zeros((nb, nb))
R2 = np.empty((nb, nb))
QF = np.empty((len(Wk), nb, nb))          # exponents for F(q): wk^T C^-1 wk, kept for later
for s in range(0, nb, BLK):
    e = min(s + BLK, nb)
    Cb = A[s:e, None] + A[None, :]
    Sb = ((2 * np.pi) ** nj / np.linalg.det(Cb)) ** 1.5
    Cib = np.linalg.inv(Cb)
    S[s:e] = Sb
    T[s:e] = Sb * 1.5 * np.einsum('ij,njk,nmkl,mli->nm', Lam, A[s:e], Cib, A)
    for w in W:
        V[s:e] += -cpair * Sb * G(np.einsum('i,nmij,j->nm', w, Cib, w))
    R2[s:e] = Sb * 3.0 * np.einsum('ij,nmji->nm', Rmat, Cib)
    for iw, wk in enumerate(Wk):
        QF[iw, s:e] = np.einsum('i,nmij,j->nm', wk, Cib, wk)
    del Cb, Cib

ev, Uo = eigh(S)
keep = ev > args.scut * ev.max()
Uo = Uo[:, keep] / np.sqrt(ev[keep])
E, cvec = eigh(Uo.T @ (T + V) @ Uo)
g = Uo @ cvec[:, 0]
norm = g @ S @ g
r2 = (g @ R2 @ g) / norm
K = (g @ T @ g) / norm


def FF(q):
    f = np.exp(-0.5 * q * q * QF).mean(axis=0)
    return (g @ (S * f) @ g) / norm


print(f'SVM coulomb N={N} Nc={args.Nc} alpha={args.alpha}  pair coupling kappa={kappa:.6g}'
      f'  ({npair} pairs)')
print(f'  nbasis={len(b)} kept={keep.sum()} seed={args.seed} ascale={args.ascale} aC={aC:.6g}')
print(f'  E0 = {E[0]:.8g}   E1 = {E[1]:.8g}   gap = {E[1] - E[0]:.6g}')
print(f'  <r^2>_ch = {r2:.8g}')
print(f'  <K> = {K:.8g}   <K>/|E0| = {K / abs(E[0]):.6f}  (= 1 exactly for Coulomb, by virial)')
if N == 2:
    print(f'  closed form: <r^2>_ch = 27/(16 alpha^2) = {27 / (16 * args.alpha ** 2):.8g}'
          f'   E0 = -4/9 alpha^2 = {-4 / 9 * args.alpha ** 2:.8g}')
print(f'  SCALING: at coupling alpha these become E0 * (alpha/{args.alpha:g})^2 and'
      f' <r^2> * ({args.alpha:g}/alpha)^2, with F at q * alpha/{args.alpha:g}')
if args.qs:
    qs = [float(x) for x in args.qs.split(',')]
    ffs = [FF(q) for q in qs]
    print('  F(q)/F(0): ' + '  '.join(f'{q:g}:{f:.6f}' for q, f in zip(qs, ffs)))
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write('Q,F_over_F0\n0.0,1.0\n')
            for q, f in zip(qs, ffs):
                fh.write(f'{q},{f:.6f}\n')
            fh.write(f'# N={N} Nc={args.Nc} alpha={args.alpha} E0={E[0]:.8g} r2={r2:.8g}\n')
