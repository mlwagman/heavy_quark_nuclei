#!/usr/bin/env python3
r"""Exact two-body reference for the FIXED-ORDER quarkonium of the currents generator.

The companion of ``mrs_two_body.py`` for ``heavy_quark_nuclei_gfmc_boosted_FV_currents.py``.
The generator's constants are read out of its own namespace (it is exec'd, as ``mrs_numerov.py``
does), so the potential solved here is the one the GFMC used:

    V_singlet(r) = -(Nc-1) * VB / r * (1 + alpha/(4 pi) * (2 beta0 log(r e^gamma mu) + aa1))
    VB = alpha CF / (Nc - 1),   so (Nc-1) VB = alpha CF

At LO the bracket is 1 and everything is closed form -- which is the point of using LO as the
anchor of a tutorial:

    a0 = 1/(mu_red alpha CF) = 1.5/alpha      (mu_red = 1/2, Nc = 3)
    E0 = -alpha^2 CF^2 mu_red/2 = -(2/9) alpha^2 ... = -4/9 alpha^2 at Nc=3 in these units
    <r^2>_ch = (3/4) a0^2 = 1.6875/alpha^2
    F(q)/F(0) = (1 + q^2 a0^2/16)^-2          an EXACT dipole, p = 2

so the Numerov solve is checked against the closed form before being trusted at NLO.  Reported
radii are one-body about the centre of mass, ``<r^2>_ch = <r^2>_rel/4``, the convention of
``gfmc_me_gevp.py`` and ``svm_baryon.py``.

Usage (from the directory holding the generator, with the jax interpreter):
    fo_two_body.py --OLO LO  --alpha 0.1 --qs 0.025,0.05,0.075,0.1,0.15,0.2
    fo_two_body.py --OLO NLO --alpha 0.1 --qs 0.025,0.05,0.075,0.1,0.15,0.2
"""
import argparse
import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal

ap = argparse.ArgumentParser()
ap.add_argument('--OLO', default='LO')
ap.add_argument('--alpha', type=float, default=0.1)
ap.add_argument('--Nc', type=int, default=3)
ap.add_argument('--nf', type=int, default=0)
ap.add_argument('--mu', type=float, default=1.0, help="the generator's --mu (renormalization scale)")
ap.add_argument('--log_mu_r', type=float, default=1.0, help="only enters the trial a0")
ap.add_argument('--h', type=float, default=0.002)
ap.add_argument('--rmax', type=float, default=None, help='default 60 a0')
ap.add_argument('--qs', default='', help='comma list of |q| for F(q)/F(0)')
ap.add_argument('--out', default='', help='optional csv for F(q)/F(0)')
a = ap.parse_args()

SRC = "heavy_quark_nuclei_gfmc_boosted_FV_currents.py"
src = open(SRC).read()
src = src[:src.index("os.makedirs(outdir")]
sys.argv = (f"x --n_step 10 --n_walkers 2 --dtau_iMev 1.0 --count_skip 5 --nf {a.nf} "
            f"--Nc {a.Nc} --N_coord 2 --masses 1 -1 --alpha {a.alpha} --mu {a.mu} "
            f"--log_mu_r {a.log_mu_r} --spoila 1 --wavefunction compact --potential full "
            f"--OLO {a.OLO} --Q_dir 1 0 0 --outdir /tmp/xx/ --Q_mag 0.0 "
            f"--no_color_vectors").split()
ns = {'__name__': 'fo_bench'}
exec(src, ns)
Nc, CF, VB = ns['Nc'], ns['CF'], ns['VB']
alpha, beta0, aa1, mu = ns['alpha'], ns['beta0'], ns['aa1'], ns['mu']
a0_trial = float(ns['a0'])
a0_exact = 1.0 / (0.5 * alpha * CF)

print(f"FIXED ORDER {a.OLO}  Nc={Nc} nf={a.nf} alpha={alpha} mu={mu}")
print(f"  CF = {CF}  VB = alpha CF/(Nc-1) = {VB}  beta0 = {beta0}  aa1 = {aa1}")
print(f"  trial a0 (spoila=1) = {a0_trial:.8g}   exact LO Bohr a0 = {a0_exact:.8g}")


def V(r):
    """The generator's ``singlet_potential_fun``, evaluated on a radial grid."""
    coul = -(Nc - 1) * VB / r
    if a.OLO == 'LO':
        return coul
    Rprime = r * np.exp(np.euler_gamma) * mu
    return coul * (1 + alpha / (4 * np.pi) * (2 * beta0 * np.log(Rprime) + aa1))


mu_red = 0.5
h = a.h
rmax = a.rmax if a.rmax is not None else 60.0 * a0_exact
r = np.arange(h, rmax, h)
n = len(r)
vals, vecs = eigh_tridiagonal(1.0 / (mu_red * h * h) + V(r),
                              -0.5 / (mu_red * h * h) * np.ones(n - 1),
                              select='i', select_range=(0, 2))
E0, E1, E2 = vals[:3]
u = vecs[:, 0]
u /= np.sqrt(np.sum(u * u) * h)
r2_rel = np.sum(u * u * r * r) * h
r2_ch = r2_rel / 4.0
Vexp = np.sum(u * u * V(r)) * h
Texp = E0 - Vexp

print(f"  E0 = {E0:.9f}   E1 = {E1:.9f}   E2 = {E2:.9f}   gap = {E1 - E0:.6g}")
print(f"  <r^2>_rel = {r2_rel:.6f}   <r^2>_ch = {r2_ch:.6f}")
print(f"  <T> = {Texp:.8f}   <T>/|E0| = {Texp / abs(E0):.6f}  (= 1 for pure Coulomb, by virial)")
if a.OLO == 'LO':
    print(f"  CLOSED FORM: E0 = -{(0.5 * (alpha * CF) ** 2 * mu_red):.9f}"
          f"   <r^2>_ch = 0.75 a0^2 = {0.75 * a0_exact ** 2:.6f}")


def FF(q):
    """(1/2) sum_k <e^{i q.(r_k - X)}> = <sinc(q r/2)> for the qqbar pair."""
    x = q * r / 2.0
    j0 = np.where(x < 1e-8, 1.0, np.sin(x) / np.maximum(x, 1e-30))
    return float(np.sum(u * u * j0) * h)


if a.qs:
    qs = [float(x) for x in a.qs.split(',')]
    ffs = [FF(q) for q in qs]
    print(f"  {'q':>8} {'F/F0':>10}" + ("  exact dipole   ratio" if a.OLO == 'LO' else ''))
    for q, f in zip(qs, ffs):
        line = f"  {q:8g} {f:10.6f}"
        if a.OLO == 'LO':
            ex = (1 + q * q * a0_exact ** 2 / 16.0) ** -2.0
            line += f"  {ex:12.6f}  {f / ex:7.5f}"
        print(line)
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write('Q,F_over_F0\n0.0,1.0\n')
            for q, f in zip(qs, ffs):
                fh.write(f'{q},{f:.8f}\n')
            fh.write(f'# OLO={a.OLO} alpha={alpha} mu={mu} E0={E0:.9g} r2_ch={r2_ch:.9g}\n')
