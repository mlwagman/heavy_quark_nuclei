#!/usr/bin/env python3
"""Radial solve of the MRS two-body problem with an arbitrary color factor.

  V(r) = -c * VB_{LO,NLO}(r/(mufac*mQ)) / r ,    c = Nc-1 (meson singlet), 1 (qq in 3bar)

Reports E0, <r^2>_rel, and the MEASURED asymptotic decay rate kappa of u(r) ~ e^{-kappa r},
which is what fixes the z-expansion branch point:  the one-body density of a two-body state
falls as e^{-4 kappa x}, so t_cut = 16 kappa^2.  For the qq pair this E0 is the baryon's
break-up threshold, so E_sep = E0(3-body) - E0(qq).

Run from the directory holding the MRS generator (``mike_20260824/code``), with the interpreter
that has jax -- the generator is exec'd for its Borel sum and running, so the potential here is
bit-for-bit the one the GFMC used.  ``--cpair 2`` reproduces ``mrs_numerov.py``.

At alpha_s(mQ) = 0.1, mu_cut = 0.0362001, mufac = 1 (the reference-mass campaign):

    order  cpair  E0             <r^2>_ch     t_cut
    LO     2      -0.017505892   56.69186     0.2800943   meson,  16|E0|
    NLO    2      -0.019235763   52.07114     0.3077722
    LO     1      -0.005324798   --           --          qq in 3bar: the baryon threshold
    NLO    1      -0.005863789   --           --

so with the variational E0(qqq) = -0.020959979 (LO) and -0.023149129 (NLO),

    t_cut(baryon) = 12 |E0(qqq) - E0(qq)| = 0.1876222 (LO), 0.2074241 (NLO).

The MEASURED tail slope comes out ~7% below sqrt(2 mu |E0|) because the potential still has a
Coulombic 1/r tail at the fit window (V(3000) = -1.1e-4), which multiplies the exponential by
r^gamma.  The branch point is set by the exponential, so use the energy value; in any case the
k_max >= 3 radius moves by under 1% over a factor 8 in t_cut.
"""
import sys, argparse, numpy as np
from scipy.linalg import eigh_tridiagonal

ap = argparse.ArgumentParser()
ap.add_argument('--OLO', default='LO')
ap.add_argument('--alpha_s', type=float, default=0.1)
ap.add_argument('--Nc', type=int, default=3)
ap.add_argument('--mu', type=float, default=None,
                help='mu_cut in units of mQ; default 3.4 * Lambda_2loop(Nc, alpha_s), which is '
                     'Ben\'s construction and is Nc dependent through beta0')
ap.add_argument('--mu_over_lam2', type=float, default=3.4,
                help='mu_cut in units of Lambda_2loop, used when --mu is not given')
ap.add_argument('--mufac', type=float, default=1.0)
ap.add_argument('--cpair', type=float, default=None,
                help='pair color factor relative to VB: Nc-1 for the singlet meson (default), '
                     '1 for the antisymmetric qq pair inside a baryon')
ap.add_argument('--h', type=float, default=0.01)
ap.add_argument('--rmax', type=float, default=3000.0)
a = ap.parse_args()
if a.cpair is None:
    a.cpair = a.Nc - 1

SRC = "heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py"
src = open(SRC).read()[:1]  # placeholder, replaced below
src = open(SRC).read()
src = src[:src.index("os.makedirs(outdir")]


def load(mu):
    """exec the generator for its VB_LO/VB_NLO and its Lambda scales at this mu_cut."""
    sys.argv = (f"x --n_step 100 --n_walkers 2 --dtau_iMev 0.5 --count_skip 5 --nf 0 "
                f"--Nc {a.Nc} --N_coord 2 --mQ 1.0 --mu {mu} --mufac {a.mufac} --Rstar 0.35 "
                f"--spoila 1 --wavefunction compact --potential full --OLO {a.OLO} "
                f"--alpha_s {a.alpha_s} --Q_dir 1 0 0 --outdir /tmp/xx/ --Q_mag 0.0").split()
    ns = {'__name__': 'mrs_bench'}
    exec(src, ns)
    return ns


# mu_cut is itself Nc dependent (Lambda_2loop runs with beta0 = 11 Nc/3), so read the scale off
# a throwaway solve and then reload at the mu it implies
ns = load(1.0 if a.mu is None else a.mu)
lam1, lam2 = float(ns['Lambda_dark']), float(ns['Lambda_dark_2loop'])
if a.mu is None:
    a.mu = a.mu_over_lam2 * lam2
    ns = load(a.mu)
VB = ns['VB_LO'] if a.OLO == 'LO' else ns['VB_NLO']
mufac, mQ = ns['mufac'], ns['mQ']

def V(r):
    return -a.cpair*np.array(VB(r/(mufac*mQ)))/r

mu_red, h = 0.5, a.h
r = np.arange(h, a.rmax, h); n = len(r)
vals, vecs = eigh_tridiagonal(1.0/(mu_red*h*h) + V(r), -0.5/(mu_red*h*h)*np.ones(n-1),
                              select='i', select_range=(0, 1))
E0, E1 = vals[:2]
u = vecs[:, 0]; u /= np.sqrt(np.sum(u*u)*h)
r2 = np.sum(u*u*r*r)*h
Vexp = np.sum(u*u*V(r))*h
Texp = E0 - Vexp

# measured asymptotic decay: fit log|u| over the window where u is still above the
# numerical floor but far outside the bulk
au = np.abs(u)
tail = (r > 6*np.sqrt(r2)) & (au > 1e-9*au.max())
kap_fit = -np.polyfit(r[tail], np.log(au[tail]), 1)[0]
kap_E = np.sqrt(2*mu_red*abs(E0))
print(f"MRS {a.OLO} Nc={a.Nc} cpair={a.cpair:g} alpha_s={a.alpha_s} mufac={a.mufac}")
print(f"  Lambda_1loop = {lam1:.8g}   Lambda_2loop = {lam2:.8g}   mu_cut = {a.mu:.8g}"
      f"  (= {a.mu / lam2:.4g} Lambda_2loop)")
print(f"  E0 = {E0:.9f}   E1 = {E1:.9f}   V(rmax) = {float(V(r[-1])):.3e}")
print(f"  <r^2>_rel = {r2:.5f}   <r^2>_ch = r2/4 = {r2/4:.5f}")
# two unit-free shape numbers; both equal (3, 1) for a pure Coulomb 1s state, so they measure
# how far the running/Borel potential is from Coulomb independently of any choice of units
print(f"  SHAPE  <r^2>_rel*|E0| = {r2*abs(E0):.6f}   <T>/|E0| = {Texp/abs(E0):.6f}"
      f"   <T> = {Texp:.8f}")
print(f"  kappa from E0 = sqrt(2 mu |E0|) = {kap_E:.8f}")
print(f"  kappa measured from u(r) tail   = {kap_fit:.8f}   (window {r[tail][0]:.0f}..{r[tail][-1]:.0f})")
print(f"  t_cut = 16 kappa^2 = {16*kap_fit**2:.8f}  (from E0: {16*kap_E**2:.8f})")
