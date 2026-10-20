"""Exact 2-body solve of the ACTUAL MRS potential (exec'd from the production
GFMC code, so identical Borel/running implementation), via finite-difference
radial Hamiltonian. Benchmarks the dark meson F_N(q), E0, <r^2>.

Usage: python mrs_numerov.py --OLO NLO --alpha_s 0.3
"""
import sys, argparse, numpy as np
from scipy.linalg import eigh_tridiagonal

ap = argparse.ArgumentParser()
ap.add_argument('--OLO', default='NLO')
ap.add_argument('--alpha_s', type=float, default=0.3)
ap.add_argument('--mufac', type=float, default=1.0,
                help='renormalization-scale factor; vary 0.5/1/2 for the '
                     'perturbative (scale) uncertainty')
ap.add_argument('--mu', type=float, default=2.0)
ap.add_argument('--quiet', action='store_true')
a = ap.parse_args()

SRC = "heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py"
argv = (f"x --n_step 100 --n_walkers 2 --dtau_iMev 0.5 --count_skip 5 --nf 0 "
        f"--Nc 3 --N_coord 2 --mQ 1.0 --mu {a.mu} --mufac {a.mufac} --Rstar 0.35 "
        f"--spoila 1 --wavefunction compact --potential full --OLO {a.OLO} "
        f"--alpha_s {a.alpha_s} --Q_dir 1 0 0 --outdir /tmp/xx/ --Q_mag 0.0").split()
src = open(SRC).read()
cut = src.index("os.makedirs(outdir")
sys.argv = argv
ns = {'__name__': 'mrs_bench'}
exec(src[:cut], ns)

VB = ns['VB_LO'] if a.OLO == 'LO' else ns['VB_NLO']
Nc = ns['Nc']; mufac = ns['mufac']; mQ = ns['mQ']
print(f"trial a0 (code) = {ns['a0']}")

def V(r):
    return -1*(Nc-1)*np.array(VB(r/(mufac*mQ)))/r

mu_red = 0.5
# box must comfortably contain the bound state; scale with the trial Bohr
# radius (fixed rmax=300 silently truncated the wavefunction for a0 >~ 10,
# e.g. alpha_s=0.01 gave a spurious unbound "E1>0")
rmax = max(300.0, 40.0*float(ns['a0']))
h = 0.01
r = np.arange(h, rmax, h); n = len(r)
main = 1.0/(mu_red*h*h) + V(r)
off  = -0.5/(mu_red*h*h)*np.ones(n-1)
vals, vecs = eigh_tridiagonal(main, off, select='i', select_range=(0,1))
E0, E1 = vals[:2]
u = vecs[:,0]; u /= np.sqrt(np.sum(u*u)*h)
r2 = np.sum(u*u*r*r)*h
print(f"EXACT MRS {a.OLO} meson alpha_s={a.alpha_s}:  E0 = {E0:.6f}  E1 = {E1:.6f}  <r^2>_rel = {r2:.2f}")
print(f"  charge radius <r^2>_rel/4 = {r2/4:.2f}")

def FF(q):
    x = q*r/2.0
    j0 = np.where(x < 1e-8, 1.0, np.sin(x)/np.maximum(x, 1e-30))
    return np.sum(u*u*j0)*h
print(f"{'Q':>6} {'exact F/F0':>11}")
for Q in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]:
    print(f"{Q:>6} {FF(Q):>11.4f}")
