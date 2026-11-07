"""
Deterministic VMC of the GFMC 'compact' trial prod_{i<j} exp(-r_ij/a0) for N equal-mass quarks,
with the generator's own MRS pair potential  V(r) = -VB(r/mufac)/r  (VB includes CF/(Nc-1)).

One Metropolis sample at a0 = 1 gives the pair distances and the kinetic constant
  <T> = t_N / a0^2,  t_N = (1/2) sum_k <|sum_{j!=k} rhat_kj|^2>,   <r^2>_ch = c_N a0^2,
so <H>(a0) for any potential is a reweighting-free rescaling of that sample.

Usage: python vmc_trial.py --Nc 5 --alpha 0.03 --OLO LO --mufac 1 [--mu MU | --mufac_cut 3.4]
Prints mu_cut = 3.4 Lambda_d(2-loop) when --mu is not given, a0_opt, E_T, trial <r^2>, spoila.
"""
import sys, os, argparse, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--Nc', type=int, default=5)
ap.add_argument('--N', type=int, default=0, help='number of quarks (default Nc; 2 = meson)')
ap.add_argument('--alpha', type=float, required=True)
ap.add_argument('--OLO', default='LO')
ap.add_argument('--mufac', type=float, default=1.0)
ap.add_argument('--mu', type=float, default=0.0, help='mu_cut; default 3.4 Lambda_d(2-loop)')
ap.add_argument('--nsamp', type=int, default=200000)
ap.add_argument('--cache', default=os.path.expanduser('~/hqn_local/cfree/vmc_samples'))
a = ap.parse_args()
N = a.N or a.Nc
assert N != 2, 'meson: use mrs_numerov.py'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py')

def gen_ns(mu):
    argv = (f"x --n_step 10 --n_walkers 2 --dtau_iMev 0.5 --count_skip 5 --nf 0 --Nc {a.Nc} --N_coord {N} "
            f"--mQ 1.0 --mu {mu} --mufac {a.mufac} --Rstar 0.35 --spoila 1 --wavefunction compact --potential full "
            f"--OLO {a.OLO} --alpha_s {a.alpha} --Q_dir 1 0 0 --outdir /tmp/xx/ --Q_mag 0.0 --no_color_vectors").split()
    src = open(SRC).read()
    sys.argv = argv
    ns = {'__name__': 'vmc_trial'}
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(src[:src.index("os.makedirs(outdir")], ns)
    return ns

mu = a.mu
if mu <= 0:
    mu = 3.4*gen_ns(1.0)['Lambda_dark_2loop']
ns = gen_ns(mu)
VB = ns['VB_LO'] if a.OLO == 'LO' else ns['VB_NLO']
a0_sp1 = float(2/VB(0.35))                       # generator: a0 = spoila*2/VB(Rstar), baryon (no /(Nc-1))

# ---- sample |psi|^2 at a0 = 1 (cached per N) ----
os.makedirs(a.cache, exist_ok=True)
cf = f'{a.cache}/N{N}_{a.nsamp}.npz'
iu = np.triu_indices(N, 1)
if os.path.exists(cf):
    z = np.load(cf); r, tN, cN = z['r'], float(z['tN']), float(z['cN'])
else:
    rng = np.random.default_rng(12345); nw = 4000
    R = rng.normal(size=(nw, N, 3))
    def lp(R):
        d = np.linalg.norm(R[:, :, None] - R[:, None], axis=-1)
        return -2*d[:, iu[0], iu[1]].sum(1)
    l = lp(R); keep_r, keep_t, keep_c = [], [], []
    step = 1.2/N; it = 0
    while sum(len(x) for x in keep_r) < a.nsamp:
        Rn = R + step*rng.normal(size=R.shape); ln = lp(Rn)
        ok = np.log(rng.random(nw)) < ln - l; R[ok] = Rn[ok]; l[ok] = ln[ok]; it += 1
        if it > 2000 and it % 20 == 0:
            dv = R[:, :, None] - R[:, None]; d = np.linalg.norm(dv, axis=-1)
            keep_r.append(d[:, iu[0], iu[1]])
            rh = np.where(d[..., None] > 0, dv/np.where(d > 0, d, 1)[..., None], 0)
            g = rh.sum(2)                                        # sum_j rhat_kj, shape (nw, N, 3)
            keep_t.append(0.5*np.sum(g**2, axis=(1, 2)))
            cm = R.mean(1, keepdims=True); keep_c.append(np.mean(np.sum((R-cm)**2, -1), 1))
    r = np.concatenate(keep_r); tN = float(np.concatenate(keep_t).mean()); cN = float(np.concatenate(keep_c).mean())
    np.savez(cf, r=r, tN=tN, cN=cN)

from scipy.optimize import minimize_scalar
def E(la0):
    a0 = np.exp(la0); x = a0*r
    V = -np.array(VB(x.ravel()/a.mufac)).reshape(x.shape)/x
    return tN/a0**2 + V.sum(1).mean()
# bracket from the Coulomb estimate with the coupling at the trial size
k0 = float(VB(10.0/a.mufac))
res = minimize_scalar(E, bracket=(np.log(0.3/k0), np.log(3.0/k0)), tol=1e-8)
a0 = float(np.exp(res.x))
print(f"Nc={a.Nc} N={N} alpha={a.alpha} {a.OLO} mufac={a.mufac} mu_cut={mu:.8g}")
print(f"tN={tN:.5f} cN={cN:.5f}  a0_opt={a0:.6g}  E_T={res.fun:.8g}  r2_T={cN*a0**2:.6g}  spoila_opt={a0/a0_sp1:.6g}  a0_sp1={a0_sp1:.6g}")
