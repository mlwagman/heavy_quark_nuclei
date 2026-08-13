"""
Correct two-sided GEVP form-factor extraction from the OFF-DIAGONAL two-leg
correlator.

Why this exists
---------------
The symmetric two-leg 3pt is  C3(t1,t2) = <psiT| e^{-H t1} J e^{-H t2} |psiT>.
The ground-state form factor is

    <0|J|0> = (v^T M3 v) / (v^T M2 v),

with  M3[i,j] = C3(tau_i, tau_j)            (OFF-DIAGONAL, independent legs)
      M2[i,j] = <tau_i|tau_j> = C2(tau_i+tau_j)   (single-leg 2pt overlap metric)
and v the ground Ritz vector from the shift-GEVP on M2.

The old Lanczos_twoleg.py was WRONG: it fed the DIAGONAL C3(tau_i+tau_j) into the
off-diagonal Hankel slots. That mixes spectral rates E_m+E_n against a 2pt
projector built for rate 2E_n, so the dominant E0+E1 cross term is never removed
-> the radius overshoots badly (meson: a=8.75 vs analytic 5.0).

This estimator needs C3_N_offdiag and Ws2_offdiag, stored by the GFMC currents
code (see heavy_quark_nuclei_gfmc_boosted_FV_currents.py).

Validation: LO Coulomb meson (alpha=0.3, analytic a0=5.0, dipole
F/F0=1/(1+(q a0/4)^2)^2):  k=2 GEVP gives a0 = 4.9 +/- 0.5 at both Q=0.02,0.05.
Read the k=2 (2-state) result; higher k overfits once the tiny Coulomb gap +
finite walkers can't resolve more states (bootstrap error explodes -> flag).
"""
import h5py, numpy as np, glob, math, argparse
from scipy.linalg import eig

def sym(A): return 0.5*(A+A.T)

def ground_F(M3m, M2m, k, tau, Esh):
    """Two-sided GEVP form factor from k-dim window. Decay shift Esh only
    improves conditioning; it cancels in the ratio."""
    d = np.exp(-Esh*tau)
    M2s = M2m*np.outer(d,d); M3s = M3m*np.outer(d,d)
    H0 = sym(M2s[:k,:k]); H1 = sym(M2s[:k,1:k+1])
    w,V = eig(H1,H0); w = np.real(w)
    idx = [i for i in range(len(w)) if 0 < w[i] < 1.5]   # physical decay factors
    if not idx: return np.nan
    g = idx[int(np.argmax(w[idx]))]
    v = np.real(V[:,g])
    return (v@sym(M3s[:k,:k])@v) / (v@sym(M2s[:k,:k])@v)

def load(path):
    h = h5py.File(path,'r')
    M3 = np.real(np.array(h['C3_N_offdiag']))   # (N,N,Nw)
    M2 = np.real(np.array(h['Ws2_offdiag']))
    tau = np.array(h['tau_values']); h.close()
    return M3, M2, tau

def extract(currents_file, F0_file, k=2, nboot=400, seed=7):
    M3,M2,tau = load(currents_file)
    M30,M20,_ = load(F0_file)
    Nw = M3.shape[2]
    c = np.polyfit(tau[:6], np.log(np.abs(M20.mean(2)[0,:6])), 1)[0]
    Esh = c + 0.01
    F0 = ground_F(M30.mean(2), M20.mean(2), k, tau, Esh)
    Fc = ground_F(M3.mean(2), M2.mean(2), k, tau, Esh)
    rng = np.random.default_rng(seed); Fb=[]
    for _ in range(nboot):
        idx = rng.integers(0,Nw,Nw)
        Fb.append(ground_F(M3[:,:,idx].mean(2), M2[:,:,idx].mean(2), k, tau, Esh))
    Fb = np.array([x for x in Fb if not np.isnan(x)])
    return Fc/F0, np.percentile(Fb/F0,[16,50,84]), Esh

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--currents_file', required=True)
    p.add_argument('--F0_file', required=True, help='Q=0 currents file for normalization')
    p.add_argument('--k', type=int, default=2)
    p.add_argument('--Q', type=float, required=True)
    a = p.parse_args()
    fr, pct, Esh = extract(a.currents_file, a.F0_file, k=a.k)
    # meson dipole radius: 1 - F/F0 = (a0^2/8) Q^2 ; baryon: <r^2>=6(1-F/F0)/Q^2
    a0 = math.sqrt(abs(8*(1-fr)))/a.Q
    print(f"F/F0 = {fr:.5f}   a0(meson-dipole) = {a0:.3f}   "
          f"boot F/F0 = [{pct[0]:.5f},{pct[1]:.5f},{pct[2]:.5f}]")
