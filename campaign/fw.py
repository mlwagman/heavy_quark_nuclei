"""Forward-walking (pure) estimate of <r^2> = (1/N) sum_i |r_i - R_cm|^2 from a run's Q=0 walker file."""
import sys, glob, h5py, numpy as np
f = [x for x in glob.glob(f'{sys.argv[1]}/Rs_*Qmag0.0.h5')][0]
with h5py.File(f) as h: R = h['Rs'][:]; W = h['Ws_leg1'][:].real
cm = R.mean(axis=2, keepdims=True)
r2 = ((R - cm)**2).sum(-1).mean(-1)                    # (n_step+1, n_walkers)
n = R.shape[0] - 1; t0 = n // 3; nw = W.shape[1]; G = 40; g = np.arange(nw) % G
def est(T, m=None):
    ts = np.arange(t0, n + 1 - T)
    num, den = W[ts + T] * r2[ts], W[ts + T]
    if m is not None: num, den = num[:, m], den[:, m]
    return num.sum() / den.sum()
for T in (0, n // 6, n // 3):
    c = est(T); jk = np.array([est(T, g != k) for k in range(G)])
    e = np.sqrt((G - 1) * np.mean((jk - jk.mean())**2))
    print(f'forward walking T={T:4d} steps: <r^2> = {c:.6g} +/- {e:.3g}' + ('  (mixed)' if T == 0 else ''))
