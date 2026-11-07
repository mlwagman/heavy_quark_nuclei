"""Reduce one run directory (one Q, one seed) to a small npz and delete its h5/png:
C3[t] = C_J(t,t), W[t] = C2(2t) on the diagonal, binned over contiguous walker blocks,
and the leg-1 mixed energy E(t) = sum W H / sum W, also binned.
Usage: python reduce_sym.py <run_dir> <out.npz> [n_bins=50]"""
import sys, glob, os, re, numpy as np, h5py
d, out = sys.argv[1], sys.argv[2]; nb = int(sys.argv[3]) if len(sys.argv) > 3 else 50
fc = [f for f in glob.glob(f'{d}/currents_3pt_*.h5')]; fh = glob.glob(f'{d}/Hammys_*.h5')
assert len(fc) == 1 and len(fh) == 1, (fc, fh)
def binned(x):            # x (..., walkers) -> (..., nb)
    n = x.shape[-1]//nb*nb
    return x[..., :n].reshape(*x.shape[:-1], nb, -1).sum(-1)
with h5py.File(fc[0]) as h:
    C = np.real(np.array(h['C3_N_offdiag'])); W = np.real(np.array(h['Ws2_offdiag']))
    nt = C.shape[0]; C3 = binned(np.stack([C[i, i] for i in range(nt)])); W2 = binned(np.stack([W[i, i] for i in range(nt)]))
    tau = np.array(h['tau_values']); q = float(np.sqrt(np.array(h['q_squared']))); sym = int(h.attrs.get('sym_only', 0))
    cv = h.attrs.get('color_vectors', 'on')
with h5py.File(fh[0]) as h:
    Hm = np.array(h['Hammys']); Ws = np.array(h['Ws_leg1'])
    EW = binned(np.real(Ws*Hm)); WW = binned(np.real(Ws))
np.savez_compressed(out, Q=q, tau=tau, C3=C3, W2=W2, EW=EW, WW=WW, sym_only=sym, color_vectors=str(cv))
for f in glob.glob(f'{d}/*.h5') + glob.glob(f'{d}/*.png'): os.remove(f)
print(f'{out}: Q={q:.6g} nt={nt} bins={nb}')
