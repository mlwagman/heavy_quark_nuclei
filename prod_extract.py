"""
Production extraction of all current MEs from fixed-code GFMC runs.

For each currents_3pt_*.h5 in a directory:
  F_X(tau) = sum_w C3_X(tau,tau) / sum_w Ws2(tau,tau)   (diagonal two-leg ratio;
  valid with the FIXED r01 — no phase contamination, ratio plateaus)
  - usable window: CV(Ws2) < cv_max
  - plateau value: average over the LAST HALF of the usable window
  - bootstrap-over-walkers errors
Writes <dir>/prod_ME.csv: Q, channel, F, F_err, tau_lo, tau_hi.

Usage: python prod_extract.py <dir> [--cv_max 1.0]
"""
import h5py, numpy as np, glob, re, os, sys, csv, argparse

p = argparse.ArgumentParser()
p.add_argument('dir')
p.add_argument('--cv_max', type=float, default=1.0)
args = p.parse_args()

CHANNELS = ['C3_N','C3_S','C3_V0','C3_Vi','C3_Ai','C3_T0i','C3_Tij']

rows = []
files = sorted(f for f in glob.glob(os.path.join(args.dir,'currents_3pt_*.h5'))
               if 'korr' not in f)
for fn in files:
    m = re.search(r'Qmag([0-9.]+)\.h5', os.path.basename(fn))
    if not m: continue
    Q = float(m.group(1))
    h = h5py.File(fn,'r')
    W = np.real(np.array(h['Ws2_all']))              # (N_tau, Nw)
    tau = np.array(h['tau_values'])
    N_tau, Nw = W.shape
    cv = W.std(1)/np.maximum(np.abs(W.mean(1)),1e-30)
    cut = int(np.argmax(cv > args.cv_max)) if (cv > args.cv_max).any() else N_tau
    cut = max(cut, 4)
    lo, hi = max(1, cut//2), cut                      # last half of usable window
    rng = np.random.default_rng(12)
    boots = rng.integers(0, Nw, (200, Nw))
    for ch in CHANNELS:
        if ch not in h: continue
        C = np.real(np.array(h[ch]))                  # (Nw, N_tau[,3])
        ncomp = C.shape[2] if C.ndim == 3 else 1
        for ic in range(ncomp):
            Cs = C[:,:,ic] if C.ndim == 3 else C
            label = ch + (['_x','_y','_z'][ic] if C.ndim == 3 else '')
            vals = [Cs[:,i].sum()/W[i,:].sum() for i in range(lo,hi)]
            Fc = float(np.mean(vals))
            bs = [np.mean([Cs[b,i].sum()/W[i,b].sum() for i in range(lo,hi)])
                  for b in boots]
            rows.append(dict(Q=Q, channel=label, F=Fc, F_err=float(np.std(bs)),
                             tau_lo=float(tau[lo]), tau_hi=float(tau[hi-1])))
    h.close()
    print(f"  {os.path.basename(fn)[:40]}...  Q={Q}  window tau=[{tau[lo]:.1f},{tau[hi-1]:.1f}]")

out = os.path.join(args.dir,'prod_ME.csv')
with open(out,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=['Q','channel','F','F_err','tau_lo','tau_hi'])
    w.writeheader()
    for r in sorted(rows, key=lambda r:(r['Q'],r['channel'])): w.writerow(r)
print(f"wrote {out} ({len(rows)} rows)")
