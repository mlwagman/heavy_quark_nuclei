"""
Collect the N_d campaign: per dataset, combine the walker splits, form the raw symmetric ratio
F(Q;t) = C_J(t,t) / (N C2(2t)), average it over the late window t >= TLATE*t_max, fit
F = (1 + r2 Q^2/(6p))^-p, and take the mixed energy over the same window.  Errors: jackknife over
walker bins (every bin refit).  Also r2(t) on each time slice for the convergence plots.

Usage: python nd_collect.py <results_dir> <datasets.csv> <out_prefix> [TLATE=0.7]
"""
import sys, glob, csv, re, numpy as np
from scipy.optimize import curve_fit
rd, dsf, outp = sys.argv[1:4]; TLATE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
NQFIT = 5
NSPLIT = 8
ds = list(csv.DictReader(open(dsf)))

def fit(Q, F, p0=(1.0, 3.0)):
    s = 1/Q[0]**2
    try:
        p, _ = curve_fit(lambda q, r2, pp: (1 + r2*q*q/(6*pp))**(-pp), Q, F, p0=(p0[0]*s, p0[1]), maxfev=20000)
        return p
    except Exception:
        return (np.nan, np.nan)

rows, curves, ffs = [], {}, {}
for d in ds:
    name, N = d['name'], int(d['Nc'])
    per_q = {}
    for f in sorted(glob.glob(f'{rd}/{name}_q*_s*.npz')):
        iq = int(re.search(r'_q(\d)_s\d\.npz$', f).group(1))
        per_q.setdefault(iq, []).append(np.load(f))
    if len(per_q) < 6 or any(len(v) < NSPLIT for v in per_q.values()):
        print(f'{name}: incomplete ({sum(len(v) for v in per_q.values())}/{6*NSPLIT})'); continue
    iqs = sorted(per_q)
    Q = np.array([float(per_q[i][0]['Q']) for i in iqs])
    tau = per_q[iqs[0]][0]['tau']
    C3 = np.stack([np.concatenate([z['C3'] for z in per_q[i]], -1) for i in iqs])     # (nq, nt, B)
    W2 = np.stack([np.concatenate([z['W2'] for z in per_q[i]], -1) for i in iqs])
    EW = np.concatenate([z['EW'] for i in iqs for z in per_q[i]], -1)                 # (nstep+1, B*nq)
    WW = np.concatenate([z['WW'] for i in iqs for z in per_q[i]], -1)
    B = C3.shape[-1]; late = tau >= TLATE*tau[-1]
    nst = EW.shape[0]; elate = np.arange(nst) >= TLATE*(nst - 1)
    def est(mask=None):
        c, w = (C3.sum(-1), W2.sum(-1)) if mask is None else ((C3*mask).sum(-1), (W2*mask).sum(-1))
        F = c/w/N
        Fl = F[:, late].mean(1)
        r2, p = fit(Q[:NQFIT], Fl[:NQFIT], (Q[0]**2*float(d['r2_T_opt']), 3.0))
        r2t = np.array([fit(Q[:NQFIT], F[:NQFIT, i], (Q[0]**2*float(d['r2_T_opt']), 3.0))[0] for i in range(len(tau))])
        return F, Fl, r2, p, r2t
    F, Fl, r2, p, r2t = est()
    jk = []
    for b in range(B):
        m = np.ones(B); m[b] = 0
        jk.append(est(m)[1:])
    J = lambda k: np.array([x[k] for x in jk])
    jerr = lambda a: np.sqrt((B-1)*np.nanvar(a, axis=0))
    dFl, dr2, dp, dr2t = jerr(J(0)), jerr(J(1)), jerr(J(2)), jerr(J(3))
    # mixed energy over the late window, jackknife over the same number of blocks
    Eb = EW[elate].sum(0)/WW[elate].sum(0)
    E = EW[elate].sum()/WW[elate].sum()
    nb = EW.shape[1]
    Ejk = np.array([(EW[elate].sum() - EW[elate][:, b].sum())/(WW[elate].sum() - WW[elate][:, b].sum()) for b in range(nb)])
    dE = np.sqrt((nb-1)*Ejk.var())
    rows.append(dict(name=name, Nc=N, alpha=d['alpha'], OLO=d['OLO'], lam=d['lam'], scale=d['scale'],
                     r2=r2, r2_err=dr2, p=p, p_err=dp, E0=E, E0_err=dE, r2_T=d['r2_T'], meson_r2=d['meson_r2'],
                     rho2=N*N*r2/3, rho2_err=N*N*dr2/3, chk_drift=(r2t[late][-1] - r2t[late][0])/max(dr2, 1e-30)))
    curves[name] = np.array([tau, r2t, dr2t])
    ffs[name] = np.array([Q, Fl, dFl])
    print(f"{name}: r2 = {r2:.6g} +- {dr2:.3g} ({dr2/r2:.2%})  p = {p:.3g}  E0 = {E:.6g} +- {dE:.2g}  trial r2 {float(d['r2_T']):.6g}")
if rows:
    with open(f'{outp}_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez(f'{outp}_curves.npz', **{k: v for k, v in curves.items()})
    np.savez(f'{outp}_ff.npz', **{k: v for k, v in ffs.items()})
