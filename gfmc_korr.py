"""
Matrix elements from GFMC two-time data with korr (FRR Eq. 62).

With Krylov vectors |b_k> = e^{-H k dt}|psi_T> on the stored grid:
  2pt  C2(t_k)        = Ws2[0, k]           leg 1 at 0, leg 2 to t_k
  3pt  C_J(sig, tau)  = C3N[tau, sig]       leg 1 = source->current = tau,
                                            leg 2 = current->sink   = sigma
GFMC weights grow as e^{+c t}; a common shift e^{-Es t}, Es = c + 0.05, makes
the 2pt decay and is applied as e^{-Es(sig+tau)} to the 3pt, which leaves J_00
unchanged. The current is the rest-frame relative current, so source and sink
share one spectrum and one set of rotators.

Usage: python gfmc_korr.py <reduced.npz> <meson|baryon> [ref] [--out results.csv]
"""
import sys, csv, warnings
import numpy as np
from scipy.optimize import curve_fit
import korr.analysis
from korr.analysis.ritz_rotators import RitzRotators
from korr.analysis.matrix_elements import matrix_elements
warnings.filterwarnings('ignore')

def J00(npz, m_max=12, n_boot=100):
    d = np.load(npz); out = {}
    for iq, Q in enumerate(d['Q']):
        c2 = d['Ws2'][iq][0].T                      # (Nb, Nt)
        c3 = d['C3N'][iq].transpose(2, 1, 0)        # (Nb, sigma, tau)
        Nt = c2.shape[1]; k = np.arange(Nt)
        c = np.polyfit(k[:6], np.log(np.abs(c2.mean(0)[:6])), 1)[0]
        sh = np.exp(-(c + 0.05) * k)
        spec = korr.analysis.spectroscopy(c2 * sh, ritz_coeffs=True, m_max=min(m_max, Nt // 2),
                                          n_outer=n_boot, n_inner=n_boot,
                                          sensitivity=False, summarize=False)
        me = matrix_elements(c3 * (sh[:, None] * sh[None, :])[None],
                             RitzRotators.from_result(spec, side='L'),
                             RitzRotators.from_result(spec, side='R'), assume_aligned=True)
        out[float(Q)] = (float(np.real(np.asarray(me.final['me'])[0, 0])),
                         np.real(np.asarray(me.final_boot['me'])[:, 0, 0]))
    return out

def fit(Qs, F, E, system):
    ok = (F > 0.05) & (F < 1)                                 # start from the p=2 inversion
    b0 = np.sqrt(np.median(8*(F[ok]**-0.5 - 1)/Qs[ok]**2)) if ok.any() else 6.
    if system == 'meson':
        p, _ = curve_fit(lambda x, b: (1 + x*b*b/8.)**-2., Qs**2, F, p0=[b0], sigma=E)
        return 1.5*p[0]**2, 2.0
    p, _ = curve_fit(lambda x, b, pp: (1 + x*b*b/(4*pp))**-pp, Qs**2, F, p0=[b0, 3],
                     sigma=E, bounds=([b0/20, 1], [b0*20, 12]))
    return 1.5*p[0]**2, p[1]

def radius(o, system):
    F0 = o[0.0][0]; Qs = np.array(sorted(q for q in o if q > 0))
    F = np.array([o[q][0] / F0 for q in Qs])
    # nanstd: korr occasionally finds no reportable state in a single resample;
    # one NaN draw must not turn the whole error bar into NaN
    E = np.array([max(np.nanstd(o[q][1] / o[0.0][1], ddof=1), 1e-5) for q in Qs])
    r2, p = fit(Qs, F, E, system)
    # Error: refit every bootstrap draw. korr draws the same resamples at every Q
    # (fixed seed), so the draws keep the correlation between Q points. The error
    # of a single fit with diagonal E ignores it and comes out ~2x too small.
    rb = []
    for i in range(len(o[0.0][1])):
        Fi = np.array([o[q][1][i] / o[0.0][1][i] for q in Qs])
        if np.all(np.isfinite(Fi)):
            try: rb.append(fit(Qs, Fi, E, system)[0])
            except (RuntimeError, ValueError): pass
    return F0, r2, np.std(rb, ddof=1), p, Qs, F, E

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out_csv = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
    if out_csv in args: args.remove(out_csv)
    npz, system = args[0], args[1]
    ref = float(args[2]) if len(args) > 2 else None
    o = J00(npz)
    F0, r2, er, p, Qs, F, E = radius(o, system)
    line = f'{npz}: F(0)={F0:.5f}  <r^2>={r2:.6g} +/- {er:.3g}  p={p:.2f}'
    if ref: line += f'  ref {ref:.6g}  {100*(r2/ref-1):+.2f}%  ({(r2-ref)/er:+.1f} sigma)'
    print(line)
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['Q', 'F_over_F0', 'err'])
            for q, x, e in zip(Qs, F, E): w.writerow([q, x, e])
            w.writerow([]); w.writerow(['F0', F0]); w.writerow(['r2', r2]); w.writerow(['r2_err', er])
            w.writerow(['p', p])
            if ref: w.writerow(['ref', ref])
