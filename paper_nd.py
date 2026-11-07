"""
Tables and figures for the N_d dependence of the dark baryon (N_d = 3, 5, 7 at matched N_d alpha_d).

Inputs in paper_final/ (campaign/nd, README there):
  nd_results.csv  GFMC radius, shape p and energy per dataset (nd_collect.py), with the trial radius r2_T
  nd_ff.npz       late-time F(Q) per dataset;  nd_curves.npz  r2(tau) per dataset
  svm_baryon_lam.csv, svm_ff.csv   N_d = 3 variational radii and form factors (paper_solver.py)
Outputs in paper_final/: nd_radius.{pdf,png}, nd_convergence.{pdf,png}, table_nd.tex; numbers quoted in the text are printed.

Radii at N_d = 5, 7 are corrected for the remaining trial dependence: each start relaxes to
r2 = r2_inf + rho (r2_T - r2_inf), with rho fixed per N_d by the starts 0.7x, 1x and 1.4x the trial radius at
N_d alpha_d = 0.15 (LO, lambda = 1).  r2_corr = (r2 - rho r2_T)/(1 - rho), error including the error of rho.
Usage: python paper_nd.py [paper_final]
"""
import sys, csv, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else 'paper_final'
NA = [0.003, 0.009, 0.03, 0.06, 0.09, 0.15, 0.21]          # N_d alpha_d(m_psi); N_d = 3: alpha_d = 0.001 ... 0.07
R = {r['name']: r for r in csv.DictReader(open(f'{OUT}/nd_results.csv'))}
S = list(csv.DictReader(open(f'{OUT}/svm_baryon_lam.csv')))
FF = np.load(f'{OUT}/nd_ff.npz'); CU = np.load(f'{OUT}/nd_curves.npz')
EXF = {}
for r in csv.DictReader(open(f'{OUT}/svm_ff.csv')):
    if r['system'] == 'baryon':
        EXF.setdefault((float(r['alpha']), r['OLO'], float(r['lam'])), []).append((float(r['q']), float(r['F'])))

def name(N, na, O, lam, tag=''):
    return f'nd{N}_{O}_a{round(na/N, 7):g}_lam{lam:g}{tag}'

# ---- trial dependence: straight line through the three starts, per N_d ----
RHO = {}
for N in (5, 7):
    ks = [name(N, 0.15, 'LO', 1.0, t) for t in ('', '_cmp', '_wide')]
    t = np.array([float(R[k]['r2_T']) for k in ks]); y = np.array([float(R[k]['r2']) for k in ks]); e = np.array([float(R[k]['r2_err']) for k in ks])
    A = np.stack([np.ones(3), t], 1)/e[:, None]; cov = np.linalg.inv(A.T @ A); a, rho = cov @ A.T @ (y/e)
    chi2 = float(np.sum(((y - a - rho*t)/e)**2)); lim = a/(1 - rho)
    g = np.array([1/(1 - rho), a/(1 - rho)**2]); dlim = float(np.sqrt(g @ cov @ g))
    RHO[N] = (rho, float(np.sqrt(cov[1, 1])), lim, dlim)
    print(f'N_d={N}: starts r2_T {t.round(1)} -> r2 {y.round(1)} +- {e.round(1)}; rho = {rho:.3f}({np.sqrt(cov[1, 1]):.3f}), '
          f'r2(inf) = {lim:.1f}({dlim:.1f}), chi2 = {chi2:.2f} (1 dof); production bias {(y[0]-lim)/lim:+.2%}')

# the form factor at each q from the same three starts, taken at r2_T = r2(inf); shape compared with N_d = 3
from scipy.optimize import curve_fit
q3, F3 = np.array(EXF[(0.05, 'LO', 1.0)]).T; r2_3 = float([x for x in S if abs(float(x['alpha']) - 0.05) < 1e-9 and x['OLO'] == 'LO'
                                                            and float(x['lam']) == 1.0][0]['r2'])
FINF, PINF = {}, {}
for N in (5, 7):
    ks = [name(N, 0.15, 'LO', 1.0, t) for t in ('', '_cmp', '_wide')]
    t = np.array([float(R[k]['r2_T']) for k in ks]); Q = FF[ks[0]][0]; lim, dlim = RHO[N][2:]
    Fi, dFi, chi = [], [], 0.0
    for i in range(len(Q)):
        y = np.array([FF[k][1][i] for k in ks]); e = np.array([FF[k][2][i] for k in ks])
        A = np.stack([np.ones(3), t], 1)/e[:, None]; cov = np.linalg.inv(A.T @ A); a, b = cov @ A.T @ (y/e)
        chi += float(np.sum(((y - a - b*t)/e)**2)); g = np.array([1.0, lim])
        Fi.append(a + b*lim); dFi.append(float(np.sqrt(g @ cov @ g + (b*dlim)**2)))
    Fi, dFi = np.array(Fi), np.array(dFi); FINF[N] = (Q, Fi, dFi)
    (r2f, PINF[N]), _ = curve_fit(lambda x, r2, p: (1 + r2*x*x/(6*p))**(-p), Q[:5], Fi[:5], p0=(lim, 4.0), sigma=dFi[:5])
    dev = Fi/np.interp(Q*Q*lim, q3*q3*r2_3, F3) - 1; ddev = dFi/np.interp(Q*Q*lim, q3*q3*r2_3, F3)
    print(f'N_d={N}: F(q) on the start line chi2 = {chi:.1f} (6 dof); fit to F(inf): r2 = {r2f:.1f} (line {lim:.1f}), p = {PINF[N]:.2f}')
    print('   vs N_d=3 shape at q^2 r2 = ' + ', '.join(f'{x:.2f}: {100*d:+.1f}({100*dd:.1f})%' for x, d, dd in zip(Q*Q*lim, dev, ddev)))

def gfmc(N, na, O, lam):
    r = R[name(N, na, O, lam)]
    r2, e, rT = float(r['r2']), float(r['r2_err']), float(r['r2_T'])
    rho, drho = RHO[N][:2]
    return (r2 - rho*rT)/(1 - rho), float(np.hypot(e/(1 - rho), drho*abs(r2 - rT)/(1 - rho)**2)), float(r['p']), float(r['p_err'])

def solver(na, O, lam):
    al = na/3
    return float([x for x in S if abs(float(x['alpha']) - al) < 1e-9 and x['OLO'] == O and abs(float(x['lam']) - lam) < 1e-9][0]['r2'])

def X(N, na, O, lam):        # <r^2> m_psi^2 (N_d alpha_d)^2 and its error
    if N == 3: return solver(na, O, lam)*na**2, 0.0
    v, e, _, _ = gfmc(N, na, O, lam); return v*na**2, e*na**2

# ---- numbers for the text ----
for N in (5, 7):
    for O in ('LO', 'NLO'):
        rat = [X(N, na, O, 1.0)[0]/X(3, na, O, 1.0)[0] for na in NA]
        print(f'N_d={N} {O}: <r2>/<r2>_3 at fixed N alpha = ' + ' '.join(f'{x:.3f}' for x in rat))
    P = np.array([gfmc(N, na, O, lam)[2:] for na in NA for O in ('LO', 'NLO') for lam in (0.5, 1.0, 2.0)])
    w = 1/P[:, 1]**2; pm = (w*P[:, 0]).sum()/w.sum()
    print(f'N_d={N}: shape p = {pm:.2f}({w.sum()**-0.5:.2f}), chi2/dof = {((P[:, 0]-pm)**2*w).sum()/(len(P)-1):.2f} over {len(P)} datasets')
    es = [gfmc(N, na, O, lam)[1]/gfmc(N, na, O, lam)[0] for na in NA for O in ('LO', 'NLO') for lam in (0.5, 1.0, 2.0)]
    sh = [gfmc(N, na, O, lam)[0]/float(R[name(N, na, O, lam)]['r2']) - 1 for na in NA for O in ('LO', 'NLO') for lam in (0.5, 1.0, 2.0)]
    print(f'N_d={N}: relative errors {min(es):.2%}-{max(es):.2%}, trial corrections {min(sh):+.2%} to {max(sh):+.2%}')
for O in ('LO', 'NLO'):
    for N in (3, 5, 7):
        hw = [(max(X(N, na, O, l)[0] for l in (0.5, 1.0, 2.0)) - min(X(N, na, O, l)[0] for l in (0.5, 1.0, 2.0)))/2/X(N, na, O, 1.0)[0] for na in NA]
        print(f'{O} N_d={N}: lambda half-width ' + ' '.join(f'{x:.1%}' for x in hw))
for k in ('nd3_LO_a0.05_lam1', 'nd3_LO_a0.001_lam1', 'nd3_NLO_a0.07_lam1', 'nd3_NLO_a0.01_lam0.5'):
    r = R[k]; al, O, lam = float(r['alpha']), r['OLO'], float(r['lam'])
    s = [x for x in S if abs(float(x['alpha'])-al) < 1e-9 and x['OLO'] == O and abs(float(x['lam'])-lam) < 1e-9][0]
    d = float(r['r2'])/float(s['r2']) - 1
    print(f'check {k}: r2 {float(r["r2"]):.5g}({float(r["r2_err"]):.2g}) vs variational {float(s["r2"]):.5g}: {d:+.2%} '
          f'({(float(r["r2"])-float(s["r2"]))/float(r["r2_err"]):+.1f} sigma); E0 {float(r["E0"])/float(s["E0"])-1:+.3%}')

# ---- table ----
rows = []
for na in NA:
    c = [f'{na:g}', f'{na/3:g}']
    for N in (3, 5, 7):
        for O in ('LO', 'NLO'):
            v, e = X(N, na, O, 1.0)
            c.append(f'{v:.2f}' if N == 3 else f'{v:.1f}({round(10*e):d})' if e >= 0.095 else f'{v:.2f}({round(100*e):d})')
    rows.append(' & '.join(c) + r'\\')
with open(f'{OUT}/table_nd.tex', 'w') as f:
    f.write('% generated by paper_nd.py\n\\begin{table}[t]\n\\centering\n\\begin{tabular}{ll|cc|cc|cc}\n\\hline\\hline\n'
            '& & \\multicolumn{2}{c|}{$N_d=3$} & \\multicolumn{2}{c|}{$N_d=5$} & \\multicolumn{2}{c}{$N_d=7$}\\\\\n'
            '$N_d\\alpha_d$ & $\\alpha_d\\,(N_d=3)$ & LO & NLO & LO & NLO & LO & NLO\\\\\n\\hline\n' + '\n'.join(rows) +
            '\n\\hline\\hline\n\\end{tabular}\n'
            '\\caption{At the same $N_d\\alpha_d(m_\\psi)$ the dark baryon is larger for larger $N_d$, by a factor that depends little on '
            'the coupling. We give $\\langle r^2\\rangle m_\\psi^2(N_d\\alpha_d)^2$ at $\\lambda=1$. The $N_d=3$ values are the variational '
            'solutions of Table~\\ref{tab:smallalpha}, and the $N_d=5,7$ values are GFMC (App.~\\ref{app:gfmc:nd}), with the '
            'uncertainty in parentheses, including that of the trial correction. Fig.~\\ref{fig:nd} shows the variation with $\\lambda$.}\n\\label{tab:nd}\n\\end{table}\n')

# ---- figures ----
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
COL = {3: '#6250d6', 5: '#e34948', 7: '#008300'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 10,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
x = np.array(NA)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
for ax, O in zip(axes[:2], ('LO', 'NLO')):
    for N in (3, 5, 7):
        y = np.array([X(N, na, O, 1.0) for na in NA]); lo = np.array([X(N, na, O, 0.5)[0] for na in NA]); hi = np.array([X(N, na, O, 2.0)[0] for na in NA])
        ax.fill_between(x, np.minimum(lo, hi), np.maximum(lo, hi), color=COL[N], alpha=0.15, lw=0)
        ax.plot(x, y[:, 0], '-', color=COL[N], lw=1.4)
        if N > 3:
            ax.errorbar(x, y[:, 0], y[:, 1], fmt='o', ms=3.2, color=COL[N], mec=INK, mew=0.35, elinewidth=0.8, capsize=1.2, zorder=5)
        ax.annotate(f'$N_d={N}$', (x[0], y[0, 0]), xytext=(4, 4), textcoords='offset points', fontsize=8, color=INK)
    ax.set_xscale('log'); ax.set_xlim(2.4e-3, 0.26); ax.set_ylim(8, 62); ax.set_title(f'baryon, MRS {O}', fontsize=9)
    ax.set_xlabel(r'$N_d\,\alpha_d(m_\psi)$')
axes[0].set_ylabel(r'$\langle r^2\rangle\,m_\psi^2\,(N_d\alpha_d)^2$'); axes[1].set_yticklabels([])
ax = axes[2]; s = np.linspace(0, 16, 300)
q, F = np.array(EXF[(0.05, 'LO', 1.0)]).T; r2 = solver(0.15, 'LO', 1.0)
ax.plot(q*q*r2, F, '-', color=COL[3], lw=1.4, label=r'$N_d=3$')
for N in (5, 7):
    Q, Fi, dFi = FINF[N]; rr = RHO[N][2]; p = PINF[N]
    ax.plot(s, (1 + s/(6*p))**(-p), '-', color=COL[N], lw=1.0)
    ax.errorbar(Q*Q*rr, Fi, dFi, fmt='o', ms=3.2, color=COL[N], mec=INK, mew=0.35, elinewidth=0.8, label=f'$N_d={N}$', zorder=5)
ax.plot(s, np.exp(-s/6), ':', color=MUTED, lw=1.0, label='Gaussian')
ax.set_xlim(0, 16); ax.set_ylim(0, 1.03); ax.set_xlabel(r'$q^2\langle r^2\rangle$'); ax.set_ylabel(r'$\xi=F_1(q^2)/F_1(0)$')
ax.set_title(r'$N_d\alpha_d=0.15$, MRS LO', fontsize=9); ax.legend(loc='upper right')
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/nd_radius.{e}')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
for ax, N in zip(axes, (5, 7)):
    rho, drho, lim, dlim = RHO[N]
    ax.axhspan(lim - dlim, lim + dlim, color=COL[N], alpha=0.15, lw=0)
    for tag, f, mk, c in (('_wide', 1.4, 's', '#8a8984'), ('', 1.0, 'o', COL[N]), ('_cmp', 0.7, 'D', '#52514e')):
        t, r2, e = CU[name(N, 0.15, 'LO', 1.0, tag)]
        ax.errorbar(t/t[-1], r2, e, fmt=mk + '-', ms=2.8, lw=0.9, color=c, mec=INK, mew=0.3, elinewidth=0.7,
                    label=rf'trial $\langle r^2\rangle\times{f:g}$')
    ax.axvline(0.7, color=MUTED, lw=0.6, ls=':')
    ax.set_xlabel(r'$\tau/\tau_{\max}$ per leg'); ax.set_title(f'$N_d={N}$, $N_d\\alpha_d=0.15$, MRS LO, $\\lambda=1$', fontsize=9)
axes[0].set_ylabel(r'$\langle r^2\rangle\,m_\psi^2$'); axes[0].legend(loc='upper right')
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/nd_convergence.{e}')
plt.close(fig)
print('written', f'{OUT}/table_nd.tex nd_radius nd_convergence')
