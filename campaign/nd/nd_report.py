"""
Figures and tables for the N_d comparison at matched N_d alpha_d (same Lambda_d/m).
N_d = 3: correlated-Gaussian solver and Numerov (paper_final/svm_baryon_lam.csv).
N_d = 5, 7: colour-free GFMC (nd_collect.py) and Numerov meson.
Usage: python nd_report.py <prefix> <svm_baryon_lam.csv> <outdir>
"""
import sys, csv, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
pre, svm, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
R = list(csv.DictReader(open(f'{pre}_results.csv')))
S = list(csv.DictReader(open(svm)))
ff = np.load(f'{pre}_ff.npz'); cu = np.load(f'{pre}_curves.npz')
NA = [0.003, 0.009, 0.03, 0.06, 0.09, 0.15, 0.21]
col = {3: '#1f6fb4', 5: '#d1495b', 7: '#2a9d5c'}

def get(N, O, lam, kind='r2'):
    out = []
    for na in NA:
        al = na/N
        if N == 3:
            r = [x for x in S if abs(float(x['alpha'])-al) < 1e-9 and x['OLO'] == O and abs(float(x['lam'])-lam) < 1e-9]
            if not r: out.append((na, np.nan, 0)); continue
            out.append((na, float(r[0]['r2' if kind == 'r2' else 'r2_meson']), 0.0))
        else:
            r = [x for x in R if int(x['Nc']) == N and abs(float(x['alpha'])-al) < 1e-6 and x['OLO'] == O
                 and abs(float(x['lam'])-lam) < 1e-9 and float(x['scale']) == 1.0]
            if not r: out.append((na, np.nan, np.nan)); continue
            out.append((na, float(r[0]['r2' if kind == 'r2' else 'meson_r2']), float(r[0]['r2_err']) if kind == 'r2' else 0.0))
    return np.array(out)

# ---- Fig 1: rho^2 = (N m)^2 <r^2>/3 and <r^2>(N alpha)^2 vs N alpha, LO and NLO, lambda band ----
for kind in ('r2', 'meson'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, O in zip(axs, ('LO', 'NLO')):
        for N in (3, 5, 7):
            c = get(N, O, 1.0, kind); lo = get(N, O, 0.5, kind); hi = get(N, O, 2.0, kind)
            y = c[:, 1]*c[:, 0]**2; e = c[:, 2]*c[:, 0]**2
            band = np.sort(np.stack([lo[:, 1], c[:, 1], hi[:, 1]]), 0)*c[:, 0]**2
            ax.fill_between(c[:, 0], band[0], band[2], color=col[N], alpha=0.18, lw=0)
            ax.errorbar(c[:, 0], y, e, fmt='o-', ms=4, color=col[N], lw=1, label=f'$N_d={N}$' + (' (solver)' if N == 3 and kind == 'r2' else ''))
        ax.set_xscale('log'); ax.set_xlabel(r'$N_d\,\alpha_d(m_\psi)$'); ax.set_title(f'{"baryon" if kind == "r2" else "meson"}, {O}', fontsize=10)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel(r'$\langle r^2\rangle m_\psi^2\,(N_d\alpha_d)^2$'); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{od}/nd_{kind}_vs_Nalpha.png', dpi=150); plt.close(fig)

# ---- Fig 2: Isgur-Wise shape, F(Q)/F(0) vs Q^2 <r^2>, at N alpha = 0.15 LO lam 1 ----
fig, ax = plt.subplots(figsize=(5.5, 4))
x = np.linspace(0, 16, 200)
for N in (5, 7):
    k = f'nd{N}_LO_a{round(0.15/N, 7):g}_lam1'
    r = [z for z in R if z['name'] == k]
    if k in ff.files and r:
        Q, F, dF = ff[k]; r2 = float(r[0]['r2']); p = float(r[0]['p'])
        ax.errorbar(Q*Q*r2, F, dF, fmt='o', color=col[N], ms=4, label=f'$N_d={N}$ GFMC, $p={p:.2f}$')
        ax.plot(x, (1 + x/(6*p))**(-p), color=col[N], lw=0.8)
ax.plot(x, (1 + x/(6*3.48))**(-3.48), color=col[3], lw=0.8, label=r'$N_d=3$ solver shape, $p=3.48$')
ax.plot(x, np.exp(-x/6), 'k:', lw=0.8, label='Gaussian')
ax.set_xlabel(r'$Q^2\langle r^2\rangle$'); ax.set_ylabel(r'$F_1(Q^2)/F_1(0)$'); ax.set_ylim(0, 1.02)
ax.set_title(r'$N_d\alpha_d=0.15$, LO, $\lambda=1$', fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{od}/nd_ff_shape.png', dpi=150); plt.close(fig)

# ---- Fig 3: convergence in imaginary time, VMC-optimal vs compact starts, and N_d = 3 validation ----
pairs = [('nd5_LO_a0.03_lam1', 'nd5_LO_a0.03_lam1_cmp', 'nd5_LO_a0.03_lam1_wide', 5),
         ('nd7_LO_a0.0214286_lam1', 'nd7_LO_a0.0214286_lam1_cmp', 'nd7_LO_a0.0214286_lam1_wide', 7)]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, (a, b, wd, N) in zip(axs, pairs):
    for k, lab, c in ((a, 'VMC-optimal trial', col[N]), (b, r'trial $\langle r^2\rangle\times0.7$', 'gray'),
                      (wd, r'trial $\langle r^2\rangle\times1.4$', 'k')):
        if k in cu.files:
            t, r2, e = cu[k]
            ax.errorbar(t/t[-1], r2, e, fmt='o-', ms=3, lw=0.8, color=c, label=lab)
    rr = [z for z in R if z['name'] == a]
    if rr:
        ax.axhspan(float(rr[0]['r2']) - float(rr[0]['r2_err']), float(rr[0]['r2']) + float(rr[0]['r2_err']), color=col[N], alpha=0.15)
    ax.axvline(0.7, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r'$\tau/\tau_{\max}$ per leg ($|E_0|\tau_{\max}\approx 10$)'); ax.set_ylabel(r'$\langle r^2\rangle m_\psi^2$')
    ax.set_title(f'$N_d={N}$, $N_d\\alpha_d=0.15$, LO, $\\lambda=1$', fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{od}/nd_convergence.png', dpi=150); plt.close(fig)

# ---- validation table: N_d = 3 GFMC (this pipeline) vs solver ----
RHO = {}
with open(f'{od}/nd_validation.txt', 'w') as f:
    for z in R:
        if int(z['Nc']) != 3: continue
        al, O, lam = float(z['alpha']), z['OLO'], float(z['lam'])
        s = [x for x in S if abs(float(x['alpha'])-al) < 1e-9 and x['OLO'] == O and abs(float(x['lam'])-lam) < 1e-9][0]
        r2, e, rs = float(z['r2']), float(z['r2_err']), float(s['r2'])
        E0, dE, Es = float(z['E0']), float(z['E0_err']), float(s['E0'])
        f.write(f"{z['name']}: r2 {r2:.6g} +- {e:.3g} vs solver {rs:.6g}: {r2/rs-1:+.2%} ({(r2-rs)/e:+.1f} sigma);  "
                f"E0 {E0:.6g} +- {dE:.2g} vs {Es:.6g}: {E0/Es-1:+.3%}\n")
    # brackets: each start relaxes to r2 = r2_true + rho (r2_T - r2_true), with one residual fraction rho per N
    # at the production imaginary time.  Weighted straight-line fit of r2 against the trial r2 over the starts.
    for names in pairs:
        N = names[-1]
        pts = [(float(z['r2_T']), float(z['r2']), float(z['r2_err']), z['name']) for nm in names[:-1] for z in R if z['name'] == nm]
        if len(pts) < 2: continue
        t, y, e = (np.array([q[k] for q in pts]) for k in range(3))
        A = np.stack([np.ones_like(t), t], 1)/e[:, None]
        cov = np.linalg.inv(A.T @ A); a, rho = cov @ A.T @ (y/e)
        chi2 = float(np.sum(((y - a - rho*t)/e)**2))
        tr = a/(1 - rho)
        g = np.array([1/(1 - rho), a/(1 - rho)**2]); dtr = float(np.sqrt(g @ cov @ g))
        RHO[N] = (rho, float(np.sqrt(cov[1, 1])))
        f.write(f"bracket N_d={N}: " + "; ".join(f"{q[3].split('_')[-1] if q[3].endswith(('cmp', 'wide')) else 'VMC-opt'} "
                f"trial {q[0]:.6g} -> {q[1]:.6g} +- {q[2]:.3g}" for q in pts) +
                f"\n   line fit: residual fraction rho = {rho:.3f} +- {np.sqrt(cov[1, 1]):.3f}, r2(tau->inf) = {tr:.6g} +- {dtr:.3g}, "
                f"chi2 = {chi2:.2f} ({len(pts) - 2} dof)\n")
        pr = [q for q in pts if not q[3].endswith(('cmp', 'wide'))]
        if pr:
            t1, x1, e1, _ = pr[0]
            f.write(f"   VMC-optimal start: {x1:.6g} +- {e1:.3g}, bias vs the line limit {(x1 - tr)/tr:+.2%}; "
                    f"corrected {(x1 - rho*t1)/(1 - rho):.6g}\n")
print(open(f'{od}/nd_validation.txt').read())

# ---- main table ----
with open(f'{od}/nd_table.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['N_d', 'alpha_d', 'N_d alpha_d', 'order', 'lambda', 'r2_baryon', 'err', 'rho2', 'err', 'p', 'E0', 'r2_meson', 'r2_T', 'r2_corrected', 'corr_err'])
    for N in (3, 5, 7):
        for O in ('LO', 'NLO'):
            for lam in (0.5, 1.0, 2.0):
                b = get(N, O, lam, 'r2'); m = get(N, O, lam, 'meson')
                for (na, r2, e), (_, rm, _) in zip(b, m):
                    z = [x for x in R if int(x['Nc']) == N and abs(float(x['alpha'])-na/N) < 1e-6 and x['OLO'] == O
                         and abs(float(x['lam'])-lam) < 1e-9 and float(x['scale']) == 1.0]
                    p = float(z[0]['p']) if z else (3.48 if N == 3 else np.nan)
                    E0 = float(z[0]['E0']) if z else np.nan
                    tT = float(z[0]['r2_T']) if z else np.nan
                    rho, drho = RHO.get(N, (0.0, 0.0))
                    rc = (r2 - rho*tT)/(1 - rho) if z else r2
                    ec = np.hypot(e/(1 - rho), drho*abs(r2 - tT)/(1 - rho)**2) if z else e
                    w.writerow([N, f'{na/N:.6g}', na, O, lam, f'{r2:.6g}', f'{e:.3g}', f'{N*N*r2/3:.6g}', f'{N*N*e/3:.3g}',
                                f'{p:.3g}', f'{E0:.6g}', f'{rm:.6g}', f'{tT:.6g}', f'{rc:.6g}', f'{ec:.3g}'])
print('report written to', od)
