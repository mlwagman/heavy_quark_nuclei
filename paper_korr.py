"""
Paper tables and figures, for the couplings where the perturbative potential applies (alpha_d <= 0.07).

    table_smallalpha.tex   scales and exact radii at every coupling
    table_FF_fits.tex      GFMC form factor fits against the exact radii
    table_coulval.tex      Coulomb GFMC against the exact scaling law
    table_ansatz.tex       z-expansion against generalized dipole
    form_factors.pdf       F(q^2)/F(0), GFMC points and exact curves
    radius_vs_alpha.pdf    <r^2> and <r^2>/<r^2>_Coulomb against alpha_d
    gfmc_exact.pdf         GFMC radii relative to the exact ones

Inputs in paper_final/: korr_cache_svm.pkl (MRS baryon), korr_cache_coulval.pkl (Coulomb, gfmc_korr.J00 for each npz),
svm_baryon_lam.csv and svm_ff.csv (paper_solver.py), gfmc_radii.csv (GFMC and forward-walking radii), coul_baryon_ff.csv.
Errors on p and <r^2> come from refitting every korr bootstrap draw.

Usage: python paper_korr.py [outdir]
"""
import sys, csv, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from gfmc_korr import fit, radius
warnings.filterwarnings('ignore')

OUT = sys.argv[1] if len(sys.argv) > 1 else 'paper_final'
ALPHAS = [0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.07]
RATIO = 3.4                                   # mu_cut / Lambda_d
CB, CM = 3.826, 27/16                         # alpha_d^2 <r^2>_Coulomb, baryon and meson
cache = pickle.load(open(f'{OUT}/korr_cache_svm.pkl', 'rb'))
coul = pickle.load(open(f'{OUT}/korr_cache_coulval.pkl', 'rb'))
T = {(float(r['alpha']), r['OLO'], float(r['lam'])): r for r in csv.DictReader(open(f'{OUT}/svm_baryon_lam.csv'))}
EX = {}
for r in csv.DictReader(open(f'{OUT}/svm_ff.csv')):
    EX.setdefault((float(r['alpha']), r['OLO'], float(r['lam']), r['system']), []).append((float(r['q']), float(r['F'])))
EX = {k: np.array(sorted(v)).T for k, v in EX.items()}
G = [dict(r, alpha=float(r['alpha']), lam=float(r['lam']), **{k: float(r[k]) for k in ('r2', 'r2_err', 'r2_exact', 'r2_fw', 'r2_fw_err')})
     for r in csv.DictReader(open(f'{OUT}/gfmc_radii.csv')) if float(r['alpha']) <= ALPHAS[-1]]
val = lambda a, O, lam, c: float(T[(a, O, lam)][c])
gendip = lambda q2, r2, p: (1 + q2*r2/(6*p))**(-p)

def zfit(Qs, F, E, t, K):
    """<r^2> from a K-term z-expansion, z = (sqrt(t+q^2)-sqrt t)/(sqrt(t+q^2)+sqrt t)"""
    z = (np.sqrt(t + Qs**2) - np.sqrt(t)) / (np.sqrt(t + Qs**2) + np.sqrt(t))
    a, _ = curve_fit(lambda zz, *c: 1 + sum(c[k] * zz**(k + 1) for k in range(K)), z, F, p0=[-1] + [0] * (K - 1), sigma=E)
    return -1.5 * a[0] / t

def analyse(o, system):
    F0, r2, er, p, Qs, F, E = radius(o, system)
    draws = []
    for i in range(len(o[0.0][1])):
        Fi = np.array([o[q][1][i] / o[0.0][1][i] for q in Qs])
        if not np.all(np.isfinite(Fi)): continue
        try: draws.append((Fi, *fit(Qs, Fi, E, system)))
        except (RuntimeError, ValueError): pass
    Fd = np.array([d[0] for d in draws]); r2d = np.array([d[1] for d in draws]); pd = np.array([d[2] for d in draws])
    t = 12 / r2                                # = 16/a0^2 for the Coulomb meson
    z = {K: (zfit(Qs, F, E, t, K), np.std([zfit(Qs, Fi, E, t, K) for Fi in Fd], ddof=1)) for K in (2, 3)}
    C = np.cov(Fd.T); res = F - gendip(Qs**2, r2, p)
    chi2 = res @ np.linalg.solve(C, res) / (len(Qs) - (1 if system == 'meson' else 2))
    return dict(F0=F0, r2=r2, er=np.std(r2d, ddof=1), p=p, ep=np.std(pd, ddof=1), Qs=Qs, F=F, E=E, chi2=chi2, z2=z[2], z3=z[3])

def pexact(q, F, r2):
    """p of the generalized dipole with <r^2> fixed to its exact value, and the largest residual where F > 0.05"""
    ok = F > 0.05
    p, _ = curve_fit(lambda x, pp: gendip(x, r2, pp), q[ok]**2, F[ok], p0=[3], bounds=(1, 20))
    return p[0], np.abs(gendip(q[ok]**2, r2, p[0]) - F[ok]).max()

def pe(v, e):
    """value(error) with two significant digits on the error"""
    d = max(0, 1 - int(np.floor(np.log10(e))))
    return f'{v:.{d}f}({round(e*10**d):d})'

R = {n: analyse(o, 'baryon') for n, o in cache.items()}
RC = {n: analyse(o, 'baryon' if 'baryon' in n else 'meson') for n, o in coul.items()}
cq, cF = np.loadtxt(f'{OUT}/coul_baryon_ff.csv', delimiter=',', skiprows=2, unpack=True, comments='#')   # alpha = 0.03
PC_B = pexact(cq, cF, CB/0.03**2)
E0C_B = -0.03**-2 * float(open(f'{OUT}/coul_baryon_ff.csv').read().split('E0=')[1].split()[0])            # = <K>/alpha^2 by the virial theorem

# ---------------- scales and exact radii ----------------
def mantissa(x):
    e = int(np.floor(np.log10(x))); return f'{x/10**e:.1f}\\times10^{{{e}}}' if e < -1 else f'{x:.3f}'
lines = []
for a in ALPHAS:
    Ld = val(a, 'LO', 1.0, 'mu_cut')/RATIO; rp = np.sqrt(3*val(a, 'LO', 1.0, 'r2'))
    row = [f'{a}', f'${mantissa(Ld)}$', f'${mantissa(Ld*rp)}$', f'${mantissa(RATIO*Ld*rp)}$']
    for col, C in (('r2_meson', CM), ('r2', CB)):
        for O in ('LO', 'NLO'):
            c = val(a, O, 1.0, col); band = max(abs(val(a, O, l, col) - c) for l in (0.5, 2.0))*a*a/C
            row.append(f'${c*a*a/C:.3f}({round(band*1000):d})$')
    lines.append(' & '.join(row) + '\\\\')
open(f'{OUT}/table_smallalpha.tex', 'w').write(
    '% generated by paper_korr.py\n\\begin{table*}[t]\n\\centering\n\\begin{tabular}{lccc|cc|cc}\n\\hline\\hline\n'
    '& & & & \\multicolumn{2}{c|}{meson} & \\multicolumn{2}{c}{baryon}\\\\\n'
    '$\\alpha_d$ & $\\Lambda_d/m_\\psi$ & $\\Lambda_d r$ & $\\mu_{\\rm cut}r$ & LO & NLO & LO & NLO\\\\\n\\hline\n'
    + '\n'.join(lines) +
    '\n\\hline\\hline\n\\end{tabular}\n'
    '\\caption{The bound state stays well above the confinement scale at every coupling we use, and its radius falls below\n'
    'the Coulomb value as the coupling grows between $m_\\psi$ and the bound-state scale.\n'
    '$\\Lambda_d$ is the two-loop scale for $n_f=0$, and $r$ is the rms pair separation of the baryon at LO and $\\lambda=1$.\n'
    'The last four columns give the exact $\\langle r^2\\rangle/\\langle r^2\\rangle_{\\rm Coul}$, with\n'
    '$\\langle r^2\\rangle_{\\rm Coul}=27/(16\\alpha_d^2m_\\psi^2)$ for the meson and $3.826/(\\alpha_d^2m_\\psi^2)$ for the baryon.\n'
    'The number in parentheses is the largest change in the last digits for $\\lambda=1/2,2$.}\n'
    '\\label{tab:smallalpha}\n\\end{table*}\n')

# ---------------- form factor fits ----------------
rows = []
for system, C in (('meson', CM), ('baryon', CB)):
    for a in (0.001, 0.03):
        r = RC[f'coul_{system}_a{a}']; a2 = a*a
        pc = '$2$' if system == 'meson' else f'${r["p"]:.1f}({max(1, round(r["ep"]*10)):d})$'
        px = '$2$' if system == 'meson' else f'{PC_B[0]:.2f}'
        K = 4/9 if system == 'meson' else E0C_B       # virial theorem
        rows.append(f'Coulomb & {system} & {a} & {pc} & {px} & ${pe(r["r2"]*a2, r["er"]*a2)}$ & ${pe(r["z3"][0]*a2, r["z3"][1]*a2)}$ & {C:.3f} & {K:.3f} & {r["chi2"]:.2f}\\\\')
rows.append('\\hline')
for O in ('LO', 'NLO'):
    for a in ALPHAS:
        r = R[f'mrs_{O.lower()}_baryon_a{a}_lam1_svm']; a2 = a*a; ex = val(a, O, 1.0, 'r2')
        q, F = EX[(a, O, 1.0, 'baryon')]
        rows.append(f'MRS {O} & baryon & {a} & ${r["p"]:.1f}({max(1, round(r["ep"]*10)):d})$ & {pexact(q, F, ex)[0]:.2f} & ${pe(r["r2"]*a2, r["er"]*a2)}$ & '
                    f'${pe(r["z3"][0]*a2, r["z3"][1]*a2)}$ & {ex*a2:.3f} & {val(a, O, 1.0, "K")/a2:.3f} & {r["chi2"]:.2f}\\\\')
open(f'{OUT}/table_FF_fits.tex', 'w').write(
    '% generated by paper_korr.py\n\\begin{table*}[t]\n\\centering\n\\begin{tabular}{lllccccccc}\n\\hline\\hline\n'
    'potential & state & $\\alpha_d$ & $p$ & $p_{\\rm exact}$ & $\\alpha_d^2\\langle r^2\\rangle$ & $\\alpha_d^2\\langle r^2\\rangle_z$ & '
    '$\\alpha_d^2\\langle r^2\\rangle_{\\rm exact}$ & $\\langle K\\rangle/(\\alpha_d^2m_\\psi)$ & $\\chi^2/\\mathrm{dof}$\\\\\n\\hline\n'
    + '\n'.join(rows) +
    '\n\\hline\\hline\n\\end{tabular}\n'
    '\\caption{The GFMC charge form factor stays within $1.9\\sigma$ of the exact radius at every coupling, for $\\lambda=1$.\n'
    'The fits use $F_1(q^2)/F_1(0)=[1+q^2\\langle r^2\\rangle/(6p)]^{-p}$, Eq.~\\eqref{eq:app:gendip}, with $F_1$ from\n'
    '$\\langle J_{V^0}\\rangle$ and the Lanczos method (App.~\\ref{app:gfmc:lanczos}). Radii are in units of $m_\\psi^{-2}$.\n'
    'Uncertainties are statistical, from a bootstrap that refits every resample, and $\\chi^2/\\mathrm{dof}$ uses the bootstrap\n'
    'covariance of the $q$ points. $\\langle r^2\\rangle_z$ is the same radius from a three-term $z$-expansion\n'
    '(App.~\\ref{app:gfmc:ansatz}). The exact values come from the closed form (Coulomb meson) and the variational solution\n'
    '(baryon), and $p_{\\rm exact}$ is the fit of Eq.~\\eqref{eq:app:gendip} to the exact $F_1$ with the radius held fixed.\n'
    'The same profile sets every channel (Sec.~\\ref{app:gfmc:me2ff}): $F_{V^i}=(q^i/2M_\\Psi)F_1$ and\n'
    '$\\langle J_{T^{0i}}\\rangle=(q^i/2m_\\psi)F_1$ exactly, the scalar $F_S=F_1-(\\langle K\\rangle/2m_\\psi)g(q^2)$, and the\n'
    'spin-dependent form factors carry the same $q^2$ shape with group-theory normalizations.}\n'
    '\\label{tab:FF_fits}\n\\end{table*}\n')

# ---------------- Coulomb benchmark ----------------
rows = []
for system, C in (('meson', CM), ('baryon', CB)):
    for a in (0.5, 0.03, 0.001):
        r = RC[f'coul_{system}_a{a}']; a2 = a*a
        rows.append(f'{system} & {a} & ${pe(r["r2"]*a2, r["er"]*a2)}$ & ${100*(r["r2"]*a2/C-1):+.1f}\\%$ & ${(r["r2"]*a2-C)/(r["er"]*a2):+.1f}$\\\\')
open(f'{OUT}/table_coulval.tex', 'w').write(
    '% generated by paper_korr.py\n\\begin{table}[t]\n\\centering\n\\begin{tabular}{llccc}\n\\hline\\hline\n'
    'state & $\\alpha_d$ & $\\alpha_d^2\\langle r^2\\rangle$ & dev. & pull\\\\\n\\hline\n' + '\n'.join(rows) +
    '\n\\hline\\hline\n\\end{tabular}\n'
    '\\caption{GFMC follows the exact Coulomb scaling law over a factor 500 in $\\alpha_d$, with\n'
    '$\\alpha_d^2\\langle r^2\\rangle=27/16=1.6875$ (meson) and $3.826$ (baryon, variational).\n'
    'Runs at different $\\alpha_d$ are the same calculation in units of the Bohr radius, with\n'
    '$d\\tau\\propto\\alpha_d^{-2}$ and $q\\propto\\alpha_d$. Since a Coulomb potential has no running, $\\alpha_d$ only sets\n'
    'the units here, so these runs test the method at any $\\alpha_d$.}\n'
    '\\label{tab:coulval}\n\\end{table}\n')

# ---------------- fit-ansatz summary ----------------
groups = [('Coulomb meson', [v for k, v in RC.items() if 'meson' in k]), ('Coulomb baryon', [v for k, v in RC.items() if 'baryon' in k]),
          ('MRS baryon', list(R.values()))]
rows = []
for lab, ns in groups:
    d2 = np.array([r['z2'][0]/r['r2'] - 1 for r in ns])*100; d3 = np.array([r['z3'][0]/r['r2'] - 1 for r in ns])*100
    p3 = np.array([(r['z3'][0] - r['r2'])/r['er'] for r in ns])
    rows.append(f'{lab} & {len(ns)} & ${d2.mean():+.1f}\\%$ & ${d3.mean():+.1f}\\%$ & ${np.abs(d3).max():.1f}\\%$ & ${np.abs(p3).max():.1f}$\\\\')
open(f'{OUT}/table_ansatz.tex', 'w').write(
    '% generated by paper_korr.py\n\\begin{center}\n\\begin{tabular}{lccccc}\n\\hline\\hline\n'
    '& & 2 terms & \\multicolumn{3}{c}{3 terms}\\\\\ndatasets & $N$ & mean & mean & max & max/$\\sigma$\\\\\n\\hline\n'
    + '\n'.join(rows) + '\n\\hline\\hline\n\\end{tabular}\n\\end{center}\n')

# ---------------- figures ----------------
INK, MUTED, GRID, SHADE = '#0b0b0b', '#52514e', '#d8d7d2', '#ecebe8'
C_COUL, C_LO, C_NLO = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 10,
                     'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'axes.axisbelow': True, 'legend.frameon': False, 'legend.fontsize': 8, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
cmap = plt.get_cmap('Blues'); COL = {a: cmap(0.42 + 0.58*i/(len(ALPHAS) - 1)) for i, a in enumerate(ALPHAS)}

# form factors: against q (where direct detection reaches) and against q r (the shape)
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharey=True, constrained_layout=True)
for i, (system, col) in enumerate((('meson', 'r2_meson'), ('baryon', 'r2'))):
    for j, O in enumerate(('LO', 'NLO')):
        ax = axes[i][j]
        ax.axvspan(5e-3, 3, color=SHADE, lw=0, zorder=0)
        for a in ALPHAS:
            q, F = EX[(a, O, 1.0, system)]
            ax.plot(q, F, '-', color=COL[a], lw=1.4, zorder=2, label=f'{a}')
            if system == 'baryon':
                r = R[f'mrs_{O.lower()}_baryon_a{a}_lam1_svm']
                ax.errorbar(r['Qs'], r['F'], yerr=r['E'], fmt='o', ms=2.8, color=COL[a], mec=INK, mew=0.35, elinewidth=0.7, zorder=4)
        ax.set_xscale('log'); ax.set_xlim(8e-5, 1.2); ax.set_ylim(0, 1.04)
        ax.set_title(f'{system}, MRS {O}', fontsize=9)
        if i == 1: ax.set_xlabel(r'$q\ [m_\psi]$')
    ax = axes[i][2]
    for a in ALPHAS:
        q, F = EX[(a, 'LO', 1.0, system)]; rr = np.sqrt(val(a, 'LO', 1.0, col))
        ax.plot(q*rr, F, '-', color=COL[a], lw=1.1, zorder=2)
        if system == 'baryon':
            r = R[f'mrs_lo_baryon_a{a}_lam1_svm']
            ax.errorbar(r['Qs']*rr, r['F'], yerr=r['E'], fmt='o', ms=2.8, color=COL[a], mec=INK, mew=0.35, elinewidth=0.7, zorder=4)
    ax.set_xscale('log'); ax.set_xlim(0.05, 30); ax.set_title(f'{system}, MRS LO, scaled', fontsize=9)
    if i == 1: ax.set_xlabel(r'$q\,\sqrt{\langle r^2\rangle}$')
    axes[i][0].set_ylabel(r'$F_1(q^2)/F_1(0)$')
axes[1][0].legend(loc='lower left', title=r'$\alpha_d$', title_fontsize=8, labelspacing=0.15, handlelength=1.3, fontsize=7)
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/form_factors.{e}')
plt.close(fig)

# radius: log-log and relative to Coulomb
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, constrained_layout=True, gridspec_kw=dict(height_ratios=[1, 1]))
A = np.array(ALPHAS)
for j, (system, col, C) in enumerate((('meson', 'r2_meson', CM), ('baryon', 'r2', CB))):
    top, bot = axes[0][j], axes[1][j]
    top.plot(A, C/A**2, color=C_COUL, lw=1.4, label='Coulomb'); bot.axhline(1, color=C_COUL, lw=1.4, label='Coulomb')
    for O, cc in (('LO', C_LO), ('NLO', C_NLO)):
        y = lambda lam: np.array([val(a, O, lam, col) for a in ALPHAS])
        top.plot(A, y(1.0), '-', color=cc, lw=1.4, label=f'MRS {O}')
        bot.fill_between(A, np.minimum(y(0.5), y(2.0))*A**2/C, np.maximum(y(0.5), y(2.0))*A**2/C, color=cc, alpha=0.2, lw=0)
        bot.plot(A, y(1.0)*A**2/C, '-', color=cc, lw=1.4, label=f'MRS {O}')
        if system == 'baryon':
            for ax, f in ((top, 1.0), (bot, None)):
                g = sorted([d for d in G if d['OLO'] == O and d['lam'] == 1.0], key=lambda d: d['alpha'])
                x = np.array([d['alpha'] for d in g])*(0.97 if O == 'LO' else 1.03); s = np.array([d['alpha'] for d in g])**2/C if f is None else 1.0
                ax.errorbar(x, np.array([d['r2'] for d in g])*s, yerr=np.array([d['r2_err'] for d in g])*s, fmt='o' if O == 'LO' else 'D', ms=3.5, color=cc, mec=INK, mew=0.4,
                            elinewidth=0.8, capsize=1.5, zorder=5, label=f'GFMC {O}' if f is None else None)
    top.set_yscale('log'); top.set_title(f'dark {system}', fontsize=10); bot.set_ylim(0.25, 1.06)
    for ax in (top, bot): ax.set_xscale('log'); ax.set_xlim(8e-4, 0.085)
    bot.set_xlabel(r'$\alpha_d(m_\psi)$')
axes[0][0].set_ylabel(r'$\langle r^2\rangle\,m_\psi^2$'); axes[1][0].set_ylabel(r'$\langle r^2\rangle/\langle r^2\rangle_{\rm Coul}$')
axes[0][0].legend(loc='upper right'); axes[1][1].legend(loc='lower left', ncol=2)
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/radius_vs_alpha.{e}')
plt.close(fig)

# GFMC against exact
ST = {('LO', 0.5): ('v', C_LO, 0.86, r'LO, $\lambda=1/2$'), ('LO', 1.0): ('o', C_LO, 0.95, r'LO, $\lambda=1$'),
      ('LO', 2.0): ('^', C_LO, 1.05, r'LO, $\lambda=2$'), ('NLO', 1.0): ('D', C_NLO, 1.16, r'NLO, $\lambda=1$')}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, constrained_layout=True)
stats = {}
for ax, v, e, ttl in ((axes[0], 'r2', 'r2_err', 'from the form factor'), (axes[1], 'r2_fw', 'r2_fw_err', 'from the walkers')):
    ax.axhline(0, color=INK, lw=0.8)
    for k, (mk, cc, sh, lab) in ST.items():
        s = [d for d in G if (d['OLO'], d['lam']) == k]
        ax.errorbar([d['alpha']*sh for d in s], [100*(d[v]/d['r2_exact'] - 1) for d in s], yerr=[100*d[e]/d['r2_exact'] for d in s],
                    fmt=mk, ms=3.5, color=cc, mec=INK, mew=0.4, elinewidth=0.8, label=lab)
    dev = np.array([100*(d[v]/d['r2_exact'] - 1) for d in G]); pull = np.array([(d[v] - d['r2_exact'])/d[e] for d in G])
    stats[v] = (dev.mean(), dev.std(ddof=1)/np.sqrt(len(dev)), np.sqrt(np.mean(dev**2)), np.mean([100*d[e]/d['r2_exact'] for d in G]), np.mean(pull**2), np.abs(pull).max())
    ax.set_title(ttl, fontsize=9); ax.set_xscale('log'); ax.set_xlim(7e-4, 0.1); ax.set_ylim(-8.5, 8.5); ax.set_xlabel(r'$\alpha_d(m_\psi)$')
axes[0].set_ylabel('GFMC / exact $-$ 1  [%]'); axes[1].legend(loc='lower right', ncol=2)
for e in ('pdf', 'png'): fig.savefig(f'{OUT}/gfmc_exact.{e}')
plt.close(fig)

# ---------------- numbers for the text ----------------
print('N =', len(G), ' form factor and walkers: mean, err, rms, mean quoted err, chi2/N, max pull')
for k, s in stats.items(): print(f'  {k:6}', ' '.join(f'{x:.2f}' for x in s))
print(f'  form factor minus walkers, rms: {np.sqrt(np.mean([(100*(d["r2"] - d["r2_fw"])/d["r2_exact"])**2 for d in G])):.2f}%')
print(f'Coulomb baryon p_exact {PC_B[0]:.2f}, largest residual {PC_B[1]:.4f}, K/alpha^2 = {E0C_B:.4f}')
for system, col in (('baryon', 'r2'), ('meson', 'r2_meson')):
    for O in ('LO', 'NLO'):
        P = [pexact(*EX[(a, O, 1.0, system)], val(a, O, 1.0, col)) for a in ALPHAS]
        print(f'{system} {O}: p_exact', ' '.join(f'{p:.2f}' for p, _ in P), ' largest residual', f'{max(x for _, x in P):.4f}')
print('Coulomb family:'); [print(f'  {n:22} a2 r2 = {r["r2"]*float(n.split("_a")[1])**2:.4f} +/- {r["er"]*float(n.split("_a")[1])**2:.4f}  z3/dip {r["z3"][0]/r["r2"]-1:+.3f}') for n, r in RC.items()]
print('GFMC fits: p range', f'{min(r["p"] for r in R.values()):.2f}-{max(r["p"] for r in R.values()):.2f}', ' chi2/dof max', f'{max(r["chi2"] for r in R.values()):.2f}', ' F(0)/3-1 max', f'{max(abs(r["F0"]/3 - 1) for r in R.values()):.4f}')
