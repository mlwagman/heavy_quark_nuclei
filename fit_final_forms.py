"""
Global (Q^2, alpha) fits of all dark-hadron matrix elements for the paper.

Per (scheme, system, alpha) dataset [prod_a*/{scheme}_{system}/prod_ME.csv]:
  F_N(q^2)/N  = [1 + q^2 b^2/(4p)]^{-p}          (generalized dipole; p=2 exact
                                                   for 2-body Coulomb)
  <r^2>       = (3/2) b^2                         (slope of F_N)
  F_V0 - F_N  = +(K/2m) g(q^2),  F_N - F_S = (K/2m) g(q^2):
                fit  D(q^2) = D0 + D1 q^2         (value + derivative, per paper)
  F_Vi        = (q/2M) F_N   exactly  (Galilean; direct meas = systematic)
  F_T0i       = (q/2m) F_N   exactly  (kinematic)
  F_Ai = F_Tij = 0                     (selection rules)

alpha-dependence:
  Coulomb:  b(alpha) = c/alpha  (exact 1-parameter law)
  MRS:      3-point interpolation b(alpha) = b0 + b1*alpha + b2*alpha^2
            (saturating; radius nearly alpha-independent)

Outputs: paper_final/final_forms.csv, final_forms.tex, FF_summary.pdf
"""
import csv, glob, os, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUT = 'paper_final'
os.makedirs(OUT, exist_ok=True)
# also drop the table straight into the Overleaf project so the paper picks it up
OVERLEAF = ("/Users/user/Library/CloudStorage/Dropbox/Apps/Overleaf/"
            "Dark baryons direct detection")

SCHEMES = ['coul', 'mrs_lo', 'mrs_nlo']
SYSTEMS = ['meson', 'baryon']
ALPHAS  = [0.2, 0.3, 0.5]

def gendip(q2, b, p): return (1.0 + q2*b*b/(4*p))**(-p)

def load(scheme, system, al):
    base = f'prod_a{al}/{scheme}_{system}'
    # prefer a higher-statistics / longer-tau rerun when present
    d = base
    for suf in ('_long', '_hi'):
        if os.path.exists(base + suf + '/prod_ME.csv'):
            d = base + suf
    fn = os.path.join(d, 'prod_ME.csv')
    if not os.path.exists(fn): return None
    D = {}
    for r in csv.DictReader(open(fn)):
        D.setdefault(float(r['Q']), {})[r['channel']] = (float(r['F']), float(r['F_err']))
    K = None
    logf = os.path.join(d, 'run_Q0.0.log')
    if os.path.exists(logf):
        for line in open(logf, errors='ignore'):
            m = re.search(r'tau=0\.000.*<K>=([+-][0-9.]+)', line)
            if m: K = float(m.group(1)); break
    return D, K, d

rows = []
data_for_plot = {}
for scheme in SCHEMES:
    for system in SYSTEMS:
        N = 2 if system == 'meson' else 3
        for al in ALPHAS:
            got = load(scheme, system, al)
            if got is None: continue
            D, K, srcdir = got
            if 0.0 not in D or 'C3_N' not in D[0.0]: continue
            F0 = D[0.0]['C3_N'][0]
            Qs = np.array(sorted(q for q in D if q > 0))
            q2 = Qs**2
            F  = np.array([D[q]['C3_N'][0]/F0 for q in Qs])
            E  = np.array([max(D[q]['C3_N'][1]/F0, 1e-4) for q in Qs])
            try:
                if system == 'meson':
                    # 2-body bound state: exactly dipole, p fixed to 2
                    pm, cm = curve_fit(lambda x, b: gendip(x, b, 2.0), q2, F,
                                       p0=[6], sigma=E, absolute_sigma=True, maxfev=8000)
                    pb = np.array([pm[0], 2.0]); eb = np.array([np.sqrt(cm[0,0]), 0.0])
                    chi = np.sum(((gendip(q2, *pb)-F)/E)**2)/max(len(Qs)-1, 1)
                else:
                    # baryon: bound p to a physical range so tightly-bound (small-r)
                    # cases don't run to a degenerate |p|->inf
                    pb, cb = curve_fit(gendip, q2, F, p0=[6, 3], sigma=E,
                                       absolute_sigma=True, maxfev=8000,
                                       bounds=([0.5, 1.0], [40.0, 12.0]))
                    eb = np.sqrt(np.diag(cb))
                    chi = np.sum(((gendip(q2, *pb)-F)/E)**2)/max(len(Qs)-2, 1)
            except Exception:
                continue
            b, p = pb
            r2 = 1.5*b*b; r2e = 3*b*eb[0]
            # relativistic-correction differences (value + slope, per paper needs)
            dS = dV = (np.nan,)*4
            if all('C3_S' in D[q] for q in D):
                DS = np.array([(D[q]['C3_N'][0]-D[q]['C3_S'][0])/F0 for q in sorted(D)])
                DV = np.array([(D[q]['C3_V0'][0]-D[q]['C3_N'][0])/F0 for q in sorted(D)])
                qq = np.array(sorted(D))**2
                cS = np.polyfit(qq, DS, 1); cV = np.polyfit(qq, DV, 1)
                dS = (cS[1], cS[0]); dV = (cV[1], cV[0])
            rows.append(dict(scheme=scheme, system=system, alpha=al,
                             b=b, b_err=eb[0], p=p, p_err=eb[1],
                             r2=r2, r2_err=r2e, chi2=chi, K=K,
                             DS0=dS[0], DS1=dS[1], DV0=dV[0], DV1=dV[1],
                             src=srcdir))
            data_for_plot[(scheme, system, al)] = (Qs, F, E, pb)

with open(f'{OUT}/final_forms.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows: w.writerow(r)

# ---- alpha-dependence fits ----
alpha_rows = []
for scheme in SCHEMES:
    for system in SYSTEMS:
        sub = [r for r in rows if r['scheme'] == scheme and r['system'] == system]
        if len(sub) < 2: continue
        als = np.array([r['alpha'] for r in sub])
        bs  = np.array([r['b'] for r in sub]); bes = np.array([r['b_err'] for r in sub])
        if scheme == 'coul':
            c, cc = curve_fit(lambda a, c: c/a, als, bs, sigma=bes, absolute_sigma=True)
            chi = np.sum(((c[0]/als-bs)/bes)**2)/max(len(sub)-1, 1)
            alpha_rows.append(dict(scheme=scheme, system=system,
                                   form='b=c/alpha', params=f"c={c[0]:.3f}+-{np.sqrt(cc[0,0]):.3f}",
                                   chi2=chi))
        else:
            if len(sub) == 3:
                coef = np.polyfit(als, bs, 2)
                alpha_rows.append(dict(scheme=scheme, system=system,
                                       form='b=b2 a^2+b1 a+b0',
                                       params=f"b2={coef[0]:.2f} b1={coef[1]:.2f} b0={coef[2]:.2f}",
                                       chi2=0.0))
with open(f'{OUT}/alpha_forms.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['scheme', 'system', 'form', 'params', 'chi2'])
    w.writeheader()
    for r in alpha_rows: w.writerow(r)

# ---- LaTeX table ----
def fmt(v, e, d=2):
    return f"${v:.{d}f}({int(round(e*10**d))})$" if np.isfinite(v) and np.isfinite(e) else "--"
lab = {'coul': 'Coulomb LO', 'mrs_lo': 'MRS LO', 'mrs_nlo': 'MRS NLO'}
with open(f'{OUT}/final_forms.tex', 'w') as f:
    f.write("% auto-generated by fit_final_forms.py\n")
    f.write("\\begin{table*}[t]\n\\centering\n\\begin{tabular}{llcccccc}\n\\hline\\hline\n")
    f.write("potential & state & $\\alpha_d$ & $b$ & $p$ & $\\langle r^2\\rangle$ "
            "& $\\langle K\\rangle/m_\\psi$ & $\\chi^2/\\mathrm{dof}$\\\\\n\\hline\n")
    for r in rows:
        f.write(f"{lab[r['scheme']]} & {r['system']} & {r['alpha']} & "
                f"{fmt(r['b'], r['b_err'])} & {fmt(r['p'], r['p_err'])} & "
                f"{fmt(r['r2'], r['r2_err'], 1)} & "
                f"{r['K']:.4f} & {r['chi2']:.2f}\\\\\n" if r['K'] else
                f"{lab[r['scheme']]} & {r['system']} & {r['alpha']} & "
                f"{fmt(r['b'], r['b_err'])} & {fmt(r['p'], r['p_err'])} & "
                f"{fmt(r['r2'], r['r2_err'], 1)} & -- & {r['chi2']:.2f}\\\\\n")
    f.write("\\hline\\hline\n\\end{tabular}\n")
    f.write("\\caption{Fitted dark-hadron charge (Dirac) form factor "
            "$F_1(q^2)/F_1(0)=[1+q^2b^2/(4p)]^{-p}$, extracted from the GFMC "
            "matrix element $\\langle J_{V^0}\\rangle$; $\\langle r^2\\rangle=3b^2/2$ "
            "and $F_1(0)=N$. The same profile sets every channel "
            "(Sec.~\\ref{app:gfmc:me2ff}): $F_{V^i}=(q^i/2M_\\Psi)F_1$ and "
            "$\\langle J_{T^{0i}}\\rangle=(q^i/2m_\\psi)F_1$ exactly, the scalar "
            "$F_S=F_1-(\\langle K\\rangle/2m_\\psi)g(q^2)$ with $\\langle K\\rangle/m_\\psi$ "
            "listed, and the spin-dependent form factors carry the same $q^2$ shape "
            "with group-theory normalizations. Orbital $J_{A^i},J_{T^{ij}}$ vanish.}\n"
            "\\label{tab:FF_fits}\n\\end{table*}\n")

# ---- summary figure ----
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey='row')
cols = {0.2: '#2166AC', 0.3: '#D6604D', 0.5: '#4DAC26'}
for i, system in enumerate(SYSTEMS):
    for j, scheme in enumerate(SCHEMES):
        ax = axes[i][j]
        for al in ALPHAS:
            key = (scheme, system, al)
            if key not in data_for_plot: continue
            Qs, F, E, pb = data_for_plot[key]
            ax.errorbar(Qs**2, F, yerr=E, fmt='o', ms=4, color=cols[al],
                        label=f"$\\alpha_d={al}$")
            xx = np.linspace(0, 0.27, 200)
            ax.plot(xx, gendip(xx, *pb), '-', lw=1, color=cols[al])
        ax.set_title(f"{lab[scheme]} {system}", fontsize=10)
        if i == 1: ax.set_xlabel("$q^2$")
        if j == 0: ax.set_ylabel("$F_N(q^2)/F_N(0)$")
        ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f'{OUT}/FF_summary.pdf')
print(f"wrote {OUT}/final_forms.csv ({len(rows)} fits), alpha_forms.csv "
      f"({len(alpha_rows)}), final_forms.tex, FF_summary.pdf")
for r in rows:
    print(f"  {r['scheme']:8s} {r['system']:6s} a={r['alpha']}: "
          f"b={r['b']:.3f}({r['b_err']:.3f}) p={r['p']:.2f}({r['p_err']:.2f}) "
          f"r2={r['r2']:.1f} chi2={r['chi2']:.2f}")
for r in alpha_rows:
    print(f"  alpha-fit {r['scheme']:8s} {r['system']:6s}: {r['form']}  {r['params']}  chi2={r['chi2']:.2f}")

# ---- mirror deliverables into the Overleaf project (paper \inputs table_FF_fits) ----
if os.path.isdir(OVERLEAF):
    import shutil
    shutil.copy(f'{OUT}/final_forms.tex', os.path.join(OVERLEAF, 'table_FF_fits.tex'))
    figdir = os.path.join(OVERLEAF, 'figs')
    if os.path.isdir(figdir):
        shutil.copy(f'{OUT}/FF_summary.pdf', os.path.join(figdir, 'FF_summary.pdf'))
    print("  -> copied table_FF_fits.tex (+ figs/FF_summary.pdf) into Overleaf project")
