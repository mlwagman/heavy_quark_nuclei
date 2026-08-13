"""
Paper-grade figures for the dark-hadron form factors.
  Fig 1  form_factors.pdf : F_1(q^2)/F_1(0), 2 rows (meson,baryon) x 3 cols
          (Coulomb, MRS LO, MRS NLO); GFMC points+fits + EXACT overlays.
  Fig 2  radius_vs_alpha.pdf : <r^2>(alpha_d), meson & baryon, 3 schemes;
          Coulomb 1/alpha^2 vs MRS saturation.
Design: alpha_d is ORDERED -> sequential blue ramp + distinct markers
(secondary encoding); schemes are categorical -> Okabe-Ito (validated).
"""
import csv, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator
from scipy.optimize import curve_fit

# ---- publication style (LaTeX-like via CM mathtext, no TeX install needed) ----
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.linewidth': 0.7, 'axes.labelsize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'legend.fontsize': 7.5, 'legend.frameon': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
ALPHAS = [0.2, 0.3, 0.5]
ARAMP  = {0.2: '#6baed6', 0.3: '#2171b5', 0.5: '#08306b'}  # sequential blue
AMARK  = {0.2: 'o', 0.3: 's', 0.5: '^'}
SCHEME = {'coul': '#0072B2', 'mrs_lo': '#D55E00', 'mrs_nlo': '#009E73'}  # categorical
SLAB   = {'coul': r'Coulomb LO', 'mrs_lo': r'MRS LO', 'mrs_nlo': r'MRS NLO'}
gendip = lambda q2, b, p: (1.0 + q2*b*b/(4*p))**(-p)

OUT = 'paper_final'; os.makedirs(OUT, exist_ok=True)
OVER = ("/Users/user/Library/CloudStorage/Dropbox/Apps/Overleaf/"
        "Dark baryons direct detection/figs")

# ---------- data ----------
def load(scheme, system, al):
    base = f'prod_a{al}/{scheme}_{system}'
    d = base + '_long' if os.path.exists(base + '_long/prod_ME.csv') else base
    d = base + '_hi'   if os.path.exists(base + '_hi/prod_ME.csv')   else d
    fn = d + '/prod_ME.csv'
    if not os.path.exists(fn): return None
    D = {}
    for r in csv.DictReader(open(fn)):
        if r['channel'] == 'C3_N':
            D[float(r['Q'])] = (float(r['F']), float(r['F_err']))
    return D

# EXACT benchmarks
def coul_meson_exact(q2, al):        # analytic hydrogen dipole, a_B = 1.5/al
    aB = 1.5/al; return 1.0/(1.0 + q2*(aB/4)**2)**2
NUM_LO = {0.2:[.9975,.9845,.9402,.7896,.5219,.3137], 0.3:[.9982,.9889,.9566,.8422,.6154,.4102],
          0.5:[.9982,.9889,.9566,.8423,.6158,.4112]}
NUM_NLO= {0.2:[.9977,.9859,.9455,.8062,.5499,.3412], 0.3:[.9983,.9891,.9576,.8454,.6217,.4174],
          0.5:[.9982,.9890,.9571,.8439,.6188,.4146]}
QNUM = [0.02,0.05,0.1,0.2,0.35,0.5]
def svm_baryon_exact(al):
    fn = f'{OUT}/svm_baryon_a{al}.csv'
    if not os.path.exists(fn): return None, None
    Q=[]; F=[]
    for r in csv.reader(open(fn)):
        if r[0].startswith('#') or r[0]=='Q': continue
        Q.append(float(r[0])); F.append(float(r[1]))
    return np.array(Q), np.array(F)

def exact_curve(scheme, system, al):
    """return (Q_arr, F_arr) exact, or (None,None)."""
    if system=='meson' and scheme=='coul':
        q=np.linspace(0,0.5,200); return q, coul_meson_exact(q*q, al)
    if system=='meson' and scheme=='mrs_lo':  return np.array(QNUM), np.array(NUM_LO[al])
    if system=='meson' and scheme=='mrs_nlo': return np.array(QNUM), np.array(NUM_NLO[al])
    if system=='baryon' and scheme=='coul':   return svm_baryon_exact(al)
    return None, None

# ============ FIGURE 1: form factors ============
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey=True)
schemes = ['coul','mrs_lo','mrs_nlo']; systems=['meson','baryon']
for i, system in enumerate(systems):
    for j, scheme in enumerate(schemes):
        ax = axes[i][j]
        for al in ALPHAS:
            D = load(scheme, system, al)
            if D is None: continue
            F0 = D[0.0][0]
            Q = np.array(sorted(q for q in D if q>0)); q2 = Q*Q
            fr = np.array([D[q][0]/F0 for q in Q]); fe = np.array([max(D[q][1]/F0,1e-4) for q in Q])
            # fit for the smooth curve
            try:
                if system=='meson':
                    pb,_=curve_fit(lambda x,b: gendip(x,b,2.0), q2, fr, p0=[6], sigma=fe, absolute_sigma=True)
                    pb=[pb[0],2.0]
                else:
                    pb,_=curve_fit(gendip, q2, fr, p0=[6,3], sigma=fe, absolute_sigma=True,
                                   bounds=([.5,1],[40,12]), maxfev=8000)
            except Exception: pb=None
            c = ARAMP[al]
            if pb is not None:
                xx=np.linspace(0,0.25,200); ax.plot(xx, gendip(xx,*pb), '-', color=c, lw=1.1, zorder=3)
            ax.errorbar(q2, fr, yerr=fe, fmt=AMARK[al], color=c, ms=3.4, lw=0, elinewidth=0.8,
                        capsize=1.4, mec='white', mew=0.4, zorder=4,
                        label=rf'$\alpha_d={al}$')
        # exact overlay
        eQ,eF = exact_curve(scheme, system, ALPHAS[1])  # representative alpha=0.3
        for al in ALPHAS:
            eQ,eF = exact_curve(scheme, system, al)
            if eQ is not None:
                ax.plot(eQ*eQ, eF, ls=(0,(4,2)), color='0.35', lw=0.8, zorder=2)
        ax.set_xlim(0,0.255); ax.set_ylim(0,1.03)
        ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
        if i==0: ax.set_title(SLAB[scheme], fontsize=9)
        if i==1: ax.set_xlabel(r'$q^2\ [m_\psi^2]$')
        # plotted quantity IS the GFMC matrix element (F_1 is derived from it)
        if j==0: ax.set_ylabel(r'$\langle J_{V^0}(q)\rangle\,/\,\langle J_{V^0}(0)\rangle$')
        if i==0 and j==0:
            ax.legend(loc='lower left', handletextpad=0.4, labelspacing=0.25)
            ax.plot([],[],ls=(0,(4,2)),color='0.35',lw=0.8,label='exact')
            ax.legend(loc='lower left', handletextpad=0.4, labelspacing=0.25)
        # row label
        if j==2:
            ax.text(0.97,0.92, 'dark '+('meson' if system=='meson' else 'baryon'),
                    transform=ax.transAxes, ha='right', va='top', fontsize=8.5, style='italic')
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
fig.savefig(f'{OUT}/form_factors.pdf'); fig.savefig(f'{OUT}/form_factors.png', dpi=150)
print("wrote form_factors.pdf")

# ============ FIGURE 2: radius vs alpha ============
# read fitted radii
FR = {}
for r in csv.DictReader(open(f'{OUT}/final_forms.csv')):
    FR[(r['scheme'],r['system'],float(r['alpha']))] = (float(r['r2']), float(r['r2_err']))
fig2, axes2 = plt.subplots(1, 2, figsize=(7.0, 2.9))
for ax, system, N in zip(axes2, ['meson','baryon'], [2,3]):
    for scheme in schemes:
        xs=[]; ys=[]; es=[]
        for al in ALPHAS:
            v=FR.get((scheme,system,al))
            if v: xs.append(al); ys.append(v[0]); es.append(v[1])
        if not xs: continue
        ax.errorbar(xs, ys, yerr=es, fmt='o-', color=SCHEME[scheme], ms=4.5, lw=1.1,
                    capsize=2, mec='white', mew=0.5, label=SLAB[scheme], zorder=4)
    # exact references
    aa=np.linspace(0.18,0.55,100)
    if system=='meson':
        ax.plot(aa, 3*(1.5/aa)**2/4, ls=(0,(4,2)), color=SCHEME['coul'], lw=0.9, zorder=2)
        for al in ALPHAS:  # numerov NLO radius points
            pass
    else:
        # SVM exact points
        svm={0.2:95.9,0.3:42.8,0.5:15.3}
        ax.plot(list(svm), list(svm.values()), 'D', mfc='none', mec=SCHEME['coul'],
                ms=6, mew=1.0, zorder=5)
        ax.plot(aa, 42.8*(0.3/aa)**2, ls=(0,(4,2)), color=SCHEME['coul'], lw=0.9, zorder=2)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.18,0.56); ax.set_xticks(ALPHAS); ax.set_xticklabels(['0.2','0.3','0.5'])
    ax.set_xlabel(r'$\alpha_d$'); ax.set_ylabel(r'$\langle r^2\rangle\ [m_\psi^{-2}]$')
    ax.set_title('dark '+system, fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.legend(loc='upper right' if system=='meson' else 'lower left',
              handletextpad=0.4, labelspacing=0.3)
axes2[0].text(0.5,0.06,r'Coulomb $\propto\alpha_d^{-2}$; MRS saturates',
              transform=axes2[0].transAxes, fontsize=7.5, color='0.35')
fig2.tight_layout(pad=0.4, w_pad=1.0)
fig2.savefig(f'{OUT}/radius_vs_alpha.pdf'); fig2.savefig(f'{OUT}/radius_vs_alpha.png', dpi=150)
print("wrote radius_vs_alpha.pdf")

# mirror into overleaf
if os.path.isdir(OVER):
    import shutil
    for f in ['form_factors.pdf','radius_vs_alpha.pdf']:
        shutil.copy(f'{OUT}/{f}', os.path.join(OVER, f))
    print("copied figures into Overleaf figs/")
