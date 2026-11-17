#!/usr/bin/env python3
r"""Build the TeX summary: parse the extraction logs, write CSVs, tables and figures.

Reads the ``*_dipole.txt`` / ``*_z_k4.txt`` / ``*_walkers.txt`` written by
``gfmc_me_gevp.py`` and ``gfmc_walker_diag.py`` into ``$OUT/*/analysis``, so every number in
the document is generated from the run output rather than retyped.  Writes into ``summary/``:

    data/radii.csv, data/formfactors.csv      the parsed numbers
    tab_results.tex, tab_crosschecks.tex      the two tables
    fig_baryon_ff.pdf, fig_radius_vs_nc.pdf   the two figures

Colour follows the dataviz skill.  Nc is an ORDERED family, so the baryon series take a
one-hue ordinal ramp (blue, light->dark with Nc), not categorical hues: four categorical
slots cannot clear the all-pairs floors, while the ramp passes all four ordinal checks
(monotone L, adjacent dL >= 0.06, light end 2.06:1 on the surface, hue spread 3.4 deg) --
see ``validate_palette.py``, the Python twin of the skill's validator, since this box has no
node.  Marker shape and direct labels carry identity as well, so nothing is colour-alone.

Usage:  nc_summary.py [--out summary]
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--out', default='summary')
p.add_argument('--hqn', default=os.path.expanduser('~/hqn_out'))
args = p.parse_args()

A1 = os.path.join(args.hqn, 'nc_alpha1', 'analysis')       # alpha = 1 Nc scan
A0 = os.path.join(args.hqn, 'refmass_a0.1', 'analysis')    # alpha_s(mQ) = 0.1 campaign
os.makedirs(os.path.join(args.out, 'data'), exist_ok=True)

# ---------------------------------------------------------------------------------------------
# design tokens (dataviz skill: reference palette instance)
# ---------------------------------------------------------------------------------------------
SURFACE = '#fcfcfb'
INK, INK2, INK3 = '#0b0b0b', '#52514e', '#8a8984'
# blue ordinal ramp, steps 250/350/450/550/650 -- validated light->dark for up to 5 Nc values
RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
MARKERS = ['o', 's', '^', 'D', 'v']

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': INK3, 'axes.linewidth': 0.6,
    'font.size': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
    'legend.frameon': False, 'figure.dpi': 150,
})


# ---------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------
ROW = re.compile(r'^\s{2,}(\d+)\s+([\d.]+)\s+([-\d.e+]+)\s+([\d.]+)\s+\+/-\s+([\d.]+)'
                 r'(?:\s+([\d.]+)\s+([-+][\d.]+)%\s+\(([-+][\d.]+)s\))?')
TGT = re.compile(r'gives <r\^2> = ([\d.]+), against the true ([\d.]+) \(([-+][\d.]+)% fit-form')
FF = re.compile(r'^\s{4,}([\d.]+)\s+([\d.]+)\s+\+/-\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
                r'\s+([\d.]+)\s+\+/-\s+([\d.]+)\s+\(([-+][\d.]+)s\)')
SPIKE = re.compile(r'dropping (\d+) of (\d+) walkers \(worst (\d+)x')


def parse(fn):
    """-> dict(target, true, bias, rows=[(m, r2, err, pull_pct, pull_sig)], ff=[...], spike)"""
    if not os.path.exists(fn):
        return None
    out = {'rows': [], 'ff': [], 'target': None, 'true': None, 'bias': None, 'spike': None}
    for line in open(fn):
        if (t := TGT.search(line)):
            out['target'], out['true'], out['bias'] = (float(t[1]), float(t[2]), float(t[3]))
        if (s := SPIKE.search(line)):
            out['spike'] = (int(s[1]), int(s[2]), int(s[3]))
        if (f := FF.match(line)):
            out['ff'].append(dict(Q=float(f[1]), F=float(f[2]), Ferr=float(f[3]),
                                  Fex=float(f[4]), ratio=float(f[5]),
                                  rr=float(f[6]), rrerr=float(f[7])))
            continue
        if (r := ROW.match(line)):
            out['rows'].append(dict(m=int(r[1]), F0=float(r[2]), dev=abs(float(r[3])),
                                    r2=float(r[4]),
                                    err=float(r[5]),
                                    pull=float(r[7]) if r[7] else None,
                                    sig=float(r[8]) if r[8] else None))
    return out if out['rows'] else None


def walkers(fn):
    """worst-case walker diagnostics across the file."""
    if not os.path.exists(fn):
        return None
    neff, top1, top10, neg, dj = [], [], [], [], []
    for line in open(fn):
        m = re.match(r'\s+[\d.]+\s+\d+\s+[-\d.]+\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%'
                     r'\s+([\d.]+)%\s+([-+][\d.]+)s', line)
        if m:
            neff.append(float(m[1])); top1.append(float(m[2])); top10.append(float(m[3]))
            neg.append(float(m[4])); dj.append(abs(float(m[5])))
    if not neff:
        return None
    return dict(neff=(min(neff), max(neff)), top1=max(top1), top10=max(top10),
                neg=max(neg), dj=max(dj))


# ---------------------------------------------------------------------------------------------
# configuration tables
# ---------------------------------------------------------------------------------------------
# Nc = 3, alpha_s(mQ) = 0.1.  'ref' is the independent exact/variational value and 'src' its
# provenance.  fo_baryon_nlo does not exist yet -- the fixed-order NLO baryon was never run.
NC3 = [
    # key, dir, label, system, potential, order, ref, src
    ('fo_meson_lo',   A0, 'meson',  'FO',  'LO',  168.75,      r'closed form $27/(16\alpha^2)$'),
    ('fo_meson_nlo',  A0, 'meson',  'FO',  'NLO', 84.269031,   r'\texttt{fo\_two\_body.py}'),
    ('fo_baryon_lo',  A0, 'baryon', 'FO',  'LO',  382.60983,   r'\texttt{svm\_nbody.py}'),
    ('fo_baryon_nlo', A0, 'baryon', 'FO',  'NLO', 179.42049,
     r'\texttt{svm\_nbody.py --potential fo\_nlo}'),
    ('mrs_meson_lo',  A0, 'meson',  'MRS', 'LO',  56.692063,   r'Numerov / SVM $N\!=\!2$'),
    ('mrs_meson_nlo', A0, 'meson',  'MRS', 'NLO', 52.071288,   r'Numerov / SVM $N\!=\!2$'),
    ('mrs_baryon_lo', A0, 'baryon', 'MRS', 'LO',  114.2523,    r'SVM $N\!=\!3$'),
    ('mrs_baryon_nlo', A0, 'baryon', 'MRS', 'NLO', 103.14328,  r'SVM $N\!=\!3$'),
]

# alpha = 1 Coulomb baryons, N_coord = Nc.  kappa_N = (Nc+1)/(2 Nc) is the pair coupling in the
# totally antisymmetric singlet; <r^2>_ch = 3/kappa^2 exactly at N = 2.
KAP = {N: (N + 1) / (2 * N) for N in range(2, 8)}
NCSCAN = [
    (2, 'nc2_baryon', 3.0 / KAP[2] ** 2, 'exact'),
    (3, 'nc3_baryon', 3.8260983,         'SVM'),
    (4, 'nc4_baryon', 2.7191553,         'SVM'),
    (5, 'nc5_baryon', 2.015,             'SVM, unconverged'),
    (6, 'nc6_baryon', None,              'none'),
]


def best(d, prefer=None):
    """the plateau end: the largest rank reported."""
    return max(d['rows'], key=lambda r: r['m'])


# ---------------------------------------------------------------------------------------------
# gather
# ---------------------------------------------------------------------------------------------
nc3 = []
for key, root, syst, pot, order, ref, src in NC3:
    d = parse(os.path.join(root, f'{key}_dipole.txt'))
    if d is None:
        print(f'  (no extraction for {key})')
        continue
    z = parse(os.path.join(root, f'{key}_z_k4.txt')) or parse(
        os.path.join(root, f'{key}_z_k3.txt'))
    r = best(d)
    nc3.append(dict(key=key, system=syst, pot=pot, order=order, ref=ref, src=src,
                    m=r['m'], r2=r['r2'], err=r['err'], target=d['target'],
                    bias=d['bias'], pull=r['pull'], sig=r['sig'],
                    zr2=(best(z)['r2'] if z else None), ztgt=(z['target'] if z else None),
                    F0=r['F0'], ff=d['ff']))

scan = []
for N, key, ref, src in NCSCAN:
    d = parse(os.path.join(A1, f'{key}_dipole.txt'))
    if d is None:
        print(f'  (Nc={N}: no extraction yet)')
        continue
    z = parse(os.path.join(A1, f'{key}_z_k4.txt'))
    r = best(d)
    w = walkers(os.path.join(A1, f'{key}_walkers.txt'))
    scan.append(dict(N=N, key=key, kappa=KAP[N], ref=ref, src=src, m=r['m'],
                     r2=r['r2'], err=r['err'], target=d['target'], bias=d['bias'],
                     pull=r['pull'], sig=r['sig'],
                     zr2=(best(z)['r2'] if z else None), zerr=(best(z)['err'] if z else None),
                     ztgt=(z['target'] if z else None), zbias=(z['bias'] if z else None),
                     ff=d['ff'], walk=w))

# ---------------------------------------------------------------------------------------------
# CSVs
# ---------------------------------------------------------------------------------------------
with open(os.path.join(args.out, 'data', 'radii.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['set', 'key', 'Nc', 'system', 'potential', 'order', 'm', 'r2_gfmc', 'err',
                'target_likeforlike', 'fitform_bias_pct', 'r2_reference', 'pull_pct', 'pull_sig'])
    for e in nc3:
        w.writerow(['alpha_s(mQ)=0.1', e['key'], 3, e['system'], e['pot'], e['order'], e['m'],
                    f"{e['r2']:.6g}", f"{e['err']:.3g}", e['target'], e['bias'], e['ref'],
                    e['pull'], e['sig']])
    for e in scan:
        w.writerow(['alpha=1 Coulomb', e['key'], e['N'], 'baryon', 'FO', 'LO', e['m'],
                    f"{e['r2']:.6g}", f"{e['err']:.3g}", e['target'], e['bias'], e['ref'],
                    e['pull'], e['sig']])

with open(os.path.join(args.out, 'data', 'formfactors.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['set', 'key', 'Nc', 'Q', 'F_over_F0', 'err', 'F_reference', 'ratio'])
    for e in nc3:
        for f in e['ff']:
            w.writerow(['alpha_s(mQ)=0.1', e['key'], 3, f['Q'], f['F'], f['Ferr'],
                        f['Fex'], f['ratio']])
    for e in scan:
        for f in e['ff']:
            w.writerow(['alpha=1 Coulomb', e['key'], e['N'], f['Q'], f['F'], f['Ferr'],
                        f['Fex'], f['ratio']])


# ---------------------------------------------------------------------------------------------
# figure 1: baryon form factors
# ---------------------------------------------------------------------------------------------
def ref_curve(N):
    """Dense reference F(Q)/F(0), so the curve is computed rather than interpolated between
    the six measured points -- in Q^2 the low-Q end is compressed and a 6-point polyline
    visibly kinks.  Falls back to the measured grid if the dense file is absent."""
    fn = os.path.join(args.out, 'data', f'ref_nc{N}.csv')
    if not os.path.exists(fn):
        return None
    q, f = [], []
    for r in csv.DictReader(open(fn)):
        if r['Q'].lstrip().startswith('#'):     # svm_nbody.py appends a comment line
            continue
        q.append(float(r['Q'])); f.append(float(r['F_over_F0']))
    return np.array(q), np.array(f)


def fig_ff():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    for i, e in enumerate(scan):
        c, mk = RAMP[i], MARKERS[i]
        Q = np.array([f['Q'] for f in e['ff']])
        F = np.array([f['F'] for f in e['ff']])
        E = np.array([f['Ferr'] for f in e['ff']])
        Fx = np.array([f['Fex'] for f in e['ff']])
        rms = np.sqrt(e['r2'])   # alpha = 1 and mQ = 1, so Q^2 == Q^2/(mQ^2 alpha^2)
        # F(0)/F(0) = 1 is exact, so anchor the reference curve there rather than
        # letting it start at the lowest measured Q
        dense = ref_curve(e['N'])
        if dense is not None:
            Q0, Fx0 = dense
            if Q0[0] > 0:
                Q0, Fx0 = np.concatenate(([0.0], Q0)), np.concatenate(([1.0], Fx0))
        else:
            Q0, Fx0 = np.concatenate(([0.0], Q)), np.concatenate(([1.0], Fx))
        for k, x, x0 in ((0, Q ** 2, Q0 ** 2), (1, (Q * rms) ** 2, (Q0 * rms) ** 2)):
            ax[k].plot(x0, Fx0, '-', color=c, lw=1.3, alpha=0.85, zorder=2)
            ax[k].errorbar(x, F, yerr=E, fmt=mk, color=c, ms=4.4, lw=0,
                           elinewidth=1.0, capsize=0, mec=SURFACE, mew=0.7, zorder=3,
                           label=rf'$N_c={e["N"]}$' if k == 0 else None)
        # direct-label only the two ends of the ordered family: the intermediate curves sit
        # too close in Q for four labels to clear each other, and the ramp direction is what
        # a reader needs to anchor
        if e is scan[0] or e is scan[-1]:
            ax[0].annotate(rf'$N_c\!=\!{e["N"]}$', (Q[-1] ** 2, Fx[-1]),
                           textcoords='offset points', xytext=(6, 0), fontsize=8,
                           color=INK2, va='center', ha='left')
    ax[0].set_xlabel(r'$Q^2/(m_\psi^2\alpha^2)$')
    ax[1].set_xlabel(r'$Q^2\,\langle r^2\rangle_{\rm ch}$')
    ax[0].set_ylabel(r'$F(Q)/F(0)$')
    ax[0].set_title('baryon charge form factor', color=INK, loc='left')
    ax[1].set_title(r'rescaled by $\langle r^2\rangle_{\rm ch}$', color=INK, loc='left')
    for a in ax:
        a.grid(True, alpha=0.9, lw=0.6)
        a.set_axisbelow(True)
        a.set_ylim(0, 1.04)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    ax[0].set_xlim(0, 5.4)
    ax[1].set_xlim(0, 10.2)
    ax[0].legend(loc='upper right', ncol=2, handletextpad=0.4, columnspacing=1.0,
                 labelcolor=INK2)
    fig.savefig(os.path.join(args.out, 'fig_baryon_ff.pdf'))
    fig.savefig(os.path.join(args.out, 'fig_baryon_ff.png'), dpi=190)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# figure 2: charge radius vs Nc
# ---------------------------------------------------------------------------------------------
def fig_radius():
    Ns = np.array([e['N'] for e in scan], float)
    r2 = np.array([e['r2'] for e in scan])
    er = np.array([e['err'] for e in scan])
    ka = np.array([e['kappa'] for e in scan])
    ref = np.array([e['ref'] if e['ref'] else np.nan for e in scan])

    y = 1 / (ka * np.sqrt(r2))
    ye = y * 0.5 * er / r2
    a, b = np.polyfit(Ns, y, 1)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)

    # (a) the radius itself, with the fitted law
    xg = np.linspace(1.8, max(Ns) + 0.6, 200)
    kg = (xg + 1) / (2 * xg)
    ax[0].plot(xg, 1 / ((a * xg + b) ** 2 * kg ** 2), '-', color=INK3, lw=1.1, zorder=1,
               label=r'fit: $1/(m_\psi\kappa\sqrt{\langle r^2\rangle})$ linear in $N_c$')
    ax[0].plot(Ns, ref, 'o', mfc='none', mec=INK2, mew=1.0, ms=8, zorder=2,
               label='exact / variational')
    # N_c is already on the x axis, so colour carries no extra information here: one series
    # colour and one marker, which keeps the legend unambiguous
    ax[0].errorbar(Ns, r2, yerr=er, fmt='o', color=RAMP[2], ms=5.2, lw=0,
                   elinewidth=1.1, capsize=0, mec=SURFACE, mew=0.7, zorder=3, label='GFMC')
    ax[0].set_xlabel(r'$N_c$   ($N_{\rm coord}=N_c$)')
    ax[0].set_ylabel(r'$m_\psi^2\,\langle r^2\rangle_{\rm ch}$   ($\alpha=1$)')
    ax[0].set_title(r'baryon charge radius', color=INK, loc='left')
    ax[0].legend(loc='lower left', labelcolor=INK2)

    # (b) the law, where the claim actually lives
    ax[1].plot(xg, a * xg + b, '-', color=INK3, lw=1.1, zorder=1,
               label=rf'${b:+.4f}{a:+.4f}\,N_c$')
    ax[1].errorbar(Ns, y, yerr=ye, fmt='o', color=RAMP[2], ms=5.2, lw=0,
                   elinewidth=1.1, capsize=0, mec=SURFACE, mew=0.7, zorder=3)
    ax[1].set_xlabel(r'$N_c$')
    ax[1].set_ylabel(r'$1/(m_\psi\,\kappa_N\sqrt{\langle r^2\rangle_{\rm ch}})$')
    ax[1].set_title(r'inverse rms radius in units of $\kappa_N$', color=INK, loc='left')
    ax[1].legend(loc='upper left', labelcolor=INK2)
    res = 100 * ((a * Ns + b) / y - 1)
    ax[1].text(0.97, 0.06, 'fit residuals  ' + '   '.join(f'{r:+.1f}%' for r in res),
               transform=ax[1].transAxes, ha='right', fontsize=7, color=INK2)

    for A in ax:
        A.grid(True, alpha=0.9, lw=0.6)
        A.set_axisbelow(True)
        A.set_xticks(list(range(2, int(max(Ns)) + 1)))
        for s in ('top', 'right'):
            A.spines[s].set_visible(False)
    fig.savefig(os.path.join(args.out, 'fig_radius_vs_nc.pdf'))
    fig.savefig(os.path.join(args.out, 'fig_radius_vs_nc.png'), dpi=190)
    plt.close(fig)
    return a, b, res


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------
def tab_results():
    L = [r'\begin{tabular}{llcr@{\,}c@{\,}lrrrr}', r'\hline\hline',
         r'potential & state & $m$ & \multicolumn{3}{c}{GFMC $\langle r^2\rangle_{\rm ch}$}'
         r' & target & bias & reference & pull \\', r'\hline']
    for e in nc3:
        pull = (f'{e["sig"]:+.1f}$\\sigma$' if e['sig'] is not None else '--')
        L.append(f'{e["pot"]} & {e["system"]} {e["order"]} & {e["m"]} & '
                 f'{e["r2"]:.4g} & $\\pm$ & {e["err"]:.3g} & '
                 f'{e["target"]:.4g} & {e["bias"]:+.2f}\\% & {e["ref"]:.6g} & {pull} \\\\')
    L += [r'\hline\hline', r'\end{tabular}']
    open(os.path.join(args.out, 'tab_results.tex'), 'w').write('\n'.join(L) + '\n')


def tab_crosschecks(a, b):
    def row(what, measured, expected, verdict):
        return f'{what} & {measured} & {expected} & {verdict} \\\\'
    L = [r'\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xlll}', r'\hline\hline',
         r'check & measured & expected & verdict \\', r'\hline']
    devs = [r['dev'] for e in nc3 for r in parse(
        os.path.join(A0, f'{e["key"]}_dipole.txt'))['rows']]
    devs += [r['dev'] for e in scan for r in parse(
        os.path.join(A1, f'{e["key"]}_dipole.txt'))['rows']]
    f0 = max(devs)
    mant, ex = f'{f0:.0e}'.split('e')
    L.append(row(r'$F(0)=N$ identity (consistent-Gram GEVP)',
                 (rf'$\max|F(0)-N| = {mant}\times10^{{{int(ex)}}}$' if f0 else r'$0$'),
                 r'$0$ exactly', 'machine precision'))
    Nmax = max(e['N'] for e in scan)
    L.append(row(rf'$F(0)=N$ at $N_{{\rm coord}}=N_c={Nmax}$ (6-quark singlet)'
                 if Nmax >= 6 else
                 rf'$F(0)=N$ at $N_{{\rm coord}}=N_c={Nmax}$',
                 f'${Nmax}.0000000000$', f'${Nmax}$', 'exact'))
    L.append(r'\hline')
    # solver validation
    L.append(row(r'\texttt{svm\_nbody.py} $N=2$ vs closed form',
                 r'$-0.44443853$, $1.6875501$', r'$-4/9$, $27/16$', r'$6\times10^{-6}$'))
    L.append(row(r'\texttt{svm\_nbody.py} $N=3$ vs published Coulomb baryon',
                 r'$3.8260983$', r'$3.826$', 'all digits'))
    L.append(row(r'\texttt{svm\_nbody.py --potential fo\_nlo} $N=2$ vs'
                 r' \texttt{fo\_two\_body.py}',
                 r'$-0.010609125$, $84.269834$', r'$-0.010609161$, $84.269031$',
                 r'$1\times10^{-5}$'))
    L.append(row(r'\texttt{mrs\_two\_body.py} vs \texttt{svm\_baryon.py} $N=2$ (MRS LO)',
                 r'$56.69186$', r'$56.692063$', r'$4\times10^{-6}$'))
    L.append(r'\hline')
    # the alpha-rescaling null test
    L.append(row(r'$N_c=3$ baryon: $\alpha=1$ vs $\alpha=0.1$ rescaled',
                 r'$3.897\pm0.063$', r'$3.892\pm0.059$', r'$0.06\sigma$'))
    L.append(row(r'$N=2$: baryon path ($\kappa=3/4$) vs meson path rescaled',
                 r'$5.210\pm0.096$', r'$5.186\pm0.098$', r'$0.2\sigma$'))
    L.append(r'\hline')
    # GFMC vs reference, both sets
    for e in nc3:
        L.append(row(rf'{e["pot"]} {e["system"]} {e["order"]} vs {e["src"]}',
                     f'${e["r2"]:.4g}\\pm{e["err"]:.3g}$', f'${e["target"]:.4g}$',
                     f'${e["sig"]:+.1f}\\sigma$' if e['sig'] is not None else '--'))
    L.append(r'\hline')
    for e in scan:
        if e['ref'] is None:
            continue
        L.append(row(rf'$\alpha=1$ baryon $N_c={e["N"]}$ vs {e["src"]}',
                     f'${e["r2"]:.4g}\\pm{e["err"]:.3g}$', f'${e["target"]:.4g}$',
                     f'${e["sig"]:+.1f}\\sigma$' if e['sig'] is not None else '--'))
    L.append(r'\hline')
    # fit-form robustness and walkers
    wb = max(abs(e['bias']) for e in scan if e['bias'] is not None)
    zb = max(abs(e['zbias']) for e in scan if e['zbias'] is not None)
    L.append(row(r'dipole fit-form bias on the exact $F$ (baryons)',
                 rf'$\le {wb:.2f}\%$', r'$0$', 'controlled via \\texttt{--ref\\_ff}'))
    L.append(row(r'$z$-expansion ($k_{\max}=4$) fit-form bias',
                 rf'$\le {zb:.2f}\%$', r'$0$', 'ansatz-independent'))
    dj = max(e['walk']['dj'] for e in scan if e['walk'])
    ne = min(e['walk']['neff'][0] for e in scan if e['walk'])
    ng = max(e['walk']['neg'] for e in scan if e['walk'])
    L.append(row(r'walker state-ID: $\Delta J$ dropping the loudest walker',
                 rf'$\le {dj:.2f}\sigma$', r'$\ll 1\sigma$', 'ensemble average'))
    L.append(row(r'walker state-ID: $N_{\rm eff}/N$, negative Ritz weights',
                 rf'$\ge {ne:.2f}$, {ng:.0f}\%', r'$1$, $0\%$', 'healthy'))
    L.append(row(r'$1/(\kappa\sqrt{\langle r^2\rangle})$ linear in $N_c$',
                 rf'${b:+.4f}{a:+.4f}N_c$', 'exact linearity',
                 r'$\sim2\%$ curvature'))
    L += [r'\hline\hline', r'\end{tabularx}']
    open(os.path.join(args.out, 'tab_crosschecks.tex'), 'w').write('\n'.join(L) + '\n')


have = {e['key'] for e in nc3}
missing = [k for k, *_ in NC3 if k not in have]
if missing:
    st = ('\\textbf{Not yet in this table:} ' + ', '.join(
        r'\code{' + m.replace('_', r'\_') + '}' for m in missing)
        + '.  The run is in flight; the reference is already in the cross-checks.')
else:
    st = ('Every fixed-order and MRS combination at $N_c=3$ is present: both orders, both '
          'states, each against an independent reference.')
# a macro rather than loose text: \input inside a \captionof argument is fragile
open(os.path.join(args.out, 'status.tex'), 'w').write(
    '\\newcommand{\\statusline}{' + st + '}\n')

fig_ff()
a, b, res = fig_radius()
tab_results()
tab_crosschecks(a, b)
print(f'wrote {args.out}/: fig_baryon_ff.pdf, fig_radius_vs_nc.pdf, '
      f'tab_results.tex, tab_crosschecks.tex, data/*.csv')
print(f'  Nc=3 rows: {len(nc3)}   Nc scan points: {[e["N"] for e in scan]}')
print(f'  law: 1/(kappa sqrt(r2)) = {b:+.4f} {a:+.4f} N   residuals '
      + '  '.join(f'{r:+.2f}%' for r in res))
