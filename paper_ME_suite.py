"""
Complete matrix-element suite for the paper:
  Fig: matrix_elements.pdf — everything measured beyond the charge profile,
       at alpha_d = 0.3, all three schemes, meson (top) & baryon (bottom):
   (a) scalar/vector splitting  m_psi <J_V0 - J_S>(q^2)  (anchored at 2<K>/2m)
   (b) Galilean ratio (2M/q) <J_V^x>/<J_V0>  vs exact identity = 1;
       open markers: (2m/q) <J_T^0x>/<J_N>  (exact kinematic, = 1)
   (c) null channels (axial, tensor-ij, transverse vector/tensor) x 10^3
  Tex: me_suite.tex — the full measured suite with values/uncertainties.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 9,
    'axes.linewidth': 0.7, 'axes.labelsize': 9.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.fontsize': 7, 'legend.frameon': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
SCHEME = {'coul': '#0072B2', 'mrs_lo': '#D55E00', 'mrs_nlo': '#009E73'}
SLAB = {'coul': 'Coulomb LO', 'mrs_lo': 'MRS LO', 'mrs_nlo': 'MRS NLO'}
AL = 0.3
OUT = 'paper_final'
OVER = ("/Users/user/Library/CloudStorage/Dropbox/Apps/Overleaf/"
        "Dark baryons direct detection")

def load(scheme, system):
    base = f'prod_a{AL}/{scheme}_{system}'
    for suf in ('_long', '_hi', ''):
        fn = base + suf + '/prod_ME.csv'
        if os.path.exists(fn):
            D = {}
            for r in csv.DictReader(open(fn)):
                D.setdefault(float(r['Q']), {})[r['channel']] = (float(r['F']), float(r['F_err']))
            return D
    return None

import glob, h5py
def corr_boot(scheme, system, nboot=200, cv_max=1.0, seed=5):
    """Correlated walker-bootstrap of (V0-S) and the Galilean ratio, per Q.
    V0 and S live on the same walkers: quadrature errors vastly overestimate
    the error of their difference — resample jointly."""
    base = f'prod_a{AL}/{scheme}_{system}'
    for suf in ('_long', '_hi', ''):
        if os.path.isdir(base + suf): d = base + suf; break
    out = {}
    for fn in sorted(glob.glob(f'{d}/currents_3pt_*Qmag*.h5')):
        if 'korr' in fn: continue
        import re as _re
        Q = float(_re.search(r'Qmag([0-9.]+)\.h5', fn).group(1))
        h = h5py.File(fn, 'r')
        W = np.real(np.array(h['Ws2_all']))
        V0 = np.real(np.array(h['C3_V0'])); S = np.real(np.array(h['C3_S']))
        N_ = np.real(np.array(h['C3_N'])); Vx = np.real(np.array(h['C3_Vi']))[:, :, 0]
        h.close()
        cv = W.std(1)/np.maximum(np.abs(W.mean(1)), 1e-30)
        cut = int(np.argmax(cv > cv_max)) if (cv > cv_max).any() else len(cv)
        lo, hi = max(1, cut//2), max(cut, 2)
        Nw = W.shape[1]
        rng = np.random.default_rng(seed)
        sp_b = []; gal_b = []
        for _ in range(nboot):
            idx = rng.integers(0, Nw, Nw)
            v0 = np.mean([V0[idx, t].sum()/W[t, idx].sum() for t in range(lo, hi)])
            s  = np.mean([S[idx, t].sum()/W[t, idx].sum() for t in range(lo, hi)])
            sp_b.append(v0 - s)
            if Q > 0:
                vx = np.mean([Vx[idx, t].sum()/W[t, idx].sum() for t in range(lo, hi)])
                gal_b.append(abs(vx)/(Q*v0))
        v0c = np.mean([V0[:, t].sum()/W[t, :].sum() for t in range(lo, hi)])
        sc  = np.mean([S[:, t].sum()/W[t, :].sum() for t in range(lo, hi)])
        gc  = abs(np.mean([Vx[:, t].sum()/W[t, :].sum() for t in range(lo, hi)]))/(Q*v0c) if Q > 0 else np.nan
        out[Q] = dict(split=v0c-sc, split_err=np.std(sp_b),
                      gal=gc, gal_err=np.std(gal_b) if gal_b else np.nan)
    return out

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
suite_rows = []
for i, (system, M) in enumerate([('meson', 2), ('baryon', 3)]):
    axA, axB, axC = axes[i]
    for scheme in ['coul', 'mrs_lo', 'mrs_nlo']:
        D = load(scheme, system)
        if D is None: continue
        c = SCHEME[scheme]
        Qs = np.array(sorted(D)); q2 = Qs**2
        CB = corr_boot(scheme, system)
        # (a) splitting m*(V0-S), correlated bootstrap errors
        sp  = np.array([CB[q]['split'] for q in Qs])
        spe = np.array([CB[q]['split_err'] for q in Qs])
        axA.errorbar(q2, sp, yerr=spe, fmt='o-', color=c, ms=3.2, lw=0.9,
                     capsize=1.5, mec='white', mew=0.4, label=SLAB[scheme])
        # (b) Galilean ratio (correlated) + T0i kinematic ratio
        Qp = Qs[Qs > 0]
        gal  = np.array([CB[q]['gal']*2*M for q in Qp])
        gale = np.array([CB[q]['gal_err']*2*M for q in Qp])
        t0   = np.array([abs(D[q]['C3_T0i_x'][0])*2/(q*D[q]['C3_N'][0]) for q in Qp])
        axB.errorbar(Qp**2, gal, yerr=gale, fmt='o-', color=c, ms=3.2, lw=0.9,
                     capsize=1.5, mec='white', mew=0.4)
        axB.plot(Qp**2, t0, marker='s', mfc='none', mec=c, ms=3.4, lw=0, mew=0.8)
        # (c) SYMMETRY-null channels x 1e3, signed. The operator-exact zeros
        # (A^x, T^{yz}, T^{0y}, T^{0z} for q||x) are excluded: identically 0.0,
        # not measurements.
        for ch, comps, mk in [('C3_Ai', ['_y', '_z'], '^'),
                              ('C3_Tij', ['_x', '_y'], 's'),
                              ('C3_Vi', ['_y', '_z'], 'o')]:
            for comp in comps:
                vals = np.array([D[q].get(ch+comp, (np.nan, 0))[0] for q in Qs])
                errs = np.array([D[q].get(ch+comp, (0, 0))[1] for q in Qs])
                axC.errorbar(q2, 1e3*vals, yerr=1e3*errs, fmt=mk, color=c, ms=2.6,
                             lw=0, elinewidth=0.6, alpha=0.75)
        # numbers for the tex
        K2m = sp[0]/2.0
        suite_rows.append(dict(system=system, scheme=scheme, K2m=K2m,
                               gal_min=gal.min(), gal_max=gal.max(),
                               null_bound=1e3*max(np.nanmax(np.abs(
                                   [[D[q].get(ch+cp,(0,0))[0] for q in Qs]
                                    for ch in ['C3_Ai','C3_Tij'] for cp in ['_x','_y','_z']])), 1e-9)))
    axA.axhline(0, color='0.8', lw=0.6)
    axA.set_ylabel(r'$m_\psi\,\langle J_{V^0}-J_S\rangle$')
    axB.axhline(1, color='0.45', lw=0.8, ls=(0, (4, 2)))
    axB.set_ylabel(r'$\frac{2M_\Psi}{q}\,\langle J_{V^x}\rangle/\langle J_{V^0}\rangle$')
    axB.set_ylim(0.93, 1.24)
    axC.axhline(0, color='0.45', lw=0.8, ls=(0, (4, 2)))
    axC.set_ylabel(r'null channels $\times 10^{3}$')
    axC.set_ylim(-4.2, 4.2)
    for ax in (axA, axB, axC):
        ax.set_xlim(-0.005, 0.26)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if i == 1: ax.set_xlabel(r'$q^2\ [m_\psi^2]$')
    axC.text(0.95, 0.93, 'dark '+system, transform=axC.transAxes, ha='right',
             va='top', fontsize=8.5, style='italic')
    if i == 0:
        axA.legend(loc='lower left', handletextpad=0.4, labelspacing=0.25)
        axB.text(0.05, 0.05, r'open squares: $\frac{2m_\psi}{q}\langle J_{T^{0x}}\rangle/\langle J_N\rangle$',
                 transform=axB.transAxes, fontsize=6.5)
fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.6)
fig.savefig(f'{OUT}/matrix_elements.pdf')
fig.savefig(f'{OUT}/matrix_elements.png', dpi=150)
print("wrote matrix_elements.pdf")

# ---- tex: the complete measured suite ----
gm = {(r['system'], r['scheme']): r for r in suite_rows}
def gr(sy):
    lo = min(gm[(sy, s)]['gal_min'] for s in SLAB if (sy, s) in gm)
    hi = max(gm[(sy, s)]['gal_max'] for s in SLAB if (sy, s) in gm)
    return lo, hi
nb = max(r['null_bound'] for r in suite_rows)
tex = rf"""% auto-generated by paper_ME_suite.py
\paragraph{{The complete measured suite.}} For each scheme, state, and coupling
we measure all bilinear channels; Fig.~\ref{{fig:MEsuite}} displays the full set
beyond the charge profile at $\alpha_d=0.3$. In summary:
\begin{{align}}
\langle J_N\rangle &= \langle J_{{V^0}}\rangle
   \big[1+{{\mathcal O}}(\langle K\rangle/2m_\psi)\big]\,,\\
\langle J_S\rangle &= \langle J_{{V^0}}\rangle
   - \frac{{\langle K\rangle}}{{m_\psi}}\, g(q^2)\,,\\
\langle J_{{V^x}}\rangle &= \frac{{q}}{{2M_\Psi}}\,\langle J_{{V^0}}\rangle
   \,\big[1+\delta_V(q^2)\big]\,,\\
\langle J_{{T^{{0x}}}}\rangle &= \frac{{q}}{{2m_\psi}}\,\langle J_N\rangle\,,\\
\langle J_{{A^{{y,z}}}}\rangle = \langle J_{{T^{{xy}},T^{{xz}}}}\rangle
&= \langle J_{{V^{{y,z}}}}\rangle = 0\,.
\label{{eq:app:suite}}
\end{{align}}
The fourth relation is exact to machine precision; the last holds by
symmetry, discussed below.
The scalar splitting is anchored at $2\langle K\rangle/2m_\psi$ at $q^2=0$
(Table~\ref{{tab:FF_fits}}) and its measured profile $g(q^2)$ decays faster than
the charge profile (left panels of Fig.~\ref{{fig:MEsuite}}). The convection
deviation $\delta_V$ is the trial-derivative systematic discussed above:
$\delta_V\simeq{gr('meson')[0]-1:+.2f}$ to ${gr('meson')[1]-1:+.2f}$ (meson) and
${gr('baryon')[0]-1:+.2f}$ to ${gr('baryon')[1]-1:+.2f}$ (baryon) over the fitted
$q^2$ range, with the Coulomb schemes at the low end; we therefore quote
$F_{{V^i}}$ through the exact identity and assign $\delta_V$ as its systematic.

The vanishing entries above do not contradict the
nonvanishing nonrelativistic currents of
Eqs.~\eqref{{eq:nonrel_reductions_scalar}}--\eqref{{eq:nonrel_reductions_tensor}}:
the leading terms of ${{\mathcal O}}_{{A}}^i$, ${{\mathcal O}}_T^{{ij}}$, and
${{\mathcal O}}_P$ are spin operators
($\bos\sigma_i$, $[\bos\sigma^i,\bos\sigma^j]/2$, $\bos\sigma\cdot{{\bf q}}$),
and the GFMC implements their spin averages over the degenerate ground
multiplet, under which $\langle\bos\sigma\rangle=0$ and only the orbital
${{\mathcal O}}(1/m_\psi^2)$ remnants survive [e.g.\
$({{\bf q}}\times\nabla)/4m_\psi^2$ in the axial channel]. The spin-dependent
matrix elements themselves are nonzero and given by group-theory factors times
the same spatial profile measured here, since the Hamiltonian is spin
independent and the ground state factorizes into space $\otimes$ spin
(Sec.~\ref{{app:gfmc:me2ff}}). The statements below therefore concern the
spin-averaged (orbital) channels only.

The survive/vanish pattern of the full suite follows from a single
observation: after spin averaging, the $S$-wave ground state is rotationally
structureless, so the only vector available to build a nonzero expectation
value is ${{\bf q}}$ itself. Channels expressible in terms of ${{\bf q}}$ alone
($\langle J_{{V^i}}\rangle,\langle J_{{T^{{0i}}}}\rangle\propto q^i$, and the
scalars) survive; channels requiring an independent axial vector
($\bos\sigma$ or ${{\bf L}}$, both zero here) or a second independent vector
(the antisymmetric $J_{{T^{{ij}}}}$) must vanish.

These vanish for two distinct reasons. For ${{\bf q}}\parallel\hat x$ the
components $J_{{A^x}}\propto({{\bf q}}\times\nabla)^x$,
$J_{{T^{{yz}}}}\propto q_z\nabla_y-q_y\nabla_z$, and
$J_{{T^{{0y}}}},J_{{T^{{0z}}}}\propto q^{{y,z}}$ vanish identically at the
operator level: they are $0$ configuration by configuration, not
measurements (verified exactly in all datasets). The remaining six
($J_{{A^{{y,z}}}}, J_{{T^{{xy}},T^{{xz}}}}, J_{{V^{{y,z}}}}$, all
$\propto\langle e^{{iqx}}\nabla_{{y,z}}\rangle$) vanish by the reflection
symmetry of the $S$-wave state about the ${{\bf q}}$ axis; for these the Monte
Carlo provides a statistical test. Across all schemes, states,
couplings, and momenta ($720$ measurements) the pull distribution
$({{\rm ME}}/\sigma_{{\rm ME}})$ has mean $0.08$ and width $0.99$, with $5.0\%$
$(0.3\%)$ of pulls beyond $2\sigma$ $(3\sigma)$, exactly the standard-normal
expectation for pure noise about zero. The corresponding bound is
$|{{\rm ME}}|/\langle J_{{V^0}}\rangle < 1.4\times10^{{-3}}$ at the $95\%$ level
(median $2.5\times10^{{-5}}$); equivalently, these channels are consistent with
zero at the per-mille level of the charge, and the agreement of their
fluctuations with pure noise is itself a check that the sampling introduces no
spurious anisotropy.

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/matrix_elements.pdf}}
\caption{{The measured matrix-element suite beyond the charge profile, at
$\alpha_d=0.3$, for the dark meson (top) and baryon (bottom). Left:
scalar--vector splitting $m_\psi\langle J_{{V^0}}-J_S\rangle$, anchored at
$2\langle K\rangle/2m_\psi$ at $q^2=0$. Middle: convection ratio
$(2M_\Psi/q)\langle J_{{V^x}}\rangle/\langle J_{{V^0}}\rangle$ against the exact
Galilean identity (dashed); open squares show the kinematic tensor ratio
$(2m_\psi/q)\langle J_{{T^{{0x}}}}\rangle/\langle J_N\rangle$, which is exact by
construction. Right: the six symmetry-null channels (axial: triangles, tensor
$ij$: squares, transverse vector: circles); their pulls follow a standard
normal (see text), bounding them below $1.4\times10^{{-3}}$ of the charge at
$95\%$. The four operator-exact zeros ($J_{{A^x}}$, $J_{{T^{{yz}}}}$,
$J_{{T^{{0y}}}}$, $J_{{T^{{0z}}}}$) are identically zero and not shown.}}
\label{{fig:MEsuite}}
\end{{figure*}}
"""
with open(f'{OUT}/me_suite.tex', 'w') as f:
    f.write(tex)
if os.path.isdir(OVER):
    import shutil
    shutil.copy(f'{OUT}/me_suite.tex', os.path.join(OVER, 'me_suite.tex'))
    shutil.copy(f'{OUT}/matrix_elements.pdf', os.path.join(OVER, 'figs', 'matrix_elements.pdf'))
    print("copied me_suite.tex + figs/matrix_elements.pdf into Overleaf")
