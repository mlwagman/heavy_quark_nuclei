#!/usr/bin/env python3
r"""Ground-state matrix elements from the GFMC two-time objects by a consistent-Gram GEVP.

Why this exists
---------------
Both existing extractions build the Ritz rotators from a *reduced* 2pt object and then
sandwich the *full* two-time 3pt matrix:

    gfmc_korr.py      Gram from Hankel(Ws2_offdiag[0, :])      (row 0)
    gfmc_me_korr.py   Gram from Hankel(antidiagonal means)     (all i+j = const cells)

Both estimate the same expectation as ``Ws2_offdiag[i, j]`` but are *different noise
realizations* of it.  The FRR ratio then inherits the mismatch, and it is not a small effect:
at ``Q = 0`` the current is :math:`J_N = N\mathbb{1}`, so the measured 3pt matrix satisfies
``C3_N_offdiag == N * Ws2_offdiag`` **identically** (verified: max abs difference exactly 0),
and therefore

    v^dag (N G) v / v^dag G v = N

for *any* vector at *any* rank.  Yet gfmc_korr.py returns 2.27 instead of 2 on such data.
Feeding it a 3pt built from the same object its Gram comes from returns 2 to 1e-15, which
isolates the mismatch as the cause rather than anything inside korr.

The construction here
---------------------
On the stored grid :math:`t_k = k\Delta` the Krylov vectors are
:math:`|b_k\rangle = e^{-H t_k}|\psi_T\rangle`, and the generator already stores exactly the
two objects that construction needs, measured on the *same* walkers:

    G[i, j]  = <b_i | b_j>       = Ws2_offdiag[i, j]        = C_2(t_i + t_j)
    M[i, j]  = <b_i | J | b_j>   = C3_N_offdiag[i, j]       = C_J(t_i, t_j)
    T[i, j]  = <b_i | e^{-H\Delta} | b_j> = Ws2_offdiag[i, j+1]

so the ground-state Ritz vector solves the generalized eigenproblem :math:`T v = \lambda G v`
with :math:`\lambda = e^{-E_0\Delta}` the largest eigenvalue, and

    J_00 = v^dag M v / v^dag G v .

Because G and M come from one measurement, ``Q = 0`` gives exactly N by construction, at every
rank -- the charge check becomes an identity rather than a test, which is the point.  G is
symmetrised and solved through a truncated eigendecomposition (drop eigenvalues below
``--gevp_cut`` times the largest) because the measured Gram is near-singular at large rank.

Usage
-----
    gfmc_me_gevp.py --dir <run_dir> [--m 2 3 4 ...] [--n_block 50] [--n_boot 200]
                    [--spike_leverage 100] [--system meson|baryon] [--ref <r^2>]

Prints J_00(Q) and the fitted radius for each rank so the plateau can be read off in the
observable itself, which is the selection rule that survived the earlier m_max study (the
``Q = 0`` charge check cannot select a rank here, by construction, and could not before either:
at ``Q = 0`` the current commutes with H, so J_fi is diagonal and the check is blind to exactly
the f != i admixture the rank controls).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py
from scipy.optimize import curve_fit

p = argparse.ArgumentParser()
p.add_argument('--dir', required=True, help='directory of currents_3pt_*.h5')
p.add_argument('--current', default='C3_N')
p.add_argument('--m', type=int, nargs='+', default=[2, 3, 4, 5, 6, 8, 10, 12],
               help='Krylov ranks to report')
p.add_argument('--n_block', type=int, default=50, help='walkers per bootstrap block')
p.add_argument('--n_boot', type=int, default=200)
p.add_argument('--seed', type=int, default=17)
p.add_argument('--spike_leverage', type=float, default=0.0,
               help='drop walkers with max_ij |w|/median_w |w| above this (0 = off)')
p.add_argument('--gevp_cut', type=float, default=1e-2,
               help='drop Gram eigenvalues below this fraction of the largest. The MEASURED Gram '
                    'is not positive definite -- it carries negative eigenvalues (observed down '
                    'to -6e-2 of the largest) and a dense floor of ~1e-4 noise modes. Whitening '
                    'by 1/sqrt(lambda) amplifies those, so a 1e-10 cut makes the GEVP explode '
                    '(values spanning 4e-9 to 2698 with 1e7 error bars). 1e-2 is stable, and the '
                    'answer is insensitive across 1e-2..1e-1.')
p.add_argument('--extra_decay_rel', type=float, default=0.1875,
               help='extra decay as a fraction of the measured growth rate (scale covariant; '
                    'a fixed absolute value breaks at small alpha)')
p.add_argument('--system', default=None, choices=['meson', 'baryon'],
               help='picks the dipole variant: p=2 for the meson, free p for the baryon. '
                    'Irrelevant for --fit z.')
p.add_argument('--N', type=int, default=None,
               help='number of constituents, i.e. F(0). Read from the Ncoord<N> in the filename '
                    'when present, so general-Nc baryons (N_coord = Nc > 3) work without it.')
p.add_argument('--ref', type=float, default=None, help='reference <r^2> to compare against')
p.add_argument('--q_max', type=float, default=None,
               help='restrict the radius fit to Q <= this. The p=2 dipole is NOT a faithful '
                    'estimator of <r^2> for the MRS form factor: fitting the EXACT Numerov F/F0 '
                    'with it returns +1.0%% over Q<=0.15 but +4.3%% over Q<=0.4 (LO; +1.1%%/+4.9%% '
                    'at NLO), because the shape is not a dipole and the large-Q points pull the '
                    'fit. Over the wide grid that bias exceeds our statistics, so restrict the '
                    'fit, and compare fit-to-fit via --ref_ff.')
p.add_argument('--f0_analytic', action='store_true',
               help='Take F(0) = N_coord analytically instead of from a Q=0 run. Exact here: '
                    'with the consistent-Gram GEVP, C3_N_offdiag == N*Ws2_offdiag identically at '
                    'Q=0 (max abs difference 0.0), so F(0) = N for every vector, at every rank, '
                    'in every bootstrap draw -- F0d[b] is a constant and adds no variance. Lets a '
                    'replica/seed study skip its Q=0 runs, halving the cost per seed.')
p.add_argument('--ref_ff', default=None,
               help='exact F/F0 on the same Q grid, as "Q:F,Q:F,...". Reports the GFMC F/F0 '
                    'against it per Q, and refits the exact values with the identical fit to give '
                    'a like-for-like target, which is what the GFMC number should be judged '
                    'against rather than the true <r^2>.')
p.add_argument('--fit', default='dipole', choices=['dipole', 'z'],
               help="radius ansatz. 'dipole' is the historical one: (1 + Q^2 b^2/4p)^-p with "
                    "p = 2 for the meson (exact only for Coulomb) and p free for the baryon. "
                    "'z' is the conformal-map expansion with t_0 = 0, where the radius is the "
                    "linear coefficient alone and nothing is assumed about the shape.")
p.add_argument('--t_cut', type=float, default=None,
               help="branch point in Q^2 for --fit z. The one-body density of an N-body state "
                    "falls as e^{-sqrt(t_cut) x}: for the meson x = r/2 and |psi|^2 ~ e^{-2 kappa r} "
                    "give t_cut = 16 kappa^2 with kappa = sqrt(2 mu |E0|); for the baryon the "
                    "struck quark sits at x = -2y/3 from the CM, giving t_cut = 9 kappa_2^2 = "
                    "12 |E_sep| with E_sep = E0(qqq) - E0(qq in 3bar).")
p.add_argument('--k_max', type=int, default=2, help='highest power of z for --fit z')
args = p.parse_args()
if args.fit == 'z' and args.t_cut is None:
    p.error('--fit z needs --t_cut')


def q_of(fn):
    m = re.search(r'Qmag([0-9.eE+-]+)\.h5$', fn)
    return float(m.group(1)) if m else None


def leverage(w):
    """Per-walker spike score, max over cells of |w| / median over walkers."""
    lev = np.zeros(w.shape[2])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            med = np.median(np.abs(w[i, j]))
            if med > 0:
                lev = np.maximum(lev, np.abs(w[i, j]) / med)
    return lev


def read(fn, current, keep=None):
    with h5py.File(fn, 'r') as f:
        for k in (f'{current}_offdiag', 'Ws2_offdiag'):
            if k not in f:
                raise SystemExit(f'{fn}: no {k} (pre-fix generator?)')
        w = np.real(f['Ws2_offdiag'][:])
        c = np.real(f[f'{current}_offdiag'][:])
    if keep is not None:
        w, c = w[:, :, keep], c[:, :, keep]
    return w, c


def block(x, n_block):
    """Average contiguous walker blocks; returns (n_samples, ...) leading axis."""
    n = x.shape[-1] // n_block
    head = x[..., :n * n_block].reshape(*x.shape[:-1], n, n_block).mean(-1)
    return np.moveaxis(head, -1, 0)


def j00(G, M, m, cut):
    r"""Ground-state J_00 from the consistent-Gram GEVP at rank m.

    G, M are (Nt, Nt).  Uses G[:m,:m], T = G[:m,1:m+1], M[:m,:m].
    """
    g = 0.5 * (G[:m, :m] + G[:m, :m].T)
    t = G[:m, 1:m + 1]
    t = 0.5 * (t + t.T)
    ev, U = np.linalg.eigh(g)
    ok = ev > cut * ev.max()
    if ok.sum() == 0:
        return np.nan
    Uk = U[:, ok] / np.sqrt(ev[ok])          # whitening: Uk^dag g Uk = 1
    a = Uk.T @ t @ Uk
    a = 0.5 * (a + a.T)
    lam, V = np.linalg.eigh(a)
    v = Uk @ V[:, np.argmax(lam)]            # largest lambda = e^{-E0 Delta}
    den = v @ g @ v
    if den == 0 or not np.isfinite(den):
        return np.nan
    return float(v @ (0.5 * (M[:m, :m] + M[:m, :m].T)) @ v / den)


def analyse_q(w, c, ms, cut, decay_rel, n_block, n_boot, rng):
    """J_00 at each rank for one Q, with a block bootstrap."""
    W = block(w, n_block)                    # (ns, Nt, Nt)
    C = block(c, n_block)
    ns, Nt, _ = W.shape
    Wm, Cm = W.mean(0), C.mean(0)
    # conditioning shift: common e^{-Es(i+j)} leaves J_00 invariant (it is E -> E + Es)
    row = np.abs(Wm[0])
    k = np.arange(Nt)
    use = row > 0
    growth = np.polyfit(k[use][:min(11, use.sum())],
                        np.log(row[use][:min(11, use.sum())]), 1)[0]
    Es = growth + decay_rel * abs(growth)
    sh = np.exp(-Es * k)
    S = sh[:, None] * sh[None, :]
    out = {}
    for m in ms:
        if m + 1 > Nt:
            continue
        cen = j00(Wm * S, Cm * S, m, cut)
        draws = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, ns, ns)
            draws[b] = j00(W[idx].mean(0) * S, C[idx].mean(0) * S, m, cut)
        out[m] = (cen, float(np.nanstd(draws, ddof=1)), draws)
    return out


def zvar(q2, t_cut):
    s, s0 = np.sqrt(t_cut + q2), np.sqrt(t_cut)
    return (s - s0) / (s + s0)


def fit_radius(Qs, F, E, system):
    """p=2 dipole for the meson, free p for the baryon; matches gfmc_korr.py's convention.

    With ``--fit z`` the same call returns the z-expansion radius instead, so the bootstrap
    refit and the ``--ref_ff`` like-for-like target automatically use whichever ansatz is
    selected -- the comparison stays fit-to-fit.
    """
    if args.fit == 'z':
        # F/F(0) = 1 + sum_{k>=1} c_k z^k.  The constant is FIXED to 1 rather than fitted:
        # z(0) = 0 and F(0)/F(0) = 1 by construction, so a free c_0 is a redundant parameter
        # that the large-Q points pull on.  Letting it float costs ~1% of bias at k_max = 3
        # (measured on the exact reference form factors), which is the whole effect we are
        # trying to resolve.  Same convention as Ben's paper_korr.py zfit.
        z = zvar(Qs ** 2, args.t_cut)
        A = np.vstack([z ** k for k in range(1, args.k_max + 1)]).T / E[:, None]
        c, *_ = np.linalg.lstsq(A, (F - 1) / E, rcond=None)
        return -1.5 * c[0] / args.t_cut, float(args.k_max)
    ok = (F > 0.05) & (F < 1)
    b0 = np.sqrt(np.median(8 * (F[ok] ** -0.5 - 1) / Qs[ok] ** 2)) if ok.any() else 6.
    if system == 'meson':
        q, _ = curve_fit(lambda x, b: (1 + x * b * b / 8.) ** -2., Qs ** 2, F, p0=[b0], sigma=E)
        return 1.5 * q[0] ** 2, 2.0
    q, _ = curve_fit(lambda x, b, pp: (1 + x * b * b / (4 * pp)) ** -pp, Qs ** 2, F,
                     p0=[b0, 3], sigma=E, bounds=([b0 / 20, 1], [b0 * 20, 12]))
    return 1.5 * q[0] ** 2, q[1]


def main():
    files = {q_of(f): f for f in sorted(glob.glob(os.path.join(args.dir, 'currents_3pt_*.h5')))
             if q_of(f) is not None and 'korr' not in f}
    if not files:
        raise SystemExit(f'no currents_3pt_*.h5 in {args.dir}')
    qs = sorted(files)
    system = args.system
    if system is None:
        system = 'baryon' if 'baryon' in args.dir else 'meson'
    if args.N is not None:
        N = args.N
    else:
        m = re.search(r'Ncoord(\d+)', os.path.basename(files[qs[0]]))
        N = int(m.group(1)) if m else (3 if system == 'baryon' else 2)

    # one drop set, fixed on Q=0, so every Q describes the same ensemble
    keep = None
    if args.spike_leverage > 0:
        w0, _ = read(files[qs[0]], args.current)
        lev = leverage(w0)
        bad = lev > args.spike_leverage
        keep = np.where(~bad)[0]
        srt = np.sort(lev)[::-1]
        print(f'spike screen: dropping {bad.sum()} of {len(lev)} walkers '
              f'(worst {srt[0]:.0f}x, next {srt[1]:.0f}x, {srt[2]:.0f}x; '
              f'threshold {args.spike_leverage:g}x)')

    print(f'{args.dir}\n  system={system}  N={N}  {len(qs)} Q values: '
          + ' '.join(f'{q:g}' for q in qs))

    res = {}
    for q in qs:
        w, c = read(files[q], args.current, keep)
        rng = np.random.default_rng(args.seed)      # same resamples at every Q
        res[q] = analyse_q(w, c, args.m, args.gevp_cut, args.extra_decay_rel,
                           args.n_block, args.n_boot, rng)

    ref_ff = None
    if args.ref_ff:
        ref_ff = {}
        for tok in args.ref_ff.split(','):
            a, b = tok.split(':')
            ref_ff[float(a)] = float(b)

    ms = sorted(set.intersection(*(set(r) for r in res.values())))
    fit_target = args.ref
    if ref_ff:
        pos_all = [q for q in qs if q > 0 and (args.q_max is None or q <= args.q_max + 1e-12)]
        have = [q for q in pos_all if q in ref_ff]
        if len(have) >= 2:
            try:
                fit_target = fit_radius(np.array(have),
                                        np.array([ref_ff[q] for q in have]),
                                        np.ones(len(have)), system)[0]
                print(f'\n  like-for-like target: the identical fit applied to the EXACT F/F0 on '
                      f'this grid\n  gives <r^2> = {fit_target:.3f}'
                      + (f', against the true {args.ref:.3f} '
                         f'({100 * (fit_target / args.ref - 1):+.2f}% fit-form bias)'
                         if args.ref else ''))
            except Exception as e:
                print(f'  (could not refit the reference form factor: {e!s:.50})')

    print(f'\n  {"m":>3s} {"F(0)":>12s} {"dev from N":>12s}   <r^2> +/- err'
          + ('      target     pull' if fit_target else ''))
    for m in ms:
        if args.f0_analytic:
            F0, F0e, F0d = float(N), 0.0, np.full(args.n_boot, float(N))
        else:
            F0, F0e, F0d = res[qs[0]][m]
        pos = [q for q in qs if q > 0 and (args.q_max is None or q <= args.q_max + 1e-12)]
        F = np.array([res[q][m][0] / F0 for q in pos])
        E = np.array([max(np.nanstd(res[q][m][2] / F0d, ddof=1), 1e-5) for q in pos])
        try:
            r2, pw = fit_radius(np.array(pos), F, E, system)
        except Exception as e:
            print(f'  {m:3d} {F0:12.6f} {F0 - N:12.2e}   fit failed: {e!s:.40}')
            continue
        # refit each bootstrap draw, keeping the Q-Q correlation
        rb = []
        for b in range(args.n_boot):
            Fb = np.array([res[q][m][2][b] / F0d[b] for q in pos])
            if np.all(np.isfinite(Fb)):
                try:
                    rb.append(fit_radius(np.array(pos), Fb, E, system)[0])
                except Exception:
                    pass
        er = float(np.std(rb, ddof=1)) if len(rb) > 2 else float('nan')
        line = f'  {m:3d} {F0:12.6f} {F0 - N:12.2e}   {r2:9.4g} +/- {er:<8.3g}'
        if fit_target:
            line += (f'  {fit_target:8.4g}  {100 * (r2 / fit_target - 1):+6.2f}%'
                     f'  ({(r2 - fit_target) / er:+.1f}s)')
        print(line)
        last = (m, pos, F, E, F0d)

    if ref_ff and 'last' in dir():
        m, pos, F, E, F0d = last
        print(f'\n  F(Q)/F(0) at m={m}, model independent:')
        print(f'    {"Q":>6s} {"GFMC":>18s} {"exact":>9s} {"ratio":>8s}'
              f'   {"<r^2> ratio":>19s}')
        for q, f, e in zip(pos, F, E):
            if q not in ref_ff:
                print(f'    {q:6g} {f:9.5f} +/- {e:<6.4f} {"-":>9s} {"-":>8s}')
                continue
            # (1 - F)/(1 - F_exact) at fixed Q tends to <r^2>_GFMC/<r^2>_exact as Q -> 0 and
            # needs no ansatz at all: at the lowest Q it is the sharpest statement the data
            # supports about the radius, free of every fit-form question.
            rr = (1 - f) / (1 - ref_ff[q])
            d = np.array([(1 - res[q][m][2][b] / F0d[b]) / (1 - ref_ff[q])
                          for b in range(args.n_boot)])
            rre = float(np.nanstd(d, ddof=1))
            print(f'    {q:6g} {f:9.5f} +/- {e:<6.4f} {ref_ff[q]:9.5f} {f / ref_ff[q]:8.4f}'
                  f'   {rr:8.4f} +/- {rre:<6.4f} ({(rr - 1) / rre:+.1f}s)')

    print('\n  F(0) must equal N exactly here -- G and M are one measurement, so any')
    print('  deviation is a bug, not a statistical statement.  Choose m from the <r^2> plateau,')
    print('  and judge <r^2> against the like-for-like target, not the true value.')


if __name__ == '__main__':
    main()
