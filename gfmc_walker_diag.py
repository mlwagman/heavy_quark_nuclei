#!/usr/bin/env python3
r"""Walker-level diagnostics for the consistent-Gram GEVP: is the answer carried by the ensemble
or by a handful of walkers?

The GEVP estimator at rank m is :math:`J = v^\dagger M v / v^\dagger G v` with
:math:`G[i,j] = \langle W_{ij}\rangle_w` and :math:`M[i,j] = \langle C_{ij}\rangle_w`, so once the
Ritz vector v is fixed from the ensemble mean, every walker contributes a pair

.. math::
    d_w = v^\dagger G_w v   \quad\text{(its weight in the Ritz state)},
    \qquad n_w = v^\dagger M_w v,
    \qquad J = \frac{\sum_w n_w}{\sum_w d_w} .

``d_w`` is the state-identification weight: the Ritz vector picks a state, and ``d_w`` says how
much of that state each walker carries.  A healthy ensemble spreads it over many walkers.  The
summary below is therefore about ``d_w``:

* ``N_eff/N`` -- :math:`(\sum d_w)^2 / (N \sum d_w^2)`, the effective fraction of the ensemble
  actually contributing.  1 is a flat weight, 1/N is one walker carrying everything.
* ``top1`` / ``top10`` -- the share of :math:`\sum|d_w|` held by the single largest and the ten
  largest walkers.  With 5000 walkers a flat ensemble gives 0.02% and 0.2%.
* ``neg`` -- the fraction of walkers with :math:`d_w < 0`.  The measured Gram is not positive
  definite, so a few are expected; many are not.
* ``dJ(top1)`` -- how far J moves when the single highest-\|d_w\| walker is dropped, in units of
  the bootstrap error.  This is the number that matters: a result that survives its loudest
  walker is one the error bar describes.

Also prints a log-spaced ASCII histogram of :math:`|d_w|` over its median, which is where a fat
tail shows up as a detached right-hand bin rather than a smooth falloff.

Usage
-----
    gfmc_walker_diag.py --dir <run_dir> [--m 8 12 16] [--current C3_N] [--gevp_cut 1e-2]
"""
import argparse
import glob
import os
import re

import numpy as np
import h5py

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--dir', required=True, help='directory of currents_3pt_*.h5')
p.add_argument('--current', default='C3_N')
p.add_argument('--m', type=int, nargs='+', default=[8, 12, 16])
p.add_argument('--gevp_cut', type=float, default=1e-2)
p.add_argument('--n_block', type=int, default=50, help='walkers per bootstrap block')
p.add_argument('--n_boot', type=int, default=200)
p.add_argument('--seed', type=int, default=17)
p.add_argument('--hist', action='store_true', help='print the |d_w| histogram for every (Q, m)')
args = p.parse_args()


def q_of(fn):
    m = re.search(r'Qmag([0-9.eE+-]+)\.h5$', fn)
    return float(m.group(1)) if m else None


def ritz(G, m, cut):
    """Ground-state Ritz vector of T v = lambda G v at rank m, from the ensemble-mean Gram."""
    g = 0.5 * (G[:m, :m] + G[:m, :m].T)
    t = G[:m, 1:m + 1]
    t = 0.5 * (t + t.T)
    ev, U = np.linalg.eigh(g)
    ok = ev > cut * ev.max()
    if ok.sum() == 0:
        return None
    Uk = U[:, ok] / np.sqrt(ev[ok])
    a = Uk.T @ t @ Uk
    lam, V = np.linalg.eigh(0.5 * (a + a.T))
    return Uk @ V[:, np.argmax(lam)]


def histogram(x, nbin=12):
    """Log-spaced ASCII histogram of |x|/median(|x|)."""
    a = np.abs(x)
    med = np.median(a[a > 0]) if np.any(a > 0) else 1.0
    r = a / med
    lo, hi = -2.0, max(2.0, np.log10(r.max()) + 1e-9)
    edges = np.linspace(lo, hi, nbin + 1)
    cnt, _ = np.histogram(np.log10(np.maximum(r, 1e-12)), bins=edges)
    wide = cnt.max() if cnt.max() else 1
    out = []
    for k in range(nbin):
        bar = '#' * int(round(40 * cnt[k] / wide))
        out.append(f'      10^{edges[k]:+5.2f}..{edges[k + 1]:+5.2f}  {cnt[k]:6d} |{bar}')
    return out


def main():
    files = {q_of(f): f for f in sorted(glob.glob(os.path.join(args.dir, 'currents_3pt_*.h5')))
             if q_of(f) is not None and 'korr' not in f}
    if not files:
        raise SystemExit(f'no currents_3pt_*.h5 in {args.dir}')
    nm = os.path.basename(files[sorted(files)[0]])
    mm = re.search(r'Ncoord(\d+)', nm)
    print(f'{args.dir}\n  N_coord = {mm.group(1) if mm else "?"}   current = {args.current}   '
          f'gevp_cut = {args.gevp_cut:g}')
    print(f'\n  {"Q":>6s} {"m":>3s} {"J":>10s} {"N_eff/N":>9s} {"top1":>8s} {"top10":>8s}'
          f' {"neg":>7s} {"dJ(top1)":>10s}')
    for q in sorted(files):
        with h5py.File(files[q], 'r') as f:
            W = np.real(f['Ws2_offdiag'][:])                 # (Nt, Nt, n_walkers)
            C = np.real(f[f'{args.current}_offdiag'][:])
        G, M = W.mean(-1), C.mean(-1)
        nw = W.shape[-1]
        rng = np.random.default_rng(args.seed)
        nblk = nw // args.n_block
        for m in args.m:
            if m + 1 > G.shape[0]:
                continue
            v = ritz(G, m, args.gevp_cut)
            if v is None:
                continue
            d = np.einsum('i,ijw,j->w', v, W[:m, :m], v)      # per-walker Ritz weight
            n = np.einsum('i,ijw,j->w', v, C[:m, :m], v)
            J = n.sum() / d.sum()
            ad = np.abs(d)
            neff = d.sum() ** 2 / (nw * (d ** 2).sum())
            order = np.argsort(ad)[::-1]
            top1, top10 = ad[order[0]] / ad.sum(), ad[order[:10]].sum() / ad.sum()
            neg = (d < 0).mean()
            # bootstrap error on J at this rank, same blocking as the extraction
            db = d[:nblk * args.n_block].reshape(nblk, args.n_block).sum(1)
            nb_ = n[:nblk * args.n_block].reshape(nblk, args.n_block).sum(1)
            idx = rng.integers(0, nblk, size=(args.n_boot, nblk))
            draws = nb_[idx].sum(1) / db[idx].sum(1)
            err = float(np.nanstd(draws, ddof=1))
            k = order[0]
            J_drop = (n.sum() - n[k]) / (d.sum() - d[k])
            dJ = (J_drop - J) / err if err > 0 else np.nan
            print(f'  {q:6g} {m:3d} {J:10.5f} {neff:9.4f} {top1:8.2%} {top10:8.2%}'
                  f' {neg:7.2%} {dJ:+9.2f}s')
            if args.hist:
                print(f'    |d_w| / median, log10 bins  ({nw} walkers):')
                print('\n'.join(histogram(d)))
    print(f'\n  A flat {nw}-walker ensemble gives N_eff/N = 1, top1 = {1 / nw:.2%}, '
          f'top10 = {10 / nw:.2%}.')
    print('  dJ(top1) is the shift from dropping the single loudest walker, in bootstrap sigma:')
    print('  below ~0.3s the result is an ensemble average, not one walker.')


if __name__ == '__main__':
    main()
