#!/usr/bin/env python3
r"""Export the fixed-order quarkonium two-leg GFMC data as CSV for the korr tutorial.

The tutorial has to stand alone inside the korr repository, so the two-time arrays go out as
plain CSV with a documented column map, binned over walkers to a manageable number of samples.

What is exported, per order (LO, NLO)
-------------------------------------
``*_c2.csv``   the two-point two-time array :math:`G[i,j] = \langle b_i|b_j\rangle = C_2(t_i+t_j)`
               (generator key ``Ws2_offdiag``), one row per sample, 169 columns ``c2_i_j``.
               **One file per order, not per Q**: ``Ws2_offdiag`` is bit-identical across the Q
               runs (verified, max abs difference exactly 0) because every Q run uses the same
               seed and walkers and the current momentum enters only the three-point junction.
               That also means the Q points are perfectly correlated through a shared two-point
               function, which a fit across Q has to respect.
``*_c3.csv``   the three-point two-time array :math:`M[i,j] = \langle b_i|J|b_j\rangle =
               C_J(t_i,t_j)` (key ``C3_N_offdiag``), with a leading ``Q`` column.
``*_times.csv``  the per-leg imaginary times :math:`t_k = k\Delta` and the generator's step
               indices, so the grid step is explicit.

Both arrays are stored real: the imaginary part of ``C3_N_offdiag`` is at the 1e-12 level of the
real part (the current is Hermitian and the ensemble is real up to roundoff).

The arrays are **not** symmetric in :math:`(i,j)` -- leg 1 carries the source evolution and leg 2
the sink, and their statistical realizations differ -- so the full square is written, not a
triangle.

The exact identity worth knowing before any analysis: at :math:`Q=0` the number operator is
:math:`J_N = N\mathbb{1}`, and the generator's two arrays satisfy
``C3_N_offdiag == N * Ws2_offdiag`` *element by element, per walker* (checked here, ratio exactly
2.0).  Any Ritz contraction therefore returns :math:`F(0) = N` identically, at every rank -- so
the charge check is an identity, not a test, and cannot be used to pick a rank.

Usage
-----
    export_korr_tutorial_data.py --run_dir <dir> --tag fo_lo --out_dir <korr>/tutorials/data
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--run_dir', required=True, help='directory of currents_3pt_*.h5')
p.add_argument('--tag', required=True, help="output stem, e.g. 'fo_lo'")
p.add_argument('--out_dir', required=True)
p.add_argument('--prefix', default='quarkonium_', help='filename prefix')
p.add_argument('--n_block', type=int, default=50, help='walkers averaged per exported sample')
p.add_argument('--n_t', type=int, default=None, help='keep only the first n_t times (default all)')
p.add_argument('--fmt', default='%.9e', help='per-value format')
p.add_argument('--spike_leverage', type=float, default=100.0,
               help='drop walkers whose leverage max_ij |w| / median_walkers |w| exceeds this '
                    '(0 = off). A single walker with a runaway weight biases a Hankel mean '
                    'directly and binning cannot remove it: on the fixed-order LO meson two of '
                    '5000 walkers sit at 3607x and 3171x the median while the third worst is 65x, '
                    'and keeping them moves <r^2> from -2.7%% to -8.9%% of the exact 168.75 at '
                    'rank 12 while inflating the error 2.7x. The drop set is fixed once on the '
                    'Q=0 file so every Q describes the same ensemble.')
p.add_argument('--current', default='C3_N_offdiag')
p.add_argument('--gram', default='Ws2_offdiag')
args = p.parse_args()


def q_of(fn):
    m = re.search(r'Qmag([0-9.eE+-]+)\.h5$', fn)
    return float(m.group(1)) if m else None


files = {q_of(f): f for f in sorted(glob.glob(os.path.join(args.run_dir, 'currents_3pt_*.h5')))
         if q_of(f) is not None and 'korr' not in f}
if not files:
    raise SystemExit(f'no currents_3pt_*.h5 in {args.run_dir}')
qs = sorted(files)
os.makedirs(args.out_dir, exist_ok=True)


def read(fn):
    with h5py.File(fn) as h:
        G = np.array(h[args.gram])
        M = np.array(h[args.current])
        tv = np.array(h['tau_values'])
        ti = np.array(h['t_indices'])
    return G, M, tv, ti


def binned(A, n_block, keep=None):
    """``(n_t, n_t, n_walkers) -> (n_samp, n_t, n_t)`` by averaging blocks of walkers."""
    A = np.moveaxis(np.real(A), -1, 0)
    if keep is not None:
        A = A[keep]
    n = (A.shape[0] // n_block) * n_block
    return A[:n].reshape(-1, n_block, *A.shape[1:]).mean(axis=1)


def leverage(w):
    """Per-walker spike score: max over cells of |w| / the median over walkers of that cell."""
    lev = np.zeros(w.shape[-1])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            med = np.median(np.abs(w[i, j]))
            if med > 0:
                lev = np.maximum(lev, np.abs(w[i, j]) / med)
    return lev


# ---------------------------------------------------------------------------------------------
# read, check the invariants the tutorial will rely on, and bin
# ---------------------------------------------------------------------------------------------
G0, M0, tv, ti = read(files[qs[0]])
nt = args.n_t or G0.shape[0]
assert qs[0] == 0.0, f'expected a Q=0 run first, got {qs[0]}'

keep = None
if args.spike_leverage > 0:
    lev = leverage(np.real(G0))
    keep = lev <= args.spike_leverage
    srt = np.sort(lev)[::-1]
    print(f'  spike screen: dropping {int((~keep).sum())} of {len(lev)} walkers '
          f'(worst {srt[0]:.0f}x, next {srt[1]:.0f}x, {srt[2]:.0f}x; '
          f'threshold {args.spike_leverage:g}x)')

imax = 0.0
gram_dev = 0.0
samples = {}
for q in qs:
    G, M, tvq, tiq = read(files[q])
    assert np.array_equal(tvq, tv) and np.array_equal(tiq, ti), f'time grid differs at Q={q}'
    gram_dev = max(gram_dev, float(np.abs(G - G0).max()))
    imax = max(imax, float(np.abs(np.imag(M)).max() / np.abs(np.real(M)).max()))
    samples[q] = binned(M, args.n_block, keep)[:, :nt, :nt]

ratio = np.real(M0) / np.real(G0)
N = int(round(float(ratio.flat[0])))
print(f'{args.run_dir}')
print(f'  {len(qs)} Q values: ' + ' '.join(f'{q:g}' for q in qs))
print(f'  grid: {G0.shape[0]} per-leg times, step Delta = {tv[1] - tv[0]:g}, '
      f'exporting {nt}; {G0.shape[-1]} walkers -> {len(samples[qs[0]])} samples '
      f'of {args.n_block}')
print(f'  Ws2_offdiag identical across Q:        max abs difference {gram_dev:.3e}')
print(f'  C3_N_offdiag imaginary part / real:    {imax:.3e}')
print(f'  C3_N_offdiag / Ws2_offdiag at Q=0:     [{ratio.min():.12g}, {ratio.max():.12g}] '
      f'-> N = {N}')
assert gram_dev == 0.0, 'the 2pt array is NOT shared across Q; export it per Q instead'
assert np.allclose(ratio, N, rtol=0, atol=1e-12), 'the Q=0 identity C3 = N*Ws2 does not hold'

g_samp = binned(G0, args.n_block, keep)[:, :nt, :nt]
cols = [f'{{}}_{i}_{j}' for i in range(nt) for j in range(nt)]


def write(path, header_lines, colnames, rows):
    with open(path, 'w') as fh:
        for line in header_lines:
            fh.write(f'# {line}\n')
        fh.write(','.join(colnames) + '\n')
        for r in rows:
            fh.write(','.join(r) + '\n')
    print(f'  -> {path}  ({os.path.getsize(path) / 1e6:.2f} MB)')


prov = [
    f'quarkonium two-leg GFMC, tag {args.tag}',
    f'source directory {args.run_dir}',
    f'{G0.shape[-1]} walkers'
    + (f', {int((~keep).sum())} dropped by the leverage > {args.spike_leverage:g} spike screen'
       if keep is not None else '')
    + f', binned by {args.n_block} -> {len(g_samp)} samples',
    f'{nt} per-leg times t_k = k*Delta with Delta = {tv[1] - tv[0]:g} (see the *_times.csv)',
    'columns <kind>_i_j are the (i,j) entry of the two-time array, i = leg 1, j = leg 2',
    'real part only; see export_korr_tutorial_data.py in the heavy_quark_nuclei repo',
]

write(os.path.join(args.out_dir, f'{args.prefix}{args.tag}_c2.csv'),
      prov + ['C2 two-time array Ws2_offdiag[i,j] = C_2(t_i + t_j), shared by every Q'],
      ['sample'] + [c.format('c2') for c in cols],
      [[str(s)] + [args.fmt % v for v in g_samp[s].ravel()] for s in range(len(g_samp))])

rows = []
for q in qs:
    A = samples[q]
    for s in range(len(A)):
        rows.append([f'{q:g}', str(s)] + [args.fmt % v for v in A[s].ravel()])
write(os.path.join(args.out_dir, f'{args.prefix}{args.tag}_c3.csv'),
      prov + ['C3 two-time array C3_N_offdiag[i,j] = C_J(t_i, t_j) for the number operator',
              f'at Q = 0 this equals {N} * the C2 array element by element, exactly'],
      ['Q', 'sample'] + [c.format('c3') for c in cols],
      rows)

write(os.path.join(args.out_dir, f'{args.prefix}{args.tag}_times.csv'),
      prov + ['per-leg imaginary times and the generator step index'],
      ['k', 'step_index', 'tau'],
      [[str(k), str(int(ti[k])), args.fmt % tv[k]] for k in range(nt)])
