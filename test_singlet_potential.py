r"""Equivalence test for the color-vector-free (``--no_color_vectors``) GFMC path.

A :math:`q\bar q` pair has a unique color singlet, :math:`\delta_{ij}/\sqrt{N_I}`, and so does
an :math:`N_I=3` :math:`qqq` state, :math:`\epsilon_{ijk}/\sqrt{2N_I}`.  Either one is an *exact*
eigenvector of the pairwise potential that ``afdmc_lib_col.make_pairwise_potential`` builds,
because the symmetric and octet projectors annihilate it.  The :math:`3^{N_{\rm coord}}`-component
vector the generators carry therefore never mixes, and only its eigenvalue

.. math::  \lambda(R) \;=\; \sum_{i<j} v_{ij}(|R_i - R_j|), \qquad
           v = \texttt{OSing}\ (q\bar q), \quad \texttt{OA}\ (qq)

enters anything.  ``make_pairwise_singlet_potential`` propagates that scalar on a trivial
1-dimensional color space, which is both cheaper and -- the point of this test -- numerically
identical.

The identity is exact rather than statistical, so this is a strong test.  Every random draw in
the generator (Metropolis initialization, the forward/backward heatbath, the accept/reject
uniforms) is taken in coordinate space only, so at a fixed ``--seed`` the two modes walk the
*same* walkers through the *same* configurations.  They also apply the same truncated propagator
:math:`1 - (d\tau/2)V + (d\tau^2/8)V^2`, because ``compute_VS`` is written against the shape of
``S`` rather than against ``NI``.  Agreement should therefore be at roundoff -- the only
difference is the order floating-point sums happen in -- not at the few-percent level that
"consistent within errors" would allow.

Each configuration is run twice, with and without the flag, and *every* dataset in the resulting
HDF5 files is compared entry by entry.  The wall-clock of both runs is reported, so this doubles
as the speedup measurement.

Usage::

    python3 test_singlet_potential.py                  # meson + baryon + MRS meson, small
    python3 test_singlet_potential.py --n_walkers 200 --n_step 40
    python3 test_singlet_potential.py --only mrs_meson
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, 'mike_20260824', 'code')
PY = os.environ.get('PY', '/Users/mwagman/opt/anaconda3/bin/python3')

FO = 'heavy_quark_nuclei_gfmc_boosted_FV_currents.py'
MRS = 'heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py'

# (label, script, N_coord, physics args).  Q != 0 on purpose: at Q = 0 the current is
# N_coord * identity and commutes with H, which would hide an error in the 3pt legs.
CONFIGS = [
    ('meson', FO, 2, ['--nf', '0', '--Nc', '3', '--N_coord', '2', '--masses', '1', '-1',
                      '--alpha', '0.3', '--spoila', '1']),
    ('baryon', FO, 3, ['--nf', '0', '--Nc', '3', '--N_coord', '3', '--masses', '1', '1', '1',
                       '--alpha', '0.3', '--spoila', '1']),
    ('mrs_meson', MRS, 2, ['--nf', '0', '--Nc', '3', '--N_coord', '2', '--masses', '1', '-1',
                           '--alpha_s', '0.3', '--OLO', 'LO', '--spoila', '0.9241',
                           '--mQ', '1.0', '--mu', '2.0', '--mufac', '1.0', '--Rstar', '0.35']),
]

ENV = dict(os.environ,
           JAX_PLATFORMS='cpu',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def run(script, args, outdir, no_color):
    """Run one generator; return wall-clock seconds."""
    os.makedirs(outdir, exist_ok=True)
    cmd = [PY, script] + args + ['--outdir', outdir + os.sep]
    if no_color:
        cmd.append('--no_color_vectors')
    t0 = time.perf_counter()
    with open(os.path.join(outdir, 'log.txt'), 'w') as log:
        r = subprocess.run(cmd, cwd=CODE, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"    FAILED (exit {r.returncode}); tail of {outdir}/log.txt:")
        with open(os.path.join(outdir, 'log.txt')) as log:
            for line in log.readlines()[-15:]:
                print('      ' + line.rstrip())
        return None
    return dt


def step_times(outdir):
    """Per-step wall-clock the generator prints, as a list of floats.

    Total run time is dominated by fixed overhead at test sizes (jax tracing and
    compilation, Metropolis thermalization, matplotlib), so the honest measure of the
    saving is the steady-state cost of a GFMC step.  The median over steps past the
    first discards the compile-included first step.
    """
    fn = os.path.join(outdir, 'log.txt')
    ts = []
    with open(fn) as f:
        for line in f:
            if line.startswith('computed step in '):
                try:
                    ts.append(float(line.split()[-1].rstrip('s')))
                except ValueError:
                    pass
    return ts


def median_step(outdir):
    ts = step_times(outdir)
    return float(np.median(ts[1:])) if len(ts) > 1 else (ts[0] if ts else float('nan'))


def datasets(fn):
    """{name: array} for every dataset in an HDF5 file, recursively."""
    out = {}
    with h5py.File(fn, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
        attrs = dict(f.attrs)
    return out, attrs


def compare(fn_a, fn_b, tol):
    """Max scale-relative difference over all datasets. Returns (worst_name, worst_rel, rows)."""
    da, aa = datasets(fn_a)
    db, ab = datasets(fn_b)
    assert aa.get('color_vectors') == 'on', f"expected color_vectors=on in {fn_a}, got {aa}"
    assert ab.get('color_vectors') == 'off', f"expected color_vectors=off in {fn_b}, got {ab}"
    rows, worst, worst_name = [], 0.0, '-'
    for name in sorted(set(da) | set(db)):
        if name not in da or name not in db:
            rows.append((name, 'MISSING', np.nan))
            worst, worst_name = np.inf, name
            continue
        a, b = np.asarray(da[name]), np.asarray(db[name])
        if a.shape != b.shape:
            rows.append((name, f'SHAPE {a.shape} vs {b.shape}', np.nan))
            worst, worst_name = np.inf, name
            continue
        if a.dtype.kind not in 'fci':
            continue
        scale = np.max(np.abs(b)) if b.size else 0.0
        if scale == 0.0:
            rel = float(np.max(np.abs(a))) if a.size else 0.0
        else:
            rel = float(np.max(np.abs(a - b)) / scale)
        rows.append((name, f'{a.shape}', rel))
        if rel > worst:
            worst, worst_name = rel, name
    return worst_name, worst, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_walkers', type=int, default=60)
    p.add_argument('--n_step', type=int, default=20)
    p.add_argument('--count_skip', type=int, default=5)
    p.add_argument('--dtau', type=float, default=0.5)
    p.add_argument('--Q_mag', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tol', type=float, default=1e-9)
    p.add_argument('--only', type=str, default=None, help='run just this config label')
    p.add_argument('--workdir', type=str,
                   default=os.path.join(HERE, 'singlet_equiv_test'))
    p.add_argument('--keep', action='store_true', help='do not delete the work directory first')
    args = p.parse_args()

    if not args.keep and os.path.isdir(args.workdir):
        shutil.rmtree(args.workdir)

    common = ['--n_step', str(args.n_step), '--n_walkers', str(args.n_walkers),
              '--dtau_iMev', str(args.dtau), '--count_skip', str(args.count_skip),
              '--wavefunction', 'compact', '--potential', 'full', '--color', '1x1',
              '--Q_dir', '1', '0', '0', '--Q_mag', str(args.Q_mag), '--seed', str(args.seed)]

    print(f"n_walkers={args.n_walkers} n_step={args.n_step} count_skip={args.count_skip} "
          f"dtau={args.dtau} Q={args.Q_mag} seed={args.seed}  tol={args.tol:g}")

    failures, timings = [], []
    for label, script, n_coord, phys in CONFIGS:
        if args.only and label != args.only:
            continue
        print(f"\n{'='*94}\n{label}  ({script}, N_coord={n_coord})\n{'='*94}")
        dirs = {}
        for mode, flag in [('color', False), ('singlet', True)]:
            d = os.path.join(args.workdir, f'{label}_{mode}')
            print(f"  running {mode:>7} ...", end='', flush=True)
            dt = run(script, common + phys, d, flag)
            if dt is None:
                failures.append(f'{label}/{mode} crashed')
                break
            print(f" {dt:7.1f}s")
            dirs[mode] = (d, dt)
        if len(dirs) != 2:
            continue

        t_col, t_sing = dirs['color'][1], dirs['singlet'][1]
        timings.append((label, t_col, t_sing,
                        median_step(dirs['color'][0]), median_step(dirs['singlet'][0])))

        for pattern in ['currents_3pt_*.h5', 'Hammys_*.h5']:
            fa = glob.glob(os.path.join(dirs['color'][0], pattern))
            fb = glob.glob(os.path.join(dirs['singlet'][0], pattern))
            if not fa or not fb:
                failures.append(f'{label}: no {pattern}')
                continue
            if pattern.startswith('currents'):
                name, worst, rows = compare(fa[0], fb[0], args.tol)
                print(f"\n  {'dataset':<24} {'shape':<22} {'max rel diff':>14}")
                for nm, shp, rel in rows:
                    flag = '  <-- FAIL' if (not np.isfinite(rel) or rel > args.tol) else ''
                    rs = f'{rel:14.3e}' if np.isfinite(rel) else f'{"n/a":>14}'
                    print(f"  {nm:<24} {str(shp):<22} {rs}{flag}")
            else:
                # Hammys has no attrs to check; compare raw
                da, _ = datasets(fa[0])
                db, _ = datasets(fb[0])
                worst, name = 0.0, '-'
                for k in sorted(set(da) & set(db)):
                    a, b = np.asarray(da[k]), np.asarray(db[k])
                    if a.shape != b.shape or a.dtype.kind not in 'fci':
                        continue
                    s = np.max(np.abs(b))
                    rel = float(np.max(np.abs(a - b)) / s) if s else 0.0
                    if rel > worst:
                        worst, name = rel, k
                print(f"  Hammys: worst dataset {name!r} at {worst:.3e}")
            if not np.isfinite(worst) or worst > args.tol:
                failures.append(f'{label}/{pattern}: {name} differs by {worst:.3e}')
            else:
                print(f"  -> {pattern}: all datasets agree, worst {worst:.3e} "
                      f"({name!r})  [tol {args.tol:g}]")

    print(f"\n{'='*94}\nTIMING\n{'='*94}")
    print(f"  {'':<14} {'----- whole run -----':>32} {'--- median GFMC step ---':>34}")
    print(f"  {'config':<14} {'color vecs':>12} {'singlet':>12} {'speedup':>7} "
          f"{'color vecs':>13} {'singlet':>12} {'speedup':>8}")
    for label, t_col, t_sing, s_col, s_sing in timings:
        sp = f'{s_col/s_sing:7.2f}x' if s_sing else f'{"n/a":>8}'
        print(f"  {label:<14} {t_col:11.1f}s {t_sing:11.1f}s {t_col/t_sing:6.2f}x "
              f"{s_col:12.3f}s {s_sing:11.3f}s {sp}")
    print("\n  Whole-run times include fixed overhead -- jax tracing/compilation, Metropolis")
    print("  thermalization, matplotlib -- which does not grow with n_step, so at production")
    print("  n_step the median-step ratio is the one that governs. It is measured past the")
    print("  first step, so it excludes compilation.")

    print(f"\n{'='*94}")
    if failures:
        print("FAIL")
        for f in failures:
            print('  ' + f)
        return 1
    print("PASS: the color-vector and scalar-singlet paths are numerically identical.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
