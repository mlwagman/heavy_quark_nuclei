#!/usr/bin/env python3
"""Collect the tau=0 variational energy from a spoila scan and locate the optimum.

<H>_VMC = mean_walkers Hammys[0], valid because Ws_leg1[0] == 1 identically (no propagation
has happened at t=0), so the walker mean is <psi|H|psi>/<psi|psi> for the sampled |psi|^2.

Reports, per config:
  * <H> +/- sem at each spoila, and the ratio to the production (radius-matched) spoila
  * the variational optimum from a parabola through the three lowest points
  * sigma(E_L)/|<H>|, the local-energy relative spread -- 0 for an exact eigenstate, so it is an
    independent trial-quality measure, and its minimum need NOT coincide with the energy minimum
"""
import argparse, glob, os, re, sys
import numpy as np
import h5py

p = argparse.ArgumentParser()
p.add_argument('--dir', required=True, help='scan directory holding one subdir per config')
p.add_argument('--production', default='', help='name=spoila,name=spoila,... radius-matched values')
p.add_argument('--configs', default='', help='comma list to restrict to')
a = p.parse_args()

prod = {}
for tok in a.production.split(','):
    if '=' in tok:
        k, v = tok.split('=', 1)
        prod[k.strip()] = float(v)

SP = re.compile(r'_spoila([0-9.eE+-]+?)_')

def collect(d):
    out = []
    for f in glob.glob(os.path.join(d, 'Hammys_*.h5')):
        m = SP.search(os.path.basename(f))
        if not m:
            continue
        try:
            with h5py.File(f, 'r') as h:
                H = np.array(h['Hammys'])
                W = np.array(h['Ws_leg1'])
        except Exception as e:
            print(f'    (skip {os.path.basename(f)}: {e})', file=sys.stderr)
            continue
        h0 = np.real(H[0]); w0 = np.real(W[0])
        if not np.allclose(w0, 1.0):
            print(f'    (WARNING {os.path.basename(f)}: Ws_leg1[0] != 1, '
                  f'min={w0.min():.4g} max={w0.max():.4g} -- not a pure VMC slice)')
        n = len(h0)
        # Walker index is Metropolis chain order, so the samples are serially correlated at
        # n_skip 10.  Quote a BLOCKED error (20 contiguous blocks) next to the naive sem; their
        # ratio is sqrt(2 tau_int) and says directly how badly the naive one lies.
        nb = 20
        k = n // nb
        blocks = h0[:nb * k].reshape(nb, k).mean(axis=1)
        sem_blk = blocks.std(ddof=1) / np.sqrt(nb)
        sem_nai = h0.std(ddof=1) / np.sqrt(n)
        # Robust spread: sigma(E_L) is tail-dominated and MRS has genuine fat tails, so also
        # report the normal-equivalent sigma from the IQR, which ignores the extreme walkers.
        q1, q3 = np.percentile(h0, [25, 75])
        iqr_sig = (q3 - q1) / 1.3489795
        out.append((float(m.group(1)), h0.mean(), sem_nai, sem_blk, h0.std(ddof=1),
                    iqr_sig, np.median(h0), n))
    return sorted(out)

def parabola_min(xs, ys):
    """Vertex of the parabola through the 3 points bracketing the lowest y."""
    i = int(np.argmin(ys))
    if i == 0 or i == len(xs) - 1:
        return None, 'minimum at grid edge -- widen the scan'
    x = np.array(xs[i - 1:i + 2], float); y = np.array(ys[i - 1:i + 2], float)
    c = np.polyfit(x, y, 2)
    if c[0] <= 0:
        return None, 'not convex at the minimum'
    return -c[1] / (2 * c[0]), ''

names = [n for n in sorted(os.listdir(a.dir))
         if os.path.isdir(os.path.join(a.dir, n))]
if a.configs:
    want = [c.strip() for c in a.configs.split(',')]
    names = [n for n in names if n in want]

summary = []
for name in names:
    rows = collect(os.path.join(a.dir, name))
    if len(rows) < 2:
        print(f'\n== {name}: {len(rows)} point(s), skipping'); continue
    sp0 = prod.get(name)
    print(f'\n{"=" * 78}\n{name}' + (f'   production spoila = {sp0:.6f}' if sp0 else '')
          + f'   ({len(rows)} points)\n{"=" * 78}')
    hdr = (f'{"spoila":>10s} {"ratio":>7s} {"<H>_tau0":>14s} {"sem_blk":>9s} {"blk/nai":>8s} '
           f'{"sig/|H|":>8s} {"IQRsig/|H|":>11s}')
    print(hdr); print('-' * len(hdr))
    xs, ys, es, rs = [], [], [], []
    best = min(r[1] for r in rows)
    bestr = min(r[5] / abs(r[1]) for r in rows)
    for sp, mean, sem_n, sem_b, sd, iqr, med, n in rows:
        ratio = sp / sp0 if sp0 else float('nan')
        mark = '  <-- Emin' if mean == best else ''
        if iqr / abs(mean) == bestr:
            mark += '  <-- IQRmin'
        star = ' *prod*' if sp0 and abs(sp - sp0) / sp0 < 2e-3 else ''
        infl = sem_b / sem_n if sem_n > 0 else float('nan')
        print(f'{sp:10.6f} {ratio:7.3f} {mean:14.8f} {sem_b:9.2e} {infl:8.2f} '
              f'{sd / abs(mean):8.4f} {iqr / abs(mean):11.4f}{star}{mark}')
        xs.append(sp); ys.append(mean); es.append(sem_b); rs.append(iqr / abs(mean))
    # variational sanity: no point may lie below the lowest energy by more than its error
    lo = int(np.argmin(ys))
    print(f'\n  lowest energy {ys[lo]:.8f} +/- {es[lo]:.2e} at spoila {xs[lo]:.6f}')
    iq = int(np.argmin(rs))
    print(f'  IQR-spread minimum at spoila {xs[iq]:.6f}  (ratio '
          + (f'{xs[iq] / sp0:.3f})' if sp0 else 'n/a)'))
    opt, why = parabola_min(xs, ys)
    if opt is None:
        print(f'\n  variational optimum: NOT BRACKETED ({why})')
        summary.append((name, sp0, None, None))
    else:
        print(f'\n  variational optimum  spoila* = {opt:.6f}')
        if sp0:
            dev = 100 * (sp0 / opt - 1)
            print(f'  production / optimum = {sp0 / opt:.4f}   ({dev:+.2f}% in spoila, '
                  f'{2 * dev:+.2f}% in a0^2 ~ <r^2>)')
        summary.append((name, sp0, opt, None))

print(f'\n{"=" * 78}\nSUMMARY   radius-matched spoila vs tau=0 variational optimum\n{"=" * 78}')
print(f'{"config":22s} {"production":>11s} {"VMC opt":>10s} {"prod/opt":>9s}')
for name, sp0, opt, _ in summary:
    s0 = f'{sp0:.6f}' if sp0 else '-'
    so = f'{opt:.6f}' if opt else 'n/a'
    rr = f'{sp0 / opt:.4f}' if (sp0 and opt) else '-'
    print(f'{name:22s} {s0:>11s} {so:>10s} {rr:>9s}')
