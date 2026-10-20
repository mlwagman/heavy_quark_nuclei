"""
Variationally tune the trial wavefunction size (--spoila) for a GFMC dataset.

The GFMC trial is a Coulomb wavefunction whose size comes from
    a0 = spoila * 2 / VB(Rstar)
with Rstar FIXED, so a0 does not track the true ground state of the MRS
potential. The mismatch reaches +15%, and because finite-tau projection cannot
remove it the extracted radius inherits a large bias (measured: +18.7% at
NLO alpha=0.3, cured to +0.22% by tuning spoila).

<H> at tau=0 is a pure VMC expectation value, so by the variational principle
the best trial MINIMISES it. That needs no benchmark, which is the point: the
six MRS baryon datasets have no exact solver (mrs_numerov.py is 2-body,
svm_baryon.py is Coulomb-only).

Runs are deliberately tiny (n_step 4) -- only the tau=0 energy is used.

Usage:
  python optimize_spoila.py --OLO LO --alpha_s 0.2 --N_coord 3
  python optimize_spoila.py --OLO NLO --alpha_s 0.3 --N_coord 2 --exact 112.90
"""
import argparse, subprocess, sys, os, re, glob, tempfile
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--OLO', default='LO')
ap.add_argument('--alpha_s', type=float, required=True)
ap.add_argument('--N_coord', type=int, required=True)
ap.add_argument('--grid', default='0.80,0.85,0.90,0.95,1.00,1.05')
ap.add_argument('--n_walkers', type=int, default=2000)
ap.add_argument('--exact', type=float, default=None,
                help='exact <r^2>_rel, if known, to cross-check the answer')
a = ap.parse_args()

MRS = 'heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py'
CSKIP = 5 if a.N_coord == 2 else 10
DTAU = 0.5 if a.N_coord == 2 else 0.3
env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

tmp = tempfile.mkdtemp(prefix='spoila_')
grid = [float(x) for x in a.grid.split(',')]
print(f'variational spoila scan: MRS {a.OLO} alpha_s={a.alpha_s} '
      f'N_coord={a.N_coord}')
print(f"{'spoila':>8} {'<H>(tau=0)':>13}")
res = []
for sp in grid:
    cmd = [sys.executable, MRS, '--n_step', '4', '--n_walkers', str(a.n_walkers),
           '--dtau_iMev', str(DTAU), '--count_skip', str(CSKIP),
           '--nf', '0', '--Nc', '3', '--N_coord', str(a.N_coord),
           '--alpha_s', str(a.alpha_s), '--OLO', a.OLO,
           '--mQ', '1.0', '--mu', '2.0', '--mufac', '1.0', '--Rstar', '0.35',
           '--spoila', str(sp), '--wavefunction', 'compact',
           '--potential', 'full', '--color', '1x1',
           '--Q_dir', '1', '0', '0', '--Q_mag', '0.0',
           '--seed', '0', '--outdir', tmp + '/']
    if a.N_coord == 3:
        cmd += ['--masses', '1', '1', '1']
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    m = re.search(r'tau=0\.000.*?<H>=([+-][0-9.]+)', out)
    for f in glob.glob(tmp + '/*.h5'):
        os.remove(f)
    if not m:
        print(f'{sp:>8} {"no <H>":>13}')
        continue
    H = float(m.group(1))
    res.append((sp, H))
    print(f'{sp:>8} {H:13.6f}')

if len(res) < 3:
    raise SystemExit('not enough points to locate a minimum')
x = np.array([r[0] for r in res]); y = np.array([r[1] for r in res])
i = int(np.argmin(y))
if i == 0 or i == len(x) - 1:
    print(f'\nWARNING: minimum at the EDGE of the grid (spoila={x[i]}). '
          f'Widen --grid.')
    best = x[i]
else:
    c = np.polyfit(x[i-1:i+2], y[i-1:i+2], 2)
    best = -c[1] / (2 * c[0])
print(f'\noptimal spoila = {best:.4f}   (grid min at {x[i]}, '
      f'<H> = {y[i]:.6f})')
if a.exact:
    print(f'cross-check: exact <r^2>_rel = {a.exact} -> E0 comparison only; '
          f'trial/true matching is the independent route.')
