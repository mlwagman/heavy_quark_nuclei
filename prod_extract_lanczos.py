"""
Production extraction using Mike's Lanczos/Korrelator code (Lanczos_twoleg.py),
replacing the diagonal-ratio plateau-average method.

For each currents_3pt_*.h5 in a directory, runs Lanczos_twoleg.py (subprocess,
so it uses exactly the same code path as standalone use) and aggregates the
per-file <file>_twoleg_Lanczos.csv outputs into <dir>/prod_ME.csv with the
same schema as the old prod_extract.py (Q, channel, F, F_err), so every
downstream script (prod_sanity.py, fit_final_forms.py, paper_figures.py,
paper_ME_suite.py) is unchanged.

Usage: python prod_extract_lanczos.py <dir> [--n_block 5 --n_boot 200]
"""
import subprocess, glob, re, os, sys, csv, argparse

p = argparse.ArgumentParser()
p.add_argument('dir')
p.add_argument('--n_block', type=int, default=5)
p.add_argument('--n_boot', type=int, default=200)
p.add_argument('--thresh', type=float, default=0.05)
p.add_argument('--max_tau_pts', type=int, default=None,
               help='Ritz rank cap (m_max = max_tau_pts/2). The Lanczos_twoleg '
                    'default of 10 (rank 5) leaves large excited-state '
                    'contamination for diffuse/weakly-bound states.')
p.add_argument('--out_name', default='prod_ME.csv')
p.add_argument('--estimator', choices=['final', 'stab'], default='final',
               help="'final' = highest Ritz rank (the historical behaviour); "
                    "'stab' = rank chosen by stability, with the residual rank "
                    "dependence carried in the error. See stability_select in "
                    "Lanczos_twoleg.py.")
args = p.parse_args()

files = sorted(f for f in glob.glob(os.path.join(args.dir, 'currents_3pt_*.h5'))
               if 'korr' not in f)
rows = []
failed = []
for fn in files:
    m = re.search(r'Qmag([0-9.]+)\.h5', os.path.basename(fn))
    if not m: continue
    Q = float(m.group(1))
    out_csv = fn[:-3] + '_twoleg_Lanczos.csv'
    cmd = [sys.executable, 'Lanczos_twoleg.py', '--currents_file', fn,
           '--n_block', str(args.n_block), '--n_boot', str(args.n_boot),
           '--thresh', str(args.thresh)]
    if args.max_tau_pts is not None:
        cmd += ['--max_tau_pts', str(args.max_tau_pts)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__) or '.')
    if r.returncode != 0 or not os.path.exists(out_csv):
        print(f"  FAILED Q={Q}: {fn}")
        print(r.stdout[-800:]); print(r.stderr[-800:])
        failed.append((Q, fn))
        continue
    with open(out_csv) as fh:
        for row in csv.DictReader(fh):
            if args.estimator == 'stab' and row.get('F_stab') not in (None, ''):
                F, Ferr = row['F_stab'], row['F_stab_err']
            else:
                F, Ferr = row['F_final'], row['F_err']
            rows.append(dict(Q=Q, channel=row['current'], F=F, F_err=Ferr))
    print(f"  Q={Q}: OK ({os.path.basename(fn)[:50]}...)")

out = os.path.join(args.dir, args.out_name)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['Q', 'channel', 'F', 'F_err'])
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r['Q'], r['channel'])):
        w.writerow(r)
print(f"wrote {out} ({len(rows)} rows, {len(failed)} file(s) failed)")
if failed:
    print("FAILED FILES:", failed)
