"""
Plan the N_d campaign: for each dataset, mu_cut = 3.4 Lambda_d(2-loop), the VMC-optimal compact trial
(vmc_trial.py), dtau = 0.0125/|E_T|, Q = x/sqrt(r2_T), and the Numerov meson radius.
Writes datasets.csv and queue.txt (one line per (dataset, Q, walker split)).

Grid: N_d alpha_d = 0.003 .. 0.21 (the N_d = 3 grid alpha_d = 0.001 .. 0.07), so Lambda_d/m and mu_cut are
the same for every N_d.  N_d = 3 validation datasets (solver references) and compact-start brackets first.
"""
import subprocess, re, sys, os, csv
from concurrent.futures import ThreadPoolExecutor
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
NA = [0.003, 0.009, 0.03, 0.06, 0.09, 0.15, 0.21]
XQ = [0.151, 0.378, 0.756, 1.51, 2.65, 3.78]
NSPLIT, NW, NSTEP, CS, EDT = 4, 4000, 800, 50, 0.0125

ds = []   # (name, Nc, alpha, OLO, lam, trial_scale)
val = [(3, 0.05, 'LO', 1.0), (3, 0.001, 'LO', 1.0), (3, 0.07, 'NLO', 1.0), (3, 0.01, 'NLO', 0.5)]
for nc, al, O, lam in val:
    ds.append((f'nd3_{O}_a{al:g}_lam{lam:g}', nc, al, O, lam, 1.0))
ds.append(('nd5_LO_a0.03_lam1_cmp', 5, 0.03, 'LO', 1.0, 0.7))
ds.append(('nd7_LO_a0.0214286_lam1_cmp', 7, round(0.15/7, 7), 'LO', 1.0, 0.7))
for nc in (5, 7):
    for na in NA:
        al = round(na/nc, 7)
        for O in ('LO', 'NLO'):
            for lam in (0.5, 1.0, 2.0):
                ds.append((f'nd{nc}_{O}_a{al:g}_lam{lam:g}', nc, al, O, lam, 1.0))

def vmc(d):
    name, nc, al, O, lam, sc = d
    t = subprocess.run([PY, f'{HERE}/vmc_trial.py', '--Nc', str(nc), '--alpha', str(al), '--OLO', O, '--mufac', str(lam),
                        '--nsamp', '100000'], capture_output=True, text=True, cwd=HERE, env=dict(os.environ, JAX_PLATFORMS='cpu'))
    o = t.stdout
    try:
        mu = float(re.search(r'mu_cut=([0-9.eE+-]+)', o).group(1))
        g = {k: float(re.search(rf'{k}=([0-9.eE+-]+)', o).group(1)) for k in ('E_T', 'r2_T', 'spoila_opt', 'a0_opt', 'cN')}
    except AttributeError:
        sys.exit(f'{name}: vmc failed\n{o}\n{t.stderr[-2000:]}')
    m = None
    if nc != 3 or True:
        tm = subprocess.run([PY, f'{HERE}/mrs_numerov.py', '--Nc', str(nc), '--OLO', O, '--alpha_s', str(al), '--mu', repr(mu),
                             '--mufac', str(lam)], capture_output=True, text=True, cwd=HERE, env=dict(os.environ, JAX_PLATFORMS='cpu'))
        mm = re.search(r'charge radius <r\^2>_rel/4 = ([0-9.eE+-]+)', tm.stdout)
        m = float(mm.group(1)) if mm else float('nan')
    sp = g['spoila_opt']*sc**0.5
    print(f"{name}: mu {mu:.6g} spoila {sp:.6g} r2_T {g['r2_T']*sc:.6g} E_T {g['E_T']:.6g} meson {m}", flush=True)
    return dict(name=name, Nc=nc, alpha=al, OLO=O, lam=lam, scale=sc, mu=mu, spoila=sp, r2_T=g['r2_T']*sc,
                r2_T_opt=g['r2_T'], E_T=g['E_T'], dtau=EDT/abs(g['E_T']), meson_r2=m)

# the per-N sample caches must exist before the threads start
for nc in sorted({d[1] for d in ds}):
    subprocess.run([PY, f'{HERE}/vmc_trial.py', '--Nc', str(nc), '--alpha', '0.03', '--nsamp', '100000'],
                   capture_output=True, cwd=HERE, env=dict(os.environ, JAX_PLATFORMS='cpu'))
with ThreadPoolExecutor(int(os.environ.get('NPROC', 4))) as ex:
    rows = list(ex.map(vmc, ds))
with open('datasets.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
n = 0
with open('queue.txt', 'w') as f:
    for r in rows:
        for iq, x in enumerate(XQ):
            Q = x/r['r2_T_opt']**0.5
            for s in range(NSPLIT):
                f.write(f"{r['name']}_q{iq}_s{s}|--alpha_s {r['alpha']} --Nc {r['Nc']} --N_coord {r['Nc']} --OLO {r['OLO']} "
                        f"--mu {r['mu']!r} --mufac {r['lam']} --spoila {r['spoila']:.6f} --dtau_iMev {r['dtau']:.6g} "
                        f"--Q_mag {Q:.6g} --n_step {NSTEP} --count_skip {CS} --n_walkers {NW} --seed {9000000 + n}\n")
                n += 1
print(f'{len(rows)} datasets, {n} queue lines')
