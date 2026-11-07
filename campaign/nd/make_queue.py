"""Queue from datasets.csv: one line per (dataset, Q, walker split), in priority order:
validation and brackets, N_d=5 lambda=1, N_d=5 lambda=1/2,2, N_d=7 lambda=1, N_d=7 lambda=1/2,2.
Usage: python make_queue.py datasets.csv queue.txt [NSPLIT=8] [NW=2000]"""
import csv, sys
ds, out = sys.argv[1], sys.argv[2]
NSPLIT = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NW = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
XQ = [0.151, 0.378, 0.756, 1.51, 2.65, 3.78]; NSTEP, CS = 800, 50
rows = list(csv.DictReader(open(ds)))
def prio(r):
    N, lam = int(r['Nc']), float(r['lam'])
    if N == 3 or float(r['scale']) != 1.0: return 0
    return {5: 1, 7: 3}[N] + (0 if lam == 1.0 else 1)
rows.sort(key=prio)   # stable: keeps the alpha/order sequence inside each group
n = 0
with open(out, 'w') as f:
    for r in rows:
        for iq, x in enumerate(XQ):
            Q = x/float(r['r2_T_opt'])**0.5
            for s in range(NSPLIT):
                f.write(f"{r['name']}_q{iq}_s{s}|--alpha_s {r['alpha']} --Nc {r['Nc']} --N_coord {r['Nc']} --OLO {r['OLO']} "
                        f"--mu {r['mu']} --mufac {r['lam']} --spoila {float(r['spoila']):.6f} --dtau_iMev {float(r['dtau']):.6g} "
                        f"--Q_mag {Q:.6g} --n_step {NSTEP} --count_skip {CS} --n_walkers {NW} --seed {9100000 + n}\n")
                n += 1
print(f'{len(rows)} datasets, {n} lines -> {out}')
