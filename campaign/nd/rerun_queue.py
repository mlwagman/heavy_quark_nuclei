"""Add the wide-trial brackets (scale 1.4) to datasets.csv and write queue_rerun.txt with every line that has
no result yet: brackets first, then the remaining N_d=5 lines, then N_d=7 (lambda=1 before lambda=1/2,2).
Lines from queue.txt keep their seeds; the new bracket lines get seeds 9200000+n.  Run from code/."""
import csv, os
rows = list(csv.DictReader(open('datasets.csv')))
names = {r['name'] for r in rows}
new = []
for base in ('nd5_LO_a0.03_lam1', 'nd7_LO_a0.0214286_lam1'):
    if base + '_wide' in names:
        continue
    r = dict(next(x for x in rows if x['name'] == base))
    r['name'] = base + '_wide'; r['scale'] = '1.4'
    r['spoila'] = repr(float(r['spoila'])*1.4**0.5); r['r2_T'] = repr(float(r['r2_T_opt'])*1.4)
    new.append(r)
if new:
    rows = rows[:6] + new + rows[6:]
    with open('datasets.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
XQ = [0.151, 0.378, 0.756, 1.51, 2.65, 3.78]
wide, n = [], 0
for r in [x for x in rows if x['name'].endswith('_wide')]:
    for iq, x in enumerate(XQ):
        Q = x/float(r['r2_T_opt'])**0.5
        for s in range(8):
            wide.append(f"{r['name']}_q{iq}_s{s}|--alpha_s {r['alpha']} --Nc {r['Nc']} --N_coord {r['Nc']} --OLO {r['OLO']} "
                        f"--mu {r['mu']} --mufac {r['lam']} --spoila {float(r['spoila']):.6f} --dtau_iMev {float(r['dtau']):.6g} "
                        f"--Q_mag {Q:.6g} --n_step 800 --count_skip 50 --n_walkers 2000 --seed {9200000 + n}\n")
            n += 1
done = lambda l: os.path.exists('../results/' + l.split('|')[0] + '.npz')
todo = [l for l in open('queue.txt') if not done(l)]
cmp_ = [l for l in todo if '_cmp_' in l]
n5 = [l for l in todo if l.startswith('nd5') and '_cmp_' not in l]
n7 = [l for l in todo if l.startswith('nd7') and '_cmp_' not in l]
n7 = [l for l in n7 if '_lam1_' in l] + [l for l in n7 if '_lam1_' not in l]
q = [l for l in wide if not done(l)] + cmp_ + n5 + n7
open('queue_rerun.txt', 'w').writelines(q)
print(f'new datasets {[r["name"] for r in new]}; wide {len(wide)} cmp {len(cmp_)} nd5 {len(n5)} nd7 {len(n7)} total {len(q)}')
