import glob, os, re, sys
import numpy as np
from scipy import stats
REF = 56.69
rows = {12: [], 16: []}
for s in range(201, 217):
    txt = open(f'/tmp/rep_{s}.txt').read()
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 6 and f[0] in ('12', '16') and f[4] == '+/-':
            rows[int(f[0])].append((s, float(f[3]), float(f[5])))
for m in (12, 16):
    r = rows[m]
    v = np.array([x[1] for x in r]); e = np.array([x[2] for x in r]); n = len(v)
    mean = v.mean(); scat = v.std(ddof=1)
    e_rms = np.sqrt((e ** 2).mean()); ratio = scat / e_rms
    rel = 1 / np.sqrt(2 * (n - 1))
    chi2 = (((v - mean) / e) ** 2).sum(); dof = n - 1
    p = 1 - stats.chi2.cdf(chi2, dof)
    lo, hi = scat * np.sqrt((n - 1) / stats.chi2.ppf([.975, .025], n - 1)) / e_rms
    print(f'== m = {m}   {n} independent seeds, 5000 walkers, Q=0.1 only')
    print('   ' + '  '.join(f'{x:.2f}' for x in v))
    print(f'   mean       {mean:.3f} +/- {scat/np.sqrt(n):.3f}  (ref {REF}, '
          f'{100*(mean/REF-1):+.2f}%, {(mean-REF)/(scat/np.sqrt(n)):+.2f}s)')
    print(f'   scatter    {scat:.3f}      within-run rms {e_rms:.3f}')
    print(f'   RATIO      {ratio:.3f} +/- {ratio*rel:.3f}   95% CI [{lo:.3f}, {hi:.3f}]')
    print(f'   chi2/dof   {chi2/dof:.2f} ({chi2:.1f}/{dof})   p = {p:.4f}')
    print()
# Nw dependence: global-mode model ratio^2 = 1 + (sig_g^2/sig_w^2) Nw
r1000 = 1.192
r5000 = rows[16] and np.array([x[1] for x in rows[16]]).std(ddof=1) / np.sqrt(
    (np.array([x[2] for x in rows[16]]) ** 2).mean())
print(f'Nw dependence (m=16): ratio(1000) = {r1000:.3f}, ratio(5000) = {r5000:.3f}')
print('  global-mode model predicts ratio^2-1 scales linearly with Nw:')
print(f'    (r^2-1) at 1000 = {r1000**2-1:.3f};  at 5000 = {r5000**2-1:.3f};  '
      f'ratio of those = {(r5000**2-1)/(r1000**2-1):.2f}  (model: 5.0)')
