"""MRS at mu_cut = 3.4 Lambda_d(2-loop) (= 1 GeV / Lambda_QCD), lambda = mufac in {1/2, 1, 2}.
Writes code/lam_queue_{1,2}.txt (baryons, split over two nodes) and results/numerov_lam.csv
(exact meson radii for every alpha, order, lambda)."""
import subprocess, re, sys, math
PY = sys.executable
RATIO = 3.4
b0 = 11 / (4 * math.pi); b1 = 102 / (16 * math.pi**2)
def mucut(a):
    # 3.4 x the 2-loop Lambda_d (the pole of the NLO coupling), same cutoff at LO and NLO,
    # like the paper's 1 GeV ~ 3.4 Lambda_QCD at every order
    from scipy.optimize import brentq
    L1 = 1 / (b0 * a)
    L = brentq(lambda L: 1/(b0*L) - b1*math.log(L)/(b0**3*L*L) - a, 0.5*L1, L1)
    return RATIO * math.exp(-L / 2)
def numerov(a, O, mu, lam):
    out = subprocess.run([PY, 'mrs_numerov.py', '--OLO', O, '--alpha_s', str(a), '--mu', f'{mu:.6g}', '--mufac', str(lam)],
                         capture_output=True, text=True, cwd='code').stdout
    a0 = float(re.search(r'trial a0 \(code\) = ([0-9.eE+-]+)', out).group(1))
    r2 = float(re.search(r'charge radius <r\^2>_rel/4 = ([0-9.eE+-]+)', out).group(1))
    return a0, r2
aref = (4 * 26.89 / 3) ** 0.5
AL = [0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 0.3, 0.5]
csv = ['alpha,OLO,lam,mu_cut,r2_meson,r2_coul_meson']
jobs = []; seed = 5001
for a in AL:
    mu = mucut(a)
    for O in ('LO', 'NLO'):
        for lam in (0.5, 1.0, 2.0):
            a0, r2 = numerov(a, O, mu, lam)
            csv.append(f'{a},{O},{lam},{mu:.6g},{r2:.8g},{27/(16*a*a):.8g}')
            print(f'alpha={a:<6} {O:3} lam={lam:<4} mu_cut={mu:<10.4g} r2/Coul={r2/(27/(16*a*a)):.4f}', flush=True)
            if O == 'NLO' and lam != 1.0: continue
            k = (4 * r2 / 3) ** 0.5 / aref; sp = (4 * r2 / 3) ** 0.5 / a0
            tag = {0.5: 'lamh', 1.0: 'lam1', 2.0: 'lam2'}[lam]
            jobs.append(f'mrs_{O.lower()}_baryon_a{a}_{tag}|baryon|-|{k:.6g}|--alpha_s {a} --OLO {O} --N_coord 3 '
                        f'--spoila OPT{sp:.4f} --n_step 100 --n_walkers 4000 --dtau_iMev {0.3*k*k:.6g} '
                        f'--count_skip 10 --mu {mu:.6g} --mufac {lam} --seed {seed}')
            seed += 1
open('results/numerov_lam.csv', 'w').write('\n'.join(csv) + '\n')
half = (len(jobs) + 1) // 2
open('code/lam_queue_1.txt', 'w').write('\n'.join(jobs[:half]) + '\n')
open('code/lam_queue_2.txt', 'w').write('\n'.join(jobs[half:]) + '\n')
print(len(jobs), 'baryon datasets in two queues')
