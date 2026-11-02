"""
Exact references for the paper, for every (alpha, order, lambda) of the MRS scan:
variational baryon (svm_baryon.py N=3, two independent bases), meson (Numerov for <r^2>,
svm_baryon.py N=2 for <K> and F(q)), and F(q)/F(0) on a grid in q r, plus the GFMC Q values.

Usage (from the directory holding the generator): python paper_solver.py table.csv out.csv ff.csv queue1 [queue2 ...]
"""
import csv, os, re, subprocess, sys, tempfile
from multiprocessing import Pool
import numpy as np

PY = sys.executable
RE = {k: re.compile(p) for k, p in dict(E0=r'E0 = ([0-9.eE+-]+)', r2=r'<r\^2>_ch = ([0-9.eE+-]+)', K=r'<K> = ([0-9.eE+-]+)').items()}
XG = np.geomspace(0.03, 20, 45)                       # grid in q sqrt(<r^2>)

def svm(a, O, lam, mu, asc, N, nbasis, seed, qs=()):
    cmd = [PY, 'svm_baryon.py', '--alpha', str(a), '--potential', 'mrs', '--OLO', O, '--mu', str(mu), '--mufac', str(lam),
           '--ascale', str(asc), '--N', str(N), '--nbasis', str(nbasis), '--seed', str(seed)] + (['--symmetrize'] if N == 3 else [])
    out = {}
    with tempfile.TemporaryDirectory() as td:
        if len(qs): cmd += ['--qs', ','.join(f'{q:.8g}' for q in qs), '--out', f'{td}/f.csv']
        t = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        for k, p in RE.items(): out[k] = float(p.search(t).group(1))
        if len(qs):
            rows = [l.split(',') for l in open(f'{td}/f.csv') if l[0] not in 'Q#']
            out['F'] = [(float(q), float(f)) for q, f in rows if float(q) > 0]
    return out

def numerov(a, O, lam, mu):
    t = subprocess.run([PY, 'mrs_numerov.py', '--OLO', O, '--alpha_s', str(a), '--mu', str(mu), '--mufac', str(lam)],
                       capture_output=True, text=True, check=True).stdout
    return float(re.search(r'charge radius <r\^2>_rel/4 = ([0-9.eE+-]+)', t).group(1))

def one(job):
    a, O, lam, mu, r2_old, rm_old, Qd = job
    asc = (rm_old*16*a*a/27)**0.5
    b = svm(a, O, lam, mu, asc, 3, 600, 1, sorted(set(XG/r2_old**0.5) | set(Qd)))
    c = svm(a, O, lam, mu, asc, 3, 300, 2)
    m = svm(a, O, lam, mu, asc, 2, 250, 1, XG/rm_old**0.5)
    return a, O, lam, mu, b, c, m, numerov(a, O, lam, mu)

if __name__ == '__main__':
    table, out_csv, ff_csv, queues = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    Qd = {}
    for q in queues:
        for line in open(q):
            if not line.strip(): continue
            f = line.split('|')
            k = float(f[3]); g = re.search(r'--alpha_s ([0-9.]+) --OLO (\w+).*--mufac ([0-9.]+)', f[4])
            Qd[(float(g[1]), g[2], float(g[3]))] = [x/k for x in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)]
    jobs = [(float(r['alpha']), r['OLO'], float(r['lam']), r['mu_cut'], float(r['r2']), float(r['r2_meson']),
             Qd.get((float(r['alpha']), r['OLO'], float(r['lam'])), [])) for r in csv.DictReader(open(table))]
    os.environ.update(JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    with Pool(int(os.environ.get('NPROC', 30))) as pool: res = pool.map(one, jobs, chunksize=1)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow('alpha OLO lam mu_cut E0 r2 K E0_chk r2_chk r2_meson E0_meson K_meson r2_meson_svm'.split())
        for a, O, lam, mu, b, c, m, rn in res:
            w.writerow([a, O, lam, mu, f"{b['E0']:.8g}", f"{b['r2']:.8g}", f"{b['K']:.8g}", f"{c['E0']:.8g}", f"{c['r2']:.8g}",
                        f'{rn:.8g}', f"{m['E0']:.8g}", f"{m['K']:.8g}", f"{m['r2']:.8g}"])
    with open(ff_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow('alpha OLO lam system q F'.split())
        for a, O, lam, mu, b, c, m, rn in res:
            for sys_, d in (('baryon', b), ('meson', m)):
                for q, x in d['F']: w.writerow([a, O, lam, sys_, f'{q:.8g}', f'{x:.6f}'])
    print('SOLVER_DONE')
