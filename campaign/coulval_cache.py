# korr J00 with the calibrated m_max for the scaled Coulomb family, sequential
import glob, os, pickle, sys
sys.path.insert(0, '/Users/user/bassi/heavy_quark_nuclei')
import gfmc_korr
res = {}
for f in sorted(glob.glob('/Users/user/hqn_local/coulval_npz/*.npz')):
    n = os.path.basename(f)[:-4]; N = 3 if 'baryon' in n else 2
    res[n] = gfmc_korr.J00(f, N=N)
    F0, r2, er, p, *_ = gfmc_korr.radius(res[n], 'baryon' if N == 3 else 'meson')
    a = float(n.split('_a')[1]); print(n, f'F0={F0:.5f} a2r2={r2*a*a:.4f} +/- {er*a*a:.4f} p={p:.2f}', flush=True)
pickle.dump(res, open('/Users/user/bassi/heavy_quark_nuclei/paper_final/korr_cache_coulval.pkl', 'wb'))
print('COULVAL_DONE')
