# korr J00 for the in-range SVM-trial datasets, in parallel; one pickle
import glob, os, pickle, re, sys
from multiprocessing import Pool
sys.path.insert(0, 'code')
import gfmc_korr
INR = {0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.07}
def one(f):
    name = os.path.basename(f)[:-4]
    print(name, 'start', flush=True)
    return name, gfmc_korr.J00(f, N=3)
fs = [f for f in sorted(glob.glob('results/*_svm.npz')) if float(re.search(r'_a([0-9.]+)_', f).group(1)) in INR]
with Pool(28) as p: res = dict(p.map(one, fs, chunksize=1))
pickle.dump(res, open('results/korr_cache_svm.pkl', 'wb'))
print('CACHE_DONE', len(res))
