# growth of the statistical error of the raw ratio C3(t,t)/C2(2t) with contiguous-walker block size
import glob, re, numpy as np
BS = [1, 2, 4, 8, 16, 32, 64]
def jk_err(c, w):
    nb = len(c); j = (c.sum() - c)/(w.sum() - w)
    return np.sqrt((nb - 1)*np.mean((j - j.mean())**2))
res = {}
for f in sorted(glob.glob('results/*_svm.npz')):
    a = float(re.search(r'_a([0-9.]+)_', f).group(1))
    d = np.load(f); W, C = d['Ws2'], d['C3N']; nt = W.shape[1]
    g = []
    for iq in (2, 3, 4, 5):
        for i in (0, nt//2, nt - 1):
            c, w = C[iq, i, i], W[iq, i, i]
            e = []
            for B in BS:
                n = len(c)//B
                e.append(jk_err(c[:n*B].reshape(n, B).mean(1), w[:n*B].reshape(n, B).mean(1)))
            g.append(np.array(e)/e[0])
    res.setdefault(a, []).append(np.mean(g, axis=0))
print('alpha  n   err(B)/err(1) for block size B (in bins of 4 walkers): ' + ' '.join(f'{b:>5}' for b in BS))
for a in sorted(res):
    print(f'{a:<6} {len(res[a]):<3} ' + ' '*50 + ' '.join(f'{x:5.2f}' for x in np.mean(res[a], axis=0)))
