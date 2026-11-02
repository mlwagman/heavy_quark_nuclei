import glob, re, numpy as np
out = []
for f in sorted(glob.glob("results/*_svm.npz")):
    if float(re.search(r"_a([0-9.]+)_", f).group(1)) > 0.07: continue
    d = np.load(f); W, C = d["Ws2"], d["C3N"]; nt = W.shape[1]
    for iq in (1, 2, 3):
        def r(i):
            c, w = C[iq, i, i], W[iq, i, i]
            return c.mean()/w.mean(), (c.sum() - c)/(w.sum() - w)
        r0, j0 = r(0); r1, j1 = r(nt - 1); nb = len(j0); dj = j1 - j0
        out.append((r1 - r0)/np.sqrt((nb - 1)*np.mean((dj - dj.mean())**2)))
out = np.array(out); print(len(out), "pairs: drift/sigma mean %+.2f rms %.2f max %.1f" % (out.mean(), np.sqrt(np.mean(out**2)), abs(out).max()))
