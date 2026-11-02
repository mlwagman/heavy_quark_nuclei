import csv, glob, re, numpy as np
E = {(float(r["alpha"]), r["OLO"], float(r["lam"])): float(r["E0"]) for r in csv.DictReader(open("results/svm_baryon_lam.csv"))}
LAM = {"lamh": 0.5, "lam1": 1.0, "lam2": 2.0}
d = []
for f in sorted(glob.glob("logs/mrs_*_baryon_a*_svm.log")):
    m = re.search(r"mrs_(n?lo)_baryon_a([0-9.]+)_(lam.)_svm", f); a = float(m[2])
    if a > 0.07: continue
    H = [float(x) for x in re.findall(r"<H>=([-+0-9.eE]+) MeV", open(f).read())]
    # every run prints <H> per slice; use the second half of each Q block and block by Q for the error
    n = len(H)//7; blocks = [np.mean(H[i*n + n//2:(i + 1)*n]) for i in range(7)]
    e0 = E[(a, m[1].upper(), LAM[m[3]])]
    d.append((a, 100*(np.mean(blocks)/e0 - 1), 100*np.std(blocks, ddof=1)/np.sqrt(7)/abs(e0)))
d = np.array(d); print(len(d), "datasets: <H>/E0-1 mean %+.3f%%, rms %.3f%%, mean err %.3f%%, max |dev| %.3f%%" % (d[:,1].mean(), np.sqrt(np.mean(d[:,1]**2)), d[:,2].mean(), abs(d[:,1]).max()))
