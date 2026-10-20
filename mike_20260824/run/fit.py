"""Fit F(q^2)/F(0) with the generalized dipole and print <r^2>.
   p=2 fixed for mesons, free for baryons. <r^2> = 1.5 b^2.
   Usage: fit.py <prod_ME.csv> <meson|baryon> [expected]"""
import csv, sys
import numpy as np
from scipy.optimize import curve_fit
fn, system = sys.argv[1], sys.argv[2]
exp = float(sys.argv[3]) if len(sys.argv) > 3 else None
def gendip(q2, b, p): return (1.0 + q2*b*b/(4*p))**(-p)
D = {float(r['Q']): (float(r['F']), float(r['F_err']))
     for r in csv.DictReader(open(fn)) if r['channel'] == 'C3_N'}
if 0.0 not in D or len(D) < 3:
    sys.exit(f"{fn}: only {len(D)} usable points; extraction failed")
F0 = D[0.0][0]; Qs = np.array(sorted(q for q in D if q > 0))
F = np.array([D[q][0]/F0 for q in Qs]); E = np.array([max(D[q][1]/abs(F0), 1e-4) for q in Qs])
if system == 'meson':
    pm, cm = curve_fit(lambda x, b: gendip(x, b, 2.0), Qs**2, F, p0=[6],
                       sigma=E, absolute_sigma=True, maxfev=8000)
    b, be, p = pm[0], np.sqrt(cm[0, 0]), 2.0
else:
    pb, cb = curve_fit(gendip, Qs**2, F, p0=[6, 3], sigma=E, absolute_sigma=True,
                       maxfev=8000, bounds=([0.5, 1.0], [200.0, 12.0]))
    b, be, p = pb[0], np.sqrt(np.diag(cb))[0], pb[1]
r2, err = 1.5*b*b, 3*b*be
print(f"\nF(0) = {F0:.4f}   (should equal N_coord)")
print(f"<r^2> = {r2:.3f} +/- {err:.3f}   (b={b:.4f}, p={p:.2f})")
if exp: print(f"expected {exp:.3f}   ->  {100*(r2/exp-1):+.2f}%  ({(r2-exp)/err:+.1f} sigma)")
