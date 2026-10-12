"""
Radius with the Ritz-rank systematic made explicit.

The form factor drifts with Ritz rank m, and the drift (4-7%) exceeds the
statistical error (~1.5%). Quoting a single m -- in particular m_max, the old
default -- hides it and makes the answer depend on how many tau points happened
to be stored, since m_max = (stored points)/2.

For each dataset this refits <r^2> at every m and reports

    <r^2> = value(stat)(rank)

where 'value' and 'stat' come from the m with the smallest statistical error,
and 'rank' is the half-spread over the usable m range -- a conservative
systematic, since no plateau in m exists.

Usage: python radius_with_rank_systematic.py <dir> [<dir> ...]
"""
import csv, glob, re, os, sys
import numpy as np
from scipy.optimize import curve_fit

def gendip(q2,b,p): return (1.0+q2*b*b/(4*p))**(-p)

def curve(d, system, mmin_frac=0.0):
    rows={}
    for fn in sorted(glob.glob(f'{d}/*_twoleg_Lanczos_vsm.csv')):
        mm=re.search(r'Qmag([0-9.]+)',fn)
        if not mm: continue
        rows[float(mm.group(1))]={int(r['m']):(float(r['F']),float(r['F_err']))
                                  for r in csv.DictReader(open(fn)) if r['current']=='C3_N'}
    if 0.0 not in rows or len(rows)<4: return None
    Qs=np.array([q for q in sorted(rows) if q>0])
    out=[]
    for m in sorted(rows[0.0]):
        if any(m not in rows[q] for q in rows): continue
        F=np.array([rows[q][m][0]/rows[0.0][m][0] for q in Qs])
        E=np.array([max(rows[q][m][1]/abs(rows[0.0][m][0]),1e-6) for q in Qs])
        try:
            if system=='meson':
                p,c=curve_fit(lambda x,b:gendip(x,b,2.0),Qs**2,F,p0=[6],sigma=E,
                              absolute_sigma=True,maxfev=20000)
                r2,er=1.5*p[0]**2, 3*p[0]*np.sqrt(c[0,0])
            else:
                p,c=curve_fit(gendip,Qs**2,F,p0=[6,3],sigma=E,absolute_sigma=True,
                              maxfev=20000,bounds=([0.5,1.],[1e5,12.]))
                r2,er=1.5*p[0]**2, 3*p[0]*np.sqrt(np.diag(c))[0]
        except Exception: continue
        out.append((m,r2,er))
    return out or None

for d in sys.argv[1:]:
    system = 'baryon' if 'baryon' in d else 'meson'
    c = curve(d, system)
    if not c:
        print(f"{os.path.basename(d):30} no m-curve"); continue
    ms=np.array([x[0] for x in c]); v=np.array([x[1] for x in c]); e=np.array([x[2] for x in c])
    k=int(np.argmin(e))                       # smallest statistical error
    syst=0.5*(v.max()-v.min())                # half-spread over m
    print(f"{os.path.basename(d):30} <r^2> = {v[k]:8.3f} ({e[k]:.3f})stat ({syst:.3f})rank "
          f"  [m={ms[k]}, m_max={ms[-1]}, drift {100*(v.max()-v.min())/v.mean():.1f}%, "
          f"endpoint {v[-1]:.3f}]")
