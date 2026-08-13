"""
Sanity battery on a prod_ME.csv: exact/identity checks per directory.

Checks:
 1. F_N(0) = N_coord exactly
 2. Galilean: |F_Vi_x(Q)/F_N(Q)| = Q/(2M) at every Q
 3. Kinematic: F_T0i_x(Q) = (Q/2m)*F_N(Q) at every Q
 4. F_S(0) = N - <K>/2m, F_V0(0) = N + <K>/2m  (K parsed from run_Q0.0.log)
 5. Forbidden channels ~ 0 (Ai, T0i_y/z, Tij_z)
 6. [meson Coulomb only] F_N(Q)/F_N(0) vs exact dipole [1+(Q a_B/4)^2]^-2

Usage: python prod_sanity.py <dir> --N 2 --M 2 [--aB 5.0]
"""
import csv, os, re, sys, argparse, numpy as np

p = argparse.ArgumentParser()
p.add_argument('dir'); p.add_argument('--N', type=int, required=True)
p.add_argument('--M', type=float, required=True)
p.add_argument('--aB', type=float, default=None, help='exact Bohr radius (Coulomb meson)')
p.add_argument('--m1', type=float, default=1.0)
args = p.parse_args()

D = {}
for r in csv.DictReader(open(os.path.join(args.dir,'prod_ME.csv'))):
    D.setdefault(float(r['Q']),{})[r['channel']] = (float(r['F']), float(r['F_err']))
Qs = sorted(D)
ok = lambda b: "PASS" if b else "FAIL"

print(f"===== SANITY {args.dir} (N={args.N}, M={args.M}) =====")
FN0,_ = D[0.0]['C3_N']
print(f"[1] F_N(0) = {FN0:.6f} vs {args.N}:  {ok(abs(FN0-args.N)<1e-3)}")

print(f"[2] Galilean |F_Vi/F_N| = Q/(2M):")
for Q in Qs:
    if Q==0 or 'C3_Vi_x' not in D[Q]: continue
    r = abs(D[Q]['C3_Vi_x'][0]/D[Q]['C3_N'][0]); pred = Q/(2*args.M)
    print(f"    Q={Q}: {r:.5f} vs {pred:.5f}  ratio={r/pred:.3f}  {ok(abs(r/pred-1)<0.05)}")

print(f"[3] Kinematic F_T0i = (Q/2m) F_N:")
for Q in Qs:
    if Q==0 or 'C3_T0i_x' not in D[Q]: continue
    r = abs(D[Q]['C3_T0i_x'][0]); pred = Q/(2*args.m1)*abs(D[Q]['C3_N'][0])
    print(f"    Q={Q}: {r:.5f} vs {pred:.5f}  {ok(abs(r/pred-1)<0.02)}")

logf = os.path.join(args.dir,'run_Q0.0.log')
if os.path.exists(logf):
    K = None
    for line in open(logf, errors='ignore'):
        m = re.search(r'tau=0\.000.*<K>=([+-][0-9.]+)', line)
        if m: K = float(m.group(1)); break
    if K is not None:
        FS0 = D[0.0].get('C3_S',(np.nan,0))[0]; FV00 = D[0.0].get('C3_V0',(np.nan,0))[0]
        pS, pV = args.N - K/(2*args.m1), args.N + K/(2*args.m1)
        print(f"[4] F_S(0)  = {FS0:.6f} vs N-<K>/2m = {pS:.6f}  {ok(abs(FS0-pS)<5e-3)}")
        print(f"    F_V0(0) = {FV00:.6f} vs N+<K>/2m = {pV:.6f}  {ok(abs(FV00-pV)<5e-3)}")

bad = []
for Q in Qs:
    for ch,(F,Fe) in D[Q].items():
        if any(k in ch for k in ['Ai','T0i_y','T0i_z','Tij_z','Vi_y','Vi_z']):
            if abs(F) > max(5*Fe, 5e-3): bad.append((Q,ch,F))
print(f"[5] forbidden channels ~0: {ok(not bad)}" + (f"  {bad[:4]}" if bad else ""))

if args.aB:
    print(f"[6] exact dipole (a_B={args.aB}):")
    for Q in Qs:
        if Q==0: continue
        fr = D[Q]['C3_N'][0]/FN0; fe = D[Q]['C3_N'][1]/FN0
        ex = 1.0/(1.0+(Q*args.aB/4)**2)**2
        pull = (fr-ex)/max(fe,1e-6)
        print(f"    Q={Q}: F/F0={fr:.4f}±{fe:.4f} exact={ex:.4f} pull={pull:+.1f}  {ok(abs(pull)<3)}")
