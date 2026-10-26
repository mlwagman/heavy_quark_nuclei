"""
Correlated-Gaussian variational solve of the equal-mass dark baryon (N=3) or
meson (N=2), for the Coulomb or the MRS potential.  Trial-free reference for
the GFMC E0 and <r^2>_ch.

  H = sum_k p_k^2/2m + sum_{k<l} V(r_kl),   V(r) = -c VB(r/(mufac mQ)) / r
  c = Nc-1 (meson), 1 (baryon pair in the 3bar channel)
  Coulomb: VB = alpha CF/(Nc-1)  ->  kappa = 4/3 alpha (meson), 2/3 alpha (baryon)
  MRS:     VB_LO / VB_NLO exec'd from the production GFMC generator, so the
           Borel sum, running and mu'(r) = sqrt(mu^2 + mufac^2/r^2) are identical.

Correlated-Gaussian matrix elements (Suzuki-Varga), d=3.  For a central pair
potential, r_kl = w^T x is Gaussian with variance beta = w^T C^-1 w per
component, so <V> = -c s G(beta) with
  G(beta) = sqrt(2/(pi beta)) int_0^inf e^-u VB(sqrt(2 beta u)) du,
tabulated once on a log-beta grid (exact for Coulomb).

Basis: random pair widths, log-uniform in [bmin, bmax] x a_C x ascale, with
a_C = 1/(mu_red kappa) the Coulomb pair Bohr radius.  Check convergence by
varying --nbasis and --seed.

Usage: python svm_baryon.py --alpha 0.3
       python svm_baryon.py --alpha 0.1 --potential mrs --OLO LO --mu 1e-3 --mufac 1 --ascale 0.6
"""
import sys, argparse, numpy as np
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline

ap = argparse.ArgumentParser()
ap.add_argument('--alpha', type=float, default=0.3)
ap.add_argument('--N', type=int, default=3, choices=[2, 3])
ap.add_argument('--potential', default='coulomb', choices=['coulomb', 'mrs'])
ap.add_argument('--OLO', default='LO')
ap.add_argument('--mu', type=float, default=2.0, help='mu_cut in units of mQ')
ap.add_argument('--mufac', type=float, default=1.0, help='lambda in mu\'(r)')
ap.add_argument('--nbasis', type=int, default=250)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--bmin', type=float, default=0.05, help='in units of the basis scale')
ap.add_argument('--bmax', type=float, default=6.0)
ap.add_argument('--ascale', type=float, default=1.0,
                help='basis scale relative to the Coulomb Bohr radius (MRS: ~sqrt(r2/r2_Coul))')
ap.add_argument('--scut', type=float, default=1e-10,
                help='drop overlap eigen-directions below scut x max (ghost-state guard)')
ap.add_argument('--symmetrize', action='store_true',
                help='N=3: add all 6 permutations of each width triple, so the state is exactly symmetric')
ap.add_argument('--save_trial', default='',
                help='npz with pair coefficients a=1/b^2 (n, npair) and amplitudes c: the GFMC trial '
                     'psi(R) = sum_i c_i exp(-1/2 sum_p a_ip r_p^2), pairs ordered (01),(02),(12)')
ap.add_argument('--trial_curve', default='',
                help='npz trial (from --save_trial): add its Gaussians to the basis and print the exact '
                     'imaginary-time curves <r^2>(t) and F(q; t,t) = <T|e^-Ht J e^-Ht|T>/<T|e^-2Ht|T>')
ap.add_argument('--taus', default='0,6,12,18,24,30,45,60,90,120,180,300', help='per-leg times for --trial_curve')
ap.add_argument('--qs', default='', help='comma list of |q| for F(q)/F(0)')
ap.add_argument('--out', default='', help='optional csv for F(q)/F(0)')
args = ap.parse_args()

m, Nc, CF = 1.0, 3, 4.0/3.0
N = args.N
cpair = (Nc - 1) if N == 2 else 1.0
VB_coul = args.alpha*CF/(Nc - 1)

if N == 2:
    # x = r1-r2, mu = m/2;  r_k - Rcm = +-x/2;  <r^2>_ch = |x|^2/4
    Lam = np.diag([1.0/(m/2)])
    W = [np.array([1.0])]
    Wk = [np.array([0.5]), np.array([-0.5])]
    Rmat = np.diag([0.25])
else:
    # x1 = r1-r2, x2 = (r1+r2)/2 - r3;  mu1 = m/2, mu2 = 2m/3
    Lam = np.diag([1.0/(m/2), 1.0/(2*m/3)])
    W = [np.array([1.0, 0.0]), np.array([0.5, 1.0]), np.array([-0.5, 1.0])]
    Wk = [np.array([0.5, 1/3]), np.array([-0.5, 1/3]), np.array([0.0, -2/3])]
    Rmat = np.diag([0.5, 2.0/3.0])/3.0
nj = Lam.shape[0]

kappa = cpair*VB_coul
aC = 1.0/(0.5*m*kappa)                       # Coulomb pair Bohr radius
L0 = aC*args.ascale

# ---------------- pair-potential kernel G(beta) ----------------
if args.potential == 'coulomb':
    def G(beta):
        return VB_coul*np.sqrt(2.0/(np.pi*beta))
else:
    SRC = "heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py"
    argv = (f"x --n_step 100 --n_walkers 2 --dtau_iMev 0.5 --count_skip 5 --nf 0 "
            f"--Nc 3 --N_coord 2 --mQ 1.0 --mu {args.mu} --mufac {args.mufac} --Rstar 0.35 "
            f"--spoila 1 --wavefunction compact --potential full --OLO {args.OLO} "
            f"--alpha_s {args.alpha} --Q_dir 1 0 0 --outdir /tmp/xx/ --Q_mag 0.0").split()
    src = open(SRC).read()
    sys.argv = argv
    ns = {'__name__': 'svm_bench'}
    exec(src[:src.index("os.makedirs(outdir")], ns)
    VBf = ns['VB_LO'] if args.OLO == 'LO' else ns['VB_NLO']
    lr = np.linspace(np.log(1e-8*L0), np.log(1e4*L0), 4001)
    VBt = np.array(VBf(np.exp(lr)/(args.mufac*1.0)), dtype=float)
    assert np.all(np.isfinite(VBt)), 'non-finite VB in table'
    s_u = np.linspace(-46.0, np.log(80.0), 6001); u = np.exp(s_u)
    def Gtab(beta):
        lrr = 0.5*np.log(2.0*beta[:, None]*u[None, :])
        vb = np.interp(lrr, lr, VBt)
        return np.sqrt(2.0/(np.pi*beta))*np.trapezoid(np.exp(-u)*u*vb, s_u, axis=1)
    lbg = np.linspace(np.log((1e-3*L0)**2), np.log((1e3*L0)**2), 1201)
    # sqrt(beta) G is smooth and bounded; a cubic spline keeps V consistent with S
    # to ~1e-10 (linear interpolation left 1e-6 errors that the near-null overlap
    # directions amplified into spurious deep eigenvalues)
    Gs = CubicSpline(lbg, Gtab(np.exp(lbg))*np.exp(0.5*lbg))
    def G(beta):
        lb = np.log(beta)
        assert lb.min() > lbg[0] and lb.max() < lbg[-1], 'beta outside table'
        return Gs(lb)/np.sqrt(beta)

# ---------------- basis ----------------
rng = np.random.default_rng(args.seed)
npair = len(W)
b = L0*np.exp(rng.uniform(np.log(args.bmin), np.log(args.bmax), size=(args.nbasis, npair)))
if args.symmetrize and N == 3:
    # particle permutations act on the three pairs as all permutations of (b12, b13, b23)
    import itertools
    b = np.concatenate([b[:, list(p)] for p in itertools.permutations(range(3))])
if args.trial_curve:
    _tr = np.load(args.trial_curve)
    assert int(_tr['N']) == N
    b = np.concatenate([b, 1.0/np.sqrt(_tr['a'])])
    xt = np.concatenate([np.zeros(len(b) - len(_tr['c'])), _tr['c']])   # trial in the combined basis
A = np.einsum('np,pi,pj->nij', 1.0/b**2, np.array(W), np.array(W))

C = A[:, None] + A[None, :]
if nj == 1:
    dC = C[..., 0, 0]; Ci = 1.0/C
else:
    dC = C[..., 0, 0]*C[..., 1, 1] - C[..., 0, 1]*C[..., 1, 0]
    Ci = np.empty_like(C)
    Ci[..., 0, 0] = C[..., 1, 1]/dC; Ci[..., 1, 1] = C[..., 0, 0]/dC
    Ci[..., 0, 1] = -C[..., 0, 1]/dC; Ci[..., 1, 0] = -C[..., 1, 0]/dC
S = ((2*np.pi)**nj/dC)**1.5
T = S*1.5*np.einsum('ij,njk,nmkl,mli->nm', Lam, A, Ci, A)
V = np.zeros_like(S)
for w in W:
    V += -cpair*S*G(np.einsum('i,nmij,j->nm', w, Ci, w))
R2 = S*3.0*np.einsum('ij,nmji->nm', Rmat, Ci)

ev, U = eigh(S)
keep = ev > args.scut*ev.max()
U = U[:, keep]/np.sqrt(ev[keep])
E, c = eigh(U.T @ (T + V) @ U)
g = U @ c[:, 0]
norm = g @ S @ g
r2 = (g @ R2 @ g)/norm

def FF(q):
    f = np.mean([np.exp(-0.5*q*q*np.einsum('i,nmij,j->nm', wk, Ci, wk)) for wk in Wk], axis=0)
    return (g @ (S*f) @ g)/norm

r2C = 27.0/(16*args.alpha**2) if N == 2 else None
tag = f"{args.potential}{'' if args.potential == 'coulomb' else ' '+args.OLO} N={N} alpha={args.alpha} mu={args.mu:.4g} mufac={args.mufac}"
print(f"SVM {tag}  nbasis={len(b)} kept={keep.sum()} seed={args.seed}")
print(f"  E0 = {E[0]:.8g}   E1 = {E[1]:.8g}")
print(f"  <r^2>_ch = {r2:.8g}" + (f"   / Coulomb = {r2/r2C:.5f}" if r2C else ''))
if args.trial_curve:
    Gv = U @ c                                       # all eigenvectors, G^T S G = 1
    cn = Gv.T @ S @ xt                               # trial amplitudes on eigenstates
    dE = E - E[0]
    assert dE.min() > -1e-12, 'ghost state below E0'
    R2n = Gv.T @ R2 @ Gv
    qs_c = [float(x) for x in args.qs.split(',')] if args.qs else [0.05, 0.1, 0.2]
    Jn = []
    for q in qs_c:
        f = np.mean([np.exp(-0.5*q*q*np.einsum('i,nmij,j->nm', wk, Ci, wk)) for wk in Wk], axis=0)
        Jn.append(Gv.T @ (S*f) @ Gv)
    print(f"  trial overlap |<0|T>|^2/<T|T> = {cn[0]**2/np.sum(cn**2):.5f}")
    print(f"  {'t/leg':>6} {'<r2>(t)':>9} {'/exact':>7}  " + "  ".join(f"F({q:g})" for q in qs_c))
    for t in [float(x) for x in args.taus.split(',')]:
        v = cn*np.exp(-dE*t); nn = v @ v
        print(f"  {t:>6g} {v @ R2n @ v/nn:>9.3f} {v @ R2n @ v/nn/r2:>7.4f}  "
              + "  ".join(f"{v @ J @ v/nn:.5f}" for J in Jn))
if args.save_trial:
    cc = g/g.sum()                                   # psi(0) = 1
    np.savez(args.save_trial, a=1.0/b**2, c=cc, E0=E[0], r2=r2, N=N, alpha=args.alpha)
    print(f"  trial -> {args.save_trial}: {len(cc)} Gaussians, max|c_i|/psi(0) = {np.abs(cc).max():.3g}")
if args.qs and not args.trial_curve:
    qs = [float(x) for x in args.qs.split(',')]
    ffs = [FF(q) for q in qs]
    print("  F(q)/F(0):", "  ".join(f"{q:g}:{f:.5f}" for q, f in zip(qs, ffs)))
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write("Q,F_over_F0\n0.0,1.0\n")
            for q, f in zip(qs, ffs): fh.write(f"{q},{f:.6f}\n")
            fh.write(f"# E0={E[0]:.8g} r2={r2:.8g}\n")
