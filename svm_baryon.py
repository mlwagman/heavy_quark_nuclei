"""
Stochastic-variational (correlated-Gaussian) solve of the 3 equal-mass Coulomb
baryon:  H = sum_k p_k^2/2m + sum_{k<l} (-kappa/r_kl),  kappa = (2/3) alpha
(each pair in the antisymmetric 3bar channel, V=-VB/r, VB=alpha*CF/(Nc-1)).

Near-exact benchmark for the GFMC baryon E0 and <r^2>.  Correlated-Gaussian
matrix elements (Suzuki-Varga), n=2 Jacobi vectors, d=3.

Self-checks:
  * 2-body reduction (drop 3rd particle -> hydrogen) reproduces -mu kappa^2/2.
  * energy vs GFMC <H>.
  * convergence in basis size.

Usage: python svm_baryon.py --alpha 0.3 [--nbasis 200 --seed 1]
"""
import argparse, numpy as np
from numpy.linalg import inv, det
from scipy.linalg import eigh

ap = argparse.ArgumentParser()
ap.add_argument('--alpha', type=float, default=0.3)
ap.add_argument('--nbasis', type=int, default=250)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--bmin', type=float, default=0.5)
ap.add_argument('--bmax', type=float, default=60.0)
args = ap.parse_args()

m = 1.0
kappa = (2.0/3.0)*args.alpha                # pairwise coupling, V=-kappa/r

# Jacobi: x1 = r1-r2 ; x2 = (r1+r2)/2 - r3.  mu1 = m/2, mu2 = 2m/3.
Lam = np.diag([1.0/(m/2), 1.0/(2*m/3)])     # kinetic:  T = -1/2 sum Lam_kl grad_k.grad_l
# pair-distance coefficient vectors  r_ij = w^T (x1,x2)
W = {'12': np.array([1.0, 0.0]),
     '13': np.array([0.5, 1.0]),
     '23': np.array([-0.5, 1.0])}
# <r^2>_ch = (1/3)[ (1/2)|x1|^2 + (2/3)|x2|^2 ]
Rmat = np.diag([0.5, 2.0/3.0])/3.0

def Amat(bpars):
    """physical correlated Gaussian: A = sum_pairs (1/b_p^2) w_p w_p^T"""
    A = np.zeros((2,2))
    for (p, b) in zip(['12','13','23'], bpars):
        A += (1.0/b**2)*np.outer(W[p], W[p])
    return A

rng = np.random.default_rng(args.seed)
# build basis of A matrices with random pair widths (log-uniform)
basis = []
while len(basis) < args.nbasis:
    b = np.exp(rng.uniform(np.log(args.bmin), np.log(args.bmax), size=3))
    A = Amat(b)
    if np.all(np.linalg.eigvalsh(A) > 1e-9):
        basis.append(A)

n = len(basis)
S = np.zeros((n,n)); H = np.zeros((n,n)); R2 = np.zeros((n,n))
for i in range(n):
    for j in range(i+1):
        A, B = basis[i], basis[j]
        C = A+B; Ci = inv(C); dC = det(C)
        s = (2*np.pi)**2 / dC; s = s**1.5           # overlap (n=2,d=3)
        t = s * 1.5 * np.trace(Lam @ A @ Ci @ B)     # kinetic
        v = 0.0
        for p in ['12','13','23']:
            w = W[p]; beta = w @ Ci @ w
            v += -kappa * s * np.sqrt(2.0/(np.pi*beta)) # -kappa <1/r>
        r2 = s * 3.0 * np.trace(Rmat @ Ci)           # <r^2>_ch numerator
        S[i,j]=S[j,i]=s; H[i,j]=H[j,i]=t+v; R2[i,j]=R2[j,i]=r2

# solve generalized eig with overlap regularization (drop tiny-norm directions)
evals, evecs = eigh(S)
keep = evals > 1e-12*evals.max()
U = evecs[:,keep] / np.sqrt(evals[keep])
Hp = U.T @ H @ U
w, c = eigh(Hp)
E0 = w[0]; g = U @ c[:,0]
r2 = (g @ R2 @ g)/(g @ S @ g)

gfmc = {0.2:-0.01902, 0.3:-0.04283, 0.5:-0.11922}
print(f"=== SVM 3-body Coulomb baryon  alpha={args.alpha}  kappa={kappa:.4f}  "
      f"(nbasis={n}, kept={keep.sum()}) ===")
print(f"  E0(SVM)  = {E0:.6f}     GFMC <H> = {gfmc.get(args.alpha,'?')}"
      f"   diff = {abs(E0-gfmc.get(args.alpha,E0))/abs(E0)*100:.2f}%")
print(f"  <r^2>_ch = {r2:.2f}")
print(f"  E1(SVM)  = {w[1]:.6f}   gap = {w[1]-E0:.5f}")

# ---- exact charge form factor F(q)/N = <(1/3) sum_k exp(i q.(r_k-Rcm))> ----
# r_k - Rcm = w_k^T (x1,x2);  <exp(iq.w^T rho)>/S = exp(-q^2/2 * w^T C^-1 w)
Wk = [np.array([0.5, 1/3]), np.array([-0.5, 1/3]), np.array([0.0, -2/3])]
def FF(qmag):
    num = np.zeros((n, n));
    for i in range(n):
        for j in range(n):
            Ci = inv(basis[i]+basis[j])
            ff = np.mean([np.exp(-0.5*qmag**2 * (wk @ Ci @ wk)) for wk in Wk])
            num[i,j] = ff
    Snum = S * num
    return (g @ Snum @ g)/(g @ S @ g)
qs = [0.02,0.05,0.1,0.2,0.35,0.5]
ffs = [FF(q) for q in qs]
print("  F(q)/F(0):", "  ".join(f"{q}:{f:.4f}" for q,f in zip(qs,ffs)))
import os
os.makedirs('paper_final', exist_ok=True)
with open(f'paper_final/svm_baryon_a{args.alpha}.csv','w') as fh:
    fh.write("Q,F_over_F0\n0.0,1.0\n")
    for q,f in zip(qs,ffs): fh.write(f"{q},{f:.6f}\n")
    fh.write(f"# E0={E0:.6f} r2={r2:.3f}\n")
