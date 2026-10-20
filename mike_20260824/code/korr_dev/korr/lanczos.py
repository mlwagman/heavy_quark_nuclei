#!/usr/bin/env python3
"""Implementation of scalar Lanczos analysis.
"""

import numpy as np
from tqdm import tqdm

from . import mnp
from .mnp import mp_array, downcast
from .eig_util import fix_sign
from .util import zip_items

## Utility
def is_diag(x, atol=1e-8):
    return mnp.allclose(x - np.diag(np.diag(x)), 0, atol=atol)

## Lanczos recursion
def make_lanczos_basis_offdiag(Ct):
    Nt = len(Ct)

    alpha = {}
    beta = {}
    Tnm = {} # <n|T^m|n> = <T^m>_n
    Xnm = {} # <n|T^m|n+1>

    # Compute low-depth special cases
    # n=1
    alpha[1] = Ct[1]
    beta[2] = (Ct[2] - alpha[1]**2)**0.5
    Tnm[1] = {ii : c for ii,c in enumerate(Ct)} 
    Xnm[1] = {m : (Tnm[1][m+1] - alpha[1] * Tnm[1][m]) / beta[2] for m in range(1, Nt-1)} # <1|A^m|2>

    # n = 2
    Tnm[2] = {} # <T^m>_2
    for m in range(1, Nt-2):
        Tnm[2][m] = 1/beta[2]**2 * ( Tnm[1][m+2] - 2 * alpha[1] * Tnm[1][m+1] + alpha[1]**2 * Tnm[1][m] )
    alpha[2] = Tnm[2][1]

    # Xnm[2] = {} # <2|T^m|3>
    # for m in range(1, Nt-2):
    #     Xnm[2][m] = 1/beta[3] * (Tnm[2][m+1] - alpha[2]*Tnm[2][m] - beta[2]*Xnm[1][m])

    # Build rest
    for n in range(3, Nt//2+1):
        beta[n] = ( Tnm[n-1][2] - alpha[n-1]**2 - beta[n-1]**2 )**0.5

        Xnm[n-1] = {
            m : 1/beta[n] * (Tnm[n-1][m+1] - alpha[n-1]*Tnm[n-1][m] - beta[n-1] * Xnm[n-2][m])
            for m in range(1, (Nt-2*n)+1)}

        dTm = {}
        for m in range(1, (Nt-2*n)+2):
            dT = Tnm[n-1][m+2] - 2 * alpha[n-1] * Tnm[n-1][m+1] + alpha[n-1]**2 * Tnm[n-1][m]
            dT += beta[n-1]**2 * Tnm[n-2][m]
            dT -= 2 * beta[n-1] * Xnm[n-2][m+1]
            dT += 2 * alpha[n-1] * beta[n-1] * Xnm[n-2][m]
            dTm[m] = dT / beta[n]**2
        Tnm[n] = dTm

        alpha[n] = Tnm[n][1]
    return alpha, beta, Tnm, Xnm
    
def load_lanczos_matrix(alpha, beta):
    T = np.zeros((len(alpha), len(alpha)), dtype=object)
    for n in range(len(alpha)):
        T[n,n] = alpha[n+1]
    for n in range(len(alpha)-1):
        T[n,n+1] = beta[n+2]
        T[n+1,n] = beta[n+2]
    return T


## Krylov coefficients (auxiliary recursion)
def make_krylov_coeffs(alpha, beta, Nt=None):
    k = max(beta.keys())
    if Nt is None: Nt = 2*k
    a = {} # a_jt = {j : (a_j)_t}
    a[1] = np.array([1] + [0]*(Nt-1), dtype=object)
    a[2] = np.array([-alpha[1]/beta[2], 1/beta[2]] + [0]*(Nt-2), dtype=object)
    for j in range(3, k+1):
        da = np.zeros(Nt, dtype=object)
        da -= beta[j-1] /beta[j] * a[j-2]
        da -= alpha[j-1]/beta[j] * a[j-1]
        da += 1/beta[j] * np.array([0] + list(a[j-1])[:-1], dtype=object) # roll w/ zero pad
        a[j] = da
    return a
# a_ = np.stack(list(map(list, a.values())), axis=1) # {j: (a_j)_t} -> a_tj

def make_krylov_matrix(alpha, beta, Nt):
    a = make_krylov_coeffs(alpha, beta, Nt=Nt)
    a = {j : downcast(a_jt) for j, a_jt in a.items()}
    A = np.stack(list(map(list, a.values())), axis=1) # {j: (a_j)_t} -> a_tj
    return A

def mp_make_krylov_matrix(alpha, beta, Nt):
    a = make_krylov_coeffs(alpha, beta, Nt=Nt)
    a = {j : mp_array(a_jt) for j, a_jt in a.items()}
    A = np.stack(list(map(list, a.values())), axis=1) # {j: (a_j)_t} -> a_tj
    return A


## Functions to compute all eigenvalues
## TODO: deprecate immediately

def mp_compute_all_eigs(T, progress=False):
    all_evals = {}
    all_evecs = {}
    all_k = range(1, T.shape[0]+1)
    for k in (tqdm(all_k) if progress else all_k):
        evals, evecs = mnp.eig(T[:k, :k])
        all_evals[k] = evals
        all_evecs[k] = evecs
    return all_evals, all_evecs

def mp_compute_all_eigs_and_CW_evals(T):
    all_evals = {}
    all_evals_CW = {1 : np.array([], dtype=object)}
    all_evecs = {}
    for k in tqdm(range(1, T.shape[0]+1)):
        evals, evecs = mnp.eig(T[:k, :k])        
        all_evals[k] = evals
        all_evecs[k] = evecs
        
        if k > 1:
            evals_CW, _ = mnp.eig(T[1:k, 1:k])
            sort_idx = np.argsort(-downcast(evals_CW).real)
            evals_CW = evals_CW[sort_idx]
            all_evals_CW[k] = evals_CW
    return all_evals, all_evecs, all_evals_CW


def compute_all_eigs(T):
    def _eig(X):
        evals, evecs = np.linalg.eig(X)
        evecs = fix_sign(evecs)
        # SORT
        sort_idx = np.argsort(-downcast(evals))
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        return evals, evecs
    T = downcast(T)
    all_evals = {}
    all_evecs = {}
    for k in range(1, T.shape[0]+1):
        evals, evecs = _eig(T[:k, :k])
        all_evals[k], all_evecs[k] = evals, evecs
    return all_evals, all_evecs

def compute_all_eigs_and_CW(T):
    def _eig(X):
        evals, evecs = np.linalg.eig(X)
        evecs = fix_sign(evecs)
        # SORT
        sort_idx = np.argsort(-downcast(evals))
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        return evals, evecs
    T = downcast(T)
    all_evals = {}
    all_evecs = {}
    all_evals_CW = {}
    all_evecs_CW = {}
    for k in range(1, T.shape[0]+1):
        evals, evecs = _eig(T[:k, :k])
        all_evals[k], all_evecs[k] = evals, evecs
        
        evals_CW, evecs_CW = _eig(T[1:k, 1:k])
        all_evals_CW[k], all_evecs_CW[k] = evals_CW, evecs_CW
    return all_evals, all_evecs, all_evals_CW, all_evecs_CW

def compute_all_eigs_and_CW_evals(T):
    def _eig(X):
        evals, evecs = np.linalg.eig(X)
        evecs = fix_sign(evecs)
        # SORT
        sort_idx = np.argsort(-downcast(evals))
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        return evals, evecs
    T = downcast(T)
    all_evals = {}
    all_evecs = {}
    all_evals_CW = {}
    for k in range(1, T.shape[0]+1):
        evals, evecs = _eig(T[:k, :k])
        evals_CW, _ = _eig(T[1:k, 1:k])
        all_evals[k], all_evecs[k] = evals, evecs
        all_evals_CW[k] = evals_CW
    return all_evals, all_evecs, all_evals_CW


## Filtering
def get_hermitian_subspace(ev, O, atol=1e-8, filter_ZZ=True):
    # NB ONLY DIAG CORRS
    ev_is_cx = ~np.isclose(ev.imag / np.abs(ev), 0, atol=atol)
    O = downcast(O)
    Oinv = np.linalg.inv(O)

    Zl = Oinv[:,0]
    Zr = O[0].conj()

    # Complex evals have complex Zl*Zr
    ZZ = Zl*Zr.conj()
    ZZ = ZZ / np.abs(ZZ).sum() # overlap fraction
    assert not np.any( np.isclose(ZZ[ev_is_cx].imag / np.abs(ZZ[ev_is_cx]), 0) )
    # Compute norms
    lnorm = Zl/Zr # |Nk|^2
    rnorm = Zr/Zl # 1/|Nk|^2

    # non-cx, pos Zl * Zr -> normalizable Ritz
    is_herm = ~ev_is_cx
    norm_pos = lnorm.real > 0
    is_herm &= norm_pos ## also need this
    is_herm &= np.isfinite(lnorm)
    is_herm &= np.isfinite(rnorm)
    is_herm &= ~np.isclose(lnorm, 0)
    is_herm &= ~np.isclose(rnorm, 0)
    if filter_ZZ:
        is_herm &= np.isclose(ZZ.imag / np.abs(ZZ), 0)
        is_herm &= ZZ.real > 0
        is_herm &= ~np.isclose(ZZ, 0)
    
    return is_herm


## Ritz coeffs & derived quantities
def make_ritz_coeffs(all_evecs, Kr, Kl, inv_mp=False):
    all_Pr, all_Pl = {}, {}
    for m, O in all_evecs.items():
        Oinv = mnp.inv(O) if inv_mp else np.linalg.inv(O)
        Pr = Kr[:,:m] @ O
        Pl = Oinv @ Kl[:m,:]
        # norm w/ overlap trick
        znorm = (Oinv[:,0].conj() / O[0,:])**0.5 # N_k
        Pr = Pr * znorm[None,:]
        Pl = Pl / znorm[:,None]
        all_Pr[m], all_Pl[m] = Pr, Pl
    return all_Pr, all_Pl


## Overlaps
def make_all_overlaps(all_evecs, C, inv_mp=False):
    abs_ = mnp.abs if inv_mp else np.abs
    all_Zr, all_Zl = {}, {}
    for m, evec in all_evecs.items():
        evec_inv = mnp.inv(evec) if inv_mp else np.linalg.inv(evec)
        Nk = abs_(evec_inv[:,0].conj()/evec[0])**0.5 # |Nk|        
        all_Zl[m] = evec_inv[:,0] * abs_(C[0])**0.5 / Nk
        all_Zr[m] = ( Nk * evec[0] * abs_(C[0])**0.5 ).conj()
    return all_Zr, all_Zl

## Matrix elements
def make_all_J(C3, C, all_Pr, all_Pl):
#     assert downcast(C[0]).real > 0, "TESTME"
    if downcast(C[0]).real < 0: print("WARNING:", np.complex(C[0]))
    Cnorm = C[0]
    all_J = {}
    for m, (Pr, Pl) in zip_items(all_Pr, all_Pl):
        all_J[m] = Pl @ C3 @ Pr / Cnorm
    return all_J

def mp_make_all_J(C3, C, all_Pr, all_Pl):
#     assert downcast(C[0]).real > 0, "TESTME"
    if downcast(C[0]).real < 0: print("WARNING: C(0)<0", downcast(C[0]).real)
    Cnorm = C[0]
    all_J = {}
    for m, (Pr, Pl) in zip_items(all_Pr, all_Pl):
        all_J[m] = Pl @ C3 @ Pr / Cnorm
    return all_J

make_all_J = mp_make_all_J

def mp_make_all_J_(C3, Cnorm, all_Pr, all_Pl):
    assert downcast(Cnorm).real > 0, "TESTME"
    all_J = {}
    for m, (Pr, Pl) in zip_items(all_Pr, all_Pl):
        all_J[m] = Pl @ C3 @ Pr / Cnorm
    return all_J



## Residual bounds
def compute_wvnorms(Kr, Kl, KK, atol=1e-8):
    assert mnp.allclose(Kl @ KK @ Kr, np.eye(Kr.shape[1]), atol=atol)
    
    A, B = Kr, Kl.conj().T
    assert mnp.allclose(B.conj().T @ KK @ A, np.eye(B.shape[1]), atol=atol) # <w_i|v_j>
    vnorm = A.conj().T @ KK @ A  # <v_i|v_j>
    wnorm = B.conj().T @ KK @ B  # <w_i|w_j>
    
    # required for self-consistency: <v1|v1> = <w1|w1> = 1
    assert mnp.isclose(vnorm[0,0], 1, atol=atol)
    assert mnp.isclose(wnorm[0,0], 1, atol=atol)

    # assert np.allclose(downcast(vnorm), np.eye(vnorm.shape[0]))
    # assert np.allclose(downcast(wnorm), np.eye(wnorm.shape[0]))
    # sometimes -1 on diag
    assert mnp.allclose(np.abs(downcast(vnorm)), np.eye(vnorm.shape[0]), atol=atol)
    assert mnp.allclose(np.abs(downcast(wnorm)), np.eye(wnorm.shape[0]), atol=atol)
    
    return vnorm, wnorm

def compute_bounds(Kr, Kl, KK, all_evecs, beta, gamma, atol=1e-8):
    vnorm, wnorm = compute_wvnorms(Kr, Kl, KK, atol=atol)
    assert mnp.allclose(vnorm, wnorm, atol=atol)
    assert is_diag(vnorm, atol=atol)
    wvnorm = np.diag(vnorm)
    
    all_lbounds = {}
    all_rbounds = {}
    all_Rr = {}
    all_Rl = {}
    for m, O in all_evecs.items():
        if m+1 not in beta: continue
        Oinv = np.linalg.inv(downcast(O)) # TODO: CHECK PREC
            
        # note 1-indexing
        # Rr^(m)_k = |O^m_mk|^2 |gamma_m+1|^2 <v_m+1|v_m+1> / \sum_ij O^m*_ik <v_i|v_j> Om_jk
        Rr = (np.abs(O[m-1,:] * downcast(gamma[m+1]))**2 
            * vnorm[m+1-1, m+1-1] / np.diag(O.conj().T @ vnorm[:m,:m] @ O))
        # Rl^(m)_k = |Oinv^m_km|^2 |beta_m+1|^2 <w_m+1|w_m+1> / \sum_ij Oinv^m_ki <w_i|w_j> Oinv^m*_kj
        Rl = (np.abs(Oinv[:,m-1] * downcast(beta[m+1]))**2
            * wnorm[m+1-1, m+1-1] / np.diag(Oinv @ wnorm[:m,:m] @ Oinv.conj().T))
        
        all_Rr[m] = Rr
        all_rbounds[m] = np.abs(Rr)**0.5
        all_Rl[m] = Rl
        all_lbounds[m] = np.abs(Rl)**0.5
        
    return all_rbounds, all_lbounds, all_Rr, all_Rl

def compute_bounds_from_vnorms(vnorm, wnorm, all_evecs, beta, gamma):
#     # DIAGONAL ONLY:
#     assert is_diag(vnorm)
#     assert is_diag(wnorm)
    
    all_lbounds = {}
    all_rbounds = {}
    all_Rr = {}
    all_Rl = {}
    for m, O in all_evecs.items():
        if m+1 not in beta: continue
        Oinv = np.linalg.inv(downcast(O)) # TODO: CHECK PREC
            
        # note 1-indexing
        # Rr^(m)_k = |O^m_mk|^2 |gamma_m+1|^2 <v_m+1|v_m+1> / \sum_ij O^m*_ik <v_i|v_j> Om_jk
        Rr = (np.abs(O[m-1,:] * downcast(gamma[m+1]))**2 
            * vnorm[m+1-1, m+1-1] / np.diag(O.conj().T @ vnorm[:m,:m] @ O))
        # Rl^(m)_k = |Oinv^m_km|^2 |beta_m+1|^2 <w_m+1|w_m+1> / \sum_ij Oinv^m_ki <w_i|w_j> Oinv^m*_kj
        Rl = (np.abs(Oinv[:,m-1] * downcast(beta[m+1]))**2
            * wnorm[m+1-1, m+1-1] / np.diag(Oinv @ wnorm[:m,:m] @ Oinv.conj().T))
        
        all_Rr[m] = Rr
        all_rbounds[m] = np.abs(Rr)**0.5
        all_Rl[m] = Rl
        all_lbounds[m] = np.abs(Rl)**0.5
        
    return all_rbounds, all_lbounds, all_Rr, all_Rl

# vnorm_vv, wnorm_vv = compute_wvnorms(Kr_vv, Kr_vv.T, KK_vv/Cvv[0])
# vnorm_ww, wnorm_ww = compute_wvnorms(Kr_ww, Kr_ww.T, KK_ww/Cww[0])
# *_, all_Br_vv_, all_Bl_vv_ = compute_bounds_from_vnorms(vnorm_vv, wnorm_vv,
#     all_evec_vv, beta_vv, beta_vv)
# *_, all_Br_ww_, all_Bl_ww_ = compute_bounds_from_vnorms(vnorm_ww, wnorm_ww,
#     all_evec_ww, beta_ww, beta_ww)
# for m, (B,B_) in zip_items(all_Br_vv, all_Br_vv_): assert mnp.allclose(B,B_)
# for m, (B,B_) in zip_items(all_Bl_vv, all_Bl_vv_): assert mnp.allclose(B,B_)
# for m, (B,B_) in zip_items(all_Br_ww, all_Br_ww_): assert mnp.allclose(B,B_)
# for m, (B,B_) in zip_items(all_Bl_ww, all_Bl_ww_): assert mnp.allclose(B,B_)

# TODO: unspecialize from beta=gamma convention
def compute_off_diag_bounds(Kr_vw, Kl_vw, Cvw, KK_vv, KK_ww, beta_vw, all_evec_vw):
    # v=r, w=l
    vnorm_vw = Kr_vw.conj().T @ KK_vv @ Kr_vw / mnp.abs(Cvw[0])
    wnorm_vw = Kl_vw @ KK_ww @ Kl_vw.conj().T / mnp.abs(Cvw[0])
    return compute_bounds_from_vnorms(
        vnorm_vw, wnorm_vw,
        all_evec_vw, beta_vw, beta_vw)

