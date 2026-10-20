#!/usr/bin/env python3
"""Implementation of block Lanczos analysis.
TODO: reimplement everything for beta, gamma version
"""

import numpy as np
import scipy.special
import numba

from tqdm import tqdm

from . import mnp
from .mnp import downcast as downcast_
from .eig_util import fix_sign, inv_block_evec, mp_inv_block_evec
from .util import zip_items, unpack_ragged_dict
from .util import rotate_block_C3pt
from .util import make_block_hankel

## Utility

def downcast_dict(d):
    return {k : downcast_(v) for k,v in d.items()}


### O,Oinv,Rho,Tau version

@numba.jit(forceobj=True) # minor speedup
def orthonormalize_corr_mx(Ct, tau_convention=lambda norm: norm**0.5):
    # Ct ~ t,ab
    bdim = Ct.shape[-1]
    norm, rot = mnp.eig(Ct[0])
    unrot = mnp.inv(rot)
    tau = tau_convention(norm)
    rho = norm / tau
    F = np.diag(1/tau) @ unrot @ Ct @ rot @ np.diag(1/rho)
    assert mnp.allclose(F[0], np.eye(bdim))
    return F, rho, tau, rot, unrot

def run_block_recursion(Ct, tau_convention=lambda norm: norm**0.5):
    Nt,_,bdim = Ct.shape
    assert Ct.shape == (Nt, bdim,bdim)
    n_steps = Nt//2
    
    ## Setup
    C = {1 : Ct} # <r_ia|T^t|r_ib>
    F = {} # <w_ia|T^t|v_jb>
    G = {} # <w_ja|T^t|v_(j-1)b>
    H = {} # <w_(j-1)a|T^t|v_jb>

    O = {}
    Oinv = {}
    Rho = {}
    Tau = {}
    Alpha = {}


    ## Step 1:
    F[1], Rho[1],Tau[1], O[1],Oinv[1] = orthonormalize_corr_mx(C[1], tau_convention)
    Alpha[1] = F[1][1]
    
    if Nt <= 2:
        return Alpha, Rho,Tau, O,Oinv, C,F,G,H

    ## Step 2:
    # compute C2(t) = <s_2a|T^t|r_2b>
    C[2] = ( # np.zeros((Nt-2, bdim, bdim), dtype=object)
        F[1][2:] # F1_ab(t+2)
        - Alpha[1] @ F[1][1:-1] # \sum_c alpha1_ac F1_cb(t+1)
        - F[1][1:-1] @ Alpha[1] # \sum_c F1_ac alpha1_cb
        + Alpha[1] @ F[1][:-2] @ Alpha[1] # \sum_cd alpha1_ac F1_cd(t) alpha1_db
    )
    # orthonormalize
    F[2], Rho[2],Tau[2], O[2],Oinv[2] = orthonormalize_corr_mx(C[2], tau_convention)
    # compute generalized corrs / blocks of T^m
    Alpha[2] = F[2][1]
    G[2] = np.diag(1/Tau[2]) @ Oinv[2] @ ( F[1][1:] - Alpha[1] @ F[1][:-1] )
    H[2] = ( F[1][1:] - F[1][:-1] @ Alpha[1] ) @ O[2] @ np.diag(1/Rho[2])


    ## Step j
    for j in range(3, n_steps+1):
        C[j] = (
            F[j-1][2:] # F^(j-1)_ab(t+2)
            - F[j-1][1:-1] @ Alpha[j-1]
            - Alpha[j-1] @ F[j-1][1:-1]
            + Alpha[j-1] @ F[j-1][:-2] @ Alpha[j-1]
            + np.diag(Rho[j-1]) @ Oinv[j-1] @ F[j-2][:-4] @ O[j-1] @ np.diag(Tau[j-1])
            - ( G[j-1][1:-2] - Alpha[j-1] @ G[j-1][:-3] ) @ O[j-1] @ np.diag(Tau[j-1])
            - np.diag(Rho[j-1]) @ Oinv[j-1] @ ( H[j-1][1:-2] - H[j-1][:-3] @ Alpha[j-1] )
        )
        F[j], Rho[j],Tau[j], O[j],Oinv[j] = orthonormalize_corr_mx(C[j], tau_convention)
        Alpha[j] = F[j][1]
        G[j] = np.diag(1/Tau[j]) @ Oinv[j] @ (
            F[j-1][1:] - Alpha[j-1] @ F[j-1][:-1]
            - np.diag(Rho[j-1]) @ Oinv[j-1] @ H[j-1][:-2]

        )
        H[j] = ( 
            F[j-1][1:] - F[j-1][:-1] @ Alpha[j-1] 
            - G[j-1][:-2] @ O[j-1] @ np.diag(Tau[j-1])
        ) @ O[j] @ np.diag(1/Rho[j])
        
    return Alpha, Rho,Tau, O,Oinv, C,F,G,H

def load_block_lanczos_matrix(Alpha, Rho,Tau, O,Oinv):
    n_steps = len(Alpha)
    assert len(Rho) == len(Tau) == n_steps
    assert len(O) == len(Oinv) == n_steps
    bdim = Alpha[1].shape[0]
    for a in Alpha.values(): assert a.shape == (bdim, bdim)
    for r in Rho.values(): assert r.shape == (bdim,)
    for t in Tau.values(): assert t.shape == (bdim,)
    for o in O.values(): assert o.shape == (bdim,bdim)
    for o in Oinv.values(): assert o.shape == (bdim,bdim)

    Tm = np.zeros((n_steps,bdim, n_steps,bdim), dtype=object)
    for i, alpha in Alpha.items():
        Tm[i-1,:,i-1,:] = alpha
    for i in range(1, n_steps+1):
        # lower off-diag
        if i > 1:
            j = i - 1
            m = max(i,j)
            Tm[i-1,:,j-1,:] = np.diag(Rho[m]) @ Oinv[m]
        # upper off-diag
        if i < n_steps:
            j = i + 1
            m = max(i,j)
            Tm[i-1,:,j-1,:] = O[m] @ np.diag(Tau[m])
    return Tm

def make_block_krylov_coeffs(Alpha,Rho,Tau, O,Oinv, for_psi=False, dtype=object):
#     assert not for_psi # DEBUG
    
    # |v_ja> = \sum_bt T^t |v_1b> K^R_jt,ba = \sum_bt |kR_tb> K^R_jt,ba
    # <w_ja| = \sum_bt K^L_jt,ab <w_1b| T^t = \sum_bt K^L_jt,ab <kL_tb|
    n_steps = len(Alpha)
    assert tuple(sorted(Alpha.keys())) == tuple(range(1,n_steps+1))
    bdim = Alpha[1].shape[0]
    
    for k,v in Alpha.items(): assert v.shape == (bdim,bdim), k
    for k,v in O.items(): assert v.shape == (bdim,bdim), k
    for k,v in Oinv.items(): assert v.shape == (bdim,bdim), k

    shape = (n_steps, bdim,bdim) # K[j]_t,ba

    G = {j : O[j] @ np.diag(1/Rho[j]) for j in range(2, n_steps+1)}
    B = {j : np.diag(1/Tau[j]) @ Oinv[j] for j in range(2, n_steps+1)}

    Kr = {}
    Kr[1] = np.stack([np.eye(bdim, dtype=dtype)] + [np.zeros((bdim,bdim))]*(n_steps-1), axis=0)
    Kr[2] = np.stack([-Alpha[1] @ G[2], G[2]] 
                   + [np.zeros((bdim,bdim))]*(n_steps-2), axis=0)

    Kl = {}
    Kl[1] = np.stack([np.eye(bdim, dtype=dtype)] + [np.zeros((bdim,bdim))]*(n_steps-1), axis=0)
    Kl[2] = np.stack([-B[2] @ Alpha[1], B[2]] 
                   + [np.zeros((bdim,bdim))]*(n_steps-2), axis=0)

    for j in range(2, n_steps):
        dKr = np.zeros_like(Kr[j])
        for t in range(n_steps):
            if t > 0:
                dKr[t] += Kr[j][t-1]
            dKr[t] -= Kr[j][t] @ Alpha[j]
            dKr[t] -= Kr[j-1][t] @ O[j] @ np.diag(Tau[j])
        Kr[j+1] = dKr @ G[j+1]

        dKl = np.zeros_like(Kl[j])
        for t in range(n_steps):
            if t > 0:
                dKl[t] += Kl[j][t-1]
            dKl[t] -= Alpha[j] @ Kl[j][t]
            dKl[t] -= np.diag(Rho[j]) @ Oinv[j] @ Kl[j-1][t]
        Kl[j+1] = B[j+1] @ dKl
        
    if for_psi:
        # redef for application to |psi_a> i.e.
        # |v_ja> = \sum_bt T^t |psi_b> K^R_jt,ba
        # <w_ja| = \sum_bt K^L_jt,ab <psi_b| T^t
        Rfac = O[1] @ np.diag(1/Rho[1])
        Lfac = np.diag(1/Tau[1]) @ Oinv[1]
        Kr = {j : Rfac[None,:,:] @ K for j,K in Kr.items()}
        Kl = {j : K @ Lfac[None,:,:] for j,K in Kl.items()}
        
    return Kr, Kl

def mp_compute_all_eigs_block(T, progress=False):
    n_steps, bdim, *_ = T.shape
    assert T.shape == (n_steps, bdim, n_steps, bdim)    
    
    all_evals = {}
    all_evecs = {}
    all_k = range(1, n_steps+1)
    for k in (tqdm(all_k) if progress else all_k):
        evals, evecs = mnp.eig(T[:k, :, :k, :].reshape(k*bdim, k*bdim))
        evecs = evecs.reshape(k,bdim, k*bdim) # (ia)k -> iak

        all_evals[k] = evals
        all_evecs[k] = evecs
    return all_evals, all_evecs

def compute_all_eigs_block(T, progress=False):
    n_steps, bdim, *_ = T.shape
    assert T.shape == (n_steps, bdim, n_steps, bdim)
    
    def _eig(X):
        evals, evecs = np.linalg.eig(X)
        evecs = fix_sign(evecs)
        # SORT
#         sort_idx = np.argsort(-downcast_(evals))
        sort_idx = np.argsort(-np.round(downcast_(evals), 9)) # ~ DEBUG
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        return evals, evecs
    
    all_evals = {}
    all_evecs = {}
    all_k = range(1, n_steps+1)
    for k in (tqdm(all_k) if progress else all_k):
        evals, evecs = _eig(T[:k, :, :k, :].reshape(k*bdim, k*bdim))
        evecs = evecs.reshape(k,bdim, k*bdim) # block convention: (ia)k -> iak 
        all_evals[k] = evals
        all_evecs[k] = evecs
    return all_evals, all_evecs

def compute_all_eigs_and_CW_block(T, progress=False):
    n_steps, bdim, *_ = T.shape
    assert T.shape == (n_steps, bdim, n_steps, bdim)
    
    def _eig(X):
        evals, evecs = np.linalg.eig(X)
        evecs = fix_sign(evecs)
        # SORT
#         sort_idx = np.argsort(-downcast_(evals))
        sort_idx = np.argsort(-np.round(downcast_(evals), 6)) # ~ DEBUG
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        return evals, evecs
        
    all_evals = {}
    all_evals_CW = {}
    all_evecs = {}
    all_k = range(1, n_steps+1)
    for k in (tqdm(all_k) if progress else all_k):
        evals, evecs = _eig(T[:k, :, :k, :].reshape(k*bdim, k*bdim))
        evecs = evecs.reshape(k,bdim, k*bdim) # block convention: (ia)k -> iak 
        all_evals[k], all_evecs[k] = evals, evecs
        all_evals_CW[k], _ = _eig(T[1:k, :, 1:k, :].reshape((k-1)*bdim, (k-1)*bdim))
    return all_evals, all_evecs, all_evals_CW





def get_block_hermitian_subspace(evals, omega, O,Oinv, Tau,Rho,
    filter_ZZ=True
):
    k, bdim, dim_total = omega.shape
    assert dim_total == k*bdim
    assert evals.shape == (dim_total,)
    # just pass entire dicts
    assert O[1].shape == (bdim, bdim)
    assert Oinv[1].shape == (bdim, bdim)
    assert Tau[1].shape == (bdim,)
    assert Rho[1].shape == (bdim,)
    
    omega_inv = inv_block_evec(omega)
    
    ## Diagnosis by norm trick: compute Zr, Zl from derived quantities
    Zr = ( O[1] @ np.diag(Tau[1]) @ omega[0] ).conj().T
    Zl = omega_inv[:,0] @ np.diag(Rho[1]) @ Oinv[1]
    znorm = Zl / Zr # |Nk|^2
    
    # downcast for convenience
    evals = downcast_(evals)
    znorm = downcast_(znorm)
    
    # conditions on eigvals
    ev_is_re = np.isclose(evals.imag / np.abs(evals), 0)
    ev_is_cx = ~ev_is_re
    
    # conditions on norm
    znorm_same = np.isclose(znorm[:,0][:,None], znorm).all(1)
    znorm_real = np.isclose(znorm.imag/np.abs(znorm), 0).all(1)
    znorm_nonzero = ~np.isclose(znorm, 0).all(1)
    znorm_finite = np.isfinite(znorm).all(1)
    znorm_pos = (znorm.real > 0).all(1)
    
    is_herm = (ev_is_re
        & znorm_same & znorm_real & znorm_nonzero & znorm_finite & znorm_pos
    )
    
    if filter_ZZ:
        # for numerical reasons, look at ZZ = Zr* Zl also
        ZZ = Zr.conj() * Zl 
        ZZ = ZZ / mnp.abs(ZZ).sum(0)[None,:] # overlap fraction (no overall norm ambiguity)
        ZZ = downcast_(ZZ)

        # conditions on overlap product
        ZZ_real = np.isclose(ZZ.imag/np.abs(ZZ), 0).all(1)
        ZZ_nonzero = (~np.isclose(ZZ, 0)).all(1)
        ZZ_finite = np.isfinite(ZZ).all(1)

        is_herm &= ZZ_real & ZZ_finite & ZZ_nonzero
    return is_herm

def make_all_block_ritz_coeffs(all_evals, all_evecs, Kr,Kl, O,Oinv, Rho,Tau,
    inv_mp=False, normalize=True, fix_sign=True
):
    all_Pr, all_Pl = {}, {}
    for m, (ev, omega) in zip_items(all_evals, all_evecs):
        omega_inv = mp_inv_block_evec(omega) if inv_mp else inv_block_evec(omega)
        
        # Make Ritz coeffs (no einsum)
        Pr = (Kr[:m,:,:,:,None] * omega[:,None,None,:,:]).sum((0,3)) # \sum_ia itba1,i11ak ~ tbk
        Pr = Pr.transpose((2,0,1)) # ~ ktb
        Pl = (Kl[None,:m,:,:,:] * omega_inv[:,:,None,:,None]).sum((1,3)) # \sum_ia 1itab,ki1a1 ~ ktb

#         if normalize:
#             # Norm trick
#             Zr = ( O[1] @ np.diag(Tau[1]) @ omega[0] ).conj().T
#             Zl = omega_inv[:,0] @ np.diag(Rho[1]) @ Oinv[1]
#             znorm = Zl / Zr # |Nk|^2
#             znorm = znorm.mean(1)**0.5 # mean trivial over physical
#             znorm = mnp.abs(znorm) if inv_mp else np.abs(znorm) # TODO: CHECKME

#             Pr = Pr * znorm[:,None,None]
#             Pl = Pl / znorm[:,None,None]
        if normalize or fix_sign:
            Zr = ( O[1] @ np.diag(Tau[1]) @ omega[0] ).conj().T
            Zl = omega_inv[:,0] @ np.diag(Rho[1]) @ Oinv[1]
        if normalize:
            # Norm trick
            znorm = Zl / Zr # |Nk|^2
            znorm = znorm.mean(1)**0.5 # mean trivial over physical
            znorm = mnp.abs(znorm) if inv_mp else np.abs(znorm) # TODO: CHECKME
            Pr = Pr * znorm[:,None,None]
            Pl = Pl / znorm[:,None,None]
        if fix_sign:
            # enforce Zr_{k0} > 0
            abs_ = mnp.abs if inv_mp else np.abs
            phase = ( Zr[:,0] / abs_(Zr[:,0]) ) # ~ k
            phase = phase[:,None,None] # ~ k11
            Pr = Pr * phase
            Pl = Pl / phase
        
        all_Pr[m] = Pr
        all_Pl[m] = Pl
    return all_Pr, all_Pl


def make_all_block_J(
    C3ij, all_Pr, all_Pl, O,Oinv, Rho,Tau,
    rotate_C3=True
):
    Nt,_,bdim,_ = C3ij.shape
    assert C3ij.shape == (Nt,Nt,bdim,bdim)
    assert O[1].shape == (bdim,bdim)
    assert Oinv[1].shape == (bdim,bdim)
    assert Rho[1].shape == (bdim,)
    assert Tau[1].shape == (bdim,)
    assert Nt%2 == 0
    
    # get 3pt in initial Lanczos basis <vL_1a|...|vR_1b>
    A3ij = (np.diag(1/Tau[1]) @ Oinv[1])[None,None] @ C3ij @ (O[1] @ np.diag(1/Rho[1]))[None,None]
    if rotate_C3:
        A3ij = rotate_block_C3pt(A3ij) # (tf,tau) -> (sigma,tau)
    # trim and get in matrix form
    A3ij = A3ij[:Nt//2, :Nt//2]
    A3ij = A3ij.transpose(0,2, 1,3) # stab->satb
    A3ij = A3ij.reshape(Nt//2*bdim, Nt//2*bdim) # (sa)(tb)
    
    all_J = {}
    for m, (Pr,Pl) in zip_items(all_Pr, all_Pl):
        assert Pr.shape == (m*bdim, Nt//2, bdim)
        assert Pl.shape == (m*bdim, Nt//2, bdim)
        # get in matrix form
        Pl = Pl.reshape(m*bdim, Nt//2*bdim)
        Pr = Pr.reshape(m*bdim, Nt//2*bdim)
        Pr = Pr.T
        all_J[m] = Pl @ A3ij @ Pr
        
    return all_J

def compute_block_wvnorms_symm(Kr, Kl, Cij, O,Oinv, Rho,Tau, 
    downcast=True, matrixform=True, do_check=False
):
    print("WARNING: this version of wvnorms is technically incorrect")
    print("due to incorrect O,Oinv ambiguities, but works out correctly")
    print("because Oinv = O^H. DEPRECATE ME")
    
    Nt, bdim, _ = Cij.shape
    assert Cij.shape == (Nt, bdim,bdim)

    # get A_ab = <vL_1a|T|vR_1b>
    Aij = (np.diag(1/Tau[1]) @ Oinv[1])[None] @ Cij @ (O[1] @ np.diag(1/Rho[1]))[None]
    
    if downcast:
        Aij = downcast_(Aij)
        Kr = downcast_(Kr)
        Kl = downcast_(Kl)

    KKij = make_block_hankel(Aij)
    KKij = KKij[:Nt//2,:,:Nt//2,:] # satb
    
    # <vL_ia|vR_jb> = delta_ij delta_ab
    # = \sum_stcd Kl_isac <vL_1c| T^s T^t |vR_1d> Kr_jtdb
    # = \sum_stcd Kl_isac KK_sctd Kr_jtdb
    # = \sum_(sc)(td) Kl_(ia)(sc) KK_(sc)(td) Kr_(td)(jb)
    # <vR_ia|vR_jb>
    # = \sum_stcd Kr*_isca <vR_1c| T*^s T^t |vR_1d> Kr_jtdb
    # ~ \sum_stcd Kr*_isca KK_sctd Kr_jtdb
    # = \sum_(sc)(td) Kr*_(ia)(sc) KK_(sc)(td) Kr_(td)(jb)
    # <vL_ia|vL_jb>
    # = \sum_stcd Kl_isac <vL_1c| T^s T^t |vR_1d> Kl*_jtbd
    # = \sum_stcd Kl_isac KK_sctd Kl*_jtbd
    # = \sum_(sc)(td) Kl_(ia)(sc) KK_(sc)(td) Kl*_(td)(jb)
    if not matrixform:
        # v1: einsum
        if do_check:
            check = np.einsum('isac,sctd,jtdb->iajb', Kl, KKij, Kr)
            assert mnp.allclose(check.reshape(Nt//2*bdim, -1), np.eye(Nt//2*bdim))
        vnorm = np.einsum('isca,sctd,jtdb->iajb', Kr.conj(), KKij, Kr)
        wnorm = np.einsum('isac,sctd,jtbd->iajb', Kl, KKij, Kl.conj())

    else:
        # v2: matrixform
        Kr_ = Kr.transpose(1,2, 0,3).reshape(Nt//2*bdim, Nt//2*bdim) # jtdb -> tdjb -> (td)(jb)
        Kl_ = Kl.transpose(0,2, 1,3).reshape(Nt//2*bdim, Nt//2*bdim) # isac -> iasc -> (ia)(sc)
        KKij_ = KKij.reshape(Nt//2*bdim, Nt//2*bdim) # sctd->(sc)(td)
        if do_check:
            check = Kl_ @ KKij_ @ Kr_ # (ia)(sc),(sc)(td),(td)(jb) -> (ia)(jb)
            assert mnp.allclose(check, np.eye(Nt//2*bdim))
        vnorm = Kr_.conj().T @ KKij_ @ Kr_ # (ia)(sc)* (sc)(td) (td)(jb) -> (ia)(jb)
        wnorm = Kl_ @ KKij_ @ Kl_.conj().T
        vnorm = vnorm.reshape(Nt//2, bdim, Nt//2, bdim)
        wnorm = wnorm.reshape(Nt//2, bdim, Nt//2, bdim)

    return vnorm, wnorm # ~ iajb


def compute_block_wvnorms(Kr, Kl, Cij, O,Oinv, Rho,Tau, 
    downcast=True, matrixform=True, do_check=False
):
    if do_check: raise NotImplementedError
        
    ## Compute LL and RR vnorm and wnorm
    Nt, bdim, _ = Cij.shape
    assert Cij.shape == (Nt, bdim,bdim)

    # get A_ab = <vL_1a|T|vR_1b>
    R = (O[1] @ np.diag(1/Rho[1]))
    L = (np.diag(1/Tau[1]) @ Oinv[1])
    Aij_rr = (R.conj().T)[None] @ Cij @ R[None]
    Aij_ll = L[None] @ Cij @ (L.conj().T)[None]
    
    if downcast:
        Aij_rr = downcast_(Aij_rr)
        Aij_ll = downcast_(Aij_ll)
        Kr = downcast_(Kr)
        Kl = downcast_(Kl)

    KKij_rr = make_block_hankel(Aij_rr)[:Nt//2,:,:Nt//2,:] # satb
    KKij_ll = make_block_hankel(Aij_ll)[:Nt//2,:,:Nt//2,:] # satb
    
    # <vL_ia|vR_jb> = delta_ij delta_ab
    # = \sum_stcd Kl_isac <vL_1c| T^s T^t |vR_1d> Kr_jtdb
    # = \sum_stcd Kl_isac KK_sctd Kr_jtdb
    # = \sum_(sc)(td) Kl_(ia)(sc) KK_(sc)(td) Kr_(td)(jb)
    # <vR_ia|vR_jb>
    # = \sum_stcd Kr*_isca <vR_1c| T*^s T^t |vR_1d> Kr_jtdb
    # ~ \sum_stcd Kr*_isca KK_sctd Kr_jtdb
    # = \sum_(sc)(td) Kr*_(ia)(sc) KK_(sc)(td) Kr_(td)(jb)
    # <vL_ia|vL_jb>
    # = \sum_stcd Kl_isac <vL_1c| T^s T^t |vR_1d> Kl*_jtbd
    # = \sum_stcd Kl_isac KK_sctd Kl*_jtbd
    # = \sum_(sc)(td) Kl_(ia)(sc) KK_(sc)(td) Kl*_(td)(jb)
    if not matrixform:
        # v1: einsum
#         if do_check:
#             check = np.einsum('isac,sctd,jtdb->iajb', Kl, KKij, Kr)
#             assert mnp.allclose(check.reshape(Nt//2*bdim, -1), np.eye(Nt//2*bdim))
        vnorm = np.einsum('isca,sctd,jtdb->iajb', Kr.conj(), KKij_rr, Kr)
        wnorm = np.einsum('isac,sctd,jtbd->iajb', Kl, KKij_ll, Kl.conj())

    else:
        # v2: matrixform
        Kr_ = Kr.transpose(1,2, 0,3).reshape(Nt//2*bdim, Nt//2*bdim) # jtdb -> tdjb -> (td)(jb)
        Kl_ = Kl.transpose(0,2, 1,3).reshape(Nt//2*bdim, Nt//2*bdim) # isac -> iasc -> (ia)(sc)
        KKij_rr_ = KKij_rr.reshape(Nt//2*bdim, Nt//2*bdim) # sctd->(sc)(td)
        KKij_ll_ = KKij_ll.reshape(Nt//2*bdim, Nt//2*bdim) # sctd->(sc)(td)
#         if do_check:
#             check = Kl_ @ KKij_ @ Kr_ # (ia)(sc),(sc)(td),(td)(jb) -> (ia)(jb)
#             assert mnp.allclose(check, np.eye(Nt//2*bdim))
        vnorm = Kr_.conj().T @ KKij_rr_ @ Kr_ # (ia)(sc)* (sc)(td) (td)(jb) -> (ia)(jb)
        wnorm = Kl_ @ KKij_ll_ @ Kl_.conj().T
        vnorm = vnorm.reshape(Nt//2, bdim, Nt//2, bdim)
        wnorm = wnorm.reshape(Nt//2, bdim, Nt//2, bdim)
        
    return vnorm, wnorm # ~ iajb


def mp_compute_all_block_bounds(Cij, Kr, Kl, all_evecs, O,Oinv, Rho,Tau,
    downcast=False, matrixform=True, do_check=False, inv_mp=True,
    progress=False
):
    vnorm, wnorm = compute_block_wvnorms(Kr, Kl, Cij, O,Oinv, Rho,Tau, 
        downcast=downcast, matrixform=matrixform, do_check=do_check)
    
    Nt, bdim, _ = Cij.shape
    assert Cij.shape == (Nt,bdim,bdim)

    Beta = {m : o @ np.diag(t) for m,(o,t) in zip_items(O, Tau)}
    Gamma = {m : np.diag(r) @ oi for m,(r,oi) in zip_items(Rho, Oinv)}
#     for m, (b,g) in zip_items(Beta, Gamma):
#         if m == 1: continue
#         assert mnp.allclose(g, Tm[m-1,:,m-2,:])
#         assert mnp.allclose(b, Tm[m-2,:,m-1,:])

    all_Br = {}
    all_Bl = {}
    for m, omega in (tqdm(list(all_evecs.items())) if progress else all_evecs.items()):
        if m+1 not in Beta: continue

        # omega ~ iak
        omega_inv = mp_inv_block_evec(omega) if inv_mp else inv_block_evec(omega)

#         # Norm trick -> |Nk|^2
#         Zr = ( O[1] @ np.diag(Tau[1]) @ omega[0] ).conj().T
#         Zl = omega_inv[:,0] @ np.diag(Rho[1]) @ Oinv[1]
#         Nk2 = ( Zl / Zr ).mean(1) # ~ k  (mean trivial over physical)
#         Nk2 = mnp.abs(Nk2) if inv_mp else np.abs(Nk2) # TODO: CHECKME
        Nk2 = 1 # using ynorm from vnorm below assumes this

        # <y|y> factors (TODO: CHECKME AGAINST PR PL)
        if not matrixform:
            ynorm_r = np.einsum('iak,iajb,jbk->k', omega.conj(), vnorm[:m,:,:m,:], omega)
            ynorm_l = np.einsum('kia,iajb,kjb->k', omega_inv, wnorm[:m,:,:m,:], omega_inv.conj())
        else:
            omega_ = omega.reshape(m*bdim, m*bdim) # iak->(ia)k
            omega_inv_ = omega_inv.reshape(m*bdim, m*bdim) # kia->k(ia)
            vnorm_ = vnorm[:m,:,:m,:].reshape(m*bdim, m*bdim) # iajb->(ia)(jb)
            wnorm_ = wnorm[:m,:,:m,:].reshape(m*bdim, m*bdim) # iajb->(ia)(jb)
            ynorm_r = np.diag( omega_.conj().T @ vnorm_ @ omega_ ) # k(ia)*,(ia)(jb),(jb)k
            ynorm_l = np.diag( omega_inv_ @ wnorm_ @ omega_inv_.conj().T ) # k(ia),(ia)(jb),(jb)k*
            
        if downcast:
            ynorm_r, ynorm_l, Nk2 = downcast_(ynorm_r), downcast_(ynorm_l), downcast_(Nk2)

        Vr = vnorm[m+1-1, :, m+1-1, :][None,:,:] * (Nk2 / ynorm_r)[:,None,None] # 1ab * k11 -> kab
        Vl = wnorm[m+1-1, :, m+1-1, :][None,:,:] / (Nk2 * ynorm_l)[:,None,None] # 1ab / k11 -> kab
        
        # precomp
        Br = Gamma[m+1] @ omega[-1] # \sum_d G_(m+1)cd o_mdk ~ ck
        Bl = omega_inv[:,-1,:] @ Beta[m+1] # ka,ab->kb  \sum_d oi_kmd B_(m+1)dc ~ kc
        if downcast:
            Br = downcast_(Br)
            Bl = downcast_(Bl)
        # finish
        if not matrixform:
            Br = np.einsum('ak,kab,bk->k', Br.conj(), Vr, Br, optimize="optimal")
            Bl = np.einsum('ka,kab,kb->k', Bl, Vl, Bl.conj(), optimize="optimal")
        else:
            Br = Br.T # ak->ka
            Br = (Br[:,:,None].conj() * Vr * Br[:,None,:]).sum((1,2)) # ka1,kab,k1b->k
            Bl = (Bl[:,:,None] * Vl * Bl[:,None,:].conj()).sum((1,2)) # ka1,kab,k1b->k

        all_Br[m] = Br
        all_Bl[m] = Bl
    return all_Br, all_Bl

def make_all_block_ynorms(Cij, all_Pr, all_Pl, O,Oinv, Rho,Tau,
    downcast=False, do_check=True
):
    raise NotImplementedError("Fix L/R in Aij")
    
    Nt, bdim, _ = Cij.shape
    assert Cij.shape == (Nt, bdim,bdim)

    # get A_ab = <vL_1a|T|vR_1b>
    Aij = (np.diag(1/Tau[1]) @ Oinv[1])[None] @ Cij @ (O[1] @ np.diag(1/Rho[1]))[None]

    if downcast:
        Aij = downcast_(Aij)
        all_Pr = downcast_dict(all_Pr)
        all_Pl = downcast_dict(all_Pl)

    KKij = make_block_hankel(Aij)
    KKij = KKij[:Nt//2,:,:Nt//2,:] # satb

    all_ynorm_lr = {}
    all_ynorm_ll = {}
    all_ynorm_rr = {}
    for m, (Pr,Pl) in zip_items(all_Pr, all_Pl):
        d = m*bdim
        KKij_ = KKij[:m,:,:m,:].reshape(d,d) # satb -> (sa)(tb)
        Pr = Pr[:,:m,:].reshape(d, d).T # ktb -> (tb)k
        Pl = Pl[:,:m,:].reshape(d, d) # ktb -> k(tb)

        ynorm_lr = Pl @ KKij_ @ Pr
        if do_check:
            assert mnp.allclose(ynorm_lr, np.eye(bdim*m)) #, atol=1e-5, rtol=1e-5)
        all_ynorm_lr[m] = ynorm_lr
        all_ynorm_rr[m] = Pr.conj().T @ KKij_ @ Pr
        all_ynorm_ll[m] = Pl @ KKij_ @ Pl.conj().T
    
    return all_ynorm_rr, all_ynorm_ll, all_ynorm_lr

def make_all_block_ynorms_from_vnorms(vnorm, wnorm, all_evecs,
    O,Oinv, Rho,Tau,
    downcast=False, matrixform=True, inv_mp=True
):
    if downcast:
        vnorm, wnorm = downcast_(vnorm), downcast_(wnorm)
        all_evecs = {m : downcast_(omega) for m,omega in all_evecs.items()}
    
    all_ynorm_rr = {}
    all_ynorm_ll = {}
    for m, omega in all_evecs.items():
        omega_inv = mp_inv_block_evec(omega) if inv_mp else inv_block_evec(omega)
        if downcast: omega_inv = downcast_(omega_inv)

        # Norm trick -> |Nk|^2
        Zr = ( O[1] @ np.diag(Tau[1]) @ omega[0] ).conj().T
        Zl = omega_inv[:,0] @ np.diag(Rho[1]) @ Oinv[1]
        Nk2 = ( Zl / Zr ).mean(1) # ~ k  (mean trivial over physical)
        Nk2 = mnp.abs(Nk2) if inv_mp else np.abs(Nk2) # TODO: CHECKME
        assert mnp.allclose(Nk2, 1) # NOISELESS ONLY, TESTME WHEN NOT

        if matrixform:
            _,bdim,_ = omega.shape
            assert omega.shape == (m, bdim, m*bdim)
            
            omega_ = omega.reshape(m*bdim, m*bdim) # iak->(ia)k
            omega_inv_ = omega_inv.reshape(m*bdim, m*bdim) # kia->k(ia)
            vnorm_ = vnorm[:m,:,:m,:].reshape(m*bdim, m*bdim) # iajb->(ia)(jb)
            wnorm_ = wnorm[:m,:,:m,:].reshape(m*bdim, m*bdim) # iajb->(ia)(jb)
            ynorm_r = omega_.conj().T @ vnorm_ @ omega_ # k(ia)*,(ia)(jb),(jb)k
            ynorm_l = omega_inv_ @ wnorm_ @ omega_inv_.conj().T # k(ia),(ia)(jb),(jb)k*
        else:
            ynorm_r = np.einsum('iak,iajb,jbl->kl', omega.conj(), vnorm[:m,:,:m,:], omega)
            ynorm_l = np.einsum('kia,iajb,ljb->kl', omega_inv, wnorm[:m,:,:m,:], omega_inv.conj())

        all_ynorm_rr[m] = ynorm_r * Nk2
        all_ynorm_ll[m] = ynorm_l / Nk2
        
    return all_ynorm_rr, all_ynorm_ll

def make_all_block_overlaps(all_evecs, O,Oinv, Rho,Tau,
    normalize=True, fix_sign=True,
    inv_mp=False, downcast=True,
):
    # WARNING: sometimes gives Zr*,Zl* vs non-Ritz version for cx Zs
    all_Zr, all_Zl = {}, {}
    for m, omega in all_evecs.items():
        
        omega_inv = mp_inv_block_evec(omega) if inv_mp else inv_block_evec(omega)
        if downcast: omega_inv = downcast_(omega_inv)

        Zr = ( O[1] @ np.diag(Tau[1]) @ omega[0] ).conj().T # ~ ka
        Zl = omega_inv[:,0] @ np.diag(Rho[1]) @ Oinv[1] # ~ ka
        
        if normalize:
            # Norm trick -> |Nk|^2
            Nk = ( Zl / Zr )**0.5 # ~ ka
            Nk = mnp.abs(Nk) if inv_mp else np.abs(Nk) # TODO: CHECKME
            Nk = Nk.mean(1)[:,None] # mean trivial over physical
            Zr = Nk.conj() * Zr
            Zl = Zl / Nk
            
        if downcast:
            Zr, Zl = downcast_(Zr), downcast_(Zl)

        if fix_sign:
            # fix Zr_k0 > 0
            phase = ( Zr[:,0] / mnp.abs(Zr[:,0]) ).conj()[:,None] # ~ k1
            Zr, Zl = Zr*phase, Zl*phase
            
        all_Zr[m] = Zr
        all_Zl[m] = Zl
    return all_Zr, all_Zl

def compute_block_kps(all_evals, true_ev, Z):
    n_states, r = Z.shape
    ev_inf = true_ev.min()
    C0 = Z.conj().T @ Z # C_ij(0)

    M, ev = unpack_ragged_dict(all_evals) # ev ~ km
    ev = downcast_(ev).real

    Gamma = 1 + 2 * (true_ev[:-r] - true_ev[r:]) / (true_ev[r:] - ev_inf) # ~ k
    Tmi = np.array([[
        (scipy.special.chebyt(m-k-1)(Gamma[k]) if m-1 >= k else np.nan)
        for m in M] for k in range(n_states-r)])

    all_K = {} # {m : K^(m)_i}
    for m in M:
        Ki = [1.]
        for i in range(1, m*r):
            Ki.append(np.product([
                (ev[j,m-1] - ev_inf) / (ev[j,m-1] - true_ev[i])
                for j in range(i)
            ]))
        all_K[m] = np.array(Ki)
    _, K = unpack_ragged_dict(all_K) # ~ km
    K = K[:n_states-r]

    X = Z @ mnp.inv(C0) @ Z.conj().T # ~ k k'
    tk = [mnp.inv(X[k:k+r, k:k+r])[0,0] for k in range(n_states-r)]
    tk = downcast_(tk).real

    lhs = (true_ev[:,None] - ev) / (true_ev[:,None] - ev_inf)
    rhs = (K**2 / Tmi**2) * (tk-1)[:,None]
    
    print(lhs.shape, rhs.shape)

    print("Satisfied:", np.all( (lhs[:n_states-r] <= rhs) | ~np.isfinite(rhs) ))
    return lhs, rhs

def compute_block_kps_extrap(all_evals, true_ev, Z, M):
    n_states, r = Z.shape
    ev_inf = true_ev.min()
    C0 = Z.conj().T @ Z # C_ij(0)

    M_, ev = unpack_ragged_dict(all_evals) # ev ~ km
    ev = downcast_(ev).real
    
    m_max = M_.max()

    Gamma = 1 + 2 * (true_ev[:-r] - true_ev[r:]) / (true_ev[r:] - ev_inf) # ~ k
    Tmi = np.array([[
        (scipy.special.chebyt(m-k-1)(Gamma[k]) if m-1 >= k else np.nan)
        for m in M] for k in range(n_states-r)])

    all_K = {} # {m : K^(m)_i}
    for m in M:
        Ki = [1.]
        for i in range(1, min(m*r, m_max*r)):
            m_ = min(m_max, m)
            Ki.append(np.product([
                (ev[j,m_-1] - ev_inf) / (ev[j,m_-1] - true_ev[i])
                for j in range(i)
            ]))
        all_K[m] = np.array(Ki)
    _, K = unpack_ragged_dict(all_K) # ~ km
    K = K[:n_states-r]

    X = Z @ mnp.inv(C0) @ Z.conj().T # ~ k k'
    tk = [mnp.inv(X[k:k+r, k:k+r])[0,0] for k in range(n_states-r)]
    tk = downcast_(tk).real

    lhs = (true_ev[:,None] - ev) / (true_ev[:,None] - ev_inf)
    rhs = (K**2 / Tmi**2) * (tk-1)[:,None]
    
    print(lhs.shape, rhs.shape)

    print("Satisfied:", np.all( 
        (lhs[:n_states-r] <= rhs[:, :m_max]) | ~np.isfinite(rhs[:, :m_max]) ))
    return lhs, rhs

def compute_all_eigs_and_CW_block_(T, progress=False):
    n_steps, bdim, *_ = T.shape
    assert T.shape == (n_steps, bdim, n_steps, bdim)
    
    def _eig(X):
        evals, evecs = np.linalg.eig(X)
        evecs = fix_sign(evecs)
        # SORT
#         sort_idx = np.argsort(-downcast_(evals))
        sort_idx = np.argsort(-np.round(downcast_(evals), 6)) # ~ DEBUG
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]
        return evals, evecs
    def _ko(T_, a):
        return np.delete(np.delete(T_, a, axis=0), a, axis=1)
        
    all_evals = {}
    all_evals_CW = {}
    all_evecs = {}
    all_k = range(1, n_steps+1)
    for k in (tqdm(all_k) if progress else all_k):
        T_ = T[:k, :, :k, :].reshape(k*bdim, k*bdim) # iajb->(ia)(jb)
        
        evals, evecs = _eig(T_)
        evecs = evecs.reshape(k,bdim, k*bdim) # block convention: (ia)k -> iak 
        all_evals[k], all_evecs[k] = evals, evecs
        
        CW = []
        for a in range(bdim):
            evals_CW, _ = _eig(_ko(T_, a))
            CW.append(evals_CW)
        all_evals_CW[k] = np.stack(CW, axis=1)
    return all_evals, all_evecs, all_evals_CW


### Beta, Gamma version

def orthonormalize_corr_mx_(Ct, tau_convention=lambda norm: norm**0.5):
    # Ct ~ t,ab
    bdim = Ct.shape[-1]
    norm, rot = mnp.eig(Ct[0])
    unrot = mnp.inv(rot)
    assert mnp.allclose(rot @ np.diag(norm) @ unrot, Ct[0])
    
    tau = tau_convention(norm)
    rho = norm / tau
    Gamma = np.diag(rho) @ unrot
    Beta = rot @ np.diag(tau)
    assert mnp.allclose(Beta @ Gamma, Ct[0])
    assert mnp.allclose(Gamma @ Beta, np.diag(norm))
    
    F = mnp.inv(Beta) @ Ct @ mnp.inv(Gamma)
    assert mnp.allclose(F[0], np.eye(bdim))
    
    return F, Beta, Gamma

def run_block_recursion_(Ct, tau_convention=lambda norm: norm**0.5):
    Nt,_,bdim = Ct.shape
    assert Ct.shape == (Nt, bdim,bdim)
    n_steps = Nt//2
    
    ## Setup
    C = {1 : Ct} # <r_ia|T^t|r_ib>
    F = {} # <w_ia|T^t|v_jb>
    G = {} # <w_ja|T^t|v_(j-1)b>
    H = {} # <w_(j-1)a|T^t|v_jb>

    Alpha = {}
    Beta = {}
    Gamma = {}

    ## Step 1:
    F[1], Beta[1], Gamma[1] = orthonormalize_corr_mx_(C[1], tau_convention)
    Alpha[1] = F[1][1]
    
    if Nt <= 2:
        return Alpha,Beta,Gamma, C,F,G,H

    ## Step 2:
    # compute C2(t) = <s_2a|T^t|r_2b>
    C[2] = ( # np.zeros((Nt-2, bdim, bdim), dtype=object)
        F[1][2:] # F1_ab(t+2)
        - Alpha[1] @ F[1][1:-1] # \sum_c alpha1_ac F1_cb(t+1)
        - F[1][1:-1] @ Alpha[1] # \sum_c F1_ac alpha1_cb
        + Alpha[1] @ F[1][:-2] @ Alpha[1] # \sum_cd alpha1_ac F1_cd(t) alpha1_db
    )
    # orthonormalize
    F[2], Beta[2], Gamma[2] = orthonormalize_corr_mx_(C[2], tau_convention)
    # compute generalized corrs / blocks of T^m
    Alpha[2] = F[2][1]
    G[2] = mnp.inv(Beta[2]) @ ( F[1][1:] - Alpha[1] @ F[1][:-1] )
    H[2] = ( F[1][1:] - F[1][:-1] @ Alpha[1] ) @ mnp.inv(Gamma[2])
    
    ## Step j
    for j in range(3, n_steps+1):
        C[j] = (
            F[j-1][2:] # F^(j-1)_ab(t+2)
            - F[j-1][1:-1] @ Alpha[j-1]
            - Alpha[j-1] @ F[j-1][1:-1]
            + Alpha[j-1] @ F[j-1][:-2] @ Alpha[j-1]
            + Gamma[j-1] @ F[j-2][:-4] @ Beta[j-1]
            - ( G[j-1][1:-2] - Alpha[j-1] @ G[j-1][:-3] ) @ Beta[j-1]
            - Gamma[j-1] @ ( H[j-1][1:-2] - H[j-1][:-3] @ Alpha[j-1] )
        )
        F[j], Beta[j], Gamma[j] = orthonormalize_corr_mx_(C[j], tau_convention)
        Alpha[j] = F[j][1]
        G[j] = mnp.inv(Beta[j]) @ (
            F[j-1][1:] - Alpha[j-1] @ F[j-1][:-1]
            - Gamma[j-1] @ H[j-1][:-2]

        )
        H[j] = ( 
            F[j-1][1:] - F[j-1][:-1] @ Alpha[j-1] 
            - G[j-1][:-2] @ Beta[j-1]
        ) @ mnp.inv(Gamma[j])
        
    return Alpha,Beta,Gamma, C,F,G,H


def load_block_lanczos_matrix_(Alpha, Beta, Gamma):
    n_steps = len(Alpha)
    assert len(Beta) == len(Gamma) == n_steps
    bdim = Alpha[1].shape[0]
    for a in Alpha.values(): assert a.shape == (bdim, bdim)
    for b in Beta.values(): assert b.shape == (bdim,bdim)
    for g in Gamma.values(): assert g.shape == (bdim,bdim)

    Tm = np.zeros((n_steps,bdim, n_steps,bdim), dtype=object)
    for i, alpha in Alpha.items():
        Tm[i-1,:,i-1,:] = alpha
    for i in range(1, n_steps+1):
        # lower off-diag
        if i > 1:
            j = i - 1
            m = max(i,j)
            Tm[i-1,:,j-1,:] = Gamma[m]
        # upper off-diag
        if i < n_steps:
            j = i + 1
            m = max(i,j)
            Tm[i-1,:,j-1,:] = Beta[m]
    return Tm

def make_block_krylov_coeffs_(Alpha, Beta, Gamma, for_psi=False, inv_mp=False):
    assert not for_psi # DEBUG
    
    inv = mnp.inv if inv_mp else np.linalg.inv 
    
    # |v_ja> = \sum_bt T^t |v_1b> K^R_jt,ba = \sum_bt |kR_tb> K^R_jt,ba
    # <w_ja| = \sum_bt K^L_jt,ab <w_1b| T^t = \sum_bt K^L_jt,ab <kL_tb|
    n_steps = len(Alpha)
    assert tuple(sorted(Alpha.keys())) == tuple(range(1,n_steps+1))
    bdim = Alpha[1].shape[0]
    
    for k,v in Alpha.items(): assert v.shape == (bdim,bdim), k
    for k,v in Beta.items(): assert v.shape == (bdim,bdim), k
    for k,v in Gamma.items(): assert v.shape == (bdim,bdim), k

    shape = (n_steps, bdim,bdim) # K[j]_t,ba

    G = {j : inv(g) for j,g in Gamma.items()}
    B = {j : inv(b) for j,b in Beta.items()}

    Kr = {}
    Kr[1] = np.stack([np.eye(bdim, dtype=object)] + [np.zeros((bdim,bdim))]*(n_steps-1), axis=0)
    Kr[2] = np.stack([-Alpha[1] @ G[2], G[2]] 
                   + [np.zeros((bdim,bdim))]*(n_steps-2), axis=0)

    Kl = {}
    Kl[1] = np.stack([np.eye(bdim, dtype=object)] + [np.zeros((bdim,bdim))]*(n_steps-1), axis=0)
    Kl[2] = np.stack([-B[2] @ Alpha[1], B[2]] 
                   + [np.zeros((bdim,bdim))]*(n_steps-2), axis=0)

    for j in range(2, n_steps):
        dKr = np.zeros_like(Kr[j])
        for t in range(n_steps):
            if t > 0:
                dKr[t] += Kr[j][t-1]
            dKr[t] -= Kr[j][t] @ Alpha[j]
            dKr[t] -= Kr[j-1][t] @ Beta[j]
        Kr[j+1] = dKr @ G[j+1]

        dKl = np.zeros_like(Kl[j])
        for t in range(n_steps):
            if t > 0:
                dKl[t] += Kl[j][t-1]
            dKl[t] -= Alpha[j] @ Kl[j][t]
            dKl[t] -= Gamma[j] @ Kl[j-1][t]
        Kl[j+1] = B[j+1] @ dKl
        
    if for_psi:
        # redef for application to |psi_a> i.e.
        # |v_ja> = \sum_bt T^t |psi_b> K^R_jt,ba
        # <w_ja| = \sum_bt K^L_jt,ab <psi_b| T^t
        Kr = {j : G[1][None,:,:] @ K for j,K in Kr.items()}
        Kl = {j : K @ B[1][None,:,:] for j,K in Kl.items()}
        
    return Kr, Kl

def make_all_block_ritz_coeffs_(all_evals, all_evecs, Kr,Kl, Beta, Gamma,
    normalize=True, fix_sign=True,
    inv_mp=False,
):
    all_Pr, all_Pl = {}, {}
    for m, (ev, omega) in zip_items(all_evals, all_evecs):
        omega_inv = mp_inv_block_evec(omega) if inv_mp else inv_block_evec(omega)
        
        # Make Ritz coeffs (no einsum)
        Pr = (Kr[:m,:,:,:,None] * omega[:,None,None,:,:]).sum((0,3)) # \sum_ia itba1,i11ak ~ tbk
        Pr = Pr.transpose((2,0,1)) # ~ ktb
        Pl = (Kl[None,:m,:,:,:] * omega_inv[:,:,None,:,None]).sum((1,3)) # \sum_ia 1itab,ki1a1 ~ ktb

        if normalize or fix_sign:
            Zr = ( Beta[1] @ omega[0] ).conj().T
            Zl = omega_inv[:,0] @ Gamma[1]
        if normalize:
            # Norm trick
            znorm = Zl / Zr # |Nk|^2
            znorm = znorm.mean(1)**0.5 # mean trivial over physical
            znorm = mnp.abs(znorm) if inv_mp else np.abs(znorm) # TODO: CHECKME
            Pr = Pr * znorm[:,None,None]
            Pl = Pl / znorm[:,None,None]
        if fix_sign:
            # enforce Zr_{k0} > 0
            abs_ = mnp.abs if inv_mp else np.abs
            phase = ( Zr[:,0] / abs_(Zr[:,0]) ) # ~ k
            phase = phase[:,None,None] # ~ k11
            Pr = Pr * phase
            Pl = Pl / phase
            
        all_Pr[m] = Pr
        all_Pl[m] = Pl
    return all_Pr, all_Pl

def make_all_block_overlaps_(all_evecs, Beta, Gamma,
    normalize=True, fix_sign=True,
    inv_mp=False, downcast=True,
):
    # WARNING: sometimes gives Zr*,Zl* vs non-Ritz version for cx Zs
    all_Zr, all_Zl = {}, {}
    for m, omega in all_evecs.items():
        omega_inv = mp_inv_block_evec(omega) if inv_mp else inv_block_evec(omega)
        if downcast: omega_inv = downcast_(omega_inv)

        Zr = ( Beta[1] @ omega[0] ).conj().T # ~ ka
        Zl = omega_inv[:,0] @ Gamma[1] # ~ ka
        
        if normalize:
            # Norm trick -> |Nk|^2
            Nk = ( Zl / Zr )**0.5 # ~ ka
            Nk = mnp.abs(Nk) if inv_mp else np.abs(Nk) # TODO: CHECKME
            Nk = Nk.mean(1)[:,None] # ~ k1, mean trivial over physical
            Zr = Nk.conj() * Zr
            Zl = Zl / Nk
            
        if downcast:
            Zr, Zl = downcast_(Zr), downcast_(Zl)
            
        if fix_sign:
            assert downcast
            # fix Zr_k0 > 0
            phase = ( Zr[:,0] / np.abs(Zr[:,0]) ).conj()[:,None] # ~ k1
            Zr, Zl = Zr*phase, Zl*phase
            
        all_Zr[m] = Zr
        all_Zl[m] = Zl
    return all_Zr, all_Zl


def make_block_ynorm_symm_(Cij, all_Pr, all_Pl, Beta, Gamma,
    downcast=False, do_check=True, inv_mp=False
):
    Nt, bdim, _ = Cij.shape
    assert Cij.shape == (Nt, bdim,bdim)

    inv = mnp.inv if inv_mp else np.linalg.inv
    
    # get A_ab = <vL_1a|T|vR_1b>
    Aij = inv(Beta[1])[None] @ Cij @ inv(Gamma[1])[None]

    if downcast:
        Aij = downcast_(Aij)
        all_Pr = downcast_dict(all_Pr)
        all_Pl = downcast_dict(all_Pl)

    KKij = make_block_hankel(Aij)
    KKij = KKij[:Nt//2,:,:Nt//2,:] # satb

    all_ynorm_lr = {}
#     all_ynorm_ll = {}
#     all_ynorm_rr = {}
    for m, (Pr,Pl) in zip_items(all_Pr, all_Pl):
        d = m*bdim
        KKij_ = KKij[:m,:,:m,:].reshape(d,d) # satb -> (sa)(tb)
        Pr = Pr[:,:m,:].reshape(d, d).T # ktb -> (tb)k
        Pl = Pl[:,:m,:].reshape(d, d) # ktb -> k(tb)

        ynorm_lr = Pl @ KKij_ @ Pr
        if do_check:
            assert mnp.allclose(ynorm_lr, np.eye(bdim*m)) #, atol=1e-5, rtol=1e-5)
        all_ynorm_lr[m] = ynorm_lr
#         all_ynorm_rr[m] = Pr.conj().T @ KKij_ @ Pr
#         all_ynorm_ll[m] = Pl @ KKij_ @ Pl.conj().T
    
#     return all_ynorm_rr, all_ynorm_ll, all_ynorm_lr
    return all_ynorm_lr


def make_all_block_overlaps_PrPl_(all_Pr, all_Pl, Cij, Beta,Gamma,
    inv_mp=False
):
    # WARNING: sometimes gives Zr*,Zl* vs non-Ritz version for cx Zs
    Nt, bdim,_ = Cij.shape
    assert Cij.shape == (Nt,bdim,bdim)    
    inv = mnp.inv if inv_mp else np.linalg.inv
    
    R = inv(Gamma[1])
    L = inv(Beta[1])
    
    CR = Cij @ R[None] # ~ tab
    LC = L[None] @ Cij # ~ tab
    CR = CR.transpose(1,0,2) # ~ atb
    LC = LC.transpose(2,0,1) # ~ bta
        
    all_Zr, all_Zl = {}, {}
    for m, (Pr, Pl) in zip_items(all_Pr, all_Pl):
        # Zr*_ka = <psi_a|yR_k> = \sum_tb <psi_a|T^t|vR_1b> Pr_ktb
        # = \sum_tbc <psi_a|T^t|psi_c> O^1_cb / Rho^1_b Pr_ktb
        # = \sum_tbc C(t)_ac O^1_cb / Rho^1_b Pr_ktb
        # = \sum_tbc C(t)_ac R_cb Pr_ktb
        # = \sum_tb CR_atb Pr_ktb
        Zr = (CR[None,:,:m,:] * Pr[:,None,:m,:]).sum((2,3)) # 1atb,k1tb->ka
        Zr = Zr.conj()
        
        # Zl_ka = <yL_k|psi_a> = \sum_tb Pl_ktb <vL_1b|T^t|psi_a>
        # = \sum_tbd Pl_ktb L_bd <psi_d|T^t|psi_a>
        # = \sum_tbd Pl_ktb L_bd C(t)_da
        # = \sum_tb Pl_ktb [\sum_b L_bd C(t)_da = LC_atb]
        # = \sum_tb Pl_ktb LC_atb # NOTE: swapped b<->a vs def
        Zl = (Pl[:,None,:m,:] * LC[None,:,:m,:]).sum((2,3)) # k1tb,1atb->ka
        
        all_Zr[m] = Zr
        all_Zl[m] = Zl
    return all_Zr, all_Zl