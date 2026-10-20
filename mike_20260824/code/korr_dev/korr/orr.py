#!/usr/bin/env python3
"""Implementation of generic oblique Rayleigh-Ritz.
"""

import numpy as np
from .util import make_hankel, make_block_hankel

## Utility
def adjoint(x):
    return x.conj().swapaxes(-2,-1)
def diag(x):
    return np.diagonal(x, axis1=-2, axis2=-1)


## Residual bounds
def _make_block_rr_bound_P(ev, P, H0ab, H1ab, H2ab, abs=np.abs, real=np.real):
    return (
        diag(adjoint(P) @ H2ab @ P)
        - 2 * real(ev) * diag(adjoint(P) @ H1ab @ P)
        + abs(ev)**2 * diag(adjoint(P) @ H0ab @ P)
    ) / diag(adjoint(P) @ H0ab @ P)

# TODO: redo w/ Gab, Mab arguments
def make_block_rr_bounds_PrPl(Cab, ev, Pr, Pl, abs=np.abs, real=np.real):
    # TODO: better checking for extra timeslices in Cab vs m
    *batch, Nt, bdim, _ = Cab.shape
    assert Cab.shape == (*batch, Nt, bdim,bdim)
    
    *batch_, m, bdim_, d = Pr.shape
    assert batch == batch_
    assert bdim == bdim_
    assert d == m*bdim
    assert Pl.shape == (*batch, d, m, bdim)
    
    assert Nt >= 2*(m+1), f"Need more timeslices to compute res bound: {Nt} < {2*(m+1)}"
        
    H0ab = make_block_hankel(Cab, n=0)
    H2ab = H0ab[..., 1:m+1,:, 1:m+1,:]
    H0ab = H0ab[..., :m,:, :m,:]
    H1ab = make_block_hankel(Cab, n=1)
    H1ab = H1ab[..., :m,:, :m,:]
    
    Pr = Pr.reshape(*batch, d,d)
    Pl = Pl.reshape(*batch, d,d)
    H0ab = H0ab.reshape(*batch, d,d)
    H1ab = H1ab.reshape(*batch, d,d)
    H2ab = H2ab.reshape(*batch, d,d)
    
    Br = _make_block_rr_bound_P(ev, Pr, H0ab, H1ab, H2ab, abs=abs, real=real)
    Bl = _make_block_rr_bound_P(ev, Pl.conj().swapaxes(-2,-1), H0ab, H1ab, H2ab, abs=abs, real=real)
    
    return Br, Bl

# TODO: redo w/ Gab, Mab arguments
def make_block_rr_bounds_PrPl_offdiag(Cab_rr, Cab_ll, ev, Pr, Pl, abs=np.abs, real=np.real):
    *batch, Nt, bdim, _ = Cab_rr.shape
    assert Cab_rr.shape == (*batch, Nt, bdim,bdim)
    assert Cab_ll.shape == (*batch, Nt, bdim,bdim)
    
    *batch_, m, bdim_, d = Pr.shape
    assert batch == batch_
    assert bdim == bdim_
    assert d == m*bdim
    assert Pl.shape == (*batch, d, m, bdim)
    Pr = Pr.reshape(*batch, d,d)
    Pl = Pl.reshape(*batch, d,d)

    # rr
    H0ab = make_block_hankel(Cab_rr, n=0)
    H2ab = H0ab[..., 1:m+1,:, 1:m+1,:]
    H0ab = H0ab[..., :m,:, :m,:]
    H1ab = make_block_hankel(Cab_rr, n=1)
    H1ab = H1ab[..., :m,:, :m,:]
    
    H0ab = H0ab.reshape(*batch, d,d)
    H1ab = H1ab.reshape(*batch, d,d)
    H2ab = H2ab.reshape(*batch, d,d)
    
    Br = _make_block_rr_bound_P(ev, Pr, H0ab, H1ab, H2ab, abs=abs, real=real)
    
    # ll
    H0ab = make_block_hankel(Cab_ll, n=0)
    H2ab = H0ab[..., 1:m+1,:, 1:m+1,:]
    H0ab = H0ab[..., :m,:, :m,:]
    H1ab = make_block_hankel(Cab_ll, n=1)
    H1ab = H1ab[..., :m,:, :m,:]
    
    H0ab = H0ab.reshape(*batch, d,d)
    H1ab = H1ab.reshape(*batch, d,d)
    H2ab = H2ab.reshape(*batch, d,d)
    
    Bl = _make_block_rr_bound_P(ev, Pl.conj().swapaxes(-2,-1), H0ab, H1ab, H2ab, abs=abs, real=real)
    
    return Br, Bl
        

## Symmetric Rayleigh Quotients
def make_block_rr_srq(Gab, Mab, P):
    *batch, Nt,bdim, _,_ = Gab.shape
    assert Gab.shape == (*batch, Nt,bdim, Nt,bdim)
    assert Mab.shape == Gab.shape
    
    # note: assumes P already in dxd form
    # assume P ~ Pr; must take Pl^H before passing
    *batch_, m, bdim, d = P.shape
    assert batch_ == batch
    assert d == m*bdim
    P = P.reshape(*batch, d,d)
    
    Gab = Gab[..., :m,:, :m,:].reshape(*batch, d,d)
    Mab = Mab[..., :m,:, :m,:].reshape(*batch, d,d)
    
    norm = diag(adjoint(P) @ Gab @ P)
    return diag(adjoint(P) @ Mab @ P) / norm, norm

## Off-diag code (redundant w/ make_block_rr_srq above)
def make_offdiag_ynorms(Pr, Pl, Clr, Crr, Cll):
    Glr = make_hankel(Clr)
    Gll = make_hankel(Cll)
    Grr = make_hankel(Crr)

    *batch, m, n_states = Pr.shape # ~ *tk
    assert Pl.shape == (*batch, n_states, m) # ~ *kt
    assert Glr.shape == (*batch, m, m)
    assert Gll.shape == (*batch, m, m)
    assert Grr.shape == (*batch, m, m)
    
    ynorm_lr = np.einsum('...ks,...st,...tk->...k', Pl, Glr, Pr, optimize=True)
    ynorm_rr = np.einsum('...sk,...st,...tk->...k', Pr.conj(), Grr, Pr, optimize=True)
    ynorm_ll = np.einsum('...ks,...st,...kt->...k', Pl, Gll, Pl.conj(), optimize=True)
    
    return ynorm_lr, ynorm_rr, ynorm_ll

def make_offdiag_ynorms_and_srq(Pr, Pl, Clr, Crr, Cll):
    Glr = make_hankel(Clr)
    Gll = make_hankel(Cll)
    Grr = make_hankel(Crr)

    *batch, m, n_states = Pr.shape # ~ *tk
    assert Pl.shape == (*batch, n_states, m) # ~ *kt
    assert Glr.shape == (*batch, m, m)
    assert Gll.shape == (*batch, m, m)
    assert Grr.shape == (*batch, m, m)
    
    ynorm_lr = np.einsum('...ks,...st,...tk->...k', Pl, Glr, Pr, optimize=True)
    ynorm_rr = np.einsum('...sk,...st,...tk->...k', Pr.conj(), Grr, Pr, optimize=True)
    ynorm_ll = np.einsum('...ks,...st,...kt->...k', Pl, Gll, Pl.conj(), optimize=True)
    
    Mll = make_hankel(Cll, 1)
    Mrr = make_hankel(Crr, 1)
    
    srq_rr = np.einsum('...sk,...st,...tk->...k', Pr.conj(), Mrr, Pr, optimize=True)
    srq_rr = srq_rr / ynorm_rr
    
    srq_ll = np.einsum('...ks,...st,...kt->...k', Pl, Mll, Pl.conj(), optimize=True)
    srq_ll = srq_ll / ynorm_ll
    
    return ynorm_lr, ynorm_rr, ynorm_ll, srq_rr, srq_ll


## Cullum-Willoughby
def compute_cw_rr_matrices(G, M, Ct):
    # TODO: MORE TESTING
    # G_st = H0_st - C(s) C(t) / C(0)
    # M_st = H1_st - C(s) C(1+t) / C(0) - C(1+s) C(t) / C(0) + C(s) C(1) C(t) / C(0)^2

    *batch, m, _ = G.shape
    assert G.shape == (*batch, m, m)
    assert M.shape == (*batch, m, m)
    assert Ct.shape == (*batch, 2*m)
    
    C0 = Ct[...,0][...,None,None]
    C1 = Ct[...,1][...,None,None]
    Gcw = G - Ct[..., :m,None] * Ct[..., None,:m] / C0
    Mcw = (M
        - Ct[..., :m,None] * Ct[..., None,1:m+1] / C0
        - Ct[..., 1:m+1,None] * Ct[..., None,:m] / C0
        + Ct[..., :m,None] * C1 * Ct[...,None,:m] / C0**2
    )

    # first row/col all zero
    Gcw = Gcw[..., 1:,1:]
    Mcw = Mcw[..., 1:,1:]
    return Gcw, Mcw

