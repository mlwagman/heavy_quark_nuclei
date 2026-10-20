#!/usr/bin/env python3
"""Oblique Rayleigh-Ritz special-cased to the right GPOF oblique convention.

TODO: block structure on evecs?
"""

import numpy as np

from .util import isreal
from .util import make_hankel, make_block_hankel
from .eig_util import eig
from .norm_util import norm_trick, normalize_overlaps
from .norm_util import block_norm_trick, normalize_block_overlaps

from .orr import make_block_rr_bounds_PrPl, make_block_rr_bounds_PrPl_offdiag, make_block_rr_srq

## Utility
def adjoint(x):
    return x.conj().swapaxes(-2,-1)
def diag(x):
    return np.diagonal(x, axis1=-2, axis2=-1)


## Basic
def diagonalize_rgpof(G, M, eig=eig, inv=np.linalg.inv):
    T = inv(G) @ M
    ev, evec = eig(T)
    return ev, evec

def diagonalize_block_rgpof(Gab, Mab, eig=eig, inv=np.linalg.inv):
    *batch, m,bdim, _,_ = Gab.shape
    assert Gab.shape == (*batch, m,bdim, m,bdim)
    assert Mab.shape == (*batch, m,bdim, m,bdim)
    d = m*bdim
    G = Gab.reshape(*batch, d,d)
    M = Mab.reshape(*batch, d,d)
    ev, evec = diagonalize_rgpof(G, M, eig=eig, inv=inv)
    # evec = evec.reshape(*batch, m,bdim, d) # *(ia)k -> *iak  ## TODO: BLOCK STRUCTURE
    return ev, evec


## Overlaps
def make_rgpof_overlaps(evec, C2pt, 
    inv=np.linalg.inv, abs=np.abs, real=np.real
):
    # TODO: logsumexp version? evec vs np.ascontiguousarray(evec) made a difference in a few Nk
    *batch, m, _ = evec.shape
    assert tuple(evec.shape) == (*batch, m,m)
    *batch_, Nt = C2pt.shape
    assert batch == batch_
    assert Nt >= m # really >= 2m, but only need m for this op

    # HARDCODE: RGPOF convention w/ El=G and Er=1
    evec_inv = inv(evec)
    # unnormalized overlaps
    unZl = evec_inv[..., :,0]
    unZr = (C2pt[..., :m][...,None,:] @ evec).conj().squeeze(-2)
    Nk, is_norm = norm_trick(unZr, unZl, abs=abs, real=real)
    Zr, Zl = normalize_overlaps(unZr, unZl, Nk)
    return Zr, Zl, Nk, is_norm


def make_block_rgpof_overlaps(evec, Cab, evec_inv=None, 
    inv=np.linalg.inv, abs=np.abs, real=np.real, isclose=np.isclose
):
    # HARDCODE: RGPOF convention w/ El=G and Er=1
    *batch, Nt, bdim, _ = Cab.shape
    assert Cab.shape == (*batch, Nt, bdim, bdim)

    *_, d, _ = evec.shape
    assert evec.shape == (*batch, d, d)
    m = d//bdim
    assert Nt >= m # really >= 2m, but only need m for this op
    
    if evec_inv is None: evec_inv = inv(evec)

    # Cab ~ (s,a,b)
    # evec ~ (sb),k
    # Zr*_ka = \sum_sb C_ab(s) evec_(sb),k Nk
    # Zl_ka = 1/Nk inv[evec]_k,0a
    C_ = Cab[..., :m, :,:] # trim to relevant part
    C_ = C_.swapaxes(-3,-2).reshape(*batch, bdim, m*bdim) # *sab->*asb->*a(sb)
    unZr = (C_ @ evec).conj().swapaxes(-2,-1) # *ak -> *ka
    
    unZl = evec_inv.reshape(*batch, m*bdim, m, bdim)[..., 0, :] # *k(ta) -> *k,ta -> *ka

    Nk, is_norm = block_norm_trick(unZr, unZl, abs=abs, isclose=isclose, real=real)
    Zr, Zl = normalize_block_overlaps(unZr, unZl, Nk)

    return Zr, Zl, Nk, is_norm


def make_block_rgpof_ritz_coeffs(evec, Nk, Gab, evec_inv=None, inv=np.linalg.inv):
    # HARDCODE: RGPOF convention w/ El=G and Er=1
    *batch, d, _ = evec.shape
    assert evec.shape == (*batch, d,d)
    *batch_, m, bdim, _,_ = Gab.shape
    assert batch == batch_
    assert d == m*bdim
    assert Gab.shape == (*batch, m,bdim, m,bdim)
    assert Nk.shape == (*batch, d)
    
    Gab_inv = inv(Gab.reshape(*batch, d,d))
    if evec_inv is None: evec_inv = inv(evec)
    
    Pr = evec * Nk[...,None,:]  # ~ *(ta)k
    Pr = Pr.reshape(*batch, m,bdim, d) # ~ *ta,k
    
    Pl = 1/Nk[...,:,None] * evec_inv @ Gab_inv # ~ k(ta)
    Pl = Pl.reshape(*batch, d, m,bdim) # ~ *k,ta
    
    return Pr, Pl




