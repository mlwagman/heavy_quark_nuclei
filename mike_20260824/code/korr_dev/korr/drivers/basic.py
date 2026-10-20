#!/usr/bin/env python3
"""Basic prepackaged analyses using batch indices and with filtering
applied ex-post-facto rather than on the fly.
"""

import numpy as np
from tqdm import tqdm

from ..util import make_block_hankel
from ..util import isreal
from ..util import unpack_ragged_dict
from ..outlier import symm_discard

from ..eig_util import eig
from ..rgpof import diagonalize_rgpof, diagonalize_block_rgpof
from ..rgpof import make_block_rgpof_overlaps, make_block_rgpof_ritz_coeffs
from ..orr import make_block_rr_bounds_PrPl, make_block_rr_bounds_PrPl_offdiag
from ..orr import make_block_rr_srq
from ..filter import compute_block_ZCW

def run_block_rgpof(bs_Cab, m, 
    eig=eig, inv=np.linalg.inv,
    real=np.real, abs=np.abs, isclose=np.isclose
):
    # TODO: refactor to pass Hankels around
    *batch, Nt, bdim,_ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim,bdim)
    
    Gab = make_block_hankel(bs_Cab, 0)[..., :m, :, :m, :]
    Mab = make_block_hankel(bs_Cab, 1)[..., :m, :, :m, :]
    d = m*bdim

    G,M = Gab.reshape(*batch, d,d), Mab.reshape(*batch, d,d)
    ev, evec = diagonalize_rgpof(G, M, eig=eig, inv=inv)

    Zr, Zl, Nk, is_norm = make_block_rgpof_overlaps(evec, bs_Cab, 
        inv=inv, abs=abs, real=real, isclose=isclose)
#     is_herm = isreal(Nk) & isreal(ev)
    # is_herm = isreal(Zr / Zl).all(-1) & isreal(ev) & ((Zr.conj() * Zl).real > 0).all(-1)
    ZZ = Zr.conj() * Zl # diagonal amplitude Zr*_ka Zl_ka
    is_herm = is_norm & isreal(ev) & isreal(ZZ).all(-1) & (real(ZZ) > 0).all(-1)
    is_real = isreal(ev) & isreal(ZZ).all(-1)

    # C0inv = np.linalg.inv(bs_Cab[...,0,:,:])
    # ZCW = (Zr.conj()[...,:,None] * C0inv[...,None,:,:] * Zl[...,None,:]).sum((-2,-1))
    # ZCW = np.abs(ZCW)
    ZCW = compute_block_ZCW(Zr, Zl, bs_Cab, inv=inv, abs=abs)

    Pr, Pl = make_block_rgpof_ritz_coeffs(evec, Nk, Gab, inv=inv)
    # # check id
    # id_ = Pl.reshape(*batch, d,d) @ Gab.reshape(*batch, d,d) @ Pr.reshape(*batch, d,d)
    # assert np.allclose(id_, np.eye(d)[None])

    if 2*m < Nt:
        Br, Bl = make_block_rr_bounds_PrPl(bs_Cab, ev, Pr, Pl)
    else:
        Br = np.nan * np.zeros_like(ev)
        Bl = np.nan * np.zeros_like(ev)

    return dict(
        ev=ev, #evec=evec,
        Zr=Zr, Zl=Zl,
        is_herm=is_herm, is_real=is_real,
        ZCW=ZCW,
        Br=Br, Bl=Bl,
        Pr=Pr, Pl=Pl
    )

def run_all_block_rgpof(bs_Cab,
    progress=True, 
    eig=eig, inv=np.linalg.inv,
    abs=np.abs, real=np.real, isclose=np.isclose
):
    # TODO: refactor to pass Hankels around
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2
    
    analysis = {}
    all_m = range(1, m_max+1)
    for m in tqdm(all_m) if progress else all_m:
        danalysis = run_block_rgpof(bs_Cab, m, 
            eig=eig, inv=inv, abs=abs, real=real, isclose=isclose)
        for k,v in danalysis.items():
            if k not in analysis: analysis[k] = {}
            analysis[k][m] = v
    return analysis


def run_block_rgpof_offdiag(bs_Cab, bs_Cab_rr, bs_Cab_ll, m,
    eig=eig, inv=np.linalg.inv,
    abs=np.abs, real=np.real, isclose=np.isclose
):
    # TODO: refactor to take Hankel args
    *batch, Nt, bdim,_ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim,bdim)
    assert bs_Cab.shape == bs_Cab_rr.shape
    assert bs_Cab.shape == bs_Cab_ll.shape
    d = m*bdim
    
    Gab = make_block_hankel(bs_Cab, 0)[..., :m,:, :m,:]
    Mab = make_block_hankel(bs_Cab, 1)[..., :m,:, :m,:]

    G,M = Gab.reshape(*batch, d,d), Mab.reshape(*batch, d,d)
    ev, evec = diagonalize_rgpof(G, M, eig=eig, inv=inv)

    Zr, Zl, Nk, _ = make_block_rgpof_overlaps(evec, bs_Cab, 
        inv=inv, abs=abs, real=real, isclose=isclose)
#     is_herm = isreal(Nk) & isreal(ev)
#     is_herm = isreal(Zr.conj() / Zl).all(-1) & isreal(ev) & ((Zr.conj() * Zl).real > 0).all(-1)
    is_herm = isreal(ev) # TODO: ADD isreal(ZZ)

    # C0inv = np.linalg.inv(bs_Cab[...,0,:,:])
    # ZCW = (Zr.conj()[...,:,None] * C0inv[...,None,:,:] * Zl[...,None,:]).sum((-2,-1))
    # ZCW = np.abs(ZCW)
    ZCW = compute_block_ZCW(Zr, Zl, bs_Cab)

    Pr, Pl = make_block_rgpof_ritz_coeffs(evec, Nk, Gab)
    # # check id
    # id_ = Pl.reshape(*batch, d,d) @ Gab.reshape(*batch, d,d) @ Pr.reshape(*batch, d,d)
    # assert np.allclose(id_, np.eye(d)[None])

    Br, Bl = make_block_rr_bounds_PrPl_offdiag(bs_Cab_rr, bs_Cab_ll, ev, Pr, Pl)
    
    ## norms, SRQs
    Pr = Pr.reshape(*batch, d,d)
    Gab_rr = make_block_hankel(bs_Cab_rr, 0)[..., :m,:, :m,:]
    Mab_rr = make_block_hankel(bs_Cab_rr, 1)[..., :m,:, :m,:]
#     ynorm_r = make_block_rr_ynorm(Gab_rr, Pr, Pr.swapaxes(-2,-1).conj())
    srq_r, ynorm_r = make_block_rr_srq(Gab_rr, Mab_rr, Pr)

    Pl = Pl.reshape(*batch, d,d)
    Gab_ll = make_block_hankel(bs_Cab_ll, 0)[..., :m,:, :m,:]
    Mab_ll = make_block_hankel(bs_Cab_ll, 1)[..., :m,:, :m,:]
#     ynorm_l = make_block_rr_ynorm(Gab_ll, Pl.swapaxes(-2,-1).conj(), Pl)
    srq_l, ynorm_l = make_block_rr_srq(Gab_ll, Mab_ll, Pl.swapaxes(-2,-1).conj())

    return dict(
        ev=ev, #evec=evec,
        Zr=Zr, Zl=Zl,
        is_herm=is_herm,
        ZCW=ZCW,
        Br=Br, Bl=Bl,
        ynorm_r=ynorm_r, srq_r=srq_r,
        ynorm_l=ynorm_l, srq_l=srq_l,
        Pr=Pr, Pl=Pl
    )

def run_all_block_rgpof_offdiag(bs_Cab, bs_Cab_rr, bs_Cab_ll, 
    progress=True,
    eig=eig, inv=np.linalg.inv,
    abs=np.abs, real=np.real, isclose=np.isclose
):
    # TODO: refactor to take Hankel args
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2
    
    analysis = {}
    all_m = range(1, m_max+1)
    for m in tqdm(all_m) if progress else all_m:
        danalysis = run_block_rgpof_offdiag(bs_Cab, bs_Cab_rr, bs_Cab_ll, m,
            eig=eig, inv=inv, abs=abs, real=real, isclose=isclose)
        for k,v in danalysis.items():
            if k not in analysis: analysis[k] = {}
            analysis[k][m] = v
    return analysis

from ..orr import make_offdiag_ynorms_and_srq
from ..norm_util import offdiag_norm_trick
def run_2x2_offdiag_gpof(Cab, m,
                         eig=eig, inv=np.linalg.inv,
                         real=np.real, abs=np.abs, isclose=np.isclose
                         ):
    from korr.orr import make_block_rr_bounds_PrPl_offdiag
    *batch, Nt, bdim, _ = Cab.shape
    assert Cab.shape == (*batch, Nt, bdim, bdim)
    assert bdim == 2
    
    Cab_full = Cab
    Cab = Cab[..., :2*m, :,:]
    Clr = Cab[..., 0,1]
    Cll = Cab[..., 0,0]
    Crr = Cab[..., 1,1]
    
    analysis = run_block_rgpof(Clr[...,None,None], m=m,
                               eig=eig, inv=inv,
                               real=real, abs=abs, isclose=isclose)
    
    if 2*m < Nt:
        Br, Bl = make_block_rr_bounds_PrPl_offdiag(
            Cab_rr=Cab_full[...,1,1,None,None],
            Cab_ll=Cab_full[...,0,0,None,None],
            ev=analysis['ev'], Pr=analysis['Pr'], Pl=analysis['Pl'],
            abs=abs, real=real)
    else:
        Br = np.nan * np.zeros_like(analysis['ev'])
        Bl = np.nan * np.zeros_like(analysis['ev'])
    
#     pick = analysis['is_real']
#     analysis = filter_block_analysis(analysis, pick)
    
    Pr = analysis['Pr'].squeeze(-2) # ~*tk
    Pl = analysis['Pl'].squeeze(-1) # ~*kt
#     ynorm_lr, ynorm_rr, ynorm_ll = make_offdiag_ynorms(Pr, Pl, Clr, Crr, Cll)
    ynorm_lr, ynorm_rr, ynorm_ll, srq_rr, srq_ll = make_offdiag_ynorms_and_srq(Pr, Pl, Clr, Crr, Cll)
    Nk2_r, Nk2_l, is_herm_r, is_herm_l = offdiag_norm_trick(Pr, Pl, Clr, Crr, Cll, real=real)
    analysis.update(dict(
        Br=Br, Bl=Bl,
        ynorm_lr=ynorm_lr, ynorm_rr=ynorm_rr, ynorm_ll=ynorm_ll,
        srq_rr=srq_rr, srq_ll=srq_ll,
        Nk2_r=Nk2_r, Nk2_l=Nk2_l,
        is_herm_r=is_herm_r, is_herm_l=is_herm_l
    ))
    
    return analysis

## Utilities for filtering analyses
def unpack_filter_offdiag(analysis, eps_ZCW,
    obs_1d=['srq_r', 'srq_l', 'ynorm_r', 'ynorm_l'],
    override_herm=True,
    take_inner_med=True
):
    filtered = {}

    M, all_ev = unpack_ragged_dict(analysis['ev'])
    _, all_is_herm = unpack_ragged_dict(analysis['is_herm'])
    if override_herm:
        all_is_herm = isreal(all_ev)
    _, all_ZCW = unpack_ragged_dict(analysis['ZCW'])
    _, all_Br = unpack_ragged_dict(analysis['Br'])
    _, all_Bl = unpack_ragged_dict(analysis['Bl'])
    
    # preprocessing
    for k in obs_1d:
        filtered[k] = unpack_ragged_dict(analysis[k])[1]
    # special cases
    all_ev = all_ev.real
    filtered['ev'] = all_ev
    filtered['Br'] = np.abs(all_Br)**0.5
    filtered['Bl'] = np.abs(all_Bl)**0.5
    filtered['B'] = np.abs(np.stack([all_Br, all_Bl])).min(0)**0.5

    # make picks
    pick = all_is_herm.copy()
    pick = pick & (all_ZCW > eps_ZCW)
    # figure out sort
    all_ev = np.where(pick, all_ev, -np.inf)
    sort_idx = np.argsort(-all_ev, axis=-2) 

    # mask picks
    filtered = {k : np.where(pick, o, np.nan) for k,o in filtered.items()}
    # apply sort
    filtered = {k : np.take_along_axis(o, sort_idx, axis=-2) for k,o in filtered.items()}
    
    if take_inner_med:
        # robust inner boot med
        filtered = {k : symm_discard(o.swapaxes(0,1), force_nan=True).swapaxes(1,0) 
                    for k,o in filtered.items()}
        # inner boot med
        filtered = {k : np.median(o, axis=1) for k,o in filtered.items()}
        # discard missing values
        filtered = {k : symm_discard(o) for k,o in filtered.items()}
    
    return M, filtered
def unpack_filter_diag(analysis, eps_ZCW,
    obs_1d=[],
    take_inner_med=True
):
    return unpack_filter_offdiag(analysis, eps_ZCW=eps_ZCW, 
        obs_1d=obs_1d, override_herm=False,
        take_inner_med=take_inner_med)

