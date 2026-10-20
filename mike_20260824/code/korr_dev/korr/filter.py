#!/usr/bin/env python3
"""Utilities for spurious state filtering.
"""

import numpy as np

from .util import zip_items
from .util import isreal
from .util import pad_for_batched_apply, apply_batch_sort, trim_axis

def compute_ZCW(Zr, Zl, C, abs=np.abs):
    assert Zr.shape == Zl.shape
    return abs(Zr.conj() * Zl / C[...,:1])

def compute_block_ZCW(Zr, Zl, Cab, abs=np.abs, inv=np.linalg.inv):
    assert Zr.shape == Zl.shape
    C0inv = inv(Cab[...,0,:,:])
    ZCW = (Zr.conj()[...,:,None] * C0inv[...,None,:,:] * Zl[...,None,:]).sum((-2,-1))
    ZCW = abs(ZCW)
    return ZCW

# criteria whose violation signal an unphysical eigenvalue
def get_physical_mask(is_herm, ev, staggered=False):
    mask = np.all(is_herm, axis=-1)
    mask = np.logical_and(mask, np.max(ev, axis=-1) < 1.0)
    if staggered:
        mask = np.logical_and(mask, np.min(ev, axis=-1) > -1.0)
    else:
        mask = np.logical_and(mask, np.min(ev, axis=-1) > 0.0)
    return mask

def pick_phys_ZCW_cut(all_ZCW, all_is_herm, all_ev, F_ZCW=10, staggered=False):
    all_m = list(all_ZCW.keys())
    m_last = np.zeros_like(all_ZCW[all_m[0]][...,0], dtype=int)
    ZCW_last = np.zeros_like(all_ZCW[all_m[0]][...,0])
    # find last m where all hermitian and corresponding ZCW
    for m, (ZCW, is_herm, ev) in zip_items(all_ZCW, all_is_herm, all_ev):
        mask = get_physical_mask(is_herm, ev, staggered=staggered)
        m_last[mask] = m
        ZCW_last[mask] = np.min(ZCW[mask], axis=-1)
    m_last = np.median(m_last, axis=-1)
    ZCW_last = np.median(ZCW_last, axis=-1)
    print("min eps_ZCW = ", np.min(ZCW_last)/F_ZCW)
    assert(ZCW_last > 0).all()
    return ZCW_last / F_ZCW, m_last

# apply ZCW filter
def apply_ZCW_test(analysis, F_ZCW=10, filter_herm=True, eps_ZCW=None, store_P=False, staggered=False):
    # extract Ritz values, Zs, Bs, and filtering quantities
    all_ev = analysis["ev"]
    all_Zl = analysis["Zl"]
    all_Zr = analysis["Zr"]
    all_Bl = analysis["Bl"]
    all_Br = analysis["Br"]
    if store_P:
        all_Pl = analysis["Pl"]
        all_Pr = analysis["Pr"]
    all_ZCW = analysis["ZCW"]
    all_is_herm = analysis["is_herm"]
    # extract iteration and bootstrap info
    all_m = all_ev.keys()
    # holders for results
    filtered_all_ev = {m : [] for m in all_m}
    filtered_all_Z = {m : [] for m in all_m}
    filtered_all_B = {m : [] for m in all_m}
    filtered_all_P = {m : [] for m in all_m}

    # determine eps_ZCW
    if eps_ZCW is None:
        eps_ZCW, m_last = pick_phys_ZCW_cut(all_ZCW, all_is_herm, all_ev, F_ZCW=F_ZCW, staggered=staggered)
    else:
        eps_ZCW = eps_ZCW * np.ones_like(all_ZCW[list(all_m)[0]][...,0,0])
        m_last = 2*np.ones_like(eps_ZCW)

    # make mask for non-spurious states
    all_picks = {}
    for m, (ZCW, is_herm) in zip_items(all_ZCW, all_is_herm):
        # hermitian subspace and ZCW
        if filter_herm:
            if len(ZCW.shape) > 2:
                all_picks[m] = np.logical_and(is_herm, ZCW > eps_ZCW[...,None,None])
            else:
                all_picks[m] = np.logical_and(is_herm, ZCW > eps_ZCW[...,None])
        # just ZCW
        else:
            if len(ZCW.shape) > 2:
                all_picks[m] = ZCW > eps_ZCW[...,None,None]
            else:
                all_picks[m] = ZCW > eps_ZCW[...,None]
    print("ZCW shape = ", ZCW.shape)

    # apply filter
    nested=False
    for m, (bs_ev, bs_Zr, bs_Zl, bs_Br, bs_Bl, bs_pick) in zip_items(all_ev,  all_Zr, all_Zl,
                                                 all_Br, all_Bl, all_picks):
        for ev, Zr, Zl, Br, Bl, pick in zip(bs_ev, bs_Zr, bs_Zl, bs_Br, bs_Bl, bs_pick):
            if len(ev.shape) > 1:
                # assume 2d nested bootstrap shape
                nested=True
                filtered_all_ev[m].append([])
                filtered_all_Z[m].append([])
                filtered_all_B[m].append([])
                for ibs_ev, ibs_Zr, ibs_Zl, ibs_Br, ibs_Bl, ibs_pick in zip(ev, Zr, Zl, Br, Bl, pick):
                    filtered_all_ev[m][-1].append(np.real(ibs_ev[ibs_pick]))
                    filtered_all_Z[m][-1].append(np.real(ibs_Zr[ibs_pick]))
                    filtered_all_B[m][-1].append(np.real(ibs_Br[ibs_pick]))
            else:
                # assume 1d bootstrap shape
                filtered_all_ev[m].append(np.real(ev[pick]))
                filtered_all_Z[m].append(np.real(Zr[pick]))
                filtered_all_B[m].append(np.real(Br[pick]))
    if store_P:
        for m, (bs_Pr, bs_Pl, bs_pick) in zip_items(all_Pr, all_Pl, all_picks):
            for Pr, Pl, pick in zip(bs_Pr, bs_Pl, bs_pick):
                if nested:
                    # assume 2d nested bootstrap shape
                    filtered_all_P[m].append([])
                    for ibs_Pr, ibs_Pl, ibs_pick in zip(Pr, Pl, pick):
                        filtered_all_P[m][-1].append(ibs_Pl[ibs_pick])
                else:
                    # assume 1d bootstrap shape
                    filtered_all_P[m].append(Pl[pick])
        return dict(
            ev=filtered_all_ev,
            Z=filtered_all_Z,
            B=filtered_all_B,
            P=filtered_all_P,
            eps_ZCW=eps_ZCW,
            m_last=m_last
        )
    else:
        return dict(
            ev=filtered_all_ev,
            Z=filtered_all_Z,
            B=filtered_all_B,
            eps_ZCW=eps_ZCW,
            m_last=m_last
        )

## Utility
def is_subset_of(m0, m1):
    # is m0 a subset of m1?
    return np.all( (m0 & m1) | ~m0)


## Cullum-Willoughby
def make_CW_picks(ev, ev_CW, p0, p0_CW, c=1, f=10):
    ev_CW = ev_CW[p0_CW] # discard unpicked CWs

    if len(ev_CW) == 0: return p0, np.nan, np.empty((len(ev),0))
    if not p0.any(): return p0, np.nan, np.empty((len(ev),0))

    dmin = np.abs(ev[:,None] - ev_CW[None,:]).min(1) # min dist of ev to a picked CW
    assert dmin.shape == ev.shape

    # use available dmin to make cut
    pick_dmin = dmin[p0] # discard already-unpicked evs
    cut = (pick_dmin.max() - pick_dmin.min()) / (f * (len(pick_dmin) + c))

    # pick ev w/ no CW nearby w/in cut
    return (dmin > cut) & p0, cut, dmin

def make_bs_pick_CW(
    bs_ev, bs_ev_CW, bs_is_normed, pre_norm=False, post_norm=True, c=1, f=10,
    allow_cx=False, restrict_CW_re=True
):
    bs_ev_is_re = np.isclose(bs_ev.imag, 0)
    bs_ev_CW_is_re = np.isclose(bs_ev_CW.imag, 0)
    assert is_subset_of(bs_is_normed, bs_ev_is_re)

    if not allow_cx:
        bs_pick0 = (bs_is_normed if pre_norm else bs_ev_is_re)
    else:
        bs_pick0 = (bs_is_normed if pre_norm else np.ones_like(bs_ev_is_re, dtype=bool))
    if restrict_CW_re:
        bs_CW_pick0 = bs_ev_CW_is_re
    else:
        bs_CW_pick0 = np.ones_like(bs_ev_CW, dtype=bool)

    bs_picks = []
    bs_cuts = []
    bs_dmin = []
    for ev, ev_CW, p0, p0_CW in zip(bs_ev, bs_ev_CW, bs_pick0, bs_CW_pick0):
        pick, cut, dmin = make_CW_picks(ev, ev_CW, p0, p0_CW, f=f, c=c)
        bs_picks.append(pick)
        bs_cuts.append(cut)
        bs_dmin.append(dmin)
    bs_picks = np.stack(bs_picks)
    bs_cuts = np.array(bs_cuts)
    bs_dmin = np.stack(dmin)
    if post_norm:
        bs_picks &= bs_is_normed
    return bs_picks, bs_cuts, bs_dmin

def make_all_bs_pick_CW(all_bs_evals, all_bs_evals_CW, all_bs_is_normed,
    pre_norm=False, post_norm=False, c=1, f=10,
    allow_cx=False, restrict_CW_re=True
):
    all_bs_picks = {}
    all_bs_cuts = {}
    all_bs_dmin = {}
    for k, (bs_ev, bs_ev_CW, bs_is_normed) in zip_items(
        all_bs_evals, all_bs_evals_CW, all_bs_is_normed,
        strict=False
    ):
        all_bs_picks[k], all_bs_cuts[k], all_bs_dmin[k] = make_bs_pick_CW(
            bs_ev, bs_ev_CW, bs_is_normed,
            pre_norm=pre_norm, post_norm=post_norm, f=f, c=c,
            allow_cx=allow_cx, restrict_CW_re=restrict_CW_re)

    return all_bs_picks, all_bs_cuts, all_bs_dmin


## MWCW cut picker

def make_all_bs_CW_picks_MW(
    all_bs_evals, all_bs_evals_CW,
    Delta=4, K_CW=3,
    return_extras=False
):
    # NOTE: assumes bs_evals, bs_evals_CW have been pre-trimmed
    N_it = len(all_bs_evals)
    N_boot = all_bs_evals[1].shape[0]

    all_d_CW = {}
    all_N_plus = []
    for m, (bs_ev, bs_ev_CW) in zip_items(all_bs_evals, all_bs_evals_CW):
        all_N_plus.append( np.isfinite(bs_ev).sum() )

        if m == 1: continue
        assert bs_ev_CW.shape[1] > 0, bs_ev_CW.shape

        d = np.abs(bs_ev[:,:,None] - bs_ev_CW[:,None,:])
        if (d == 0).any():
            print(f"WARNING: discarding {(d==0).sum()} exact zeros at m={m}")
            d[d == 0] = np.inf # exclude from min
        d[~np.isfinite(d)] = np.inf # exclude from min
        d = d.min(2)
        check = np.where(~((~np.isfinite(d) == ~np.isfinite(bs_ev))))
#         assert (~np.isfinite(d) == ~np.isfinite(bs_ev)).all() # all inf d_CW are missing eigs
        assert (~np.isfinite(d) == (
            ~np.isfinite(bs_ev)  # all inf d_CW are missing eigs
            | ((~np.isfinite(bs_ev_CW)).all(1)[:,None])) # or all CW eigs are missing
        ).all()
        d[~np.isfinite(d)] = np.nan
        all_d_CW[m] = d

    N_plus = np.sum(all_N_plus)

    N_lam = int(np.rint(N_plus / N_boot / N_it))
    N_bins = int(np.rint(Delta * N_lam))
    delta_CW = N_boot * (N_it - N_lam) * K_CW/Delta

    d_CW = np.concatenate([d[np.isfinite(d)] for d in all_d_CW.values()])
    log_d_CW = np.log(d_CW)

    hist, bins = np.histogram(log_d_CW, bins=N_bins)
    assert (hist > delta_CW).any()

    cut_idx = np.argmax(hist > delta_CW)
    cut_CW = np.exp(bins[cut_idx-1])

    picks = {m : (d > cut_CW) for m,d in all_d_CW.items()}
    picks[1] = np.ones_like(all_bs_evals[1], dtype=bool)

    if not return_extras:
        return picks, cut_CW, all_d_CW
    else:
        return picks, cut_CW, all_d_CW, bins, hist, delta_CW


## ZCW

def make_ZCW_picks(ZCW, p0, c=1, f=10):
    dmin = np.abs(ZCW) # min dist of ev to a picked CW

    # use available dmin to make cut
    pick_dmin = dmin[p0] # discard already-unpicked evs
    cut = (pick_dmin.max() - pick_dmin.min()) / (f * (len(pick_dmin) + c))

    # pick ev w/ no CW nearby w/in cut
    return (dmin > cut) & p0, cut

def make_bs_pick_ZCW(
    bs_ev, bs_ZCW, bs_is_herm, pre_herm=False, post_herm=True, c=1, f=10,
    allow_cx=False
):
    bs_ev_is_re = isreal(bs_ev)
    assert is_subset_of(bs_is_herm, bs_ev_is_re)

    if pre_herm: bs_pick0 = bs_is_herm
    elif not allow_cx: bs_pick0 = bs_ev_is_re
    else: bs_pick0 = np.ones_like(bs_ev_is_re, dtype=bool)

    bs_picks = []
    bs_cuts = []
    bs_dmin = []
    for ZCW, p0 in zip(bs_ZCW, bs_pick0):
        pick, cut = make_ZCW_picks(ZCW, p0, f=f, c=c)
        bs_picks.append(pick)
        bs_cuts.append(cut)
    bs_picks = np.stack(bs_picks)
    bs_cuts = np.array(bs_cuts)
    if post_herm:
        bs_picks &= bs_is_herm
    return bs_picks, bs_cuts

def make_all_bs_pick_ZCW(all_bs_evals, all_bs_ZCW, all_bs_is_herm,
    pre_herm=False, post_herm=False, c=1, f=10,
    allow_cx=False
):
    all_bs_picks = {}
    all_bs_cuts = {}
    for k, (bs_ev, bs_ZCW, bs_is_herm) in zip_items(
        all_bs_evals, all_bs_ZCW, all_bs_is_herm,
        strict=False
    ):
        all_bs_picks[k], all_bs_cuts[k] = make_bs_pick_ZCW(
            bs_ev, bs_ZCW, bs_is_herm,
            pre_herm=pre_herm, post_herm=post_herm, f=f, c=c,
            allow_cx=allow_cx)

    return all_bs_picks, all_bs_cuts

def make_all_bs_ZCW_picks_MW(
    all_bs_evals, all_bs_ZCW,
    Delta=4, K_CW=3,
    return_extras=False
):
    # NOTE: assumes bs_evals, bs_evals_CW have been pre-trimmed
    N_it = len(all_bs_evals)
    N_boot = all_bs_evals[1].shape[0]

    all_N_plus = [np.isfinite(bs_ev).sum() for bs_ev in all_bs_evals.values()]
    N_plus = np.sum(all_N_plus)

    N_lam = int(np.rint(N_plus / N_boot / N_it))
    N_bins = int(np.rint(Delta * N_lam))
    delta_CW = N_boot * (N_it - N_lam) * K_CW/Delta

    d_CW = np.concatenate([d[np.isfinite(d)] for d in all_bs_ZCW.values()])
    log_d_CW = np.log(d_CW)
    hist, bins = np.histogram(log_d_CW, bins=N_bins)
    assert (hist > delta_CW).any()

    cut_idx = np.argmax(hist > delta_CW)
    cut_CW = np.exp(bins[cut_idx-1])

    picks = {m : (d > cut_CW) for m,d in all_bs_ZCW.items()}
    picks[1] = np.ones_like(all_bs_evals[1], dtype=bool)

    if not return_extras:
        return picks, cut_CW
    else:
        return picks, cut_CW, delta_CW, bins, hist


## ZCW cut picker
def pick_herm_ZCW_cut(all_ZCW, all_is_herm, F_ZCW=10):
    # find last m where all hermitian
    m_last = 0
    for m, is_herm in {m : all_is_herm[m] for m in sorted(all_is_herm)}.items():
        if not is_herm.all(): break
        m_last = m
    is_herm, ZCW = all_is_herm[m_last], all_ZCW[m_last]
    assert is_herm.shape == ZCW.shape # don't prefilter herms
    assert is_herm.all()
#     print(m_last, ZCW, is_herm)
    return np.abs(ZCW).min() / F_ZCW, m_last


## Batched filtering utils
def filter_sort_trim(x, pick, sort_idx, axis):
    pick = pad_for_batched_apply(pick, x, axis=axis)
    x = np.where(pick, x, np.nan)
    sort_idx = pad_for_batched_apply(sort_idx, x, axis=axis)
    x = apply_batch_sort(x, sort_idx, axis=axis)
    n_max = pick.sum(axis).max()
    x = trim_axis(x, n_max, axis)
    return x

def get_filtered_sort_idx(ev, pick):
    ev = np.where(pick, ev, -np.inf)
    return np.argsort(-ev, axis=-1)


def _filter_analysis(analysis, pick, axis_for_obs):
    # TODO: ABSOLUTELY NEEDS BETTER ERROR OUTPUT
    sort_idx = get_filtered_sort_idx(analysis['ev'], pick)

    filtered = {}
    for o, axis in axis_for_obs.items():
        if o in analysis:
            filtered[o] = filter_sort_trim(analysis[o],
                            pick=pick, sort_idx=sort_idx, axis=axis)

    # carry over leftovers
    for k,v in analysis.items():
        if k not in filtered:
            filtered[k] = v

    return filtered

def filter_scalar_analysis(analysis, pick):
    return _filter_analysis(analysis, pick, axis_for_obs={
        'ev' : -1, 'Br' : -1, 'Bl' : -1,
        'srq_rr' : -1, 'srq_ll' : -1,
        'E' : -1, 'B' : -1, 'res_lower' : -1, 'res_upper' : -1,
        'gap_lower' : -1, 'gap_upper' : -1,
        'is_herm' : -1, # ~ *k
        'Zr' : -1, 'Zl' : -1, # ~ *k
        'ZCW' : -1, # ~ *k
        'Pr' : -1, # ~ *tk
        'Pl' : -2, # ~ *kt
    })

def filter_offdiag_analysis(analysis, pick):
    return _filter_analysis(analysis, pick, axis_for_obs={
        'ev' : -1, 'Br' : -1, 'Bl' : -1, # ~ *k
        'E' : -1, 'B' : -1, 'res_lower' : -1, 'res_upper' : -1,
        'is_herm' : -1, # ~ *k
        'Zr' : -2, 'Zl' : -2, # ~ *ka
        'ZCW' : -1, # ~ *k
        'Pr' : -1, # ~ *tak
        'Pl' : -3, # ~ *kta
        'Nk2_r' : -1, 'Nk2_l' : -1,
        'srq_rr' : -1, 'srq_ll' : -1,
        'ynorm_rr' : -1, 'ynorm_ll' : -1, 'ynorm_lr' : -1,
    })

def filter_block_analysis(analysis, pick):
    return _filter_analysis(analysis, pick, axis_for_obs={
        'ev' : -1, 'Br' : -1, 'Bl' : -1, # ~ *k
        'srq_rr' : -1, 'srq_ll' : -1,
        'E' : -1, 'B' : -1, 'res_lower' : -1, 'res_upper' : -1,
        'gap_lower' : -1, 'gap_upper' : -1,
        'is_herm' : -1, # ~ *k
        'Zr' : -2, 'Zl' : -2, # ~ *ka
        'ZCW' : -1, # ~ *k
        'Pr' : -1, # ~ *tak
        'Pl' : -3, # ~ *kta
    })
