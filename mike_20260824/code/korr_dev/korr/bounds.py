#!/usr/bin/env python3
"""Two-sided windows related to residual bounds
"""

import numpy as np

# compute residual bounds and add to dictionary
def get_res_bounds(analysis, log=np.log, abs=np.abs):
    ev = analysis['ev']
    B = analysis['B']
    res_lower = -log(ev + abs(B)**0.5)
    res_upper = -log(ev - abs(B)**0.5)

    analysis['res_lower'] = res_lower
    analysis['res_upper'] = res_upper

# compute gap bounds and add to dictionary
def get_gap_bounds(analysis, E_gap_bound=None, log=np.log, abs=np.abs,
                   never_abs=False):
    ev = analysis['ev']
    B = analysis['B']
    gap = np.zeros(ev.shape)
    if E_gap_bound is not None:
        gap = np.transpose(np.array([np.abs(ev[...,k] - np.exp(-E_gap_bound))
               for k in range(ev.shape[-1])]))
    elif ev.shape[-1] > 1:
        gap = np.zeros(ev.shape)
        for k in range(ev.shape[-1]):
            diffs = np.abs(ev - ev[...,k][...,None])
            gap[...,k] = np.sort(diffs, axis=-1)[...,1]
    if never_abs:
        gap_lower = -log(ev + B/gap)
        gap_upper = -log(ev - B/gap)
    else:
        gap_lower = -log(ev + abs(B)/gap)
        gap_upper = -log(ev - abs(B)/gap)

    # set nan gap bound to residual bound
    gap_lower = np.where(np.isnan(gap_lower), analysis['res_lower'], gap_lower)
    #gap_lower = np.where(analysis['res_lower'] > gap_lower, analysis['res_lower'], gap_lower)
    gap_upper = np.where(np.isnan(gap_upper), analysis['res_upper'], gap_upper)
    #gap_upper = np.where(analysis['res_upper'] < gap_upper, analysis['res_upper'], gap_upper)

    analysis['gap_lower'] = gap_lower
    analysis['gap_upper'] = gap_upper

# compute gap bounds and add to dictionary
def get_offdiag_gap_bounds(analysis, E_gap_bound=None, log=np.log, abs=np.abs,
                   never_abs=False):
    ev = analysis['ev']
    Bl = analysis['Bl']
    Br = analysis['Br']
    gap = np.zeros(ev.shape)
    if E_gap_bound is not None:
        gap = np.transpose(np.array([np.abs(ev[...,k] - np.exp(-E_gap_bound))
               for k in range(ev.shape[-1])]))
    elif ev.shape[-1] > 1:
        gap = np.zeros(ev.shape)
        for k in range(ev.shape[-1]):
            diffs = np.abs(ev - ev[...,k][...,None])
            gap[...,k] = np.sort(diffs, axis=-1)[...,1]
    if never_abs:
        gap_ll_lower = -log(ev + Bl/gap)
        gap_ll_upper = -log(ev - Bl/gap)
        gap_rr_lower = -log(ev + Br/gap)
        gap_rr_upper = -log(ev - Br/gap)
    else:
        gap_ll_lower = -log(ev + abs(Bl)/gap)
        gap_ll_upper = -log(ev - abs(Bl)/gap)
        gap_rr_lower = -log(ev + abs(Br)/gap)
        gap_rr_upper = -log(ev - abs(Br)/gap)

    # set nan gap bound to residual bound
    gap_ll_lower = np.where(np.isnan(gap_ll_lower), analysis['res_lower'], gap_ll_lower)
    gap_ll_upper = np.where(np.isnan(gap_ll_upper), analysis['res_upper'], gap_ll_upper)
    gap_rr_lower = np.where(np.isnan(gap_rr_lower), analysis['res_lower'], gap_rr_lower)
    gap_rr_upper = np.where(np.isnan(gap_rr_upper), analysis['res_upper'], gap_rr_upper)

    analysis['gap_ll_lower'] = gap_ll_lower
    analysis['gap_ll_upper'] = gap_ll_upper
    analysis['gap_rr_lower'] = gap_rr_lower
    analysis['gap_rr_upper'] = gap_rr_upper
