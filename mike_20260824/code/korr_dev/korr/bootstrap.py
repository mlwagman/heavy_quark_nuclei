#!/usr/bin/env python3
"""Utilities for bootstrapping.

TODO:
- Decompose bs_tensor, nbs_tensor functions to reduce redundancy
"""

import numpy as np

def make_bs_tensor(N_samps, N_boots, N_sub=None, weights=None):
    if weights is None: weights = np.ones(N_samps)
    if N_sub is None: N_sub = N_samps
    weights = np.float64(weights)/np.sum(weights) # for weighted bootstrap
    samp_ids = np.arange(N_samps)
    bs_samp_ids = np.random.choice(samp_ids, p=weights, size=(N_boots, N_sub), replace=True)
    bs_tensor = np.zeros((N_boots, N_samps))
    for bs_id in range(N_boots):
        samp_ids, counts = np.unique(bs_samp_ids[bs_id], return_counts=True)
        bs_tensor[bs_id, samp_ids] = counts
    return bs_tensor / N_sub # (id_bs, id_samp), for dotting into things to get sum

def make_nbs_tensor(N_samps, N_outer, N_inner, N_sub=None, weights=None):
    if weights is None: weights = np.ones(N_samps)
    if N_sub is None: N_sub = N_samps
    weights = np.float64(weights)/np.sum(weights) # for weighted bootstrap

    # outer boot
    samp_ids = np.arange(N_samps)
    bs_samp_ids = np.random.choice(samp_ids, p=weights, size=(N_outer, N_sub), replace=True)

    # inner boot: no weights
    nest_samp_ids = [np.random.choice(bs_id, size=(N_inner, N_sub), replace=True)
        for bs_id in bs_samp_ids]
    nest_samp_ids = np.stack(nest_samp_ids, axis=0)

    # make bs tensor
    assert nest_samp_ids.shape == (N_outer, N_inner, N_sub)
    nest_samp_ids = nest_samp_ids.reshape(-1, N_sub)
    bs_tensor = np.zeros((N_outer*N_inner, N_samps), dtype=int)
    for bs_id in range(nest_samp_ids.shape[0]):
        samp_ids, counts = np.unique(nest_samp_ids[bs_id], return_counts=True)
        bs_tensor[bs_id, samp_ids] = counts
    bs_tensor = bs_tensor / N_sub
    bs_tensor = bs_tensor.reshape((N_outer, N_inner, N_samps))
    
    return bs_tensor
