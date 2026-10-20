"""Utilities for state labeling.
"""

import numpy as np

from .util import make_block_hankel
from .util import pad_for_batched_apply
from .util import nan_take_along_axis

def compute_label_dot(Cab, Pl_label, Pr):
    *batch, Nt, bdim,_ = Cab.shape
    assert Cab.shape == (*batch, Nt, bdim,bdim)

    *_, d_label,m_label,_ = Pl_label.shape
    assert Pl_label.shape == (*batch, d_label, m_label,bdim) # ~ *kta

    *_, m,_,d = Pr.shape
    assert Pr.shape == (*batch, m,bdim,d) # ~ *tak

    Gab = make_block_hankel(Cab, 0)
    Gab = Gab[..., :m_label,:, :m,:]
    return np.einsum('...ksa,...satb,...tbl->...kl', Pl_label, Gab, Pr, optimize=True)

def label_states_relative_threshold(dot, eps):
    dot = np.where(np.isfinite(dot), dot, 0)
    # create a label for each state
    (N_boot, N_labels, N_states) = dot.shape
    N_states_labeled = min([N_labels, N_states])
    label_idx = np.zeros((N_boot,N_states))
    for b in range(N_boot):
        # inherit ordering from sort by default
        label_idx[b,:] = np.arange(N_states)
        for k in range(0,N_states_labeled):
            # if overlap is pathologically small compared to a different label
            if np.abs(dot[b,k,k])/np.argmax(np.abs(dot[b,k,:])) < eps:
                # label as that state
                label_idx[b,k] = np.argmax(np.abs(dot[b,k,:]))
            else:
                # dislocation, label as missing state
                label_idx[b,k] = np.nan
    #label_idx = np.argmax(dot, axis=-1)
    return label_idx

def label_states_threshold(dot, eps):
    dot = np.where(np.isfinite(dot), dot, -np.inf)
    # create a label for each state
    (N_boot, N_labels, N_states) = dot.shape
    N_states_labeled = min([N_labels, N_states])
    label_idx = np.zeros((N_boot,N_states))
    for b in range(N_boot):
        # inherit ordering from sort by default
        label_idx[b,:] = np.arange(N_states)
        for k in range(0,N_states_labeled):
            # if overlap is pathologically small
            if np.abs(dot[b,k,k]) < eps:
                # if theres another state with large overlap
                if np.argmax(dot[b,k,:]) > eps:
                    # label as that state
                    label_idx[b,k] = np.argmax(dot[b,k,:])
                else:
                    # dislocation, label as missing state
                    label_idx[b,k] = np.nan
    return label_idx

def label_states_argmax(dot):
    dot = np.where(np.isfinite(dot), np.abs(dot), -np.inf)
    label_idx = np.argmax(dot, axis=-1)
    # label doesn't claim a state all overlaps are nan
    # (usually means missing label vector)
    label_idx = np.where(np.isfinite(dot).any(-1), label_idx, np.nan)
    return label_idx


def _label_analysis(analysis, label_idx, axis_for_obs):
    labeled = {}
    for o, axis in axis_for_obs.items():
        if o in analysis:
            label = pad_for_batched_apply(label_idx, analysis[o], axis=axis)
            labeled[o] = nan_take_along_axis(analysis[o], label, axis=axis)

    # carry over leftovers
    for k,v in analysis.items():
        if k not in labeled:
            labeled[k] = v

    return labeled

def label_scalar_analysis(analysis, label_idx):
    return _label_analysis(analysis, label_idx, axis_for_obs={
        'ev' : -1, 'Br' : -1, 'Bl' : -1,
        'E' : -1, 'B' : -1, 'res_lower' : -1, 'res_upper' : -1,
        'gap_lower' : -1, 'gap_upper' : -1,
        'is_herm' : -1, # ~ *k
        'Zr' : -1, 'Zl' : -1, # ~ *k
        'ZCW' : -1, # ~ *k
        'Pr' : -1, # ~ *tk
        'Pl' : -2, # ~ *kt
    })

def label_block_analysis(analysis, label_idx):
    return _label_analysis(analysis, label_idx, axis_for_obs={
        'ev' : -1, 'Br' : -1, 'Bl' : -1, # ~ *k
        'E' : -1, 'B' : -1, 'res_lower' : -1, 'res_upper' : -1,
        'gap_lower' : -1, 'gap_upper' : -1,
        'is_herm' : -1, # ~ *k
        'Zr' : -2, 'Zl' : -2, # ~ *ka
        'ZCW' : -1, # ~ *k
        'Pr' : -1, # ~ *tak
        'Pl' : -3, # ~ *kta
    })
