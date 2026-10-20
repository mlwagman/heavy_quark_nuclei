#!/usr/bin/env python3
"""General utility functions.
"""

import numpy as np

from .mnp import downcast

## Utilities for working with ragged data
def stack_rpadded(X, pad=np.nan, axis=0):
    shapes = np.stack([x.shape for x in X], axis=0)
    assert len(shapes.shape) == 2
    if shapes.shape[1] == 0: return np.array(X) # edge case: X is a list of scalars
    n_max = np.max(shapes, axis=0)
    all_dL = n_max[None,:] - shapes
    assert np.all(all_dL >= 0)
    res = [np.pad(x, pad_width=[(0,d) for d in dL], constant_values=pad) for x, dL in zip(X,all_dL)]
    return np.stack(res, axis=axis)

def rpad_to_shape(x, shape, pad=np.nan, axis=None):
    assert len(x.shape) == len(shape), (x.shape, shape)
    pad_widths = []
    for aa, (l0,l1) in enumerate(zip(x.shape, shape)):
        if axis is not None and aa != axis:
            pad_widths.append((0,0))
            continue
        assert l0 <= l1, (l0,l1, x.shape, shape)
        pad_widths.append((0, l1-l0))
    return np.pad(x, pad_width=pad_widths, constant_values=pad)

def trim_axis(x, n_max, axis):
    axis = axis % len(x.shape) # for - axis
    return x[(slice(None),)*axis + (slice(0,n_max),)] # ~ [:,...,:,:n_max,:,...,:]

def thresh_mask(mask, thresh=0.95, axis=0):
    m = mask.mean(axis) >= thresh
    m = np.expand_dims(m, axis)
    return mask & m
def thresh_missing(x, thresh=0.95, axis=0):
    m = np.isfinite(x)
    m = thresh_mask(m, thresh=thresh, axis=axis)
    return np.where(m, x, np.nan)

def nan_take_along_axis(arr, indices, axis=-1):
    # Version which can treat missingness (NaNs) in indices
    # TESTME CAREFULLY
    keep = np.isfinite(indices) # find nans in idx
    indices = np.where(keep, indices, -1) # fill in -1 instead of nan
    indices = indices.astype(int) # cast to int for indexing
    x = np.take_along_axis(arr, indices, axis=axis) # take
    x = np.where(keep, x, np.nan) # mask out bad indices
    return x

## Utilities for batched sorting
def pad_for_batched_apply(y, x, axis=-1):
    ## Pad y to broadcast last axis w/ `axis` of x
    assert len(y.shape) <= len(x.shape)
    axis = axis % len(x.shape) # for - axis
    # pad until last dim of y if axis
    assert axis >= len(y.shape)-1
    while len(y.shape)-1 < axis:
        y = y[...,None,:] # add trivial dims before last one
    # pad end of y until same dims as x
    while len(y.shape) < len(x.shape):
        y = y[...,None]
    # checks
    assert len(y.shape) == len(x.shape)
    assert all([((d0==d1) or (d1 == 1)) or (dd == axis) for dd,(d0,d1) in enumerate(zip(x.shape, y.shape))])
    return y

def apply_batch_sort(x, sort_idx, axis=-1):
    sort_idx = pad_for_batched_apply(sort_idx, x)
    return np.take_along_axis(x, sort_idx, axis=axis)


## Checks
def isreal(x, rtol=1e-5, atol=1e-8):
    x = downcast(x)
    return np.isclose(x.imag / np.abs(x), 0, rtol=rtol, atol=atol)
def allreal(x, rtol=1e-5, atol=1e-8):
    return isreal(x, rtol=rtol, atol=atol).all()


## Dictionary manipulation
def zip_items(*all_d, strict=True):
    assert all([isinstance(d, dict) for d in all_d])
    if strict:
        keys = sorted(list(all_d[0].keys()))
        assert all([set(d.keys()) == set(keys) for d in all_d])
    else:
        # take sorted union
        keys = set(all_d[0].keys())
        for d in all_d:
            keys = keys & set(d.keys())
        keys = sorted(keys)
    for k in keys:
        vals = tuple(d[k] for d in all_d)
        yield k, vals

def unpack_ragged_dict(data):
    X = np.array(list(data.keys()))
    Y = stack_rpadded(list(data.values()), axis=-1)
    return X,Y

def stack_dicts(all_d, axis=0, stack=stack_rpadded):
    # [{k : x0}, {k : x1}] -> {k : stack(x0,x1,...)}
    # TODO: TESTME CAREFULLY
    
    assert all([isinstance(d, dict) for d in all_d])
    # all dicts must have same keys TODO: allow missingness?
    keys = list(all_d[0].keys())
    assert all([set(d.keys()) == set(keys) for d in all_d])
    
    stacked = {k : [] for k in keys}
    for d in all_d:
        for k in keys:
            stacked[k].append(d[k])
    stacked = {k : stack(v, axis=axis) for k,v in stacked.items()}
    return stacked


## Correlator manipulation
def make_hankel(C2pt, n=0):
    *batch, Nt = C2pt.shape
    m_max = Nt//2
    H = np.zeros((*batch, m_max,m_max), dtype=C2pt.dtype)
    for t in range(m_max):
        for s in range(m_max):
            if t+s+n >= Nt: continue
            H[..., t,s] = C2pt[..., t+s+n]
    return H

def make_block_hankel(Cab, n=0):
    *batch, Nt, bdim, _ = Cab.shape
    assert Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2
    Hab = np.zeros((*batch, m_max,bdim, m_max,bdim), dtype=Cab.dtype)
    for t in range(m_max):
        for s in range(m_max):
            if t+s+n >= Nt: continue
            Hab[...,s,:,t,:] = Cab[...,t+s+n,:,:]
    return Hab


def rotate_C3pt(C3pt):
    *_,Nt,_ = C3pt.shape
    assert C3pt.shape[-2:] == (Nt,Nt)
    C3pt_ = np.zeros_like(C3pt)
    for s in range(Nt):
        for t in range(Nt):
            if s+t >= Nt: continue
            C3pt_[...,s,t] = C3pt[...,s+t,t]
    return C3pt_
def unrotate_C3pt(C3pt_rot):
    *_,Nt,_ = C3pt_rot.shape
    assert C3pt_rot.shape[-2:] == (Nt,Nt)
    C3pt = np.zeros_like(C3pt_rot)
    for s in range(Nt):
        for t in range(Nt):
            if s+t >= Nt: continue
            C3pt[...,s+t,t] = C3pt_rot[...,s,t]
    return C3pt


# rotation isn't physical though
def rotate_C3pt_lossless(C3pt):
    Nt,_ = C3pt.shape
    assert C3pt.shape == (Nt,Nt)
    C3pt_rot = np.zeros_like(C3pt)
    for s in range(Nt):
        for t in range(Nt):
            C3pt_rot[s,t] = C3pt[(s+t)%Nt,t]
    return C3pt_rot
def unrotate_C3pt_lossless(C3pt_rot):
    Nt,_ = C3pt_rot.shape
    assert C3pt_rot.shape == (Nt,Nt)
    C3pt = np.zeros_like(C3pt_rot)
    for s in range(Nt):
        for t in range(Nt):
            C3pt[(s+t)%Nt,t] = C3pt_rot[s,t]
    return C3pt


def rotate_block_C3pt(C3pt):
    Nt,*_ = C3pt.shape
    assert C3pt.shape[:2] == (Nt,Nt)
    C3pt_ = np.zeros_like(C3pt)
    for s in range(Nt):
        for t in range(Nt):
            if s+t >= Nt: continue
            C3pt_[s,t,...] = C3pt[s+t,t,...]
    return C3pt_

def unrotate_block_C3pt(C3pt_rot):
    Nt,*_ = C3pt_rot.shape
    assert C3pt_rot.shape[:2] == (Nt,Nt)
    C3pt = np.zeros_like(C3pt_rot)
    for s in range(Nt):
        for t in range(Nt):
            if s+t >= Nt: continue
            C3pt[s+t,t,...] = C3pt_rot[s,t,...]
    return C3pt

def effective_mass(C2pt):
    Nt, = C2pt.shape
    eff_mass = np.zeros((Nt-1,))
    for t in range(Nt-1):
        eff_mass[t] = np.log( np.abs( C2pt[t] / C2pt[t+1] ) )
    return eff_mass

def bs_effective_mass(C2pt):
    Nboot,Nt = C2pt.shape
    eff_mass = np.zeros((Nboot,Nt-1))
    for t in range(Nt-1):
        eff_mass[:,t] = np.log( np.abs( C2pt[:,t] / C2pt[:,t+1] ) )
    return eff_mass

def bin_data(data, Nbin):
    Nmeas = data.shape[0]
    bin_data = np.zeros( (Nmeas//Nbin,)+data.shape[1:], dtype=data.dtype )
    for n in range(Nmeas//Nbin):
        for b in range(Nbin):
            bin_data[n] += data[n*Nbin+b]/Nbin
    return bin_data
