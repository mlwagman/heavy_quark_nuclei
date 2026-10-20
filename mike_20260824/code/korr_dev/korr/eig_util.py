#!/usr/bin/env python3
"""Utilities for eigendecompositions.
"""

import numpy as np
import copy

from . import mnp
from .mnp import downcast
from .util import allreal

def sort_eigs(ev, evec, prec=9):
    *batch, d = ev.shape
    assert tuple(evec.shape) == (*batch, d,d)
    sort_idx = np.argsort(-np.round(downcast(ev), prec), axis=-1)
    ev = np.take_along_axis(ev, sort_idx, axis=-1)
    evec = np.take_along_axis(evec, sort_idx[...,None,:], axis=-1)
    return ev, evec

def fix_evec_sign(evec, abs=np.abs):
    # evec_0k is real
    # if evec_0k=0, then evec_ik is real for first nonzero i
    idx = np.argmax(evec != 0, axis=-2)
    x = np.take_along_axis(evec, idx[...,None,:], axis=-2).squeeze(-2)
    phase = x/abs(x)
    evec = evec / phase[...,None,:]
    return evec

def eig(A):
    ev, evec = np.linalg.eig(A)
    ev, evec = sort_eigs(ev, evec)
    evec = fix_evec_sign(evec)
    return ev, evec

def mp_eig(A):
    ev, evec = mnp.eig(A)
    ev, evec = sort_eigs(ev, evec)
    evec = fix_evec_sign(evec, abs=mnp.abs)
    return ev, evec

def mp_inv_block_evec(O):
    *batch, m,bdim,n = O.shape
    assert m*bdim == n, (m,bdim,n)
    return mnp.inv(O.reshape(*batch, n,n)).reshape(*batch, n, m,bdim)
def inv_block_evec(O):
    *batch, m,bdim,n = O.shape
    assert m*bdim == n, (m,bdim,n)
    return np.linalg.inv(O.reshape(*batch, n,n)).reshape(*batch, n, m,bdim)

## GEVP solvers
def scipy_gevp(C, C0):
    import scipy.linalg
    ev, evec = scipy.linalg.eig(C, C0)
    sort_idx = np.argsort(-ev)
    ev = ev[sort_idx]
    evec = evec[:, sort_idx]
    evec = fix_evec_sign(evec)
    return ev, evec

def mp_solve_gevp(C, C0, fix_norm=True):
    # TODO: FINISH FIXING L/R GEVP CONVENTIONS
    C0_inv = mnp.inv(C0)
    gev, gevec_r = mnp.eig(C0_inv @ C)
    gev_, gevec_l = mnp.eig(C0_inv.conj().T @ C.conj().T)
    assert allreal(downcast(gev_)) # don't bother re-sorting
    gevec_l = gevec_l.conj().T
    assert mnp.allclose(gev, gev_)
    if fix_norm:
        norm = 1 / np.diag(gevec_l @ C0 @ gevec_r + 0j)**0.5
        gevec_r = gevec_r * norm[None,:]
        gevec_l = gevec_l * norm[:,None]
    return gev, gevec_r, gevec_l

