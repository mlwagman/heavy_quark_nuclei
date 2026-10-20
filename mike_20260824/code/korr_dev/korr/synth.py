#!/usr/bin/env python3
"""Utilities for making synthetic correlator data.
"""

import numpy as np
from tqdm import tqdm
from itertools import product

from . import mnp
from .mnp import mp_array

## High-precision mpmath versions
def mp_make_spec(Z, E, Nt):
    Z = mp_array(Z)
    E = mp_array(E)
    n_states, = Z.shape
    assert E.shape == (n_states,)
    Z, E = Z[:,None], E[:,None] # ~ n t
    lam = mnp.exp(-E) # ~ n t
    t = np.arange(Nt)[None,:] # ~ n t
    return Z * lam**t

def mp_make_2pt_offdiag(Zw, Zv, E, Nt):
    # \sum_n <chi|n> <n|chi> lam_n^t
    # = \sum_n Zw*_n Zv_n lam_n^t
    spec = mp_make_spec(Z=Zv, E=E, Nt=Nt) # spec_nt = Zv_n exp[-En t]
    Zw = mp_array(Zw)
    n_states, = Zw.shape
    assert spec.shape == (n_states, Nt)
    return (Zw[:,None] * spec).sum(0)

def mp_make_2pt(Z, E, Nt):
    spec = mp_make_spec(Z=Z, E=E, Nt=Nt) # spec_nt = Zn exp[-En t]
    Z = mp_array(Z)
    n_states, = Z.shape
    assert spec.shape == (n_states, Nt)
    return (Z[:,None] * spec).sum(0)

def mp_make_3pt_diag(Z, E, M, Nt):
    Z = mp_array(Z)
    E = mp_array(E)
    M = mp_array(M)
    n_states, = Z.shape
    assert E.shape == (n_states,)
    assert M.shape == (n_states, n_states)
    
    C3pt = np.zeros((Nt,Nt), dtype=object)
    tf = np.arange(Nt)[:,None]
    tau = np.arange(Nt)[None,:]
    dt = tf-tau
    from itertools import product
    for n,m in tqdm(list(product(range(n_states), range(n_states)))):
        C3pt += Z[n] * Z[m] * M[n,m] * mnp.exp(-E[n]*dt -E[m]*tau)
    
    return C3pt

def mp_make_3pt_rot_diag(Z, E, M, Nt):
    Z, E = mp_array(Z), mp_array(E)
    M = mp_array(M) # M_fi
    n_states, = Z.shape
    assert E.shape == (n_states,)
    assert Z.shape == (n_states,)

    # spec_nt ~ Z_n exp[-En t] ~ n t
    spec = mp_make_spec(Z=Z, E=E, Nt=Nt)

    # Reshape all to ~ nf ni dt tau
    spec_f = spec[:,None,:,None] 
    spec_i = spec[None,:,None,:]
    M = M[:,:,None,None]

    C3pt_rot = (spec_f * M * spec_i).sum((0,1))
    return C3pt_rot


def mp_make_3pt(Zi, Zf, Ei, Ef, M, Nt):
    Zi, Zf = mp_array(Zi), mp_array(Zf)
    Ei, Ef = mp_array(Ei), mp_array(Ef)
    M = mp_array(M) # M_fi
    n_states, = Zi.shape
    assert Zf.shape == (n_states,)
    assert Ei.shape == (n_states,)
    assert Ef.shape == (n_states,)
    assert M.shape == (n_states, n_states)
    
    C3pt = np.zeros((Nt,Nt), dtype=object)
    tf = np.arange(Nt)[:,None]
    tau = np.arange(Nt)[None,:]
    dt = tf-tau
    from itertools import product
    for n,m in tqdm(list(product(range(n_states), range(n_states)))):
        C3pt += Zf[n] * Zi[m] * M[n,m] * mnp.exp(-Ef[n]*dt -Ei[m]*tau)
    
    return C3pt

def mp_make_3pt_rot(Zi, Zf, Ei, Ef, M, Nt):
    Zi, Zf = mp_array(Zi), mp_array(Zf)
    Ei, Ef = mp_array(Ei), mp_array(Ef)
    M = mp_array(M) # M_fi
    n_states, = Zi.shape
    assert Zf.shape == (n_states,)
    assert Ei.shape == (n_states,)
    assert Ef.shape == (n_states,)
    assert M.shape == (n_states, n_states)

    # spec_nt ~ Z_n exp[-En t] ~ n t
    spec_i = mp_make_spec(Z=Zi, E=Ei, Nt=Nt)
    spec_f = mp_make_spec(Z=Zf, E=Ef, Nt=Nt)

    # Reshape all to ~ nf ni dt tau
    spec_f = spec_f[:,None,:,None] 
    spec_i = spec_i[None,:,None,:]
    M = M[:,:,None,None]

    C3pt_rot = (spec_f * M * spec_i).sum((0,1))
    return C3pt_rot


## Low-precision numpy versions
def make_2pt(Z, E, Nt):
    n_states, = Z.shape
    assert E.shape == (n_states,)
    t = np.arange(Nt)
    C2pt = np.zeros(Nt)
    for n in range(n_states):
        C2pt[t] += Z[n]**2 * np.exp(-E[n]*t)
    return C2pt

def make_3pt_diag(Z, E, M, Nt):
    n_states, = Z.shape
    assert E.shape == (n_states,)
    assert M.shape == (n_states, n_states)

    C3pt = np.zeros((Nt,Nt))
    tf = np.arange(Nt)[:,None]
    tau = np.arange(Nt)[None,:]
    dt = tf-tau
    for n,m in product(range(n_states), range(n_states)):
        C3pt += Z[n] * Z[m] * M[n,m] * np.exp(-E[n]*dt -E[m]*tau)
    return C3pt

def make_3pt(Zi, Zf, Ei, Ef, M, Nt):
    n_states, = Zi.shape
    assert Zf.shape == (n_states,)
    assert Ei.shape == (n_states,)
    assert Ef.shape == (n_states,)
    assert M.shape == (n_states, n_states)

    C3pt = np.zeros((Nt,Nt))
    tf = np.arange(Nt)[:,None]
    tau = np.arange(Nt)[None,:]
    dt = tf-tau
    for n,m in product(range(n_states), range(n_states)):
        C3pt += Zf[n] * Zi[m] * M[n,m] * np.exp(-Ef[n]*dt -Ei[m]*tau)
    return C3pt

def make_spec(Z, E, Nt):
    n_states, = Z.shape
    assert E.shape == (n_states,)
    spec = np.zeros((n_states, Nt))
    t = np.arange(Nt)
    for n in range(n_states):
        spec[n] = Z[n] * np.exp(-E[n]*t)
    return spec
