#!/usr/bin/env python3
"""Utilities to work with numpy arrays full of mpmath complex numbers.

NOTE: No package imports! Should be standalone submodule.

TODO:
- Add support for batch dimensions.
- Hand-code isclose and allclose
"""

import numpy as np
from mpmath import mp

# from .eig import fix_evec_sign


## Batching
# decorator to treat all but last n_dims as batch dimensions
# handles multiple returns, but only one input allowed
def batched_function(n_dims):
    def decorator(f):
        def batched_f(x, *args, **kwargs):
            batch = x.shape[:-n_dims]
            x = x.reshape(-1, *x.shape[-n_dims:]) # flatten batch dim
            fx = [f(b, *args, **kwargs) for b in x] # apply function to each batch element
            if isinstance(fx[0], tuple):
                # multiple returns
                assert all([isinstance(y, tuple) for y in fx]) # everything must be tuple
                lens = np.array(list(map(len,fx)))
                assert (lens == lens[0]).all(), lens # all same length
                stacked = []
                for y in zip(*fx): # iterate over each return
                    y = np.stack(y, axis=0)
                    y = y.reshape(*batch, *y.shape[1:]) # unflatten batch dims
                    stacked.append(y)
                return tuple(stacked)
            else:
                # scalar return
                fx = np.stack(fx, axis=0)
                fx = fx.reshape(*batch, *fx.shape[1:]) # unflatten batch dims
                return fx
        return batched_f
    return decorator

def mp_array(A):
    return np.array(list(map(mp.mpc, A.ravel())), dtype=object).reshape(A.shape)

def downcast(A):
    return A.astype(complex)

conj = np.vectorize(lambda x: x.conjugate())
log = np.vectorize(mp.log)
exp = np.vectorize(mp.exp)
abs = np.vectorize(mp.fabs)
imag = np.vectorize(mp.im)
real = np.vectorize(mp.re)

# same default tol as np.isclose
def isclose(x, y, rtol=1e-5, atol=1e-8):
    return np.isclose(np.complex128(x), np.complex128(y), rtol=rtol, atol=atol)
def allclose(x, y, rtol=1e-5, atol=1e-8):
    return isclose(x, y, rtol=rtol, atol=atol).all()

@batched_function(2)
def powm(A, t):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    A = mp.matrix(A.tolist())
    Ainv = A**t
    return np.array(Ainv.tolist())

def inv(A):
    return powm(A, -1)

@batched_function(2)
def eig(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    A = mp.matrix(A.tolist())
    evals, evecs = mp.eig(A)
    evals = np.array(evals)
    evecs = np.array(evecs.tolist())
    
    # UNIT-NORMALIZE
    # evecs /= (evecs * evecs.conj()).sum(0)[None,:]**0.5
    evecs /= (abs(evecs)**2).sum(0)[None,:]**0.5
    
    # # SORT
    # sort_idx = np.argsort(-np.complex_(evals))
    # evals = evals[sort_idx]
    # evecs = evecs[:, sort_idx]
    
    # # Fix evec
    # evecs = fix_evec_sign(evecs, abs=abs) # call w/ mp abs
    
    return evals, evecs

@batched_function(2)
def expm(A, n):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    A = mp.matrix(A.tolist())
    An = A**n
    return np.array(An.tolist())

