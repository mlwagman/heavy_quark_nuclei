#!/usr/bin/env python3
"""Utilities to compute norms of various vectors.
"""

import numpy as np
from .util import isreal

def norm_trick(Zr, Zl, 
    abs=np.abs, real=np.real
):
    assert Zr.shape == Zl.shape
    Nk2 = Zl / Zr # = |Nk|^2

    is_norm = isreal(Nk2) & (real(Nk2) > 0) # normalizable?
    Nk = abs(Nk2)**0.5

    # overlap phase convention: real, positive Zr
    phase = Zr / abs(Zr)
    Nk = phase * Nk # gets applied to Zr conjugated
    
    return Nk, is_norm

def normalize_overlaps(Zr, Zl, Nk):
    assert Zr.shape == Zl.shape # ~ *k
    assert Zr.shape == Nk.shape # ~ *k
    Zr = Nk.conj() * Zr
    Zl = Zl / Nk
    return Zr, Zl


def block_norm_trick(Zr, Zl, 
    abs=np.abs, isclose=np.isclose, real=np.real
):
    assert Zr.shape == Zl.shape
    # assumes Zr, Zl ~ *ka
    Nka2 = Zl / Zr # ~ *ka, = |N_ka|^2

    # normalizable?
    is_norm = isclose(Nka2[...,:1], Nka2, rtol=1e-4).all(-1) # Nk same for all a?
    Nk2 = Nka2.mean(-1) # ~ *k, mean over a trivial for physical k
    is_norm &= isreal(Nk2) & (real(Nk2) > 0)
    Nk = abs(Nk2)**0.5

    # overlap phase convention: Zr_k0 real, positive
    phase = Zr[...,:,0] / abs(Zr[...,:,0]) # ~ *k
    Nk = phase * Nk # gets applied to Zr conjugated

    return Nk, is_norm

def normalize_block_overlaps(Zr, Zl, Nk):
    assert Zr.shape == Zl.shape # ~ *ka
    assert Nk.shape == Zr.shape[:-1] # ~ *k
    Nk = Nk[...,:,None] # ~ *k1
    Zr = Nk.conj() * Zr
    Zl = Zl / Nk
    return Zr, Zl


def offdiag_norm_trick(Pr, Pl, Clr, Crr, Cll, real=np.real):
    *batch, m, n_states = Pr.shape # ~ *tk
    assert Pl.shape == (*batch, n_states, m) # ~ *kt
    assert Clr.shape[:-1] == (*batch,)
    assert Crr.shape[:-1] == (*batch,)
    assert Cll.shape[:-1] == (*batch,)
    
    # test vector = |psi^L> = |psi_0>
    num_l = np.einsum('...kt,...t->...k', Pl, Cll[..., :m], optimize=True)
    den_l = np.einsum('...t,...tk->...k', Clr[..., :m], Pr, optimize=True)
#     assert nanisclose(den_l, prefiltered['Zr'].squeeze(-1)).all()

    # test vector = |psi^R> = |psi_1>
    num_r = np.einsum('...kt,...t->...k', Pl, Clr[..., :m], optimize=True)
    den_r = np.einsum('...t,...tk->...k', Crr[..., :m], Pr, optimize=True)
    # assert nanisclose(num_r, prefiltered['Zl'].squeeze(-1)).all()
#     print(nanisclose(num_r, prefiltered['Zl'].squeeze(-1)).mean()) # ~ 1

    Nk2_l = num_l / den_l
    Nk2_r = num_r / den_r
    
    is_herm_r = isreal(Nk2_r) & (real(Nk2_r) > 0)
    is_herm_l = isreal(Nk2_l) & (real(Nk2_l) > 0)
    
    return Nk2_r, Nk2_l, is_herm_r, is_herm_l
