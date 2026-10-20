#!/usr/bin/env python3
"""Utilities for clustering Ritz vectors.
"""

import numpy as np

from .util import make_block_hankel, stack_rpadded
from .label import compute_label_dot

_block_axis_for_obs = {
    'ev' : -1, 'Br' : -1, 'Bl' : -1, # ~ *k
    'E' : -1, 'res_lower' : -1, 'res_upper' : -1,
    'gap_lower' : -1, 'gap_upper' : -1,
    'is_herm' : -1, # ~ *k
    'Zr' : -2, 'Zl' : -2, # ~ *ka
    'ZCW' : -1, # ~ *k
    'Pr' : -1, # ~ *tak
    'Pl' : -3, # ~ *kta
}

def find_converging_seqs(all_Pl, all_Pr, Cab,
    m_min=1, m_max=None,
    abs_dot=True,
    thresh_converged=0.99
):
    # assumes no batch indices
    assert len(Cab.shape) == 3 # ~ tab
    assert all([len(Pl.shape) == 3 for Pl in all_Pl.values()]) # ~ kta
    assert all([len(Pr.shape) == 3 for Pr in all_Pr.values()]) # ~ tak
    
    assert set(all_Pl.keys()) == set(all_Pr.keys()) # same m
    all_m = sorted(list(all_Pl.keys()))
    if m_max is None:
        m_max = max(all_m)
    if m_max < 0:
        m_max = max(all_m) + m_max
    assert min(all_m) <= m_min
    assert max(all_m) >= m_max
    all_m = range(m_min, m_max+1)
    
    ## Find converged sequences
    all_seqs = [] # ~ ([m],[k])
    for m in all_m[:-1]:
        # dot = <yL^(m)_k | yR^(m+1)_l>
        Pl0 = all_Pl[m]
        Pr1 = all_Pr[m+1]
        dot = compute_label_dot(Cab, Pl0, Pr1) # ~ kl
        if abs_dot: dot = np.abs(dot)
        
        # Forward links: which state in next iter is most similar?
        links = []
        for k0 in range(dot.shape[0]):
            if not np.isfinite(dot[k0]).any(): continue
            k1 = np.nanargmax(dot[k0])
            if dot[k0,k1] > thresh_converged:
                links.append([k0,k1])
        links = sorted(list(set(map(tuple, links)))) # unique links
        
        # Use links to build trajectories
        for k0,k1 in links:
            for all_m, seq in all_seqs:
                # continue trajectory if found
                if all_m[-1] == m and seq[-1] == k0:
                    all_m.append(m+1)
                    seq.append(k1)
                    break
            else:
                # seed a new trajectory
                all_seqs.append(([m,m+1], [k0,k1]))
                
    return all_seqs

def pick_from_seqs(all_seqs, Cab, all_Pl, all_Pr,
    min_len=2,
    abs_dot=True,
    thresh_duplicate=0.5,
    rule='latest',
    all_B=None
):
    # assumes no batch indices
    assert len(Cab.shape) == 3 # ~ tab
    assert all([len(Pl.shape) == 3 for Pl in all_Pl.values()]) # ~ kta
    assert all([len(Pr.shape) == 3 for Pr in all_Pr.values()]) # ~ tak

    assert set(all_Pl.keys()) == set(all_Pr.keys()) # same m
    
    assert rule in ('latest', 'longest', 'leastB', 'earliest')
    
    # Clip length
    all_seqs = [(m,seq) for (m,seq) in all_seqs if len(seq) >= min_len]
    
    ## Consider final states in each seq
    # Compute <yL^(m0)_k0 | yR^(m1)_k1> between final states
    # Remove duplicated final states
    all_unique = set([(m[-1],seq[-1]) for (m,seq) in all_seqs])
    Gab = make_block_hankel(Cab)
    for ff, (all_m0, all_k0) in enumerate(all_seqs):
        m0, k0 = all_m0[-1], all_k0[-1]
        if (m0,k0) not in all_unique:
            continue # already removed
        Pl = all_Pl[m0][k0] # ~ sa
        PlG = np.einsum('sa,satb->tb', Pl, Gab[:m0,:, :,:], optimize=True)
        for all_m1, all_k1 in all_seqs[ff+1:]:            
            m1, k1 = all_m1[-1], all_k1[-1]
            if (m0,k0) == (m1,k1):
                continue # degenerate final state
            if (m1,k1) not in all_unique:
                continue # already removed
            Pr = all_Pr[m1][:,:,k1] # ~ tb
            dot = np.einsum('tb,tb', PlG[:m1,:], Pr, optimize=True)
            if abs_dot: dot = np.abs(dot)
                
            if dot > thresh_duplicate:
                # duplicate found, pick one to remove
                if rule == 'latest':
                    remove = (0 if m0 < m1 else 1)
                elif rule == 'earliest':
                    remove = (0 if m0 > m1 else 1)
                elif rule == 'longest':
                    remove = (0 if len(all_m0) < len(all_m1) else 1)
                elif rule == 'leastB':
                    B0 = all_B[m0][k0]
                    B1 = all_B[m1][k1]
                    remove = (0 if B0 < B1 else 1)
                    
                # do the removal
                if remove == 0:
                    all_unique.remove((m0,k0))
                    break # don't need to test against (m0,k0) anymore
                else:
                    all_unique.remove((m1,k1))

    # Sort on state index
    all_unique = sorted(all_unique, key=lambda mk: mk[1])
    return all_unique

def pick_from_seqs_Bmin(all_seqs, Cab, all_Pl, all_Pr, all_B,
    min_len=2,
    abs_dot=True,
    thresh_duplicate=0.5,
):
    # assumes no batch indices
    assert len(Cab.shape) == 3 # ~ tab
    assert all([len(Pl.shape) == 3 for Pl in all_Pl.values()]) # ~ kta
    assert all([len(Pr.shape) == 3 for Pr in all_Pr.values()]) # ~ tak

    assert set(all_Pl.keys()) == set(all_Pr.keys()) # same m
    
    # Clip length
    all_seqs = [(m,seq) for (m,seq) in all_seqs if len(seq) >= min_len]
    
    all_Bmin = []
    for all_m, seq in all_seqs:
        B = [all_B[m][k] for m,k in zip(all_m,seq)]
        idx = np.argmin(B)
        all_Bmin.append((all_m[idx], seq[idx]))
    
    ## Consider final states in each seq
    # Compute <yL^(m0)_k0 | yR^(m1)_k1> between final states
    # Remove duplicated final states
    all_unique = set([(m[-1],seq[-1]) for (m,seq) in all_seqs])
    Gab = make_block_hankel(Cab)
    for ff, (m0,k0) in enumerate(all_Bmin):
        if (m0,k0) not in all_unique:
            continue # already removed
        Pl = all_Pl[m0][k0] # ~ sa
        PlG = np.einsum('sa,satb->tb', Pl, Gab[:m0,:, :,:], optimize=True)
        for m1, k1 in all_Bmin:
            if (m0,k0) == (m1,k1):
                continue # degenerate final state
            if (m1,k1) not in all_unique:
                continue # already removed
            Pr = all_Pr[m1][:,:,k1] # ~ tb
            dot = np.einsum('tb,tb', PlG[:m1,:], Pr, optimize=True)
            if abs_dot: dot = np.abs(dot)
                
            if dot > thresh_duplicate:
                # duplicate found, pick one to remove
                B0 = all_B[m0][k0]
                B1 = all_B[m1][k1]
                remove = (0 if B0 < B1 else 1)
                    
                # do the removal
                if remove == 0:
                    all_unique.remove((m0,k0))
                    break # don't need to test against (m0,k0) anymore
                else:
                    all_unique.remove((m1,k1))

    # Sort on state index
    all_unique = sorted(all_unique, key=lambda mk: mk[1])
    return all_unique


def pick_states(all_analyses, all_states,
    axis_for_obs=_block_axis_for_obs
):
    # all_analyses ~ {m : {obs : ...}}
    assert all([len(an['ev'].shape) == 1 for an in all_analyses.values()]) # assume no batch indices
    
    # pick out states for each obs by m,k
    all_obs = list(list(all_analyses.values())[0].keys())
    picked = {}
    for m,k in all_states:
        for o in all_obs:
            if o in axis_for_obs:
                v = np.take(all_analyses[m][o], k, axis=axis_for_obs[o])
                if o not in picked: picked[o] = []
                picked[o].append(v)
    
    # stack states w/ appropriate padding
    stacked = {}
    for o,v in picked.items():
        # special case logic for Pr, Pl: zero-pad along t
        if o in ['Pr', 'Pl']:
            stacked[o] = stack_rpadded(v, pad=0, axis=axis_for_obs[o])
        else:
            # should always be square, but add nans for edge cases
            stacked[o] = stack_rpadded(v, pad=np.nan, axis=axis_for_obs[o])
    
    return stacked


def make_bs_convergence_picks(all_filtered, Cab,
    m_min=1, m_max=10,
    thresh_converged=0.99,
    thresh_duplicate=0.5,
    rule='latest'
):

    # Flatten batch dimensions
    batch_shape = list(all_filtered.values())[0]['ev'].shape[:-1] # all ev dims but last are batch
    B = np.prod(batch_shape) # unraveled batch size
    d_batch = len(batch_shape) # number of batch dims
    all_flat = {m : {o : v.reshape(B, *v.shape[d_batch:]) for o,v in an.items()} 
                for m,an in all_filtered.items()}
    Cab_flat = Cab.reshape(B, *Cab.shape[d_batch:])

    bs_picked = {}
    for bb in range(B):
        
        sub_Cab = Cab_flat[bb]
        sub_analyses = {m : {obs : v[bb] for obs,v in an.items()} for m,an in all_flat.items()}

        all_Pl = {m : an['Pl'] for m,an in sub_analyses.items()}
        all_Pr = {m : an['Pr'] for m,an in sub_analyses.items()}
        all_seqs = find_converging_seqs(all_Pl=all_Pl, all_Pr=all_Pr, Cab=sub_Cab,
            m_min=m_min, m_max=m_max, thresh_converged=thresh_converged)
        all_picks = pick_from_seqs(all_seqs, Cab=sub_Cab, all_Pl=all_Pl, all_Pr=all_Pr,
            rule=rule, thresh_duplicate=thresh_duplicate)
        sub_picked = pick_states(sub_analyses, all_picks)

        # accumulate picked observables
        for o,v in sub_picked.items():
            if o not in bs_picked: bs_picked[o] = []
            bs_picked[o].append(v)

    assert all([len(v) == B for v in bs_picked.values()])

    # Stack and unflatten batch dimensions
    stacked = {o : stack_rpadded(v) for o,v in bs_picked.items() if o not in ['Pr', 'Pl']}
    # special-case logic for Pr,Pl
    stacked['Pl'] = stack_rpadded(bs_picked['Pl'], pad=((np.nan,np.nan),(0,0),(np.nan,np.nan)), axis=0)
    stacked['Pr'] = stack_rpadded(bs_picked['Pr'], pad=((0,0),(np.nan,np.nan),(np.nan,np.nan)), axis=0)

    # Unflatten batch dims
    stacked = {o : v.reshape(*batch_shape, *v.shape[1:]) for o,v in stacked.items()}

    return stacked

def cluster_boot_states(all_picked, Cab,
    abs_dot=True,
    renorm_dot=False,
    group_thresh=0.7,
    sort=True
):
    B, Nt, bdim, _ = Cab.shape
    assert Cab.shape == (B, Nt, bdim, bdim)
    assert len(all_picked['ev'].shape) == 2 # ~ Bk, assume single batch dim
    assert all_picked['ev'].shape[0] == B
    
    ## precompute <y^(b)_k|y^{b'}_l>
    Pl = all_picked['Pl'] # ~ Bksa
    Pr = all_picked['Pr'] # ~ Btbk
    m_max = Pl.shape[-2]
    assert Pr.shape[-3] == m_max
    
    # use mean Cab for metric
    Cab0 = Cab.mean(0)[:2*m_max]
    Gab0 = make_block_hankel(Cab0)
    
    print(Pl.shape, Gab0.shape, Pr.shape)
    dot = np.einsum('Bksa,satb,Ctbl->BCkl', Pl, Gab0, Pr, optimize=True)
    if renorm_dot:
        lnorm = np.einsum('Bksa,satb,Bktb->Bk', Pl, Gab0, Pl.conj(), optimize=True)
        rnorm = np.einsum('Bsak,satb,Btbk->Bk', Pr.conj(), Gab0, Pr, optimize=True)
        dot = dot / (lnorm[:,None,:,None] * rnorm[None,:,None,:])
    if abs_dot: dot = np.abs(dot)
        
    ## Make clusters
    # seed with first boot
    label_groups = [[(0,k)] for k in range(dot.shape[-1]) if np.isfinite(dot[0,0,k,k])]
    # add to groups one boot at a time
    for b0 in range(1,dot.shape[0]):
        for k0 in range(dot.shape[-1]):
            if ~np.isfinite(dot[b0,b0,k0,k0]):
                continue # missing (b0,k0), do not associate
            for gg, group in enumerate(label_groups):
                mismatch = [dot[b0,b1,k0,k1] < group_thresh for b1,k1 in group]
                if not any(mismatch):
                    # add to group
                    group.append((b0,k0))
                    break # only allocate to one group
            else:
                # no match found, make new group
                label_groups.append([(b0,k0)])

    # Sanity check: at most one choice per boot
    for group in label_groups:
        all_b = [b for b,k in group]
        assert len(all_b) == len(set(all_b)), f"Multiple boots in {group}"
    
    # Optionally, sort groups by median ev
    if sort:
        ev_for_group = [np.nanmedian([all_picked['ev'][b,k] for b,k in g]) for g in label_groups]
        sort_idx = np.argsort(-np.array(ev_for_group).real)
        label_groups = [label_groups[ii] for ii in sort_idx]
    
    return label_groups

def relabel_analysis(all_picked, label_groups,
    min_group_size=1,
    axis_for_obs=_block_axis_for_obs
):
    assert len(all_picked['ev'].shape) == 2 # assume only one batch dim
    N_boots = all_picked['ev'].shape[0]

    all_relabeled = {}
    for group in label_groups:
        if len(group) < min_group_size: continue
        drelabel = {}
        for obs, v_src in all_picked.items():
            v_dest = [None] * N_boots
            for b0,k0 in group:
                v_dest[b0] = np.take(v_src[b0], k0, axis=axis_for_obs[obs])
            ex = [x for x in v_dest if x is not None][0] # find non-None example to steal shape
            v_dest = [(np.nan * np.zeros_like(ex) if x is None else x) for x in v_dest]
            v_dest = stack_rpadded(v_dest, axis=0) # cast to empty array for compatibility
            drelabel[obs] = v_dest

        for obs, v in drelabel.items():
            if obs not in all_relabeled: all_relabeled[obs] = []
            all_relabeled[obs].append(v)

    all_relabeled = {obs : stack_rpadded(v, axis=axis_for_obs[obs]) 
                     for obs,v in all_relabeled.items()}
    return all_relabeled

