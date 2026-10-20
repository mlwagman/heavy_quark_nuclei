#!/usr/bin/env python3
"""Outlier-robust estimators.
"""

import numpy as np
import math
from .util import thresh_missing

# batched inner median with missingness thresholds
def inner_median(x, thresh=0.95, thresh_inner=None):
    # x ~ (outer, inner, ...)
    if thresh_inner is None: # overrides thresh if provided
        thresh_inner = thresh
    x = thresh_missing(x, thresh=thresh_inner, axis=1)
    x = np.nanmedian(np.real(x), axis=1)
    x = thresh_missing(x, thresh=thresh, axis=0)
    return x

# batched median with missingness thresholds
def batched_median(x, thresh=0.95):
    # x ~ (inner, ...)
    x = thresh_missing(x, thresh=thresh, axis=0)
    x = np.nanmedian(np.real(x), axis=0)
    return x

# median final few iterations
def final_iter_median(x, N_states=1, m_window=1, final_m=None):
    all_m = list(x.keys())
    # find last step with residuals available
    if final_m is None:
        final_m = all_m[-2]
    # create output dict
    x_median = { k : v.copy() for k,v in x[final_m].items() }
    for key,val in x_median.items():
        # get values from steps m\in[final_m-(m_window-1), final_m]
        final_few = np.array([x[m][key] for m in all_m[(final_m-m_window+1):(final_m+1)]])
        # take median
        final_few_med = np.nanmedian(final_few, axis=0)
        # store in output
        x_median[key] = final_few_med
    return x_median

def homogenize_analyses(x, N_states=1, state_ax=0):
    all_m = list(x.keys())
    m_long = all_m[np.argmax([x[m]['ev'].shape[-1] for m in all_m])]
    for m,an in x.items():
        for key,val in an.items():
            if key == 'Pr':
                val = np.einsum('...tak->...kta', val)
            # in case too small
            if val.shape[state_ax] < N_states:
                ind = ''
                # only zero, one, or two-level bootstrap supported
                assert(state_ax < 3)
                ind_list = ['b', 'i']
                dims = [N_states-val.shape[state_ax]]
                for ax in range(0, state_ax):
                    ind = ind + ind_list[ax]
                    dims.append(val.shape[ax])
                # kbi or substring
                ind_T = 'k' + ind
                # bik or substring
                ind = ind + 'k'
                # transpose state_ax <-> 0
                ein_T = ind + '...->' + ind_T + '...'
                ein_Tdag = ind_T + '...->' + ind + '...'
                val_T = np.einsum(ein_T, val)
                # build nan padding
                ones_T = np.nan*np.ones(tuple(dims))
                ein_broadcast = ind_T + '...,' + ind_T[1:] + '...->' + ind_T + '...'
                if key == 'Pl' or key == 'Pr':
                    ones_T = np.einsum(ein_broadcast, ones_T, x[m_long]['Pl'].take(indices=0, axis=state_ax)[...,0:m,:])
                else:
                    ones_T = np.einsum(ein_broadcast, ones_T, x[m_long][key].take(indices=0, axis=state_ax))
                # add nan padding
                if state_ax == 0:
                    val = np.concatenate([an[key], ones_T])
                else:
                    # ones_T inner dims may differ from val_T due to different
                    # reference analysis; force match since ones_T is all NaN
                    if val_T.shape[1:] != ones_T.shape[1:]:
                        ones_T = np.full(
                            (ones_T.shape[0],) + val_T.shape[1:],
                            np.nan, dtype=val_T.dtype)
                    val = np.einsum(ein_Tdag, np.vstack([val_T, ones_T]))
            # in case too big
            val = val.take(indices=range(0,N_states), axis=state_ax)
            if key == 'Pr':
                val = np.einsum('...kta->...tak', val)
            x[m][key] = val
    return x


# 1sigma empirical bootstrap confidence intervals
def make_boot_ci(boot_x, sigma=1, side="", cv=0):
    # positions of 1sigma confidence intervals
    # ~ np.sqrt(np.mean(boot_x**2) - np.mean(boot_x)**2)
    lQ = (1 - math.erf(sigma/math.sqrt(2)))/2
    uQ = 1 - lQ
    # empirical bootstrap confidence interval width
    upper = np.nanquantile(boot_x, uQ, method='averaged_inverted_cdf', axis=0)
    lower = np.nanquantile(boot_x, lQ, method='averaged_inverted_cdf', axis=0)
    if side == "lower":
        return cv - lower
    elif side == "upper":
        return upper - cv
    else:
        return 0.5*(upper-lower)

def minres(analysis):
    all_m = sorted(list(analysis.keys()))
    N_iter = len(all_m)
    (N_boot, N_states) = analysis[all_m[-1]]['Bl'].shape
    # holder for return dictionary
    minres_analysis = analysis[all_m[-1]]
    for b in range(N_boot):
        for k in range(N_states):
            # store Bs from each iteration
            Bs = np.nan*np.zeros(N_iter)
            # skip last iteration, nans
            for mi, m in enumerate(all_m[:-1]):
                B_shape = analysis[m]['Bl'][b].shape
                if B_shape[-1] > k:
                    Bs[mi] = min([np.abs(analysis[m]['Bl'][b,k]), np.abs(analysis[m]['Br'][b,k])])
            # find minres B
            min_B = np.nanmin(Bs)
            # if all iterations are nan for this state, leave as is (the last iter is nan in particular)
            if np.isnan(min_B):
                continue
            # find minres B index
            minres_m = [all_m[mi] for mi, B in enumerate(Bs) if B == min_B][-1]
            # store values at minres iteration
            for key,val in minres_analysis.items():
                new_val = analysis[minres_m][key]
                val[b,k] = new_val[b,k]
    return minres_analysis

def nbs_minres(analysis):
    all_m = sorted(list(analysis.keys()))
    N_iter = len(all_m)
    (N_outer, N_inner, N_states) = analysis[all_m[-1]]['Bl'].shape
    # holder for return dictionary
    minres_analysis = analysis[all_m[-1]]
    for b in range(N_outer):
        for ib in range(N_inner):
            for k in range(N_states):
                # store Bs from each iteration
                Bs = np.nan*np.zeros(N_iter)
                # skip last iteration, nans
                for mi, m in enumerate(all_m[:-1]):
                    B_shape = analysis[m]['Bl'][b,ib].shape
                    if B_shape[-1] > k:
                        Bs[mi] = min([np.abs(analysis[m]['Bl'][b,ib,k]), np.abs(analysis[m]['Br'][b,ib,k])])
                # find minres B
                min_B = np.nanmin(Bs)
                # if all iterations are nan for this state, leave as is (the last iter is nan in particular)
                if np.isnan(min_B):
                    continue
                # find minres B index
                minres_m = [all_m[mi] for mi, B in enumerate(Bs) if B == min_B][-1]
                # store values at minres iteration
                for key,val in minres_analysis.items():
                    new_val = analysis[minres_m][key]
                    val[b,ib,k] = new_val[b,ib,k]
    return minres_analysis


# get bootstrap median for one iteration
def apply_median_iter(x, Q=0.05):
    N_boot = len(x)
    # max number of states in fraction Q of bootstraps
    N_states = int(np.quantile([len(x[b]) for b in range(N_boot)], Q))
    x_array = np.zeros(N_states)
    # check whether data is scalar or vector (block case)
    nbs = 1
    for b in range(N_boot):
        if len(x[b]) > 0:
            if len(np.array(x[b]).shape) > 1:
                nbs = np.array(x[b]).shape[-1]
    if nbs > 1:
        x_array = np.zeros((N_states,nbs))
        for n in range(N_states):
            # bootstrap median for each state
            x_list = [ [] for a in range(nbs) ]
            for b in range(N_boot):
                if len(x[b]) > n:
                    # loop over operators in block case
                    for a in range(nbs):
                        x_list[a].append(x[b][n][a])
            for a in range(nbs):
                x_array[n,a] = np.median(np.array(x_list[a]))
    else:
        for n in range(N_states):
            # bootstrap median for each state
            x_list = [ ]
            for b in range(N_boot):
                if len(x[b]) > n:
                    x_list.append(x[b][n])
            x_array[n] = np.median(np.array(x_list))
    return x_array

# get bootstrap median for all iterations
def apply_median(all_x, Q=0.05):
    all_m = all_x.keys()
    med_x = {}
    for m in all_m:
        med_x[m] = apply_median_iter(all_x[m], Q)
    return med_x

# get nested bootstrap median for all iterations
def apply_nested_median(all_x, Q=0.05):
    all_m = all_x.keys()
    N_boot = len(all_x[list(all_m)[0]])
    nested_med_x = {m: [] for m in all_m}
    for m in all_m:
        # inner bootstrap median for all states
        med_x = []
        for x_inner in all_x[m]:
             x_inner_med = apply_median_iter(x_inner, Q)
             med_x.append(x_inner_med)
        # max number of states in fraction Q of outer bootstraps
        N_states = int(np.quantile([len(med_x[b]) for b in range(N_boot)], Q))
        # inner bootstrap median for all states
        x_outer_list = []
        for n in range(N_states):
            x_list = []
            for b in range(N_boot):
                if len(med_x[b]) > n:
                    x_list.append(med_x[b][n])
            x_outer_list.append(np.array(x_list))
        nested_med_x[m] = x_outer_list
    return(nested_med_x)

def get_boot_err(boot_x):
    # positions of 1sigma confidence intervals
    # ~ np.sqrt(np.mean(boot_x**2) - np.mean(boot_x)**2)
    lQ = (1 - math.erf(1/math.sqrt(2)))/2
    uQ = 1 - lQ
    # empirical bootstrap confidence interval width
    upper = np.quantile(boot_x, uQ, method='averaged_inverted_cdf')
    lower = np.quantile(boot_x, lQ, method='averaged_inverted_cdf')
    return 0.5*(upper-lower)

# statistical errors from nested bootstrap
def get_final_nested_median_err(nested_med_x, nested_med_final_m, N_states):
    N_boot = nested_med_final_m.shape[0]
    nbs = 1
    err_x = np.zeros((N_states))
    boot_x = np.zeros((N_boot,N_states))
    # check whether data is scalar or vector (block case)
    boot_op_shape = np.array(nested_med_x[nested_med_final_m[0,0]][0]).shape
    if len(boot_op_shape) > 1:
        nbs = boot_op_shape[-1]
        err_x = np.zeros((N_states, nbs))
        boot_x = np.zeros((N_boot, N_states, nbs))
    for n in range(N_states):
        x_list = []
        for b in range(N_boot):
            # get all x from final iteration
            if nested_med_final_m[b,n] == 0:
                final_x = []
            else:
                final_x = nested_med_x[nested_med_final_m[b,n]]
            # check whether this state/boot exist
            if len(final_x) > n:
                if len(final_x[n]) > b:
                    # add to list of final x
                    x_list.append(final_x[n][b])
        if len(x_list) == 0:
            err_x[n] = np.nan
            continue
        if nbs > 1:
            # loop over operators in block case
            for a in range(nbs):
                this_x = np.array(x_list)[:,a]
                err_x[n,a] = get_boot_err(this_x)
                boot_x[:len(this_x),n,a] = this_x
        else:
            err_x[n] = get_boot_err(x_list)
            boot_x[:len(x_list),n] = np.array(x_list)
    return err_x, boot_x

# statistical errors from nested bootstrap
def get_nested_median_err(nested_med_x):
    all_m = nested_med_x.keys()
    err_x = {m: [] for m in all_m}
    # loop over iterations
    for m in all_m:
        N_states = len(nested_med_x[m])
        nbs = 1
        err_x[m] = np.zeros(N_states)
        # check whether data is scalar or vector (block case)
        boot_op_shape = np.array(nested_med_x[m][0]).shape
        if len(boot_op_shape) > 1:
            nbs = boot_op_shape[-1]
            err_x[m] = np.zeros((N_states, nbs))
        # loop over states
        for n in range(N_states):
            x_list = nested_med_x[m][n]
            if nbs > 1:
                # loop over operators in block case
                for a in range(nbs):
                    err_x[m][n,a] = get_boot_err(np.array(x_list)[:,a])
            else:
                err_x[m][n] = get_boot_err(x_list)
    return err_x

# get bootstrap median for one iteration
def get_bounds_median_iter(ev, B, Q=0.05, true_E=None):
    N_boot = len(ev)
    # max number of states in fraction Q of bootstraps
    N_states = int(np.quantile([len(ev[b]) for b in range(N_boot)], Q))
    res_lower_array = np.zeros(N_states)
    res_upper_array = np.zeros(N_states)
    gap_lower_array = np.zeros(N_states)
    gap_upper_array = np.zeros(N_states)
    # check whether data is scalar or vector (block case)
    for n in range(N_states):
        # bootstrap median for each state
        res_lower_list = [ ]
        res_upper_list = [ ]
        gap_lower_list = [ ]
        gap_upper_list = [ ]
        for b in range(N_boot):
            if len(ev[b]) > n:
                this_ev = ev[b][n]
                this_B = np.abs(B[b][n])
                #this_B = np.max([B[b][n], 0])
                res_lower_list.append( -np.log(this_ev + np.sqrt(this_B)) )
                res_upper_list.append( -np.log(this_ev - np.sqrt(this_B)) )
                #res_upper_list.append( -np.log(np.max([this_ev - np.sqrt(this_B),1e-8])) )
                gap = None
                if true_E is not None:
                    gap = np.abs(this_ev - np.exp(-true_E))
                elif len(ev[b]) > 1:
                    diffs = np.abs(ev[b] - this_ev)
                    gap = np.sort(diffs)[1]
                if gap is not None:
                    gap_lower_list.append( -np.log(this_ev + this_B/gap) )
                    gap_upper_list.append( -np.log(this_ev - this_B/gap) )
                    #gap_upper_list.append( -np.log(np.max([this_ev - this_B/gap,1e-8])) )
        res_lower_array[n] = np.nanmedian(np.array(res_lower_list))
        res_upper_array[n] = np.nanmedian(np.array(res_upper_list))
        if len(gap_upper_list) > 0:
            gap_lower_array[n] = np.nanmedian(np.array(gap_lower_list))
            gap_upper_array[n] = np.nanmedian(np.array(gap_upper_list))
    return res_lower_array, res_upper_array, gap_lower_array, gap_upper_array

# get bootstrap median for all iterations
def get_bounds_median(all_ev, all_B, Q=0.05, true_E=None):
    all_m = all_ev.keys()
    med_res_lower = {}
    med_res_upper = {}
    med_gap_lower = {}
    med_gap_upper = {}
    for m in all_m:
        med_res_lower[m], med_res_upper[m], med_gap_lower[m], med_gap_upper[m] = get_bounds_median_iter(all_ev[m], all_B[m], Q=Q, true_E=true_E)
    return med_res_lower, med_res_upper, med_gap_lower, med_gap_upper

# get nested bootstrap median for all iterations
def get_bounds_nested_median(all_ev, all_B, Q=0.05, true_E=None):
    all_m = all_ev.keys()
    N_boot = len(all_ev[list(all_m)[0]])
    nested_med_res_lower = {m: [] for m in all_m}
    nested_med_res_upper = {m: [] for m in all_m}
    nested_med_gap_lower = {m: [] for m in all_m}
    nested_med_gap_upper = {m: [] for m in all_m}
    for m in all_m:
        # inner bootstrap median for all states
        med_res_lower = []
        med_res_upper = []
        med_gap_lower = []
        med_gap_upper = []
        for b in range(N_boot):
            bounds_inner_med = get_bounds_median_iter(all_ev[m][b], all_B[m][b], Q=Q, true_E=true_E)
            med_res_lower.append(bounds_inner_med[0])
            med_res_upper.append(bounds_inner_med[1])
            med_gap_lower.append(bounds_inner_med[2])
            med_gap_upper.append(bounds_inner_med[3])
        # max number of states in fraction Q of outer bootstraps
        N_states = int(np.quantile([len(med_res_lower[b]) for b in range(N_boot)], Q))
        # inner bootstrap median for all states
        res_lower_outer_list = []
        res_upper_outer_list = []
        gap_lower_outer_list = []
        gap_upper_outer_list = []
        for n in range(N_states):
            res_lower_list = []
            res_upper_list = []
            gap_lower_list = []
            gap_upper_list = []
            for b in range(N_boot):
                if len(med_res_lower[b]) > n:
                    res_lower_list.append(med_res_lower[b][n])
                    res_upper_list.append(med_res_upper[b][n])
                if len(med_gap_lower[b]) > n:
                    gap_lower_list.append(med_gap_lower[b][n])
                    gap_upper_list.append(med_gap_upper[b][n])
            res_lower_outer_list.append(np.array(res_lower_list))
            res_upper_outer_list.append(np.array(res_upper_list))
            gap_lower_outer_list.append(np.array(gap_lower_list))
            gap_upper_outer_list.append(np.array(gap_upper_list))
        nested_med_res_lower[m] = res_lower_outer_list
        nested_med_res_upper[m] = res_upper_outer_list
        nested_med_gap_lower[m] = gap_lower_outer_list
        nested_med_gap_upper[m] = gap_upper_outer_list
    return nested_med_res_lower, nested_med_res_upper, nested_med_gap_lower, nested_med_gap_upper

# imitates gv.dataset.avg_data(..., bstrap=True)
def outlier_robust_avg(data):
    N_bs, *shape = data.shape
    data = data.reshape(N_bs, -1)

    bs_mu = np.median(data, axis=0)
    bs_corr = np.corrcoef(data, rowvar=False)

    bdry = (0.5-0.341344746)*100
    bs_dL = bs_mu - np.percentile(data, bdry, axis=0)
    bs_dR = np.percentile(data, 100-bdry, axis=0) - bs_mu
    spreads = np.max(
        np.concatenate((np.expand_dims(bs_dL, 0), np.expand_dims(bs_dR, 0)), axis=0),
        axis=0
    )
    bs_cov = bs_corr * np.einsum('a,b', spreads, spreads)

    bs_mu = bs_mu.reshape(shape)
    bs_cov = bs_cov.reshape(shape+shape)

    return bs_mu, bs_cov

# imitates gv.dataset.avg_data(..., bstrap=True), but doesn't compute a covariance matrix
def outlier_robust_avg_decorr(data):
    N_bs, *shape = data.shape
    data = data.reshape(N_bs, -1)

    bs_mu = np.median(data, axis=0)

    bdry = (0.5-0.341344746)*100
    bs_dL = bs_mu - np.percentile(data, bdry, axis=0)
    bs_dR = np.percentile(data, 100-bdry, axis=0) - bs_mu
    spreads = np.max(
        np.concatenate((np.expand_dims(bs_dL, 0), np.expand_dims(bs_dR, 0)), axis=0),
        axis=0
    )

    bs_mu = bs_mu.reshape(shape)
    spreads = spreads.reshape(shape)

    return bs_mu, spreads

# TODO: make not suck
# TODO: add axis argument
def symm_discard(x, force_nan=False):
    # symmetrically replace non-finite values with +/- inf
    # iff +/-1sigma worth of values are left afterwards
    # considers axis 0, all other dims are batch-like
    shape = x.shape
    x = x.reshape(x.shape[0], -1)
    bdry = 15.865525400000003
    keep = np.isfinite(x).mean(0) > (1 - 2*bdry/100)
    x = x.copy()
    for ii in range(x.shape[1]):
        if not keep[ii]:
            if force_nan: x[:,ii] = np.nan
            continue
        idx = np.sort(np.where(~np.isfinite(x[:,ii]))[0])
        assert len(idx) < x.shape[0]*(2*bdry/100)
        lo = idx[:len(idx)//2]
        hi = idx[len(idx)//2:]
        x[lo,ii] = -np.inf
        x[hi,ii] = np.inf
    x = x.reshape(shape)
    return x
