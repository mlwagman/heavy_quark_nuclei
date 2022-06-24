### Eval script for deuteron GFMC deformation.

import argparse
import analysis as al
import getpass
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.interpolate
import pickle
import paper_plt
import tqdm.auto as tqdm
import afdmc_lib as adl
import os
import pickle
from afdmc_lib import NI,NS,mp_Mev,fm_Mev
import jax
import sys
from heavy_quark_nuclei_variational_test import *


paper_plt.load_latex_config()

parser = argparse.ArgumentParser()
parser.add_argument('--n_walkers', type=int, default=1000)
parser.add_argument('--dtau_iMev', type=float, required=True)
parser.add_argument('--n_step', type=int, required=True)
parser.add_argument('--resampling', type=int, default=3)
globals().update(vars(parser.parse_args()))

# Set up GFMC
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)

### Load potential and WF
DR = 0.01
N_STEPS = 1024
N_OPS = 6
rs = np.linspace(0, N_STEPS*DR, num=N_STEPS, endpoint=False)
vnn = np.fromfile('av6p_table.dat', dtype=np.float64).reshape(N_STEPS, N_OPS)
AV6p = {}
for i in range(N_OPS):
    AV6p[f'O{i+1}'] = adl.chebyshev.make_interp_function_rsq(
        rs**2, vnn[:,i], rsqi=np.linspace(0.0, 20.0**2, num=100000), ncheb=150)

av6p_potential = adl.make_pairwise_potential(AV6p)

estimate_av6p_H = adl.make_twobody_estimate_H(AV6p)

f_R, df_R, ddf_R = adl.chebyshev.load_nn_wavefunction_rsq('psi_deuteron_av4p_ale.dat', ncheb=150)

def old_laplacian(R):
    rsq = norm_3vec_sq(R)
    return (6*df_R + 4*rsq*ddf_R)*fm_Mev**2

f_R_norm, df_R_norm, ddf_R_norm = adl.normalize_wf(f_R, df_R, ddf_R)

quadrature_batch = 10000
dR = 20 / quadrature_batch
R_av4p = np.linspace([[-1e-4,0,0], [1e-4,0,0]], [[-10,0,0], [10,0,0]],
                     num=quadrature_batch, endpoint=True)
print(np.mean(R_av4p[:,0] + R_av4p[:,1]))
Rs = R_av4p[:,0] - R_av4p[:,1]
S_av4p = np.zeros(shape=(quadrature_batch,) + (NI,NS)*2)
# antisymmetric spin-iso WF
S_av4p[:,0,0,1,0] = 1/np.sqrt(2)
S_av4p[:,1,0,0,0] = -1/np.sqrt(2)

### check f is normalized
Rs = np.linspace([0,0,0], [20,0,0], endpoint=False, num=10000)
f = f_R_norm(Rs)
print('norm f =', np.sum(4*np.pi * dR * adl.norm_3vec(Rs)**2 * f**2))

deuteron_trial_weight = adl.make_wf_weight(f_R_norm)

### metropolis
Rs_fname = f'metropolis_N{n_walkers}_Rs.npy'
Ss_fname = f'metropolis_N{n_walkers}_Ss.npy'
if not os.path.exists(Rs_fname) or not os.path.exists(Ss_fname):
    print('Generating/writing wavefunction metropolis samples...')
    R0 = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
    samples = adl.metropolis(R0, deuteron_trial_weight, n_therm=1000, n_step=n_walkers, n_skip=10, eps=1.0)
    Rs_metropolis = np.array([R[0]-R[1] for R,_ in samples])
    # Ws_metropolis = np.array([W for _,W in samples])

    S_av4p_metropolis = np.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*2).astype(np.complex128)
    # antisymmetric spin-iso WF
    S_av4p_metropolis[:,0,0,1,0] = 1/np.sqrt(2)
    S_av4p_metropolis[:,1,0,0,0] = -1/np.sqrt(2)
    np.save(Rs_fname, Rs_metropolis)
    np.save(Ss_fname, S_av4p_metropolis)

print('Loading wavefunction samples...')
Rs_metropolis = np.load(Rs_fname)
S_av4p_metropolis = np.load(Ss_fname)


### TEST: Measure <H> at tau = 0. This looks good.
# f = f_R_norm(Rs_metropolis)
# df = df_R_norm(Rs_metropolis)
# ddf = ddf_R_norm(Rs_metropolis)
# estH = estimate_av6p_H(
#     Rs_metropolis, S_av4p_metropolis, np.ones_like(f),
#     S_av4p_metropolis, f, df, ddf, m_Mev=mp_Mev, Nboot=100, verbose=True)
# print(estH)
# import sys
# sys.exit()

# Trivial contour deformation
deform_f = lambda x, params: x
params = (np.zeros((n_step+1)),)

print('Running GFMC evolution:')
AVcoeffs = AV6p
potential = adl.make_pairwise_potential(AVcoeffs)
rand_draws = np.random.random(size=(n_step, Rs_metropolis.shape[0]))
gfmc = adl.gfmc_twobody_deform(
    Rs_metropolis, S_av4p_metropolis, f_R_norm, params,
    rand_draws=rand_draws, tau_iMev=tau_iMev, N=n_step, potential=potential,
    deform_f=deform_f, m_Mev=adl.mp_Mev,
    resampling_freq=resampling)
gfmc_Rs = np.array([Rs for Rs,_,_,_, in gfmc])
gfmc_Ws = np.array([Ws for _,_,_,Ws, in gfmc])
gfmc_Ss = np.array([Ss for _,_,Ss,_, in gfmc])

print('GFMC tau=0 weights:', gfmc_Ws[0])
print('GFMC tau=dtau weights:', gfmc_Ws[1])

# measure H
print('Measuring <H>...')
res = adl.measure_gfmc_obs_deform(
    gfmc, estimate_av6p_H,
    f_R_norm, df_R_norm, ddf_R_norm, verbose=False)
Hs = res['H']


def old_laplacian(R,f_R, df_R, ddf_R,fm_Mev):
    rsq = norm_3vec_sq(R)
    return (6*df_R + 4*rsq*ddf_R)*fm_Mev**2
# get raw samples of H terms
#Ks = np.array([
#    adl.compute_K(dRs, f_R_norm(dRs), df_R_norm(dRs), ddf_R_norm(dRs), m_Mev=adl.mp_Mev)
#    for dRs in map(adl.to_relative, gfmc_Rs)])
Ks = np.array([old_laplacian(dRs,f_R_norm(dRs), df_R_norm(dRs),ddf_R_norm(dRs))/(f* m_MeV)
    for dRs in map(adl.to_relative, gfmc_Rs)])
Vs = np.array([
    sum([
        AVcoeffs[name](dRs) * adl.compute_O(adl.two_body_ops[name](dRs), S, S_av4p_metropolis)
        for name in AVcoeffs
    ])
    for dRs, S in zip(map(adl.to_relative, gfmc_Rs), gfmc_Ss)])

# NOTE: These match the directly evaluated <H> correctly!
# print('undeform <H> = ',
#       [al.bootstrap(Ks + Vs, Ws, Nboot=100, f=adl.rw_mean)
#        for Ks,Vs,Ws in zip(Ks, Vs, gfmc_Ws)])
# print('deform <H> = ',
#       [al.bootstrap(Ks + Vs, Ws, Nboot=100, f=adl.rw_mean)
#        for Ks,Vs,Ws in zip(Ks_deform, Vs_deform, gfmc_deform_Ws)])

# plot H
fig, ax = plt.subplots(1,1, figsize=(4,3))
al.add_errorbar(np.transpose(Hs), ax=ax, xs=xs, color='xkcd:forest green', label=r'$\left< H \right>$', marker='o')
ax.set_ylim(-5, 6.5)
ax.legend()

def make_H_err_plt(xs, H_errs):
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    for a in ax:
        a.plot(xs, np.sqrt(H_errs), label=r"sqrt(bootstrap variance)", color='k')
    ax[0].set_yscale('log')
    fig.legend()

make_H_err_plt(xs, Hs[:,1])

def add_noise_plot(xs, Ks, Vs, Ws, *, ax, label):
    print(Ks.shape, Ws.shape, xs.shape)
    Hs = Ks + Vs
    num_re_loss = np.mean(np.real(Ws * Hs)**2, axis=1)
    num_im_loss = np.mean(np.imag(Ws * Hs)**2, axis=1)
    den_re_loss = np.mean(np.real(Ws)**2, axis=1)
    den_im_loss = np.mean(np.imag(Ws)**2, axis=1)
    assert(num_re_loss.shape == xs.shape)
    ax.plot(xs, 0.5*np.log(num_re_loss + num_im_loss) + 0.5*np.log(den_re_loss + den_im_loss),
            marker='o', label=label)

fig, ax = plt.subplots(1,1, figsize=(4,3))
add_noise_plot(xs, Ks, Vs, gfmc_Ws, ax=ax, label='var(H)')
fig.legend()

plt.show()
