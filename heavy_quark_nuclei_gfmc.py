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

# Build Coulomb Potential


### Load potential and WF
AV_Coulomb = {}
AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)

Coulomb_potential = adl.make_pairwise_potential(AV_Coulomb)

# Build Coulomb wavefunction

trial_wvfn = wvfn()
print(trial_wvfn.A)
f_R = lambda R: trial_wvfn.psi(torch.from_numpy(np.asarray(R))).detach().numpy()
laplacian_f_R = lambda R: trial_wvfn.laplacian(torch.from_numpy(np.asarray(R))).detach().numpy()

deuteron_trial_weight = lambda R: np.abs(f_R(R)**2)

### metropolis
Rs_fname = f'metropolis_N{n_walkers}_Rs.npy'
Ss_fname = f'metropolis_N{n_walkers}_Ss.npy'
if not os.path.exists(Rs_fname) or not os.path.exists(Ss_fname):
    print('Generating/writing wavefunction metropolis samples...')
    #R0 = np.array([[[-0.5, 0, 0], [0.5, 0, 0]]])
    #print(R0.shape)
    #samples = adl.metropolis(R0, deuteron_trial_weight, n_therm=1000, n_step=n_walkers,# n_skip=10, eps=1.0)
    #print(samples.shape)
    #print(samples[:,0].shape)
    Rs_metropolis = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=n_walkers, n_skip=10, eps=trial_wvfn.A[0].item()/N_coord**2)[0]
    #Rs_metropolis = np.array([R for R,_ in samples])
    print(Rs_metropolis.shape)
    print(Rs_metropolis)
    # Ws_metropolis = np.array([W for _,W in samples])

    S_av4p_metropolis = np.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
    # antisymmetric spin-iso WF
    S_av4p_metropolis[:,(0,)*NS*NI*N_coord] = 1
    np.save(Rs_fname, Rs_metropolis)
    np.save(Ss_fname, S_av4p_metropolis)


print('Loading wavefunction samples...')
Rs_metropolis = np.load(Rs_fname)
S_av4p_metropolis = np.load(Ss_fname)

print(S_av4p_metropolis.shape)

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
rand_draws = np.random.random(size=(n_step, Rs_metropolis.shape[0]))
gfmc = adl.gfmc_deform(
    Rs_metropolis, S_av4p_metropolis, f_R, params,
    rand_draws=rand_draws, tau_iMev=tau_iMev, N=n_step, potential=Coulomb_potential,
    deform_f=deform_f, m_Mev=adl.mp_Mev,
    resampling_freq=resampling)
gfmc_Rs = np.array([Rs for Rs,_,_,_, in gfmc])
gfmc_Ws = np.array([Ws for _,_,_,Ws, in gfmc])
gfmc_Ss = np.array([Ss for _,_,Ss,_, in gfmc])

print('GFMC tau=0 weights:', gfmc_Ws[0])
print('GFMC tau=dtau weights:', gfmc_Ws[1])

# measure H
print('Measuring <H>...')

Ks = []
for R in tqdm.tqdm(gfmc_Rs):
    #Ks.append(-1/2*laplacian_f_R(R) / f_R(R) / adl.mp_Mev)
    Ks.append(-1/2*laplacian_f_R(R) / f_R(R) / 1)
Ks = np.array(Ks)

#Ks *= fm_Mev**2

#Vs = np.array([
#    sum([
#        AV_Coulomb[name](dRs) * adl.compute_O(adl.two_body_ops[name](dRs), S, S_av4p_metropolis)
#        for name in AV_Coulomb
#    ])
#    for dRs, S in zip(map(adl.to_relative, gfmc_Rs), gfmc_Ss)])

# TODO IS THIS RIGHT? MORE ROBUST TO COPY THE POTENTIAL CREATION FUNCTION
Vs = np.array([
    sum([
        AV_Coulomb[name](Rs) # * adl.compute_O(adl.two_body_ops[name](Rs), S, S_av4p_metropolis)
        for name in AV_Coulomb
    ])
    for Rs in gfmc_Rs])

Hs = np.array([al.bootstrap(Ks + Vs, Ws, Nboot=100, f=adl.rw_mean)
        for Ks,Vs,Ws in zip(Ks, Vs, gfmc_Ws)])

ave_Ks = np.array([al.bootstrap(Ks, Ws, Nboot=100, f=adl.rw_mean)
        for Ks,Vs,Ws in zip(Ks, Vs, gfmc_Ws)])
ave_Vs = np.array([al.bootstrap(Vs, Ws, Nboot=100, f=adl.rw_mean)
        for Ks,Vs,Ws in zip(Ks, Vs, gfmc_Ws)])

print("H=",Hs,"\n\n")
print("K=",ave_Ks,"\n\n")
print("V=",ave_Vs,"\n\n")

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
ax.set_ylim(-1, 0)
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
