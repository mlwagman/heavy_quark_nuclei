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
from itertools import repeat


paper_plt.load_latex_config()

parser = argparse.ArgumentParser()
parser.add_argument('--n_walkers', type=int, default=1000)
parser.add_argument('--dtau_iMev', type=float, required=True)
parser.add_argument('--n_step', type=int, required=True)
parser.add_argument('--resampling', type=int, default=3)
globals().update(vars(parser.parse_args()))

# imaginary time points for GFMC evolution
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)

# build Coulomb potential
AV_Coulomb = {}
AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)
Coulomb_potential = adl.make_pairwise_potential(AV_Coulomb)

# build Coulomb ground-state trial wavefunction
trial_wvfn = wvfn()
print(trial_wvfn.A)
f_R = lambda R: trial_wvfn.psi(torch.from_numpy(np.asarray(R))).detach().numpy()
laplacian_f_R = lambda R: trial_wvfn.laplacian(torch.from_numpy(np.asarray(R))).detach().numpy()

# Metropolis
Rs_metropolis = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=n_walkers, n_skip=10, eps=trial_wvfn.A[0].item()/N_coord**2)[0]
Rs_metropolis = Rs_metropolis.detach().numpy()
# build trial wavefunction
S_av4p_metropolis = np.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
print("built Metropolis wavefunction ensemble")
# trial wavefunction spin-flavor structure is |up,u> x |up,u> x ... x |up,u>
spin_slice = (slice(0,None),) + (0,)*2*N_coord
S_av4p_metropolis[spin_slice] = 1
print("spin-flavor wavefunction shape = ", S_av4p_metropolis.shape)

# trivial contour deformation
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

Vs = []
for Rs in gfmc_Rs:
    VSI,_ = Coulomb_potential(Rs)
    V_ind = (slice(0,None),) + (0,)*NS*NI*N_coord
    Vs.append(VSI[V_ind])

Vs = np.array(Vs)
print(Vs.shape)

Hs = np.array([al.bootstrap(Ks + Vs, Ws, Nboot=100, f=adl.rw_mean)
        for Ks,Vs,Ws in zip(Ks, Vs, gfmc_Ws)])

ave_Ks = np.array([al.bootstrap(Ks, Ws, Nboot=100, f=adl.rw_mean)
        for Ks,Vs,Ws in zip(Ks, Vs, gfmc_Ws)])
ave_Vs = np.array([al.bootstrap(Vs, Ws, Nboot=100, f=adl.rw_mean)
        for Ks,Vs,Ws in zip(Ks, Vs, gfmc_Ws)])

print("first walker")
print(gfmc_Rs.shape)
print("R = ",gfmc_Rs[0][0])
x = gfmc_Rs[0][:,:,0]
y = gfmc_Rs[0][:,:,1]
z = gfmc_Rs[0][:,:,2]
r_n = np.sqrt(x**2 + y**2 + z**2)
t_n = np.arctan2(np.sqrt(x**2 + y**2), z)
p_n = np.arctan2(y, x)
print("r = ",r_n[0,0])
print("theta = ",t_n[0,0])
print("phi = ",p_n[0,0])
print("psi(R) = ",f_R(gfmc_Rs[0])[0])
print("K(R) = ",Ks[0,0])
print("V(R) = ",Vs[0,0])
print("H(R) = ",Ks[0,0]+Vs[0,0])

print("\n", Ks.shape)

print("\nsecond walker")
print("R = ",gfmc_Rs[0][1])
x = gfmc_Rs[0][:,:,0]
y = gfmc_Rs[0][:,:,1]
z = gfmc_Rs[0][:,:,2]
r_n = np.sqrt(x**2 + y**2 + z**2)
t_n = np.arctan2(np.sqrt(x**2 + y**2), z)
p_n = np.arctan2(y, x)
print("r = ",r_n[0,1])
print("theta = ",t_n[0,1])
print("phi = ",p_n[0,1])
print("psi(R) = ",f_R(gfmc_Rs[0])[1])
print("K(R) = ",Ks[0,1])
print("V(R) = ",Vs[0,1])
print("H(R) = ",Ks[0,0]+Vs[0,0])

print("H=",Hs,"\n\n")
print("K=",ave_Ks,"\n\n")
print("V=",ave_Vs,"\n\n")

#with h5py.File('Hammys_'+"nCoord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+'.hdf5', 'w') as f:
#    dset = f.create_dataset("default", data=Hs)

with h5py.File('Hammys_'+"nCoord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+'.hdf5', 'w') as f:
    dset = f.create_dataset("default", data=Ks+Vs)


with h5py.File('Hammys_'+"nCoord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+'.hdf5', 'r') as f:
    data = f['default']
    print(data)

# plot H
fig, ax = plt.subplots(1,1, figsize=(4,3))
al.add_errorbar(np.transpose(Hs/(VB**2)), ax=ax, xs=xs, color='xkcd:forest green', label=r'$\left< H \right>$', marker='o')
if N_coord == 2:
    ax.set_ylim(-.26, -.24)
elif N_coord == 3:
    ax.set_ylim(-1.1, -1.05)
elif N_coord == 4:
    ax.set_ylim(-2.5, -3.5)
elif N_coord == 5:
    ax.set_ylim(-5, -6)
elif N_coord == 6:
    ax.set_ylim(-9, -11)
elif N_coord == 7:
    ax.set_ylim(-15, -18)
ax.legend()

plt.show()
