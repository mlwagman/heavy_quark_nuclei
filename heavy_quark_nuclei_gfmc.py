### Eval script for deuteron GFMC deformation.

import argparse
import analysis as al
import getpass
import matplotlib.pyplot as plt
import numpy as onp
import scipy
import scipy.interpolate
import scipy.integrate
import scipy.special
import pickle
import paper_plt
import tqdm.auto as tqdm
import afdmc_lib as adl
import os
import pickle
from afdmc_lib import NI,NS,mp_Mev,fm_Mev
import jax
import jax.numpy as np
import sys
from itertools import repeat
import time
import h5py
import math
import mpmath
from functools import partial

from itertools import permutations
import torch
import torch.nn as nn

onp.random.seed(0)

paper_plt.load_latex_config()

parser = argparse.ArgumentParser()
parser.add_argument('--n_walkers', type=int, default=1000)
parser.add_argument('--dtau_iMev', type=float, required=True)
parser.add_argument('--n_step', type=int, required=True)
parser.add_argument('--n_skip', type=int, default=20)
parser.add_argument('--resampling', type=int, default=None)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--mufac', type=float, default=1.0)
parser.add_argument('--Nc', type=int, default=3)
parser.add_argument('--N_coord', type=int, default=2)
parser.add_argument('--nf', type=int, default=5)
parser.add_argument('--OLO', type=str, default="LO")
parser.add_argument('--spoila', type=int, default=1)
parser.add_argument('--spoilf', type=str, default="hwf")
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--input_Rs_database', type=str, default="")
parser.add_argument('--log_mu_r', type=float, default=1)
globals().update(vars(parser.parse_args()))

#######################################################################################

CF = (Nc**2 - 1)/(2*Nc)
VB = alpha*CF
if N_coord > 2:
    VB = alpha*CF/(Nc-1)
SingC3 = -(Nc+1)/8
cutoff = 1;
a0 = spoila*2/VB;
#a0=4.514

a_cutoff = 1.0

#VB=.1
#print(VB)
#quit()
# imaginary time points for GFMC evolution
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)

beta0 = 11/3*Nc - 2/3*nf
beta1 = 34/3*Nc**2 - 20/3*Nc*nf/2 - 2*CF*nf
beta2 = 2857/54*Nc**3 + CF**2*nf-205/9*Nc*CF*nf/2-1415/27*Nc**2*nf/2+44/9*CF*(nf/2)**2+158/27*Nc*(nf/2)**2
aa1 = 31/9*Nc-10/9*nf
zeta3 = scipy.special.zeta(3)
zeta5 = scipy.special.zeta(5)
zeta51 = 1/2 + 1/3 + 1/7 + 1/51 + 1/4284
zeta6 = scipy.special.zeta(6)
aa2 = ( 4343/162 + 6*np.pi**2 - np.pi**4/4 + 22/3*zeta3 )*Nc**2 - ( 1798/81 + 56/3*zeta3 )*Nc*nf/2 - ( 55/3 - 16*zeta3  )*CF*nf/2 + (10/9*nf)**2
dFF = (18-Nc**2+Nc**4)/(96*Nc**2)
dFA = Nc*(Nc**2+6)/48
alpha4 = float(mpmath.polylog(4,1/2))*0+(-np.log(2))**4/(4*3*2*1)
ss6 = zeta51+zeta6
aa30 = dFA*( np.pi**2*( 7432/9-4736*alpha4+np.log(2)*(14752/3-3472*zeta3)-6616*zeta3/3)  +  np.pi**4*(-156+560*np.log(2)/3+496*np.log(2)**2/3)+1511*np.pi**6/45)  + Nc**3*(385645/2916 + np.pi**2*( -953/54 +584/3*alpha4 +175/2*zeta3 + np.log(2)*(-922/9+217*zeta3/3) ) +584*zeta3/3 + np.pi**4*( 1349/270-20*np.log(2)/9-40*np.log(2)**2/9 ) -1927/6*zeta5 -143/2*zeta3**2-4621/3024*np.pi**6+144*ss6  )
aa31 = dFF*( np.pi**2*(1264/9-976*zeta3/3+np.log(2)*(64+672*zeta3)) + np.pi**4*(-184/3+32/3*np.log(2)-32*np.log(2)**2) +10/3*np.pi**6 ) + CF**2/2*(286/9+296/3*zeta3-160*zeta5)+Nc*CF/2*(-71281/162+264*zeta3+80*zeta5)+Nc**2/2*(-58747/486+np.pi**2*(17/27-32*alpha4+np.log(2)*(-4/3-14*zeta3)-19/3*zeta3)-356*zeta3+np.pi**4*(-157/54-5*np.log(2)/9+np.log(2)**2)+1091*zeta5/6+57/2*zeta3**2+761*np.pi**6/2520-48*ss6)
aa32 = Nc/4*(12541/243+368/3*zeta3+64*np.pi**4/135)+CF/4*(14002/81-416*zeta3/3)
aa33 = -(20/9)**3*1/8
aa3 = aa30+aa31*nf+aa32*nf**2+aa33*nf**3

L = log_mu_r
VB_LO = VB

VB_NLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L))

VB_NNLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L) + (alpha/(4*np.pi))**2*( beta0**2*(4*L**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*L + aa2 ) )
if N_coord > 2:
   VB_NNLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L) + (alpha/(4*np.pi))**2*( beta0**2*(4*L**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*L + aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2)  ) )


if OLO == "LO":
    a0=spoila*2/VB_LO
elif OLO == "NLO":
    a0=spoila*2/VB_NLO
elif OLO == "mNLO":
    a0=spoila*2/VB_NLO
elif OLO == "NNLO":
    a0=spoila*2/VB_NNLO

@partial(jax.jit)
def V3(r1, r2):
   R = lambda x, y: x*r1 - y*r2
   r1_norm = adl.norm_3vec(r1)
   r1_hat = r1 / r1_norm[...,np.newaxis]
   r2_norm = adl.norm_3vec(r1)
   r2_hat = r2 / r2_norm[...,np.newaxis]
   r1_hat_dot_r2_hat = np.sum(r1_hat*r2_hat, axis=-1)
   R_norm = lambda x, y: adl.norm_3vec(R(x,y))
   R_hat = lambda x, y: R(x,y) / R_norm(x,y)[...,np.newaxis]
   r1_hat_r2_hat_dot_R_R = lambda x, y: np.sum(r1_hat*R_hat(x,y), axis=-1)*np.sum(r2_hat*R_hat(x,y), axis=-1)
   A = lambda x, y: r1_norm * np.sqrt(x*(1-x)) + r2_norm*np.sqrt(y*(1-y))

   V3_integrand = lambda x, y: 16*np.pi*( np.arctan2(R_norm(x,y),A(x,y))*r1_hat_dot_r2_hat*1/R_norm(x,y)*(-1*A(x,y)**2/R_norm(x,y)**2+1) + r1_hat_dot_r2_hat*A(x,y)/R_norm(x,y)**2
           + np.arctan2(R_norm(x,y),A(x,y))*r1_hat_r2_hat_dot_R_R(x,y)*1/R_norm(x,y)*(3*A(x,y)**2/R_norm(x,y)**2+1) - 3*r1_hat_r2_hat_dot_R_R(x,y)*A(x,y)/R_norm(x,y)**2)

   int_points = 100
   dx = 1/int_points
   x_grid = np.arange(dx, stop=1, step=dx)
   y_grid = np.arange(0, stop=1, step=dx)
   #V3_grid = np.transpose(np.array([[V3_integrand(x,y) for y in y_grid] for x in x_grid]), (2,0,1))

   y_vmap_f = jax.vmap(V3_integrand, (None, 0))
   xy_vmap_f = jax.vmap(y_vmap_f, (0, None))
   V3_grid = np.transpose( xy_vmap_f(x_grid, y_grid), (2,0,1))

   V3_integral = np.trapz( np.trapz(V3_grid, dx=1/int_points), dx=1/int_points)

   return V3_integral

Rprime = lambda R: adl.norm_3vec(R)*np.exp(np.euler_gamma)*mu
# build Coulomb potential
AV_Coulomb = {}
B3_Coulomb = {}
if OLO == "LO":
    @partial(jax.jit)
    def potential_fun(R):
	    return -1*VB/adl.norm_3vec(R)
elif OLO == "NLO":
    @partial(jax.jit)
    def potential_fun(R):
        return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
elif OLO == "NNLO":
    if N_coord > 2:
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2) ) )
    else:
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 ) )
    B3_Coulomb['O1'] = lambda Rij, Rjk, Rik: SingC3*2*alpha*(alpha/(4*np.pi))**2*(V3(Rij, Rjk) + V3(Rjk, Rik) + V3(Rik, Rij))
elif OLO == "N3LO":
    if N_coord > 2:
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2) ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*np.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( np.log(Rprime(R))**3 + np.pi**4/4*np.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(np.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*np.log(Rprime(R)) )
    else:
        @partial(jax.jit)
        def potential_fun(R):
            -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*np.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( np.log(Rprime(R))**3 + np.pi**4/4*np.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(np.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*np.log(Rprime(R)) )
    B3_Coulomb['O1'] = lambda Rij, Rjk, Rik: SingC3*2*alpha*(alpha/(4*np.pi))**2*(V3(Rij, Rjk) + V3(Rjk, Rik) + V3(Rik, Rij))
elif OLO == "mNLO":
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1)) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
elif OLO == "mNNLO":
    if N_coord > 2:
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2) ) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
    else:
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 ) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
    B3_Coulomb['O1'] = lambda Rij, Rjk, Rik: SingC3*2*alpha*(alpha/(4*np.pi))**2*(V3(Rij, Rjk) + V3(Rjk, Rik) + V3(Rik, Rij))
elif OLO == "mN3LO":
    if N_coord > 2:
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2) ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*np.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( np.log(Rprime(R))**3 + np.pi**4/4*np.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(np.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*np.log(Rprime(R)) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
    else:
        @partial(jax.jit)
        def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*np.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))+ aa2 ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*np.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( np.log(Rprime(R))**3 + np.pi**4/4*np.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(np.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*np.log(Rprime(R)) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
else:
	print("order not supported")
	throw(0)

AV_Coulomb['O1'] = potential_fun

Coulomb_potential = adl.make_pairwise_potential(AV_Coulomb, B3_Coulomb)

# build Coulomb ground-state trial wavefunction
@partial(jax.jit)
def f_R(Rs):
    #N_walkers = Rs.shape[0]
    #assert Rs.shape == (N_walkers, N_coord, 3)
    psi = 1
    for i in range(N_coord):
       	for j in range(N_coord):
            if i!=j and j>=i:
                ri = Rs[...,i,:]
                rj = Rs[...,j,:]
                rij_norm = adl.norm_3vec(ri - rj)
                psi = psi*np.exp(-rij_norm/a0)
    return psi


@partial(jax.jit)
def laplacian_f_R(Rs):
    #N_walkers = Rs.shape[0]
    #assert Rs.shape == (N_walkers, N_coord, 3)
    nabla_psi_tot = 0
    # terms where laplacian hits one piece of wvfn
    # laplacian hits r_kl
    for k in range(N_coord):
        for l in range(N_coord):
            if k!=l and l>=k:
                # wvfn includes r_ij
                nabla_psi = 1
                for i in range(N_coord):
                    for j in range(N_coord):
                        if i!=j and j>=i:
                            ri = Rs[...,i,:]
                            rj = Rs[...,j,:]
                            rij_norm = adl.norm_3vec(ri - rj)
                            # nabla_k^2 r_kl = nabla_l^2 r_kl
                            # factor of two included to account for both terms appearing in laplacian
                            if k == i and l == j:
                                nabla_psi = nabla_psi * (2/a0**2 - 4/(a0*rij_norm)) * np.exp(-rij_norm/a0)
                            else:
                                nabla_psi = nabla_psi * np.exp(-rij_norm/a0)
                nabla_psi_tot += nabla_psi
    # terms where gradients hit separate pieces of wvfn
    # laplacian involves particle a
    for a in range(N_coord):
        # first gradient involves r_kl
        for k in range(N_coord):
            for l in range(N_coord):
                if k!=l and l>=k and (a==k or a==l):
                    # second gradient involves r_mn
                    for m in range(N_coord):
                        for n in range(N_coord):
                            if m!=n and n>=m and (m!=k or n!=l) and (a==m or a==n):
                                # sum over the 3-d components of gradient
                                for x in range(3):
                                    # wvfn involves r_ij
                                    nabla_psi = 1
                                    for i in range(N_coord):
                                        for j in range(N_coord):
                                            if i!=j and j>=i:
                                                ri = Rs[...,i,:]
                                                rj = Rs[...,j,:]
                                                rij_norm = adl.norm_3vec(ri - rj)
                                                rsign = 0
                                                # grad_a r_ij = rsign * (ri - rj)
                                                if a == i:
                                                    rsign = 1
                                                elif a == j:
                                                    rsign = -1
                                                if (k == i and l == j) or (m == i and n == j):
                                                    nabla_psi = rsign * nabla_psi * (ri[:,x] - rj[:,x])/(a0*rij_norm) * np.exp(-rij_norm/a0)
                                                else:
                                                    nabla_psi = nabla_psi * np.exp(-rij_norm/a0)
                                    nabla_psi_tot += nabla_psi
    return nabla_psi_tot


# Metropolis
if input_Rs_database == "":
    R0 = onp.random.normal(size=(N_coord,3))
    # set center of mass position to 0
    R0 -= onp.mean(R0, axis=1, keepdims=True)
    samples = adl.metropolis(R0, f_R, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=2*a0/N_coord**2)
    Rs_metropolis = np.array([R for R,_ in samples])
else:
    f = h5py.File(input_Rs_database, 'r')
    Rs_metropolis = f["Rs"][-1]
print(Rs_metropolis)
# build trial wavefunction
S_av4p_metropolis = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
print("built Metropolis wavefunction ensemble")
# trial wavefunction spin-flavor structure is |up,u> x |up,u> x ... x |up,u>
spin_slice = (slice(0,None),) + (0,)*2*N_coord
S_av4p_metropolis[spin_slice] = 1
print("spin-flavor wavefunction shape = ", S_av4p_metropolis.shape)

#print("old ", f_R_old(Rs_metropolis))
#print("new ", f_R(Rs_metropolis))

#print("old laplacian ", laplacian_f_R_old(Rs_metropolis))
#print("new laplacian ", laplacian_f_R(Rs_metropolis))

# trivial contour deformation
deform_f = lambda x, params: x
params = (np.zeros((n_step+1)),)

print('Running GFMC evolution:')
rand_draws = onp.random.random(size=(n_step, Rs_metropolis.shape[0]))
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
#for R in tqdm.tqdm(gfmc_Rs):
for count, R in enumerate(gfmc_Rs):
    print('Calculating Laplacian for step ', count)
    K_time = time.time()
    #Ks.append(-1/2*laplacian_f_R(R) / f_R(R) / adl.mp_Mev)
    Ks.append(-1/2*laplacian_f_R(R) / f_R(R) / 1)
    print(f"calculated kinetic in {time.time() - K_time} sec")
Ks = np.array(Ks)

#Ks *= fm_Mev**2

#Vs = np.array([
#    sum([
#        AV_Coulomb[name](dRs) * adl.compute_O(adl.two_body_ops[name](dRs), S, S_av4p_metropolis)
#        for name in AV_Coulomb
#    ])
#    for dRs, S in zip(map(adl.to_relative, gfmc_Rs), gfmc_Ss)])

Vs = []
for count, R in enumerate(gfmc_Rs):
    print('Calculating potential for step ', count)
    V_time = time.time()
    VSI,_ = Coulomb_potential(R)
    V_ind = (slice(0,None),) + (0,)*NS*NI*N_coord
    print(f"calculated potential in {time.time() - V_time} sec")
    Vs.append(VSI[V_ind])

Vs = np.array(Vs)
print(Vs.shape)

Hs = np.array([al.bootstrap(K + V, W, Nboot=100, f=adl.rw_mean)
        for K,V,W in zip(Ks, Vs, gfmc_Ws)])

ave_Ks = np.array([al.bootstrap(K, W, Nboot=100, f=adl.rw_mean)
        for K,V,W in zip(Ks, Vs, gfmc_Ws)])
ave_Vs = np.array([al.bootstrap(V, W, Nboot=100, f=adl.rw_mean)
        for K,V,W in zip(Ks, Vs, gfmc_Ws)])

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
print("H(R) = ",Ks[0,1]+Vs[0,1])

print("H=",Hs,"\n\n")
print("K=",ave_Ks,"\n\n")
print("V=",ave_Vs,"\n\n")

tag = str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_Nf"+str(nf) + "_alpha"+str(alpha) + "_spoila"+str(spoila) + "_spoilf"+str(spoilf) + "_log_mu_r"+str(log_mu_r)

with h5py.File(outdir+'Hammys_'+tag+'.h5', 'w') as f:
    dset = f.create_dataset("Hammys", data=Ks+Vs)
    dset = f.create_dataset("Ws", data=gfmc_Ws)

with h5py.File(outdir+'Hammys_'+tag+'.h5', 'r') as f:
    data = f['Hammys']
    print(data)

#with h5py.File(outdir+'Rs_'+tag+'.h5', 'w') as f:
#    dset = f.create_dataset("Rs", data=gfmc_Rs)
#    dset = f.create_dataset("Ws", data=gfmc_Ws)
#
#with h5py.File(outdir+'Rs_'+tag+'.h5', 'r') as f:
#    data = f['Rs']
#    print(data)
#
## plot H
#fig, ax = plt.subplots(1,1, figsize=(4,3))
#al.add_errorbar(np.transpose(Hs/(VB**2)), ax=ax, xs=xs, color='xkcd:forest green', label=r'$\left< H \right>$', marker='o')
#if N_coord == 2:
#    ax.set_ylim(-.26, -.24)
#elif N_coord == 3:
#    ax.set_ylim(-1.1, -1.05)
#elif N_coord == 4:
#    ax.set_ylim(-2.5, -3.5)
#elif N_coord == 5:
#    ax.set_ylim(-5, -6)
#elif N_coord == 6:
#    ax.set_ylim(-9, -11)
#elif N_coord == 7:
#    ax.set_ylim(-15, -18)
#ax.legend()
#
#plt.savefig(outdir+'Hammy_gfmc_plot_'+tag+'.pdf')
