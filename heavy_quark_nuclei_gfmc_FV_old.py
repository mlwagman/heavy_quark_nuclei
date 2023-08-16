### Eval script for deuteron GFMC deformation.

import argparse
import analysis as al
import getpass
import matplotlib.pyplot as plt
import numpy as onp
import scipy
import scipy.interpolate
import scipy.integrate
import jax.scipy
import jax.scipy.special
import pickle
import paper_plt
import tqdm.auto as tqdm
import afdmc_lib_col as adl
import os
import pickle
from afdmc_lib_col_old import NI,NS,mp_Mev,fm_Mev
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
parser.add_argument('--n_skip', type=int, default=100)
parser.add_argument('--resampling', type=int, default=None)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--mufac', type=float, default=1.0)
parser.add_argument('--Nc', type=int, default=3)
parser.add_argument('--N_coord', type=int, default=3)
parser.add_argument('--nf', type=int, default=5)
parser.add_argument('--OLO', type=str, default="LO")
parser.add_argument('--spoila', type=float, default=1)
parser.add_argument('--spoilf', type=str, default="hwf")
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--input_Rs_database', type=str, default="")
parser.add_argument('--log_mu_r', type=float, default=1)
parser.add_argument('--cutoff', type=float, default=0.0)
parser.add_argument('--L', type=float, default=0.0)
parser.add_argument('--Lcut', type=int, default=5)
parser.add_argument('--spoilS', type=float, default=1)
parser.add_argument('--wavefunction', type=str, default="compact")
parser.add_argument('--potential', type=str, default="full")
parser.add_argument('--spoilaket', type=float, default=1)
parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
globals().update(vars(parser.parse_args()))

#######################################################################################

volume = "infinite"
if L > 1e-2:
    volume = "finite"

masses = onp.ones(N_coord)
if N_coord == 2:
    masses = [1,-1]
if N_coord == 4:
    masses = [1,-1,1,-1]

print("masses = ", masses)

if wavefunction == "asymmetric":
    bra_wavefunction = "product"
    ket_wavefunction = "compact"
else:
    bra_wavefunction = wavefunction
    ket_wavefunction = wavefunction

#assert Nc == NI

CF = (Nc**2 - 1)/(2*Nc)
VB = alpha*CF/(Nc-1)
SingC3 = -(Nc+1)/8

#a0=4.514

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
VB_LO = VB

VB_NLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*log_mu_r))

if OLO == "LO":
    a0=spoila*2/VB_LO
elif OLO == "NLO":
    a0=spoila*2/VB_NLO

if N_coord == 2 or N_coord == 4:
    a0 /= Nc-1


ket_a0 = a0
if wavefunction == "asymmetric":
    ket_a0 = a0*spoilaket

print("a0 = ", a0)
print("ket_a0 = ", ket_a0)

Rprime = lambda R: adl.norm_3vec(R)*np.exp(np.euler_gamma)*mu
# build Coulomb potential
AV_Coulomb = {}
B3_Coulomb = {}

# Generate the nn array representing 3D shifts in a cubic grid
pp = np.array([np.array([i, j, k]) for i in range(-Lcut, Lcut+1) for j in range(-Lcut, Lcut+1) for k in range(-Lcut, Lcut+1)])

print(pp.shape)

print("zero mode is ", pp[Lcut*(2*Lcut+1)*(2*Lcut+1)+Lcut*(2*Lcut+1)+Lcut])

nn = np.delete(pp, Lcut*(2*Lcut+1)*(2*Lcut+1)+Lcut*(2*Lcut+1)+Lcut, axis=0)

print(nn.shape)
print(pp.shape)

#from jax.config import config
#config.update('jax_disable_jit', True)

@partial(jax.jit)
def FV_Coulomb_with_zero_mode(R, L, nn):
    Rdotp = np.einsum('bi,ki->bk', R, pp)
    RdotR = np.sum( R*R, axis=1 )
    pdotp = np.sum( pp*pp, axis=1 )
    pmRL = np.sqrt( Rdotp*(-2.0*L) + pdotp[(np.newaxis,slice(None))]*L*L + RdotR[(slice(None),np.newaxis)] )
    sums = np.sum( 1.0/pmRL, axis=1 )
    #assert( (np.abs(sums/(np.pi*L) - FV_Coulomb_slow(R,L,nn)) < 1e-6).all() )
    #print(sums/(np.pi*L))
    #print(FV_Coulomb_slow(R,L,nn))
    return sums

@partial(jax.jit)
def FV_Coulomb(R, L, nn):
    sums = np.zeros(n_walkers)
    sums += -1
    Rdotn = np.einsum('bi,ki->bk', R, nn)
    n_mag_sq = np.sum( nn*nn, axis=1 )
    sums += np.sum( np.exp((2*np.pi*1j/L)*Rdotn)*np.exp(-n_mag_sq)/n_mag_sq, axis=1 )
    Rdotp = np.einsum('bi,ki->bk', R, pp)
    RdotR = np.sum( R*R, axis=1 )
    pdotp = np.sum( pp*pp, axis=1 )
    pmRL = np.sqrt( Rdotp*(-2.0/L) + pdotp[(np.newaxis,slice(None))] + (1.0/L)**2*RdotR[(slice(None),np.newaxis)] )
    sums += np.sum( np.pi/pmRL*(1-jax.scipy.special.erf(np.pi*pmRL)), axis=1 )
    #assert( (np.abs(sums/(np.pi*L) - FV_Coulomb_slow(R,L,nn)) < 1e-6).all() )
    #print(sums/(np.pi*L))
    #print(FV_Coulomb_slow(R,L,nn))
    return sums/(np.pi*L)

def FV_Coulomb_slow(R, L, nn):
    sums = np.zeros(n_walkers)
    sums += -1
    for i in range(len(nn)):
        n=nn[i]
        n_mag = adl.norm_3vec(n)
        sums += np.exp(-n_mag**2)/n_mag**2*np.exp(2*np.pi*1j*np.sum(n*R,axis=1)/L)
        #print(n)
        #print(R)
        #print(n*R)
        #print(np.sum(n*R,axis=1))
        #sums += 1/n_mag**2*np.exp(2*np.pi*1j*np.sum(n*R,axis=1)/L)
        #print("n = ", n)
        #print(sums)
    for i in range(len(pp)):
        n=pp[i]
        n_mag = adl.norm_3vec(n)
        sums += np.pi/adl.norm_3vec(n - R/L)*(1-jax.scipy.special.erf(np.pi*adl.norm_3vec(n - R/L)))
        #print("n = ", n)
        #print(sums)
    #print(sums/(np.pi*L))
    #print(1/adl.norm_3vec(R))
    #exit(1)
    return sums/(np.pi*L)

if OLO == "LO":
    @partial(jax.jit)
    def potential_fun(R):
            return -1*VB/adl.norm_3vec(R)
    @partial(jax.jit)
    def symmetric_potential_fun(R):
            return spoilS*(Nc - 1)/(Nc + 1)*VB/adl.norm_3vec(R)
    @partial(jax.jit)
    def singlet_potential_fun(R):
            return -1*(Nc - 1)*VB/adl.norm_3vec(R)
    @partial(jax.jit)
    def octet_potential_fun(R):
            return spoilS*(Nc - 1)/CF/(2*Nc)*VB/adl.norm_3vec(R)
            #return -1*(CF-Nc/2)*(Nc - 1)/CF*VB/adl.norm_3vec(R)
    @partial(jax.jit)
    def potential_fun_sum(R):
            return -1*VB*FV_Coulomb(R, L, nn)
    @partial(jax.jit)
    def symmetric_potential_fun_sum(R):
            return spoilS*(Nc - 1)/(Nc + 1)*VB*FV_Coulomb(R, L, nn)
    @partial(jax.jit)
    def singlet_potential_fun_sum(R):
            return -1*(Nc - 1)*VB*FV_Coulomb(R, L, nn)
    @partial(jax.jit)
    def octet_potential_fun_sum(R):
            return spoilS*(Nc - 1)/CF/(2*Nc)*VB*FV_Coulomb(R, L, nn)
elif OLO == "NLO":
    @partial(jax.jit)
    def potential_fun(R):
        return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    @partial(jax.jit)
    def potential_fun_sum(R):
            return calculate_sum(potential_fun, R, L, nn)
    def symmetric_potential_fun(R):
        return (Nc - 1)/(Nc + 1)*VB*spoilS/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    def symmetric_potential_fun_sum(R):
            return calculate_sum(symmetric_potential_fun, R, L, nn)
    @partial(jax.jit)
    def singlet_potential_fun(R):
        return -1*(Nc - 1)*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    @partial(jax.jit)
    def singlet_potential_fun_sum(R):
            return calculate_sum(potential_fun, R, L, nn)
    def octet_potential_fun(R):
        return spoilS*(Nc - 1)/CF/(2*Nc)*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    def octet_potential_fun_sum(R):
            return calculate_sum(symmetric_potential_fun, R, L, nn)
else:
        print("order not supported")
        throw(0)

def trivial_fun(R):
    return 0*adl.norm_3vec(R)+1

# MODE 1
#AV_Coulomb['O1'] = potential_fun
#AV_Coulomb['O1'] = symmetric_potential_fun

# MODE 2

print("volume = ", volume)

if volume == "finite":
    AV_Coulomb['OA'] = potential_fun_sum
    AV_Coulomb['OS'] = symmetric_potential_fun_sum
    AV_Coulomb['OSing'] = singlet_potential_fun_sum
    AV_Coulomb['OO'] = octet_potential_fun_sum
else:
    #AV_Coulomb['OA'] = trivial_fun
    #AV_Coulomb['OS'] = trivial_fun
    #AV_Coulomb['OSing'] = trivial_fun
    #AV_Coulomb['OO'] = trivial_fun
    AV_Coulomb['OA'] = potential_fun
    AV_Coulomb['OS'] = symmetric_potential_fun
    AV_Coulomb['OSing'] = singlet_potential_fun
    AV_Coulomb['OO'] = octet_potential_fun
    #AV_Coulomb['O1'] = potential_fun


print("AV_Coulomb = ", AV_Coulomb)

#AV_Coulomb['OSing'] = trivial_fun
#AV_Coulomb['OO'] = trivial_fun
#AV_Coulomb['O1'] = trivial_fun

if potential == "product":
    Coulomb_potential = adl.make_pairwise_product_potential(AV_Coulomb, B3_Coulomb, masses)
else:
    Coulomb_potential = adl.make_pairwise_potential(AV_Coulomb, B3_Coulomb, masses)





# build Coulomb ground-state trial wavefunction
@partial(jax.jit)
def f_R_slow(Rs, wavefunction=wavefunction):
    #N_walkers = Rs.shape[0]
    #assert Rs.shape == (N_walkers, N_coord, 3)
    psi = 1
    for i in range(N_coord):
       	for j in range(N_coord):
            if i!=j and j>=i:
                if wavefunction == "product":
                    baryon_0 = 1
                    if i < N_coord/2:
                        baryon_0 = 0
                    baryon_1 = 1
                    if j < N_coord/2:
                        baryon_1 = 0
                    if baryon_0 != baryon_1:
                        continue
                ri = Rs[...,i,:]
                rj = Rs[...,j,:]
                rij_norm = adl.norm_3vec(ri - rj)
                psi = psi*np.exp(-rij_norm/a0)
    return psi

pairs = np.array([np.array([i, j]) for i in range(0,N_coord) for j in range(0, i)])
print("pairs = ", pairs)

product_pairs = []
for i in range(N_coord):
    for j in range(N_coord):
        if i!=j and j>=i:
            baryon_0 = 1
            if i < N_coord/2:
                baryon_0 = 0
            baryon_1 = 1
            if j < N_coord/2:
                baryon_1 = 0
            if baryon_0 != baryon_1:
                continue
            product_pairs.append(np.array([i,j]))
product_pairs = np.array(product_pairs)
print("product pairs = ", product_pairs)

@partial(jax.jit, static_argnums=(1,))
def f_R(Rs, wavefunction=bra_wavefunction, a0=a0):

    def r_norm(pair):
        [i,j] = pair
        rdiff = Rs[...,i,:] - Rs[...,j,:]
        rij_norm = np.sqrt( np.sum(rdiff*rdiff, axis=-1) )
        return rij_norm

    if wavefunction == "product":
        r_sum = np.sum( jax.lax.map(r_norm, product_pairs), axis=0 )
    else:
        r_sum = np.sum( jax.lax.map(r_norm, pairs), axis=0 )
    return np.exp(-r_sum/a0)

def f_R_sq(Rs):
    return np.abs( f_R(Rs) )**2

def f_R_braket(Rs):
    return np.abs( f_R(Rs, wavefunction=bra_wavefunction) * f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0) )

def f_R_braket_tempered(Rs, fac):
    return np.abs( f_R(Rs, wavefunction=bra_wavefunction) * f_R(Rs, wavefunction=ket_wavefunction, a0=fac*ket_a0) )

def f_R_braket_phase(Rs):
    prod = f_R(Rs, wavefunction=bra_wavefunction) * f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0)
    return prod / np.abs( prod )

@partial(jax.jit)
def laplacian_f_R(Rs, wavefunction=bra_wavefunction):
    #N_walkers = Rs.shape[0]
    #assert Rs.shape == (N_walkers, N_coord, 3)
    nabla_psi_tot = 0
    # terms where laplacian hits one piece of wvfn
    # laplacian hits r_kl
    for k in range(N_coord):
        for l in range(N_coord):
            if k!=l and l>=k:
                if wavefunction == "product":
                    baryon_0 = 1
                    if k < N_coord/2:
                        baryon_0 = 0
                    baryon_1 = 1
                    if l < N_coord/2:
                        baryon_1 = 0
                    if baryon_0 != baryon_1:
                        continue
                # wvfn includes r_ij
                nabla_psi = 1
                for i in range(N_coord):
                    for j in range(N_coord):
                        if i!=j and j>=i:
                            if wavefunction == "product":
                                baryon_0 = 1
                                if i < N_coord/2:
                                    baryon_0 = 0
                                baryon_1 = 1
                                if j < N_coord/2:
                                    baryon_1 = 0
                                if baryon_0 != baryon_1:
                                    continue
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
                    if wavefunction == "product":
                        baryon_0 = 1
                        if k < N_coord/2:
                            baryon_0 = 0
                        baryon_1 = 1
                        if l < N_coord/2:
                            baryon_1 = 0
                        if baryon_0 != baryon_1:
                            continue
                    # second gradient involves r_mn
                    for m in range(N_coord):
                        for n in range(N_coord):
                            if m!=n and n>=m and (m!=k or n!=l) and (a==m or a==n):
                                if wavefunction == "product":
                                    baryon_0 = 1
                                    if m < N_coord/2:
                                        baryon_0 = 0
                                    baryon_1 = 1
                                    if n < N_coord/2:
                                        baryon_1 = 0
                                    if baryon_0 != baryon_1:
                                        continue
                                # sum over the 3-d components of gradient
                                for x in range(3):
                                    # wvfn involves r_ij
                                    nabla_psi = 1
                                    for i in range(N_coord):
                                        for j in range(N_coord):
                                            if i!=j and j>=i:
                                                if wavefunction == "product":
                                                    baryon_0 = 1
                                                    if i < N_coord/2:
                                                        baryon_0 = 0
                                                    baryon_1 = 1
                                                    if j < N_coord/2:
                                                        baryon_1 = 0
                                                    if baryon_0 != baryon_1:
                                                        continue
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
    met_time = time.time()
    R0 = onp.random.normal(size=(N_coord,3))
    # set center of mass position to 0
    #R0 -= onp.mean(R0, axis=1, keepdims=True)
    R0 -= onp.mean(R0, axis=0, keepdims=True)
    print("R0 = ", R0)
    samples = adl.metropolis(R0, f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=4*2*a0/N_coord**2)
    #samples = adl.metropolis(R0, f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=2*a0/N_coord**2)

#    if N_coord == 2:
#        samples = adl.direct_sample_quarkonium(n_walkers, f_R_braket, a0=a0)
#    elif N_coord == 4:
#        samples1 = adl.direct_sample_quarkonium(n_walkers, f_R_braket, a0=a0)
#        samples2 = adl.direct_sample_quarkonium(n_walkers, f_R_braket, a0=a0)
#        samples = []
#        for n in range(n_walkers):
#            this_R = onp.concatenate((samples1[n][0], samples2[n][0]), axis=0)
#            this_W = f_R_braket(this_R)
#            samples.append((this_R, this_W))
#    else:
#        samples = adl.metropolis(R0, f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=2*a0/N_coord**2)
    fac_list = [1/2, 1.0, 2]
    streams = len(fac_list)
    R0_list = [ onp.random.normal(size=(N_coord,3)) for s in range(0,streams) ]
    for s in range(streams):
        R0_list[s] -= onp.mean(R0_list[s], axis=0, keepdims=True)
    print("R0 = ", R0_list[0])
    #samples = adl.parallel_tempered_metropolis(fac_list, R0_list, f_R_braket_tempered, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=2*a0/N_coord**2)
    #print(samples)
    print("first walker")
    print("R = ",samples[0])
    print(f"metropolis in {time.time() - met_time} sec")
    Rs_metropolis = np.array([R for R,_ in samples])
else:
    f = h5py.File(input_Rs_database, 'r')
    Rs_metropolis = f["Rs"][-1]
#print(Rs_metropolis)
# build trial wavefunction
S_av4p_metropolis = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
print("built Metropolis wavefunction ensemble")
# trial wavefunction spin-flavor structure is |up,u> x |up,u> x ... x |up,u>

def levi_civita(i, j, k):
    if i == j or j == k or i == k:
        return 0
    if (i,j,k) in [(0,1,2), (1,2,0), (2,0,1)]:
        return 1
    else:
        return -1

def kronecker_delta(i, j):
    return 1 if i == j else 0

print("spin-flavor wavefunction shape = ", S_av4p_metropolis.shape)

if N_coord == 3:
  for i in range(NI):
   for j in range(NI):
    for k in range(NI):
     if i != j and j != k and i != k:
      spin_slice = (slice(0, None),) + (i,0,j,0,k,0)
      S_av4p_metropolis[spin_slice] = levi_civita(i, j, k) / np.sqrt(6)

# symmetric
#S_av4p_metropolis = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
#spin_slice = (slice(0,None),) + (0,)*2*N_coord
#S_av4p_metropolis[spin_slice] = 1

if N_coord == 2:
  for i in range(NI):
   for j in range(NI):
        if i == j:
          spin_slice = (slice(0, None),) + (i,0,j,0)
          S_av4p_metropolis[spin_slice] = kronecker_delta(i, j)/np.sqrt(NI)

# adjoint
#S_av4p_metropolis = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
#spin_slice = (slice(0,None),) + (0,0,1,0)
#S_av4p_metropolis[spin_slice] = 1/np.sqrt(2)
#spin_slice = (slice(0,None),) + (1,0,0,0)
#S_av4p_metropolis[spin_slice] = 1/np.sqrt(2)

# adjoint again
#S_av4p_metropolis = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
#spin_slice = (slice(0,None),) + (0,0,0,0)
#S_av4p_metropolis[spin_slice] = 1/np.sqrt(6)
#spin_slice = (slice(0,None),) + (1,0,1,0)
#S_av4p_metropolis[spin_slice] = 1/np.sqrt(6)
#spin_slice = (slice(0,None),) + (2,0,2,0)
#S_av4p_metropolis[spin_slice] = -2/np.sqrt(6)

if N_coord == 4:
  for i in range(NI):
   for j in range(NI):
    for k in range(NI):
     for l in range(NI):
        if i == j and k == l:
          # up up up up up up
          spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,0)
          # up up up down down down
          #spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,1,m,1,n,1)
          S_av4p_metropolis[spin_slice] = kronecker_delta(i, j)*kronecker_delta(k,l)/3

if N_coord == 6:
  for i in range(NI):
   for j in range(NI):
    for k in range(NI):
     for l in range(NI):
      for m in range(NI):
       for n in range(NI):
        if i != j and j != k and i != k and l != m and m != n and n != l:
          # up up up up up up
          spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,0,m,0,n,0)
          # up up up down down down
          #spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,1,m,1,n,1)
          S_av4p_metropolis[spin_slice] = levi_civita(i, j, k)*levi_civita(l, m, n) / 6
          #spin_slice = (slice(0, None),) + (i,0,j,0,k,0,0,0,0,0,0,0)
          #S_av4p_metropolis[spin_slice] = levi_civita(i, j, k) / np.sqrt(6)




#print(S_av4p_metropolis)

print("spin-flavor wavefunction shape = ", S_av4p_metropolis.shape)
S_av4p_metropolis_norm = adl.inner(S_av4p_metropolis, S_av4p_metropolis)
assert (np.abs(S_av4p_metropolis_norm - 1.0) < 1e-6).all()
print("spin-flavor wavefunction normalization = ", S_av4p_metropolis_norm)

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

phase_Ws = f_R_braket_phase(gfmc_Rs)
gfmc_Ws *= phase_Ws

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
    Ks.append(-1/2*laplacian_f_R(R) / f_R(R, wavefunction=bra_wavefunction) / 1)
    print(f"calculated kinetic in {time.time() - K_time} sec")
Ks = np.array(Ks)

Vs = []
for count, R in enumerate(gfmc_Rs):
    print('Calculating potential for step ', count)
    V_time = time.time()
    S = gfmc_Ss[count]
    V_SI, V_SD = Coulomb_potential(R)
    #if N_coord == 6:
    #  print("V_SD has ", V_SD[0,0,0,1,0,2,0,0,0,1,0,2,0,0,0,1,0,2,0,0,0,1,0,2,0])
    #  print("V_SD has ", V_SD[0,0,0,1,0,2,0,0,0,1,0,2,0,0,0,1,0,2,0,0,0,2,0,1,0])
    #  print("V_SD has ", V_SD[0,0,0,1,0,2,0,0,0,2,0,1,0,0,0,1,0,2,0,0,0,2,0,1,0])
    V_SD_S = adl.batched_apply(V_SD, S)
    #if N_coord == 6:
    #  print("S(0,1,2,0,1,2) = ", S[0,0,0,1,0,2,0,0,0,1,0,2,0])
    #  print("V_SD_S(0,1,2,0,1,2) = ", V_SD_S[0,0,0,1,0,2,0,0,0,1,0,2,0])
    #  print("S(0,2,1,0,1,2) = ", S[0,0,0,2,0,1,0,0,0,1,0,2,0])
    #  print("V_SD_S(0,2,1,0,1,2) = ", V_SD_S[0,0,0,2,0,1,0,0,0,1,0,2,0])
    #  print("S(0,1,2,0,2,1) = ", S[0,0,0,1,0,2,0,0,0,2,0,1,0])
      #print("V_SD_S(0,1,2,0,2,1) = ", V_SD_S[0,0,0,1,0,2,0,0,0,2,0,1,0])
      #print("S(0,1,2,0,:,:) = ", S[0,0,0,1,0,2,0,0,0,:,0,:,0])
      #print("V_SD_S(0,1,2,0,:,:) = ", V_SD_S[0,0,0,1,0,2,0,0,0,:,0,:,0])
    #print("S_T shape is ", S_av4p_metropolis.shape)
    #print("S shape is ", S.shape)
    #print("V_SI shape is ", V_SI.shape)
    #print("V_SI_S shape is ", V_SI_S.shape)
    broadcast_SI = ((slice(None),) + (np.newaxis,)*N_coord*2)
    V_SI_S = adl.batched_apply(V_SI, S)
    print("V_SD shape is ", V_SD.shape)
    print("V_SD L2 norm is ", np.sqrt(np.mean(V_SD**2)))
    print("V_SD Linfinity norm is ", np.max(np.abs(V_SD)))
    #print("V_SD_S shape is ", V_SD_S.shape)
    #print("V_SD_S L2 norm is ", np.sqrt(np.mean(V_SD_S**2)))
    #print("V_SD_S Linfinity norm is ", np.max(np.abs(V_SD_S)))
    V_tot = adl.inner(S_av4p_metropolis, V_SD_S + V_SI_S) / adl.inner(S_av4p_metropolis, S)
    #print("V_tot shape is ", V_tot.shape)
    #print("V_tot L2 norm is ", np.sqrt(np.mean(V_tot**2)))
    #print("V_tot Linfinity norm is ", np.max(np.abs(V_tot)))
    print(f"calculated potential in {time.time() - V_time} sec")
    Vs.append(V_tot)

Vs = np.array(Vs)

print(Vs.shape)

#if verbose:
    #ave_Vs = np.array([al.bootstrap(V, W, Nboot=100, f=adl.rw_mean)
    #        for V,W in zip(Vs, gfmc_Ws)])
#    ave_Vs = np.array([al.bootstrap(Vs[0], gfmc_Ws[0], Nboot=100, f=adl.rw_mean)])
#    print("V[tau=0] = ",ave_Vs,"\n\n")
#    ave_Vs = np.array([al.bootstrap(Vs[-1], gfmc_Ws[-1], Nboot=100, f=adl.rw_mean)])
#    print("V[last tau] = ",ave_Vs,"\n\n")

#Ks *= fm_Mev**2

#Vs = np.array([
#    sum([
#        AV_Coulomb[name](dRs) * adl.compute_O(adl.two_body_ops[name](dRs), S, S_av4p_metropolis)
#        for name in AV_Coulomb
#    ])
#    for dRs, S in zip(map(adl.to_relative, gfmc_Rs), gfmc_Ss)])

if volume == "finite":
    tag = str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_nskip" + str(n_skip) + "_Nf"+str(nf) + "_alpha"+str(alpha) + "_spoila"+str(spoila) + "_spoilf"+str(spoilf) + "_spoilS"+str(spoilS) + "_log_mu_r"+str(log_mu_r) + "_wavefunction_"+str(wavefunction) + "_potential_"+str(potential)+"_L"+str(L)
else:
    tag = str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_nskip" + str(n_skip) + "_Nf"+str(nf) + "_alpha"+str(alpha) + "_spoila"+str(spoila) + "_spoilf"+str(spoilf)+ "_spoilS"+str(spoilS) + "_log_mu_r"+str(log_mu_r) + "_wavefunction_"+str(wavefunction) + "_potential_"+str(potential)


with h5py.File(outdir+'Hammys_'+tag+'.h5', 'w') as f:
    dset = f.create_dataset("Hammys", data=Ks+Vs)
    dset = f.create_dataset("Ws", data=gfmc_Ws)

with h5py.File(outdir+'Hammys_'+tag+'.h5', 'r') as f:
    data = f['Hammys']
    print(data)

with h5py.File(outdir+'Rs_'+tag+'.h5', 'w') as f:
    dset = f.create_dataset("Rs", data=gfmc_Rs)
    dset = f.create_dataset("Ws", data=gfmc_Ws)

with h5py.File(outdir+'Rs_'+tag+'.h5', 'r') as f:
    data = f['Rs']
    print(data)

dset = Ks+Vs
print("dset = ", dset)
last_point = n_walkers//8
def tauint(t, littlec):
     return 1 + 2 * np.sum(littlec[1:t])
for tau_ac in range(0,n_step,10):
    sub_dset = np.real(dset[tau_ac] - np.mean(dset[tau_ac]))
    auto_corr = []
    c0 = np.mean(sub_dset * sub_dset)
    print("sub_dset = ", sub_dset)
    print("c0 = ", c0)
    auto_corr.append(c0)
    for i in range(1,n_walkers//4):
        auto_corr.append(np.mean(sub_dset[i:] * sub_dset[:-i]))
    littlec = np.asarray(auto_corr) / c0
    print("tau = ", tau_ac)
    print("integrated autocorrelation time = ", tauint(last_point, littlec))

if verbose:

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
