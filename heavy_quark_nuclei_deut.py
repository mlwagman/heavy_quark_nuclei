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
import afdmc_lib_deut as adl
import os
import pickle
from afdmc_lib_deut import NI,NS,mp_Mev,fm_Mev
import jax
import jax.numpy as np
import sys
from itertools import repeat
import time
import h5py
import math
import mpmath
from functools import partial

from itertools import permutations, chain
import torch
import torch.nn as nn
from collections import defaultdict

onp.random.seed(0)

paper_plt.load_latex_config()

parser = argparse.ArgumentParser()
parser.add_argument('--ferm_symm', type=str, choices=['s', 'a', 'mas'],default='a')
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
parser.add_argument('--radial_n', type=int, default=1)
parser.add_argument('--spoila', type=float, default=1)
parser.add_argument('--g', type=float, default=0)
parser.add_argument('--gfac', type=float, default=1)
parser.add_argument('--afac', type=float, default=1)
parser.add_argument('--samefac', type=float, default=1)
parser.add_argument('--spoilf', type=str, default="hwf")
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--input_Rs_database', type=str, default="")
parser.add_argument('--log_mu_r', type=float, default=1)
parser.add_argument('--cutoff', type=float, default=0.0)
parser.add_argument('--L', type=float, default=0.0)
parser.add_argument('--Lcut', type=int, default=5)
parser.add_argument('--spoilS', type=float, default=1)
parser.add_argument('--wavefunction', type=str, default="compact")
parser.add_argument('--color', type=str, default="1x1")
parser.add_argument('--potential', type=str, default="full")
parser.add_argument('--spoilaket', type=float, default=1)
parser.add_argument('--masses', type=float, default=0., nargs='+')
parser.add_argument('--verbose', dest='verbose', action='store_true', 
                    default=False)
globals().update(vars(parser.parse_args()))

##############################################################################

volume = "infinite"
if L > 1e-2:
    volume = "finite"

if masses == 0.:
    masses = onp.ones(N_coord)
    if N_coord == 2:
        masses = [1,-1]
    elif N_coord == 3:
        masses = [1,1,1]
    elif N_coord == 4:
        masses = [1,-1,1,-1]
    elif N_coord == 6:
        masses = [1,1,1,1,1,1]

masses_copy = masses

swapI = 1
for i in range(1,N_coord):
    if masses[1]*masses[i] > 0:
        swapI = i

def count_transpositions(perm):
    visited = [False] * len(perm)
    transpositions = 0

    for i in range(len(perm)):
        if not visited[i]:
            cycle_length = 0
            x = i
            while not visited[x]:
                visited[x] = True
                x = perm[x]
                cycle_length += 1
            if cycle_length > 1:
                transpositions += cycle_length - 1

    return (-1) ** transpositions

def unique_group_permutations(masses):
    # Group indices by mass
    mass_dict = {}
    for idx, mass in enumerate(masses):
        if mass not in mass_dict:
            mass_dict[mass] = []
        mass_dict[mass].append(idx)

    # Generate permutations for each group of identical masses
    perms_per_group = {mass: list(permutations(indices)) 
                        for mass, indices in mass_dict.items() 
                        if len(indices) > 1}

    # Start with the identity permutation
    complete_perms = [list(range(len(masses)))]

    # Combine permutations for each group
    for mass, perm_group in perms_per_group.items():
        new_complete_perms = []
        for base_perm in complete_perms:
            for group_perm in perm_group:
                new_perm = base_perm.copy()
                for i, idx in enumerate(group_perm):
                    new_perm[mass_dict[mass][i]] = base_perm[idx]
                new_complete_perms.append(new_perm)
        complete_perms = new_complete_perms

    # Calculate antisymmetrization factors for each permutation
    antisym_factors = [count_transpositions(perm) for perm in complete_perms]

    return complete_perms, antisym_factors

def count_transpositions_baryons(perm, group1, group2):
    transpositions = 1  # Start with a factor of +1
    # Moved outside the inner loop to print only once per permutation
    print("trying perm = ", perm)
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            # Check if the current pair involves a within-group swap 
            # (intra-baryon swap)
            if (i in group1 and j in group1) or (i in group2 and j in group2):
                if perm[i] > perm[j]:  # If out of order, it's a transposition
                    transpositions *= -1
                    print("add transposition minus sign for i = ", i, 
                          ", j = ", j, ", perm[i] = ", perm[i], 
                          ", perm[j] = ", perm[j])
            # Check if the current pair involves a between-group swap 
            # (inter-baryon swap)
            # We do nothing here because inter-baryon swaps 
            # do not affect the sign.
            elif (i in group1 and j in group2) \
                   or (i in group2 and j in group1):
                # Inter-baryon swaps do not change the transposition sign
                continue  

    return transpositions

def count_transpositions_baryons_no_intra_no_sign_change(perm, group1, 
                                                         group2):
    print("trying perm = ", perm)  # Print the permutation once per call
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            # Check if the current pair involves a within-group swap 
            # (intra-baryon swap)
            if ((i in group1 and j in group1) 
                    or (i in group2 and j in group2)) \
                    and (masses[i] == masses[j]) and (perm[i] > perm[j]):
                return 0
    return 1  # Always return +1 (no sign change)

def unique_group_permutations_baryons(masses):
    # Group the indices by mass
    mass_dict = {}
    for idx, mass in enumerate(masses):
        if mass not in mass_dict:
            mass_dict[mass] = []
        mass_dict[mass].append(idx)

    print("mass_dict = ", mass_dict)

    # Generate permutations for each group of identical masses
    perms_per_group = {mass: list(permutations(indices)) 
                       for mass, indices in mass_dict.items() 
                       if len(indices) > 1}

    print("perms_per_group = ", perms_per_group)

    # Start with the identity permutation
    complete_perms = [list(range(len(masses)))]

    # Define the two baryon groups
    group1 = [0, 1, 2]  # m1, m2, m3
    group2 = [3, 4, 5]  # m4, m5, m6

    # Combine permutations for each group
    for mass, perm_group in perms_per_group.items():
        new_complete_perms = []
        for base_perm in complete_perms:
            for group_perm in perm_group:
                new_perm = base_perm.copy()
                for i, idx in enumerate(group_perm):
                    new_perm[mass_dict[mass][i]] = base_perm[idx]
                new_complete_perms.append(new_perm)
        complete_perms = new_complete_perms

    print("len complete_perms = ", len(complete_perms))

    # Calculate antisymmetrization factors for each permutation, 
    # considering baryon groups
    unique_factors = \
      [count_transpositions_baryons_no_intra_no_sign_change(perm, 
            group1, group2) for perm in complete_perms]

    unique_complete_perms = []

    for p in range(len(complete_perms)):
        if unique_factors[p] != 0:
            unique_complete_perms.append(complete_perms[p])

    antisym_factors = [count_transpositions_baryons(perm, group1, group2) 
                         for perm in unique_complete_perms]

    return unique_complete_perms, antisym_factors


print("ferm_symm = ", ferm_symm)

if ferm_symm == "a" or ferm_symm == "s":
    perms, antisym_factors = unique_group_permutations(masses)
else:
    perms, antisym_factors = unique_group_permutations_baryons(masses)

# just doing interesting 2-flavor case
if N_coord == 4:
    # hardcoded in (q qbar) (q qbar) ordering
    assert(swapI != 1)
    perms = [(0,1,2,3),(2,1,0,3)]
    if ferm_symm == "s":
        antisym_factors=[1,1]
    else:   
        antisym_factors=[1,-1]

if ferm_symm == "s":
    antisym_factors=[1] * len(perms)

# reset all masses to +/- 1 now that we have computed perms from them
masses_print = masses
masses /= onp.abs(masses)
print("masses = ", masses)

# Display permutations with antisymmetrization factors
print("Unique permutations of indices and their antisymmetrization factors:")
for perm, factor in zip(perms, antisym_factors):
    print(f"Permutation: {perm}, Factor: {factor}")

print("length of perms = ", len(perms))

# TODO: WORKS! Gives list (0,1,2,3,4,5),...

bra_wavefunction = wavefunction
ket_wavefunction = wavefunction

CF = (Nc**2 - 1)/(2*Nc)
VB = alpha*CF/(Nc-1)
SingC3 = -(Nc+1)/8

# imaginary time points for GFMC evolution
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)

beta0 = 11/3*Nc - 2/3*nf
beta1 = 34/3*Nc**2 - 20/3*Nc*nf/2 - 2*CF*nf
aa1 = 31/9*Nc-10/9*nf
zeta3 = scipy.special.zeta(3)
aa2 = ( 4343/162 + 6*np.pi**2 - np.pi**4/4 + 22/3*zeta3 )*Nc**2 \
      - ( 1798/81 + 56/3*zeta3 )*Nc*nf/2 \
      - ( 55/3 - 16*zeta3  )*CF*nf/2 + (10/9*nf)**2

VB_LO = VB
VB_NLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*log_mu_r))
VB_NNLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*log_mu_r) 
                + (alpha/(4*np.pi))**2*( beta0**2*(4*log_mu_r**2 
                    + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*log_mu_r + aa2 ) )

if OLO == "LO":
    a0=spoila*2/VB_LO
elif OLO == "NLO":
    a0=spoila*2/VB_NLO
elif OLO == "NNLO":
    a0=spoila*2/VB_NNLO

if N_coord == 2 or N_coord == 4:
    a0 /= Nc-1

ket_a0 = a0
biga0 = a0

if wavefunction == "product":
    biga0 = a0*afac

ket_afac = afac*spoilaket

print("a0 = ", a0)
print("ket_a0 = ", ket_a0)
print("afac = ", afac)
print("ket_afac = ", ket_afac)

Rprime = lambda R: adl.norm_3vec(R)*np.exp(np.euler_gamma)*mu
# build Coulomb potential
AV_Coulomb = {}
B3_Coulomb = {}

# Generate the nn array representing 3D shifts in a cubic grid
pp = np.array([np.array([i, j, k]) for i in range(-Lcut, Lcut+1) 
              for j in range(-Lcut, Lcut+1) for k in range(-Lcut, Lcut+1)])

print(pp.shape)

print("zero mode is ", pp[Lcut*(2*Lcut+1)*(2*Lcut+1)+Lcut*(2*Lcut+1)+Lcut])

nn = np.delete(pp, Lcut*(2*Lcut+1)*(2*Lcut+1)+Lcut*(2*Lcut+1)+Lcut, axis=0)

print(nn.shape)
print(pp.shape)

@partial(jax.jit)
def FV_Coulomb(R, L, nn):
    sums = np.zeros(n_walkers)
    sums += -1
    Rdotn = np.einsum('bi,ki->bk', R, nn)
    n_mag_sq = np.sum( nn*nn, axis=1 )
    sums += np.sum( np.exp((2*np.pi*1j/L)*Rdotn)
                    *np.exp(-n_mag_sq)/n_mag_sq, axis=1 )
    Rdotp = np.einsum('bi,ki->bk', R, pp)
    RdotR = np.sum( R*R, axis=1 )
    pdotp = np.sum( pp*pp, axis=1 )
    pmRL = np.sqrt( Rdotp*(-2.0/L) + pdotp[(np.newaxis,slice(None))] 
                    + (1.0/L)**2*RdotR[(slice(None),np.newaxis)] )
    sums += np.sum( np.pi/pmRL*(1-jax.scipy.special.erf(np.pi*pmRL)), axis=1 )
    return sums/(np.pi*L)

def FV_Coulomb_slow(R, L, nn):
    sums = np.zeros(n_walkers)
    sums += -1
    for i in range(len(nn)):
        n=nn[i]
        n_mag = adl.norm_3vec(n)
        sums += np.exp(-n_mag**2)/n_mag**2*np.exp(2*np.pi*1j*np.sum(n*R,axis=1)/L)
    for i in range(len(pp)):
        n=pp[i]
        n_mag = adl.norm_3vec(n)
        sums += np.pi/adl.norm_3vec(n - R/L)*(1-jax.scipy.special.erf(np.pi*adl.norm_3vec(n - R/L)))
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
        return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)
                  * (2*beta0*np.log(Rprime(R))+aa1))
    @partial(jax.jit)
    def potential_fun_sum(R):
        return calculate_sum(potential_fun, R, L, nn)
    def symmetric_potential_fun(R):
        return (Nc - 1)/(Nc + 1)*VB*spoilS/adl.norm_3vec(R) \
                * (1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    def symmetric_potential_fun_sum(R):
        return calculate_sum(symmetric_potential_fun, R, L, nn)
    @partial(jax.jit)
    def singlet_potential_fun(R):
        return -1*(Nc - 1)*VB/adl.norm_3vec(R) \
                * (1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    @partial(jax.jit)
    def singlet_potential_fun_sum(R):
        return calculate_sum(potential_fun, R, L, nn)
    def octet_potential_fun(R):
        return spoilS*(Nc - 1)/CF/(2*Nc)*VB/adl.norm_3vec(R) \
                * (1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    def octet_potential_fun_sum(R):
        return calculate_sum(symmetric_potential_fun, R, L, nn)
elif OLO == "NNLO":
    @partial(jax.jit)
    def potential_fun(R):
        return -1*spoilS*VB/adl.norm_3vec(R) \
                * (1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) 
                    + (alpha/(4*np.pi))**2*( beta0**2
                        * (4*np.log(Rprime(R))**2 + np.pi**2/3) 
                        + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))
                        + aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2) ) )
    @partial(jax.jit)
    def potential_fun_sum(R):
        return calculate_sum(potential_fun, R, L, nn)
    def symmetric_potential_fun(R):
        return spoilS*(Nc - 1)/(Nc + 1)*VB/adl.norm_3vec(R) \
                * (1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) 
                    + (alpha/(4*np.pi))**2*( beta0**2
                        * (4*np.log(Rprime(R))**2 + np.pi**2/3) 
                        + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))
                        + aa2 + Nc*(Nc+2)/2*((np.pi)**4-12*(np.pi)**2) ) )
    def symmetric_potential_fun_sum(R):
        return calculate_sum(symmetric_potential_fun, R, L, nn)
    @partial(jax.jit)
    def singlet_potential_fun(R):
        return -1*(Nc - 1)*VB/adl.norm_3vec(R) \
                * (1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) 
                    + (alpha/(4*np.pi))**2*( beta0**2
                       * (4*np.log(Rprime(R))**2 + np.pi**2/3) 
                       + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R)) + aa2 ) )
    @partial(jax.jit)
    def singlet_potential_fun_sum(R):
        return calculate_sum(potential_fun, R, L, nn)
    def octet_potential_fun(R):
        return spoilS*(Nc - 1)/CF/(2*Nc)*VB/adl.norm_3vec(R) \
                * (1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1) 
                    + (alpha/(4*np.pi))**2*( beta0**2
                        * (4*np.log(Rprime(R))**2 + np.pi**2/3) 
                        + 2*( beta1+2*beta0*aa1 )*np.log(Rprime(R))
                        + aa2 + (Nc**2)*((np.pi)**4-12*(np.pi)**2) ) )
    def octet_potential_fun_sum(R):
        return calculate_sum(symmetric_potential_fun, R, L, nn)
else:
    print("order not supported")
    throw(0)

print("volume = ", volume)

if volume == "finite":
    AV_Coulomb['OA'] = potential_fun_sum
    AV_Coulomb['OS'] = symmetric_potential_fun_sum
    AV_Coulomb['OSing'] = singlet_potential_fun_sum
    AV_Coulomb['OO'] = octet_potential_fun_sum
else:
    AV_Coulomb['OA'] = potential_fun
    AV_Coulomb['OS'] = symmetric_potential_fun
    AV_Coulomb['OSing'] = singlet_potential_fun
    AV_Coulomb['OO'] = octet_potential_fun

print("AV_Coulomb = ", AV_Coulomb)

if potential == "product":
    Coulomb_potential = adl.make_pairwise_product_potential(AV_Coulomb, 
                              B3_Coulomb, masses)
else:
    Coulomb_potential = adl.make_pairwise_potential(AV_Coulomb, 
                              B3_Coulomb, masses)

# build Coulomb ground-state trial wavefunction
@partial(jax.jit, static_argnums=(1,))
def f_R_slow(Rs, wavefunction=bra_wavefunction, a0=a0):
    psi = 1
    for i in range(N_coord):
       	for j in range(N_coord):
            thisa0=a0
            if i!=j and j>=i:
                if wavefunction == "product":
                    baryon_0 = 1
                    if i < N_coord/2:
                        baryon_0 = 0
                    baryon_1 = 1
                    if j < N_coord/2:
                        baryon_1 = 0
                    if baryon_0 != baryon_1:
                        thisa0 = biga0
                ri = Rs[...,i,:]
                rj = Rs[...,j,:]
                rij_norm = adl.norm_3vec(ri - rj)
                psi = psi*np.exp(-rij_norm/thisa0)
    return psi

pairs = np.array([np.array([j, i]) for i in range(0,N_coord) 
                    for j in range(0, i)])
print("pairs = ", pairs)

same_pairs = []
for i in range(N_coord):
    for j in range(N_coord):
        if i!=j and j>=i:
            if masses[i]*masses[j] > 0:
                same_pairs.append(np.array([i,j]))
same_pairs = np.array(same_pairs)
print("same_pairs = ", same_pairs)

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

diquark_pairs = []
for i in range(N_coord):
    for j in range(N_coord):
        if i!=j and j>=i:
            diquark_0 = 2
            if i < 2:
                diquark_0 = 0
            elif i < 4:
                diquark_0 = 1
            diquark_1 = 2
            if j < 2:
                diquark_1 = 0
            elif j < 4:
                diquark_1 = 1
            if diquark_0 != diquark_1:
                continue
            diquark_pairs.append(np.array([i,j]))
diquark_pairs = np.array(diquark_pairs)
print("diquark pairs = ", diquark_pairs)

absmasses=np.abs(np.array(masses))
def hydrogen_wvfn(r, n):
    psi = np.exp(-r/n)
    if n == 1:
        return psi
    if n == 2:
        return psi*(2 - r)
    if n == 3:
        return psi*(27 - 18*r + 2*r**2)
    if n == 4:
        return psi*(192 - 144*r + 24*r**2 - r**3)
    if n == 5:
        return psi*(9375 - 7500*r + 1500*r**2 - 100*r**3 + 2*r**4)
    if n == 6:
        return psi*(174960 - 145800*r + 32400*r**2 - 2700*r**3 
                    + 90*r**4 - r**5)
    if n == 7:
        return psi*(37059435 - 31765230*r + 7563150*r**2 - 720300*r**3 
                    + 30870*r**4 - 588*r**5 + 4*r**6)


#TODO: ADD PERMS OPTIONS IF N_COORD=6

@partial(jax.jit, static_argnums=(2,))
def f_R(Rs,perm=None, wavefunction=bra_wavefunction, a0=a0, 
        afac=afac, masses=absmasses):
    # check permuting simply gives same and only do for one perm
    # Apply permutations if provided and N_coord is 6

    if perm is not None and (N_coord == 4 or N_coord == 6):
        print(f"Perm type: {type(perm)}")
        print(f"Perm contents: {perm}")

        perm = np.array(perm)

        if Rs.shape[-2] == N_coord:
            # Handle both (N_coord, 3) and (N_walkers, N_coord, 3) shapes
            if Rs.ndim == 2:
                # Shape is (N_coord, 3)
                Rs = Rs[perm, :]
            elif Rs.ndim == 3:
                # Shape is (N_walkers, N_coord, 3)
                Rs = Rs[:, perm, :]
            else:
                raise ValueError(f"Unexpected shape for Rs: {Rs.shape}")

    def r_norm(pair):
        [i,j] = pair
        rdiff = Rs[...,i,:] - Rs[...,j,:]
        mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
        rij_norm = np.sqrt( np.sum(rdiff*rdiff, axis=-1) )
        return rij_norm * mij

    #UNIT TESTED

    # Setup pair lists for the mappings 
    # (these need to be predefined or passed into the function)

    if wavefunction == "product":
        r_sum = np.sum( jax.lax.map(r_norm, product_pairs), axis=0 ) \
                  *(1/a0-1/(a0*afac)) \
                  + np.sum( jax.lax.map(r_norm, pairs), axis=0 )/(a0*afac)
        r_sum += np.sum( jax.lax.map(r_norm, same_pairs), axis=0 ) \
                  *(1/(a0*afac*samefac)-1/(a0*afac))
    elif wavefunction == "diquark":
        r_sum = np.sum( jax.lax.map(r_norm, diquark_pairs), axis=0 ) \
                  * (1/a0-1/(a0*afac)) \
                  + np.sum( jax.lax.map(r_norm, pairs), axis=0 )/(a0*afac)
    else:
        r_sum = np.sum( jax.lax.map(r_norm, pairs), axis=0 )/a0

    psi = hydrogen_wvfn(r_sum, radial_n)
    afac *= gfac
    a0 /= gfac

    # Additional logic for N_coord == 4
    if N_coord == 4 and abs(g) > 0:
        Rs_T = Rs
        Rs_T = Rs_T.at[...,1,:].set(Rs[...,swapI,:])
        Rs_T = Rs_T.at[...,swapI,:].set(Rs[...,1,:])

        def r_norm_T(pair):
            [i,j] = pair
            rdiff = Rs_T[...,i,:] - Rs_T[...,j,:]
            rij_norm = np.sqrt( np.sum(rdiff*rdiff, axis=-1) )
            mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
            return rij_norm * mij

        if wavefunction == "product":
            r_sum_T = np.sum( jax.lax.map(r_norm_T, product_pairs), axis=0 ) \
                      * (1/a0-1/(a0*afac)) \
                      + np.sum(jax.lax.map(r_norm_T, pairs), axis=0)/(a0*afac)
            r_sum_T += np.sum( jax.lax.map(r_norm_T, same_pairs), axis=0 ) \
                      *(1/(a0*afac*samefac)-1/(a0*afac))
        elif wavefunction == "diquark":
            r_sum_T = np.sum( jax.lax.map(r_norm_T, diquark_pairs), axis=0 ) \
                      *(1/a0-1/(a0*afac)) \
                      + np.sum(jax.lax.map(r_norm_T, pairs), axis=0)/(a0*afac)
        else:
            r_sum_T = np.sum( jax.lax.map(r_norm_T, pairs), axis=0 )/a0

        psi += g * hydrogen_wvfn(r_sum_T, radial_n)
    return psi

#TODO: ADD PERMS OPTIONS

# Ensure this matches the earlier permutation list format
@partial(jax.jit, static_argnums=(2,))
def laplacian_f_R(Rs,perm=None, wavefunction=bra_wavefunction, a0=a0, 
                  afac=afac, masses=absmasses):
    if radial_n > 1:
        assert N_coord == 2
    nabla_psi_tot = 0

    if perm is not None and (N_coord == 4 or N_coord == 6):
        print("Inside f_R:")
        print(f"Perm type: {type(perm)}")
        print(f"Perm contents: {perm}")

        perm = np.array(perm)

        if Rs.shape[-2] == N_coord:
            # Handle both (N_coord, 3) and (N_walkers, N_coord, 3) shapes
            if Rs.ndim == 2:
                # Shape is (N_coord, 3)
                Rs = Rs[perm, :]
            elif Rs.ndim == 3:
                # Shape is (N_walkers, N_coord, 3)
                Rs = Rs[:, perm, :]
            else:
                raise ValueError(f"Unexpected shape for Rs: {Rs.shape}")

    #UNIT TESTED

    for k in range(N_coord):
        for l in range(N_coord):
            if k!=l and l>=k:
                # wvfn includes r_ij
                nabla_psi = 1
                for i in range(N_coord):
                    for j in range(N_coord):
                        thisa0 = a0
                        if i!=j and j>=i:
                            if wavefunction == "product":
                                baryon_0 = 1
                                if i < N_coord/2:
                                    baryon_0 = 0
                                baryon_1 = 1
                                if j < N_coord/2:
                                    baryon_1 = 0
                                if baryon_0 != baryon_1:
                                    thisa0 *= afac
                                    #continue
                                if masses_copy[i]*masses_copy[j] > 0:
                                    thisa0 *= samefac
                            elif wavefunction == "diquark":
                                diquark_0 = 2
                                if i < 2:
                                    diquark_0 = 0
                                elif i < 4:
                                    diquark_0 = 1
                                diquark_1 = 2
                                if j < 2:
                                    diquark_1 = 0
                                elif j < 4:
                                    diquark_1 = 1
                                if diquark_0 != diquark_1:
                                    thisa0 *= afac
                                if masses_copy[i]*masses_copy[j] > 0:
                                    thisa0 *= samefac
                            ri = Rs[...,i,:]
                            rj = Rs[...,j,:]
                            rij_norm = adl.norm_3vec(ri - rj)
                            mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
                            thisa0 /= mij
                            # nabla_k^2 r_kl = nabla_l^2 r_kl
                            # factor of two included to account for both 
                            # terms appearing in laplacian
                            if k == i and l == j:
                                nabla_psi = nabla_psi \
                                    * ((1/(radial_n*thisa0)**2 
                                        - 2/(thisa0*rij_norm))/masses[k] 
                                      + (1/(radial_n*thisa0)**2 
                                        - 2/(thisa0*rij_norm))/masses[l]) \
                                    * hydrogen_wvfn(rij_norm/thisa0, radial_n)
                            else:
                                nabla_psi = nabla_psi \
                                    * hydrogen_wvfn(rij_norm/thisa0, radial_n)
                nabla_psi_tot += nabla_psi

    # Terms where gradients hit separate pieces of the wavefunction
    for a in range(N_coord):
        # first gradient involves r_kl
        for k in range(N_coord):
            for l in range(N_coord):
                if k!=l and l>=k and (a==k or a==l):
                    # second gradient involves r_mn
                    for m in range(N_coord):
                        for n in range(N_coord):
                            if (m!=n and n>=m and (m!=k or n!=l) 
                                    and (a==m or a==n)):
                                # sum over the 3-d components of gradient
                                # wvfn involves r_ij
                                nabla_psi = 1
                                for i in range(N_coord):
                                    for j in range(N_coord):
                                        thisa0 = a0
                                        if i!=j and j>=i:
                                            if wavefunction == "product":
                                                baryon_0 = 1
                                                if i < N_coord/2:
                                                    baryon_0 = 0
                                                baryon_1 = 1
                                                if j < N_coord/2:
                                                    baryon_1 = 0
                                                if baryon_0 != baryon_1:
                                                    thisa0 *= afac
                                                    #continue
                                                if (masses_copy[i]
                                                     * masses_copy[j]) > 0:
                                                    thisa0 *= samefac
                                            elif wavefunction == "diquark":
                                                diquark_0 = 2
                                                if i < 2:
                                                    diquark_0 = 0
                                                elif i < 4:
                                                    diquark_0 = 1
                                                diquark_1 = 2
                                                if j < 2:
                                                    diquark_1 = 0
                                                elif j < 4:
                                                    diquark_1 = 1
                                                if diquark_0 != diquark_1:
                                                    thisa0 *= afac
                                            ri = Rs[...,i,:]
                                            rj = Rs[...,j,:]
                                            rij_norm = adl.norm_3vec(ri - rj)
                                            mij = 2*masses[i]*masses[j] \
                                                   / (masses[i]+masses[j])
                                            thisa0 /= mij
                                            rsign = 0
                                            new_shape = onp.array(ri.shape)
                                            new_shape[-1] = 1
                                            # grad_a r_ij = rsign * (ri - rj)
                                            if a == i:
                                                rsign = 1
                                            elif a == j:
                                                rsign = -1
                                            if ((k == i and l == j) 
                                                    or (m == i and n == j)):
                                                nabla_psi = rsign \
                                                  * nabla_psi * (ri - rj) \
                                                  * (np.exp(-rij_norm/thisa0) 
                                                      / (thisa0*rij_norm)).reshape(new_shape)
                                            else:
                                                nabla_psi = nabla_psi \
                                                  * np.exp(-rij_norm/thisa0).reshape(new_shape)
                                nabla_psi_tot += np.sum(nabla_psi, axis=-1) \
                                                  / np.abs(masses[a])
    return nabla_psi_tot


if N_coord >= 6 and verbose:
    print("No JIT for Laplacian")
    def laplacian_f_R(Rs,perm=None, wavefunction=bra_wavefunction, a0=a0, 
                      afac=afac, masses=absmasses):
        nabla_psi_tot = 0

        if perm is not None and (N_coord == 4 or N_coord == 6):
            print("Inside f_R:")
            print(f"Perm type: {type(perm)}")
            print(f"Perm contents: {perm}")

            perm = np.array(perm)

            if Rs.shape[-2] == N_coord:
                # Handle both (N_coord, 3) and (N_walkers, N_coord, 3) shapes
                if Rs.ndim == 2:
                    # Shape is (N_coord, 3)
                    Rs = Rs[perm, :]
                elif Rs.ndim == 3:
                    # Shape is (N_walkers, N_coord, 3)
                    Rs = Rs[:, perm, :]
                else:
                    raise ValueError(f"Unexpected shape for Rs: {Rs.shape}")

        #UNIT TESTED
        # terms where laplacian hits one piece of wvfn
        # laplacian hits r_kl
        for k in range(N_coord):
            for l in range(N_coord):
                if k!=l and l>=k:
                    # wvfn includes r_ij
                    nabla_psi = 1
                    for i in range(N_coord):
                        for j in range(N_coord):
                            thisa0 = a0
                            if i!=j and j>=i:
                                if wavefunction == "product":
                                    baryon_0 = 1
                                    if i < N_coord/2:
                                        baryon_0 = 0
                                    baryon_1 = 1
                                    if j < N_coord/2:
                                        baryon_1 = 0
                                    if baryon_0 != baryon_1:
                                        thisa0 *= afac
                                        #continue
                                elif wavefunction == "diquark":
                                    diquark_0 = 2
                                    if i < 2:
                                        diquark_0 = 0
                                    elif i < 4:
                                        diquark_0 = 1
                                    diquark_1 = 2
                                    if j < 2:
                                        diquark_1 = 0
                                    elif j < 4:
                                        diquark_1 = 1
                                    if diquark_0 != diquark_1:
                                        thisa0 *= afac
                                ri = Rs[...,i,:]
                                rj = Rs[...,j,:]
                                rij_norm = adl.norm_3vec(ri - rj)
                                mij = 2*masses[i]*masses[j] \
                                      / (masses[i]+masses[j])
                                thisa0 /= mij
                                # nabla_k^2 r_kl = nabla_l^2 r_kl
                                # factor of two included to account for 
                                # both terms appearing in laplacian
                                if k == i and l == j:
                                    nabla_psi = nabla_psi \
                                        * ((1/thisa0**2 
                                            - 2/(thisa0*rij_norm))/masses[k] 
                                          + (1/thisa0**2 
                                            - 2/(thisa0*rij_norm))
                                          /masses[l]) \
                                        * np.exp(-rij_norm/thisa0)
                                else:
                                    nabla_psi = nabla_psi \
                                                * np.exp(-rij_norm/thisa0)
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
                                if (m!=n and n>=m and (m!=k or n!=l) 
                                        and (a==m or a==n)):
                                    # sum over the 3-d components of gradient
                                    for x in range(3):
                                        # wvfn involves r_ij
                                        nabla_psi = 1
                                        for i in range(N_coord):
                                            for j in range(N_coord):
                                                thisa0 = a0
                                                if i!=j and j>=i:
                                                    if wavefunction == "product":
                                                        baryon_0 = 1
                                                        if i < N_coord/2:
                                                            baryon_0 = 0
                                                        baryon_1 = 1
                                                        if j < N_coord/2:
                                                            baryon_1 = 0
                                                        if baryon_0 != baryon_1:
                                                            thisa0 *= afac
                                                    elif wavefunction == "diquark":
                                                        diquark_0 = 2
                                                        if i < 2:
                                                            diquark_0 = 0
                                                        elif i < 4:
                                                            diquark_0 = 1
                                                        diquark_1 = 2
                                                        if j < 2:
                                                            diquark_1 = 0
                                                        elif j < 4:
                                                            diquark_1 = 1
                                                        if diquark_0 != diquark_1:
                                                            thisa0 *= afac
                                                    ri = Rs[...,i,:]
                                                    rj = Rs[...,j,:]
                                                    rij_norm = adl.norm_3vec(ri - rj)
                                                    mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
                                                    thisa0 /= mij
                                                    rsign = 0
                                                    # grad_a r_ij = rsign * (ri - rj)
                                                    if a == i:
                                                        rsign = 1
                                                    elif a == j:
                                                        rsign = -1
                                                    if (k == i and l == j) or (m == i and n == j):
                                                        nabla_psi = rsign * nabla_psi * (ri[:,x] - rj[:,x])/(thisa0*rij_norm) * np.exp(-rij_norm/thisa0)
                                                    else:
                                                        nabla_psi = nabla_psi * np.exp(-rij_norm/thisa0)
                                        nabla_psi_tot += nabla_psi / np.abs(masses[a])
        return nabla_psi_tot

else:
    print("JIT Laplacian")

# build trial wavefunction

# TODO: ADD PERMS OPTIONS AND BUILD MULTIPLE SPIN-FLAV WVFNS - 
# DEFINE FUNCTION THAT DOES ALL THIS IF N_COORD==6

S_av4p_metropolis = onp.zeros(shape=(N_coord,) + (NI,NS)*N_coord).astype(np.complex128)
print("built Metropolis wavefunction ensemble")

# trial wavefunction spin-flavor structure is |up,u> x |up,u> x ... x |up,u>

def levi_civita(i, j, k):
    if i == j or j == k or i == k:
        return 0
    if (i,j,k) in [(0,1,2), (1,2,0), (2,0,1)]:
        return 1
    else:
        return -1

def TAAA(i, j, k, l, m, n):
    return (levi_civita(i,j,m)*levi_civita(k,l,n) 
              - levi_civita(i,j,n)*levi_civita(k,l,m))/(4*np.sqrt(3))

def TAAS(i, j, k, l, m, n):
    return (levi_civita(i,j,m)*levi_civita(k,l,n) 
              + levi_civita(i,j,n)*levi_civita(k,l,m))/(4*np.sqrt(6))

def TASA(i, j, k, l, m, n):
    return (levi_civita(i,j,k)*levi_civita(m,n,l) 
              + levi_civita(i,j,l)*levi_civita(m,n,k))/(4*np.sqrt(6))

def TSAA(i, j, k, l, m, n):
    return (levi_civita(m,n,i)*levi_civita(k,l,j) 
              + levi_civita(m,n,j)*levi_civita(k,l,i))/(4*np.sqrt(6))

def TSSS(i, j, k, l, m, n):
    return (levi_civita(i,k,m)*levi_civita(j,l,n) 
              + levi_civita(i,k,n)*levi_civita(j,l,m) 
              + levi_civita(j,k,m)*levi_civita(i,l,n) 
              + levi_civita(j,k,n)*levi_civita(i,l,m))/(12*np.sqrt(2))

def kronecker_delta(i, j):
    return 1 if i == j else 0

print("spin-flavor wavefunction shape = ", S_av4p_metropolis.shape)

if N_coord == 3:
    for i in range(NI):
        for j in range(NI):
            for k in range(NI):
                if i != j and j != k and i != k:
                    spin_slice = (slice(0, None),) + (i,0,j,0,k,0)
                    S_av4p_metropolis[spin_slice] \
                        = levi_civita(i, j, k) / np.sqrt(2*NI)

if N_coord == 2:
    for i in range(NI):
        for j in range(NI):
            if i == j:
                spin_slice = (slice(0, None),) + (i,0,j,0)
                S_av4p_metropolis[spin_slice] \
                    = kronecker_delta(i, j)/np.sqrt(NI)

def generate_wavefunction_tensor(NI, NS, N_coord, full_permutations, color):

    # Initialize base tensor to hold wavefunctions
    S_av4p_metropolis_set = []

    # Example shape for Rs_metropolis, ensure this matches your actual usage
    #Rs_metropolis = onp.random.normal(size=(N_coord,3))/np.mean(absmasses)  
    # Placeholder for actual Rs_metropolis data

    for perm in full_permutations:
        S_av4p_metropolis = onp.zeros((n_walkers,) + (NI, NS) * N_coord, 
                              dtype=np.complex128)
        if N_coord == 4:
            for i_old in range(NI):
                for j_old in range(NI):
                    for k_old in range(NI):
                        for l_old in range(NI):
                            spin_slice = (slice(0, None),) \
                                  + (i_old,0,j_old,0,k_old,0,l_old,0)
                            original_indices = [i_old, j_old, k_old, l_old]
                            permuted_indices = [original_indices[idx] 
                                                for idx in perm]
                            i, j, k, l = tuple(permuted_indices)
                            if color == "1x1":
                                if swapI == 1:
                                    # 1 x 1 -- Q Q Qbar Qbar
                                    S_av4p_metropolis[spin_slice] = kronecker_delta(i, k)*kronecker_delta(j,l)/NI
                                else:
                                    # 1 x 1 -- Q Qbar Q Qbar
                                    S_av4p_metropolis[spin_slice] = kronecker_delta(i, j)*kronecker_delta(k,l)/NI
                            elif color == "3x3bar":
                                if swapI == 1:
                                    # 3bar x 3 -- Q Q Qbar Qbar
                                    S_av4p_metropolis[spin_slice] += kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2-2*NI)
                                    S_av4p_metropolis[spin_slice] -= kronecker_delta(i, l)*kronecker_delta(j, k)/np.sqrt(2*NI**2-2*NI)
                                else:
                                    # 3bar x 3 -- Q Qbar Q Qbar
                                    S_av4p_metropolis[spin_slice] += kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2-2*NI)
                                    S_av4p_metropolis[spin_slice] -= kronecker_delta(i, l)*kronecker_delta(k, j)/np.sqrt(2*NI**2-2*NI)
                            elif color == "3x3bar-1":
                                if swapI == 1:
                                    # 3bar x 3 -- Q Q Qbar Qbar
                                    part3x3bar = kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2-2*NI)
                                    part3x3bar -= kronecker_delta(i, l)*kronecker_delta(j, k)/np.sqrt(2*NI**2-2*NI)
                                    part1x1 = kronecker_delta(i, k)*kronecker_delta(j,l)/NI
                                    S_av4p_metropolis[spin_slice] = (part3x3bar - (1/np.sqrt(3))*part1x1)/np.sqrt(2/3)
                                else:
                                    # 3bar x 3 -- Q Qbar Q Qbar
                                    part3x3bar = kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2-2*NI)
                                    part3x3bar -= kronecker_delta(i, l)*kronecker_delta(k, j)/np.sqrt(2*NI**2-2*NI)
                                    part1x1 = kronecker_delta(i, j)*kronecker_delta(k,l)/NI
                                    S_av4p_metropolis[spin_slice] = (part3x3bar - (1/np.sqrt(3))*part1x1)/np.sqrt(2/3)
                            elif color == "3x3bar-6x6bar":
                                theta_c = 1.0
                                if swapI == 1:
                                    # 3bar x 3 -- Q Q Qbar Qbar
                                    part3x3bar = kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2-2*NI)
                                    part3x3bar -= kronecker_delta(i, l)*kronecker_delta(j, k)/np.sqrt(2*NI**2-2*NI)
                                    part6x6bar = kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2+2*NI)
                                    part6x6bar += kronecker_delta(i, l)*kronecker_delta(j,k)/np.sqrt(2*NI**2+2*NI)
                                    S_av4p_metropolis[spin_slice] = np.cos(theta_c)*part3x3bar + np.sin(theta_c)*part6x6bar
                                else:
                                    # 3bar x 3 -- Q Qbar Q Qbar
                                    part3x3bar = kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2-2*NI)
                                    part3x3bar -= kronecker_delta(i, l)*kronecker_delta(k, j)/np.sqrt(2*NI**2-2*NI)
                                    part6x6bar = kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2+2*NI)
                                    part6x6bar += kronecker_delta(i, l)*kronecker_delta(k,j)/np.sqrt(2*NI**2+2*NI)
                                    S_av4p_metropolis[spin_slice] = np.cos(theta_c)*part3x3bar + np.sin(theta_c)*part6x6bar
                            elif color == "6x6bar":
                                if swapI == 1:
                                    # 6bar x 6 -- Q Q Qbar Qbar
                                    S_av4p_metropolis[spin_slice] += kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2+2*NI)
                                    S_av4p_metropolis[spin_slice] += kronecker_delta(i, l)*kronecker_delta(j,k)/np.sqrt(2*NI**2+2*NI)
                                else:
                                    # 6bar x 6 -- Q Qbar Q Qbar
                                    S_av4p_metropolis[spin_slice] += kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2+2*NI)
                                    S_av4p_metropolis[spin_slice] += kronecker_delta(i, l)*kronecker_delta(k,j)/np.sqrt(2*NI**2+2*NI)
        if N_coord == 6:
            original_indices = ["i", "j", "k", "l", "m", "n"]
            permuted_indices = [original_indices[idx] for idx in perm]
            print("original = ", original_indices)
            print("permuted = ", permuted_indices)

            for i_old in range(NI):
                for j_old in range(NI):
                    for k_old in range(NI):
                        for l_old in range(NI):
                            for m_old in range(NI):
                                for n_old in range(NI):
                                    original_indices = [i_old, j_old, k_old, l_old, m_old, n_old]
                                    permuted_indices = [original_indices[idx] for idx in perm]
                                    i, j, k, l, m, n = tuple(permuted_indices)

                                    spin_slice = (slice(0, None),) + (i_old,0,j_old,0,k_old,0,l_old,0,m_old,0,n_old,0)

                                    # Assign values based on color using explicit parameters
                                    if color == "1x1":
                                        S_av4p_metropolis[spin_slice] = levi_civita(i, j, k) * levi_civita(l, m, n) / 6
                                    elif color == "AAA":
                                        S_av4p_metropolis[spin_slice] = TAAA(i, j, k, l, m, n)
                                    elif color == "AAS":
                                        S_av4p_metropolis[spin_slice] = TAAS(i, j, k, l, m, n)
                                    elif color == "ASA":
                                        S_av4p_metropolis[spin_slice] = TASA(i, j, k, l, m, n)
                                    elif color == "SAA":
                                        S_av4p_metropolis[spin_slice] = TSAA(i, j, k, l, m, n)
                                    elif color == "SSS":
                                        S_av4p_metropolis[spin_slice] = TSSS(i, j, k, l, m, n)

        S_av4p_metropolis_set.append(S_av4p_metropolis)

    print(f"Generated {len(S_av4p_metropolis_set)} wavefunction ensembles based on permutations.")
    return S_av4p_metropolis_set

if N_coord == 6 or N_coord == 4:
    S_av4p_metropolis_set = generate_wavefunction_tensor(NI, NS, N_coord, 
                                                         perms, color)

    def f_R_braket(Rs):
        total_wvfn = np.zeros((NI, NS) * N_coord, dtype=np.complex128)
        for ii in range(len(perms)):
            f_R_tensor = f_R(Rs, perms[ii], wavefunction=bra_wavefunction)
            # use the 1-walker version of tensor if Rs only has 1 walker
            if len(Rs.shape) == 2:
                S_av4_tensor = S_av4p_metropolis_set[ii][0]
            # otherwise fall back on full version of S
            else:
                S_av4_tensor = S_av4p_metropolis_set[ii]
            total_wvfn += antisym_factors[ii] * S_av4_tensor * f_R_tensor
        Ss=np.array([total_wvfn])
        #CHECK SAV4^2 =1!!
        result = np.abs( adl.inner(Ss,Ss) )
        if len(Rs.shape) != 2:
            result /= n_walkers
        return result

else:
    def f_R_braket(Rs):
        print("total wvfn inner = ", 
               f_R(Rs, wavefunction=bra_wavefunction)**2 )
        return np.abs( f_R(Rs, wavefunction=bra_wavefunction)**2 )

def f_R_braket_phase(Rs):
    prod = f_R(Rs, wavefunction=bra_wavefunction) \
           * f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0, 
                 afac=ket_afac)
    return prod / np.abs( prod )


# Metropolis
if input_Rs_database == "":
    met_time = time.time()
    R0 = onp.random.normal(size=(N_coord,3))/np.mean(absmasses)
    # set center of mass position to 0
    R0 -= onp.transpose(onp.transpose(onp.mean(onp.transpose(
              onp.transpose(R0)*absmasses), axis=0, keepdims=True))
            /onp.mean(absmasses))
    print("R0 = ", R0)
    print("NCOORD = ", N_coord)
    print("f_R_braket(R0) = ", f_R_braket(R0))

    #FOR PERM TEST

    # TODO ACTUALLY CHANNGE BACK PLEASE
    if color == "6x6bar" or color == "SSS":
        samples = adl.metropolis(R0, f_R_braket, n_therm=50*n_skip, 
                                 n_step=n_walkers, n_skip=n_skip, 
                                 eps=4*2*a0*afac/N_coord**2*radial_n, 
                                 masses=absmasses)
    else:
        samples = adl.metropolis(R0, f_R_braket, n_therm=50*n_skip, 
                                 n_step=n_walkers, n_skip=n_skip, 
                                 eps=4*2*a0/N_coord**2*radial_n, 
                                 masses=absmasses)


    fac_list = [1/2, 1.0, 2]
    streams = len(fac_list)
    R0_list = [onp.random.normal(size=(N_coord,3)) for s in range(0,streams)]
    for s in range(streams):
        R0_list[s] -= onp.mean(R0_list[s], axis=0, keepdims=True)

    print("first walker")
    print("R = ",samples[0])
    print(f"metropolis in {time.time() - met_time} sec")
    Rs_metropolis = np.array([R for R,_ in samples])
else:
    f = h5py.File(input_Rs_database, 'r')
    Rs_metropolis = f["Rs"][-1]

print("spin-flavor wavefunction shape = ", S_av4p_metropolis_set[0].shape)
S_av4p_metropolis_norm = adl.inner(S_av4p_metropolis_set[0], S_av4p_metropolis_set[0])
print("spin-flavor wavefunction normalization = ", S_av4p_metropolis_norm)
assert (np.abs(S_av4p_metropolis_norm - 1.0) < 1e-6).all()

# trivial contour deformation
deform_f = lambda x, params: x
params = (np.zeros((n_step+1)),)


def trial_wvfn(R):
    psi = np.zeros((n_walkers,)  + (NI, NS) * N_coord, dtype=np.complex128)
    for ii in range(len(perms)):
        psi += antisym_factors[ii]*np.einsum("i,i...->i...", 
                  f_R(R,perms[ii],wavefunction=bra_wavefunction), 
                  S_av4p_metropolis_set[ii])
    return psi

print('Running GFMC evolution:')

#TODO: DEFINE SUM(S_av4p_metropolis_I*F_R_I) AND F_R=1
#WE WANT SUM_perms(S[perm]*f_R[perm])
if n_step > 0:
    rand_draws = onp.random.random(size=(n_step, Rs_metropolis.shape[0]))
    gfmc = adl.gfmc_deform(
        Rs_metropolis, trial_wvfn, params,
        rand_draws=rand_draws, tau_iMev=tau_iMev, 
        N=n_step, potential=Coulomb_potential,
        deform_f=deform_f, m_Mev=np.abs(np.array(masses)),
        resampling_freq=resampling)
    gfmc_Rs = np.array([Rs for Rs,_,_,_, in gfmc])
    gfmc_Ws = np.array([Ws for _,_,_,Ws, in gfmc])
    gfmc_Ss = np.array([Ss for _,_,Ss,_, in gfmc])
else:
    gfmc_Rs = np.array([Rs_metropolis])
    gfmc_Ws = np.array([0*Rs_metropolis[:,1,1]+1])
    gfmc_Ss = np.array([S_av4p_metropolis_set])

phase_Ws = f_R_braket_phase(gfmc_Rs)
print('phase Ws', phase_Ws)

gfmc_Ws= np.einsum('nk...,nk->nk...', gfmc_Ws, phase_Ws)

print('GFMC tau=0 weights:', gfmc_Ws[0])
if n_step > 0:
    print('GFMC tau=dtau weights:', gfmc_Ws[1])

# measure H
print('Measuring <H>...')

#TODO: DEFINE SUM(S_av4p_metropolis_I*LAPLACIAN_F_R_I) AND F_R=1

Ks = []
#for R in tqdm.tqdm(gfmc_Rs):
for count, R in enumerate(gfmc_Rs):
    K_time = time.time()
    if N_coord != 6:
       print('Calculating Laplacian for step ', count)
       K_time = time.time()
       K_term = -1/2*laplacian_f_R(R) / f_R(R, wavefunction=bra_wavefunction)

    if N_coord == 4 and abs(g) > 0:
        R_T = R
        R_T = R_T.at[...,1,:].set(R[...,swapI,:])
        R_T = R_T.at[...,swapI,:].set(R[...,1,:])

        K_term += -1/2*laplacian_f_R(R_T, afac=afac*gfac, a0=a0/gfac) \
                  / f_R(R_T, wavefunction=bra_wavefunction) * g
        Ks.append(K_term)

    if N_coord == 4 or N_coord == 6:
        total_wvfn = np.zeros((Rs_metropolis.shape[0],) + (NI, NS) * N_coord, 
                              dtype=np.complex128)
        total_lap = np.zeros((Rs_metropolis.shape[0],)  + (NI, NS) * N_coord, 
                              dtype=np.complex128)
        for ii in range(len(perms)):
            test=antisym_factors[ii]*np.einsum("i,i...->i...", 
                  f_R(R,perms[ii],wavefunction=bra_wavefunction), 
                  S_av4p_metropolis_set[ii])
            total_wvfn += antisym_factors[ii]*np.einsum("i,i...->i...", 
                            f_R(R,perms[ii],wavefunction=bra_wavefunction), 
                            S_av4p_metropolis_set[ii])
            total_lap += antisym_factors[ii]*np.einsum("i,i...->i...", 
                            laplacian_f_R(R,perms[ii],
                              wavefunction=bra_wavefunction), 
                            S_av4p_metropolis_set[ii])
        numerator = adl.inner(total_wvfn,total_lap)
        denominator = adl.inner(total_wvfn,total_wvfn)
        K_term = -1/2*numerator/denominator
        norm = adl.inner(total_wvfn,total_wvfn)
        print("NORM=",norm)
        Ks.append(K_term)

    print(f"calculated kinetic in {time.time() - K_time} sec")
Ks = np.array(Ks)

#TODO: DEFINE SUM(S_av4p_metropolis_I*F_R_I) AND F_R=1

Vs = []
for count, R in enumerate(gfmc_Rs):
    print('Calculating potential for step ', count)
    V_time = time.time()
    S = gfmc_Ss[count]
    print("R shape = ",R.shape)
    V_SI, V_SD = Coulomb_potential(R)
    print("shape of VSI = ",V_SI.shape)
    print("shape of VSD = ",V_SD.shape)
    broadcast_SI = ((slice(None),) + (np.newaxis,)*N_coord*2)

    if N_coord == 4 or N_coord == 6:
        total_wvfn = np.zeros((Rs_metropolis.shape[0],) + (NI, NS) * N_coord, 
                      dtype=np.complex128)
        for ii in range(len(perms)):
            total_wvfn += antisym_factors[ii]*np.einsum("i,i...->i...", 
                            f_R(R,perms[ii],wavefunction=bra_wavefunction), 
                            S_av4p_metropolis_set[ii])
        print("total wvfn = ",total_wvfn.shape)
        numerator = adl.inner(total_wvfn,
                        adl.batched_apply(V_SD + V_SI,total_wvfn))
        denominator = adl.inner(total_wvfn,total_wvfn)
        V_tot= numerator/denominator
    else:
        V_tot = adl.inner(S_av4p_metropolis, V_SD_S + V_SI_S) \
                  / adl.inner(S_av4p_metropolis, S)
    print(f"calculated potential in {time.time() - V_time} sec")
    Vs.append(V_tot)

Vs = np.array(Vs)

print(Vs.shape)

volume_string = ""
if volume == "finite":
  volume_string = "_L" + str(L)

tag = str(OLO) + "_dtau" + str(dtau_iMev) + "_Nstep"+str(n_step) \
      + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) \
      + "_nskip" + str(n_skip) + "_Nf"+str(nf) + "_alpha"+str(alpha) \
      + "_spoila"+str(spoila) + "_log_mu_r"+str(log_mu_r) \
      + "_wavefunction_"+str(wavefunction) + "_potential_"+str(potential) \
      + volume_string + "_afac"+str(afac)+"_masses"+str(masses_print) \
      +"_color_"+color+"_g"+str(g)+"_ferm_symm"+str(ferm_symm)

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

if verbose:
    last_point = n_walkers//8
    if last_point > 50:
        last_point = 50
    def tauint(t, littlec):
         return 1 + 2 * np.sum(littlec[1:t])
    tau_ac=0
    sub_dset = np.real(dset[tau_ac] - np.mean(dset[tau_ac]))
    auto_corr = []
    c0 = np.mean(sub_dset * sub_dset)
    print("sub_dset = ", sub_dset)
    print("c0 = ", c0)
    auto_corr.append(c0)
    for i in range(1,2*last_point):
        auto_corr.append(np.mean(sub_dset[i:] * sub_dset[:-i]))
    littlec = np.asarray(auto_corr) / c0
    print("tau = ", tau_ac)
    print("integrated autocorrelation time = ", tauint(last_point, littlec))

    Hs = np.array([al.bootstrap(K + V, W, Nboot=100, f=adl.rw_mean)
            for K,V,W in zip(Ks, Vs, gfmc_Ws)])

    Hs_opt = np.array([al.bootstrap(-V**2/(4*K), W, Nboot=100, f=adl.rw_mean)
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
    print("H_opt=",Hs_opt,"\n\n")
    print("K=",ave_Ks,"\n\n")
    print("V=",ave_Vs,"\n\n")
