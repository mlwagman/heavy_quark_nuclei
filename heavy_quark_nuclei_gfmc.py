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
from itertools import repeat
import time
import h5py
import math

from sympy import simplify, summation, sqrt, Eq, Integral, oo, pprint, symbols, Symbol, log, exp, diff, Sum, factorial, IndexedBase, Function, cos, sin, atan, acot, pi, atan2, trigsimp, lambdify, re, im
from sympy.physics.hydrogen import R_nl, Psi_nlm

from itertools import permutations
import torch
import torch.nn as nn

paper_plt.load_latex_config()

parser = argparse.ArgumentParser()
parser.add_argument('--n_walkers', type=int, default=1000)
parser.add_argument('--dtau_iMev', type=float, required=True)
parser.add_argument('--n_step', type=int, required=True)
parser.add_argument('--n_skip', type=int, default=100)
parser.add_argument('--resampling', type=int, default=None)
parser.add_argument('--VB', type=float, default=1)
parser.add_argument('--N_coord', type=int, default=2)
parser.add_argument('--a0', type=float, default=2)
globals().update(vars(parser.parse_args()))

#######################################################################################
cutoff = 1;
a0 = 2/VB;

rr = np.full((N_coord,N_coord), fill_value = '',dtype=object)
for i in range(N_coord):
    for j in range(N_coord):
        rr[i][j] = Symbol('rr[{},{}]'.format(i, j))
tt = np.full((N_coord,N_coord), fill_value = '',dtype=object)
for i in range(N_coord):
    for j in range(N_coord):
        tt[i][j] = Symbol('tt[{},{}]'.format(i, j))
pp = np.full((N_coord,N_coord), fill_value = '',dtype=object)
for i in range(N_coord):
    for j in range(N_coord):
        pp[i][j] = Symbol('pp[{},{}]'.format(i, j))

C = symbols('C0:%d'%N_coord, real=True);
A = symbols('A0:%d'%N_coord, Positive=True);

# Define spherical coords
r = symbols('r0:%d'%N_coord, positive=True);
t = symbols('t0:%d'%N_coord, real=True);
p = symbols('p0:%d'%N_coord, real=True);

# Define cartesian coords
x = symbols('x0:%d'%N_coord, real=True);
y = symbols('y0:%d'%N_coord, real=True);
z = symbols('z0:%d'%N_coord, real=True);

# Define color vector
v = symbols('v0:%d'%N_coord);

#   Potential coupling and a0
#a, Z = symbols("a Z", positive=True)
B = symbols("B", real=True)
a = symbols("a", real=True)

def laPlaceSpher(f,r,t,p):
    dfdr = diff(f,r)
    rpart = 1/r**2*diff(r**2*(diff(f,r)),r)
    tpart = 1/(r**2*sin(t))*diff(sin(t)*(diff(f,t)),t)
    ppart = 1/(r**2*sin(t)**2)*diff(diff(f,p),p)
    nabla = rpart + tpart + ppart
    return nabla

def rInv(i,j):
    return 1/sqrt(r[i]**2 + r[j]**2 - 2*r[j]*r[i]*sin(t[i])*sin(t[j])*cos(p[i]-p[j]) - 2*r[j]*r[i]*cos(t[i])*cos(t[j]))


def Potential(rr,B,N_coord):
    V0 = -B*sum(o for i, a in enumerate(rr) for j, o  in enumerate(a) if i!=j and j>=i);
    for i in range(N_coord):
        for j in range(N_coord):
            if i!=j and j>=i:
                V0 = V0.subs(rr[i][j],rInv(i,j))
    return V0

# Define HWF quantum numbers
n, l, m, phi, theta, Z = symbols("n l m phi theta Z")

# Spher (i,j) to Cart(i),Cart(j) to Spher(i),Spher(j)
def rrSpher(i,j,r,t,p):
    return  sqrt(r[j]*(r[j]-2*r[i]*(sin(t[i])*sin(t[j])*cos(p[i]-p[j])+cos(t[i])*cos(t[j]))) +r[i]**2);
def ttSpher(i,j,r,t,p):
    # experimentally this works assuming atan2(y, x) = atan(y / x) + phase
    return atan2(sqrt((cos(p[i])*r[i]*sin(t[i]) - cos(p[j])*r[j]*sin(t[j]))**2 + (r[i]*sin(t[i])*sin(p[i]) - r[j]*sin(t[j])*sin(p[j]))**2), (cos(t[i])*r[i] - cos(t[j])*r[j]));
def ppSpher(i,j,r,t,p):
    return  atan2((r[i]*sin(t[i])*sin(p[i]) - r[j]*sin(t[j])*sin(p[j])), (cos(p[i])*r[i]*sin(t[i]) - cos(p[j])*r[j]*sin(t[j])));

print(simplify(rrSpher(1,1,r,t,p)))

#  Define chi(r_i) where psi(r1,..,rn)=chi(r1)*...*chi(rn)
def Chi(k, N_coord, n, l, m, Z, r, t, p, v, col):
     Chi =  0
     for j in range(0,N_coord):
         if k!=j and j>=k:
             Chi = Chi + v[col]*1/(N_coord-1)*Psi_nlm(n, l, m, rrSpher(k,j,r,t,p), ppSpher(k,j,r,t,p), ttSpher(k,j,r,t,p), Z)
         elif k!=j and k>=j:
             Chi = Chi + v[col]*1/(N_coord-1)*Psi_nlm(n, l, m, rrSpher(j,k,r,t,p), ppSpher(j,k,r,t,p), ttSpher(j,k,r,t,p), Z)
         else:
             Chi = Chi
     return Chi

def Chi_no_v_test(N_coord, r, t, p, C, A):
    if (N_coord == 2):
        Chi = C[0]*exp(-rrSpher(0,1,r,t,p)/A[0])
    elif (N_coord == 3):
        Chi = C[0]*exp(-(rrSpher(0,1,r,t,p)+ rrSpher(0,2,r,t,p) + rrSpher(1,2,r,t,p))/A[0])
    elif (N_coord == 4):
        Chi = C[0]*exp(-(rrSpher(0,1,r,t,p)+ rrSpher(0,2,r,t,p) + rrSpher(0,3,r,t,p)
        + rrSpher(1,2,r,t,p)+ rrSpher(1,3,r,t,p) + rrSpher(2,3,r,t,p))/A[0])
        + C[1]*(exp(-(rrSpher(0,1,r,t,p))/A[1]) + exp(-(rrSpher(0,2,r,t,p))/A[1]) + exp(-(rrSpher(0,3,r,t,p))/A[1])
        + exp(-(rrSpher(1,2,r,t,p))/A[1])  + exp(-(rrSpher(1,3,r,t,p))/A[1])  + exp(-(rrSpher(2,3,r,t,p))/A[1]))
    else:
        Chi=1
    return Chi

def Chi_no_v(N_coord, r, t, p, C, A):
    Chi = 1;
    for i in range(N_coord):
        for j in range(N_coord):
            if i!=j and j>=i:
                Chi = Chi*exp(-rrSpher(i,j,r,t,p)/A[0])
    return C[0]*Chi

print(simplify(Chi_no_v_test(N_coord, r, t, p, C, A)))

#  Define psi(r1,..,rn)=chi(r1)*...*chi(rn)

def psi_no_v(N_coord, r, t, p, C, A):
    psi = Chi_no_v(N_coord, r, t, p, C, A)
    psi = psi.rewrite(cos)
    modules = {'sin': math.sin, 'cos': math.cos} #, 're': torch.real, 'im': torch.imag
    return lambdify([C, A, r, t, p], psi, modules)

def nabla_psi_no_v(N_coord, r, t, p, C, A):
    psi = Chi_no_v(N_coord, r, t, p, C, A)
    nabla_wvfn = 0.0j
    for a in range(N_coord):
        nabla_wvfn += laPlaceSpher(psi, r[a], t[a], p[a])
    nabla_wvfn = nabla_wvfn.rewrite(cos)
    modules = {'sin': math.sin, 'cos': math.cos}
    return lambdify([C, A, r, t, p], nabla_wvfn, modules)

#######################################################################################
N_skip = 10
N_refresh_metropolis = 1
patience_factor = 10

print(f'precomputing wavefunctions')
psi_time = time.time()
#psitab = []
#psitab.append(psi_no_v(N_coord, r, t, p, C, A))
psitab = psi_no_v(N_coord, r, t, p, C, A)
print(f"precomputed wavefunctions in {time.time() - psi_time} sec")

print(f'precomputing wavefunction Laplacians')
nabla_psi_time = time.time()
#nabla_psitab = []
#nabla_psitab.append(nabla_psi_no_v(N_coord, r, t, p, C, A))
nabla_psitab = nabla_psi_no_v(N_coord, r, t, p, C, A)
print(f"precomputed wavefunction Laplacians in {time.time() - nabla_psi_time} sec")

def total_Psi_nlm(Rs, A_n, C_n, psi_fn):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    # convert to spherical
    x = Rs[:,:,0]
    y = Rs[:,:,1]
    z = Rs[:,:,2]
    r_n = torch.sqrt(x**2 + y**2 + z**2)
    t_n = torch.atan2(torch.sqrt(x**2 + y**2), z)
    p_n = torch.atan2(y, x)
    # evaluate wavefunction
    for i in range(N_walkers):
       Psi_nlm_s[i] = psi_fn(C_n, A_n, r_n[i], t_n[i], p_n[i])
    return Psi_nlm_s

def nabla_total_Psi_nlm(Rs, A_n, C_n, nabla_psi_fn):
    nabla_psi_time = time.time()
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    nabla_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    # convert to spherical
    x = Rs[:,:,0]
    y = Rs[:,:,1]
    z = Rs[:,:,2]
    r_n = torch.sqrt(x**2 + y**2 + z**2)
    t_n = torch.atan2(torch.sqrt(x**2 + y**2), z)
    p_n = torch.atan2(y, x)
    # evaluate wavefunction
    for i in range(N_walkers):
        nabla_Psi_nlm_s[i] = nabla_psi_fn(C_n, A_n, r_n[i], t_n[i], p_n[i])
    print(f"calculated nabla in {time.time() - nabla_psi_time} sec")
    return nabla_Psi_nlm_s

def potential_no_Psi_nlm(Rs, A_n, C_n, psi_fn):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    V_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    wvfn = total_Psi_nlm(Rs, A_n, C_n, psi_fn)
    for i in range(N_walkers):
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        # evaluate potential
        V = 0
        for a in range(N_coord):
            for b in range(N_coord):
                if b > a:
                    V += -VB/np.sqrt( (x[a]-x[b])**2 + (y[a]-y[b])**2 + (z[a]-z[b])**2 )
        V_Psi_nlm_s[i] = V
    return V_Psi_nlm_s

def potential_total_Psi_nlm(Rs, A_n, C_n, psi_fn):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    V_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    wvfn = total_Psi_nlm(Rs, A_n, C_n, psi_fn)
    for i in range(N_walkers):
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        # evaluate potential
        V = 0
        for a in range(N_coord):
            for b in range(N_coord):
                if b > a:
                    V += -VB/np.sqrt( (x[a]-x[b])**2 + (y[a]-y[b])**2 + (z[a]-z[b])**2 )
        V_Psi_nlm_s[i] = V * wvfn[i]
    return V_Psi_nlm_s

def K_Psi_nlm(Rs, A, C, nabla_psi_fn):
    K_psi = -1/2*nabla_total_Psi_nlm(Rs, A, C, nabla_psi_fn)
    return K_psi


def V_Psi_nlm(Rs, A, C, psi_fn):
    V_psi = potential_total_Psi_nlm(Rs, A, C, psi_fn)
    return V_psi
atan2
def hammy_Psi_nlm(Rs, A, C, psi_fn, nabla_psi_fn):
    K_psi = K_Psi_nlm(Rs, A, C, nabla_psi_fn)
    V_psi = V_Psi_nlm(Rs, A, C, psi_fn)
    H_psi = K_psi + V_psi
    return H_psi


def draw_coordinates(shape, *, eps=1.0, axis=1):
    dR = eps/np.sqrt(2) * torch.normal(torch.ones(shape))
    # subtract mean to keep center of mass fixed
    dR -= torch.mean(dR, axis=axis, keepdims=True)
    return dR

def metropolis_coordinate_ensemble(this_psi, *, n_therm, N_walkers, n_skip, eps):
    # array of walkers to be generated
    Rs = torch.zeros((N_walkers, N_coord, 3))
    psi2s = torch.zeros((N_walkers))
    this_walker = 0
    # store acceptance ratio
    acc = 0
    # initial condition to start metropolis
    R = torch.normal(torch.ones((1,N_coord,3)))
    # set center of mass position to 0
    R -= torch.mean(R, axis=1, keepdims=True)
    # metropolis updates
    print("Running Metropolis")
    for i in tqdm.tqdm(range(-n_therm, N_walkers*n_skip)):
        # update
        dR = draw_coordinates(R.shape, eps=eps, axis=1)
        new_R = R + dR
        # accept/reject based on |psi(R)|^2
        abspsi = torch.abs(this_psi(R))
        p_R = abspsi**2
        abspsi_new = torch.abs(this_psi(new_R))
        p_new_R = abspsi_new**2
        if (torch.rand(1) < (p_new_R / p_R) and not torch.isnan(p_new_R) and p_new_R > 0 and p_new_R < 1 ):
            R = new_R #accept
            p_R = p_new_R
            if i >= 0:
                acc += 1
        # store walker every skip updates
        if i >= 0 and (i+1) % n_skip == 0:
            Rs[this_walker,:,:] = R
            psi2s[this_walker] = p_R
            this_walker += 1
            #print(f'iteration {i+1}')
            #print(f'|psi(R)|^2 = {p_R}')
            #print(f'Total acc frac = {acc / (i+1)}')
    print(f'Total acc frac = {acc / (i+1)}')
    # return coordinates R and respective |psi(R)|^2
    return Rs, psi2s

class wvfn(nn.Module):
    # Psi(R) = sum_{n,l,m} c_{n,l,m} psi(n, l, m, R)
    def __init__(self):
        super(wvfn, self).__init__()
        # register Bohr radius a and c_{n,l,m,k,j} as pytorch paramters
        self.A = nn.Parameter(a0*torch.ones(N_coord, dtype=torch.double))
        self.C = nn.Parameter(torch.cat((
            torch.ones((N_coord-1), dtype=torch.complex64),
            torch.ones((1), dtype=torch.complex64))))
        #self.A = nn.Parameter(2/VB*torch.ones(1, dtype=torch.double))
        #self.C = nn.Parameter(torch.ones(1, dtype=torch.complex64))
    # For N_coord>1 C and A have Length N_coord not 1
    def psi(self, Rs):
        A_n=self.A
        C_n=self.C
        psi = total_Psi_nlm(Rs, A_n, C_n, psitab)
        return psi
    def psi2(self, Rs):
        return torch.pow(torch.abs(self.psi(Rs)), 2)
    def laplacian(self, Rs):
        A_n=self.A
        C_n=self.C
        return nabla_total_Psi_nlm(Rs, A_n, C_n, nabla_psitab)

    def coulPot(self, Rs):
        A_n=self.A
        C_n=self.C
        return potential_no_Psi_nlm(Rs, A_n, C_n, psitab)

    def hammy(self, Rs):
        A_n=self.A
        C_n=self.C
        H_psi = hammy_Psi_nlm(Rs, A_n, C_n, psitab, nabla_psitab)
        psistar = torch.conj(total_Psi_nlm(Rs, A_n, C_n, psitab))
        return psistar*H_psi
    def forward(self, Rs):
        A_n=self.A
        C_n=self.C
        H_psi = hammy_Psi_nlm(Rs, A_n, C_n, psitab, nabla_psitab)
        psistar = torch.conj(total_Psi_nlm(Rs, A_n, C_n, psitab))
        return psistar*H_psi / VB**2, torch.pow(torch.abs(psistar), 2)
#######################################################################################

#VB=.1
#print(VB)
#quit()
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
Rs_metropolis = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=n_walkers, n_skip=n_skip, eps=trial_wvfn.A[0].item()/N_coord**2)[0]
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

#with h5py.File('Hammys_'+"N_coord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+'.hdf5', 'w') as f:
#    dset = f.create_dataset("default", data=Hs)

with h5py.File('Hammys_'+"N_coord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+"_a0="+str(a0)+'.hdf5', 'w') as f:
    dset = f.create_dataset("Hammys", data=Ks+Vs)

with h5py.File('Rs_'+"N_coord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+"_a0="+str(a0)+'.hdf5', 'w') as f:
    dset = f.create_dataset("Rs", data=gfmc_Rs)


with h5py.File('Hammys_'+"N_coord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+"_a0="+str(a0)+'.hdf5', 'r') as f:
    data = f['Hammys']
    print(data)

with h5py.File('Rs_'+"N_coord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+"_a0="+str(a0)+'.hdf5', 'r') as f:
    data = f['Rs']
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

plt.savefig('Hammys_'+"N_coord="+str(N_coord)+"_B="+str(VB)+"_nStep="+str(n_step)+"_dtau="+str(dtau_iMev)+"_a0="+str(a0)+'.pdf')
