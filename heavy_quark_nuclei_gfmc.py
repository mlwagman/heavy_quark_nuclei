### Eval script for deuteron GFMC deformation.

import argparse
import analysis as al
import getpass
import matplotlib.pyplot as plt
import numpy as np
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
import sys
from itertools import repeat
import time
import h5py
import math
import mpmath
from functools import partial

from sympy import simplify, summation, sqrt, Eq, Integral, oo, pprint, symbols, Symbol, log, exp, diff, Sum, factorial, IndexedBase, Function, cos, sin, atan, acot, pi, atan2, trigsimp, lambdify, re, im
from sympy.physics.hydrogen import R_nl, Psi_nlm

from itertools import permutations
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)

paper_plt.load_latex_config()

parser = argparse.ArgumentParser()
parser.add_argument('--n_walkers', type=int, default=1000)
parser.add_argument('--dtau_iMev', type=float, required=True)
parser.add_argument('--n_step', type=int, required=True)
parser.add_argument('--n_skip', type=int, default=200)
parser.add_argument('--resampling', type=int, default=None)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--mufac', type=float, default=1.0)
parser.add_argument('--Nc', type=int, default=2)
parser.add_argument('--N_coord', type=int, default=2)
parser.add_argument('--nf', type=int, default=5)
parser.add_argument('--OLO', type=str, default="LO")
parser.add_argument('--spoila', type=int, default=1)
parser.add_argument('--spoilf', type=str, default="hwf")
parser.add_argument('--outdir', type=str, required=True)
globals().update(vars(parser.parse_args()))

#######################################################################################

CF = (Nc**2 - 1)/(2*Nc)
VB = alpha*CF
if N_coord > 2:
    VB = alpha*CF/(Nc-1)
SingC3 = -(Nc+1)/8
cutoff = 1;
a0 = spoila*2/VB;

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
    if spoilf == "hwf":
    	for i in range(N_coord):
       		for j in range(N_coord):
            		if i!=j and j>=i:
                		Chi = Chi*exp(-rrSpher(i,j,r,t,p)/A[0])
    elif spoilf == "gauss":
    	for i in range(N_coord):
        	for j in range(N_coord):
            		if i!=j and j>=i:
                		Chi = Chi*exp(-1/2*(rrSpher(i,j,r,t,p)/A[0])**2)
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
    #for i in tqdm.tqdm(range(-n_therm, N_walkers*n_skip)):
    for i in range(-n_therm, N_walkers*n_skip): # update
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
            print(f'walker {this_walker}, iteration {i+1}')
            print(f'acc frac = {acc / (i+1)} \n')
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
alpha4 = float(mpmath.polylog(4,1/2))*0+(-jax.numpy.log(2))**4/(4*3*2*1)
ss6 = zeta51+zeta6
aa30 = dFA*( np.pi**2*( 7432/9-4736*alpha4+jax.numpy.log(2)*(14752/3-3472*zeta3)-6616*zeta3/3)  +  np.pi**4*(-156+560*jax.numpy.log(2)/3+496*jax.numpy.log(2)**2/3)+1511*np.pi**6/45)  + Nc**3*(385645/2916 + np.pi**2*( -953/54 +584/3*alpha4 +175/2*zeta3 + jax.numpy.log(2)*(-922/9+217*zeta3/3) ) +584*zeta3/3 + np.pi**4*( 1349/270-20*jax.numpy.log(2)/9-40*jax.numpy.log(2)**2/9 ) -1927/6*zeta5 -143/2*zeta3**2-4621/3024*np.pi**6+144*ss6  )
aa31 = dFF*( np.pi**2*(1264/9-976*zeta3/3+jax.numpy.log(2)*(64+672*zeta3)) + np.pi**4*(-184/3+32/3*jax.numpy.log(2)-32*jax.numpy.log(2)**2) +10/3*np.pi**6 ) + CF**2/2*(286/9+296/3*zeta3-160*zeta5)+Nc*CF/2*(-71281/162+264*zeta3+80*zeta5)+Nc**2/2*(-58747/486+np.pi**2*(17/27-32*alpha4+jax.numpy.log(2)*(-4/3-14*zeta3)-19/3*zeta3)-356*zeta3+np.pi**4*(-157/54-5*jax.numpy.log(2)/9+jax.numpy.log(2)**2)+1091*zeta5/6+57/2*zeta3**2+761*np.pi**6/2520-48*ss6)
aa32 = Nc/4*(12541/243+368/3*zeta3+64*np.pi**4/135)+CF/4*(14002/81-416*zeta3/3)
aa33 = -(20/9)**3*1/8
aa3 = aa30+aa31*nf+aa32*nf**2+aa33*nf**3

@partial(jax.jit)
def V3(r1, r2):
   R = lambda x, y: x*r1 - y*r2
   r1_norm = adl.norm_3vec(r1)
   r1_hat = r1 / r1_norm[...,jax.numpy.newaxis]
   r2_norm = adl.norm_3vec(r1)
   r2_hat = r2 / r2_norm[...,jax.numpy.newaxis]
   r1_hat_dot_r2_hat = jax.numpy.sum(r1_hat*r2_hat, axis=-1)
   R_norm = lambda x, y: adl.norm_3vec(R(x,y))
   R_hat = lambda x, y: R(x,y) / R_norm(x,y)[...,jax.numpy.newaxis]
   r1_hat_r2_hat_dot_R_R = lambda x, y: jax.numpy.sum(r1_hat*R_hat(x,y), axis=-1)*jax.numpy.sum(r2_hat*R_hat(x,y), axis=-1)
   A = lambda x, y: r1_norm * jax.numpy.sqrt(x*(1-x)) + r2_norm*jax.numpy.sqrt(y*(1-y))

   V3_integrand = lambda x, y: 16*jax.numpy.pi*( jax.numpy.arctan2(R_norm(x,y),A(x,y))*r1_hat_dot_r2_hat*1/R_norm(x,y)*(-1*A(x,y)**2/R_norm(x,y)**2+1) + r1_hat_dot_r2_hat*A(x,y)/R_norm(x,y)**2
           + jax.numpy.arctan2(R_norm(x,y),A(x,y))*r1_hat_r2_hat_dot_R_R(x,y)*1/R_norm(x,y)*(3*A(x,y)**2/R_norm(x,y)**2+1) - 3*r1_hat_r2_hat_dot_R_R(x,y)*A(x,y)/R_norm(x,y)**2)

   int_points = 100
   dx = 1/int_points
   x_grid = jax.numpy.arange(dx, stop=1, step=dx)
   y_grid = jax.numpy.arange(0, stop=1, step=dx)
   #V3_grid = jax.numpy.transpose(jax.numpy.array([[V3_integrand(x,y) for y in y_grid] for x in x_grid]), (2,0,1))

   y_vmap_f = jax.vmap(V3_integrand, (None, 0))
   xy_vmap_f = jax.vmap(y_vmap_f, (0, None))
   V3_grid = jax.numpy.transpose( xy_vmap_f(x_grid, y_grid), (2,0,1))

   V3_integral = jax.numpy.trapz( jax.numpy.trapz(V3_grid, dx=1/int_points), dx=1/int_points)

   return V3_integral

Rprime = lambda R: adl.norm_3vec(R)*jax.numpy.exp(np.euler_gamma)*mu
# build Coulomb potential
AV_Coulomb = {}
B3_Coulomb = {}
if OLO == "LO":
	AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)
elif OLO == "NLO":
	AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1))
elif OLO == "NNLO":
    if N_coord > 2:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 + 2*Nc/(Nc-1)*((np.pi)**4-12*(np.pi)**2) ) )
    else:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 ) )
    B3_Coulomb['O1'] = lambda Rij, Rjk, Rik: SingC3*2*alpha*(alpha/(4*np.pi))**2*(V3(Rij, Rjk) + V3(Rjk, Rik) + V3(Rik, Rij))
elif OLO == "N3LO":
    if N_coord > 2:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 + 2*Nc/(Nc-1)*((np.pi)**4-12*(np.pi)**2) ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*jax.numpy.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( jax.numpy.log(Rprime(R))**3 + np.pi**4/4*jax.numpy.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(jax.numpy.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*jax.numpy.log(Rprime(R)) )
    else:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*jax.numpy.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( jax.numpy.log(Rprime(R))**3 + np.pi**4/4*jax.numpy.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(jax.numpy.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*jax.numpy.log(Rprime(R)) )
    B3_Coulomb['O1'] = lambda Rij, Rjk, Rik: SingC3*2*alpha*(alpha/(4*np.pi))**2*(V3(Rij, Rjk) + V3(Rjk, Rik) + V3(Rik, Rij))
elif OLO == "mNLO":
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1)) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
elif OLO == "mNNLO":
    if N_coord > 2:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 + 2*Nc/(Nc-1)*((np.pi)**4-12*(np.pi)**2) ) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
    else:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 ) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
    B3_Coulomb['O1'] = lambda Rij, Rjk, Rik: SingC3*2*alpha*(alpha/(4*np.pi))**2*(V3(Rij, Rjk) + V3(Rjk, Rik) + V3(Rik, Rij))
elif OLO == "mN3LO":
    if N_coord > 2:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 + 2*Nc/(Nc-1)*((np.pi)**4-12*(np.pi)**2) ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*jax.numpy.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( jax.numpy.log(Rprime(R))**3 + np.pi**4/4*jax.numpy.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(jax.numpy.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*jax.numpy.log(Rprime(R)) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
    else:
        AV_Coulomb['O1'] = lambda R: -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*jax.numpy.log(Rprime(R))+aa1) + (alpha/(4*np.pi))**2*( beta0**2*(4*jax.numpy.log(Rprime(R))**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*jax.numpy.log(Rprime(R))+ aa2 ) ) + (alpha/(4*np.pi))**3*( 64*np.pi**2/3*Nc**3*jax.numpy.log(adl.norm_3vec(R)) + aa3 + 64*np.pi**2/3*Nc**3*np.euler_gamma + 512*beta0**3*( jax.numpy.log(Rprime(R))**3 + np.pi**4/4*jax.numpy.log(Rprime(R))+2*zeta3 ) + (640*beta0*beta1 + 192*beta0**2*aa1)*(jax.numpy.log(Rprime(R))**2+np.pi**2/12) + (128*beta2+64*beta1*aa1+24*beta0*aa2)*jax.numpy.log(Rprime(R)) ) -1*CF*Nc*alpha**2/(N_coord-1)/(adl.norm_3vec(R)**2)
else:
	print("order not supported")
	throw(0)

Coulomb_potential = adl.make_pairwise_potential(AV_Coulomb, B3_Coulomb)

# build Coulomb ground-state trial wavefunction
trial_wvfn = wvfn()
print(trial_wvfn.A)
f_R = lambda R: trial_wvfn.psi(torch.from_numpy(np.asarray(R))).detach().numpy()

laplacian_f_R = lambda R: trial_wvfn.laplacian(torch.from_numpy(np.asarray(R))).detach().numpy()

# Metropolis
Rs_metropolis = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=n_walkers, n_skip=n_skip, eps=2*trial_wvfn.A[0].item()/N_coord**2)[0]
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
#for R in tqdm.tqdm(gfmc_Rs):
for count, R in enumerate(gfmc_Rs):
    print('Calculating Laplacian for step ', count)
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
for count, R in enumerate(gfmc_Rs):
    print('Calculating potential for step ', count)
    V_time = time.time()
    VSI,_ = Coulomb_potential(R)
    V_ind = (slice(0,None),) + (0,)*NS*NI*N_coord
    print(f"calculated potential in {time.time() - V_time} sec")
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

tag = str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_Nf"+str(nf) + "_alpha"+str(alpha) + "_spoila"+str(spoila) + "_spoilf"+str(spoilf)

with h5py.File(outdir+'Hammys_'+tag+'.h5', 'w') as f:
    dset = f.create_dataset("Hammys", data=Ks+Vs)

with h5py.File(outdir+'Rs_'+tag+'.h5', 'w') as f:
    dset = f.create_dataset("Rs", data=gfmc_Rs)


with h5py.File(outdir+'Hammys_'+tag+'.h5', 'r') as f:
    data = f['Hammys']
    print(data)

with h5py.File(outdir+'Rs_'+tag+'.h5', 'r') as f:
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

plt.savefig(outdir+'Hammy_gfmc_plot_'+tag+'.pdf')
