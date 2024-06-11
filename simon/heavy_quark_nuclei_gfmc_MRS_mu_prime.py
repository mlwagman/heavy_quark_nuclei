### Eval script for deuteron GFMC deformation.

import argparse
import analysis as al
import getpass
import matplotlib.pyplot as plt
import numpy as onp
import scipy
import scipy.special as spl ####### NEW
import scipy.interpolate
import scipy.integrate
from scipy.optimize import root_scalar ###### NEW
import jax.scipy
import jax.scipy.special
import pickle
import paper_plt
import tqdm.auto as tqdm
import afdmc_lib_col as adl
import os
import pickle
from afdmc_lib_col import NI,NS,mp_Mev,fm_Mev
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
#parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--mu', type=float, default=2.0) # or default=1.0
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
#parser.add_argument('--L_pert', type=int, default=3) ################## NEW set at 3 because we don't go to 3-loop calculation
parser.add_argument('--Rstar', type=float, default=0.35)  # or default=1.91 ?
parser.add_argument('--mQ', type=float, default=0.0)  ############################ NEW
parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
globals().update(vars(parser.parse_args()))

#######################################################################################
###################################################################################################################
Tf = 1/2
CA = Nc
CF = (Nc**2 - 1)/(2*Nc)
Pi = onp.pi
NA = Nc**2 - 1

# Zeta function using scipy
def Zeta(s):
    return spl.zeta(s, 1)  # '1' is the q parameter, for Riemann Zeta it's set to 1

###################################################################################################################
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
dAA = Nc**2 * (Nc**2 + 36) / 24 ############################### NEW
alpha4 = float(mpmath.polylog(4,1/2))+(-np.log(2))**4/(4*3*2*1) ############################ NEW, we remove the "*0" at the end of the first term
ss6 = zeta51+zeta6 ############ different from Andreas' orginal code (mistake from Andreas ??)
aa30 = dFA*( np.pi**2*( 7432/9-4736*alpha4+np.log(2)*(14752/3-3472*zeta3)-6616*zeta3/3)  +  np.pi**4*(-156+560*np.log(2)/3+496*np.log(2)**2/3)+1511*np.pi**6/45)  + Nc**3*(385645/2916 + np.pi**2*( -953/54 +584/3*alpha4 +175/2*zeta3 + np.log(2)*(-922/9+217*zeta3/3) ) +584*zeta3/3 + np.pi**4*( 1349/270-20*np.log(2)/9-40*np.log(2)**2/9 ) -1927/6*zeta5 -143/2*zeta3**2-4621/3024*np.pi**6+144*ss6  )
aa31 = dFF*( np.pi**2*(1264/9-976*zeta3/3+np.log(2)*(64+672*zeta3)) + np.pi**4*(-184/3+32/3*np.log(2)-32*np.log(2)**2) +10/3*np.pi**6 ) + CF**2/2*(286/9+296/3*zeta3-160*zeta5)+Nc*CF/2*(-71281/162+264*zeta3+80*zeta5)+Nc**2/2*(-58747/486+np.pi**2*(17/27-32*alpha4+np.log(2)*(-4/3-14*zeta3)-19/3*zeta3)-356*zeta3+np.pi**4*(-157/54-5*np.log(2)/9+np.log(2)**2)+1091*zeta5/6+57/2*zeta3**2+761*np.pi**6/2520-48*ss6)
aa32 = Nc/4*(12541/243+368/3*zeta3+64*np.pi**4/135)+CF/4*(14002/81-416*zeta3/3)
aa33 = -(20/9)**3*1/8
aa3 = aa30+aa31*nf+aa32*nf**2+aa33*nf**3
###################################################################################################################


def Beta(k):
    if k == 0:
        return 1 / (4 * Pi)**(k+1) * beta0
    elif k == 1:
        return 1 / (4 * Pi)**(k+1) * beta1
    elif k == 2:
        return 1 / (4 * Pi)**(k+1) * beta2
    elif k == 3:
        return 1 / (4 * Pi)**(k+1) * beta3
    else:
        None
        
        
def delta(k):
    return Beta(k) / Beta(0) - (Beta(1) / Beta(0))**k

#MS_bar SCHEME ONLY ! Wrong: are from 2008 and it is a numerical approximation. Should use https://arxiv.org/pdf/1608.02603.pdf
def a(k, rV=2/3): # What is rV ?
    if k == 0:
        return 1
    elif k == 1:
        return 1/(4 * Pi) * (31/9 * CA - 20/9 * Tf * nf)
    elif k == 2:
        return 1/(4 * Pi)**2 * ((4343/162 + 4 * Pi**2 - Pi**4/4 + 22/3 * Zeta(3)) * CA**2 -
      ((1798/81 + 56/3 * Zeta(3)) * CA + (55/3 - 16 * Zeta(3)) * (CA**2 - 1)/(2*CA)) * Tf * nf +
      (20/9 * Tf * nf)**2)
    elif k == 3:
        a_3_us = 1/(4 * Pi)**3 * (16/3 * (onp.log(CA * rV) + onp.euler_gamma - 5/6) * Pi**2 * CA**3)
        additional_term = 1/(4 * Pi)**3 * (
    (385645/2916 + 584/3 * Zeta(3) - 1927/6 * Zeta(5) - 143/2 * Zeta(3)**2 + 144 * ss6 -
     (953/54 - 175/2 * Zeta(3) + (922/9 - 217/3 * Zeta(3)) * onp.log(2) - 584/3 * alpha4) * Pi**2 +
     (1349/270 - 20/9 * onp.log(2) - 40/9 * onp.log(2)**2) * Pi**4 - 4621/3024 * Pi**6) * CA**3 +
    (7432/9 - 6616/3 * Zeta(3) + (14752/3 - 3472 * Zeta(3)) * onp.log(2) -
     4736 * alpha4 + Pi**2 * (-156 + 560/3 * onp.log(2) +
     496/3 * onp.log(2)**2) + Pi**4 * 1511/45) * Pi**2 * dFA -
    ((58747/486 + 356 * Zeta(3) - 1091/6 * Zeta(5) - 57/2 * Zeta(3)**2 + 48 * ss6 -
      (17/27 - 19/3 * Zeta(3) - 4/3 * onp.log(2) - 14 * onp.log(2) * Zeta(3) - 32 * alpha4) * Pi**2 +
      (157/54 + 5/9 * onp.log(2) - onp.log(2)**2) * Pi**4 - 761/2520 * Pi**6) * CA**2 +
     (71281/162 - 264 * Zeta(3) - 80 * Zeta(5)) * CA * CF -
     (286/9 + 296/3 * Zeta(3) - 160 * Zeta(5)) * CF**2) * Tf * nf +
    (1264/9 - 976/3 * Zeta(3) + 32 * (2 + 21 * Zeta(3)) * onp.log(2) -
     Pi**2 * (184/3 - 32/3 * onp.log(2) + 32 * onp.log(2)**2) + Pi**4 * 10/3) * Pi**2 * (CF) * nf * dFF -
    ((12541/243 + 368/3 * Zeta(3) + 64/135 * Pi**4) * CA +
     (14002/81 - 416/3 * Zeta(3)) * CF) * Tf**2 * nf**2 -
    (20/9 * Tf * nf)**3)
        return a_3_us + additional_term
    else:
        return None  # Undefined for other values
    
#MS_bar SCHEME ONLY !
def a(k, alpha=1): # rV means ratio of the potentiel V ?
    if k == 0:
        return 1
    elif k == 1:
        return 1/(4 * Pi) * aa1
    elif k == 2:
        return 1/(4 * Pi)**2 * aa2
    elif k == 3:
        #aa3_us = 16/3 * Pi**2 * CA**3 * (onp.log(CA * rV) + onp.euler_gamma - 5/6)
        aa3_us = 16/3 * Pi**2 * CA**3 * (onp.log(CA/2 * alpha) - 5/6 + onp.euler_gamma + onp.log(2))
        return 1/(4 * Pi)**3 * (aa3 + aa3_us)
    else:
        return None  # Undefined for other values

    
def L_fun(n):
    if n == 1:
        return onp.euler_gamma
    elif n == 2:
        return onp.euler_gamma**2 + 0.5 * Zeta(2)
    elif n == 3:
        return onp.euler_gamma**3 + 1.5 * Zeta(2) * onp.euler_gamma + 2 * Zeta(3)
    else:
        return None  # Undefined for other values

def v(k, o, alpha=1):
    if k == 0:
        return a(0)
    elif k == 1:
        return a(1) + 2 * a(0) * Beta(0) * L_fun(1)
    elif k == 2:
        singulet = a(2) + 2 * (2 * a(1) * Beta(0) + a(0) * Beta(1)) * L_fun(1) + a(0) * (2 * Beta(0))**2 * L_fun(2)
        if o == 'singulet':
            return singulet
        elif o == 'antisymmetric':
            return singulet + 1 / (4 * Pi)**2 * (Nc * (Nc - 2) / 2 * (Pi**4 - 12 * Pi**2))
        elif o == 'symmetric':
            return singulet + 1 / (4 * Pi)**2 * (Nc * (Nc + 2) / 2 * (Pi**4 - 12 * Pi**2))
        elif o == 'octet':
            return singulet + 1 / (4 * Pi)**2 * (CA**2 * (Pi**4 - 12 * Pi**2))
    elif k == 3:
        singulet = a(3, alpha) + 2 * (3 * a(2) * Beta(0) + 2 * a(1) * Beta(1) + a(0) * Beta(2)) * L_fun(1) + \
                   3 * a(1) * (2 * Beta(0))**2 * L_fun(2) + 10 * a(0) * Beta(0) * Beta(1) * L_fun(2) + \
                   a(0) * (2 * Beta(0))**3 * L_fun(3)
        if o == 'singulet':
            return singulet
        elif o == 'antisymmetric':
            return singulet #? need to be calculate
        elif o == 'symmetric':
            return singulet #? need to be calculate
        elif o == 'octet': # too complexe and we will not use it
            l2 = onp.log(2)
            daa3nf1 = CA * dFF * (Pi**2 * (88/9 - 32*l2/3 + 248*Zeta(3)/3 - 112*Zeta(3)*l2) + Pi**4 * (28/9 - 16*l2/9 + 16*l2**2/3) - 5*Pi**6/9) + dFA * (Pi**2 * (4/3 - 192*alpha4 - 16*l2/3 - 176*Zeta(3)/3 - 56*l2*Zeta(3)/3) + Pi**4 * (-1/54 + 5*l2/27 - 2*l2**2/9) - 23*Pi**6/270)
            daa3nf0 = CA * dFA * (Pi**2 * (-2356/9 + 3520*alpha4 - 7376*l2/3 + 1420*Zeta(3) + 1736*Zeta(3)*l2) + Pi**4 * (66 - 200*l2/3 - 248*l2**2/3) - 511*Pi**6/18) + dAA * (Pi**2 * (50/3 - 1184*alpha4/3 + 3688*l2/9 - 370*Zeta(3)/3 - 868*l2*Zeta(3)/3) + Pi**4 * (-197/9 + 140*l2/9 + 124*l2**2/9) + 1871*Pi**6/540) + CA**4 * (Pi**2 * (257/54 - 512*alpha4/9 + 922*l2/27 - 220*Zeta(3)/9 - 217*l2*Zeta(3)/9) + Pi**4 * (-25/54 + 20*l2/27 + 31*l2**2/27) + 2897 * Pi**6/6480)
            octet_us = 0#? need to be calculate
            return singulet + 1 / (4 * Pi)**3 * (daa3nf1/(CF - CA/2) * nf + daa3nf0/(CF - CA/2)) + octet_us
    else:
        return None  # Undefined for other values
    

def vl(l, s, o, alpha=1): # s is mu/Q or more precisely it is s*mu/Q when mu=Q
    if l == 0:
        return v(0, o)
    elif l == 1:
        return v(1, o) + 2 * v(0, o) * Beta(0) * onp.log(s)
    elif l == 2:
        return v(2, o) + 2 * (2 * v(1, o) * Beta(0) + v(0, o) * Beta(1)) * onp.log(s) + v(0, o) * (2 * Beta(0) * onp.log(s))**2
    elif l == 3:
        return v(3, o, alpha) + 3 * v(2, o) * (2 * Beta(0) * onp.log(s)) + \
               v(1, o) * (4 * Beta(1) * onp.log(s) + 3 * (2 * Beta(0) * onp.log(s))**2) + \
               v(0, o) * (2 * Beta(2) * onp.log(s) + 10 * Beta(0) * Beta(1) * onp.log(s)**2 + (2 * Beta(0) * onp.log(s))**3)
    else:
        return None  # Undefined for other values

    
def f(k, o, alpha=1):
    v_val = v(k, o, alpha)
    sum_part = sum(j * v(j - 1, o, alpha) * Beta(k - j) for j in range(1, k + 1))

    return v_val - 2 * sum_part


def fl(l, s, o, alpha=1):
    if l == 0:
        return f(0, o)
    elif l == 1:
        return f(1, o) + 2 * f(0, o) * Beta(0) * onp.log(s)
    elif l == 2:
        return f(2, o) + 2 * (2 * f(1, o) * Beta(0) + f(0, o) * Beta(1)) * onp.log(s) + f(0, o) * (2 * Beta(0) * onp.log(s))**2
    elif l == 3:
        return f(3, o, alpha) + 3 * f(2, o) * (2 * Beta(0) * onp.log(s)) + \
               f(1, o) * (4 * Beta(1) * onp.log(s) + 3 * (2 * Beta(0) * onp.log(s))**2) + \
               f(0, o) * (2 * Beta(2) * onp.log(s) + 10 * Beta(0) * Beta(1) * onp.log(s)**2 + (2 * Beta(0) * onp.log(s))**3)
    else:
        return None  # Undefined for other values
                       
                       
                       
def fK(l, s, o, alpha):
    if l == 3:
        additional_term = delta(2) * fl(0, s, o)
        return fl(3, s, o, alpha) + additional_term
    else:
        return fl(l, s, o)
    
b = Beta(1) / (2 * Beta(0)**2)
    
#V0 = sum((k+1) * spl.gamma(1 + b) / spl.gamma(k + 2 + b) / (2 * Beta(0))**k * f(k) for k in range(0, L_pert))
#print('V0 is ------------->', V0)


#def Vl(l):
#    result = V0 * (2 * Beta(0))**l * spl.gamma(l + 1 + b) / spl.gamma(1 + b)
#    return float(result)

def V0(L_pert, s, o, alpha):
    return sum(fK(k, s, o, alpha) * spl.gamma(1 + b) / spl.gamma(k + 2 + b) * (k + 1) / (2 * Beta(0))**k for k in range(0, L_pert))


def errV0(L_pert, s, o, alpha): #Should I include it in the calculation of the uncertainty ?
    return 1 / V0(s, L_pert, o, alpha) * (fK(L_pert, s, o, alpha) * spl.gamma(1 + b)) / spl.gamma(L_pert + 2 + b) * (L_pert + 1) / (2 * Beta(0))**3


def QF(l):
    return (2 * Beta(0))**l * spl.gamma(b + l + 1) / spl.gamma(b + 1)


def Vl(l, L_pert, s, o, alpha):
    return V0(L_pert, s, o, alpha) * QF(l)

             
#@partial(jax.jit)
def VB_MRS_definition(alpha, Order, s, o='singulet'):
    
    if Order == 'LO':
        L_pert=1
    elif Order == 'NLO':
        L_pert=2
    elif Order == 'NNLO':
        L_pert=3
    elif Order == 'NNNLO':
        L_pert=4
    else: 
        print("Order non valide. Leading Order taken by default")
        L_pert=1
        
    def V_RS(alpha, L_pert, s, o):
        Vl_list = [Vl(l, L_pert, s, o, alpha) for l in range(0, L_pert)]
        vl_list = [vl(l, s, o, alpha) for l in range(0, L_pert)]
        #print(vl_list)
        #print(Vl_list)
        #rl_list = [0.42, 1.03, 3.69, 17.4] in the pole mass, so not our case !
        result = sum((vl_list[i] - Vl_list[i]) * alpha**(i+1) for i in range(0, L_pert))
        return result
    
    
    #@partial(jax.jit)
    def Javad_jax_real(a, alpha):
        x = 1/(2 * Beta(0) * alpha)
        term1 = np.exp(-x)
        
        def lower_incomplete_gamma(a_value, x_values):
            
            @partial(jax.jit)
            def series_sum(a, x, tolerance=1e-10, max_iter=10000):
                y = -x
                def body_fun(val):
                    n, total, last_term = val
                    n_fact = jax.scipy.special.gamma(n + 1)  # Using gamma(n+1) for n!
                    current_term = (y ** n) / (n_fact * (n + a))
                    new_total = total + current_term
                    return (n + 1, new_total, current_term)

                def cond_fun(val):
                    n, total, last_term = val
                    # Continue if the last term is larger than the tolerance and n is less than max_iter
                    return (np.max(np.abs(last_term)) > tolerance) & (n < max_iter)

                # Initial values: n=0, total=0, last_term=large value to start the loop
                n0 = 0
                total0 = np.zeros_like(y)
                last_term0 = np.ones_like(y) * np.inf

                # Run the loop
                _, total, _ = jax.lax.while_loop(cond_fun, body_fun, (n0, total0, last_term0))
                #return total * (x + 0j)**a #It will make appear the term (-x)^(-a) after the argument are taken negatively
                return total
            
            # Compute the series sum
            result = series_sum(a_value, x_values)
            #print(result)

            return result
        
        term3 = lower_incomplete_gamma(-a, -x)
        #term3 = jax.scipy.special.gammainc(-a, -x_complex) * jax.scipy.special.gamma(-a) does not work
        
        #term4 = (-x + 0j)**a
        
        result = term1 * term3 #* term4
        return np.real(result)
        #return result
    
    def Javad_float_real(a, alpha):
        x = 1/(2 * Beta(0) * alpha)
        term1 = mpmath.exp(-x)
        term3 = mpmath.gammainc(-a, 0, -x, regularized=False)
        term4 = (-x)**a
        result = term1 * term3 * term4
        return float(result.real)

    import timeout_decorator
    @timeout_decorator.timeout(5, timeout_exception=StopIteration)  # Timeout in 5 seconds
    def calc_Javad_real(a, x):
        try:
            if type(x) == float or type(x) == int:
                return Javad_float_real(a, x)
            else:
                return Javad_jax_real(a, x)
        except StopIteration:
            return 0
             
    def V_Borel(alpha, L_pert, s, o):
        b = Beta(1) / (2 * Beta(0)**2)
        result = V0(L_pert, s, o, alpha) * 1 / (2 * Beta(0)) * calc_Javad_real(b, alpha)
        return result
    #print(V_RS(alpha, L_pert, s), '\n', V_Borel(alpha, L_pert, s))
    return (V_RS(alpha, L_pert, s, o) + V_Borel(alpha, L_pert, s, o)) 


###################################################################################################################

#The 2 first/scheme-independent
def Beta0(Nf):
    Beta0 = (1 / (4 * Pi)) * (11/3 * CA - 2/3 * Nf)
    return Beta0

def Beta1(Nf):
    Beta1 = (1 / (4 * Pi))**2 * (34/3 * CA**2 - 10/3 * CA * Nf - 2 * CF * Nf)
    return Beta1

#Beta scheme-dependent: in the MS bar scheme here
def Beta2(Nf):
    Beta2 = (1 / (4 * Pi))**3 * (2857/54 * CA**3 - 1415/27 * CA**2 * Nf / 2 + 
         158/27 * CA * Nf**2 / 4 + 44/9 * CF * Nf**2 / 4 - 
         205/9 * CF * CA * Nf / 2 + CF**2 * Nf)
    return Beta2

def Beta3(Nf):
    Beta3 = (1 / (4 * Pi))**4 * (CA * CF * Nf**2 / 4 * (17152/243 + 448/9 * Zeta(3)) + 
         CA * CF**2 * Nf / 2 * (-4204/27 + 352/9 * Zeta(3)) + 
         424/243 * CA * Nf**3 / 8 + 1232/243 * CF * Nf**3 / 8 + 
         CA**2 * CF * Nf / 2 * (7073/243 - 656/9 * Zeta(3)) + 
         CA**2 * Nf**2 / 4 * (7930/81 + 224/9 * Zeta(3)) + 
         CA**3 * Nf / 2 * (-39143/81 + 136/3 * Zeta(3)) + 
         CA**4 * (150653/486 - 44/9 * Zeta(3)) + 
         CF**2 * Nf**2 / 4 * (1352/27 - 704/9 * Zeta(3)) + 
         46 * CF**3 * Nf / 2 + 
         Nf * dFA * (512/9 - 1664/3 * Zeta(3)) + 
         Nf**2 * dFF * (-704/9 + 512/3 * Zeta(3)) + 
         dAA * (-80/9 + 704/3 * Zeta(3)))
    return Beta3

Mt = 173.34
Mb = 4.7
Mc = 1.5
MZ = 91.1876

#######################################################################################
def LambdaQCD_MSbar(f):
    if f == 3:
        return 0.338
    elif f == 4:
        return 0.296
    elif f == 5:
        return 0.213
    elif f == 6:
        return 0.090
    else:
        return None  # Undefined for other values


#######################################################################################
#######################################################################################


def find_Lambda():
    
    def LQ_MSbar(Q, f):
        return np.log(Q**2 / LambdaQCD_MSbar(f)**2)
    
    def Alpha_sNf(Q, f):
        LQ_val = LQ_MSbar(Q, f)
        return 1 / (Beta0(f) * LQ_val) - \
               Beta1(f) * np.log(LQ_val) / (Beta0(f)**3 * LQ_val**2) + \
               1 / (Beta0(f)**3 * LQ_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (np.log(LQ_val)**2 - np.log(LQ_val) - 1) + Beta2(f) / Beta0(f)) + \
               1 / (Beta0(f)**4 * LQ_val**4) * (Beta1(f)**3 / Beta0(f)**3 * (-np.log(LQ_val)**3 + 2.5 * np.log(LQ_val)**2 + 2 * np.log(LQ_val) - 0.5) - \
               3 * Beta1(f) * Beta2(f) * np.log(LQ_val) / Beta0(f)**2 + Beta3(f) / (2 * Beta0(f)))

    
    def MAlpha_s(Q, f):
        alpha_s = Alpha_sNf(Q, f)
        return 1 + (-0.291667) * (alpha_s / math.pi)**2 + (-5.32389 + (f - 1) * 0.26247) * (alpha_s / math.pi)**3

    
    def Alpha_s(Q):
        if Q < 1:
            return MAlpha_s(Mc, 4) * MAlpha_s(Mb, 5) * Alpha_sNf(1, 5)
        elif 1 <= Q < Mc:
            return MAlpha_s(Mc, 4) * MAlpha_s(Mb, 5) * Alpha_sNf(Q, 5)
        elif Mc <= Q < Mb:
            return MAlpha_s(Mb, 5) * Alpha_sNf(Q, 5)
        elif Mb <= Q < Mt:
            return Alpha_sNf(Q, 5)
        elif Q >= Mt:
            return 1 / MAlpha_s(Mt, 6) * Alpha_sNf(Q, 5)
        else:
            return None
        
        
    
    def LQs_MSbar(Q, Lambda):
        return np.log(Q**2 / Lambda**2)
    
    
    def Alpha_sNf_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val) - \
               Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
               1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (np.log(LR_val)**2 - np.log(LR_val) - 1) + Beta2(f) / Beta0(f)) + \
               1 / (Beta0(f)**4 * LR_val**4) * (Beta1(f)**3 / Beta0(f)**3 * (-np.log(LR_val)**3 + 2.5 * np.log(LR_val)**2 + 2 * np.log(LR_val) - 0.5) - \
               3 * Beta1(f) * Beta2(f) * np.log(LR_val) / Beta0(f)**2 + Beta3(f) / (2 * Beta0(f)))


    def Alpha_sNf3Loop_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val) - \
               Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
               1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (np.log(LR_val)**2 - np.log(LR_val) - 1) + Beta2(f) / Beta0(f))

    def Alpha_sNf2Loop_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val) - \
               Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2)

    def Alpha_sNf1Loop_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val)
    
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf_L(MZ, 5, Lambda) - Alpha_s(MZ)
        
    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda4LoopNf5 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf_L(Mt, 6, Lambda) - Alpha_sNf_L(Mt, 5, Lambda4LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda4LoopNf6 = result.root    
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf_L(Mb, 4, Lambda) - Alpha_sNf_L(Mb, 5, Lambda4LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda4LoopNf4 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf_L(Mc, 3, Lambda) - Alpha_sNf_L(Mc, 4, Lambda4LoopNf4)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda4LoopNf3 = result.root
    
    Lambda4Loop = [Lambda4LoopNf6, Lambda4LoopNf5, Lambda4LoopNf4, Lambda4LoopNf3]
    
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf3Loop_L(MZ, 5, Lambda) - Alpha_s(MZ)
        
    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda3LoopNf5 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf3Loop_L(Mt, 6, Lambda) - Alpha_sNf3Loop_L(Mt, 5, Lambda3LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda3LoopNf6 = result.root    
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf3Loop_L(Mb, 4, Lambda) - Alpha_sNf3Loop_L(Mb, 5, Lambda3LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda3LoopNf4 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf3Loop_L(Mc, 3, Lambda) - Alpha_sNf3Loop_L(Mc, 4, Lambda3LoopNf4)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda3LoopNf3 = result.root

    Lambda3Loop = [Lambda3LoopNf6, Lambda3LoopNf5, Lambda3LoopNf4, Lambda3LoopNf3]
    
    
        # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf2Loop_L(MZ, 5, Lambda) - Alpha_s(MZ)
        
    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda2LoopNf5 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf2Loop_L(Mt, 6, Lambda) - Alpha_sNf2Loop_L(Mt, 5, Lambda2LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda2LoopNf6 = result.root    
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf2Loop_L(Mb, 4, Lambda) - Alpha_sNf2Loop_L(Mb, 5, Lambda2LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda2LoopNf4 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf2Loop_L(Mc, 3, Lambda) - Alpha_sNf2Loop_L(Mc, 4, Lambda2LoopNf4)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda2LoopNf3 = result.root
    
    Lambda2Loop = [Lambda2LoopNf6, Lambda2LoopNf5, Lambda2LoopNf4, Lambda2LoopNf3]
    
    
        # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf1Loop_L(MZ, 5, Lambda) - Alpha_s(MZ)
        
    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda1LoopNf5 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf1Loop_L(Mt, 6, Lambda) - Alpha_sNf1Loop_L(Mt, 5, Lambda1LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda1LoopNf6 = result.root    
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf1Loop_L(Mb, 4, Lambda) - Alpha_sNf1Loop_L(Mb, 5, Lambda1LoopNf5)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda1LoopNf4 = result.root
    
    # Define the function whose root we want to find
    def equation_to_solve(Lambda):
        return Alpha_sNf1Loop_L(Mc, 3, Lambda) - Alpha_sNf1Loop_L(Mc, 4, Lambda1LoopNf4)

    # Find the root using scipy.optimize.root_scalar
    result = root_scalar(equation_to_solve, x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])

    # Extract the solution
    Lambda1LoopNf3 = result.root
    
    Lambda1Loop = [Lambda1LoopNf6, Lambda1LoopNf5, Lambda1LoopNf4, Lambda1LoopNf3]
    
    return Lambda4Loop, Lambda3Loop, Lambda2Loop, Lambda1Loop


Lambda4Loop, Lambda3Loop, Lambda2Loop, Lambda1Loop = find_Lambda()

def Lambda4LoopNf(f):
    return Lambda4Loop[6-f]

def Lambda3LoopNf(f):
    return Lambda3Loop[6-f]

def Lambda2LoopNf(f):
    return Lambda2Loop[6-f]

def Lambda1LoopNf(f):
    return Lambda1Loop[6-f]



def LR_4Loop_MSbar(muPrime, f):
    return np.log(muPrime**2 / Lambda4LoopNf(f)**2)

def LR_3Loop_MSbar(muPrime, f):
    return np.log(muPrime**2 / Lambda3LoopNf(f)**2)

def LR_2Loop_MSbar(muPrime, f):
    return np.log(muPrime**2 / Lambda2LoopNf(f)**2)

def LR_1Loop_MSbar(muPrime, f):
    return np.log(muPrime**2 / Lambda1LoopNf(f)**2)



def Alpha_sNf(muPrime, f):
    LR_val = LR_4Loop_MSbar(muPrime, f)
    return 1 / (Beta0(f) * LR_val) - \
           Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
           1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (np.log(LR_val)**2 - np.log(LR_val) - 1) + Beta2(f) / Beta0(f)) + \
           1 / (Beta0(f)**4 * LR_val**4) * (Beta1(f)**3 / Beta0(f)**3 * (-np.log(LR_val)**3 + 2.5 * np.log(LR_val)**2 + 2 * np.log(LR_val) - 0.5) - \
           3 * Beta1(f) * Beta2(f) * np.log(LR_val) / Beta0(f)**2 + Beta3(f) / (2 * Beta0(f)))


def Alpha_sNf3Loop(muPrime, f):
    LR_val = LR_3Loop_MSbar(muPrime, f)
    return 1 / (Beta0(f) * LR_val) - \
           Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
           1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (np.log(LR_val)**2 - np.log(LR_val) - 1) + Beta2(f) / Beta0(f))

def Alpha_sNf2Loop(muPrime, f):
    LR_val = LR_2Loop_MSbar(muPrime, f)
    return 1 / (Beta0(f) * LR_val) - \
           Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2)

def Alpha_sNf1Loop(muPrime, f):
    LR_val = LR_1Loop_MSbar(muPrime, f)
    return 1 / (Beta0(f) * LR_val)



def MAlpha_s(Q, f):
    alpha_s = Alpha_sNf(Q, f)
    return 1 + (-0.291667) * (alpha_s / math.pi)**2 + (-5.32389 + (f - 1) * 0.26247) * (alpha_s / math.pi)**3

def MAlpha_s2Loop(Q, f):
    return 1 + (-0.291667) * (Alpha_sNf3Loop(Q, f) / math.pi)**2 

##################################


def mu_Prime(R):
    return np.sqrt(mu**2 + 1/R**2)
    #return np.sqrt(mu**2 + (mufac/R)**2) #?

@jax.jit
def Alpha_s(R):
    #inv_R = 1 / R
    muPrime = mu_Prime(R)
    result = np.where(
        muPrime < 1,
        MAlpha_s(Mc, 4) * MAlpha_s(Mb, 5) * Alpha_sNf(muPrime, 5), # ? 
        np.where(
            muPrime < Mc,
            MAlpha_s(Mc, 4) * MAlpha_s(Mb, 5) * Alpha_sNf(muPrime, 5),
            np.where(
                muPrime < Mb,
                MAlpha_s(Mb, 5) * Alpha_sNf(muPrime, 5),
                np.where(
                    muPrime < Mt,
                    Alpha_sNf(muPrime, 5),
                    1 / MAlpha_s(Mt, 6) * Alpha_sNf(muPrime, 5)
                )
            )
        )
    )
    return result


@jax.jit
def Alpha_s1Loop(R):
    #inv_R = 1 / R
    #Does we define mu before or here ?
    muPrime = mu_Prime(R)
    result = np.where(
        muPrime < 1,
        Alpha_sNf1Loop(muPrime, 3), # ?
        np.where(
            muPrime < Mc,
            Alpha_sNf1Loop(muPrime, 3),
            np.where(
                muPrime < Mb,
                Alpha_sNf1Loop(muPrime, 4),
                np.where(
                    muPrime < Mt,
                    Alpha_sNf1Loop(muPrime, 5),
                    Alpha_sNf1Loop(muPrime, 6)
                )
            )
        )
    )
    return result

@jax.jit
def Alpha_s2Loop(R):
    #inv_R = 1 / R
    muPrime = mu_Prime(R)
    result = np.where(
        muPrime < 1,
        Alpha_sNf2Loop(muPrime, 3),  # ?
        np.where(
            muPrime < Mc,
            Alpha_sNf2Loop(muPrime, 3), 
            np.where(
                muPrime < Mb,
                Alpha_sNf2Loop(muPrime, 4), 
                np.where(
                    muPrime < Mt,
                    Alpha_sNf2Loop(muPrime, 5),
                    Alpha_sNf2Loop(muPrime, 6)
                )
            )
        )
    )
    return result

@jax.jit
def Alpha_s3Loop(R):
    #inv_R = 1 / R
    muPrime = mu_Prime(R)
    result = np.where(
        muPrime < 1,
        MAlpha_s2Loop(Mc, 4) * MAlpha_s2Loop(Mb, 5) * Alpha_sNf3Loop(muPrime, 5), #?
        np.where(
            muPrime < Mc,
            MAlpha_s2Loop(Mc, 4) * MAlpha_s2Loop(Mb, 5) * Alpha_sNf3Loop(muPrime, 5),
            np.where(
                muPrime < Mb,
                MAlpha_s2Loop(Mb, 5) * Alpha_sNf3Loop(muPrime, 5),
                np.where(
                    muPrime < Mt,
                    Alpha_sNf3Loop(muPrime, 5),
                    1 / MAlpha_s2Loop(Mt, 6) * Alpha_sNf3Loop(muPrime, 5)
                )
            )
        )
    )
    return result



# muPrime = np.sqrt(mu**2 + (mufac/R)**2)
#######################################################################################
#######################################################################################

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

print("masses = ", masses)
print("spatial wavefunction = ", wavefunction)
print("g = ", g)
print("color wavefunction = ", color)

#if wavefunction == "asymmetric":
#    bra_wavefunction = "product"
#    ket_wavefunction = "compact"
#else:
bra_wavefunction = wavefunction
ket_wavefunction = wavefunction

#assert Nc == NI

#CF = (Nc**2 - 1)/(2*Nc)c
#VB = alpha*CF/(Nc-1) 

SingC3 = -(Nc+1)/8

#a0=4.514

#VB=.1
#print(VB)
#quit()
# imaginary time points for GFMC evolution
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)



#######################################################################################
#Rprime = lambda R: adl.norm_3vec(R)*np.exp(np.euler_gamma)*mu

VB_LO = lambda R: CF * VB_MRS_definition(Alpha_s1Loop(R), 'LO', 1) / (Nc - 1)

#VB_NLO(R) = Alpha_s2Loop(R)*CF/(Nc-1) * (1 + Alpha_s2Loop(R)/(4*np.pi)*(aa1 + 2*beta0*log_mu_r))
#VB_NNLO(R) = Alpha_s3Loop(R)*CF/(Nc-1) * (1 + Alpha_s3Loop(R)/(4*np.pi)*(aa1 + 2*beta0*L) + (Alpha_s3Loop(R)/(4*np.pi))**2*( beta0**2*(4*L**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*L + aa2 ) )

VB_NLO = lambda R: CF * VB_MRS_definition(Alpha_s2Loop(R), 'NLO', 1) / (Nc - 1)
VB_NNLO = lambda R: CF * VB_MRS_definition(Alpha_s3Loop(R), 'NNLO', 1) / (Nc - 1)
VB_NNNLO = lambda R: CF * VB_MRS_definition(Alpha_s(R), 'NNNLO', 1) / (Nc - 1) ############ NEW

#ANTISYMMETRIC AND SYMMETRIC:

VB_NNLO_antisym = lambda R: CF * VB_MRS_definition(Alpha_s3Loop(R), 'NNLO', 1, 'antisymmetric') / (Nc - 1)
VB_NNLO_sym = lambda R: CF * VB_MRS_definition(Alpha_s3Loop(R), 'NNLO', 1, 'symmetric') / (Nc - 1)


#OCTET :

VB_NNLO_octet = lambda R: (CA/2 - CF) * VB_MRS_definition(Alpha_s3Loop(R), 'NNLO', 1, 'octet') 

#######################################################################################
#Can only have a guess of the mass, so we take the mass of the quark at MS bar:
if mQ == 0:
    if nf == 3:
        mQ = Mc
    elif nf == 4:
        mQ = Mb
    elif nf == 5:
        mQ = Mt
    else: None 
else: None
#NEED TO BE CONFIRMED THAT IT MAKE SENSE OR NOT
print('mQ is ', mQ)
#######################################################################################

print('VB_LO Rstar is ', VB_LO(Rstar))
print('VB_NNLO Rstar is ', VB_NNLO(Rstar))
print('VB_NNNLO Rstar is ', VB_NNNLO(Rstar))



#r0 = 2/alpha ? <--- initial guess, no need to be precise

if OLO == "LO":
    a0=spoila*2/VB_LO(Rstar) ##### ? NEW
elif OLO == "NLO":
    a0=spoila*2/VB_NLO(Rstar) ##### ? NEW
elif OLO == "NNLO":
    a0=spoila*2/VB_NNLO(Rstar) ##### ? NEW

if N_coord == 2 or N_coord == 4:
    a0 /= Nc-1


ket_a0 = a0

biga0 = a0

if wavefunction == "product":
    biga0 = a0*afac

#if correlator == "asymmetric":
#ket_a0 = a0*spoilaket
ket_afac = afac*spoilaket

print("a0 = ", a0)
print("ket_a0 = ", ket_a0)
print("afac = ", afac)
print("ket_afac = ", ket_afac)

Rprime = lambda R: adl.norm_3vec(R)*np.exp(np.euler_gamma)*mu*mufac ######################### MODIFIED
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
    #print("FV Coulomb test")
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
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * VB_LO(norm_R/(mufac*mQ)) / norm_R
        #potential_result = -1 * VB_LO(norm_R/(mufac*mQ)) / norm_R # ?????????????????????????????????
        return potential_result

    @partial(jax.jit)
    def symmetric_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * (Nc - 1) / (Nc + 1) * VB_LO(norm_R/(mufac*mQ)) / norm_R
        return potential_result

    @partial(jax.jit)
    def singlet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * (Nc - 1) * VB_LO(norm_R/(mufac*mQ)) / norm_R
        #potential_result = -1 * (Nc - 1) * VB_LO(norm_R/mQ) / norm_R # ?
        return potential_result

    @partial(jax.jit)
    def octet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * (Nc - 1) / CF / (2 * Nc) * VB_LO(norm_R/(mufac*mQ)) / norm_R
        return potential_result

    @partial(jax.jit)
    def potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * VB_LO(norm_R/(mufac*mQ)) * FV_Coulomb(R, L, nn)
        return potential_result

    @partial(jax.jit)
    def symmetric_potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * (Nc - 1) / (Nc + 1) * VB_LO(norm_R/(mufac*mQ)) * FV_Coulomb(R, L, nn)
        return potential_result

    @partial(jax.jit)
    def singlet_potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * (Nc - 1) * VB_LO(norm_R/(mufac*mQ)) * FV_Coulomb(R, L, nn)
        return potential_result

    @partial(jax.jit)
    def octet_potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * (Nc - 1) / CF / (2 * Nc) * VB_LO(norm_R/(mufac*mQ)) * FV_Coulomb(R, L, nn)
        return potential_result


elif OLO == "NLO":
    @partial(jax.jit)
    def potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * VB_NLO(norm_R/(mufac*mQ)) / norm_R #* (1 + alpha_s2loop / (4 * np.pi) * aa1) # No more mu dependence but 1/R
        return potential_result

    @partial(jax.jit)
    def potential_fun_sum(R):
        return calculate_sum(potential_fun, R, L, nn)
    def symmetric_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        VB_NLO_val = VB_NLO(norm_R/(mufac*mQ))
        potential_result = (Nc - 1) / (Nc + 1) * VB_NLO_val * spoilS / norm_R #* (1 + alpha_s2loop / (4 * np.pi) * aa1)
        return potential_result

    def symmetric_potential_fun_sum(R):
        return calculate_sum(symmetric_potential_fun, R, L, nn)
    @partial(jax.jit)
    def singlet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        VB_NLO_val = VB_NLO(norm_R/(mufac*mQ))  
        potential_result = -1 * (Nc - 1) * VB_NLO_val / norm_R #* (1 + alpha_s2loop / (4 * np.pi) * aa1)
        return potential_result

    @partial(jax.jit)
    def singlet_potential_fun_sum(R):
        return calculate_sum(singlet_potential_fun, R, L, nn)

    @partial(jax.jit)
    def octet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        VB_NLO_val = VB_NLO(norm_R/(mufac*mQ))
        potential_result = spoilS * (Nc - 1) / CF / (2 * Nc) * VB_NLO_val / norm_R #* (1 + alpha_s2loop / (4 * np.pi) * aa1)
        return potential_result

    def octet_potential_fun_sum(R):
        return calculate_sum(octet_potential_fun, R, L, nn)


elif OLO == "NNLO":
    @partial(jax.jit)
    def potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * spoilS * VB_NNLO_antisym(norm_R/(mufac*mQ)) / norm_R  #* (1 + Alpha_s3Loop(norm_R) / (4 * np.pi) * aa1 + (Alpha_s3Loop(norm_R) / (4 * np.pi))**2 * (beta0**2 * np.pi**2 / 3 + aa2 + Nc * (Nc - 2) / 2 * ((np.pi)**4 - 12 * (np.pi)**2)))
        return potential_result

    @partial(jax.jit)
    def potential_fun_sum(R):
        return calculate_sum(potential_fun, R, L, nn)

    def symmetric_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * (Nc - 1) / (Nc + 1) * VB_NNLO_sym(norm_R/(mufac*mQ)) / norm_R #* (1 + Alpha_s3Loop(norm_R) / (4 * np.pi) * aa1 + (Alpha_s3Loop(norm_R) / (4 * np.pi))**2 * (beta0**2 * np.pi**2 / 3 + aa2 + Nc * (Nc + 2) / 2 * ((np.pi)**4 - 12 * (np.pi)**2)))
        return potential_result

    def symmetric_potential_fun_sum(R):
        return calculate_sum(symmetric_potential_fun, R, L, nn)

    def singlet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * (Nc - 1) * VB_NNLO(norm_R/(mufac*mQ)) / norm_R #* (1 + Alpha_s3Loop(norm_R) / (4 * np.pi) * aa1 + (Alpha_s3Loop(norm_R) / (4 * np.pi))**2 * (beta0**2 * np.pi**2 / 3 + aa2))
        return potential_result

    def singlet_potential_fun_sum(R):
        return calculate_sum(singlet_potential_fun, R, L, nn)

    def octet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * VB_NNLO_octet(norm_R/(mufac*mQ)) / norm_R #* (1 + Alpha_s3Loop(norm_R) / (4 * np.pi) * aa1 + (Alpha_s3Loop(norm_R) / (4 * np.pi))**2 * (beta0**2 * np.pi**2 / 3 + aa2 + (Nc**2) * ((np.pi)**4 - 12 * (np.pi)**2)))
        return potential_result

    def octet_potential_fun_sum(R):
        return calculate_sum(octet_potential_fun, R, L, nn)
    
elif OLO == "NNNLO": #not viable yet
    @partial(jax.jit)
    def potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * spoilS * VB_NNNLO_antisym(norm_R/(mufac*mQ)) / norm_R #+ an other color factor
        return potential_result

    @partial(jax.jit)
    def potential_fun_sum(R):
        return calculate_sum(potential_fun, R, L, nn)

    def symmetric_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * (Nc - 1) / (Nc + 1) * VB_NNNLO_sym(norm_R/(mufac*mQ)) / norm_R #+ an other color factor(np.pi)**2)))
        return potential_result

    def symmetric_potential_fun_sum(R):
        return calculate_sum(symmetric_potential_fun, R, L, nn)

    def singlet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = -1 * (Nc - 1) * VB_NNNLO(norm_R/(mufac*mQ)) / norm_R 
        return potential_result

    def singlet_potential_fun_sum(R):
        return calculate_sum(singlet_potential_fun, R, L, nn)

    def octet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        potential_result = spoilS * VB_NNNLO_octet(norm_R/(mufac*mQ)) / norm_R 
        return potential_result

    def octet_potential_fun_sum(R):
        return calculate_sum(octet_potential_fun, R, L, nn)

else:
        print("order not supported")
        throw(0)
    

def trivial_fun(R):
    return 0*adl.norm_3vec(R)+1

def trivial_fun_1(R):
    return 0*adl.norm_3vec(R)-4/3

def trivial_fun_8(R):
    return 0*adl.norm_3vec(R)+1/6

def trivial_fun_A(R):
    return 0*adl.norm_3vec(R)-2/3

def trivial_fun_S(R):
    return 0*adl.norm_3vec(R)+1/3

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
    #AV_Coulomb['OA'] = trivial_fun_A
    #AV_Coulomb['OS'] = trivial_fun_S
    #AV_Coulomb['OSing'] = trivial_fun_1
    #AV_Coulomb['OO'] = trivial_fun_8
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
@partial(jax.jit, static_argnums=(1,))
def f_R_slow(Rs, wavefunction=bra_wavefunction, a0=a0):
    #N_walkers = Rs.shape[0]
    #assert Rs.shape == (N_walkers, N_coord, 3)
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

pairs = np.array([np.array([j, i]) for i in range(0,N_coord) for j in range(0, i)])
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
        return psi*(174960 - 145800*r + 32400*r**2 - 2700*r**3 + 90*r**4 - r**5)
    if n == 7:
        return psi*(37059435 - 31765230*r + 7563150*r**2 - 720300*r**3 + 30870*r**4 - 588*r**5 + 4*r**6)


@partial(jax.jit, static_argnums=(1,))
def f_R(Rs, wavefunction=bra_wavefunction, a0=a0, afac=afac, masses=absmasses):

    def r_norm(pair):
        [i,j] = pair
        rdiff = Rs[...,i,:] - Rs[...,j,:]
        mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
        rij_norm = np.sqrt( np.sum(rdiff*rdiff, axis=-1) )
        return rij_norm * mij

    if wavefunction == "product":
        r_sum = np.sum( jax.lax.map(r_norm, product_pairs), axis=0 )*(1/a0-1/(a0*afac)) + np.sum( jax.lax.map(r_norm, pairs), axis=0 )/(a0*afac)
        r_sum += np.sum( jax.lax.map(r_norm, same_pairs), axis=0 )*(1/(a0*afac*samefac)-1/(a0*afac))
    elif wavefunction == "diquark":
        r_sum = np.sum( jax.lax.map(r_norm, diquark_pairs), axis=0 )*(1/a0-1/(a0*afac)) + np.sum( jax.lax.map(r_norm, pairs), axis=0 )/(a0*afac)
    else:
        r_sum = np.sum( jax.lax.map(r_norm, pairs), axis=0 )/a0

    #psi = np.exp(-r_sum)
    psi = hydrogen_wvfn(r_sum, radial_n)
    afac *= gfac
    a0 /= gfac

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
            r_sum_T = np.sum( jax.lax.map(r_norm_T, product_pairs), axis=0 )*(1/a0-1/(a0*afac)) + np.sum( jax.lax.map(r_norm_T, pairs), axis=0 )/(a0*afac)
            r_sum_T += np.sum( jax.lax.map(r_norm_T, same_pairs), axis=0 )*(1/(a0*afac*samefac)-1/(a0*afac))
        elif wavefunction == "diquark":
            r_sum_T = np.sum( jax.lax.map(r_norm_T, diquark_pairs), axis=0 )*(1/a0-1/(a0*afac)) + np.sum( jax.lax.map(r_norm_T, pairs), axis=0 )/(a0*afac)
        else:
            r_sum_T = np.sum( jax.lax.map(r_norm_T, pairs), axis=0 )/a0

        #psi += g * np.exp(-r_sum_T)
        psi += g * hydrogen_wvfn(r_sum_T, radial_n)
    return psi

def f_R_sq(Rs):
    return np.abs( f_R(Rs) )**2

def f_R_braket(Rs):
    #return np.abs( f_R(Rs, wavefunction=bra_wavefunction) * f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0, afac=ket_afac) )
    return np.abs( f_R(Rs, wavefunction=bra_wavefunction)**2 )
    #return np.abs( f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0, afac=ket_afac)**2 )

def f_R_braket_tempered(Rs, fac):
    return np.abs( f_R(Rs, wavefunction=bra_wavefunction) * f_R(Rs, wavefunction=ket_wavefunction, a0=fac*ket_a0) )

def f_R_braket_phase(Rs):
    prod = f_R(Rs, wavefunction=bra_wavefunction) * f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0, afac=ket_afac)
    #prod = ( f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0, afac=ket_afac) / f_R(Rs, wavefunction=bra_wavefunction) )
    #prod = ( f_R(Rs, wavefunction=bra_wavefunction) / f_R(Rs, wavefunction=ket_wavefunction, a0=ket_a0, afac=ket_afac) )**2
    return prod / np.abs( prod )

@partial(jax.jit)
def laplacian_f_R(Rs, wavefunction=bra_wavefunction, a0=a0, afac=afac, masses=absmasses):
    if radial_n > 1:
        assert N_coord == 2
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
                            # factor of two included to account for both terms appearing in laplacian
                            if k == i and l == j:
                                #nabla_psi = nabla_psi * (2/thisa0**2 - 4/(thisa0*rij_norm)) * np.exp(-rij_norm/thisa0)
                                #nabla_psi = nabla_psi * ((1/thisa0**2 - 2/(thisa0*rij_norm))/masses[k] + (1/thisa0**2 - 2/(thisa0*rij_norm))/masses[l]) * np.exp(-rij_norm/thisa0)
                                nabla_psi = nabla_psi * ((1/(radial_n*thisa0)**2 - 2/(thisa0*rij_norm))/masses[k] + (1/(radial_n*thisa0)**2 - 2/(thisa0*rij_norm))/masses[l]) * hydrogen_wvfn(rij_norm/thisa0, radial_n)
                            else:
                                nabla_psi = nabla_psi * hydrogen_wvfn(rij_norm/thisa0, radial_n)
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

if N_coord >= 6 and verbose:
    print("No JIT for Laplacian")
    def laplacian_f_R(Rs, wavefunction=bra_wavefunction, a0=a0, afac=afac, masses=absmasses):
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
                                mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
                                thisa0 /= mij
                                # nabla_k^2 r_kl = nabla_l^2 r_kl
                                # factor of two included to account for both terms appearing in laplacian
                                if k == i and l == j:
                                    #nabla_psi = nabla_psi * (2/thisa0**2 - 4/(thisa0*rij_norm)) * np.exp(-rij_norm/thisa0)
                                    nabla_psi = nabla_psi * ((1/thisa0**2 - 2/(thisa0*rij_norm))/masses[k] + (1/thisa0**2 - 2/(thisa0*rij_norm))/masses[l]) * np.exp(-rij_norm/thisa0)
                                else:
                                    nabla_psi = nabla_psi * np.exp(-rij_norm/thisa0)
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

N_inner = 2
if N_coord % 3 == 0:
    N_inner = 3

N_outer = N_coord//N_inner

# Metropolis
if input_Rs_database == "":
    met_time = time.time()
    R0 = onp.random.normal(size=(N_coord,3))/np.mean(absmasses)
    # set center of mass position to 0
    #R0 -= onp.mean(R0, axis=1, keepdims=True)
    #R0 -= onp.mean(R0*absmasses, axis=0, keepdims=True)/absmasses
    R0 -= onp.transpose(onp.transpose(onp.mean(onp.transpose(onp.transpose(R0)*absmasses), axis=0, keepdims=True))/onp.mean(absmasses))
    print("R0 = ", R0)
    print("NCOORD = ", N_coord)
    print("NINNER = ", N_inner)
    print("NOUTER = ", N_outer)
    #if N_coord % 3 == 0:
        #samples = adl.direct_sample_metropolis(N_inner, N_outer, f_R_braket, a0*afac/3, n_therm=500, n_step=n_walkers, n_skip=n_skip, a0=a0/2)
    #else:
        #samples = adl.direct_sample_metropolis(N_inner, N_outer, f_R_braket, a0*afac, n_therm=500, n_step=n_walkers, n_skip=n_skip, a0=a0)
    #samples = adl.metropolis(R0, f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=2*a0/N_coord**2)
    if color == "6x6bar" or color == "SSS":
        samples = adl.metropolis(R0, f_R_braket, n_therm=500*n_skip, n_step=n_walkers, n_skip=n_skip, eps=4*2*a0*afac/N_coord**2*radial_n, masses=absmasses)
    else:
        #samples = adl.metropolis(R0, f_R_braket, n_therm=500*n_skip, n_step=n_walkers, n_skip=n_skip, eps=4*2*a0/N_coord**2, masses=absmasses)
        samples = adl.metropolis(R0, f_R_braket, n_therm=500*n_skip, n_step=n_walkers, n_skip=n_skip, eps=4*2*a0/4**2*radial_n, masses=absmasses)

    #samples = adl.metropolis(R0, f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=2*a0/N_coord**2)

    #samples = adl.metropolis(R0, f_R_braket, n_therm=500*n_skip, n_step=n_walkers, n_skip=n_skip, eps=8*2*a0/N_coord**2)

    #samples = adl.direct_sample_metropolis(f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, a0=a0)
    #samples = adl.metropolis(R0, f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=4*2*a0/N_coord**2)
    #samples = adl.metropolis(R0, f_R_braket, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=0.1*2*a0/N_coord**2)

    fac_list = [1/2, 1.0, 2]
    streams = len(fac_list)
    R0_list = [ onp.random.normal(size=(N_coord,3)) for s in range(0,streams) ]
    for s in range(streams):
        R0_list[s] -= onp.mean(R0_list[s], axis=0, keepdims=True)
    #print("R0 = ", R0_list[0])
    #samples = adl.parallel_tempered_metropolis(fac_list, R0_list, f_R_braket_tempered, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=8*2*a0/N_coord**2)
    #samples = adl.parallel_tempered_metropolis(fac_list, R0_list, f_R_braket_tempered, n_therm=500, n_step=n_walkers, n_skip=n_skip, eps=4*2*a0/N_coord**2)
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

def TAAA(i, j, k, l, m, n):
    return (levi_civita(i,j,m)*levi_civita(k,l,n) - levi_civita(i,j,n)*levi_civita(k,l,m))/(4*np.sqrt(3))

def TAAS(i, j, k, l, m, n):
    return (levi_civita(i,j,m)*levi_civita(k,l,n) + levi_civita(i,j,n)*levi_civita(k,l,m))/(4*np.sqrt(6))

def TASA(i, j, k, l, m, n):
    return (levi_civita(i,j,k)*levi_civita(m,n,l) + levi_civita(i,j,l)*levi_civita(m,n,k))/(4*np.sqrt(6))

def TSAA(i, j, k, l, m, n):
    return (levi_civita(m,n,i)*levi_civita(k,l,j) + levi_civita(m,n,j)*levi_civita(k,l,i))/(4*np.sqrt(6))

def TSSS(i, j, k, l, m, n):
    return (levi_civita(i,k,m)*levi_civita(j,l,n) + levi_civita(i,k,n)*levi_civita(j,l,m) + levi_civita(j,k,m)*levi_civita(i,l,n) + levi_civita(j,k,n)*levi_civita(i,l,m))/(12*np.sqrt(2))

def kronecker_delta(i, j):
    return 1 if i == j else 0

print("spin-flavor wavefunction shape = ", S_av4p_metropolis.shape)

if N_coord == 3:
  for i in range(NI):
   for j in range(NI):
    for k in range(NI):
     if i != j and j != k and i != k:
      spin_slice = (slice(0, None),) + (i,0,j,0,k,0)
      S_av4p_metropolis[spin_slice] = levi_civita(i, j, k) / np.sqrt(2*NI)

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
        #if i == j and k == l:
        spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,0)
        if color == "1x1":
            if swapI == 1:
                # 1 x 1 -- Q Q Qbar Qbar
                #print("QQ QbarQbar ordering")
                S_av4p_metropolis[spin_slice] = kronecker_delta(i, k)*kronecker_delta(j,l)/NI
            else:
                # 1 x 1 -- Q Qbar Q Qbar
                #print("QQbar QQbar ordering")
                S_av4p_metropolis[spin_slice] = kronecker_delta(i, j)*kronecker_delta(k,l)/NI
        elif color == "3x3bar":
            if swapI == 1:
                # 3bar x 3 -- Q Q Qbar Qbar
                #print("QQ QbarQbar ordering")
                S_av4p_metropolis[spin_slice] += kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2-2*NI)
                S_av4p_metropolis[spin_slice] -= kronecker_delta(i, l)*kronecker_delta(j, k)/np.sqrt(2*NI**2-2*NI)
            else:
                # 3bar x 3 -- Q Qbar Q Qbar
                #print("QQbar QQbar ordering")
                S_av4p_metropolis[spin_slice] += kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2-2*NI)
                S_av4p_metropolis[spin_slice] -= kronecker_delta(i, l)*kronecker_delta(k, j)/np.sqrt(2*NI**2-2*NI)
        elif color == "3x3bar-1":
            if swapI == 1:
                # 3bar x 3 -- Q Q Qbar Qbar
                #print("QQ QbarQbar ordering")
                part3x3bar = kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2-2*NI)
                part3x3bar -= kronecker_delta(i, l)*kronecker_delta(j, k)/np.sqrt(2*NI**2-2*NI)
                part1x1 = kronecker_delta(i, k)*kronecker_delta(j,l)/NI
                S_av4p_metropolis[spin_slice] = (part3x3bar - (1/np.sqrt(3))*part1x1)/np.sqrt(2/3)
            else:
                # 3bar x 3 -- Q Qbar Q Qbar
                #print("QQbar QQbar ordering")
                part3x3bar = kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2-2*NI)
                part3x3bar -= kronecker_delta(i, l)*kronecker_delta(k, j)/np.sqrt(2*NI**2-2*NI)
                part1x1 = kronecker_delta(i, j)*kronecker_delta(k,l)/NI
                S_av4p_metropolis[spin_slice] = (part3x3bar - (1/np.sqrt(3))*part1x1)/np.sqrt(2/3)
        elif color == "3x3bar-6x6bar":
            theta_c = 1.0
            if swapI == 1:
                # 3bar x 3 -- Q Q Qbar Qbar
                #print("QQ QbarQbar ordering")
                part3x3bar = kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2-2*NI)
                part3x3bar -= kronecker_delta(i, l)*kronecker_delta(j, k)/np.sqrt(2*NI**2-2*NI)
                part6x6bar = kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2+2*NI)
                part6x6bar += kronecker_delta(i, l)*kronecker_delta(j,k)/np.sqrt(2*NI**2+2*NI)
                S_av4p_metropolis[spin_slice] = np.cos(theta_c)*part3x3bar + np.sin(theta_c)*part6x6bar
            else:
                # 3bar x 3 -- Q Qbar Q Qbar
                #print("QQbar QQbar ordering")
                part3x3bar = kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2-2*NI)
                part3x3bar -= kronecker_delta(i, l)*kronecker_delta(k, j)/np.sqrt(2*NI**2-2*NI)
                part6x6bar = kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2+2*NI)
                part6x6bar += kronecker_delta(i, l)*kronecker_delta(k,j)/np.sqrt(2*NI**2+2*NI)
                S_av4p_metropolis[spin_slice] = np.cos(theta_c)*part3x3bar + np.sin(theta_c)*part6x6bar
        elif color == "6x6bar":
            if swapI == 1:
                # 6bar x 6 -- Q Q Qbar Qbar
                #print("QQ QbarQbar ordering")
                S_av4p_metropolis[spin_slice] += kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2+2*NI)
                S_av4p_metropolis[spin_slice] += kronecker_delta(i, l)*kronecker_delta(j,k)/np.sqrt(2*NI**2+2*NI)
            else:
                # 6bar x 6 -- Q Qbar Q Qbar
                #print("QQbar QQbar ordering")
                S_av4p_metropolis[spin_slice] += kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2+2*NI)
                S_av4p_metropolis[spin_slice] += kronecker_delta(i, l)*kronecker_delta(k,j)/np.sqrt(2*NI**2+2*NI)

if N_coord == 6:
  for i in range(NI):
   for j in range(NI):
    for k in range(NI):
     for l in range(NI):
      for m in range(NI):
       for n in range(NI):
          # up up up up up up
          spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,0,m,0,n,0)
          # up up up down down down
          #spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,1,m,1,n,1)
          if color == "1x1":
              S_av4p_metropolis[spin_slice] = levi_civita(i, j, k)*levi_civita(l, m, n) / 6
          # color tensors are ordered as diquark, diquark, diquark, each baryon has one diquark
          elif color == "AAA":
              #S_av4p_metropolis[spin_slice] = TAAA(i, j, l, m, k, n)
              S_av4p_metropolis[spin_slice] = TAAA(i, j, k, l, m, n)
          elif color == "AAS":
              #S_av4p_metropolis[spin_slice] = TAAS(i, j, l, m, k, n)
              S_av4p_metropolis[spin_slice] = TAAS(i, j, k, l, m, n)
          elif color == "ASA":
              #S_av4p_metropolis[spin_slice] = TASA(i, j, l, m, k, n)
              S_av4p_metropolis[spin_slice] = TASA(i, j, k, l, m, n)
          elif color == "SAA":
              #S_av4p_metropolis[spin_slice] = TSAA(i, j, l, m, k, n)
              S_av4p_metropolis[spin_slice] = TSAA(i, j, k, l, m, n)
          elif color == "SSS":
              #S_av4p_metropolis[spin_slice] = TSSS(i, j, l, m, k, n)
              S_av4p_metropolis[spin_slice] = TSSS(i, j, k, l, m, n)




#print(S_av4p_metropolis)

print("spin-flavor wavefunction shape = ", S_av4p_metropolis.shape)
S_av4p_metropolis_norm = adl.inner(S_av4p_metropolis, S_av4p_metropolis)
print("spin-flavor wavefunction normalization = ", S_av4p_metropolis_norm)
assert (np.abs(S_av4p_metropolis_norm - 1.0) < 1e-6).all()

#print("old ", f_R_old(Rs_metropolis))
#print("new ", f_R(Rs_metropolis))

#print("old laplacian ", laplacian_f_R_old(Rs_metropolis))
#print("new laplacian ", laplacian_f_R(Rs_metropolis))

# trivial contour deformation
deform_f = lambda x, params: x
params = (np.zeros((n_step+1)),)

print('Running GFMC evolution:')
if n_step > 0:
    rand_draws = onp.random.random(size=(n_step, Rs_metropolis.shape[0]))
    gfmc = adl.gfmc_deform(
        Rs_metropolis, S_av4p_metropolis, f_R, params,
        rand_draws=rand_draws, tau_iMev=tau_iMev, N=n_step, potential=Coulomb_potential,
        deform_f=deform_f, m_Mev=np.abs(np.array(masses)),
        resampling_freq=resampling)
    gfmc_Rs = np.array([Rs for Rs,_,_,_, in gfmc])
    gfmc_Ws = np.array([Ws for _,_,_,Ws, in gfmc])
    gfmc_Ss = np.array([Ss for _,_,Ss,_, in gfmc])
else:
    gfmc_Rs = np.array([Rs_metropolis])
    gfmc_Ws = np.array([0*Rs_metropolis[:,1,1]+1])
    gfmc_Ss = np.array([S_av4p_metropolis])

phase_Ws = f_R_braket_phase(gfmc_Rs)
print('phase Ws', phase_Ws)
gfmc_Ws *= phase_Ws

print('GFMC tau=0 weights:', gfmc_Ws[0])
if n_step > 0:
    print('GFMC tau=dtau weights:', gfmc_Ws[1])

# measure H
print('Measuring <H>...')

Ks = []
#for R in tqdm.tqdm(gfmc_Rs):
for count, R in enumerate(gfmc_Rs):
    print('Calculating Laplacian for step ', count)
    K_time = time.time()
    #Ks.append(-1/2*laplacian_f_R(R) / f_R(R) / adl.mp_Mev)
    K_term = -1/2*laplacian_f_R(R) / f_R(R, wavefunction=bra_wavefunction)

    if N_coord == 4 and abs(g) > 0:
        R_T = R
        R_T = R_T.at[...,1,:].set(R[...,swapI,:])
        R_T = R_T.at[...,swapI,:].set(R[...,1,:])

        K_term += -1/2*laplacian_f_R(R_T, afac=afac*gfac, a0=a0/gfac) / f_R(R_T, wavefunction=bra_wavefunction) * g

    Ks.append(K_term)

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

if N_coord == 4:
  S_1x1 = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
  S_3x3bar = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
  S_6x6bar = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)
  for i in range(NI):
   for j in range(NI):
    for k in range(NI):
     for l in range(NI):
        #if i == j and k == l:
        spin_slice = (slice(0, None),) + (i,0,j,0,k,0,l,0)
        if swapI == 1:
            S_1x1[spin_slice] = kronecker_delta(i, k)*kronecker_delta(j,l)/NI
            S_3x3bar[spin_slice] += kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2-2*NI)
            S_3x3bar[spin_slice] -= kronecker_delta(i, l)*kronecker_delta(j, k)/np.sqrt(2*NI**2-2*NI)
            S_6x6bar[spin_slice] += kronecker_delta(i, k)*kronecker_delta(j,l)/np.sqrt(2*NI**2+2*NI)
            S_6x6bar[spin_slice] += kronecker_delta(i, l)*kronecker_delta(j,k)/np.sqrt(2*NI**2+2*NI)
        else:
            S_1x1[spin_slice] = kronecker_delta(i, j)*kronecker_delta(k,l)/NI
            S_3x3bar[spin_slice] += kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2-2*NI)
            S_3x3bar[spin_slice] -= kronecker_delta(i, l)*kronecker_delta(k, j)/np.sqrt(2*NI**2-2*NI)
            S_6x6bar[spin_slice] += kronecker_delta(i, j)*kronecker_delta(k,l)/np.sqrt(2*NI**2+2*NI)
            S_6x6bar[spin_slice] += kronecker_delta(i, l)*kronecker_delta(k,j)/np.sqrt(2*NI**2+2*NI)

  Z_1x1 = []
  Z_3x3bar = []
  Z_6x6bar = []
  Z_norm = []
  for count, R in enumerate(gfmc_Rs):
    print('Calculating potential for step ', count)
    V_time = time.time()
    S = gfmc_Ss[count]
    Z_1x1.append(adl.inner(S_1x1, S) / np.sqrt(adl.inner(S, S)))
    Z_3x3bar.append(adl.inner(S_3x3bar, S) / np.sqrt(adl.inner(S, S)))
    Z_6x6bar.append(adl.inner(S_6x6bar, S) / np.sqrt(adl.inner(S, S)))
    Z_norm.append(np.sqrt( adl.inner(S_6x6bar, S)**2 + adl.inner(S_3x3bar, S)**2 ) / np.sqrt(adl.inner(S, S)))
  Z_1x1 = np.array(Z_1x1)
  Z_3x3bar = np.array(Z_3x3bar)
  Z_6x6bar = np.array(Z_6x6bar)
  Z_norm = np.array(Z_norm)

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
    #tag = str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_nskip" + str(n_skip) + "_Nf"+str(nf) + "_alpha"+str(alpha) + "_spoila"+str(spoila) + "_spoilaket"+str(spoilaket) + "_spoilf"+str(spoilf) + "_spoilS"+str(spoilS) + "_log_mu_r"+str(log_mu_r) + "_wavefunction_"+str(wavefunction) + "_potential_"+str(potential)+"_L"+str(L)+"_afac"+str(afac)+"_masses"+str(masses)+"_color_"+color+"_g"+str(g)
    tag = "muPrime_" + str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_nskip" + str(n_skip) + "_Nf"+str(nf) + "_radial_n"+str(radial_n) + "_mQ"+str(mQ) + "_mu"+str(mu) + "_mufac"+str(mufac) + "_Rstar" + str(Rstar) + "_spoila"+str(spoila) + "_log_mu_r"+str(log_mu_r) + "_wavefunction_"+str(wavefunction) + "_potential_"+str(potential)+"_L"+str(L)+"_afac"+str(afac)+"_masses"+str(masses)+"_color_"+color+"_g"+str(g)
else:
    #tag = str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_nskip" + str(n_skip) + "_Nf"+str(nf) + "_alpha"+str(alpha) + "_spoila"+str(spoila) + "_spoilaket"+str(spoilaket) + "_spoilf"+str(spoilf)+ "_spoilS"+str(spoilS) + "_log_mu_r"+str(log_mu_r) + "_wavefunction_"+str(wavefunction) + "_potential_"+str(potential)+"_afac"+str(afac)+"_masses"+str(masses)+"_color_"+color+"_g"+str(g)
    tag = "muPrime_" + str(OLO) + "_dtau"+str(dtau_iMev) + "_Nstep"+str(n_step) + "_Nwalkers"+str(n_walkers) + "_Ncoord"+str(N_coord) + "_Nc"+str(Nc) + "_nskip" + str(n_skip) + "_Nf"+str(nf) + "_radial_n"+str(radial_n) + "_mQ"+str(mQ) + "_mu"+str(mu) + "_mufac"+str(mufac) + "_Rstar" + str(Rstar) + "_spoila"+str(spoila) + "_log_mu_r"+str(log_mu_r) + "_wavefunction_"+str(wavefunction) + "_potential_"+str(potential)+"_afac"+str(afac)+"_masses"+str(masses)+"_color_"+color+"_g"+str(g)


with h5py.File(outdir+'Hammys_'+tag+'.h5', 'w') as f:
    dset = f.create_dataset("Hammys", data=Ks+Vs)
    dset = f.create_dataset("Ws", data=gfmc_Ws)

if N_coord == 4:
    with h5py.File(outdir+'Zs_'+tag+'.h5', 'w') as f:
        dset = f.create_dataset("Z_1x1", data=Z_1x1)
        dset = f.create_dataset("Z_3x3bar", data=Z_3x3bar)
        dset = f.create_dataset("Z_6x6bar", data=Z_6x6bar)

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

    if N_coord == 4:
      ave_Z_1x1 = np.array([al.bootstrap(Z, Nboot=100, f=al.rmean)
              for Z in Z_1x1])
      ave_Z_3x3bar = np.array([al.bootstrap(Z, Nboot=100, f=al.rmean)
              for Z in Z_3x3bar])
      ave_Z_6x6bar = np.array([al.bootstrap(Z, Nboot=100, f=al.rmean)
              for Z in Z_6x6bar])
      ave_Z_norm = np.array([al.bootstrap(Z, Nboot=100, f=al.rmean)
              for Z in Z_norm])

      print("Z_1x1=",ave_Z_1x1,"\n\n")
      print("Z_3x3bar=",ave_Z_3x3bar,"\n\n")
      print("Z_6x6bar=",ave_Z_6x6bar,"\n\n")
      print("Z_norm=",ave_Z_norm,"\n\n")
