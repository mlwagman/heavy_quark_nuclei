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
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--mufac', type=float, default=1.0)
parser.add_argument('--Nc', type=int, default=3)
parser.add_argument('--N_coord', type=int, default=3)
parser.add_argument('--nf', type=int, default=5)
parser.add_argument('--OLO', type=str, default="LO")
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
parser.add_argument('--Rc', type=float, default=1)  ############################ it's more about 0.5
parser.add_argument('--V_0_r_test', type=float, default=0.0)  ####### NEW
parser.add_argument('--r_test', type=float, default=0.03)  ### NEW
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
ss6 = zeta51+zeta6 
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

#MS_bar SCHEME ONLY ! https://arxiv.org/pdf/1608.02603.pdf
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
def VB_MRS_definition(alpha, L_pert, s, o='singulet'):
    
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



def LR_4Loop_MSbar(R, f):
    return np.log(1 / (R**2 * Lambda4LoopNf(f)**2))

def LR_3Loop_MSbar(R, f):
    return np.log(1 / (R**2 * Lambda3LoopNf(f)**2))

def LR_2Loop_MSbar(R, f):
    return np.log(1 / (R**2 * Lambda2LoopNf(f)**2))

def LR_1Loop_MSbar(R, f):
    return np.log(1 / (R**2 * Lambda1LoopNf(f)**2))



def Alpha_sNf(R, f):
    LR_val = LR_4Loop_MSbar(R, f)
    return 1 / (Beta0(f) * LR_val) - \
           Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
           1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (np.log(LR_val)**2 - np.log(LR_val) - 1) + Beta2(f) / Beta0(f)) + \
           1 / (Beta0(f)**4 * LR_val**4) * (Beta1(f)**3 / Beta0(f)**3 * (-np.log(LR_val)**3 + 2.5 * np.log(LR_val)**2 + 2 * np.log(LR_val) - 0.5) - \
           3 * Beta1(f) * Beta2(f) * np.log(LR_val) / Beta0(f)**2 + Beta3(f) / (2 * Beta0(f)))


def Alpha_sNf3Loop(R, f):
    LR_val = LR_3Loop_MSbar(R, f)
    return 1 / (Beta0(f) * LR_val) - \
           Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
           1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (np.log(LR_val)**2 - np.log(LR_val) - 1) + Beta2(f) / Beta0(f))

def Alpha_sNf2Loop(R, f):
    LR_val = LR_2Loop_MSbar(R, f)
    return 1 / (Beta0(f) * LR_val) - \
           Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2)

def Alpha_sNf1Loop(R, f):
    LR_val = LR_1Loop_MSbar(R, f)
    return 1 / (Beta0(f) * LR_val)



def MAlpha_s(Q, f):
    alpha_s = Alpha_sNf(1/Q, f)
    return 1 + (-0.291667) * (alpha_s / math.pi)**2 + (-5.32389 + (f - 1) * 0.26247) * (alpha_s / math.pi)**3

def MAlpha_s2Loop(Q, f):
    return 1 + (-0.291667) * (Alpha_sNf3Loop(1/Q, f) / math.pi)**2 

##################################

@jax.jit
def Alpha_s(R):
    inv_R = 1 / R

    result = np.where(
        inv_R < 1,
        MAlpha_s(Mc, 4) * MAlpha_s(Mb, 5) * Alpha_sNf(1, 5),
        np.where(
            inv_R < Mc,
            MAlpha_s(Mc, 4) * MAlpha_s(Mb, 5) * Alpha_sNf(R, 5),
            np.where(
                inv_R < Mb,
                MAlpha_s(Mb, 5) * Alpha_sNf(R, 5),
                np.where(
                    inv_R < Mt,
                    Alpha_sNf(R, 5),
                    1 / MAlpha_s(Mt, 6) * Alpha_sNf(R, 5)
                )
            )
        )
    )
    return result


@jax.jit
def Alpha_s1Loop(R):
    inv_R = 1 / R

    result = np.where(
        inv_R < 1,
        Alpha_sNf1Loop(1, 3),
        np.where(
            inv_R < Mc,
            Alpha_sNf1Loop(R, 3),
            np.where(
                inv_R < Mb,
                Alpha_sNf1Loop(R, 4),
                np.where(
                    inv_R < Mt,
                    Alpha_sNf1Loop(R, 5),
                    Alpha_sNf1Loop(R, 6)
                )
            )
        )
    )
    return result

@jax.jit
def Alpha_s2Loop(R):
    inv_R = 1 / R

    result = np.where(
        inv_R < 1,
        Alpha_sNf2Loop(1, 3), 
        np.where(
            inv_R < Mc,
            Alpha_sNf2Loop(R, 3),  
            np.where(
                inv_R < Mb,
                Alpha_sNf2Loop(R, 4), 
                np.where(
                    inv_R < Mt,
                    Alpha_sNf2Loop(R, 5), 
                    Alpha_sNf2Loop(R, 6) 
                )
            )
        )
    )
    return result

@jax.jit
def Alpha_s3Loop(R):
    inv_R = 1 / R

    result = np.where(
        inv_R < 1,
        MAlpha_s2Loop(Mc, 4) * MAlpha_s2Loop(Mb, 5) * Alpha_sNf3Loop(1, 5),
        np.where(
            inv_R < Mc,
            MAlpha_s2Loop(Mc, 4) * MAlpha_s2Loop(Mb, 5) * Alpha_sNf3Loop(R, 5),
            np.where(
                inv_R < Mb,
                MAlpha_s2Loop(Mb, 5) * Alpha_sNf3Loop(R, 5),
                np.where(
                    inv_R < Mt,
                    Alpha_sNf3Loop(R, 5),
                    1 / MAlpha_s2Loop(Mt, 6) * Alpha_sNf3Loop(R, 5)
                )
            )
        )
    )
    return result



#muPrime = mu + 1/R #???
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

#CF = (Nc**2 - 1)/(2*Nc)
#VB = alpha*CF/(Nc-1) 
# Importante: bellow we have implemented a mu dependence where it wasn't before
VB = CF * alpha / (Nc - 1) 


print('VB is ', VB)
SingC3 = -(Nc+1)/8

#a0=4.514

#VB=.1
#print(VB)
#quit()
# imaginary time points for GFMC evolution
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)

L = log_mu_r + np.log(mufac)
Vb_LO = VB
Vb_NLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L))
Vb_NNLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L) + (alpha/(4*np.pi))**2*( beta0**2*(4*L**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*L + aa2 ) )

#######################################################################################
#Rprime = lambda R: adl.norm_3vec(R)*np.exp(np.euler_gamma)*mu

VB_LO = lambda R: CF * VB_MRS_definition(Alpha_s1Loop(R), 1, mufac) / (Nc - 1) ############################### NEW to be independent of the change in mu

#VB_NLO(R) = Alpha_s2Loop(R)*CF/(Nc-1) * (1 + Alpha_s2Loop(R)/(4*np.pi)*(aa1 + 2*beta0*log_mu_r))
#VB_NNLO(R) = Alpha_s3Loop(R)*CF/(Nc-1) * (1 + Alpha_s3Loop(R)/(4*np.pi)*(aa1 + 2*beta0*L) + (Alpha_s3Loop(R)/(4*np.pi))**2*( beta0**2*(4*L**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*L + aa2 ) )

VB_NLO = lambda R: CF * VB_MRS_definition(Alpha_s2Loop(R), 2, mufac) / (Nc - 1)
VB_NNLO = lambda R: CF * VB_MRS_definition(Alpha_s3Loop(R), 3, mufac) / (Nc - 1)

#ANTISYMMETRIC AND SYMMETRIC:

VB_NNLO_antisym = lambda R: CF * VB_MRS_definition(Alpha_s3Loop(R), 3, mufac, 'antisymmetric') / (Nc - 1)
VB_NNLO_sym = lambda R: CF * VB_MRS_definition(Alpha_s3Loop(R), 3, mufac, 'symmetric') / (Nc + 1)


#OCTET :

VB_NNLO_octet = lambda R: (CA/2 - CF) * VB_MRS_definition(Alpha_s3Loop(R), 3, mufac, 'octet') 

#######################################################################################

def find_mQ(Alpha_sLoop, alpha, fudge=4):
    def equation_to_solve(mu):
        return Alpha_sLoop(1/mu) - alpha
    t = root_scalar(equation_to_solve, x0=3, method='brentq', bracket=[0.0001, 100]).root
    #print(t)
    
    def equation_to_solve(mQ):
        return t - fudge * Alpha_sLoop(1/t) * mQ
    mQ = root_scalar(equation_to_solve, x0=1000, method='brentq', bracket=[0.0001, 100000]).root
    #print(mQ)
    return mQ


if OLO == "LO":
    mQ = find_mQ(Alpha_s1Loop, alpha)
elif OLO == "NLO":
    mQ = find_mQ(Alpha_s2Loop, alpha)
elif OLO == "NNLO":
    mQ = find_mQ(Alpha_s3Loop, alpha)
elif OLO == "NNNLO":
    mQ = find_mQ(Alpha_s, alpha)
    
print('mQ is ', mQ)
#######################################################################################


VMRS_singulet_LO = -1 * (Nc - 1) * VB_LO(r_test) / r_test
VMRS_singulet_NLO = -1 * (Nc - 1) * VB_NLO(r_test) / r_test

print('VMRS_singulet_LO is ', VMRS_singulet_LO)
print('VMRS_singulet_NLO is ', VMRS_singulet_NLO)


VMRS_antisym_LO = -1 * VB_LO(r_test) / r_test
VMRS_antisym_NLO = -1 * VB_NLO(r_test) / r_test

print('VMRS_antisym_LO is ', VMRS_antisym_LO)
print('VMRS_antisym_NLO is ', VMRS_antisym_NLO)

assert onp.allclose(VMRS_singulet_LO,V_0_r_test), "not close enough"
#onp.allclose(VMRS_singulet_LO,V_0(r_test)) 
