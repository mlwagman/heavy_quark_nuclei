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
#import afdmc_lib as adl
import os
#from afdmc_lib import NI,NS,mp_Mev,fm_Mev
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
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--log_mu_r', type=float, default=0)
parser.add_argument('--Nc', type=int, default=3)
parser.add_argument('--N_coord', type=int, default=2)
parser.add_argument('--nf', type=int, default=5)
globals().update(vars(parser.parse_args()))

#######################################################################################

L = log_mu_r

CF = (Nc**2 - 1)/(2*Nc)
VB = alpha*CF
if N_coord > 2:
    VB = alpha*CF/(Nc-1)
SingC3 = -(Nc+1)/8
cutoff = 1;
a0 = 2/VB;


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

VB_LO = VB

VB_NLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L))

VB_NNLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L) + (alpha/(4*np.pi))**2*( beta0**2*(4*L**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*L + aa2 ) )
if N_coord > 2:
   VB_NNLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*L) + (alpha/(4*np.pi))**2*( beta0**2*(4*L**2 + np.pi**2/3) + 2*( beta1+2*beta0*aa1 )*L + aa2 + Nc*(Nc-2)/2*((np.pi)**4-12*(np.pi)**2)  ) )

print("LO a = ", 2/VB_LO)
print("NLO a = ", 2/VB_NLO)
print("NNLO a = ", 2/VB_NNLO)
