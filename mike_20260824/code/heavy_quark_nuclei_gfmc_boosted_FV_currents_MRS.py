### GFMC form factor script with MRS (renormalon-subtracted) potential
### Based on heavy_quark_nuclei_gfmc_boosted_FV_currents.py
### MRS potential adapted from simon/heavy_quark_nuclei_gfmc_MRS_mu_prime.py

import argparse
import analysis as al
import matplotlib
matplotlib.use('Agg')  # headless: never pop GUI windows
import matplotlib.pyplot as plt
import numpy as onp
import scipy
import scipy.special as spl
import scipy.interpolate
import scipy.integrate
from scipy.optimize import root_scalar
import jax.scipy
import jax.scipy.special
import paper_plt
import tqdm.auto as tqdm
import afdmc_lib_col as adl
import os
from afdmc_lib_col import NI,NS,mp_Mev,fm_Mev
import jax
import jax.numpy as np
import sys
import time
import h5py
import math
import mpmath
from functools import partial
import timeout_decorator

onp.random.seed(0)
import shutil
if shutil.which("latex"):
    paper_plt.load_latex_config()

parser = argparse.ArgumentParser()
parser.add_argument('--n_walkers', type=int, default=1000)
parser.add_argument('--dtau_iMev', type=float, required=True)
parser.add_argument('--n_step', type=int, required=True)
parser.add_argument('--n_skip', type=int, default=100)
parser.add_argument('--resampling', type=int, default=None)
parser.add_argument('--mQ', type=float, default=1.0, help='Heavy quark mass (sets dimensionless scale)')
parser.add_argument('--Rstar', type=float, default=0.35, help='Reference separation for a0 determination')
parser.add_argument('--mufac', type=float, default=1.0, help='Scale factor for MRS potential')
parser.add_argument('--mu', type=float, default=2.0, help='IR cutoff scale')
parser.add_argument('--Nc', type=int, default=3)
parser.add_argument('--N_coord', type=int, default=3)
parser.add_argument('--nf', type=int, default=0)
parser.add_argument('--alpha_s', type=float, default=None, help='alpha_s at quark mass scale mQ (dark sector coupling; replaces SM running if set)')
parser.add_argument('--OLO', type=str, default="LO")
parser.add_argument('--radial_n', type=int, default=1)
parser.add_argument('--spoila', type=float, default=1)
parser.add_argument('--g', type=float, default=0)
parser.add_argument('--outdir', type=str, required=True)
parser.add_argument('--input_Rs_database', type=str, default="")
parser.add_argument('--log_mu_r', type=float, default=1)
parser.add_argument('--L', type=float, default=0.0)
parser.add_argument('--Lcut', type=int, default=5)
parser.add_argument('--spoilS', type=float, default=1)
parser.add_argument('--wavefunction', type=str, default="compact")
parser.add_argument('--color', type=str, default="1x1")
parser.add_argument('--potential', type=str, default="full")
parser.add_argument('--masses', type=float, default=0., nargs='+')
parser.add_argument('--Q_mag', type=float, default=0.1, help='Magnitude of current operator momentum')
parser.add_argument('--Q_dir', type=float, nargs=3, default=[1, 0, 0], help='Direction of current momentum')
parser.add_argument('--p_leg1_mag', type=float, default=0.0, help='Magnitude of leg-1 boost momentum')
parser.add_argument('--p_leg1_dir', type=float, nargs=3, default=[0, 0, 0], help='Direction of leg-1 momentum')
parser.add_argument('--count_skip', type=int, default=1)
parser.add_argument('--tau_min_plateau', type=float, default=None, help='Start of plateau region (default: 40% of max tau)')
parser.add_argument('--tau_max_plateau', type=float, default=None, help='End of plateau region (default: max tau)')
parser.add_argument('--seed', type=int, default=0, help='RNG seed (vary across parallel jobs for independent walker batches)')
globals().update(vars(parser.parse_args()))
onp.random.seed(seed)

# only mesons and baryons with compact wavefunction
assert N_coord in (2,3), "Only mesons (N_coord=2) and baryons (N_coord=3) supported"
assert wavefunction == "compact", "Only compact wavefunction supported"

#######################################################################################

volume = "infinite"
if L > 1e-2:
    volume = "finite"

# leg-1 state momentum, split equally among particles
p_leg1_dir_arr = onp.array(p_leg1_dir)
p_leg1_norm = onp.linalg.norm(p_leg1_dir_arr)
if p_leg1_norm > 1e-10:
    p_leg1_dir_arr = p_leg1_dir_arr / p_leg1_norm

if volume == "finite":
    mtm_single = p_leg1_dir_arr * p_leg1_mag * (2*onp.pi/L) / N_coord
else:
    mtm_single = p_leg1_dir_arr * p_leg1_mag / N_coord

mtm = onp.tile(mtm_single, (N_coord, 1))
mtm = np.array(mtm)

# current operator momentum: total momentum transfer q
Q_dir_arr = onp.array(Q_dir)
Q_dir_norm = onp.linalg.norm(Q_dir_arr)
if Q_dir_norm > 1e-10:
    Q_dir_arr = Q_dir_arr / Q_dir_norm

if volume == "finite":
    q_total = Q_dir_arr * Q_mag * (2*onp.pi/L)
else:
    q_total = Q_dir_arr * Q_mag

q_total = np.array(q_total)

# per-particle share of momentum transfer, for wavefunction phase ratio
q_vec_single = q_total / N_coord
q_vec = onp.tile(q_vec_single, (N_coord, 1))
q_vec = np.array(q_vec)

print("=" * 70)
print("MOMENTUM SETUP")
print("=" * 70)
print(f"Leg-1 momentum per particle: {onp.array(mtm[0])}")
print(f"Current TOTAL momentum q: {onp.array(q_total)}")
print(f"Current momentum per particle (wavefunction): {onp.array(q_vec[0])}")
print(f"|q|^2 = {float(np.sum(q_total**2)):.6f}")
print("=" * 70)

if masses == 0.:
    masses = onp.ones(N_coord)
    if N_coord == 2:
        masses = [1,-1]
    elif N_coord == 3:
        masses = [1,1,1]

masses_np = np.array(masses)
absmasses = np.abs(masses_np)

print(f"masses = {masses}")
print(f"|masses| = {onp.array(absmasses)}")
print(f"spatial wavefunction = {wavefunction}")
print(f"color wavefunction = {color}")

###################################################################################################################
# MRS potential: renormalon-subtracted scheme
# Adapted from simon/heavy_quark_nuclei_gfmc_MRS_mu_prime.py
###################################################################################################################

Tf = 1/2
CA = Nc
CF = (Nc**2 - 1)/(2*Nc)
Pi = onp.pi
NA = Nc**2 - 1

def Zeta(s):
    return spl.zeta(s, 1)

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
dAA = Nc**2 * (Nc**2 + 36) / 24
alpha4 = float(mpmath.polylog(4,1/2))+(-onp.log(2))**4/(4*3*2*1)
ss6 = zeta51+zeta6
aa30 = dFA*( Pi**2*( 7432/9-4736*alpha4+onp.log(2)*(14752/3-3472*zeta3)-6616*zeta3/3)  +  Pi**4*(-156+560*onp.log(2)/3+496*onp.log(2)**2/3)+1511*Pi**6/45)  + Nc**3*(385645/2916 + Pi**2*( -953/54 +584/3*alpha4 +175/2*zeta3 + onp.log(2)*(-922/9+217*zeta3/3) ) +584*zeta3/3 + Pi**4*( 1349/270-20*onp.log(2)/9-40*onp.log(2)**2/9 ) -1927/6*zeta5 -143/2*zeta3**2-4621/3024*Pi**6+144*ss6  )
aa31 = dFF*( Pi**2*(1264/9-976*zeta3/3+onp.log(2)*(64+672*zeta3)) + Pi**4*(-184/3+32/3*onp.log(2)-32*onp.log(2)**2) +10/3*Pi**6 ) + CF**2/2*(286/9+296/3*zeta3-160*zeta5)+Nc*CF/2*(-71281/162+264*zeta3+80*zeta5)+Nc**2/2*(-58747/486+Pi**2*(17/27-32*alpha4+onp.log(2)*(-4/3-14*zeta3)-19/3*zeta3)-356*zeta3+Pi**4*(-157/54-5*onp.log(2)/9+onp.log(2)**2)+1091*zeta5/6+57/2*zeta3**2+761*Pi**6/2520-48*ss6)
aa32 = Nc/4*(12541/243+368/3*zeta3+64*Pi**4/135)+CF/4*(14002/81-416*zeta3/3)
aa33 = -(20/9)**3*1/8
aa3 = aa30+aa31*nf+aa32*nf**2+aa33*nf**3


def Beta(k):
    if k == 0:
        return 1 / (4 * Pi)**(k+1) * beta0
    elif k == 1:
        return 1 / (4 * Pi)**(k+1) * beta1
    elif k == 2:
        return 1 / (4 * Pi)**(k+1) * beta2
    else:
        return None


def delta(k):
    return Beta(k) / Beta(0) - (Beta(1) / Beta(0))**k


def a(k, alpha=1):
    if k == 0:
        return 1
    elif k == 1:
        return 1/(4 * Pi) * aa1
    elif k == 2:
        return 1/(4 * Pi)**2 * aa2
    elif k == 3:
        aa3_us = 16/3 * Pi**2 * CA**3 * (onp.log(CA/2 * alpha) - 5/6 + onp.euler_gamma + onp.log(2))
        return 1/(4 * Pi)**3 * (aa3 + aa3_us)
    else:
        return None


def L_fun(n):
    if n == 1:
        return onp.euler_gamma
    elif n == 2:
        return onp.euler_gamma**2 + 0.5 * Zeta(2)
    elif n == 3:
        return onp.euler_gamma**3 + 1.5 * Zeta(2) * onp.euler_gamma + 2 * Zeta(3)
    else:
        return None


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
        else:
            return singulet
    else:
        return None


def vl(l, s, o, alpha=1):
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
        return None


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
        return None


def fK(l, s, o, alpha):
    if l == 3:
        additional_term = delta(2) * fl(0, s, o)
        return fl(3, s, o, alpha) + additional_term
    else:
        return fl(l, s, o)


b = Beta(1) / (2 * Beta(0)**2)


def V0(L_pert, s, o, alpha):
    return sum(fK(k, s, o, alpha) * spl.gamma(1 + b) / spl.gamma(k + 2 + b) * (k + 1) / (2 * Beta(0))**k for k in range(0, L_pert))


def QF(l):
    return (2 * Beta(0))**l * spl.gamma(b + l + 1) / spl.gamma(b + 1)


def Vl(l, L_pert, s, o, alpha):
    return V0(L_pert, s, o, alpha) * QF(l)


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
        result = sum((vl_list[i] - Vl_list[i]) * alpha**(i+1) for i in range(0, L_pert))
        return result

    def Javad_jax_real(a, alpha):
        x = 1/(2 * Beta(0) * alpha)
        term1 = np.exp(-x)

        def lower_incomplete_gamma(a_value, x_values):

            @partial(jax.jit)
            def series_sum(a, x, tolerance=1e-10, max_iter=10000):
                y = -x
                def body_fun(val):
                    n, total, last_term = val
                    n_fact = jax.scipy.special.gamma(n + 1)
                    current_term = (y ** n) / (n_fact * (n + a))
                    new_total = total + current_term
                    return (n + 1, new_total, current_term)

                def cond_fun(val):
                    n, total, last_term = val
                    return (np.max(np.abs(last_term)) > tolerance) & (n < max_iter)

                n0 = 0
                total0 = np.zeros_like(y)
                last_term0 = np.ones_like(y) * np.inf

                _, total, _ = jax.lax.while_loop(cond_fun, body_fun, (n0, total0, last_term0))
                return total

            result = series_sum(a_value, x_values)
            return result

        term3 = lower_incomplete_gamma(-a, -x)
        result = term1 * term3
        return np.real(result)

    def Javad_float_real(a, alpha):
        x = 1/(2 * Beta(0) * alpha)
        term1 = mpmath.exp(-x)
        term3 = mpmath.gammainc(-a, 0, -x, regularized=False)
        term4 = (-x)**a
        result = term1 * term3 * term4
        return float(result.real)

    @timeout_decorator.timeout(5, timeout_exception=StopIteration)
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

    return (V_RS(alpha, L_pert, s, o) + V_Borel(alpha, L_pert, s, o))


###################################################################################################################
# Lambda QCD and running coupling (4-loop MSbar)
###################################################################################################################

def Beta0(Nf):
    return (1 / (4 * Pi)) * (11/3 * CA - 2/3 * Nf)

def Beta1(Nf):
    return (1 / (4 * Pi))**2 * (34/3 * CA**2 - 10/3 * CA * Nf - 2 * CF * Nf)

def Beta2(Nf):
    return (1 / (4 * Pi))**3 * (2857/54 * CA**3 - 1415/27 * CA**2 * Nf / 2 +
         158/27 * CA * Nf**2 / 4 + 44/9 * CF * Nf**2 / 4 -
         205/9 * CF * CA * Nf / 2 + CF**2 * Nf)

def Beta3(Nf):
    return (1 / (4 * Pi))**4 * (CA * CF * Nf**2 / 4 * (17152/243 + 448/9 * Zeta(3)) +
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

Mt = 173.34
Mb = 4.7
Mc = 1.5
MZ = 91.1876

def LambdaQCD_MSbar(f):
    if f == 3: return 0.338
    elif f == 4: return 0.296
    elif f == 5: return 0.213
    elif f == 6: return 0.090
    else: return None


def find_Lambda():

    def LQ_MSbar(Q, f):
        return onp.log(Q**2 / LambdaQCD_MSbar(f)**2)

    def Alpha_sNf_find(Q, f):
        LQ_val = LQ_MSbar(Q, f)
        return 1 / (Beta0(f) * LQ_val) - \
               Beta1(f) * onp.log(LQ_val) / (Beta0(f)**3 * LQ_val**2) + \
               1 / (Beta0(f)**3 * LQ_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (onp.log(LQ_val)**2 - onp.log(LQ_val) - 1) + Beta2(f) / Beta0(f)) + \
               1 / (Beta0(f)**4 * LQ_val**4) * (Beta1(f)**3 / Beta0(f)**3 * (-onp.log(LQ_val)**3 + 2.5 * onp.log(LQ_val)**2 + 2 * onp.log(LQ_val) - 0.5) - \
               3 * Beta1(f) * Beta2(f) * onp.log(LQ_val) / Beta0(f)**2 + Beta3(f) / (2 * Beta0(f)))

    def MAlpha_s_find(Q, f):
        alpha_s = Alpha_sNf_find(Q, f)
        return 1 + (-0.291667) * (alpha_s / math.pi)**2 + (-5.32389 + (f - 1) * 0.26247) * (alpha_s / math.pi)**3

    def Alpha_s_find(Q):
        if Q < 1:
            return MAlpha_s_find(Mc, 4) * MAlpha_s_find(Mb, 5) * Alpha_sNf_find(1, 5)
        elif 1 <= Q < Mc:
            return MAlpha_s_find(Mc, 4) * MAlpha_s_find(Mb, 5) * Alpha_sNf_find(Q, 5)
        elif Mc <= Q < Mb:
            return MAlpha_s_find(Mb, 5) * Alpha_sNf_find(Q, 5)
        elif Mb <= Q < Mt:
            return Alpha_sNf_find(Q, 5)
        elif Q >= Mt:
            return 1 / MAlpha_s_find(Mt, 6) * Alpha_sNf_find(Q, 5)
        else:
            return None

    def LQs_MSbar(Q, Lambda):
        return onp.log(Q**2 / Lambda**2)

    def Alpha_sNf_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val) - \
               Beta1(f) * onp.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
               1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (onp.log(LR_val)**2 - onp.log(LR_val) - 1) + Beta2(f) / Beta0(f)) + \
               1 / (Beta0(f)**4 * LR_val**4) * (Beta1(f)**3 / Beta0(f)**3 * (-onp.log(LR_val)**3 + 2.5 * onp.log(LR_val)**2 + 2 * onp.log(LR_val) - 0.5) - \
               3 * Beta1(f) * Beta2(f) * onp.log(LR_val) / Beta0(f)**2 + Beta3(f) / (2 * Beta0(f)))

    def Alpha_sNf3Loop_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val) - \
               Beta1(f) * onp.log(LR_val) / (Beta0(f)**3 * LR_val**2) + \
               1 / (Beta0(f)**3 * LR_val**3) * (Beta1(f)**2 / Beta0(f)**2 * (onp.log(LR_val)**2 - onp.log(LR_val) - 1) + Beta2(f) / Beta0(f))

    def Alpha_sNf2Loop_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val) - Beta1(f) * onp.log(LR_val) / (Beta0(f)**3 * LR_val**2)

    def Alpha_sNf1Loop_L(Q, f, Lambda):
        LR_val = LQs_MSbar(Q, Lambda)
        return 1 / (Beta0(f) * LR_val)

    # 4-loop Lambda
    result = root_scalar(lambda L: Alpha_sNf_L(MZ, 5, L) - Alpha_s_find(MZ), x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])
    Lambda4LoopNf5 = result.root
    result = root_scalar(lambda L: Alpha_sNf_L(Mt, 6, L) - Alpha_sNf_L(Mt, 5, Lambda4LoopNf5), x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])
    Lambda4LoopNf6 = result.root
    result = root_scalar(lambda L: Alpha_sNf_L(Mb, 4, L) - Alpha_sNf_L(Mb, 5, Lambda4LoopNf5), x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])
    Lambda4LoopNf4 = result.root
    result = root_scalar(lambda L: Alpha_sNf_L(Mc, 3, L) - Alpha_sNf_L(Mc, 4, Lambda4LoopNf4), x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])
    Lambda4LoopNf3 = result.root
    Lambda4Loop = [Lambda4LoopNf6, Lambda4LoopNf5, Lambda4LoopNf4, Lambda4LoopNf3]

    # 3-loop Lambda
    result = root_scalar(lambda L: Alpha_sNf3Loop_L(MZ, 5, L) - Alpha_s_find(MZ), x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])
    Lambda3LoopNf5 = result.root
    result = root_scalar(lambda L: Alpha_sNf3Loop_L(Mt, 6, L) - Alpha_sNf3Loop_L(Mt, 5, Lambda3LoopNf5), x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])
    Lambda3LoopNf6 = result.root
    result = root_scalar(lambda L: Alpha_sNf3Loop_L(Mb, 4, L) - Alpha_sNf3Loop_L(Mb, 5, Lambda3LoopNf5), x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])
    Lambda3LoopNf4 = result.root
    result = root_scalar(lambda L: Alpha_sNf3Loop_L(Mc, 3, L) - Alpha_sNf3Loop_L(Mc, 4, Lambda3LoopNf4), x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])
    Lambda3LoopNf3 = result.root
    Lambda3Loop = [Lambda3LoopNf6, Lambda3LoopNf5, Lambda3LoopNf4, Lambda3LoopNf3]

    # 2-loop Lambda
    result = root_scalar(lambda L: Alpha_sNf2Loop_L(MZ, 5, L) - Alpha_s_find(MZ), x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])
    Lambda2LoopNf5 = result.root
    result = root_scalar(lambda L: Alpha_sNf2Loop_L(Mt, 6, L) - Alpha_sNf2Loop_L(Mt, 5, Lambda2LoopNf5), x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])
    Lambda2LoopNf6 = result.root
    result = root_scalar(lambda L: Alpha_sNf2Loop_L(Mb, 4, L) - Alpha_sNf2Loop_L(Mb, 5, Lambda2LoopNf5), x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])
    Lambda2LoopNf4 = result.root
    result = root_scalar(lambda L: Alpha_sNf2Loop_L(Mc, 3, L) - Alpha_sNf2Loop_L(Mc, 4, Lambda2LoopNf4), x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])
    Lambda2LoopNf3 = result.root
    Lambda2Loop = [Lambda2LoopNf6, Lambda2LoopNf5, Lambda2LoopNf4, Lambda2LoopNf3]

    # 1-loop Lambda
    result = root_scalar(lambda L: Alpha_sNf1Loop_L(MZ, 5, L) - Alpha_s_find(MZ), x0=LambdaQCD_MSbar(5), method='brentq', bracket=[0.0001, 1])
    Lambda1LoopNf5 = result.root
    result = root_scalar(lambda L: Alpha_sNf1Loop_L(Mt, 6, L) - Alpha_sNf1Loop_L(Mt, 5, Lambda1LoopNf5), x0=LambdaQCD_MSbar(6), method='brentq', bracket=[0.0001, 1])
    Lambda1LoopNf6 = result.root
    result = root_scalar(lambda L: Alpha_sNf1Loop_L(Mb, 4, L) - Alpha_sNf1Loop_L(Mb, 5, Lambda1LoopNf5), x0=LambdaQCD_MSbar(4), method='brentq', bracket=[0.0001, 1])
    Lambda1LoopNf4 = result.root
    result = root_scalar(lambda L: Alpha_sNf1Loop_L(Mc, 3, L) - Alpha_sNf1Loop_L(Mc, 4, Lambda1LoopNf4), x0=LambdaQCD_MSbar(3), method='brentq', bracket=[0.0001, 1])
    Lambda1LoopNf3 = result.root
    Lambda1Loop = [Lambda1LoopNf6, Lambda1LoopNf5, Lambda1LoopNf4, Lambda1LoopNf3]

    return Lambda4Loop, Lambda3Loop, Lambda2Loop, Lambda1Loop


if alpha_s is not None:
    # Dark sector: compute Lambda_dark from alpha_s(mQ) via 1-loop relation
    # alpha_s(mQ) = 1 / (Beta0(0) * log(mQ^2/Lambda_dark^2))
    # => Lambda_dark = mQ * exp(-1 / (2*Beta0(0)*alpha_s))
    Lambda_dark = float(mQ) * onp.exp(-1.0 / (2.0 * Beta0(0) * alpha_s))
    # NLO runs the coupling at TWO loops, so it needs its own Lambda anchored to
    # the same physical input alpha_s(mQ) -- this is what find_Lambda() does per
    # loop order in the SM branch. Solve
    #     alpha_s = 1/(b0 L) - b1 ln(L)/(b0^3 L^2),   L = ln(mQ^2/Lambda^2)
    # for L, then Lambda = mQ exp(-L/2). Reusing the 1-loop Lambda would leave
    # alpha_s^{2-loop}(mQ) != alpha_s(mQ).
    _b0, _b1 = float(Beta0(0)), float(Beta1(0))
    _L1 = 1.0 / (_b0 * alpha_s)                      # 1-loop value of L
    _sol = root_scalar(
        lambda L: 1.0/(_b0*L) - _b1*onp.log(L)/(_b0**3 * L*L) - alpha_s,
        bracket=[0.5*_L1, _L1], method='brentq')
    Lambda_dark_2loop = float(mQ) * onp.exp(-0.5 * _sol.root)
    # dummy arrays so LambdaNLoopNf helpers don't crash if called
    Lambda4Loop = Lambda3Loop = Lambda2Loop = Lambda1Loop = [Lambda_dark] * 4
else:
    Lambda4Loop, Lambda3Loop, Lambda2Loop, Lambda1Loop = find_Lambda()

def Lambda4LoopNf(f): return Lambda4Loop[6-f]
def Lambda3LoopNf(f): return Lambda3Loop[6-f]
def Lambda2LoopNf(f): return Lambda2Loop[6-f]
def Lambda1LoopNf(f): return Lambda1Loop[6-f]

def LR_4Loop_MSbar(muPrime, f): return np.log(muPrime**2 / Lambda4LoopNf(f)**2)
def LR_3Loop_MSbar(muPrime, f): return np.log(muPrime**2 / Lambda3LoopNf(f)**2)
def LR_2Loop_MSbar(muPrime, f): return np.log(muPrime**2 / Lambda2LoopNf(f)**2)
def LR_1Loop_MSbar(muPrime, f): return np.log(muPrime**2 / Lambda1LoopNf(f)**2)


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
    return 1 / (Beta0(f) * LR_val) - Beta1(f) * np.log(LR_val) / (Beta0(f)**3 * LR_val**2)

def Alpha_sNf1Loop(muPrime, f):
    LR_val = LR_1Loop_MSbar(muPrime, f)
    return 1 / (Beta0(f) * LR_val)

def MAlpha_s(Q, f):
    alpha_s = Alpha_sNf(Q, f)
    return 1 + (-0.291667) * (alpha_s / math.pi)**2 + (-5.32389 + (f - 1) * 0.26247) * (alpha_s / math.pi)**3

def MAlpha_s2Loop(Q, f):
    return 1 + (-0.291667) * (Alpha_sNf3Loop(Q, f) / math.pi)**2


def mu_Prime(R):
    return np.sqrt(mu**2 + 1/R**2)


if alpha_s is not None:
    # Dark sector: simple 1-loop running at nf=0, anchored at alpha_s(mQ)
    @jax.jit
    def Alpha_s1Loop(R):
        muPrime = mu_Prime(R)
        LR = np.log(muPrime**2 / Lambda_dark**2)
        return 1.0 / (Beta0(0) * LR)
else:
    @jax.jit
    def Alpha_s1Loop(R):
        muPrime = mu_Prime(R)
        result = np.where(
            muPrime < 1,
            Alpha_sNf1Loop(muPrime, 3),
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


if alpha_s is not None:
    # Dark sector: 2-loop running at nf=0, anchored at alpha_s(mQ).
    # This branch was MISSING: Alpha_s2Loop fell through to the SM flavour tree
    # below, which picks nf=3..6 by scale. With mu_cut=2 and mQ=1, mu' is always
    # in [2, Mb), so it always chose nf=4 and --nf 0 never reached the NLO
    # coupling. VB_NLO/VB_LO then drifted to Beta0(0)/Beta0(4)=1.32 instead of
    # ->1 as alpha_s->0, and MRS NLO failed to reduce to Coulomb at weak coupling.
    @jax.jit
    def Alpha_s2Loop(R):
        muPrime = mu_Prime(R)
        LR = np.log(muPrime**2 / Lambda_dark_2loop**2)
        return (1.0 / (Beta0(0) * LR)
                - Beta1(0) * np.log(LR) / (Beta0(0)**3 * LR**2))
else:
  @jax.jit
  def Alpha_s2Loop(R):
    muPrime = mu_Prime(R)
    result = np.where(
        muPrime < 1,
        Alpha_sNf2Loop(muPrime, 3),
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


VB_LO = lambda R: CF * VB_MRS_definition(Alpha_s1Loop(R), 'LO', 1) / (Nc - 1)
VB_NLO = lambda R: CF * VB_MRS_definition(Alpha_s2Loop(R), 'NLO', 1) / (Nc - 1)

###################################################################################################################

SingC3 = -(Nc+1)/8

# imaginary time points per leg
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)

# a0 from MRS potential at Rstar
if OLO == "LO":
    a0 = spoila * 2 / VB_LO(Rstar)
elif OLO == "NLO":
    a0 = spoila * 2 / VB_NLO(Rstar)
else:
    raise ValueError("Only LO and NLO supported")

if N_coord == 2:
    a0 /= Nc-1

print(f"mQ = {mQ}")
print(f"Rstar = {Rstar}")
print(f"mufac = {mufac}")
print(f"mu = {mu}")
if alpha_s is not None:
    print(f"alpha_s (at mQ) = {alpha_s}  [dark sector mode, nf=0]")
    print(f"Lambda_dark = {Lambda_dark:.6f}  [= mQ*exp(-1/(2*Beta0(0)*alpha_s))]")
    print(f"alpha_s(Rstar) = {float(Alpha_s1Loop(Rstar)):.6f}  [running value at Rstar]")
print(f"VB_LO(Rstar) = {VB_LO(Rstar)}")
print(f"a0 = {a0}")

os.makedirs(outdir, exist_ok=True)

pp = np.array([np.array([i, j, k]) for i in range(-Lcut, Lcut+1) for j in range(-Lcut, Lcut+1) for k in range(-Lcut, Lcut+1)])
nn = np.delete(pp, Lcut*(2*Lcut+1)*(2*Lcut+1)+Lcut*(2*Lcut+1)+Lcut, axis=0)

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
    return sums/(np.pi*L)

# MRS potentials
AV_Coulomb = {}
B3_Coulomb = {}

if OLO == "LO":
    @partial(jax.jit)
    def potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return -1 * VB_LO(norm_R / (mufac * mQ)) / norm_R

    @partial(jax.jit)
    def symmetric_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return spoilS * (Nc - 1) / (Nc + 1) * VB_LO(norm_R / (mufac * mQ)) / norm_R

    @partial(jax.jit)
    def singlet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return -1 * (Nc - 1) * VB_LO(norm_R / (mufac * mQ)) / norm_R

    @partial(jax.jit)
    def octet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return spoilS * (Nc - 1) / CF / (2 * Nc) * VB_LO(norm_R / (mufac * mQ)) / norm_R

    @partial(jax.jit)
    def potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        return -1 * VB_LO(norm_R / (mufac * mQ)) * FV_Coulomb(R, L, nn)

    @partial(jax.jit)
    def symmetric_potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        return spoilS * (Nc - 1) / (Nc + 1) * VB_LO(norm_R / (mufac * mQ)) * FV_Coulomb(R, L, nn)

    @partial(jax.jit)
    def singlet_potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        return -1 * (Nc - 1) * VB_LO(norm_R / (mufac * mQ)) * FV_Coulomb(R, L, nn)

    @partial(jax.jit)
    def octet_potential_fun_sum(R):
        norm_R = adl.norm_3vec(R)
        return spoilS * (Nc - 1) / CF / (2 * Nc) * VB_LO(norm_R / (mufac * mQ)) * FV_Coulomb(R, L, nn)

elif OLO == "NLO":
    @partial(jax.jit)
    def potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return -1 * VB_NLO(norm_R / (mufac * mQ)) / norm_R

    @partial(jax.jit)
    def singlet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return -1 * (Nc - 1) * VB_NLO(norm_R / (mufac * mQ)) / norm_R

    def symmetric_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return spoilS * (Nc - 1) / (Nc + 1) * VB_NLO(norm_R / (mufac * mQ)) / norm_R

    def octet_potential_fun(R):
        norm_R = adl.norm_3vec(R)
        return spoilS * (Nc - 1) / CF / (2 * Nc) * VB_NLO(norm_R / (mufac * mQ)) / norm_R

print(f"volume = {volume}")

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

Coulomb_potential = adl.make_pairwise_potential(AV_Coulomb, B3_Coulomb, masses)

# trial wavefunction pieces
pairs = np.array([np.array([j, i]) for i in range(0,N_coord) for j in range(0, i)])

def f_R_braket(Rs):
    psi0 = onp.array(_psi_core_no_phase(Rs[None, ...]))[0]
    return float(onp.abs(psi0)**2)

def hydrogen_wvfn(r, n):
    psi = np.exp(-r/n)
    if n == 1: return psi
    if n == 2: return psi*(2 - r)
    if n == 3: return psi*(27 - 18*r + 2*r**2)
    return psi

@partial(jax.jit)
def _psi_core_no_phase(Rs, a0=a0, masses=absmasses):
    def r_norm(pair):
        [i,j] = pair
        rdiff = Rs[...,i,:] - Rs[...,j,:]
        mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
        rij_norm = np.sqrt( np.sum(rdiff*rdiff, axis=-1) )
        return rij_norm * mij
    r_sum = np.sum( jax.lax.map(r_norm, pairs), axis=0 )/a0
    psi = hydrogen_wvfn(r_sum, radial_n)
    return psi

def f_R(Rs, a0=a0, masses=absmasses):
    psi = _psi_core_no_phase(Rs, a0=a0, masses=masses)
    phase = np.exp(1j*np.einsum('...ai,ai->...', Rs, mtm))
    return psi * phase

def f_R_norm(Rs): return np.abs(_psi_core_no_phase(Rs))
def f_R_0(Rs):    return _psi_core_no_phase(Rs)

def f_R_q(Rs, q_vec):
    return _psi_core_no_phase(Rs) * np.exp(1j*np.einsum('...ai,ai->...', Rs, q_vec))

@partial(jax.jit)
def grad_f_R(Rs, a0=a0, masses=absmasses):
    """Gradient of full wavefunction (with momentum phase). Shape: (N_coord, 3, n_walkers)"""
    N_walkers = Rs.shape[0]
    N_coord_local = Rs.shape[1]
    if radial_n > 1:
        assert N_coord_local == 2
    grad_psi_tot = np.zeros((N_coord_local, 3, N_walkers))
    for x in range(3):
        for a in range(N_coord_local):
            for k in range(N_coord_local):
                for l in range(N_coord_local):
                    if k!=l and l>=k and (a==k or a==l):
                        grad_psi = 1
                        for i in range(N_coord_local):
                            for j in range(N_coord_local):
                                thisa0 = a0
                                if i!=j and j>=i:
                                    ri = Rs[...,i,:]
                                    rj = Rs[...,j,:]
                                    rij_norm = adl.norm_3vec(ri - rj)
                                    mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
                                    thisa0 /= mij
                                    rsign = 1 if a==i else (-1 if a==j else 0)
                                    if (k == i and l == j):
                                        grad_psi = rsign * grad_psi * (ri[:,x] - rj[:,x])/(thisa0*rij_norm) * np.exp(-rij_norm/thisa0)
                                    else:
                                        grad_psi = grad_psi * np.exp(-rij_norm/thisa0)
                        grad_psi_tot = grad_psi_tot.at[a,x,:].set(grad_psi_tot[a,x,:] + grad_psi / masses[a])
    phase = np.exp(1j*np.einsum('nai,ai->n', Rs, mtm))
    grad_psi_tot *= phase
    return grad_psi_tot

@partial(jax.jit)
def laplacian_f_R(Rs, a0=a0, masses=absmasses):
    """Total laplacian of wavefunction summed over all particles."""
    N_coord_local = Rs.shape[1]
    if radial_n > 1:
        assert N_coord_local == 2
    nabla_psi_tot = 0
    for k in range(N_coord_local):
        for l in range(N_coord_local):
            if k!=l and l>=k:
                nabla_psi = 1
                for i in range(N_coord_local):
                    for j in range(N_coord_local):
                        thisa0 = a0
                        if i!=j and j>=i:
                            ri = Rs[...,i,:]; rj = Rs[...,j,:]
                            rij_norm = adl.norm_3vec(ri - rj)
                            mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
                            thisa0 /= mij
                            if k == i and l == j:
                                nabla_psi = nabla_psi * ((1/(radial_n*thisa0)**2 - 2/(thisa0*rij_norm))/masses[k] + (1/(radial_n*thisa0)**2 - 2/(thisa0*rij_norm))/masses[l]) * hydrogen_wvfn(rij_norm/thisa0, radial_n)
                            else:
                                nabla_psi = nabla_psi * hydrogen_wvfn(rij_norm/thisa0, radial_n)
                nabla_psi_tot += nabla_psi
    for a in range(N_coord_local):
        for k in range(N_coord_local):
            for l in range(N_coord_local):
                if k!=l and l>=k and (a==k or a==l):
                    for m in range(N_coord_local):
                        for n in range(N_coord_local):
                            if m!=n and n>=m and (m!=k or n!=l) and (a==m or a==n):
                                for x in range(3):
                                    nabla_psi = 1
                                    for i in range(N_coord_local):
                                        for j in range(N_coord_local):
                                            thisa0 = a0
                                            if i!=j and j>=i:
                                                ri = Rs[...,i,:]; rj = Rs[...,j,:]
                                                rij_norm = adl.norm_3vec(ri - rj)
                                                mij = 2*masses[i]*masses[j]/(masses[i]+masses[j])
                                                thisa0 /= mij
                                                rsign = 1 if a==i else (-1 if a==j else 0)
                                                if (k == i and l == j) or (m == i and n == j):
                                                    nabla_psi = rsign * nabla_psi * (ri[:,x] - rj[:,x])/(thisa0*rij_norm) * np.exp(-rij_norm/thisa0)
                                                else:
                                                    nabla_psi = nabla_psi * np.exp(-rij_norm/thisa0)
                                    nabla_psi_tot += nabla_psi / masses[a]
    phase = np.exp(1j*np.einsum('nai,ai->n', Rs, mtm))
    nabla_psi_tot *= phase
    return nabla_psi_tot

def _phase(Rs, q_vec):
    return np.exp(1j*np.einsum('nai,ai->n', Rs, q_vec))

def grad_core(Rs): return grad_f_R(Rs) / _phase(Rs, mtm)[None, None, :]
def lap_core(Rs):  return laplacian_f_R(Rs) / _phase(Rs, mtm)


###
# current operator building blocks
###

@partial(jax.jit)
def phase_per_particle(Rs, q):
    dots = np.einsum('i,nki->nk', q, Rs)
    return np.exp(1j * dots)

@partial(jax.jit)
def grad_psi_over_psi_per_particle(Rs, a0=a0, masses=absmasses):
    N_walkers = Rs.shape[0]
    N_coord_local = Rs.shape[1]

    grad_ln_psi = np.zeros((N_coord_local, 3, N_walkers))

    for k in range(N_coord_local):
        for j in range(N_coord_local):
            if j != k:
                rk = Rs[:, k, :]
                rj = Rs[:, j, :]
                r_kj = rk - rj
                r_kj_norm = np.sqrt(np.sum(r_kj**2, axis=1))
                r_kj_norm_safe = np.maximum(r_kj_norm, 1e-6)
                m_kj = 2 * masses[k] * masses[j] / (masses[k] + masses[j])
                a0_kj = a0 / m_kj
                contrib = -(1.0 / a0_kj) * r_kj / r_kj_norm_safe[:, None]
                grad_ln_psi = grad_ln_psi.at[k, :, :].set(
                    grad_ln_psi[k, :, :] + contrib.T
                )

    return grad_ln_psi

@partial(jax.jit)
def lap_psi_over_psi_per_particle(Rs, a0=a0, masses=absmasses):
    N_walkers = Rs.shape[0]
    N_coord_local = Rs.shape[1]

    lap_over_psi = np.zeros((N_coord_local, N_walkers))

    for k in range(N_coord_local):
        grad_ln_psi_k = np.zeros((N_walkers, 3))
        lap_ln_psi_k = np.zeros(N_walkers)

        for j in range(N_coord_local):
            if j != k:
                rk = Rs[:, k, :]
                rj = Rs[:, j, :]
                r_kj = rk - rj
                r_kj_norm = np.sqrt(np.sum(r_kj**2, axis=1))
                r_kj_norm_safe = np.maximum(r_kj_norm, 1e-6)
                m_kj = 2 * masses[k] * masses[j] / (masses[k] + masses[j])
                a0_kj = a0 / m_kj
                grad_ln_psi_k += -(1.0 / a0_kj) * r_kj / r_kj_norm_safe[:, None]
                lap_ln_psi_k += -(2.0 / a0_kj) / r_kj_norm_safe

        grad_sq = np.sum(grad_ln_psi_k**2, axis=1)
        lap_over_psi = lap_over_psi.at[k, :].set(grad_sq + lap_ln_psi_k)

    return lap_over_psi


###
# current operators (NR reductions of Dirac bilinears)
###

def compute_J_S(Rs, q, masses=absmasses):
    phases = phase_per_particle(Rs, q)
    grad_over_psi = grad_psi_over_psi_per_particle(Rs)
    lap_over_psi = lap_psi_over_psi_per_particle(Rs)
    N_coord_local = Rs.shape[1]
    result = np.zeros(Rs.shape[0], dtype=np.complex128)
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (4.0 * m_k**2)
        iq_dot_grad = 1j * np.einsum('i,in->n', q, grad_over_psi[k])
        bracket = 1.0 + factor * (lap_over_psi[k] + iq_dot_grad)
        result += phases[:, k] * bracket
    return result

def compute_J_V0(Rs, q, masses=absmasses):
    phases = phase_per_particle(Rs, q)
    grad_over_psi = grad_psi_over_psi_per_particle(Rs)
    lap_over_psi = lap_psi_over_psi_per_particle(Rs)
    N_coord_local = Rs.shape[1]
    result = np.zeros(Rs.shape[0], dtype=np.complex128)
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (4.0 * m_k**2)
        iq_dot_grad = 1j * np.einsum('i,in->n', q, grad_over_psi[k])
        bracket = 1.0 - factor * (lap_over_psi[k] + iq_dot_grad)
        result += phases[:, k] * bracket
    return result

def compute_J_Vi(Rs, q, masses=absmasses):
    phases = phase_per_particle(Rs, q)
    grad_over_psi = grad_psi_over_psi_per_particle(Rs)
    N_coord_local = Rs.shape[1]
    N_walkers = Rs.shape[0]
    result = np.zeros((N_walkers, 3), dtype=np.complex128)
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (2.0 * m_k)
        for i in range(3):
            bracket = 2j * grad_over_psi[k, i, :] - q[i]
            result = result.at[:, i].set(result[:, i] + phases[:, k] * factor * bracket)
    return result

def compute_J_Ai(Rs, q, masses=absmasses):
    phases = phase_per_particle(Rs, q)
    grad_over_psi = grad_psi_over_psi_per_particle(Rs)
    N_coord_local = Rs.shape[1]
    N_walkers = Rs.shape[0]
    result = np.zeros((N_walkers, 3), dtype=np.complex128)
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (4.0 * m_k**2)
        grad_k = grad_over_psi[k]
        cross_x = q[1] * grad_k[2, :] - q[2] * grad_k[1, :]
        cross_y = q[2] * grad_k[0, :] - q[0] * grad_k[2, :]
        cross_z = q[0] * grad_k[1, :] - q[1] * grad_k[0, :]
        result = result.at[:, 0].set(result[:, 0] + phases[:, k] * factor * cross_x)
        result = result.at[:, 1].set(result[:, 1] + phases[:, k] * factor * cross_y)
        result = result.at[:, 2].set(result[:, 2] + phases[:, k] * factor * cross_z)
    return result

def compute_J_T0i(Rs, q, masses=absmasses):
    phases = phase_per_particle(Rs, q)
    N_coord_local = Rs.shape[1]
    N_walkers = Rs.shape[0]
    result = np.zeros((N_walkers, 3), dtype=np.complex128)
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (2.0 * m_k)
        for i in range(3):
            result = result.at[:, i].set(result[:, i] + phases[:, k] * factor * q[i])
    return result

def compute_J_Tij(Rs, q, masses=absmasses):
    phases = phase_per_particle(Rs, q)
    grad_over_psi = grad_psi_over_psi_per_particle(Rs)
    N_coord_local = Rs.shape[1]
    N_walkers = Rs.shape[0]
    result = np.zeros((N_walkers, 3), dtype=np.complex128)
    ij_pairs = [(0, 1), (0, 2), (1, 2)]
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (4.0 * m_k**2)
        grad_k = grad_over_psi[k]
        for idx, (i, j) in enumerate(ij_pairs):
            bracket = 1j * (q[j] * grad_k[i, :] - q[i] * grad_k[j, :])
            result = result.at[:, idx].set(result[:, idx] + phases[:, k] * factor * bracket)
    return result

def compute_J_number(Rs, q, masses=absmasses):
    phases = phase_per_particle(Rs, q)
    return np.sum(phases, axis=1)

def compute_all_currents(Rs, q, masses=absmasses):
    return {
        'N': compute_J_number(Rs, q, masses),
        'S': compute_J_S(Rs, q, masses),
        'V0': compute_J_V0(Rs, q, masses),
        'Vi': compute_J_Vi(Rs, q, masses),
        'Ai': compute_J_Ai(Rs, q, masses),
        'T0i': compute_J_T0i(Rs, q, masses),
        'Tij': compute_J_Tij(Rs, q, masses),
    }


###
# metropolis sampling
###

if input_Rs_database == "":
    met_time = time.time()
    R0 = onp.random.normal(size=(N_coord,3))/onp.mean(onp.abs(masses))
    R0 -= onp.transpose(onp.transpose(onp.mean(onp.transpose(onp.transpose(R0)*onp.abs(masses)), axis=0, keepdims=True))/onp.mean(onp.abs(masses)))
    samples = adl.metropolis(R0, f_R_braket, n_therm=500*n_skip, n_step=n_walkers, n_skip=n_skip, eps=4*2*a0/N_coord**2*radial_n, masses=onp.abs(masses))
    print(f"metropolis in {time.time() - met_time} sec")
    Rs_metropolis = np.array([R for R,_ in samples])
else:
    f = h5py.File(input_Rs_database, 'r')
    Rs_metropolis = f["Rs"][-1]

# spin-flavor structure
S_av4p_metropolis = onp.zeros(shape=(Rs_metropolis.shape[0],) + (NI,NS)*N_coord).astype(np.complex128)

def levi_civita(i, j, k):
    if i == j or j == k or i == k: return 0
    return 1 if (i,j,k) in [(0,1,2), (1,2,0), (2,0,1)] else -1

def kronecker_delta(i, j): return 1 if i == j else 0

if N_coord == 3:
  for i in range(NI):
   for j in range(NI):
    for k in range(NI):
     if i != j and j != k and i != k:
      spin_slice = (slice(0, None),) + (i,0,j,0,k,0)
      S_av4p_metropolis[spin_slice] = levi_civita(i, j, k) / np.sqrt(2*NI)

if N_coord == 2:
  for i in range(NI):
   for j in range(NI):
    if i == j:
      spin_slice = (slice(0, None),) + (i,0,j,0)
      S_av4p_metropolis[spin_slice] = kronecker_delta(i, j)/np.sqrt(NI)

S_av4p_metropolis_norm = adl.inner(S_av4p_metropolis, S_av4p_metropolis)
assert (np.abs(S_av4p_metropolis_norm - 1.0) < 1e-6).all()

deform_f = lambda x, params: x
params = (np.zeros((n_step+1)),)

###
# GFMC evolution (leg-1)
###

print('\nRunning GFMC evolution (leg-1):')
if n_step > 0:
    rand_draws = onp.random.random(size=(n_step, Rs_metropolis.shape[0]))
    gfmc = adl.gfmc_deform(
        Rs_metropolis, S_av4p_metropolis, f_R_norm, params,
        rand_draws=rand_draws, tau_iMev=tau_iMev, N=n_step, potential=Coulomb_potential,
        deform_f=deform_f, m_Mev=np.abs(np.array(masses)),
        resampling_freq=resampling)
    gfmc_Rs = np.array([Rs for Rs,_,_,_ in gfmc])
    gfmc_Ws = np.array([Ws for _,_,_,Ws in gfmc])
    gfmc_Ss = np.array([Ss for _,_,Ss,_ in gfmc])
else:
    gfmc_Rs = np.array([Rs_metropolis])
    gfmc_Ws = np.array([0*Rs_metropolis[:,1,1]+1])
    gfmc_Ss = np.array([S_av4p_metropolis])

print('GFMC tau=0 weights (leg-1):', gfmc_Ws[0][:5], '...')
if n_step > 0:
    print('GFMC tau=dtau weights (leg-1):', gfmc_Ws[1][:5], '...')

###
# Hamiltonian measurements (leg-1)
###

print('\nMeasuring <H> (leg-1 grid)...')
Ks = []
for count, R in enumerate(gfmc_Rs):
    K_term = -1/2*laplacian_f_R(R) / f_R(R)
    grad = grad_f_R(R)
    K_term += 1.0/2*np.sum(np.sum( mtm**2, axis=1)/absmasses, axis=0) - 1j*np.einsum('ain,ai->n', grad, mtm) / f_R(R)
    Ks.append(K_term)
Ks = np.array(Ks)

Vs = []
for count, R in enumerate(gfmc_Rs):
    S = gfmc_Ss[count]
    V_SI, V_SD = Coulomb_potential(R)
    V_SD_S = adl.batched_apply(V_SD, S)
    V_SI_S = adl.batched_apply(V_SI, S)
    V_tot = adl.inner(S_av4p_metropolis, V_SD_S + V_SI_S) / adl.inner(S_av4p_metropolis, S)
    Vs.append(V_tot)
Vs = np.array(Vs)

print("\n" + "="*70)
print("AVERAGE HAMILTONIANS AT EACH TIME SLICE (leg-1)")
print("="*70)
for t_idx in range(len(gfmc_Rs)):
    tau = t_idx * dtau_iMev
    H = Ks[t_idx] + Vs[t_idx]
    W = gfmc_Ws[t_idx]
    W_sum = onp.sum(onp.abs(W))
    H_avg = onp.sum(onp.real(H) * onp.abs(W)) / W_sum
    K_avg = onp.sum(onp.real(Ks[t_idx]) * onp.abs(W)) / W_sum
    V_avg = onp.sum(onp.real(Vs[t_idx]) * onp.abs(W)) / W_sum
    print(f"tau={tau:5.3f} MeV^-1: <H>={H_avg:+8.5f} MeV  <K>={K_avg:+8.5f} MeV  <V>={V_avg:+8.5f} MeV")
print("="*70 + "\n")

###
# save leg-1 data
###

_alphas_str = f"_alphas{alpha_s}" if alpha_s is not None else ""
tag = f"muPrime_{OLO}_dtau{dtau_iMev}_Nstep{n_step}_Nwalkers{n_walkers}_Ncoord{N_coord}_Nc{Nc}_Nf{nf}_mQ{mQ}_mu{mu}_mufac{mufac}_Rstar{Rstar}_spoila{spoila}{_alphas_str}_wavefunction_{wavefunction}_potential_{potential}_masses{masses}_color_{color}_Qmag{Q_mag}"

with h5py.File(outdir+'Hammys_'+tag+'.h5', 'w') as f:
    f.create_dataset("Hammys", data=Ks+Vs)
    f.create_dataset("Ws_leg1", data=gfmc_Ws)
    f.create_dataset("Ks", data=Ks)
    f.create_dataset("Vs", data=Vs)

with h5py.File(outdir+'Rs_'+tag+'.h5', 'w') as f:
    f.create_dataset("Rs", data=gfmc_Rs)
    f.create_dataset("Ws_leg1", data=gfmc_Ws)


###
# 3-point correlators with all currents
###

def _run_leg2_Ws_sequence(R, S, Ws_seed, rand_draws=None, N_leg2=None):
    if N_leg2 is None:
        N_leg2 = n_step
    if rand_draws is None:
        rand_draws = onp.random.random(size=(N_leg2, R.shape[0]))
    else:
        rand_draws = rand_draws[:N_leg2]
    leg2_params = (np.zeros((N_leg2+1)),)
    leg2_tau = dtau_iMev * N_leg2
    gf2 = adl.gfmc_deform(
        R, S, f_R_norm, leg2_params,
        rand_draws=rand_draws, tau_iMev=leg2_tau, N=N_leg2, potential=Coulomb_potential,
        deform_f=deform_f, m_Mev=np.abs(np.array(masses)), resampling_freq=resampling,
        Ws=Ws_seed
    )
    seq = [Ws for _,_,_,Ws in gf2]
    if len(seq) == N_leg2:
        seq = [Ws_seed] + seq
    return seq

t_indices = [t for t in range(n_step+1) if t % count_skip == 0]
N_tau = len(t_indices)

print(f"\nComputing 3-point correlators for {N_tau} time slices...")
print(f"Computing ALL current operators: N, S, V0, Vi(x,y,z), Ai(x,y,z), T0i(x,y,z), Tij(xy,xz,yz)")

C2_2tau = onp.zeros(N_tau)
Ws2_all = onp.zeros((N_tau, n_walkers), dtype=onp.complex128)

C3_N = onp.zeros((n_walkers, N_tau), dtype=onp.complex128)
C3_S = onp.zeros((n_walkers, N_tau), dtype=onp.complex128)
C3_V0 = onp.zeros((n_walkers, N_tau), dtype=onp.complex128)
C3_Vi = onp.zeros((n_walkers, N_tau, 3), dtype=onp.complex128)
C3_Ai = onp.zeros((n_walkers, N_tau, 3), dtype=onp.complex128)
C3_T0i = onp.zeros((n_walkers, N_tau, 3), dtype=onp.complex128)
C3_Tij = onp.zeros((n_walkers, N_tau, 3), dtype=onp.complex128)

print("\n" + "="*70)
print("CURRENT OPERATOR VALUES AT EACH TIME SLICE (before leg-2 evolution)")
print("="*70)

for i_t, t in enumerate(t_indices):
    R = gfmc_Rs[t]
    S = gfmc_Ss[t]
    W1 = gfmc_Ws[t]

    tau = t * dtau_iMev
    print(f"\n--- tau = {tau:.3f} MeV^-1 (slice {i_t+1}/{N_tau}) ---")

    N_leg2 = max(t, 1)  # leg-2 only needs t steps (not full n_step)
    rand2_shared = onp.random.random(size=(N_leg2, R.shape[0]))
    rng_seed = onp.random.randint(0, 2**31)

    onp.random.seed(rng_seed)
    ws_seq_C2 = _run_leg2_Ws_sequence(R, S, W1, rand2_shared, N_leg2=N_leg2)

    Ws2 = onp.array(ws_seq_C2[min(t, N_leg2)])
    Ws2_all[i_t] = Ws2
    C2_2tau[i_t] = onp.sum(Ws2.real)

    # sink-side boosted-bra reweighting needs the CONJUGATE CM phase e^{-iq.R_cm}:
    # J = Sum_k e^{iq.x_k} = e^{+iq.R_cm} * (relative part), so conj(r01) cancels the
    # CM phase and the junction operator is the rest-frame relative current.
    # (unconjugated r01 doubles the phase -> exact spurious decay e^{-(2q)^2 tau1/(2M)})
    r01 = np.conj(f_R_q(R, q_vec)) / f_R_0(R)
    r01_arr = onp.array(r01)
    print(f"  r01 (wfn ratio): min={r01_arr.real.min():.6f}, max={r01_arr.real.max():.6f}, should be 1.0 at q=0")

    currents = compute_all_currents(R, q_total, absmasses)

    for key, val in currents.items():
        if isinstance(val, np.ndarray) or isinstance(val, onp.ndarray):
            val_arr = onp.array(val)
            n_nan = onp.sum(onp.isnan(val_arr))
            if n_nan > 0:
                print(f"  WARNING: {n_nan} NaN values in current {key}!")

    J_N_arr = onp.array(currents['N'])
    print(f"  J_N per-walker: min={J_N_arr.real.min():.6f}, max={J_N_arr.real.max():.6f}, should be {N_coord} at q=0")

    W_sum = onp.sum(onp.abs(W1))
    J_N_avg = onp.sum(currents['N'] * onp.abs(W1)) / W_sum
    J_S_avg = onp.sum(currents['S'] * onp.abs(W1)) / W_sum
    J_V0_avg = onp.sum(currents['V0'] * onp.abs(W1)) / W_sum
    print(f"  <J_N>  = {J_N_avg.real:+.6f} + {J_N_avg.imag:+.6f}i  (phase sum, expect {N_coord} at q=0)")
    print(f"  <J_S>  = {J_S_avg.real:+.6f} + {J_S_avg.imag:+.6f}i")
    print(f"  <J_V0> = {J_V0_avg.real:+.6f} + {J_V0_avg.imag:+.6f}i")

    for i, label in enumerate(['x', 'y', 'z']):
        J_Vi_avg = onp.sum(currents['Vi'][:, i] * onp.abs(W1)) / W_sum
        print(f"  <J_V{label}> = {J_Vi_avg.real:+.6f} + {J_Vi_avg.imag:+.6f}i")

    for i, label in enumerate(['x', 'y', 'z']):
        J_Ai_avg = onp.sum(currents['Ai'][:, i] * onp.abs(W1)) / W_sum
        print(f"  <J_A{label}> = {J_Ai_avg.real:+.6f} + {J_Ai_avg.imag:+.6f}i")

    for i, label in enumerate(['x', 'y', 'z']):
        J_T0i_avg = onp.sum(currents['T0i'][:, i] * onp.abs(W1)) / W_sum
        print(f"  <J_T0{label}> = {J_T0i_avg.real:+.6f} + {J_T0i_avg.imag:+.6f}i")

    for i, label in enumerate(['xy', 'xz', 'yz']):
        J_Tij_avg = onp.sum(currents['Tij'][:, i] * onp.abs(W1)) / W_sum
        print(f"  <J_T{label}> = {J_Tij_avg.real:+.6f} + {J_Tij_avg.imag:+.6f}i")

    def _safe_leg2(Ws_seed, label=""):
        """Run leg-2 only if Ws_seed is non-trivial; return zeros otherwise."""
        if onp.max(onp.abs(Ws_seed)) < 1e-30:
            print(f"  Skipping leg-2 for {label}: Ws_seed is zero (spin-0 baryon)")
            return [onp.zeros_like(Ws_seed)] * (N_leg2 + 1)
        if onp.any(onp.isnan(Ws_seed)):
            print(f"  WARNING: NaN in Ws_seed for {label}! Setting to zero.")
            return [onp.zeros_like(Ws_seed)] * (N_leg2 + 1)
        onp.random.seed(rng_seed)
        return _run_leg2_Ws_sequence(R, S, Ws_seed, rand2_shared, N_leg2=N_leg2)

    # number operator J_N
    Ws_seed_N = W1 * (currents['N'] * r01)
    ws_seq_N = _safe_leg2(Ws_seed_N, "J_N")
    C3_N[:, i_t] = onp.array(ws_seq_N[t])

    C3_N_t = onp.array(ws_seq_N[t])
    C2_t = onp.array(ws_seq_C2[t])
    valid = onp.abs(C2_t.real) > 1e-15
    if onp.sum(valid) > 0:
        ratio_per_walker = C3_N_t[valid].real / C2_t[valid].real
        print(f"  C3_N/C2 per-walker: min={ratio_per_walker.min():.10f}, max={ratio_per_walker.max():.10f}")
        direct_F_N = onp.sum(C3_N_t.real) / onp.sum(C2_t.real)
        print(f"  Direct F_N (no bootstrap) = {direct_F_N:.10f}, expected = {N_coord}")
    else:
        print(f"  All weights invalid!")

    # scalar current J_S
    Ws_seed_S = W1 * (currents['S'] * r01)
    ws_seq_S = _safe_leg2(Ws_seed_S, "J_S")
    C3_S[:, i_t] = onp.array(ws_seq_S[t])

    # vector temporal J_V0
    Ws_seed_V0 = W1 * (currents['V0'] * r01)
    ws_seq_V0 = _safe_leg2(Ws_seed_V0, "J_V0")
    C3_V0[:, i_t] = onp.array(ws_seq_V0[t])

    # vector spatial J_Vi (3 components)
    for comp in range(3):
        Ws_seed_Vi = W1 * (currents['Vi'][:, comp] * r01)
        ws_seq_Vi = _safe_leg2(Ws_seed_Vi, f"J_Vi[{comp}]")
        C3_Vi[:, i_t, comp] = onp.array(ws_seq_Vi[t])

    # axial spatial J_Ai (3 components)
    for comp in range(3):
        Ws_seed_Ai = W1 * (currents['Ai'][:, comp] * r01)
        ws_seq_Ai = _safe_leg2(Ws_seed_Ai, f"J_Ai[{comp}]")
        C3_Ai[:, i_t, comp] = onp.array(ws_seq_Ai[t])

    # tensor T0i (3 components)
    for comp in range(3):
        Ws_seed_T0i = W1 * (currents['T0i'][:, comp] * r01)
        ws_seq_T0i = _safe_leg2(Ws_seed_T0i, f"J_T0i[{comp}]")
        C3_T0i[:, i_t, comp] = onp.array(ws_seq_T0i[t])

    # tensor Tij (3 components: xy, xz, yz)
    for comp in range(3):
        Ws_seed_Tij = W1 * (currents['Tij'][:, comp] * r01)
        ws_seq_Tij = _safe_leg2(Ws_seed_Tij, f"J_Tij[{comp}]")
        C3_Tij[:, i_t, comp] = onp.array(ws_seq_Tij[t])

print("\n" + "="*70)


###
# form factor computation with bootstrap errors
###

print("\nComputing form factors with bootstrap errors...")

def compute_form_factor_bootstrap(C3_data, C2_data, n_boot=200):
    n_walkers_local = len(C3_data)
    valid_mask = (~onp.isnan(C3_data.real) & ~onp.isnan(C2_data.real))
    n_valid = onp.sum(valid_mask)
    if n_valid < 10:
        return onp.nan, onp.nan
    C3_valid = C3_data[valid_mask]
    C2_valid = C2_data[valid_mask]
    boot_vals = []
    for _ in range(n_boot):
        idx = onp.random.randint(0, n_valid, size=n_valid)
        C3_boot = onp.sum(C3_valid[idx].real)
        C2_boot = onp.sum(C2_valid[idx].real)
        if onp.abs(C2_boot) > 1e-15:
            R_boot = C3_boot / C2_boot
        else:
            R_boot = onp.nan
        boot_vals.append(R_boot)
    boot_vals = onp.array(boot_vals)
    boot_vals = boot_vals[~onp.isnan(boot_vals)]
    if len(boot_vals) < 10:
        return onp.nan, onp.nan
    return onp.mean(boot_vals), onp.std(boot_vals)

tau_values = onp.array(t_indices) * dtau_iMev

F_N = onp.zeros(N_tau)
F_N_err = onp.zeros(N_tau)
F_S = onp.zeros(N_tau)
F_S_err = onp.zeros(N_tau)
F_V0 = onp.zeros(N_tau)
F_V0_err = onp.zeros(N_tau)
F_Vi = onp.zeros((N_tau, 3))
F_Vi_err = onp.zeros((N_tau, 3))
F_Ai = onp.zeros((N_tau, 3))
F_Ai_err = onp.zeros((N_tau, 3))
F_T0i = onp.zeros((N_tau, 3))
F_T0i_err = onp.zeros((N_tau, 3))
F_Tij = onp.zeros((N_tau, 3))
F_Tij_err = onp.zeros((N_tau, 3))

for i_t in range(N_tau):
    C2_walkers = Ws2_all[i_t]
    F_N[i_t], F_N_err[i_t] = compute_form_factor_bootstrap(C3_N[:, i_t], C2_walkers)
    F_S[i_t], F_S_err[i_t] = compute_form_factor_bootstrap(C3_S[:, i_t], C2_walkers)
    F_V0[i_t], F_V0_err[i_t] = compute_form_factor_bootstrap(C3_V0[:, i_t], C2_walkers)
    for comp in range(3):
        F_Vi[i_t, comp], F_Vi_err[i_t, comp] = compute_form_factor_bootstrap(C3_Vi[:, i_t, comp], C2_walkers)
        F_Ai[i_t, comp], F_Ai_err[i_t, comp] = compute_form_factor_bootstrap(C3_Ai[:, i_t, comp], C2_walkers)
        F_T0i[i_t, comp], F_T0i_err[i_t, comp] = compute_form_factor_bootstrap(C3_T0i[:, i_t, comp], C2_walkers)
        F_Tij[i_t, comp], F_Tij_err[i_t, comp] = compute_form_factor_bootstrap(C3_Tij[:, i_t, comp], C2_walkers)


###
# print form factor results
###

print("\n" + "="*70)
print("FORM FACTORS VS IMAGINARY TIME")
print(f"|q|^2 = {float(np.sum(q_total**2)):.6f}")
print("="*70)

print("\n--- Number operator (phase sum) - expect N_coord at q=0 ---")
print(f"{'tau':>8} {'F_N':>12} {'err':>8}")
for i_t, tau in enumerate(tau_values):
    print(f"{tau:8.3f} {F_N[i_t]:12.6f} {F_N_err[i_t]:8.4f}")

print("\n--- Scalar and Vector-temporal (expect ~N_coord at q=0) ---")
print(f"{'tau':>8} {'F_S':>12} {'err':>8} {'F_V0':>12} {'err':>8}")
for i_t, tau in enumerate(tau_values):
    print(f"{tau:8.3f} {F_S[i_t]:12.6f} {F_S_err[i_t]:8.4f} {F_V0[i_t]:12.6f} {F_V0_err[i_t]:8.4f}")

print("="*70)


###
# save results
###

filename = outdir + 'currents_3pt_' + tag + '.h5'
print(f"\nSaving results to {filename}...")

with h5py.File(filename, 'w') as f:
    f.create_dataset("tau_values", data=tau_values)
    f.create_dataset("t_indices", data=onp.array(t_indices, dtype=onp.int32))
    f.create_dataset("q_total", data=onp.array(q_total))
    f.create_dataset("q_squared", data=float(np.sum(q_total**2)))
    f.create_dataset("masses", data=onp.array(masses))
    f.create_dataset("C2_2tau", data=C2_2tau)
    f.create_dataset("Ws2_all", data=Ws2_all)
    f.create_dataset("C3_N", data=C3_N)
    f.create_dataset("C3_S", data=C3_S)
    f.create_dataset("C3_V0", data=C3_V0)
    f.create_dataset("C3_Vi", data=C3_Vi)
    f.create_dataset("C3_Ai", data=C3_Ai)
    f.create_dataset("C3_T0i", data=C3_T0i)
    f.create_dataset("C3_Tij", data=C3_Tij)
    f.create_dataset("F_N", data=F_N)
    f.create_dataset("F_N_err", data=F_N_err)
    f.create_dataset("F_S", data=F_S)
    f.create_dataset("F_S_err", data=F_S_err)
    f.create_dataset("F_V0", data=F_V0)
    f.create_dataset("F_V0_err", data=F_V0_err)
    f.create_dataset("F_Vi", data=F_Vi)
    f.create_dataset("F_Vi_err", data=F_Vi_err)
    f.create_dataset("F_Ai", data=F_Ai)
    f.create_dataset("F_Ai_err", data=F_Ai_err)
    f.create_dataset("F_T0i", data=F_T0i)
    f.create_dataset("F_T0i_err", data=F_T0i_err)
    f.create_dataset("F_Tij", data=F_Tij)
    f.create_dataset("F_Tij_err", data=F_Tij_err)


###
# plots
###

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
ax.errorbar(tau_values, F_N, yerr=F_N_err, fmt='^-', label=r'$F_N$ (phase sum)', capsize=3)
ax.errorbar(tau_values, F_S, yerr=F_S_err, fmt='o-', label=r'$F_S$', capsize=3)
ax.errorbar(tau_values, F_V0, yerr=F_V0_err, fmt='s-', label=r'$F_{V^0}$', capsize=3)
ax.axhline(y=N_coord, color='gray', linestyle='--', alpha=0.5, label=f'N_coord = {N_coord}')
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title('Number, Scalar, Vector-temporal')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
labels = ['x', 'y', 'z']
colors = ['C0', 'C1', 'C2']
for i in range(3):
    ax.errorbar(tau_values, F_Vi[:, i], yerr=F_Vi_err[:, i], fmt='o-',
                label=rf'$F_{{V^{labels[i]}}}$', color=colors[i], capsize=3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Vector-spatial $F_{V^i}$')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
for i in range(3):
    ax.errorbar(tau_values, F_Ai[:, i], yerr=F_Ai_err[:, i], fmt='o-',
                label=rf'$F_{{A^{labels[i]}}}$', color=colors[i], capsize=3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Axial-spatial $F_{A^i}$')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for i in range(3):
    ax.errorbar(tau_values, F_T0i[:, i], yerr=F_T0i_err[:, i], fmt='o-',
                label=rf'$F_{{T^{{0{labels[i]}}}}}$', color=colors[i], capsize=3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Tensor $F_{T^{0i}}$')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ij_labels = ['xy', 'xz', 'yz']
for i in range(3):
    ax.errorbar(tau_values, F_Tij[:, i], yerr=F_Tij_err[:, i], fmt='o-',
                label=rf'$F_{{T^{{{ij_labels[i]}}}}}$', color=colors[i], capsize=3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Tensor $F_{T^{ij}}$')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.errorbar(tau_values, F_S/N_coord, yerr=F_S_err/N_coord, fmt='o-',
            label=r'$F_S/N$', capsize=3)
ax.errorbar(tau_values, F_V0/N_coord, yerr=F_V0_err/N_coord, fmt='s-',
            label=r'$F_{V^0}/N$', capsize=3)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Expected at q=0')
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Normalized Form Factor')
ax.set_title(f'Charge-normalized ($|q|^2$ = {float(np.sum(q_total**2)):.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = outdir + 'form_factors_all_' + tag + '.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"Saved plot to {plot_filename}")
plt.close('all')

print("\nDone.")


###
# plateau fitting with bootstrap
###

def fit_plateau_bootstrap(F_data, tau_values, tau_min=None, tau_max=None, n_boot=500):
    if tau_min is None:
        tau_min = 0.4 * tau_values[-1]
    if tau_max is None:
        tau_max = tau_values[-1]
    mask = (tau_values >= tau_min) & (tau_values <= tau_max)
    F_plateau = F_data[mask]
    if len(F_plateau) < 2:
        return F_data[-1], 0.0, tau_min, tau_max
    boot_vals = []
    for _ in range(n_boot):
        idx = onp.random.choice(len(F_plateau), size=len(F_plateau), replace=True)
        boot_vals.append(onp.mean(F_plateau[idx]))
    return onp.mean(boot_vals), onp.std(boot_vals), tau_min, tau_max


if tau_min_plateau is None:
    tau_min_plateau = 0.4 * tau_values[-1]
if tau_max_plateau is None:
    tau_max_plateau = tau_values[-1]

print("\n" + "="*70)
print("PLATEAU FITS FOR FINAL FORM FACTORS")
print(f"Plateau region: tau in [{tau_min_plateau:.2f}, {tau_max_plateau:.2f}] MeV^-1")
print("="*70)

results = {}

F_N_final, F_N_final_err, _, _ = fit_plateau_bootstrap(F_N, tau_values, tau_min_plateau, tau_max_plateau)
results['N'] = (F_N_final, F_N_final_err)

F_S_final, F_S_final_err, _, _ = fit_plateau_bootstrap(F_S, tau_values, tau_min_plateau, tau_max_plateau)
F_V0_final, F_V0_final_err, _, _ = fit_plateau_bootstrap(F_V0, tau_values, tau_min_plateau, tau_max_plateau)
results['S'] = (F_S_final, F_S_final_err)
results['V0'] = (F_V0_final, F_V0_final_err)

F_Vi_final = onp.zeros(3)
F_Vi_final_err = onp.zeros(3)
for i in range(3):
    F_Vi_final[i], F_Vi_final_err[i], _, _ = fit_plateau_bootstrap(F_Vi[:, i], tau_values, tau_min_plateau, tau_max_plateau)
results['Vi'] = (F_Vi_final, F_Vi_final_err)

F_Ai_final = onp.zeros(3)
F_Ai_final_err = onp.zeros(3)
for i in range(3):
    F_Ai_final[i], F_Ai_final_err[i], _, _ = fit_plateau_bootstrap(F_Ai[:, i], tau_values, tau_min_plateau, tau_max_plateau)
results['Ai'] = (F_Ai_final, F_Ai_final_err)

F_T0i_final = onp.zeros(3)
F_T0i_final_err = onp.zeros(3)
for i in range(3):
    F_T0i_final[i], F_T0i_final_err[i], _, _ = fit_plateau_bootstrap(F_T0i[:, i], tau_values, tau_min_plateau, tau_max_plateau)
results['T0i'] = (F_T0i_final, F_T0i_final_err)

F_Tij_final = onp.zeros(3)
F_Tij_final_err = onp.zeros(3)
for i in range(3):
    F_Tij_final[i], F_Tij_final_err[i], _, _ = fit_plateau_bootstrap(F_Tij[:, i], tau_values, tau_min_plateau, tau_max_plateau)
results['Tij'] = (F_Tij_final, F_Tij_final_err)

comp_labels = ['x', 'y', 'z']
print("\n" + "-"*50)
print("FINAL FORM FACTORS (plateau averages)")
print("-"*50)
print(f"{'Current':<12} {'Component':<10} {'F(q^2)':<14} {'Error':<10}")
print("-"*50)
print(f"{'Number':<12} {'':<10} {F_N_final:>12.6f}   {F_N_final_err:>8.6f}  (phase sum)")
print(f"{'Scalar':<12} {'':<10} {F_S_final:>12.6f}   {F_S_final_err:>8.6f}")
print(f"{'Vector-0':<12} {'':<10} {F_V0_final:>12.6f}   {F_V0_final_err:>8.6f}")
for i in range(3):
    print(f"{'Vector-i':<12} {comp_labels[i]:<10} {F_Vi_final[i]:>12.6f}   {F_Vi_final_err[i]:>8.6f}")
for i in range(3):
    print(f"{'Axial-i':<12} {comp_labels[i]:<10} {F_Ai_final[i]:>12.6f}   {F_Ai_final_err[i]:>8.6f}")
for i in range(3):
    print(f"{'Tensor-0i':<12} {comp_labels[i]:<10} {F_T0i_final[i]:>12.6f}   {F_T0i_final_err[i]:>8.6f}")
ij_labels = ['xy', 'xz', 'yz']
for i in range(3):
    print(f"{'Tensor-ij':<12} {ij_labels[i]:<10} {F_Tij_final[i]:>12.6f}   {F_Tij_final_err[i]:>8.6f}")
print("-"*50)

print(f"\nF_N / N_coord  = {F_N_final/N_coord:.6f} +/- {F_N_final_err/N_coord:.6f}  (phase sum, expect 1.0)")
print(f"F_S / N_coord  = {F_S_final/N_coord:.6f} +/- {F_S_final_err/N_coord:.6f}")
print(f"F_V0 / N_coord = {F_V0_final/N_coord:.6f} +/- {F_V0_final_err/N_coord:.6f}")


###
# plots with plateau bands
###

print("\nGenerating plots with plateau fits...")

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

def add_plateau_band(ax, val, err, tau_min, tau_max, color='gray', alpha=0.3):
    ax.axhspan(val - err, val + err, xmin=0, xmax=1, alpha=alpha, color=color, zorder=0)
    ax.axhline(y=val, color=color, linestyle='-', alpha=0.7, linewidth=1.5)

ax = axes2[0, 0]
ax.errorbar(tau_values, F_N, yerr=F_N_err, fmt='^-', label=r'$F_N$', capsize=3, zorder=2)
ax.errorbar(tau_values, F_S, yerr=F_S_err, fmt='o-', label=r'$F_S$', capsize=3, zorder=2)
ax.errorbar(tau_values, F_V0, yerr=F_V0_err, fmt='s-', label=r'$F_{V^0}$', capsize=3, zorder=2)
ax.axhline(y=N_coord, color='black', linestyle='--', alpha=0.5, label=f'N = {N_coord}')
ax.axvline(x=tau_min_plateau, color='red', linestyle=':', alpha=0.5, label='Plateau region')
add_plateau_band(ax, F_N_final, F_N_final_err, tau_min_plateau, tau_max_plateau, color='C0')
add_plateau_band(ax, F_S_final, F_S_final_err, tau_min_plateau, tau_max_plateau, color='C1')
add_plateau_band(ax, F_V0_final, F_V0_final_err, tau_min_plateau, tau_max_plateau, color='C2')
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(f'N: {F_N_final:.4f}, S: {F_S_final:.4f}, V0: {F_V0_final:.4f}')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes2[0, 1]
for i in range(3):
    ax.errorbar(tau_values, F_Vi[:, i], yerr=F_Vi_err[:, i], fmt='o-',
                label=rf'$F_{{V^{comp_labels[i]}}}$={F_Vi_final[i]:.4f}', color=colors[i], capsize=3, zorder=2)
    add_plateau_band(ax, F_Vi_final[i], F_Vi_final_err[i], tau_min_plateau, tau_max_plateau, color=colors[i])
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=tau_min_plateau, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Vector-spatial $F_{V^i}$')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes2[0, 2]
for i in range(3):
    ax.errorbar(tau_values, F_Ai[:, i], yerr=F_Ai_err[:, i], fmt='o-',
                label=rf'$F_{{A^{comp_labels[i]}}}$={F_Ai_final[i]:.4f}', color=colors[i], capsize=3, zorder=2)
    add_plateau_band(ax, F_Ai_final[i], F_Ai_final_err[i], tau_min_plateau, tau_max_plateau, color=colors[i])
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=tau_min_plateau, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Axial-spatial $F_{A^i}$')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes2[1, 0]
for i in range(3):
    ax.errorbar(tau_values, F_T0i[:, i], yerr=F_T0i_err[:, i], fmt='o-',
                label=rf'$F_{{T^{{0{comp_labels[i]}}}}}$={F_T0i_final[i]:.4f}', color=colors[i], capsize=3, zorder=2)
    add_plateau_band(ax, F_T0i_final[i], F_T0i_final_err[i], tau_min_plateau, tau_max_plateau, color=colors[i])
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=tau_min_plateau, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Tensor $F_{T^{0i}}$')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes2[1, 1]
for i in range(3):
    ax.errorbar(tau_values, F_Tij[:, i], yerr=F_Tij_err[:, i], fmt='o-',
                label=rf'$F_{{T^{{{ij_labels[i]}}}}}$={F_Tij_final[i]:.4f}', color=colors[i], capsize=3, zorder=2)
    add_plateau_band(ax, F_Tij_final[i], F_Tij_final_err[i], tau_min_plateau, tau_max_plateau, color=colors[i])
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=tau_min_plateau, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Form Factor')
ax.set_title(r'Tensor $F_{T^{ij}}$')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes2[1, 2]
F_N_norm = F_N / N_coord
F_S_norm = F_S / N_coord
F_V0_norm = F_V0 / N_coord
F_N_norm_err = F_N_err / N_coord
F_S_norm_err = F_S_err / N_coord
F_V0_norm_err = F_V0_err / N_coord
ax.errorbar(tau_values, F_N_norm, yerr=F_N_norm_err, fmt='^-',
            label=rf'$F_N/N$={F_N_final/N_coord:.4f}', capsize=3, zorder=2)
ax.errorbar(tau_values, F_S_norm, yerr=F_S_norm_err, fmt='o-',
            label=rf'$F_S/N$={F_S_final/N_coord:.4f}', capsize=3, zorder=2)
ax.errorbar(tau_values, F_V0_norm, yerr=F_V0_norm_err, fmt='s-',
            label=rf'$F_{{V^0}}/N$={F_V0_final/N_coord:.4f}', capsize=3, zorder=2)
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Expected at q=0')
ax.axvline(x=tau_min_plateau, color='red', linestyle=':', alpha=0.5)
add_plateau_band(ax, F_N_final/N_coord, F_N_final_err/N_coord, tau_min_plateau, tau_max_plateau, color='C0')
add_plateau_band(ax, F_S_final/N_coord, F_S_final_err/N_coord, tau_min_plateau, tau_max_plateau, color='C1')
add_plateau_band(ax, F_V0_final/N_coord, F_V0_final_err/N_coord, tau_min_plateau, tau_max_plateau, color='C2')
ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
ax.set_ylabel('Normalized Form Factor')
ax.set_title(f'Charge-normalized ($|q|^2$ = {float(np.sum(q_total**2)):.4f})')
ax.legend(loc='best', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename2 = outdir + 'form_factors_plateau_' + tag + '.png'
plt.savefig(plot_filename2, dpi=150, bbox_inches='tight')
print(f"Saved plateau plot to {plot_filename2}")
plt.close('all')


###
# save plateau fit results
###

with h5py.File(filename, 'a') as f:
    f.create_dataset("tau_min_plateau", data=tau_min_plateau)
    f.create_dataset("tau_max_plateau", data=tau_max_plateau)
    f.create_dataset("F_N_final", data=F_N_final)
    f.create_dataset("F_N_final_err", data=F_N_final_err)
    f.create_dataset("F_S_final", data=F_S_final)
    f.create_dataset("F_S_final_err", data=F_S_final_err)
    f.create_dataset("F_V0_final", data=F_V0_final)
    f.create_dataset("F_V0_final_err", data=F_V0_final_err)
    f.create_dataset("F_Vi_final", data=F_Vi_final)
    f.create_dataset("F_Vi_final_err", data=F_Vi_final_err)
    f.create_dataset("F_Ai_final", data=F_Ai_final)
    f.create_dataset("F_Ai_final_err", data=F_Ai_final_err)
    f.create_dataset("F_T0i_final", data=F_T0i_final)
    f.create_dataset("F_T0i_final_err", data=F_T0i_final_err)
    f.create_dataset("F_Tij_final", data=F_Tij_final)
    f.create_dataset("F_Tij_final_err", data=F_Tij_final_err)

print(f"\nAppended plateau fit results to {filename}")
print("\nDone with plateau fitting.")
