import argparse
import analysis as al
import matplotlib.pyplot as plt
import numpy as onp
import scipy
import scipy.interpolate
import scipy.integrate
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
parser.add_argument('--Nc', type=int, default=3)
parser.add_argument('--N_coord', type=int, default=3)
parser.add_argument('--nf', type=int, default=5)
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
globals().update(vars(parser.parse_args()))

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

CF = (Nc**2 - 1)/(2*Nc)
VB = alpha*CF/(Nc-1)
SingC3 = -(Nc+1)/8

# imaginary time points per leg
tau_iMev = dtau_iMev * n_step
xs = np.linspace(0, tau_iMev, endpoint=True, num=n_step+1)

beta0 = 11/3*Nc - 2/3*nf
beta1 = 34/3*Nc**2 - 20/3*Nc*nf/2 - 2*CF*nf
aa1 = 31/9*Nc-10/9*nf
zeta3 = scipy.special.zeta(3)
zeta5 = scipy.special.zeta(5)
aa2 = ( 4343/162 + 6*np.pi**2 - np.pi**4/4 + 22/3*zeta3 )*Nc**2 - ( 1798/81 + 56/3*zeta3 )*Nc*nf/2 - ( 55/3 - 16*zeta3  )*CF*nf/2 + (10/9*nf)**2

VB_LO = VB
VB_NLO = VB * (1 + alpha/(4*np.pi)*(aa1 + 2*beta0*log_mu_r))

if OLO == "LO":
    a0=spoila*2/VB_LO
elif OLO == "NLO":
    a0=spoila*2/VB_NLO
else:
    raise ValueError("Only LO and NLO supported")

if N_coord == 2:
    a0 /= Nc-1

print(f"a0 = {a0}")

os.makedirs(outdir, exist_ok=True)

Rprime = lambda R: adl.norm_3vec(R)*np.exp(np.euler_gamma)*mu
AV_Coulomb = {}
B3_Coulomb = {}

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

# potentials
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
        return -1*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    @partial(jax.jit)
    def singlet_potential_fun(R):
        return -1*(Nc - 1)*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    def symmetric_potential_fun(R):
        return (Nc - 1)/(Nc + 1)*VB*spoilS/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))
    def octet_potential_fun(R):
        return spoilS*(Nc - 1)/CF/(2*Nc)*VB/adl.norm_3vec(R)*(1 + alpha/(4*np.pi)*(2*beta0*np.log(Rprime(R))+aa1))

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
    """
    exp(i q . x_k) for each particle k.
    q is the TOTAL momentum transfer (shape (3,)), not per-particle.
    Returns shape (n_walkers, N_coord).
    """
    dots = np.einsum('i,nki->nk', q, Rs)
    return np.exp(1j * dots)

@partial(jax.jit)
def grad_psi_over_psi_per_particle(Rs, a0=a0, masses=absmasses):
    """
    (grad_k psi)/psi for each particle k, no mass factors.
    For the compact wavefunction psi = exp(-sum_{i<j} m_ij r_ij / a0),
    this is just -sum_{j!=k} (1/a0_kj) * r_hat_kj.
    Returns shape (N_coord, 3, n_walkers).
    """
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
                
                # cap distance to avoid blowup when particles are on top of each other
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
    """
    (laplacian_k psi)/psi for each particle k, no mass factors.
    Uses: laplacian_k psi / psi = |grad_k ln psi|^2 + laplacian_k ln psi.
    Returns shape (N_coord, n_walkers).
    """
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
# all return J|psi>/|psi> with per-particle mass factors 1/(4m_k^2) or 1/(2m_k)
###

def compute_J_S(Rs, q, masses=absmasses):
    """
    Scalar current: J_S = sum_k exp(iq.x_k) [1 + 1/(4m_k^2)(lap_k + iq.grad_k)] psi/psi
    """
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
    """
    Vector temporal current: J_V0 = sum_k exp(iq.x_k) [1 - 1/(4m_k^2)(lap_k + iq.grad_k)] psi/psi
    Same as scalar but with minus sign on the correction.
    """
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
    """
    Vector spatial current: J_V^i = sum_k exp(iq.x_k) * 1/(2m_k) * (2i grad_k^i - q^i) psi/psi
    Returns shape (n_walkers, 3), one component per spatial direction.
    """
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
            result = result.at[:, i].set(
                result[:, i] + phases[:, k] * factor * bracket
            )
    
    return result

def compute_J_Ai(Rs, q, masses=absmasses):
    """
    Axial spatial current: J_A^i = sum_k exp(iq.x_k) * 1/(4m_k^2) * (q x grad_k)^i psi/psi
    Returns shape (n_walkers, 3).
    """
    phases = phase_per_particle(Rs, q)
    grad_over_psi = grad_psi_over_psi_per_particle(Rs)
    
    N_coord_local = Rs.shape[1]
    N_walkers = Rs.shape[0]
    result = np.zeros((N_walkers, 3), dtype=np.complex128)
    
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (4.0 * m_k**2)
        
        grad_k = grad_over_psi[k]  # shape (3, n_walkers)
        
        cross_x = q[1] * grad_k[2, :] - q[2] * grad_k[1, :]
        cross_y = q[2] * grad_k[0, :] - q[0] * grad_k[2, :]
        cross_z = q[0] * grad_k[1, :] - q[1] * grad_k[0, :]
        
        result = result.at[:, 0].set(result[:, 0] + phases[:, k] * factor * cross_x)
        result = result.at[:, 1].set(result[:, 1] + phases[:, k] * factor * cross_y)
        result = result.at[:, 2].set(result[:, 2] + phases[:, k] * factor * cross_z)
    
    return result

def compute_J_T0i(Rs, q, masses=absmasses):
    """
    Tensor 0i current: J_T^{0i} = sum_k exp(iq.x_k) * 1/(2m_k) * q^i
    Returns shape (n_walkers, 3).
    """
    phases = phase_per_particle(Rs, q)
    
    N_coord_local = Rs.shape[1]
    N_walkers = Rs.shape[0]
    result = np.zeros((N_walkers, 3), dtype=np.complex128)
    
    for k in range(N_coord_local):
        m_k = masses[k]
        factor = 1.0 / (2.0 * m_k)
        
        for i in range(3):
            result = result.at[:, i].set(
                result[:, i] + phases[:, k] * factor * q[i]
            )
    
    return result

def compute_J_Tij(Rs, q, masses=absmasses):
    """
    Tensor ij current: J_T^{ij} = sum_k exp(iq.x_k) * 1/(4m_k^2) * i*(q^j grad_k^i - q^i grad_k^j) psi/psi
    Antisymmetric in i,j. Returns 3 independent components:
        index 0: (i,j) = (0,1) = xy
        index 1: (i,j) = (0,2) = xz
        index 2: (i,j) = (1,2) = yz
    Returns shape (n_walkers, 3).
    """
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
            result = result.at[:, idx].set(
                result[:, idx] + phases[:, k] * factor * bracket
            )
    
    return result


def compute_J_number(Rs, q, masses=absmasses):
    """
    Number operator: J_N = sum_k exp(iq.x_k).
    At q=0 this gives exactly N_coord.
    """
    phases = phase_per_particle(Rs, q)
    return np.sum(phases, axis=1)


def compute_all_currents(Rs, q, masses=absmasses):
    """Compute all current operators. Returns dict."""
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

tag = f"{OLO}_dtau{dtau_iMev}_Nstep{n_step}_Nwalkers{n_walkers}_Ncoord{N_coord}_Nc{Nc}_Nf{nf}_alpha{alpha}_spoila{spoila}_wavefunction_{wavefunction}_potential_{potential}_masses{masses}_color_{color}_Qmag{Q_mag}"

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

def _run_leg2_Ws_sequence(R, S, Ws_seed, rand_draws=None):
    """Run leg-2 GFMC evolution and return weight sequence.
    Uses f_R_norm for importance sampling (same for zero and finite momentum
    since the plane-wave phase doesn't affect the norm).
    """
    if rand_draws is None:
        rand_draws = onp.random.random(size=(n_step, R.shape[0]))
    gf2 = adl.gfmc_deform(
        R, S, f_R_norm, params,
        rand_draws=rand_draws, tau_iMev=tau_iMev, N=n_step, potential=Coulomb_potential,
        deform_f=deform_f, m_Mev=np.abs(np.array(masses)), resampling_freq=resampling,
        Ws=Ws_seed
    )
    seq = [Ws for _,_,_,Ws in gf2]
    if len(seq) == n_step:
        seq = [Ws_seed] + seq
    return seq

t_indices = [t for t in range(n_step+1) if t % count_skip == 0]
N_tau = len(t_indices)

print(f"\nComputing 3-point correlators for {N_tau} time slices...")
print(f"Computing ALL current operators: N, S, V0, Vi(x,y,z), Ai(x,y,z), T0i(x,y,z), Tij(xy,xz,yz)")

# storage for 2-point correlators (only need one set since norm is the same)
C2_2tau = onp.zeros(N_tau)
Ws2_all = onp.zeros((N_tau, n_walkers), dtype=onp.complex128)

# storage for 3-point correlators
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

    # one set of random draws for this tau slice, shared by all leg-2 evolutions
    rand2_shared = onp.random.random(size=(n_step, R.shape[0]))
    
    # one seed for this tau slice so all leg-2 runs get identical diffusion paths
    rng_seed = onp.random.randint(0, 2**31)

    # C2 correlator (only one needed -- norm is the same for zero and finite momentum)
    onp.random.seed(rng_seed)
    ws_seq_C2 = _run_leg2_Ws_sequence(R, S, W1, rand2_shared)
    
    Ws2 = onp.array(ws_seq_C2[t])
    Ws2_all[i_t] = Ws2
    C2_2tau[i_t] = onp.sum(Ws2.real)
    
    # wavefunction ratio for leg-2: psi_q / psi_0
    r01 = f_R_q(R, q_vec) / f_R_0(R)
    
    r01_arr = onp.array(r01)
    print(f"  r01 (wfn ratio): min={r01_arr.real.min():.6f}, max={r01_arr.real.max():.6f}, should be 1.0 at q=0")
    
    # compute all currents at this time slice
    currents = compute_all_currents(R, q_total, absmasses)
    
    # check for NaN in currents
    for key, val in currents.items():
        if isinstance(val, np.ndarray) or isinstance(val, onp.ndarray):
            val_arr = onp.array(val)
            n_nan = onp.sum(onp.isnan(val_arr))
            if n_nan > 0:
                print(f"  WARNING: {n_nan} NaN values in current {key}!")
    
    J_N_arr = onp.array(currents['N'])
    print(f"  J_N per-walker: min={J_N_arr.real.min():.6f}, max={J_N_arr.real.max():.6f}, should be {N_coord} at q=0")
    
    # print current values (weighted averages)
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

    # evolve with each current insertion

    # number operator J_N
    Ws_seed_N = W1 * (currents['N'] * r01)
    onp.random.seed(rng_seed)
    ws_seq_N = _run_leg2_Ws_sequence(R, S, Ws_seed_N, rand2_shared)
    C3_N[:, i_t] = onp.array(ws_seq_N[t])
    
    # per-walker check (should be exactly N_coord at q=0)
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
    if onp.any(onp.isnan(Ws_seed_S)):
        print(f"  WARNING: NaN in Ws_seed_S! Count: {onp.sum(onp.isnan(Ws_seed_S))}")
    onp.random.seed(rng_seed)
    ws_seq_S = _run_leg2_Ws_sequence(R, S, Ws_seed_S, rand2_shared)
    C3_S[:, i_t] = onp.array(ws_seq_S[t])
    
    # vector temporal J_V0
    Ws_seed_V0 = W1 * (currents['V0'] * r01)
    if onp.any(onp.isnan(Ws_seed_V0)):
        print(f"  WARNING: NaN in Ws_seed_V0! Count: {onp.sum(onp.isnan(Ws_seed_V0))}")
    onp.random.seed(rng_seed)
    ws_seq_V0 = _run_leg2_Ws_sequence(R, S, Ws_seed_V0, rand2_shared)
    C3_V0[:, i_t] = onp.array(ws_seq_V0[t])
    
    # vector spatial J_Vi (3 components)
    for comp in range(3):
        Ws_seed_Vi = W1 * (currents['Vi'][:, comp] * r01)
        onp.random.seed(rng_seed)
        ws_seq_Vi = _run_leg2_Ws_sequence(R, S, Ws_seed_Vi, rand2_shared)
        C3_Vi[:, i_t, comp] = onp.array(ws_seq_Vi[t])
    
    # axial spatial J_Ai (3 components)
    for comp in range(3):
        Ws_seed_Ai = W1 * (currents['Ai'][:, comp] * r01)
        onp.random.seed(rng_seed)
        ws_seq_Ai = _run_leg2_Ws_sequence(R, S, Ws_seed_Ai, rand2_shared)
        C3_Ai[:, i_t, comp] = onp.array(ws_seq_Ai[t])
    
    # tensor T0i (3 components)
    for comp in range(3):
        Ws_seed_T0i = W1 * (currents['T0i'][:, comp] * r01)
        onp.random.seed(rng_seed)
        ws_seq_T0i = _run_leg2_Ws_sequence(R, S, Ws_seed_T0i, rand2_shared)
        C3_T0i[:, i_t, comp] = onp.array(ws_seq_T0i[t])
    
    # tensor Tij (3 components: xy, xz, yz)
    for comp in range(3):
        Ws_seed_Tij = W1 * (currents['Tij'][:, comp] * r01)
        onp.random.seed(rng_seed)
        ws_seq_Tij = _run_leg2_Ws_sequence(R, S, Ws_seed_Tij, rand2_shared)
        C3_Tij[:, i_t, comp] = onp.array(ws_seq_Tij[t])

print("\n" + "="*70)


###
# form factor computation with bootstrap errors
###

print("\nComputing form factors with bootstrap errors...")

def compute_form_factor_bootstrap(C3_data, C2_data, n_boot=200):
    """
    Compute form factor R = C3 / C2 with bootstrap errors.
    
    C3 and C2 must be per-walker arrays from correlated sampling so that
    resampling with the same indices preserves per-walker correlations.
    """
    n_walkers_local = len(C3_data)
    
    # filter out NaN walkers, same mask for both to keep correlation
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
    
    mean = onp.mean(boot_vals)
    err = onp.std(boot_vals)
    
    return mean, err

# compute form factors for all currents
tau_values = onp.array(t_indices) * dtau_iMev

# number operator form factor
F_N = onp.zeros(N_tau)
F_N_err = onp.zeros(N_tau)

# scalar form factors
F_S = onp.zeros(N_tau)
F_S_err = onp.zeros(N_tau)
F_V0 = onp.zeros(N_tau)
F_V0_err = onp.zeros(N_tau)

# vector form factors (3 components each)
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

print("\n--- Vector-spatial (x,y,z components) ---")
print(f"{'tau':>8} {'F_Vx':>12} {'err':>8} {'F_Vy':>12} {'err':>8} {'F_Vz':>12} {'err':>8}")
for i_t, tau in enumerate(tau_values):
    print(f"{tau:8.3f} {F_Vi[i_t,0]:12.6f} {F_Vi_err[i_t,0]:8.4f} {F_Vi[i_t,1]:12.6f} {F_Vi_err[i_t,1]:8.4f} {F_Vi[i_t,2]:12.6f} {F_Vi_err[i_t,2]:8.4f}")

print("\n--- Axial-spatial (x,y,z components) ---")
print(f"{'tau':>8} {'F_Ax':>12} {'err':>8} {'F_Ay':>12} {'err':>8} {'F_Az':>12} {'err':>8}")
for i_t, tau in enumerate(tau_values):
    print(f"{tau:8.3f} {F_Ai[i_t,0]:12.6f} {F_Ai_err[i_t,0]:8.4f} {F_Ai[i_t,1]:12.6f} {F_Ai_err[i_t,1]:8.4f} {F_Ai[i_t,2]:12.6f} {F_Ai_err[i_t,2]:8.4f}")

print("\n--- Tensor T0i (x,y,z components) ---")
print(f"{'tau':>8} {'F_T0x':>12} {'err':>8} {'F_T0y':>12} {'err':>8} {'F_T0z':>12} {'err':>8}")
for i_t, tau in enumerate(tau_values):
    print(f"{tau:8.3f} {F_T0i[i_t,0]:12.6f} {F_T0i_err[i_t,0]:8.4f} {F_T0i[i_t,1]:12.6f} {F_T0i_err[i_t,1]:8.4f} {F_T0i[i_t,2]:12.6f} {F_T0i_err[i_t,2]:8.4f}")

print("\n--- Tensor Tij (xy, xz, yz components) ---")
print(f"{'tau':>8} {'F_Txy':>12} {'err':>8} {'F_Txz':>12} {'err':>8} {'F_Tyz':>12} {'err':>8}")
for i_t, tau in enumerate(tau_values):
    print(f"{tau:8.3f} {F_Tij[i_t,0]:12.6f} {F_Tij_err[i_t,0]:8.4f} {F_Tij[i_t,1]:12.6f} {F_Tij_err[i_t,1]:8.4f} {F_Tij[i_t,2]:12.6f} {F_Tij_err[i_t,2]:8.4f}")

print("="*70)


###
# physics sanity checks
###

print("\n" + "="*70)
print("PHYSICS SANITY CHECKS")
print("="*70)

print(f"\nExpected at q=0: F_N = F_S = F_V0 = N_coord = {N_coord}")
print(f"At largest tau:")
print(f"  F_N  = {F_N[-1]:.4f}  (phase sum)")
print(f"  F_S  = {F_S[-1]:.4f}")
print(f"  F_V0 = {F_V0[-1]:.4f}")

F_S_norm = F_S / N_coord
F_V0_norm = F_V0 / N_coord
print(f"\nCharge-normalized (should be ~1 at q=0):")
print(f"  F_S / N_coord = {F_S_norm[-1]:.4f}")
print(f"  F_V0 / N_coord = {F_V0_norm[-1]:.4f}")

if abs(Q_dir_arr[0]) > 0.5:
    print(f"\nq is along x-direction:")
    print(f"  F_Vx = {F_Vi[-1,0]:.6f} (expect finite)")
    print(f"  F_Vy = {F_Vi[-1,1]:.6f} (expect ~0)")
    print(f"  F_Vz = {F_Vi[-1,2]:.6f} (expect ~0)")

W_eff = onp.sum(onp.abs(gfmc_Ws[-1]))**2 / onp.sum(onp.abs(gfmc_Ws[-1])**2)
print(f"\nWeight health: N_eff/N_walkers = {W_eff/n_walkers:.3f}")
print("(Should be > 0.1 for good statistics)")

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

plt.show()

print("\nDone.")


###
# plateau fitting with bootstrap
###

def fit_plateau_bootstrap(F_data, tau_values, tau_min=None, tau_max=None, n_boot=500):
    """Fit a constant to the plateau region with bootstrap errors."""
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

# summary table
print("\n" + "-"*50)
print("FINAL FORM FACTORS (plateau averages)")
print("-"*50)
print(f"{'Current':<12} {'Component':<10} {'F(q^2)':<14} {'Error':<10}")
print("-"*50)

print(f"{'Number':<12} {'':<10} {F_N_final:>12.6f}   {F_N_final_err:>8.6f}  (phase sum)")
print(f"{'Scalar':<12} {'':<10} {F_S_final:>12.6f}   {F_S_final_err:>8.6f}")
print(f"{'Vector-0':<12} {'':<10} {F_V0_final:>12.6f}   {F_V0_final_err:>8.6f}")

comp_labels = ['x', 'y', 'z']
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

print("\n" + "-"*50)
print("CHARGE-NORMALIZED FORM FACTORS")
print("-"*50)
print(f"F_N / N_coord  = {F_N_final/N_coord:.6f} +/- {F_N_final_err/N_coord:.6f}  (phase sum, expect 1.0)")
print(f"F_S / N_coord  = {F_S_final/N_coord:.6f} +/- {F_S_final_err/N_coord:.6f}")
print(f"F_V0 / N_coord = {F_V0_final/N_coord:.6f} +/- {F_V0_final_err/N_coord:.6f}")
print("-"*50)


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
colors = ['C0', 'C1', 'C2']
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

plt.show()


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