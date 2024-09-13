import analysis as al
from functools import partial
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import jax.example_libraries.optimizers
import numpy as onp
import pickle
import time
import tqdm.auto as tqdm

from .util import hashabledict, jax_print, norm_3vec, norm_3vec_sq, to_relative

fm_Mev = 1.0
mp_Mev = 1.0

### GFMC utils
def draw_dR(shape, *, lam, axis=1, masses=1):
    dR = onp.transpose( lam/onp.sqrt(2) 
                        * onp.transpose( onp.random.normal(size=shape) ) )
    # subtract mean dR to avoid "drift" in the system
    dR -= np.transpose(np.transpose(np.mean(np.transpose(np.transpose(dR) 
              * masses), axis=axis, keepdims=True))/np.mean(masses))
    return dR

def step_G0_symm(R, *, dtau_iMev, m_Mev):
    dtau_fm = dtau_iMev * fm_Mev
    lam_fm = np.sqrt(2/m_Mev * fm_Mev * dtau_fm)
    dR = draw_dR(R.shape, lam=lam_fm)
    return R+dR, R-dR

def step_G0_symm_distinct(R, *, dtau_iMev, m_Mev):
    dtau_fm = dtau_iMev * fm_Mev
    lam_fm = np.sqrt(2/m_Mev * fm_Mev * dtau_fm)
    (n_walkers, n_coord, n_d) = R.shape
    dR = 1/onp.sqrt(2) * onp.random.normal(size=R.shape)
    for i in range(0, n_coord):
        dR[:,i,:] = dR[:,i,:] * lam_fm[i]
    # subtract mean dR to avoid "drift" in the system
    return R+dR, R-dR

### Hamiltonian
NS = 1
NI = 3

# Define the Gell-Mann matrices
sigma1 = 1/2*onp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
sigma2 = 1/2*onp.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
sigma3 = 1/2*onp.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
sigma4 = 1/2*onp.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
sigma5 = 1/2*onp.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
sigma6 = 1/2*onp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
sigma7 = 1/2*onp.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
sigma8 = 1/2*onp.array([[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(3), 0], 
                        [0, 0, -2 / np.sqrt(3)]])

# Stack the matrices along the third axis (depth)
paulis = onp.stack([
    onp.array([[0, 1], [1, 0]]), # X
    onp.array([[0, -1j], [1j, 0]]), # Y
    onp.array([[1, 0], [0, -1]]) # Z
])
gells = onp.stack([sigma1, sigma2, sigma3, sigma4, 
                   sigma5, sigma6, sigma7, sigma8])

for a in range(8):
    assert( np.einsum('ij,ji', gells[a], gells[a]) - 1.0/2 < 1e-6 )

for a in range(8):
    for b in range(a):
        assert( np.einsum('ij,ji', gells[a], gells[b]) < 1e-6 )

# Define the Levi-Civita symbol tensor
lc_tensor = onp.zeros((NI, NI, NI))
lc_tensor[0, 1, 2] = lc_tensor[1, 2, 0] = lc_tensor[2, 0, 1] = 1
lc_tensor[0, 2, 1] = lc_tensor[2, 1, 0] = lc_tensor[1, 0, 2] = -1

# QQ color symmetric potential operator
iso_del = 1/2 * 1/2 * (onp.einsum('ab,cd->acdb', onp.identity(NI), 
             onp.identity(NI)) + onp.einsum('ab,cd->cadb', onp.identity(NI), 
               onp.identity(NI)))

# QQ color antisymmetric potential operator
iso_eps = (NI - 1)/4 /onp.math.factorial(NI-1) \
            * onp.einsum('abo,cdo->abcd', lc_tensor, lc_tensor)

# QQbar color singlet potential operator
iso_sing = 1/NI * onp.einsum('ab,cd->abcd', onp.identity(NI), 
                             onp.identity(NI))

# QQbar color octet potential operator
iso_oct = np.zeros((NI,NI,NI,NI))
for a in range(8):
    iso_oct += 2*onp.einsum('ab,cd->bacd', gells[a], gells[a])

# NOTE(gkanwar): spin and isospin pieces are identical matrices, but are
# semantically different objects.
two_body_pieces = {
    # symmetric in color
    'iso_S': iso_del,
    # antisymmetric in color
    'iso_A': iso_eps,
    # symmetric in color
    'iso_sing': iso_sing,
    # antisymmetric in color
    'iso_oct': iso_oct,
    # 1 . 1
    'sp_I': onp.einsum('ij,kl->ikjl', onp.identity(NS), onp.identity(NS)),
    # sigma_i . sigma_j
    'sp_dot': sum(onp.einsum('ij,kl->ikjl', p, p) for p in paulis),
    # 1 . 1
    'iso_I': onp.einsum('ab,cd->acbd', onp.identity(NI), onp.identity(NI)),
    # tau_i . tau_j
    'iso_dot': sum(onp.einsum('ab,cd->acbd', p, p) for p in paulis)
}

three_body_pieces = {
    # 1 . 1 . 1
    'sp_I': onp.einsum('ij,kl,mn->ikmjln', onp.identity(NS), 
                       onp.identity(NS), onp.identity(NS)),
    # 1 . 1 . 1
    'iso_I': onp.einsum('ab,cd,ef->acebdf', onp.identity(NI), 
                        onp.identity(NI), onp.identity(NI)),
}

@partial(jax.jit)
def two_body_outer(two_body_iso, two_body_spin):
    return np.einsum('zacbd,zikjl->zaickbjdl', two_body_iso, two_body_spin)

@partial(jax.jit)
def three_body_outer(three_body_iso, three_body_spin):
    return np.einsum('zacebdf,zikmjln->zaickembjdlfn', three_body_iso, 
                     three_body_spin)

qq_two_body_ops = {
    'OA': two_body_outer(
        two_body_pieces['iso_A'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
    'OS': two_body_outer(
        two_body_pieces['iso_S'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
}

qqbar_two_body_ops = {
    'OSing': two_body_outer(
        two_body_pieces['iso_sing'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
    'OO': two_body_outer(
        two_body_pieces['iso_oct'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
}

def get_qq_two_body_ops(x):
    return x
def get_qqbar_two_body_ops(x):
    return x

three_body_ops = {
    'O1': lambda Rij, Rjk, Rik: three_body_outer(
        1/6*three_body_pieces['iso_I'][np.newaxis],
        three_body_pieces['sp_I'][np.newaxis]),
    'OA': lambda Rij, Rjk, Rik: three_body_outer(
        three_body_pieces['iso_I'][np.newaxis],
        three_body_pieces['sp_I'][np.newaxis]),
    'OS': lambda Rij, Rjk, Rik: three_body_outer(
        three_body_pieces['iso_I'][np.newaxis],
        three_body_pieces['sp_I'][np.newaxis]),
}

def generate_sequence(AA):
    sequence = [0, 1]
    evens = [i for i in range(4, AA, 2)]
    sequence.extend(evens)
    sequence.extend([2,3])
    odds = [i for i in range(5, AA, 2)]
    sequence.extend(odds)
    return sequence

def extend_sequence(seq):
    seqlen = len(seq)
    # batch index
    newseq = [0]
    # pad batch index and double spin/color
    for i in range(seqlen):
        newseq.append(2*seq[i] + 1)
        newseq.append(2*seq[i] + 1 + 1)
    return newseq

def generate_full_sequence(AA):
    return extend_sequence( generate_sequence(AA) )

def make_pairwise_potential(AVcoeffs, B3coeffs, masses):
    @jax.jit
    def pairwise_potential(R):
        start_pot = time.time()
        batch_size, A = R.shape[:2]
        V_SI_Mev = 0
        V_SD_Mev = np.zeros( # big matrix
            (batch_size,) + # batch of walkers
            (NI,NS)*A + # source (i1, s1, i2, s2, ...)
            (NI,NS)*A, # sink (i1', s1', i2', s2', ...)
            dtype=np.complex128
        )
        # two-body potentials
        for i in range(A):
            for j in range(A):
                if i==j:
                    continue
                Rij = R[:,i] - R[:,j]
                this_two_body_ops = qqbar_two_body_ops 
                if masses[i]*masses[j]>0:
                    this_two_body_ops = qq_two_body_ops
                elif masses[i] < masses[j]:
                    continue
                print("i = ", i, ", j = ", j)
                perm = [ l for l in range(2*A) ]
                perm[0] = 2*i
                perm[1] = 2*i+1
                perm[2*i] = 0
                perm[2*i+1] = 1
                perm_copy = perm.copy()
                j_slot = perm.index(2)
                perm[2*j] = perm_copy[j_slot]
                perm[2*j+1] = perm_copy[j_slot+1]
                perm[j_slot] = perm_copy[2*j]
                perm[j_slot+1] = perm_copy[2*j+1]
                src_perm = [ perm[l] + 1 for l in range(len(perm)) ]
                snk_perm = [ src_perm[l] + 2*A for l in range(len(perm)) ]
                full_perm = [0] + src_perm + snk_perm
                print("full perm = ",full_perm)
                scaled_O = np.zeros_like(qq_two_body_ops["OA"])
                for name,op in this_two_body_ops.items():
                    if name not in AVcoeffs: continue
                    print('including op', name)
                    Oij = op
                    vij = AVcoeffs[name](Rij)
                    broadcast_vij_inds = (slice(None),) \
                                          + (np.newaxis,)*(len(Oij.shape)-1)
                    vij = vij[broadcast_vij_inds]
                    scaled_O += vij * Oij

                # tensor in identity matrices in color and spin
                # fast path for NS == 1
                if NS == 1:
                    old_shape = scaled_O.shape
                    for alpha in range(A-2):
                        scaled_O = np.einsum('...,mn->...mn', scaled_O, 
                                             onp.identity(NI))
                        old_shape += (NI, NS, NI, NS)
                    scaled_O = np.reshape(scaled_O, old_shape)
                # general case (should work for any NS)
                else:
                    for alpha in range(A-2):
                        scaled_O = np.einsum('...,mn,op->...monp', scaled_O, 
                                      onp.identity(NI), onp.identity(NS))

                assert V_SD_Mev.shape==scaled_O.shape
                starting_perm = generate_full_sequence(2*A)
                print('starting_perm',starting_perm)
                scaled_O = np.transpose(scaled_O, axes=starting_perm)
                scaled_O_perm = np.transpose(scaled_O, axes=full_perm)
                if name == 'O1':
                    broadcast_inds = (slice(None),) + (0,)*(len(Oij.shape)-1)
                    V_SD_Mev += scaled_O[broadcast_inds]
                else:
                    V_SD_Mev += scaled_O_perm
        end_pot = time.time()
        jax.debug.print("Time spent in pairwise_potential: " + str(end_pot - start_pot))
        return V_SI_Mev, V_SD_Mev
    return pairwise_potential

def batched_apply(M, S): # compute M|S>
    batch_size, src_sink_dims = M.shape[0], M.shape[1:]
    batch_size2, src_dims = S.shape[0], S.shape[1:]
    assert (batch_size == batch_size2 or
            batch_size == 1 or batch_size2 == 1), \
            'batch size must be broadcastable'
    assert src_sink_dims == src_dims + src_dims, \
            'matrix dims must match vector dims'
    inds_M = list(range(len(M.shape)))
    inds_S = [0] + list(range(len(S.shape), 2*len(S.shape)-1))
    inds_out = [0] + list(range(1, len(S.shape)))
    return np.einsum(M, inds_M, S, inds_S, inds_out)

@partial(jax.jit)
def inner(S, Sp):
    assert Sp.shape == S.shape
    spin_iso_axes = tuple(range(1, len(S.shape)))
    return np.sum(np.conjugate(S) * Sp, axis=spin_iso_axes)

### Useful functions to bootstrap over
def rw_mean(O, w):
    return np.real(al.mean(O * w) / al.mean(w))

### Metropolis/GFMC
def metropolis(R, W, *, n_therm, n_step, n_skip, eps, masses=1):
    samples = []
    acc = 0
    print('R0=',R)
    for i in tqdm.tqdm(range(-n_therm, n_step*n_skip)):
        dR = draw_dR(R.shape, lam=eps/masses, axis=0, masses=masses)
        new_R = R + dR
        W_R = W(R)
        new_W_R = W(new_R)
        if onp.random.random() < (new_W_R / W_R):
            R = new_R # accept
            W_R = new_W_R
            acc += 1
        if i >= 0 and (i+1) % n_skip == 0:
            samples.append((R, W_R))
    print(f'Total acc frac = {acc} / {n_therm+n_skip*n_step} = {1.0*acc/(n_therm+n_skip*n_step)}')
    return samples

### Apply exp(-dtau/2 V_SI) (1 - (dtau/2) V_SD + (dtau^2/8) V_SD^2) to |S>.
#@partial(jax.jit, static_argnums=(2,))
def compute_VS(R_deform, psi, potential, *, dtau_iMev):
    start_VS = time.time()
    N_coord = R_deform.shape[1]
    old_psi = psi
    V_SI, V_SD = potential(R_deform)
    VS = batched_apply(V_SD, psi)
    VVS = batched_apply(V_SD, VS)
    psi = psi - (dtau_iMev/2) * VS + (dtau_iMev**2/8) * VVS
    end_VS = time.time()
    print("Time in compute_VS: " + str(end_VS - start_VS))
    return psi

#@partial(jax.jit, static_argnums=(8,9), static_argnames=('deform_f',))
def kinetic_step_absolute(R_fwd, R_bwd, R, R_deform, psi, u, params_i, psi0,
                 potential, *, dtau_iMev, m_Mev):
    start_kin = time.time()
    """Step forward given two possible proposals and RNG to decide"""
    # transform manifold
    R_fwd_old = R_fwd
    R_bwd_old = R_bwd

    psi_fwd = compute_VS(R_fwd, psi, potential, dtau_iMev=dtau_iMev)
    psi_bwd = compute_VS(R_bwd, psi, potential, dtau_iMev=dtau_iMev)

    w_fwd = np.abs( inner( psi0, psi_fwd ) )
    w_bwd = np.abs( inner( psi0, psi_bwd ) )

    # correct kinetic energy
    denom = 1/(2*dtau_iMev*fm_Mev**2/m_Mev)

    G_ratio_fwd_num = -np.einsum('...j,...j->...', R_fwd-R_deform, 
        R_fwd-R_deform)+np.einsum('...j,...j->...', R_fwd_old-R, R_fwd_old-R)
    G_ratio_fwd = np.exp(np.einsum('...i,i->...', G_ratio_fwd_num, denom))
    G_ratio_bwd_num = -np.einsum('...j,...j->...', R_bwd-R_deform, 
        R_bwd-R_deform)+np.einsum('...j,...j->...', R_bwd_old-R, R_bwd_old-R)
    G_ratio_bwd = np.exp(np.einsum('...i,i->...', G_ratio_bwd_num, denom))
    w_fwd = w_fwd * G_ratio_fwd
    w_bwd = w_bwd * G_ratio_bwd

    p_fwd = np.abs(w_fwd) / (np.abs(w_fwd) + np.abs(w_bwd))
    pc_fwd = w_fwd / (w_fwd + w_bwd)
    p_bwd = np.abs(w_bwd) / (np.abs(w_fwd) + np.abs(w_bwd))
    pc_bwd = w_bwd / (w_fwd + w_bwd)
    ind_fwd = u < p_fwd
    ind_fwd_R = np.expand_dims(ind_fwd, axis=(-2,-1))
    R = np.where(ind_fwd_R, R_fwd_old, R_bwd_old)
    R_deform = np.where(ind_fwd_R, R_fwd, R_bwd)
    N_coord = R_deform.shape[1]
    axis_tup = tuple([i for i in range(-2*N_coord,0)])
    ind_fwd_S = np.expand_dims(ind_fwd, axis=axis_tup)
    psi = np.where(ind_fwd_S, psi_fwd, psi_bwd)
    W = ((w_fwd + w_bwd) / 2)
    W = np.where(ind_fwd, W * pc_fwd / p_fwd, W * pc_bwd / p_bwd)
    end_kin = time.time()
    print("Time in kinetic_step_absolute: " + str(end_kin - start_kin))
    return R, R_deform, psi, W

def gfmc_deform(
        R0, psi_T, params, *, rand_draws, tau_iMev, N, potential, m_Mev,
        deform_f, resampling_freq=None):
    params0 = tuple(param[0] for param in params)
    R0_deform = deform_f(R0, *params0)
    psi0 = psi_T(R0_deform)
    W = np.ones_like( inner(psi0, psi0) )
    walkers = (R0, R0_deform, psi0, W)
    dtau_iMev = tau_iMev/N
    history = [walkers]

    for i in range(N):
        _start = time.time()
        print("step ", i)
        R, R_deform, psi, W = walkers
        print("W shape ", W.shape)

        drift = np.einsum("jik,i->jk", R, m_Mev) / np.sum(m_Mev)
        print("checking drift = 0 =", np.mean(drift, axis=0))
        print("<W^2>/<W>^2 = ", np.mean(W*W)/np.mean(W))
        print("<r_first> = ", np.mean(W*np.transpose(norm_3vec(R)[:,0]))
              / np.mean(W))
        print("<r_last> = ", np.mean(W*np.transpose(norm_3vec(R)[:,-1]))
              / np.mean(W))

        # remove previous factors (to be replaced with current factors 
        # after evolving)
        # TODO DOES THIS NEED NP.ARRAY ????
        W = W / np.abs( inner(psi0,psi) )
        print("W shape after removing previous factors ", W.shape)

        # exp(-dtau V/2)|R,S>
        psi = compute_VS(R_deform, psi, potential, dtau_iMev=dtau_iMev)

        # exp(-dtau V/2) exp(-dtau K)|R,S> using fwd/bwd heatbath
        R_fwd, R_bwd = step_G0_symm_distinct(onp.array(R), 
                          dtau_iMev=dtau_iMev, m_Mev=m_Mev)
        R_fwd, R_bwd = np.array(R_fwd), np.array(R_bwd)
        u = rand_draws[i] # np.array(onp.random.random(size=R_fwd.shape[0]))
        step_params = tuple(param[i+1] for param in params)
        R, R_deform, psi, dW = kinetic_step_absolute(
            R_fwd, R_bwd, R, R_deform, psi, u, step_params, psi0,
            potential, dtau_iMev=dtau_iMev, m_Mev=m_Mev)

        # incorporate factors <S_T|S_i> f(R_i) and leftover 
        # fwd/bwd factors from the kinetic step
        W = W * dW
        print("W shape after kinetic step ", W.shape)

        # save config for obs
        history.append((R, R_deform, psi, W))

        gfmc_Rs = np.array([Rs for Rs,_,_,_, in history])

        if resampling_freq is not None and (i+1) % resampling_freq == 0:
            assert len(W.shape) == 1, 'weights must be flat array'
            p = np.abs(W) / np.sum(np.abs(W))
            W = np.mean(np.abs(W), keepdims=True) * W / np.abs(W)
            inds = onp.random.choice(onp.arange(W.shape[0]), 
                                     size=W.shape[0], p=p)
            R = R[inds]
            R_deform = R_deform[inds]
            psi = psi[inds]
            W = W[inds]

        walkers = (R, R_deform, psi, W)
        _step_time = time.time()-_start
        print(f'computed step in {_step_time:.1f}s')

    return history
