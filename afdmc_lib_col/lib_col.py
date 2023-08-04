import analysis as al
from functools import partial
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
#import jax.experimental.optimizers
import jax.example_libraries.optimizers
import numpy as onp
import pickle
import time
import tqdm.auto as tqdm

from .util import hashabledict, jax_print, norm_3vec, norm_3vec_sq, to_relative

#fm_Mev = 197.326
#mp_Mev = 938
fm_Mev = 1.0
mp_Mev = 1.0

### GFMC utils
def draw_dR(shape, *, lam, axis=1):
    dR = lam/onp.sqrt(2) * onp.random.normal(size=shape)
    # subtract mean dR to avoid "drift" in the system
    dR -= onp.mean(dR, axis=axis, keepdims=True)
    return dR
def step_G0(R, *, dtau_iMev, m_Mev):
    dtau_fm = dtau_iMev * fm_Mev
    lam_fm = np.sqrt(2/m_Mev * fm_Mev * dtau_fm)
    dR = draw_dR(R.shape, lam=lam_fm)
    return R+dR
def step_G0_symm(R, *, dtau_iMev, m_Mev):
    dtau_fm = dtau_iMev * fm_Mev
    lam_fm = np.sqrt(2/m_Mev * fm_Mev * dtau_fm)
    # print(f'lam_fm = {lam_fm}')
    dR = draw_dR(R.shape, lam=lam_fm)
    return R+dR, R-dR

def direct_sample_quarkonium(n_meas, f_R, *, a0):
    shape = (n_meas)
    u = onp.random.uniform(size=shape)
    theta = onp.pi*onp.random.uniform(size=shape)
    phi = 2*onp.pi*onp.random.uniform(size=shape)
    r = -a0*onp.log(u)
    Rrel = onp.array([r*onp.sin(theta)*onp.cos(phi), r*onp.sin(theta)*onp.sin(phi), r*onp.cos(theta)])
    R = onp.array([Rrel/2, -Rrel/2])
    samples = [(onp.array(R[:,:,n]), f_R(R[:,:,n])) for n in range(n_meas)]
    return samples

def normalize_wf(f_R, df_R, ddf_R):
    Rs = onp.linspace([0,0,0], [20,0,0], endpoint=False, num=10000)
    rs = onp.array(norm_3vec(Rs))
    f = f_R(Rs)
    dR = onp.array(norm_3vec(Rs[0] - Rs[1]))
    fnorm = onp.sqrt(onp.sum(4*np.pi * rs**2 * f**2 * dR))
    f_R_norm = jax.jit(lambda R: f_R(R) / fnorm)
    df_R_norm = jax.jit(lambda R: df_R(R) / fnorm)
    ddf_R_norm = jax.jit(lambda R: ddf_R(R) / fnorm)
    return f_R_norm, df_R_norm, ddf_R_norm


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
sigma8 = 1/2*onp.array([[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(3), 0], [0, 0, -2 / np.sqrt(3)]])

# Stack the matrices along the third axis (depth)
paulis = onp.stack([
    onp.array([[0, 1], [1, 0]]), # X
    onp.array([[0, -1j], [1j, 0]]), # Y
    onp.array([[1, 0], [0, -1]]) # Z
])
gells = onp.stack([sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8])

for a in range(8):
    assert( np.einsum('ij,ji', gells[a], gells[a]) - 1.0/2 < 1e-6 )

for a in range(8):
    for b in range(a):
        assert( np.einsum('ij,ji', gells[a], gells[b]) < 1e-6 )

#define levi-cevita tensor
# Define the Levi-Civita symbol tensor
lc_tensor = onp.zeros((NI, NI, NI))
#lc_tensor[0, 1, 2] = lc_tensor[1, 2, 0] = lc_tensor[2, 0, 1] = 1
#lc_tensor[0, 2, 1] = lc_tensor[2, 1, 0] = lc_tensor[1, 0, 2] = -1
lc_tensor[0,0,0] = 1

# QQ color symmetric potential operator
iso_del = 1/2 * 1/2 * (onp.einsum('ab,cd->acdb', onp.identity(NI), onp.identity(NI)) + onp.einsum('ab,cd->cadb', onp.identity(NI), onp.identity(NI)))

# QQ color antisymmetric potential operator
iso_eps = (NI - 1)/4 /onp.math.factorial(NI-1) * onp.einsum('abo,cdo->abcd', lc_tensor, lc_tensor)

# QQbar color singlet potential operator
iso_sing = 1/NI * onp.einsum('ab,cd->abcd', onp.identity(NI), onp.identity(NI))

# QQbar color octet potential operator
iso_oct = np.zeros((NI,NI,NI,NI))
for a in range(8):
    iso_oct += 2*onp.einsum('ab,cd->abcd', gells[a], gells[a])

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
    'sp_I': onp.einsum('ij,kl,mn->ikmjln', onp.identity(NS), onp.identity(NS), onp.identity(NS)),
    # 1 . 1 . 1
    'iso_I': onp.einsum('ab,cd,ef->acebdf', onp.identity(NI), onp.identity(NI), onp.identity(NI)),
}
@partial(jax.jit)
def Sij(Rij):
    batch_size, nd = Rij.shape
    assert nd == 3, 'Rij must be batched 3-vectors'
    # pauli_Rij = np.einsum('xij,ax->aij', paulis, Rij)
    pauli_Rij = np.tensordot(Rij, paulis, axes=1) # \vec{R} . \vec{pauli}
    rij2 = norm_3vec_sq(Rij)
    # term1 = 3*np.einsum('aij,akl,a->aikjl', pauli_Rij, pauli_Rij, 1/rij2)
    pauli_Rij_rescale = pauli_Rij / rij2[:,np.newaxis,np.newaxis]
    term1 = 3*pauli_Rij_rescale[...,np.newaxis,np.newaxis]*pauli_Rij[:,np.newaxis,np.newaxis]
    term1 = np.swapaxes(term1, axis1=2, axis2=3)
    term2 = two_body_pieces['sp_dot'][onp.newaxis,:,:,:,:]
    return term1 - term2

@partial(jax.jit)
def two_body_outer(two_body_iso, two_body_spin):
    return np.einsum('zacbd,zikjl->zaickbjdl', two_body_iso, two_body_spin)

@partial(jax.jit)
def three_body_outer(three_body_iso, three_body_spin):
    return np.einsum('zacebdf,zikmjln->zaickembjdlfn', three_body_iso, three_body_spin)

qq_two_body_ops = {
    'OA': lambda Rij: two_body_outer(
        two_body_pieces['iso_A'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
    'OS': lambda Rij: two_body_outer(
        two_body_pieces['iso_S'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
}

qqbar_two_body_ops = {
    'OSing': lambda Rij: two_body_outer(
        two_body_pieces['iso_sing'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
    'OO': lambda Rij: two_body_outer(
        two_body_pieces['iso_oct'][np.newaxis],
        two_body_pieces['sp_I'][np.newaxis]),
}

def get_qq_two_body_ops(x):
    return x
def get_qqbar_two_body_ops(x):
    return x

three_body_ops = {
    'O1': lambda Rij, Rjk, Rik: three_body_outer(
        #three_body_pieces['iso_I'][np.newaxis],
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
    newseq = [0]
    for i in range(seqlen):
        newseq.append(2*seq[i] + 1)
        newseq.append(2*seq[i] + 1 + 1)
    return newseq

def generate_full_sequence(AA):
    return extend_sequence( generate_sequence(AA) )

def make_pairwise_potential(AVcoeffs, B3coeffs, masses):
    @jax.jit
    def pairwise_potential(R):
        batch_size, A = R.shape[:2]
        V_SI_Mev = np.zeros( # big-ass matrix
            (batch_size,) + # batch of walkers
            (NI,NS)*A + # source (i1, s1, i2, s2, ...)
            (NI,NS)*A, # sink (i1', s1', i2', s2', ...)
            dtype=np.complex128
        )
        V_SD_Mev = np.zeros( # big-ass matrix
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
                this_two_body_ops = qqbar_two_body_ops #jax.lax.cond(masses[i]*masses[j]>0, get_qq_two_body_ops, get_qqbar_two_body_ops, qq_two_body_ops)
                if masses[i]*masses[j]>0:
                    this_two_body_ops = qq_two_body_ops
                elif masses[i] < masses[j]:
                    continue
                print("i = ", i, ", j = ", j)
                for name,op in this_two_body_ops.items():
                    if name not in AVcoeffs: continue
                    print('including op', name)
                    Oij = op(Rij)
                    vij = AVcoeffs[name](Rij)
                    broadcast_vij_inds = (slice(None),) + (np.newaxis,)*(len(Oij.shape)-1)
                    vij = vij[broadcast_vij_inds]
                    scaled_O = vij * Oij
                    for alpha in range(A-2):
                        scaled_O = np.einsum('...,mn,op->...monp', scaled_O, onp.identity(NI), onp.identity(NS))
                    assert V_SI_Mev.shape==scaled_O.shape
                    basic_perm = generate_sequence(2*A)
                    print('basic_perm',basic_perm)
                    starting_perm = generate_full_sequence(2*A)
                    print('starting_perm',starting_perm)
                    scaled_O = np.transpose(scaled_O, axes=starting_perm)
                    #print("scaled_O shape =",scaled_O.shape)
                    #print("i = ",i," j = ", j)
                    #perm = [ l for l in range(A) ]
                    #perm[0] = i
                    #perm[i] = 0
                    #perm_copy = perm.copy()
                    #j_slot = perm.index(j)
                    #perm[1] = perm_copy[j_slot]
                    #perm[j_slot] = perm_copy[1]
                    #print(perm)
                    perm = [ l for l in range(2*A) ]
                    perm[0] = 2*i
                    perm[1] = 2*i+1
                    perm[2*i] = 0
                    perm[2*i+1] = 1
                    perm_copy = perm.copy()
                    j_slot = perm.index(2*j)
                    perm[2] = perm_copy[j_slot]
                    perm[3] = perm_copy[j_slot+1]
                    perm[j_slot] = perm_copy[2]
                    perm[j_slot+1] = perm_copy[3]
                    #src_perm = [ perm[l] + 1 for l in range(len(perm)) ]
                    src_perm = [ perm[l] + 1 for l in range(len(perm)) ]
                    snk_perm = [ src_perm[l] + 2*A for l in range(len(perm)) ]
                    full_perm = [0] + src_perm + snk_perm
                    #print(perm)
                    #print("full perm = ",full_perm)
                    scaled_O_perm = np.transpose(scaled_O, axes=full_perm)
                    if name == 'O1':
                        broadcast_inds = (slice(None),) + (0,)*(len(Oij.shape)-1)
                        V_SI_Mev = V_SI_Mev + scaled_O[broadcast_inds]
                    #    V_SI_Mev += scaled_O_perm[broadcast_inds]
                    else:
                        V_SD_Mev += scaled_O_perm
                        #print("O ", Oij[0,0,0,1,0,0,0,1,0])
                        #print("scaled O ", scaled_O[0,0,0,1,0,0,0,1,0,2,0,2,0])
                        #print("scaled O perm ", scaled_O_perm[0,0,0,1,0,0,0,1,0,2,0,2,0])
                        #print("V_SD ", V_SD_Mev[0,0,0,1,0,0,0,1,0,2,0,2,0])
                        #print(Oij.shape)
                        #print(scaled_O.shape)
        return V_SI_Mev, V_SD_Mev
    return pairwise_potential

def make_pairwise_product_potential(AVcoeffs, B3coeffs, masses):
    @jax.jit
    def pairwise_potential(R):
        batch_size, A = R.shape[:2]
        V_SI_Mev = np.zeros( # big-ass matrix
            (batch_size,) + # batch of walkers
            (NI,NS)*A + # source (i1, s1, i2, s2, ...)
            (NI,NS)*A, # sink (i1', s1', i2', s2', ...)
            dtype=np.complex128
        )
        V_SD_Mev = np.zeros( # big-ass matrix
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
                this_two_body_ops = qqbar_two_body_ops #jax.lax.cond(masses[i]*masses[j]>0, get_qq_two_body_ops, get_qqbar_two_body_ops, qq_two_body_ops)
                if masses[i]*masses[j]>0:
                    this_two_body_ops = qq_two_body_ops
                if i < A//2 and j >= A//2:
                    continue
                if j < A//2 and i >= A//2:
                    continue
                elif masses[i] < masses[j]:
                    continue
                print("i = ", i, ", j = ", j)
                for name,op in this_two_body_ops.items():
                    if name not in AVcoeffs: continue
                    print('including op', name)
                    Oij = op(Rij)
                    vij = AVcoeffs[name](Rij)
                    broadcast_vij_inds = (slice(None),) + (np.newaxis,)*(len(Oij.shape)-1)
                    vij = vij[broadcast_vij_inds]
                    scaled_O = vij * Oij
                    for alpha in range(A-2):
                        scaled_O = np.einsum('...,mn,op->...monp', scaled_O, onp.identity(NI), onp.identity(NS))
                    assert V_SI_Mev.shape==scaled_O.shape
                    basic_perm = generate_sequence(2*A)
                    print('basic_perm',basic_perm)
                    starting_perm = generate_full_sequence(2*A)
                    print('starting_perm',starting_perm)
                    scaled_O = np.transpose(scaled_O, axes=starting_perm)
                    #print("scaled_O shape =",scaled_O.shape)
                    #print("i = ",i," j = ", j)
                    #perm = [ l for l in range(A) ]
                    #perm[0] = i
                    #perm[i] = 0
                    #perm_copy = perm.copy()
                    #j_slot = perm.index(j)
                    #perm[1] = perm_copy[j_slot]
                    #perm[j_slot] = perm_copy[1]
                    #print(perm)
                    perm = [ l for l in range(2*A) ]
                    perm[0] = 2*i
                    perm[1] = 2*i+1
                    perm[2*i] = 0
                    perm[2*i+1] = 1
                    perm_copy = perm.copy()
                    j_slot = perm.index(2*j)
                    perm[2] = perm_copy[j_slot]
                    perm[3] = perm_copy[j_slot+1]
                    perm[j_slot] = perm_copy[2]
                    perm[j_slot+1] = perm_copy[3]
                    #src_perm = [ perm[l] + 1 for l in range(len(perm)) ]
                    src_perm = [ perm[l] + 1 for l in range(len(perm)) ]
                    snk_perm = [ src_perm[l] + 2*A for l in range(len(perm)) ]
                    full_perm = [0] + src_perm + snk_perm
                    #print(perm)
                    #print("full perm = ",full_perm)
                    scaled_O_perm = np.transpose(scaled_O, axes=full_perm)
                    if name == 'O1':
                        broadcast_inds = (slice(None),) + (0,)*(len(Oij.shape)-1)
                        V_SI_Mev = V_SI_Mev + scaled_O[broadcast_inds]
                    #    V_SI_Mev += scaled_O_perm[broadcast_inds]
                    else:
                        V_SD_Mev += scaled_O_perm
                        #print("O ", Oij[0,0,0,1,0,0,0,1,0])
                        #print("scaled O ", scaled_O[0,0,0,1,0,0,0,1,0,2,0,2,0])
                        #print("scaled O perm ", scaled_O_perm[0,0,0,1,0,0,0,1,0,2,0,2,0])
                        #print("V_SD ", V_SD_Mev[0,0,0,1,0,0,0,1,0,2,0,2,0])
                        #print(Oij.shape)
                        #print(scaled_O.shape)
        return V_SI_Mev, V_SD_Mev
    return pairwise_potential

#@partial(jax.jit)
def batched_apply(M, S): # compute M|S>
    batch_size, src_sink_dims = M.shape[0], M.shape[1:]
    batch_size2, src_dims = S.shape[0], S.shape[1:]
    print(src_sink_dims)
    print(src_dims)
    assert (batch_size == batch_size2 or
            batch_size == 1 or batch_size2 == 1), 'batch size must be broadcastable'
    assert src_sink_dims == src_dims + src_dims, 'matrix dims must match vector dims'
    inds_M = list(range(len(M.shape)))
    inds_S = [0] + list(range(len(S.shape), 2*len(S.shape)-1))
    inds_out = [0] + list(range(1, len(S.shape)))
    return np.einsum(M, inds_M, S, inds_S, inds_out)

@partial(jax.jit)
def batched_apply_transpose(M, S): # compute <S|M
    batch_size, src_sink_dims = M.shape[0], M.shape[1:]
    batch_size2, sink_dims = S.shape[0], S.shape[1:]
    assert (batch_size == batch_size2 or
            batch_size == 1 or batch_size2 == 1), 'batch size must be broadcastable'
    assert src_sink_dims == sink_dims + sink_dims, 'matrix dims must match vector dims'
    inds_M = list(range(len(M.shape)))
    inds_S = [0] + list(range(1, len(S.shape)))
    inds_out = [0] + list(range(len(S.shape), 2*len(S.shape)-1))
    return np.einsum(M, inds_M, S, inds_S, inds_out)

@partial(jax.jit)
def inner(S, Sp):
    assert Sp.shape == S.shape
    spin_iso_axes = tuple(range(1, len(S.shape)))
    return np.sum(np.conjugate(S) * Sp, axis=spin_iso_axes)

@partial(jax.jit)
def batched_vev(M, S): # compute <S|M|S>
    Sp = batched_apply(M, S)
    return np.real(inner(S, Sp))

# evaluate <psi|H|psi> where R,S are a batch of pointlike evaluations
# of |psi>, and logw is the amplitude weight for each point
def make_evaluate_H(potential):
    def evaluate_H(R, S, f, df, ddf, *, m_Mev):
        V_SI_Mev, V_SD_Mev = potential(R)
        V_Mev = V_SI_Mev + V_SD_Mev
        r = norm_3vec(to_relative(R))
        S_V_S = batched_vev(V_Mev, S)
        psi_V_psi_Mev = r**2 * f**2 * S_V_S
        psi_K_psi_Mev = r *(r * f * ddf + f * df * 2) * (fm_Mev**2) / (-m_Mev)
        print('<K> = ', np.sum(psi_K_psi_Mev))
        print('<V> = ', np.sum(psi_V_psi_Mev))
        w_Mev = psi_V_psi_Mev + psi_K_psi_Mev
        return np.sum(w_Mev)
    return evaluate_H


### Useful functions to bootstrap over
def normed_mean(O, w):
    return np.real(al.mean(O) / al.mean(w))

def rw_mean(O, w):
    return np.real(al.mean(O * w) / al.mean(w))

@partial(jax.jit)
def compute_K(R, f, df, ddf, *, m_Mev):
    # non-holomorphic version
    # rs = norm_3vec(R)
    # K_Mev = (1 / rs) * (-1) * (rs*ddf + 2*df)*fm_Mev**2/(m_Mev)
    # holomorphic version (NOTE: it is assumed that the f included in the GFMC
    # weights is appropriately deformed when K is, so that (1/f) here does not
    # spoil holomorphy)
    rsq = norm_3vec_sq(R)
    K_Mev = (-1) * (6*df + 4*rsq*ddf)*fm_Mev**2/ (f * m_Mev)
    return K_Mev

@partial(jax.jit)
def compute_O(op, S, S_T):
    S_T_dot_S = inner(S_T, S)
    return inner(S_T, batched_apply(op, S)) / S_T_dot_S

# evaluate <psi_T|H|psi(tau)>
# where R,S are a batch of pointlike evaluations
# of |psi>
def make_twobody_estimate_H(AVcoeffs):
    def estimate_H(R, S, W, S_T, f, df, ddf, *, m_Mev, Nboot, verbose=False):
        """Estimate <psi_T|H|psi(tau)>/<psi_T|psi(tau)> given metropolis samples from psi(tau)

        `R` and `S` are samples from GFMC evolution of psi_T. `W` is the reweighting
        factor from q(R) to I(R), defined by \expt{I(R)} = <psi_T(R) | psi(tau,R)>.

        `f`, `df`, and `ddf` are the scalar amplitude/derivates psi_T(R) on the
        metropolis samples `R`.
        TODO: for now `R` is relative coordinates between pair of nucleons, should
        upgrade to absolute coordinates of A nucleons.
        """
        K_Mev = onp.array(compute_K(R, f, df, ddf, m_Mev=m_Mev))
        est_K = al.bootstrap(K_Mev, onp.array(W), Nboot=Nboot, f=rw_mean)
        if verbose: tqdm.tqdm.write(f'<K> = {est_K}')
        Os = {
            name: onp.array(
                AVcoeffs[name](R) * compute_O(two_body_ops[name](R), S, S_T))
            for name in AVcoeffs
        }
        if verbose:
            for name in AVcoeffs:
                boot_val = al.bootstrap(Os[name], onp.array(W), Nboot=Nboot, f=rw_mean)
                tqdm.tqdm.write(f'<{name}> = {boot_val}')
        V_Mev = sum(Os.values())
        est_V = al.bootstrap(V_Mev, W, Nboot=Nboot, f=rw_mean)
        if verbose: tqdm.tqdm.write(f'<V> = {est_V}')
        H_Mev = K_Mev + V_Mev
        est_H = al.bootstrap(H_Mev, W, Nboot=Nboot, f=rw_mean)
        if verbose: tqdm.tqdm.write(f'<H> = {est_H}')
        res = {
            'H': est_H,
            'K': est_K,
            'V': est_V,
        }
        return res
    return estimate_H

def measure_eucl_density_response(R0, RN, *, q_Mev, i, j):
    """Evaluate <0|rho_j^dag(q)|R'><R'|e^{-H tau}|R><R|rho_i(q)|0>
    where rho_i(q) = exp(-i R_i . q)
    and R_i is the ith particle coord (deformed or undeformed) relative to the CoM.
    The full density response is the elementwise sum over all (i,j), but we may want
    to deform these components independently.
    """
    NMeas, A, Nd = R0.shape
    assert q_Mev.shape == (Nd,)
    assert RN.shape == R0.shape
    tildeR0i = (R0 - np.mean(R0, axis=1, keepdims=True))[...,i,:]
    tildeRNj = (RN - np.mean(RN, axis=1, keepdims=True))[...,j,:]
    xN = np.dot(tildeRNj, q_Mev) / fm_Mev
    x0 = np.dot(tildeR0i, q_Mev) / fm_Mev
    return np.exp(1j * xN) * np.exp(-1j * x0)


# evaluate <psi_T|H|psi(tau)>
# where R,S are a batch of pointlike evaluations
# of |psi>
# def make_twobody_sample_mean_H(AVcoeffs):
#     def estimate_H(R, S, Ws, S_T, f, df, ddf, *, m_Mev, verbose=False):
#         """Estimate <psi_T|H|psi(tau)>/<psi_T|psi(tau)> given metropolis samples from psi(tau)

#         `R` and `S` are samples from GFMC evolution of psi_T. `W` is the reweighting
#         factor from q(R) to I(R), defined by \expt{I(R)} = <psi_T(R) | psi(tau,R)>.

#         `f`, `df`, and `ddf` are the scalar amplitude/derivates psi_T(R) on the
#         metropolis samples `R`.
#         TODO: for now `R` is relative coordinates between pair of nucleons, should
#         upgrade to absolute coordinates of A nucleons.
#         """
#         W, Wp = Ws # W for <V>, Wp for <K>
#         K_Mev = np.array(compute_K(R, f, df, ddf, m_Mev=m_Mev))
#         est_K = np.mean(np.real(K_Mev))/np.mean(np.real(W))
#         if verbose: tqdm.tqdm.write(f'<K> = {est_K}')
#         Os = {
#             name: np.array(
#                 compute_O(two_body_ops[name](R), R, S, W, S_T, AVcoeff=AVcoeffs[name](R)))
#             for name in AVcoeffs
#         }
#         if verbose:
#             for name in AVcoeffs:
#                 boot_val = np.mean(np.real(Os[name]))/np.mean(np.real(W))
#                 tqdm.tqdm.write(f'<{name}> = {boot_val}')
#         V_Mev = sum(Os.values())
#         est_V = np.mean(np.real(V_Mev))/np.mean(np.real(W))
#         if verbose: tqdm.tqdm.write(f'<V> = {est_V}')
#         H_Mev = K_Mev + V_Mev
#         est_H = np.mean(np.real(H_Mev))/np.mean(np.real(W))
#         est_H2_num = np.mean(np.real(H_Mev)**2)/np.mean(np.real(W))
#         est_H2_den = np.mean(np.real(W)**2)/np.mean(np.real(W))
#         if verbose: tqdm.tqdm.write(f'<H> = {est_H}')
#         res = {
#             'H': est_H,
#             'K': est_K,
#             'V': est_V,
#             'H2_num': est_H2_num,
#             'H2_den': est_H2_den
#         }
#         return res
#     return estimate_H

### Metropolis/GFMC
def make_wf_weight(f_R):
    def weight(R):
        dR = R[...,0,:] - R[...,1,:]
        # TODO: why do we take a real here?
        return onp.real(f_R(dR))**2
    return weight

def parallel_tempered_metropolis(fac_list, R_list, W, *, n_therm, n_step, n_skip, eps):
    samples = []
    streams = len(R_list)
    target = (streams-1)//2
    print(f"Starting parallel tempered Metropolis with {streams} streams")
    acc_list = [ 0 for s in range(0,streams) ]
    swap_acc_list = [ 0 for s in range(0,streams) ]
    #for i in range(-n_therm, n_step*n_skip):
    (N_coord, N_d) = R_list[0].shape
    for i in tqdm.tqdm(range(-n_therm, n_step*n_skip)):
        # cluster update
        for s in range(streams):
            R_flat = onp.reshape(R_list[s], (N_coord*N_d))
            onp.random.shuffle(R_flat)
            R_list[s] = onp.reshape(R_flat, (N_coord,N_d))
        W_R_list = [ W(R_list[s], fac_list[s]) for s in range(0,streams) ]
        # in-stream update
        for s in range(streams):
            dR = draw_dR(R_list[s].shape, lam=eps, axis=0)
            new_R = R_list[s] + dR
            new_W_R = W(new_R, fac_list[s])
            W_R = W_R_list[s]
            if new_W_R < 1.0 and onp.random.random() < (new_W_R / W_R):
                R_list[s] = new_R # accept
                W_R_list[s] = new_W_R # accept
                acc_list[s] += 1
        # swap (0,1), (2,3), ...
        for s in range(0, streams-1, 2):
            W_R_a = W_R_list[s]
            W_R_b = W_R_list[s+1]
            W_R = W_R_a * W_R_b
            new_R_a = R_list[s+1]
            new_R_b = R_list[s]
            new_W_R_a = W(new_R_a, fac_list[s])
            new_W_R_b = W(new_R_b, fac_list[s+1])
            new_W_R = new_W_R_a * new_W_R_b
            if new_W_R < 1.0 and onp.random.random() < (new_W_R / W_R):
                R_list[s] = new_R_a # accept
                R_list[s+1] = new_R_b # accept
                W_R_list[s] = new_W_R_a # accept
                W_R_list[s+1] = new_W_R_b # accept
                swap_acc_list[s] += 1
                swap_acc_list[s+1] += 1
        # cluster update
        for s in range(streams):
            R_flat = onp.reshape(R_list[s], (N_coord*N_d))
            onp.random.shuffle(R_flat)
            R_list[s] = onp.reshape(R_flat, (N_coord,N_d))
        W_R_list = [ W(R_list[s], fac_list[s]) for s in range(0,streams) ]
        # in-stream update
        for s in range(streams):
            dR = draw_dR(R_list[s].shape, lam=eps, axis=0)
            new_R = R_list[s] + dR
            new_W_R = W(new_R, fac_list[s])
            W_R = W_R_list[s]
            if new_W_R < 1.0 and onp.random.random() < (new_W_R / W_R):
                R_list[s] = new_R # accept
                W_R_list[s] = new_W_R # accept
                acc_list[s] += 1
        # swap (1,2), (3,4), ...
        for s in range(1, streams-1, 2):
            W_R_a = W_R_list[s]
            W_R_b = W_R_list[s+1]
            W_R = W_R_a * W_R_b
            new_R_a = R_list[s+1]
            new_R_b = R_list[s]
            new_W_R_a = W(new_R_a, fac_list[s])
            new_W_R_b = W(new_R_b, fac_list[s+1])
            new_W_R = new_W_R_a * new_W_R_b
            if new_W_R < 1.0 and onp.random.random() < (new_W_R / W_R):
                R_list[s] = new_R_a # accept
                R_list[s+1] = new_R_b # accept
                W_R_list[s] = new_W_R_a # accept
                W_R_list[s+1] = new_W_R_b # accept
                swap_acc_list[s] += 1
                swap_acc_list[s+1] += 1
        # save
        if i >= 0 and (i+1) % n_skip == 0:
            samples.append((R_list[target], W_R_list[target]))
    acc = acc_list[target]
    n_tot = (n_therm+n_skip*n_step)*2
    print(f'In-stream acc frac = {acc} / {n_tot} = {1.0*acc/(n_tot)}')
    acc = swap_acc_list[target]
    print(f'Swap acc frac = {acc} / {n_tot} = {1.0*acc/(n_tot)}')
    return samples

def metropolis(R, W, *, n_therm, n_step, n_skip, eps):
    samples = []
    acc = 0
    (N_coord, N_d) = R.shape
    for i in tqdm.tqdm(range(-n_therm, n_step*n_skip)):
        R_flat = onp.reshape(R, (N_coord*N_d))
        onp.random.shuffle(R_flat)
        R = onp.reshape(R_flat, (N_coord,N_d))
        dR = draw_dR(R.shape, lam=eps, axis=0)
        new_R = R + dR
        W_R = W(R)
        new_W_R = W(new_R)
        if new_W_R < 1.0 and onp.random.random() < (new_W_R / W_R):
            R = new_R # accept
            W_R = new_W_R
            acc += 1
        if i >= 0 and (i+1) % n_skip == 0:
            samples.append((R, W_R))
    print(f'Total acc frac = {acc} / {n_therm+n_skip*n_step} = {1.0*acc/(n_therm+n_skip*n_step)}')
    return samples

### Apply exp(-dtau/2 V_SI) (1 - (dtau/2) V_SD + (dtau^2/8) V_SD^2) to |S>,
### separately returning the updated spin-isospin wavefunction
###     |S'> = (1 - (dtau/2) V_SD + (dtau^2/8) V_SD^2) |S>
### and the scalar factor exp(-dtau/2 V_SI).
@partial(jax.jit, static_argnums=(2,))
def compute_VS_separate(R_prop, S, potential, *, dtau_iMev):
    N_coord = R_prop.shape[1]
    V_SI_prop, V_SD = potential(R_prop)
    VS = batched_apply(V_SD, S)
    VVS = batched_apply(V_SD, VS)
    S_prop = S - (dtau_iMev/2) * VS + (dtau_iMev**2/8) * VVS
    return np.exp(-dtau_iMev/2 * V_SI_prop), S_prop

### Apply exp(-dtau/2 V_SI) (1 - (dtau/2) V_SD + (dtau^2/8) V_SD^2) to |S>.
#@partial(jax.jit, static_argnums=(2,))
def compute_VS(R_deform, S, potential, *, dtau_iMev):
    N_coord = R_deform.shape[1]
    old_S = S
    V_SI, V_SD = potential(R_deform)
    #print("V_SD shape ", V_SD.shape)
    #print("V_SI shape ", V_SI.shape)
    V_SD = V_SD + V_SI
    VS = batched_apply(V_SD, S)
    #print("V_SD ", V_SD[0,0,0,1,0,2,0,0,0,1,0,2,0])
    #print("norm old_S ", inner(old_S,old_S))
    #print("norm VS ", inner(VS,VS))
    #print("old_S.VS/norms  ", inner(VS,old_S) / np.sqrt( inner(VS,VS)*inner(old_S,old_S) ))
    ang = np.arccos(inner(VS, old_S) / np.sqrt( inner(VS,VS)*inner(old_S,old_S) ))
    #print("angle between old and new spin-color vec is ", ang)
    # TODO THIS OUGHT TO FAIL FOR DEUTERON
    #if (np.abs(inner(VS,VS)) > 1e-6).all():
    #    assert ((np.abs(ang) < 1e-6).all() or (np.abs(ang - np.pi) < 1e-6).all() or (np.abs(ang + np.pi) < 1e-6).all())
    VVS = batched_apply(V_SD, VS)
    S = S - (dtau_iMev/2) * VS + (dtau_iMev**2/8) * VVS
    #print("norm old_S ", inner(old_S,old_S))
    #print("norm S_f ", inner(S,S))
    #print("old_S.S_f  ", inner(S,old_S))
    #ang = np.arccos(inner(S, old_S) / np.sqrt( inner(S,S)*inner(old_S,old_S) ))
    #print("angle between old and new spin-color vec is ", ang)
    #V_SI_exp = np.zeros_like(V_SI)
    #V_SI_exp_S = batched_apply(V_SI_exp, S)
    #return V_SI_exp_S
    return S

@partial(jax.jit, static_argnums=(8,9), static_argnames=('deform_f',))
def kinetic_step(R_fwd, R_bwd, R, R_deform, S, u, params_i, _T,
                 f_R_norm, potential, *, deform_f, dtau_iMev, m_Mev):
    """Step forward given two possible proposals and RNG to decide"""
    # transform manifold
    old_S = S
    R_fwd_old = R_fwd
    R_bwd_old = R_bwd
    #R_fwd = phi_shift(R_fwd, lambda0_i)
    #R_bwd = phi_shift(R_bwd, lambda0_i)
    R_fwd = deform_f(R_fwd, *params_i)
    R_bwd = deform_f(R_bwd, *params_i)
    S_fwd = compute_VS(R_fwd, S, potential, dtau_iMev=dtau_iMev)
    S_bwd = compute_VS(R_bwd, S, potential, dtau_iMev=dtau_iMev)
    w_fwd = f_R_norm(to_relative(R_fwd)) * inner(S_T, S_fwd)
    w_bwd = f_R_norm(to_relative(R_bwd)) * inner(S_T, S_bwd)

    # correct kinetic energy
    G_ratio_fwd = np.exp(
        (-np.einsum('...ij,...ij->...', R_fwd-R_deform, R_fwd-R_deform)
         +np.einsum('...ij,...ij->...', R_fwd_old-R, R_fwd_old-R))
        / (2*dtau_iMev*fm_Mev**2/m_Mev))
    G_ratio_bwd = np.exp(
        (-np.einsum('...ij,...ij->...', R_bwd-R_deform, R_bwd-R_deform)
         + np.einsum('...ij,...ij->...', R_bwd_old-R, R_bwd_old-R))
        / (2*dtau_iMev*fm_Mev**2/m_Mev))
    w_fwd = w_fwd * G_ratio_fwd
    w_bwd = w_bwd * G_ratio_bwd

    p_fwd = np.abs(w_fwd) / (np.abs(w_fwd) + np.abs(w_bwd))
    pc_fwd = w_fwd / (w_fwd + w_bwd)
    p_bwd = np.abs(w_bwd) / (np.abs(w_fwd) + np.abs(w_bwd))
    pc_bwd = w_bwd / (w_fwd + w_bwd)
    ind_fwd = u < p_fwd
    # TODO I DONT UNDERSTAND THIS LINE
    ind_fwd_R = np.expand_dims(ind_fwd, axis=(-3,-2,-1))
    R = np.where(ind_fwd_R, R_fwd_old, R_bwd_old)
    R_deform = np.where(ind_fwd_R, R_fwd, R_bwd)
    N_coord = R_deform.shape[1]
    # TODO I REALLY DONT UNDERSTAND THIS LINE
    small_axis_tup = tuple([i for i in range(-N_coord-1,0)])
    ind_fwd_S = np.expand_dims(ind_fwd, axis=small_axis_tup)
    S = np.where(ind_fwd_S, S_fwd, S_bwd)
    # TODO check if new and old S parallel!!
    #ang = inner(S, old_S) / np.sqrt( inner(S,S)*inner(S_T,S_T) )
    #print("angle between old and new spin-color vec is ", ang)
    #jax_print(np.mean(ang), label="angle between old and new spin-color vec is ", level=0)
    #assert (np.abs(ang) < 1e-6).all()
    #exit()
    W = ((w_fwd + w_bwd) / 2)
    W = np.where(ind_fwd, W * pc_fwd / p_fwd, W * pc_bwd / p_bwd)
    return R, R_deform, S, W


### Perform a contour-deformed twobody GFMC evolution.
### We assume the initial dR are drawn from f^2(dR) [as a 3D density], which is
### passed as `f_R_norm`. The deformation `params` are defined for each of the
### `N` walker steps and can be used to compute the deformed particle
### coordinates at each step.  Evolution proceeds according to an importance
### sampling weight and returns reweighting factors `W` that can be used to
### estimate holomorphic integrals
###   <W O(Rtilde)> / <W> ~
###       \int dR_N ... dR_0 <R_N| O e^{-H dtau}|R_{N-1}> ... <R_1|e^{-H dtau}|R_0>
###       -------------------------------------------------------------------------
###       \int dR_N ... dR_0 <R_N| e^{-H dtau}|R_{N-1}> ... <R_1|e^{-H dtau}|R_0>
def gfmc_twobody_deform(
        dR_T, S_T, f_R_norm, params, *, rand_draws, tau_iMev, N, potential, m_Mev,
        deform_f, resampling_freq=None):
    R0 = np.stack((-dR_T/2, dR_T/2), axis=1)
    # sigma, kappa_0, kappa_m, zeta_m, lambda_mn, chi_mn = params
    params0 = tuple(param[0] for param in params)
    R0_deform = deform_f(R0, *params0)
    W = f_R_norm(to_relative(R0_deform)) / f_R_norm(to_relative(R0))
    walkers = (R0, R0_deform, S_T, W)
    dtau_iMev = tau_iMev/N
    history = [walkers]

    for i in tqdm.tqdm(range(N)):
        R, R_deform, S, W = walkers

        # remove previous factors (to be replaced with current factors after evolving)
        W = W / (inner(S_T, S) * f_R_norm(to_relative(R_deform)))

        # exp(-dtau V/2)|R,S>
        S = compute_VS(R_deform, S, potential, dtau_iMev=dtau_iMev)

        # exp(-dtau V/2) exp(-dtau K)|R,S> using fwd/bwd heatbath
        _start = time.time()
        R_fwd, R_bwd = step_G0_symm(onp.array(R), dtau_iMev=dtau_iMev, m_Mev=m_Mev)
        R_fwd, R_bwd = np.array(R_fwd), np.array(R_bwd)
        u = rand_draws[i] # np.array(onp.random.random(size=R_fwd.shape[0]))
        step_params = tuple(param[i+1] for param in params)
        R, R_deform, S, dW = kinetic_step(
            R_fwd, R_bwd, R, R_deform, S, u, step_params, S_T,
            f_R_norm, potential, deform_f=deform_f, dtau_iMev=dtau_iMev, m_Mev=m_Mev)

        # incorporate factors <S_T|S_i> f(R_i) and leftover fwd/bwd factors from
        # the kinetic step
        W = W * dW

        # save config for obs
        history.append((R, R_deform, S, W))

        if resampling_freq is not None and (i+1) % resampling_freq == 0:
            assert len(W.shape) == 1, 'weights must be flat array'
            p = np.abs(W) / np.sum(np.abs(W))
            W = np.mean(np.abs(W), keepdims=True) * W / np.abs(W)
            inds = onp.random.choice(onp.arange(W.shape[0]), size=W.shape[0], p=p)
            R = R[inds]
            R_deform = R_deform[inds]
            S = S[inds]
            W = W[inds]

        walkers = (R, R_deform, S, W)

    return history

#@partial(jax.jit, static_argnums=(8,9), static_argnames=('deform_f',))
def kinetic_step_absolute(R_fwd, R_bwd, R, R_deform, S, u, params_i, S_T,
                 f_R_norm, potential, *, deform_f, dtau_iMev, m_Mev):
    """Step forward given two possible proposals and RNG to decide"""
    # transform manifold
    old_S = S
    R_fwd_old = R_fwd
    R_bwd_old = R_bwd
    #R_fwd = phi_shift(R_fwd, lambda0_i)
    #R_bwd = phi_shift(R_bwd, lambda0_i)
    R_fwd = deform_f(R_fwd, *params_i)
    R_bwd = deform_f(R_bwd, *params_i)
    S_fwd = compute_VS(R_fwd, S, potential, dtau_iMev=dtau_iMev)
    S_bwd = compute_VS(R_bwd, S, potential, dtau_iMev=dtau_iMev)
    #print("norm old_S ", inner(old_S,old_S))
    #print("norm S_f ", inner(S_fwd,S_fwd))
    #print("old_S.S_f  ", inner(S_fwd,old_S))
    #ang_fwd = np.arccos(inner(S_fwd, old_S) / np.sqrt( inner(S_fwd,S_fwd)*inner(old_S,old_S) ))
    #print("fwd angle between old and new spin-color vec is ", ang_fwd)
    #ang_bwd = np.arccos(inner(S_bwd, old_S) / np.sqrt( inner(S_bwd,S_bwd)*inner(old_S,old_S) ))
    #print("bwd angle between old and new spin-color vec is ", ang_bwd)
    w_fwd = f_R_norm(R_fwd) * inner(S_T, S_fwd)
    w_bwd = f_R_norm(R_bwd) * inner(S_T, S_bwd)

    # correct kinetic energy
    # TODO distinct masses
    G_ratio_fwd = np.exp(
        (-np.einsum('...ij,...ij->...', R_fwd-R_deform, R_fwd-R_deform)
         +np.einsum('...ij,...ij->...', R_fwd_old-R, R_fwd_old-R))
        / (2*dtau_iMev*fm_Mev**2/m_Mev))
    G_ratio_bwd = np.exp(
        (-np.einsum('...ij,...ij->...', R_bwd-R_deform, R_bwd-R_deform)
         + np.einsum('...ij,...ij->...', R_bwd_old-R, R_bwd_old-R))
        / (2*dtau_iMev*fm_Mev**2/m_Mev))
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
    S = np.where(ind_fwd_S, S_fwd, S_bwd)
    # TODO check if new and old S parallel!!
    #ang = np.arccos(inner(S, old_S) / np.sqrt( inner(S,S)*inner(old_S,old_S) ))
    #print("angle between old and new spin-color vec is ", ang)
    #assert (np.abs(ang) < 1e-6).all()
    W = ((w_fwd + w_bwd) / 2)
    W = np.where(ind_fwd, W * pc_fwd / p_fwd, W * pc_bwd / p_bwd)
    return R, R_deform, S, W

def gfmc_deform(
        R0, S_T, f_R_norm, params, *, rand_draws, tau_iMev, N, potential, m_Mev,
        deform_f, resampling_freq=None):
    # sigma, kappa_0, kappa_m, zeta_m, lambda_mn, chi_mn = params
    #config.update('jax_disable_jit', True)
    params0 = tuple(param[0] for param in params)
    R0_deform = deform_f(R0, *params0)
    W = f_R_norm(R0_deform) / f_R_norm(R0)
    walkers = (R0, R0_deform, S_T, W)
    dtau_iMev = tau_iMev/N
    history = [walkers]

    #for i in tqdm.tqdm(range(N)):
    for i in range(N):
        _start = time.time()
        print("step ", i)
        R, R_deform, S, W = walkers

        # remove previous factors (to be replaced with current factors after evolving)
        W = W / (inner(S_T, S) * f_R_norm(R_deform))

        # exp(-dtau V/2)|R,S>
        S = compute_VS(R_deform, S, potential, dtau_iMev=dtau_iMev)

        # exp(-dtau V/2) exp(-dtau K)|R,S> using fwd/bwd heatbath
        R_fwd, R_bwd = step_G0_symm(onp.array(R), dtau_iMev=dtau_iMev, m_Mev=m_Mev)
        R_fwd, R_bwd = np.array(R_fwd), np.array(R_bwd)
        u = rand_draws[i] # np.array(onp.random.random(size=R_fwd.shape[0]))
        step_params = tuple(param[i+1] for param in params)
        R, R_deform, S, dW = kinetic_step_absolute(
            R_fwd, R_bwd, R, R_deform, S, u, step_params, S_T,
            f_R_norm, potential, deform_f=deform_f, dtau_iMev=dtau_iMev, m_Mev=m_Mev)

        # incorporate factors <S_T|S_i> f(R_i) and leftover fwd/bwd factors from
        # the kinetic step
        W = W * dW

        # save config for obs
        history.append((R, R_deform, S, W))

        gfmc_Rs = np.array([Rs for Rs,_,_,_, in history])

        if resampling_freq is not None and (i+1) % resampling_freq == 0:
            assert len(W.shape) == 1, 'weights must be flat array'
            p = np.abs(W) / np.sum(np.abs(W))
            W = np.mean(np.abs(W), keepdims=True) * W / np.abs(W)
            inds = onp.random.choice(onp.arange(W.shape[0]), size=W.shape[0], p=p)
            R = R[inds]
            R_deform = R_deform[inds]
            S = S[inds]
            W = W[inds]

        walkers = (R, R_deform, S, W)
        _step_time = time.time()-_start
        print(f'computed step in {_step_time:.1f}s')

    return history

def history_to_onp(history):
    onp_history = []
    for R, R_deform, S, W in history:
        onp_history.append((onp.array(R), onp.array(R_deform), onp.array(S), onp.array(W)))
    return onp_history

def measure_gfmc_obs_deform(
        gfmc_history, estimate_H, f_R_norm, df_R_norm, ddf_R_norm,
        *, Nboot=100, verbose=True, enable_H=True, enable_rhoij=True):
    Hs = []
    Ks = []
    Vs = []
    rhoijs = []
    _,R0_deform,S_T,_ = gfmc_history[0]
    for Rs, Rs_deform, Ss, Ws in tqdm.tqdm(gfmc_history):
        if enable_H:
            dRs = to_relative(Rs_deform)
            f = onp.array(f_R_norm(dRs))
            df = onp.array(df_R_norm(dRs))
            ddf = onp.array(ddf_R_norm(dRs))
            res = estimate_H(dRs, Ss, Ws, S_T, f, df, ddf, m_Mev=mp_Mev, Nboot=Nboot, verbose=verbose)
            if verbose: tqdm.tqdm.write(f'<H> = {res["H"]}')
            Hs.append(res['H'])
            Ks.append(res['K'])
            Vs.append(res['V'])
        if enable_rhoij:
            q_Mev = np.array([600.0, 0, 0])
            A = Rs.shape[1]
            rhoij = []
            for i in range(A):
                for j in range(A):
                    rhoij_vals = measure_eucl_density_response(R0_deform, Rs_deform, q_Mev=q_Mev, i=i, j=j)
                    rhoij.append(al.bootstrap(onp.array(rhoij_vals), onp.array(Ws), Nboot=Nboot, f=rw_mean))
            rhoij = np.transpose(np.array(rhoij).reshape((A, A, 2)), (2, 0, 1))
            rhoijs.append(rhoij)
    res = {}
    if enable_H:
        res['H'] = onp.array(Hs)
        res['K'] = onp.array(Ks)
        res['V'] = onp.array(Vs)
    if enable_rhoij:
        res['rhoij'] = onp.array(rhoijs)
    return res

def measure_gfmc_loss(
        params, gfmc_Rs, gfmc_Ws, S_T, AVcoeffs, f_R_norm, df_R_norm, ddf_R_norm, loss_ts,
        *, eval_local_loss, deform_f, alpha, beta, tau_iMev, N, m_Mev):

    potential = make_explicit_pairwise_potential(AVcoeffs)
    #estimate_H = make_twobody_sample_mean_H(AVcoeffs)
    dtau_iMev = tau_iMev/N

    # for i in range(1,len(gfmc_history)):
    def body(carry, array_elems):
        S, W_prod = carry
        jax_print(np.min(np.abs(inner(S, S))), label="start loop", level=3)
        jax_print(np.min(np.abs(inner(S_T, S))), level=3)
        Rs_prev, Rs_next = array_elems[:2]
        params_prev = array_elems[2::2]
        params_next = array_elems[3::2]
        Rs_deform_prev = deform_f(Rs_prev, *params_prev)
        Rs_deform_next = deform_f(Rs_next, *params_next)
        W_prod = W_prod / (inner(S_T, S) * f_R_norm(to_relative(Rs_deform_prev)))
        jax_print(W_prod, label='W_prod divided', level=3)

        # DEBUG: check pre-deform VSI
        V_SI, _ = potential(Rs_prev)
        V_ind = (slice(0,None),) + (0,)*NS*NI*N_coord
        V_half_ind = (slice(0,None),) + (0,)*NI*N_coord
        # TODO
        V_slice = (slice(0,None),) + (slice(0,1,1),)*NI*N_coord + (0,)*NS*N_coord
        #V_slice = (slice(0,None),) + (0,)*NI*N_coord + (0,)*NS*N_coord
        #V_slice = (slice(0,None),) + (0,)*NS*NI*N_coord
        V_SI = V_SI[V_slice]
        jax_print(np.sort(np.abs(V_SI[V_half_ind]))[-10:], label='VSI', level=3)

        # exp(-dtau V/2)|R,S>
        V_SI, V_SD = potential(Rs_deform_prev)
        VS = batched_apply(V_SD, S)
        VVS = batched_apply(V_SD, VS)
        S = S - (dtau_iMev/2) * VS + (dtau_iMev**2/8) * VVS
        jax_print(np.min(np.abs(inner(S, S))), label="After VSD", level=3)
        jax_print(np.min(np.abs(inner(S_T, S))), level=3)
        V_SI = V_SI[V_slice]
        jax_print(np.sort(np.abs(V_SI[V_half_ind]))[-10:], label='VSI', level=3)
        i = np.argmax(np.abs(V_SI[V_half_ind]))
        jax_print((
            norm_3vec_sq(Rs_prev[i,0]-Rs_prev[i,1]),
            norm_3vec_sq(Rs_deform_prev[i,0]-Rs_deform_prev[i,1])), label='R max', level=3)
        S = np.exp(-dtau_iMev/2 * V_SI) * S
        jax_print(np.min(np.abs(inner(S, S))), label="After VSI", level=3)
        jax_print(np.min(np.abs(inner(S_T, S))), level=3)

        # exp(-dtau V/2) exp(-dtau K)|R,S>, accounting for fwd/bwd heatbath
        Delta_Rs = Rs_next - Rs_prev
        Delta_Rs_deform = Rs_deform_next - Rs_deform_prev
        G_ratio_fwd = np.exp(
            (-np.einsum('...ij,...ij->...', Delta_Rs_deform, Delta_Rs_deform)
             +np.einsum('...ij,...ij->...', Delta_Rs, Delta_Rs))
            / (2*dtau_iMev*fm_Mev**2/m_Mev))
        Rs_next_bwd = Rs_prev - Delta_Rs
        Rs_deform_next_bwd = deform_f(Rs_next_bwd, *params_next)
        Delta_Rs_deform_bwd = Rs_deform_next_bwd - Rs_deform_prev
        G_ratio_bwd = np.exp(
            (-np.einsum('...ij,...ij->...', Delta_Rs_deform_bwd, Delta_Rs_deform_bwd)
             +np.einsum('...ij,...ij->...', Delta_Rs, Delta_Rs))
            / (2*dtau_iMev*fm_Mev**2/m_Mev))


        V_SI, V_SD = potential(Rs_deform_next)
        VS = batched_apply(V_SD, S)
        VVS = batched_apply(V_SD, VS)
        S_fwd = S - (dtau_iMev/2) * VS + (dtau_iMev**2/8) * VVS
        jax_print(np.min(np.abs(inner(S_fwd, S_fwd))), label="After VSD 2", level=3)
        jax_print(np.min(np.abs(inner(S_T, S_fwd))), level=3)
        V_SI = V_SI[V_slice]
        jax_print(np.sort(np.abs(V_SI[V_half_ind]))[-10:], label='VSI 2', level=3)
        i = np.argmax(np.abs(V_SI[V_half_ind]))
        jax_print((
            norm_3vec_sq(Rs_next[i,0]-Rs_next[i,1]),
            norm_3vec_sq(Rs_deform_next[i,0]-Rs_deform_next[i,1])), label='R max', level=3)
        S_fwd = np.exp(-dtau_iMev/2 * V_SI) * S_fwd
        jax_print(np.min(np.abs(inner(S_fwd, S_fwd))), label="After VSI 2", level=3)
        jax_print(np.min(np.abs(inner(S_T, S_fwd))), level=3)

        V_SI, V_SD = potential(Rs_deform_next_bwd)
        VS = batched_apply(V_SD, S)
        VVS = batched_apply(V_SD, VS)
        S_bwd = S - (dtau_iMev/2) * VS + (dtau_iMev**2/8) * VVS
        jax_print(np.min(np.abs(inner(S_bwd, S_bwd))), label="After VSD 2", level=3)
        jax_print(np.min(np.abs(inner(S_T, S_bwd))), level=3)
        V_SI = V_SI[V_slice]
        jax_print(np.sort(np.abs(V_SI[V_half_ind]))[-10:], label='VSI 2', level=3)
        i = np.argmax(np.abs(V_SI[V_half_ind]))
        jax_print((
            norm_3vec_sq(Rs_next[i,0]-Rs_next[i,1]),
            norm_3vec_sq(Rs_deform_next[i,0]-Rs_deform_next[i,1])), label='R max', level=3)
        S_bwd = np.exp(-dtau_iMev/2 * V_SI) * S_bwd
        jax_print(np.min(np.abs(inner(S_bwd, S_bwd))), label="After VSI 2", level=3)
        jax_print(np.min(np.abs(inner(S_T, S_bwd))), level=3)

        w_fwd = inner(S_T, S_fwd) * f_R_norm(to_relative(Rs_deform_next)) * G_ratio_fwd
        w_bwd = inner(S_T, S_bwd) * f_R_norm(to_relative(Rs_deform_next_bwd)) * G_ratio_bwd
        W_prod = W_prod * (w_fwd / np.abs(w_fwd)) * (np.abs(w_fwd) + np.abs(w_bwd)) / 2

        S = S_fwd

        # print('total err S', np.sum((S - test_S_next)**2))
        # print('err S', np.sum((S - test_S_next)**2, axis=(1,2,3,4)))
        # print('max err S', np.max(np.sum((S - test_S_next)**2, axis=(1,2,3,4))))
        # assert np.allclose(S, test_S_next)

        jax_print(np.min(np.abs(inner(S, S))), label="End loop", level=3)
        jax_print(np.min(np.abs(inner(S_T, S))), level=3)
        jax_print(W_prod, label='W_prod', level=3)

        return (S, W_prod), (S, W_prod)

    S = S_T
    R0_deform = deform_f(gfmc_Rs[0], *[param[0] for param in params])
    RN_deform = deform_f(gfmc_Rs[-1], *[param[-1] for param in params])
    dR0_deform = to_relative(R0_deform)
    dRN_deform = to_relative(RN_deform)
    W_prod = f_R_norm(dR0_deform) / f_R_norm(to_relative(gfmc_Rs[0]))
    jax_print(W_prod, label='W_prod_0', level=3)

    # lists of previous / current coords and params
    scan_input = (gfmc_Rs[:-1], gfmc_Rs[1:]) + sum(
        ((param[:-1], param[1:]) for param in params), ())
    _, all_Ss_Ws = jax.lax.scan(body, (S,W_prod), scan_input)

    eval_local_loss = partial(eval_local_loss, R0_deform=R0_deform, RN_deform=RN_deform, S_T=S_T)
    loss_pieces = jax.lax.map(eval_local_loss, (all_Ss_Ws[0][loss_ts], all_Ss_Ws[1][loss_ts]))
    return np.sum(loss_pieces)

def make_local_loss_H(f_R_norm, df_R_norm, ddf_R_norm, *, AVcoeffs, alpha, beta, m_Mev):
    @jax.jit
    def eval_local_loss_H(X_i, *, R0_deform, RN_deform, S_T):
        dRN_deform = to_relative(RN_deform)
        S, W = X_i
        # jax_print(np.abs(W),label='Wc', level=3)
        # jax_print(gfmc_Ws,label='gfmc_Ws', level=3)
        # jax_print(np.allclose(np.abs(W), gfmc_Ws[-1]), label='Weights check', level=0)
        # jax_print(np.min(np.abs(inner(S_T, S))), level=3)
        f = np.array(f_R_norm(dRN_deform))
        df = np.array(df_R_norm(dRN_deform))
        ddf = np.array(ddf_R_norm(dRN_deform))
        K_Mev = np.array(compute_K(dRN_deform, f, df, ddf, m_Mev=m_Mev))
        Os = {
            name: np.array(
                AVcoeffs[name](dRN_deform) * compute_O(two_body_ops[name](dRN_deform), S, S_T))
            for name in AVcoeffs
        }
        V_Mev = sum(Os.values())
        H_Mev = K_Mev + V_Mev
        est_K = np.real(np.mean(W * K_Mev)/np.mean(W))
        est_V = np.real(np.mean(W * V_Mev)/np.mean(W))
        est_H = np.real(np.mean(W * H_Mev)/np.mean(W))

        num_loss = np.mean(np.real(W * H_Mev)**2 + np.imag(W * H_Mev)**2)
        #num_loss = np.mean(np.real(W * K_Mev)**2 + np.imag(W * K_Mev)**2) + np.mean(np.real(W * V_Mev)**2 + np.imag(W * V_Mev)**2)
        den_loss = np.mean(np.real(W)**2 + np.imag(W)**2)
        jax_print(est_K, label='K', level=1)
        jax_print(est_V, label='V', level=1)
        jax_print(est_H, label='H', level=1)
        return alpha * np.log(num_loss) + beta * np.log(den_loss)
    return eval_local_loss_H

def make_local_loss_E(*, q_Mev, i, j, alpha, beta):
    @jax.jit
    def eval_local_loss_E(X_i, *, R0_deform, RN_deform, S_T):
        S, W = X_i
        num = measure_eucl_density_response(R0_deform, RN_deform, q_Mev=q_Mev, i=i, j=j)
        num_loss = np.mean(np.real(num)**2 + np.imag(num)**2)
        den_loss = np.mean(np.real(W)**2 + np.imag(W)**2)
        return alpha * np.log(num_loss) + beta * np.log(den_loss)
    return eval_local_loss_E


def train_gfmc_deform(
        Rs_metropolis, S_av4p_metropolis, f_R_norm, df_R_norm, ddf_R_norm, params,
        *, tau_iMev, N, AVcoeffs, deform_f, eval_local_loss,
        m_Mev, loss_t0=None, loss_tstep=1, n_iter=10000,
        mlog10step=2, patience=250, window=50, alpha=0, beta=1):

    step_size = 10**(-1*(mlog10step))
    factor = 0.1
    max_reduces = 2

    if loss_t0 is None:
        loss_t0 = N-1
    loss_ts = np.arange(loss_t0, N, loss_tstep)
    print(f'Evaluating loss on timesteps {loss_ts}')

    thisii = 0
    current = 0.0
    best = 0.0
    bad_epochs = 0
    reduces = 0

    def plateau_learn_rate(ii):
        nonlocal best
        nonlocal current
        nonlocal bad_epochs
        nonlocal reduces
        nonlocal thisii
        if ii <= window:
           print(f'starting epoch')
           best = current
        elif current < best:
            print(f'good epoch')
            bad_epochs = 0
            best = current
        else:
            if ii > thisii:
                bad_epochs += 1
                print(f'bad epoch #{bad_epochs}, learning reduced {reduces} times')
                if bad_epochs > patience:
                    reduces += 1
                    print(f'lost patience! learning reduced {reduces} times')
                    bad_epochs = 0
            thisii = ii
        return np.power(factor,reduces)*step_size

    potential = make_explicit_pairwise_potential(AVcoeffs)

    l = 10**(10)

    ### TEMP: compute jaxpr to debug long JIT times
    # rand_draws = np.array(onp.random.random(size=(N, Rs_metropolis.shape[0])))
    # gfmc_deform = gfmc_twobody_deform(
    #     Rs_metropolis, S_av4p_metropolis, f_R_norm, params,
    #     rand_draws=rand_draws, tau_iMev=tau_iMev, N=N, potential=potential, m_Mev=m_Mev,
    #     resampling_freq=None)
    # gfmc_Rs = np.array([Rs for Rs,_,_,_, in gfmc_deform])
    # gfmc_Ws = np.abs(np.array([Ws for _,_,_,Ws, in gfmc_deform]))
    # gfmc_Ss = np.abs(np.array([Ss for _,_,Ss,_, in gfmc_deform]))
    # _,_,S_T,_ = gfmc_deform[0]
    # jaxpr = jax.make_jaxpr(measure_gfmc_loss, static_argnums=(4,5,6,7))(
    #     params, gfmc_Rs, gfmc_Ws, S_T, AVcoeffs, f_R_norm, df_R_norm, ddf_R_norm,
    #     alpha, beta, tau_iMev=tau_iMev, N=N, m_Mev=m_Mev)
    # print('JIT info (fewer is better for JIT time):')
    # print('# consts', len(jaxpr.jaxpr.constvars))
    # print('# in vars', len(jaxpr.jaxpr.invars))
    # print('# out vars', len(jaxpr.jaxpr.outvars))
    # print('# eqns', len(jaxpr.jaxpr.eqns))
    # print('== JAXPR ==') # can be really verbose
    # print(jaxpr)
    # print('== END JAXPR ==')

    #loss_and_grad = jax.value_and_grad(measure_gfmc_log_abs_weight, argnums=0)
    #loss_and_grad = jax.jit(loss_and_grad, static_argnums=(4,5))
    loss_and_grad = jax.value_and_grad(measure_gfmc_loss, argnums=0)
    # TODO for debugging
    #config.update('jax_disable_jit', True)
    loss_and_grad = jax.jit(
        loss_and_grad, static_argnums=(4,5,6,7), static_argnames=('deform_f','eval_local_loss'))
    opt_init, opt_update, opt_get_params = jax.experimental.optimizers.adam(plateau_learn_rate)
    opt_state = opt_init(params)
    _jit_time = None

    history = {
        'loss': [],
        'test_loss': [],
        'lambda_0': [],
        'lambda_m': [],
        'lambda_mn': [],
        'chi_m': [],
        'chi_mn': [],
        'grad': []
    }
    param_history = []

    AVcoeffs = hashabledict(AVcoeffs)

    for i in tqdm.tqdm(range(n_iter)):

        _gen_start = time.time()

        rand_draws = np.array(onp.random.random(size=(N, Rs_metropolis.shape[0])))
        gfmc_deform = gfmc_twobody_deform(
            Rs_metropolis, S_av4p_metropolis, f_R_norm, params,
            rand_draws=rand_draws, tau_iMev=tau_iMev, N=N, potential=potential, m_Mev=m_Mev,
            deform_f=deform_f, resampling_freq=None)
        gfmc_Rs = np.array([Rs for Rs,_,_,_, in gfmc_deform])
        gfmc_Ws = np.abs(np.array([Ws for _,_,_,Ws, in gfmc_deform]))
        gfmc_Ss = np.abs(np.array([Ss for _,_,Ss,_, in gfmc_deform]))

        _,_,S_T,_ = gfmc_deform[0]

        _gen_time = time.time()-_gen_start
        tqdm.tqdm.write(f'walkers generated in {_gen_time:.1f}s')

        print("params = ", params)

        # JIT happens on first call, print out some timing info
        if _jit_time is None:
            tqdm.tqdm.write('JIT compiling loss_and_grad (this may take a while) ...')
            _jit_start = time.time()
            # jaxpr = jax.make_jaxpr(loss_and_grad, static_argnums=(4,5))(
            #   params, gfmc_Rs, S_T, AVcoeffs, tau_iMev=tau_iMev, N=N, m_Mev=m_Mev)
            # jaxpr = jax.make_jaxpr(loss_and_grad, static_argnums=(4,5,6,7))(
            #     params, gfmc_Rs, gfmc_Ws, S_T, AVcoeffs, f_R_norm, df_R_norm, ddf_R_norm, alpha, beta, tau_iMev=tau_iMev, N=N, m_Mev=m_Mev)
            # print('JIT info (fewer is better for JIT time):')
            # print('# consts', len(jaxpr.jaxpr.constvars))
            # print('# in vars', len(jaxpr.jaxpr.invars))
            # print('# out vars', len(jaxpr.jaxpr.outvars))
            # print('# eqns', len(jaxpr.jaxpr.eqns))
            # import sys
            # sys.exit()

        # l, g = loss_and_grad(params, gfmc_Rs, S_T, AVcoeffs, tau_iMev=tau_iMev, N=N, m_Mev=m_Mev)
        #l, g = loss_and_grad(params, gfmc_Rs, gfmc_Ws, S_T, potential, f_R_norm, tau_iMev=tau_iMev, N=N, m_Mev=m_Mev)
        _lg_start = time.time()

        l, g = loss_and_grad(
            params, gfmc_Rs, gfmc_Ws, S_T, AVcoeffs, f_R_norm, df_R_norm, ddf_R_norm,
            loss_ts, eval_local_loss=eval_local_loss, deform_f=deform_f,
            alpha=alpha, beta=beta, tau_iMev=tau_iMev, N=N, m_Mev=m_Mev)

        if _jit_time is None:
            _jit_time = time.time()-_jit_start
            tqdm.tqdm.write(f'JIT done in {_jit_time:.2f}s')

        l.block_until_ready() # ensure computation done before reporting time
        _lg_time = time.time()-_lg_start
        tqdm.tqdm.write(f'loss and grad in {_lg_time:.2f}s')

        # check for BAD things
        if not np.isfinite(l):
            tqdm.tqdm.write(f'Infinite/NaN loss = {l}')
            tqdm.tqdm.write('Logging out bad params')
            crash = {
                'param': opt_get_params(opt_state),
                'loss': history['loss'],
                'grad': history['grad']
            }
            tag = time.time()
            with open(f'crash_dumps/crash_dump_{tag}.pkl', 'wb') as f:
                pickle.dump(crash, f)
                np.save(f'crash_dumps/crash_gfmc_Rs_{tag}.npy', gfmc_Rs)
                np.save(f'crash_dumps/crash_gfmc_Ws_{tag}.npy', gfmc_Ws)
                np.save(f'crash_dumps/crash_Rs_{tag}.npy', Rs_metropolis)
                np.save(f'crash_dumps/crash_Ss_{tag}.npy', S_av4p_metropolis)
            # tqdm.tqdm.write('Goodbye')
            # import sys
            # sys.exit()

            # HACK: make sure indexing is still consistent with dummy values
            history['loss'].append(float('nan'))
            history['grad'].append(float('nan'))

        else:
            current += l
            if i >= window and np.isfinite(history['loss'][i-window]):
                current -= history['loss'][i-window]

            history['loss'].append(l)
            history['grad'].append(g)
            param_history.append(params)
            params_old = params
            opt_state = opt_update(i, g, opt_state)
            params = opt_get_params(opt_state)

        # tqdm.tqdm.write(f'params = {params}') # way too noisy
        tqdm.tqdm.write(f'N = {N} iteration {i+1}, loss = {l}')
        tqdm.tqdm.write('')
        if reduces >= max_reduces:
            print("plateau learn rate converged")
            break

    # set final state to be the lowest test loss
    #best_loss = l
    #for i in range(len(param_history)):
    #    if history['loss'][i] < best_loss:
    #        print("more optimal manifold at step ", i)
    #        best_loss = history['loss'][i]
    #        lambda0 = param_history[i]
    best_params = param_history[-1]
    return best_params, history
