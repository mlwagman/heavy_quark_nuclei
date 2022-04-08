# Wagman Feb 2022

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
import torch.optim as optim
import time
import h5py
import tqdm.auto as tqdm
import os
import argparse

from hydrogen import *

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})

N_coord = 2
VB = 1

def total_Psi_nlm(Rs, n, l, m, Z_n):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    for i in range(N_walkers):
        # convert to spherical
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        r_n = torch.sqrt(x**2 + y**2 + z**2)
        t_n = torch.atan2(torch.sqrt(x**2 + y**2), z) 
        p_n = torch.atan2(y, x)
        # evaluate wavefunction
        psi_fn = psi_no_v(N_coord, n, l, m, Z, r, t, p)
        Psi_nlm_s[i] = psi_fn(Z_n, r_n, t_n, p_n)
    return Psi_nlm_s

def nabla_total_Psi_nlm(Rs, n, l, m, Z_n):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    nabla_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    for i in range(N_walkers):
        # convert to spherical
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        r_n = torch.sqrt(x**2 + y**2 + z**2)
        t_n = torch.atan2(torch.sqrt(x**2 + y**2), z) 
        p_n = torch.atan2(y, x)
        # evaluate wavefunction
        nabla_psi_fn = nabla_psi_no_v(N_coord, n, l, m, Z, r, t, p)
        nabla_Psi_nlm_s[i] = nabla_psi_fn(Z_n, r_n, t_n, p_n)
    return nabla_Psi_nlm_s

def potential_total_Psi_nlm(Rs, n, l, m, Z_n):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    V_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    wvfn = total_Psi_nlm(Rs, n, l, m, Z_n)
    for i in range(N_walkers):
        # convert to spherical
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

def K_Psi_nlm(Rs, n, l, m, Z):
    K_psi = -1/2*nabla_total_Psi_nlm(Rs, n, l, m, Z)
    return K_psi

def V_Psi_nlm(Rs, n, l, m, Z):
    V_psi = potential_total_Psi_nlm(Rs, n, l, m, Z)
    return V_psi

def hammy_Psi_nlm(Rs, n, l, m, Z):
    K_psi = K_Psi_nlm(Rs, n, l, m, Z)
    V_psi = V_Psi_nlm(Rs, n, l, m, Z)
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
    #for i in tqdm.tqdm(range(-n_therm, N_walkers*n_skip)):
    for i in range(-n_therm, N_walkers*n_skip):
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


def variational_wvfn(params, Rs):
    return np.exp(-Rs**2/params[0]**2)


class wvfn(nn.Module):
    # Psi(R) = sum_{n,l,m} c_{n,l,m} psi(n, l, m, R)
    def __init__(self):
        super(wvfn, self).__init__()
        # register Bohr radius a and c_{n,l,m} as pytorch paramters
        self.a = nn.Parameter(1*torch.ones(1, dtype=torch.double))
        self.c100_re = nn.Parameter(torch.ones(1, dtype=torch.double))
        self.c100_im = nn.Parameter(0*torch.ones(1, dtype=torch.double))
        self.c200_re = nn.Parameter(torch.ones(1, dtype=torch.double))
        self.c200_im = nn.Parameter(0*torch.ones(1, dtype=torch.double))
    def psi(self, Rs):
        a_n = self.a[0]
        psi = (self.c100_re+1j*self.c100_im) * total_Psi_nlm(Rs, 1, 0, 0, 1/a_n) + (self.c200_re+1j*self.c200_im) * total_Psi_nlm(Rs, 2, 0, 0, 1/a_n)
        return psi
    def psi2(self, Rs):
        a_n = self.a[0]
        psi = (self.c100_re+1j*self.c100_im) * total_Psi_nlm(Rs, 1, 0, 0, 1/a_n) + (self.c200_re+1j*self.c200_im) * total_Psi_nlm(Rs, 2, 0, 0, 1/a_n)
        psistar = (self.c100_re-1j*self.c100_im) * torch.conj(total_Psi_nlm(Rs, 1, 0, 0, 1/a_n)) + (self.c200_re-1j*self.c200_im) * torch.conj(total_Psi_nlm(Rs, 2, 0, 0, 1/a_n))
        return psistar*psi
    def hammy(self, Rs):
        a_n = self.a[0]
        H_psi = (self.c100_re+1j*self.c100_im) * hammy_Psi_nlm(Rs, 1, 0, 0, 1/a_n) + (self.c200_re+1j*self.c200_im) * hammy_Psi_nlm(Rs, 2, 0, 0, 1/a_n)
        psistar = (self.c100_re-1j*self.c100_im) * torch.conj(total_Psi_nlm(Rs, 1, 0, 0, 1/a_n)) + (self.c200_re-1j*self.c200_im) * torch.conj(total_Psi_nlm(Rs, 2, 0, 0, 1/a_n))
        return psistar*H_psi
    def forward(self, Rs):
        a_n =  self.a[0]
        H_psi = (self.c100_re+1j*self.c100_im) * hammy_Psi_nlm(Rs, 1, 0, 0, 1/a_n) + (self.c200_re+1j*self.c200_im) * hammy_Psi_nlm(Rs, 2, 0, 0, 1/a_n)
        psistar = (self.c100_re-1j*self.c100_im) * torch.conj(total_Psi_nlm(Rs, 1, 0, 0, 1/a_n)) + (self.c200_re-1j*self.c200_im) * torch.conj(total_Psi_nlm(Rs, 2, 0, 0, 1/a_n))
        hammy = psistar*H_psi
        psi = (self.c100_re+1j*self.c100_im) * total_Psi_nlm(Rs, 1, 0, 0, 1/a_n) + (self.c200_re+1j*self.c200_im) * total_Psi_nlm(Rs, 2, 0, 0, 1/a_n)
        psi2 = psistar*psi
        return torch.mean( hammy ) / torch.mean( psi2 )

def loss_function(wvfn, Rs):
    # <psi|H|psi> / <psi|psi>
    loss = wvfn(Rs)
    return loss 
    
def train_variational_wvfn(wvfn, Rs):
    optimizer.zero_grad()
    # train net
    train_time = time.time()
    for n in tqdm.tqdm(range(N_train)):
        step_time = time.time()
        Rs, psi2s = metropolis_coordinate_ensemble(wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=10, eps=1.0)
        loss = loss_function(wvfn, Rs)
        loss.backward()
        optimizer.step()
        tqdm.tqdm.write(f"\nTraining step {n}")
        tqdm.tqdm.write(f"loss function curently {loss}")
        for name, param in wvfn.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    print(f"completed {N_train} steps of training in {time.time() - train_time} sec")
    print(f"final loss function {loss}")
    return wvfn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_particles', default=2, type=int)
    parser.add_argument('--N_walkers', default=100, type=int)
    parser.add_argument('--N_train', default=200, type=int)
    parser.add_argument('--log10_learn_rate', default=2, type=int)
    globals().update(vars(parser.parse_args()))

    # test chi
    #Hammy = hydrogen.laPlaceSpher(hydrogen.Chi(1, hydrogen.N_coord, 1, 0, 0, 1, hydrogen.r, hydrogen.t, hydrogen.p, hydrogen.v, 1),hydrogen.r[0],hydrogen.t[0],hydrogen.p[0]).subs(hydrogen.r[1],0)+(hydrogen.Potential(hydrogen.rr,hydrogen.B,hydrogen.N_coord)*hydrogen.Chi(1, hydrogen.N_coord, 1, 0, 0, 1, hydrogen.r, hydrogen.t, hydrogen.p, hydrogen.v, 1)).subs(hydrogen.r[1],0)

    #print(hydrogen.simplify(Hammy.subs(hydrogen.v[1],1)))

    B_n=1
    #a=-2/B
    a_n=2/B_n
    def psi0(Rs):
        return total_Psi_nlm(Rs, 1, 0, 0, 1/a_n)
        #return total_Psi_nlm(Rs, 2, 1, 1, 1/a_n)

    print("\nn l m = 1 0 0 without using relative coordinates")
    psi0_wvfn = Chi(1, nCoord, 1, 0, 0, 1/a, r, t, p, v, 1)
    #wvfn = Chi(1, nCoord, 2, 1, 1, 1/a, r, t, p, v, 1)
    V = (Potential(rr,B,nCoord)*psi0_wvfn)
    #print(f'V = {V}')
    K1 = -1/2*laPlaceSpher(psi0_wvfn,r[0],t[0],p[0])
    #print(f'K1 = {K1}')
    K2 = -1/2*laPlaceSpher(psi0_wvfn,r[1],t[1],p[1])
    #print(f'K2 = {K2}')
    Hammy = K1 + K2 + V
    #print(f'H = {Hammy}')
    #print(f'presimp = {(Hammy / wvfn).subs(v[1],1).subs(a,-2/B).subs(r[0],1).subs(r[1],1).subs(t[0],1).subs(t[1],1).subs(p[0],1).subs(p[1],1).subs(B,1)}')
    #Enl = simplify((Hammy / wvfn).subs(v[1],1).subs(a,-2/B).subs(r[0],1).subs(r[1],1).subs(t[0],pi/2).subs(t[1],pi/2).subs(p[0],1).subs(p[1],-1).subs(B,1))
    Enl = simplify((Hammy / psi0_wvfn).subs(v[1],1).subs(a,2/B).subs(r[0],1).subs(r[1],1).subs(t[0],np.pi/2).subs(t[1],np.pi/2).subs(p[0],1.0).subs(p[1],-1.0).subs(B,1))
    print(f"E(n={1}, l={0}) = {Enl}")

    Rs, psi2s = metropolis_coordinate_ensemble(psi0, n_therm=500, N_walkers=N_walkers, n_skip=10, eps=1.0)
    #print(f'Rs = {Rs}')

    print(Rs.shape)
    x = Rs[0,:,0]
    y = Rs[0,:,1]
    z = Rs[0,:,2]
    r_n = torch.sqrt(x**2 + y**2 + z**2)
    t_n = torch.atan2(torch.sqrt(x**2 + y**2), z) 
    p_n = torch.atan2(y, x)
    psistar = np.conjugate(psi0_wvfn)
    print(f"V = ", (psistar*V).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print(f"K1 = ", (psistar*K1).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print(f"K2 = ", (psistar*K2).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print(f"H = ", (psistar*Hammy).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print(f"|psi|^2 = ", (psistar*psi0_wvfn).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print(f"<psi|H|psi>/|psi|^2 = ", ((psistar*Hammy)/(psistar*psi0_wvfn)).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print(f"<psi|V|psi>/|psi|^2 = ", ((psistar*V)/(psistar*psi0_wvfn)).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print(f"<psi|K|psi>/|psi|^2 = ", ((psistar*(K1+K2))/(psistar*psi0_wvfn)).subs(v[1],1).subs(a,2/B).subs(r[0],r_n[0]).subs(r[1],r_n[1]).subs(t[0],t_n[0]).subs(t[1],t_n[1]).subs(p[0],p_n[0]).subs(p[1],p_n[1]).subs(B,1))
    print("\n")

    print(f"psi0 = {psi0(Rs)[0]}")
    print(f"|psi0|^2 = {np.conjugate(psi0(Rs)[0])*psi0(Rs)[0]}")
    hammy_ME = np.conjugate(psi0(Rs))*hammy_Psi_nlm(Rs, 1, 0, 0, 1/a_n)
    #hammy_ME = np.conjugate(psi0(Rs)[0])*hammy_Psi_nlm(Rs, 2, 1, 1, 1/a_n)
    print(f"|psi|^2 = ", psi2s[0])
    print(f"<psi|H|psi>/|psi|^2 = {hammy_ME[0]/psi2s[0]}")
    V_ME = np.conjugate(psi0(Rs))*V_Psi_nlm(Rs, 1, 0, 0, 1/a_n)
    #V_ME = np.conjugate(psi0(Rs)[0])*V_Psi_nlm(Rs, 2, 1, 1, 1/a_n)
    print(f"<psi|V|psi>/|psi|^2 = {V_ME[0]/psi2s[0]}")
    K_ME = np.conjugate(psi0(Rs))*K_Psi_nlm(Rs, 1, 0, 0, 1/a_n)
    #K_ME = np.conjugate(psi0(Rs)[0])*K_Psi_nlm(Rs, 2, 1, 1, 1/a_n)
    print(f"<psi|K|psi>/|psi|^2 = {K_ME[0]/psi2s[0]}")
    
    print(f'|psi|^2 = {psi2s}')

    E0 = hammy_ME / psi2s

    print(f'\nEvery element should be E0=-1/4, {E0} \n')
    print(f'<psi|H|psi>/<psi|psi> = {torch.mean(hammy_ME)/torch.mean(psi2s)} +/- {torch.abs(torch.mean(hammy_ME))/torch.mean(psi2s)*torch.sqrt(torch.var(hammy_ME)/torch.mean(hammy_ME)**2+torch.var(psi2s)/torch.mean(psi2s)**2)/np.sqrt(N_walkers)} = -1/4?')
    print(f'<psi|V|psi>/<psi|psi> = {torch.mean(V_ME)/torch.mean(psi2s)} +/- {torch.abs(torch.mean(V_ME))/torch.mean(psi2s)*torch.sqrt(torch.var(V_ME)/torch.mean(V_ME)+torch.var(psi2s)/torch.mean(psi2s)**2)/np.sqrt(N_walkers)}')
    print(f'<psi|K|psi>/<psi|psi> = {torch.mean(K_ME)/torch.mean(psi2s)} +/- {torch.abs(torch.mean(K_ME))/torch.mean(psi2s)*torch.sqrt(torch.var(K_ME)/torch.mean(K_ME)+torch.var(psi2s)/torch.mean(psi2s)**2)/np.sqrt(N_walkers)}')

    print("\n")

    # initialize wvfn
    wvfn = wvfn()
    for name, param in wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    #print("now with full wvfn\n")
    #print(f"psi0 = {wvfn.psi(Rs)[0]}")
    #print(f"|psi0|^2 = {wvfn.psi2(Rs)[0]}")
    #hammy_ME = wvfn.hammy(Rs) / wvfn.psi2(Rs)
    #hammy_ME = np.conjugate(psi0(Rs)[0])*hammy_Psi_nlm(Rs, 2, 1, 1, 1/a_n)
    #print(f"|psi|^2 = ", psi2s[0])
    #print(f'\nEvery element should be E0=-1/4, {hammy_ME} \n')
    #hammy_ME_m = wvfn(Rs)
    #print(f"<psi|H|psi>/|psi|^2 = {hammy_ME_m}")

    # set up optimizer
    optimizer = optim.Adam(wvfn.parameters(), lr=10**(-log10_learn_rate))

    # train
    wvfn = train_variational_wvfn(wvfn, Rs)

    # print results
    for name, param in wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)
