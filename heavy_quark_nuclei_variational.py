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

n_coord = 2

def total_Psi_nlm(Rs, n, l, m, Z_n):
    n_walkers = Rs.shape[0]
    assert Rs.shape == (n_walkers, n_coord, 3)
    Psi_nlm_s = np.zeros((n_walkers))
    for i in range(n_walkers):
        # convert to spherical
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        r_n = np.sqrt(x**2 + y**2 + z**2)
        t_n = np.arctan2(np.sqrt(x**2 + y**2), z) 
        p_n = np.arctan2(y, x)
        # evaluate wavefunction
        sym_psi = psi_no_v(n_coord, n, l, m, Z, r, t, p)
        for a in range(n_coord):
            sym_psi = sym_psi.subs(r[a], r_n[a]).subs(t[a], t_n[a]).subs(p[a], p_n[a])
        Psi_nlm_s[i] = sym_psi.subs(Z, Z_n)
    return Psi_nlm_s

def nabla_total_Psi_nlm(Rs, n, l, m, Z_n):
    n_walkers = Rs.shape[0]
    assert Rs.shape == (n_walkers, n_coord, 3)
    nabla_Psi_nlm_s = np.zeros((n_walkers))
    for i in range(n_walkers):
        # convert to spherical
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        r_n = np.sqrt(x**2 + y**2 + z**2)
        t_n = np.arctan2(np.sqrt(x**2 + y**2), z) 
        p_n = np.arctan2(y, x)
        # evaluate wavefunction
        sym_psi = psi_no_v(n_coord, n, l, m, Z, r, t, p)
        nabla_wvfn = 0
        for a in range(n_coord):
            nabla_wvfn += laPlaceSpher(sym_psi, r[a], t[a], p[a])
        for a in range(n_coord):
            nabla_wvfn = nabla_wvfn.subs(r[a], r_n[a]).subs(t[a], t_n[a]).subs(p[a], p_n[a])
        nabla_Psi_nlm_s[i] = nabla_wvfn.subs(Z, Z_n)
    return nabla_Psi_nlm_s

def potential_total_Psi_nlm(Rs, n, l, m, Z):
    n_walkers = Rs.shape[0]
    assert Rs.shape == (n_walkers, n_coord, 3)
    V_Psi_nlm_s = np.zeros((n_walkers))
    wvfn = total_Psi_nlm(Rs, n, l, m, Z)
    for i in range(n_walkers):
        # convert to spherical
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        # evaluate potential
        V = 0
        for a in range(n_coord):
            for b in range(n_coord):
                if b > a:
                    V += 1/np.sqrt( (x[a]-x[b])**2 + (y[a]-y[b])**2 + (z[a]-z[b])**2 )
        V_Psi_nlm_s[i] = V * wvfn[i]
    return V_Psi_nlm_s

def hammy_Psi_nlm(Rs, n, l, m, Z):
    psistar = np.conjugate(total_Psi_nlm(Rs, n, l, m, Z))
    K_psi = nabla_total_Psi_nlm(Rs, n, l, m, Z)/2 
    V_psi = potential_total_Psi_nlm(Rs, n, l, m, Z)
    H_psi = K_psi + V_psi
    print(f'<K> = {psistar * K_psi}')
    print(f'<V> = {psistar * V_psi}')
    print(f'|psi|^2 = {psistar * np.conjugate(psistar)}')
    return psistar * H_psi

    
def draw_coordinates(shape, *, eps=1.0, axis=1):
    dR = eps/np.sqrt(2) * np.random.normal(size=shape)
    # subtract mean to keep center of mass fixed
    dR -= np.mean(dR, axis=axis, keepdims=True)
    return dR

def metropolis_coordinate_ensemble(this_psi, *, n_therm, n_walkers, n_skip, eps):
    # array of walkers to be generated
    Rs = np.zeros((n_walkers, n_coord, 3))
    psi2s = np.zeros((n_walkers))
    this_walker = 0
    # store acceptance ratio
    acc = 0
    # initial condition to start metropolis
    R = np.random.normal(size=(1,n_coord,3))
    # set center of mass position to 0
    R -= np.mean(R, axis=1, keepdims=True)
    # metropolis updates
    for i in range(-n_therm, n_walkers*n_skip):
        # update
        dR = draw_coordinates(R.shape, eps=eps, axis=1)
        new_R = R + dR
        # accept/reject based on |psi(R)|^2
        p_R = np.conjugate(this_psi(R))*this_psi(R)
        p_new_R = np.conjugate(this_psi(new_R))*this_psi(new_R)
        if (np.random.random() < (p_new_R / p_R) and not np.isnan(p_new_R) and p_new_R > 0 and p_new_R < 1 ):
            R = new_R #accept
            p_R = p_new_R
            if i >= 0:
                acc += 1
        # store walker every skip updates
        if i >= 0 and (i+1) % n_skip == 0:
            Rs[this_walker,:,:] = R
            psi2s[this_walker] = p_R
            this_walker += 1
            print(f'i = {i}')
            print(f'|psi(R)|^2 = {p_R}')
            print(f'Total acc frac = {acc / (i+1)}')
    # return coordinates R and respective |psi(R)|^2
    return Rs, psi2s


def variational_wvfn(params, Rs):
    return np.exp(-Rs**2/params[0]**2)


class wvfn(nn.Module):
    # Psi(R) = sum_{n,l,m} c_{n,l,m} psi(n, l, m, R)
    # register c_{n,l,m} as pytorch paramters
    def __init__(self):
        super(wvfn, self).__init__()
        self.sigma = nn.Parameter(torch.ones(1, dtype=torch.double))
    def forward(self, Rs):
        out = torch.exp(-torch.pow(Rs, 2)/self.sigma**2)
        return out 

def loss_function(wvfn, Rs):
    # <psi|H|psi> / <psi|psi>
    # compute hammy
    return torch.sum( torch.pow( wvfn(Rs) - torch.exp(-torch.pow(Rs,2)/4), 2) )
    
def train_variational_wvfn(wvfn, Rs):
    optimizer.zero_grad()
    # train net
    train_time = time.time()
    for n in tqdm.tqdm(range(N_train)):
        step_time = time.time()
        # TODO compute loss
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
    parser.add_argument('--N_samples', default=10000, type=int)
    parser.add_argument('--N_train', default=100000, type=int)
    parser.add_argument('--log10_learn_rate', default=3, type=int)
    globals().update(vars(parser.parse_args()))

    # test chi
    #Hammy = hydrogen.laPlaceSpher(hydrogen.Chi(1, hydrogen.n_coord, 1, 0, 0, 1, hydrogen.r, hydrogen.t, hydrogen.p, hydrogen.v, 1),hydrogen.r[0],hydrogen.t[0],hydrogen.p[0]).subs(hydrogen.r[1],0)+(hydrogen.Potential(hydrogen.rr,hydrogen.B,hydrogen.n_coord)*hydrogen.Chi(1, hydrogen.n_coord, 1, 0, 0, 1, hydrogen.r, hydrogen.t, hydrogen.p, hydrogen.v, 1)).subs(hydrogen.r[1],0)

    #print(hydrogen.simplify(Hammy.subs(hydrogen.v[1],1)))

    B=1
    a=-2/B
    def psi0(Rs):
        return total_Psi_nlm(Rs, 1, 0, 0, 1/a)

    psi(n_coord, 1, 0, 0, 1, r, t, p, v)
    print("yay")
    psi_no_v(n_coord, 1, 0, 0, 1, r, t, p)

    print("psi0")
    print(psi0(np.array([[[1,1,1],[-1,-1,-1]]])))

    n_walkers = 20

    Rs, psi2s = metropolis_coordinate_ensemble(psi0, n_therm=50, n_walkers=n_walkers, n_skip=10, eps=0.5)

    hammy_ME = hammy_Psi_nlm(Rs, 1, 0, 0, 1/a)
    
    print(f'|psi|^2 = {psi2s}')

    E0 = hammy_ME / psi2s

    print(np.mean(E0))
    print(np.sqrt(np.var(E0)/n_walkers))

    throw()

    # initialize wvfn
    wvfn = wvfn()
    for name, param in wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # set up optimizer
    optimizer = optim.Adam(wvfn.parameters(), lr=10**(-log10_learn_rate))

    # initialize random coordinate sample
    Rs = torch.reshape( 10*torch.rand(N_samples*3*N_particles), (N_samples, N_particles, 3) )
    print(Rs.shape)

    # train
    wvfn = train_variational_wvfn(wvfn, Rs)

    # print results
    for name, param in wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)
