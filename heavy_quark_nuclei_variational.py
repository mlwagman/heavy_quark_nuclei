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
import os, sys
import argparse
import copy

from hydrogen import *

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})

N_coord = nCoord
VB = 0.1
N_skip = 10

psitab = []
for nnn in range(1,cutoff+1):
    psitab.append([])
    for ll in range(0,nnn):
        psitab[nnn-1].append([])
        for mm in range(-ll,ll+1):
            psitab[nnn-1][ll].append(psi_no_v(N_coord, nnn, ll, mm, Z, r, t, p, C))

def total_Psi_nlm(Rs, n, l, m, Z_n, C_n, psi_fn):
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
        Psi_nlm_s[i] = psi_fn(C_n, Z_n, r_n, t_n, p_n)
    return Psi_nlm_s

def nabla_total_Psi_nlm(Rs, n, l, m, Z_n, C_n, psi_fn):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    nabla_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    nabla_psi_fn = nabla_psi_no_v(N_coord, n, l, m, Z, r, t, p, C)
    for i in range(N_walkers):
        # convert to spherical
        x = Rs[i,:,0]
        y = Rs[i,:,1]
        z = Rs[i,:,2]
        r_n = torch.sqrt(x**2 + y**2 + z**2)
        t_n = torch.atan2(torch.sqrt(x**2 + y**2), z)
        p_n = torch.atan2(y, x)
        # evaluate wavefunction
        nabla_Psi_nlm_s[i] = nabla_psi_fn(C_n, Z_n, r_n, t_n, p_n)
    return nabla_Psi_nlm_s

def potential_total_Psi_nlm(Rs, n, l, m, Z_n, C_n, psi_fn):
    N_walkers = Rs.shape[0]
    assert Rs.shape == (N_walkers, N_coord, 3)
    V_Psi_nlm_s = torch.zeros((N_walkers), dtype=torch.complex64)
    wvfn = total_Psi_nlm(Rs, n, l, m, Z_n, C_n, psi_fn)
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

def K_Psi_nlm(Rs, n, l, m, Z, C, psi_fn):
    K_psi = -1/2*nabla_total_Psi_nlm(Rs, n, l, m, Z, C, psi_fn)
    return K_psi

def V_Psi_nlm(Rs, n, l, m, Z, C, psi_fn):
    V_psi = potential_total_Psi_nlm(Rs, n, l, m, Z, C, psi_fn)
    return V_psi
atan2
def hammy_Psi_nlm(Rs, n, l, m, Z, C, psi_fn):
    K_psi = K_Psi_nlm(Rs, n, l, m, Z, C, psi_fn)
    V_psi = V_Psi_nlm(Rs, n, l, m, Z, C, psi_fn)
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
    for i in tqdm.tqdm(range(-n_therm, N_walkers*n_skip)):
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

class wvfn(nn.Module):
    # Psi(R) = sum_{n,l,m} c_{n,l,m} psi(n, l, m, R)
    def __init__(self):
        super(wvfn, self).__init__()
        # register Bohr radius a and c_{n,l,m,k,j} as pytorch paramters
        self.a = nn.Parameter(1/VB*torch.ones(1, dtype=torch.double))
        self.C = nn.Parameter(torch.ones((cutoff, cutoff, 2*cutoff-1, N_coord, N_coord), dtype=torch.complex64))
    def psi(self, Rs):
        a_n = self.a[0]
        psi = 0
        for nnn in range(1,cutoff+1):
            for ll in range(0,nnn):
                for mm in range(-ll,ll+1):
                    psi += total_Psi_nlm(Rs, nnn, ll, mm, 1/a_n, self.C, psitab[nnn-1][ll][mm+ll])
        return psi
    def psi2(self, Rs):
        return torch.pow(torch.abs(self.psi(Rs)), 2)
    def hammy(self, Rs):
        a_n = self.a[0]
        H_psi = 0 
        psistar = 0
        for nnn in range(1,cutoff+1):
            for ll in range(0,nnn):
                for mm in range(-ll,ll+1):
                    H_psi += hammy_Psi_nlm(Rs, nnn, ll, mm, 1/a_n, self.C, psitab[nnn-1][ll][mm+ll])
                    psistar += torch.conj(total_Psi_nlm(Rs, nnn, ll, mm, 1/a_n, self.C, psitab[nnn-1][ll][mm+ll]))
        return psistar*H_psi
    def forward(self, Rs):
        a_n = self.a[0]
        H_psi = 0 
        psistar = 0
        for nnn in range(1,cutoff+1):
            for ll in range(0,nnn):
                for mm in range(-ll,ll+1):
                    H_psi += hammy_Psi_nlm(Rs, nnn, ll, mm, 1/a_n, self.C, psitab[nnn-1][ll][mm+ll])
                    psistar += torch.conj(total_Psi_nlm(Rs, nnn, ll, mm, 1/a_n, self.C, psitab[nnn-1][ll][mm+ll]))
        return psistar*H_psi / VB**2, torch.pow(torch.abs(psistar), 2)

def loss_function(wvfn, Rs):
    # <psi|H|psi> / <psi|psi>
    hammy, psi2s = wvfn(Rs)
    N_walkers = len(hammy)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'<psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')
    loss = torch.real( E_trial + np.sqrt(N_walkers)*noise_E_trial )
    return loss

def fast_loss_function(wvfn, Rs, psi2s0):
    # <psi|H|psi> / <psi|psi> 
    # quickly by reweighting using a Monte Carlo distribution proportional to 
    # |psi_0|^2 instead of generating one proportional to |psi|^2
    hammy, psi2s = wvfn(Rs)
    N_walkers = len(hammy)
    E_trial = torch.mean(hammy/psi2s0) / torch.mean(psi2s/psi2s0)
    noise_E_trial = torch.abs( torch.mean(hammy/psi2s0) / torch.mean(psi2s/psi2s0) ) * torch.sqrt( torch.var(hammy/psi2s0)/torch.mean( hammy/psi2s0 )**2 + torch.var(psi2s/psi2s0)/torch.mean( psi2s/psi2s0 )**2 ) / np.sqrt(N_walkers)
    print(f'1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')
    loss = torch.real( E_trial + np.sqrt(N_walkers)*noise_E_trial )
    return loss

def train_variational_wvfn(wvfn):
    print("\nStarting training")
    optimizer.zero_grad()
    # train net
    train_time = time.time()
    max_reduces = 2
    best_loss = 1e10
    best_iter = 0
    best_wvfn_state = copy.deepcopy(wvfn.state_dict())
    for n in tqdm.tqdm(range(N_train)):
        sep_time = time.time()
        if n % N_skip == 0:
            print("\nRefreshing walkers")
            Rs, psi2s = metropolis_coordinate_ensemble(wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=wvfn.a[0].item()/2)
            loss = loss_function(wvfn, Rs)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            if loss < best_loss:
                best_iter = n
                best_loss = loss 
                best_wvfn_state = copy.deepcopy(wvfn.state_dict())
        else:
            loss = fast_loss_function(wvfn, Rs, psi2s)
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step(loss)
            if loss < best_loss:
                best_iter = n
                best_loss = loss 
                best_wvfn_state = copy.deepcopy(wvfn.state_dict())
        tqdm.tqdm.write(f"\nTraining step {n}")
        tqdm.tqdm.write(f"loss function curently {loss}")
        for name, param in wvfn.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        lr = optimizer.param_groups[0]['lr']
        print(f"learn rate = {lr}")
        print(f"bad epochs = {scheduler.num_bad_epochs}")
        if (lr / 10**(-log10_learn_rate)) < 10**(-1*(max_reduces+.5)):
            print(f"reduced learn rate {max_reduces} times, quitting")
            break
    print(f"completed {N_train} steps of training in {time.time() - train_time} sec")
    print(f"best iteration {best_iter}")
    print(f"best loss function {best_loss} \n\n")
    wvfn.load_state_dict(best_wvfn_state)
    return best_loss, wvfn

def diagnostics():
    print("Running positronium diagnostics")
    N_coord = 2

    C_n=torch.zeros((cutoff, cutoff, 2*cutoff-1, N_coord, N_coord));
    B_n=VB
    a_n=2/B_n
    C_n[0,0,0,:,:] = 1;

    psi_fn = psi_no_v(N_coord, 1, 0, 0, Z, r, t, p, C)

    def psi0(Rs):
        return total_Psi_nlm(Rs, 1, 0, 0, 1/a_n, C_n, psi_fn)

    Rs, psi2s0 = metropolis_coordinate_ensemble(psi0, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=1.0)

    print(Rs.shape)

    print(f"psi0 = {psi0(Rs)[0]}")
    print(f"|psi0|^2 = {np.conjugate(psi0(Rs)[0])*psi0(Rs)[0]}")
    hammy_ME = np.conjugate(psi0(Rs))*hammy_Psi_nlm(Rs, 1, 0, 0, 1/a_n, C_n, psi_fn)
    print(f"|psi|^2 = ", psi2s0[0])
    print(f"<psi|H|psi>/|psi|^2 = {hammy_ME[0]/psi2s0[0]}")
    V_ME = np.conjugate(psi0(Rs))*V_Psi_nlm(Rs, 1, 0, 0, 1/a_n, C_n, psi_fn)
    print(f"<psi|V|psi>/|psi|^2 = {V_ME[0]/psi2s0[0]}")
    K_ME = np.conjugate(psi0(Rs))*K_Psi_nlm(Rs, 1, 0, 0, 1/a_n, C_n, psi_fn)
    print(f"<psi|K|psi>/|psi|^2 = {K_ME[0]/psi2s0[0]}")

    print(f'|psi|^2 = {psi2s0}')

    E0 = hammy_ME / psi2s0

    print(f'\nEvery element should be E0=-1/4, {E0} \n')
    print(f'<psi|H|psi>/<psi|psi> = {torch.mean(hammy_ME/psi2s0)} +/- {torch.sqrt(torch.var(hammy_ME/psi2s0))/np.sqrt(N_walkers)} = -1/4?')
    print(f'<psi|V|psi>/<psi|psi> = {torch.mean(V_ME/psi2s0)} +/- {torch.sqrt(torch.var(V_ME/psi2s0))/np.sqrt(N_walkers)}')
    print(f'<psi|K|psi>/<psi|psi> = {torch.mean(K_ME/psi2s0)} +/- {torch.sqrt(torch.var(K_ME/psi2s0))/np.sqrt(N_walkers)}')

    print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N_walkers', default=200, type=int)
    parser.add_argument('--N_train', default=5000, type=int)
    parser.add_argument('--log10_learn_rate', default=1, type=int)
    parser.add_argument('--output', default="./wvfn", type=str)
    globals().update(vars(parser.parse_args()))

    filename = output + "_Ncoord" + str(N_coord) + "_cutoff" + str(cutoff) + f"_potential{VB:.3f}.wvfn"
    print("saving wvfn results to "+filename+"\n")
    if os.path.exists(filename):
        print("Error - remove existing wavefunction, torch save doesn't overwrite\n\n")
        sys.exit()

    #diagnostics()

    # initialize wvfn
    trial_wvfn = wvfn()
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # initialize optimizer
    optimizer = optim.Adam(trial_wvfn.parameters(), lr=10**(-log10_learn_rate))
    N_patience = 5*N_skip
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=N_patience, threshold=0.00001, threshold_mode='abs', verbose=True)

    # train
    best_loss, trial_wvfn = train_variational_wvfn(trial_wvfn)

    # print results
    print(f'Wavefunction results:')
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    Rs, psi2s = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=trial_wvfn.a[0].item()/2)
    hammy, psi2s = trial_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')

    print(f"\n\n Round two!")
    # initialize optimizer
    optimizer = optim.Adam(trial_wvfn.parameters(), lr=10**(-log10_learn_rate-1))
    N_patience = 5*N_skip
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=N_patience, threshold=0.00001, threshold_mode='abs', verbose=True)

    # train
    best_loss, trial_wvfn = train_variational_wvfn(trial_wvfn)

    # print results
    print(f'Wavefunction results:')
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    Rs, psi2s = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=trial_wvfn.a[0].item()/2)
    hammy, psi2s = trial_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')

    print(f"\n\n Round three!")
    # initialize optimizer
    optimizer = optim.Adam(trial_wvfn.parameters(), lr=10**(-log10_learn_rate-2))
    N_patience = 5*N_skip
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=N_patience, threshold=0.00001, threshold_mode='abs', verbose=True)

    # train
    best_loss, trial_wvfn = train_variational_wvfn(trial_wvfn)

    # print results
    print(f'Wavefunction results:')
    for name, param in trial_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    Rs, psi2s = metropolis_coordinate_ensemble(trial_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=trial_wvfn.a[0].item()/2)
    hammy, psi2s = trial_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial}')

    # save best wvfn
    print(f'\n\nSaving best wavefunction to '+filename)
    torch.save(trial_wvfn.state_dict(), filename)

    new_wvfn = wvfn()
    new_dict = torch.load(filename)
    new_wvfn.load_state_dict(new_dict)
    print(f'\nVerifying saved wavefunction results:')
    for name, param in new_wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    Rs, psi2s = metropolis_coordinate_ensemble(new_wvfn.psi, n_therm=500, N_walkers=N_walkers, n_skip=N_skip, eps=trial_wvfn.a[0].item()/2)
    hammy, psi2s = new_wvfn(Rs)
    E_trial = torch.mean(hammy/psi2s)
    noise_E_trial = torch.sqrt(torch.var(hammy/psi2s))/np.sqrt(N_walkers)
    print(f'\n\n1/V^2 <psi|H|psi>/<psi|psi> = {E_trial} +/- {noise_E_trial} \n\n')
