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

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})

def variational_wvfn(params, Rs):
    return np.exp(-Rs**2/params[0]**2)


class wvfn(nn.Module):
    def __init__(self):
        super(wvfn, self).__init__()
        self.sigma = nn.Parameter(torch.ones(1, dtype=torch.double))
    def forward(self, Rs):
        out = torch.exp(-torch.pow(Rs, 2)/self.sigma**2)
        return out 

def loss_function(wvfn, Rs):
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
    parser.add_argument('--N_samples', default=10000, type=int)
    parser.add_argument('--N_train', default=100000, type=int)
    parser.add_argument('--log10_learn_rate', default=3, type=int)
    globals().update(vars(parser.parse_args()))

    # initialize wvfn
    wvfn = wvfn()
    for name, param in wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # set up optimizer
    optimizer = optim.Adam(wvfn.parameters(), lr=10**(-log10_learn_rate))

    # initialize random coordinate sample
    Rs = 10*torch.rand(N_samples)

    # train
    wvfn = train_variational_wvfn(wvfn, Rs)

    # print results
    for name, param in wvfn.named_parameters():
        if param.requires_grad:
            print(name, param.data)
