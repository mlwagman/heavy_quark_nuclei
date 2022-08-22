from julia import Main
Main.include("shrinkage.jl")
import numpy as np
import random
from analysis import *
import h5py


f = h5py.File('hamtest.hdf5', 'r')
dset = f['Hammys']
dataslice = dset[:,100]

# n_walk sets sample size - if I set sample size = n_walk then var and covar trivial
n_walk = dset.shape[1]

n_step = dset.shape[0]
n_boot = 200

#testing smaller sample sizes
n_walk = 40

#standard means for each step
means = []
for i in range(n_step):
    means.append(np.mean(np.real(dset[i,:])))
print(np.size(means))
print(means)

#testing sample means
sample_test = []
for i in range(n_step):
    y = random.sample((np.real(dset[i,:])).tolist(), n_walk)
    avg = np.mean(y)
    sample_test.append(avg)

print(np.size(sample_test))

#generating samples and taking mean for each step
boot_means = np.zeros((n_step,n_boot))
print(boot_means.shape)
for i in range(n_step):
    for j in range(n_boot):
        y = random.sample((np.real(dset[i,:])).tolist(), n_walk)
        avg = np.mean(y)
        boot_means[i,j] = avg

print(boot_means.shape)
print(boot_means)

#computing variance
boot_var = np.zeros(n_step)
for i in range(n_step):
    y = (np.mean(boot_means[i,:])-means[i])**2
    avg = np.mean(y)
    boot_var[i] = avg

print(boot_var.shape)
print(boot_var)

#computing covariance
boot_covar = np.zeros((n_step,n_step))
for i in range(n_step):
    for j in range(n_step):
        y = (np.mean(boot_means[i,:])-means[i])*(np.mean(boot_means[j,:])-means[j])
        avg = np.mean(y)
        boot_covar[i,j] = avg

print(boot_covar.shape)
print(boot_covar)

np.savetxt("boot_covar.csv",boot_covar,delimiter=',')

#print(np.mean(sample_mean))

#xcov = covar_from_boots(np.array(sample_mean))

#xgen = bootstrap_gen(*x,Nboot=200)

#xboot = bootstrap(*x, Nboot=200, f=mean)
#xcov = covar_from_boots(np.array(*bootstrap(xgen,Nboot=200,f=mean)))
#print(x)
#print(xboot)
#print(xcov)
#print(sample_mean)
#print(*xgen)
#xboot = bootstrap(*xgen, Nboot=200, f=mean)
#print(xboot)
#print(bootstrap(x,Nboot=200,f=mean))
#print(xcov)
#print(covar_from_boots(bootstrap(x,Nboot=200,f=mean)))
#print(bin_data(x,2,silent_trunc=True))
#print(Main.optimalShrinkage(x))
