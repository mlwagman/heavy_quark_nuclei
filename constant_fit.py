### Fit GFMC results to a constant

import argparse
import numpy as np
import h5py

np.random.seed(0)

# fitting parameters
parser = argparse.ArgumentParser()
# data to fit
parser.add_argument('--database', type=str, required=True)
parser.add_argument('--dataset', type=str, default="Hammys")
# how many steps in to start fit
parser.add_argument('--start_fit', type=int, default=1)
# how many steps to skip in between samples to keep correlations managable
parser.add_argument('--n_skip', type=int, default=1)
# how many steps to average to keep correlations managable
parser.add_argument('--n_block', type=int, default=1)
# how many bootstrap samples
parser.add_argument('--n_boot', type=int, default=200)
# how often to print
parser.add_argument('--n_print', type=int, default=10)
globals().update(vars(parser.parse_args()))

if n_skip > 1 and n_block > 1:
    print("DON'T SKIP AND BLOCK")
    throw()

# read data
f = h5py.File(database, 'r')
dset = f[dataset]
full_data = np.real(dset[start_fit:])
n_step = full_data.shape[0]

n_walk_full = full_data.shape[1]
if n_skip > 1:
    n_walk = n_walk_full // n_skip
elif n_block > 1:
    n_walk = n_walk_full // n_block
else :
    n_walk = n_walk_full

printmax = n_step // n_print

# block data
data = np.zeros((n_step,n_walk))
for i in range(n_walk):
    for k in range(n_block):
        data[:,i] += full_data[:,i*n_skip+k]/n_block

# sparsen data
data = np.zeros((n_step,n_walk))
for i in range(n_walk):
    data[:,i] += full_data[:,i*n_skip]

# sample mean for each step
sample_mean = np.zeros((n_step))
for n in range(n_step):
    sample_mean[n] = np.mean(data[n])

print("\n SAMPLE MEAN")
print(np.size(sample_mean))
print(sample_mean[:printmax])

# sample covariance
sample_covar = np.zeros((n_step,n_step))
for n in range(n_step):
    for m in range(n_step):
       sample_covar[n,m] = (np.mean(data[n]*data[m]) - np.mean(data[n])*np.mean(data[m])) * n_walk/(n_walk-1)

print("\n SAMPLE COVARIANCE")
print(sample_covar.shape)
print(np.array([sample_covar[n,n]/n_walk for n in range(printmax)]))

# bootstrap ensemble means
boot_ensemble = np.zeros((n_boot, n_step))
for b in range(n_boot):
    inds = np.random.randint(n_walk, size=n_walk)
    for n in range(n_step):
        this_boot = data[n][inds]
        boot_ensemble[b,n] = np.mean(this_boot)

print("\n BOOTSTRAP MEAN")
print(boot_ensemble.shape)
print(np.array([np.mean(boot_ensemble[:,n]) for n in range(printmax)]))

# bootstrap variance 
boot_var = np.zeros((n_step))
for n in range(n_step):
    boot_var[n] = (np.mean(boot_ensemble[:,n]**2) - np.mean(boot_ensemble[:,n])**2) * n_walk/(n_walk-1)

print("\n BOOTSTRAP VARIANCE")
print(boot_var.shape)
print(boot_var[:printmax])

# bootstrap covariance
boot_covar = np.zeros((n_step,n_step))
for n in range(n_step):
    for m in range(n_step):
       boot_covar[n,m] = (np.mean(boot_ensemble[:,n]*boot_ensemble[:,m]) - np.mean(boot_ensemble[:,n])*np.mean(boot_ensemble[:,m])) * n_walk/(n_walk-1)

print("\n BOOTSTRAP COVARIANCE")
print(boot_covar.shape)
print("\n DIAGONAL")
print(np.array([boot_covar[n,n] for n in range(printmax)]))
print("\n TOP ROW")
print(np.array([boot_covar[0,n] for n in range(printmax)]))

# normalized sample mean and covariance
norm_data = np.array([ (data[n,:] - sample_mean[n])/np.sqrt(sample_covar[n,n]) for n in range(n_step) ])
norm_covar = np.array([ [ sample_covar[n,m]/np.sqrt(sample_covar[n,n]*sample_covar[m,m]) for m in range(n_step) ] for n in range(n_step) ])

# determine optimal shrinkage parameter
d2 = 0.0
for n in range(n_step):
    for m in range(n_step):
        if n == m:
            d2 += (norm_covar[n,m] - 1)**2
        else:
            d2 += norm_covar[n,m]**2
d2 /= n_step
b2 = 0.0
for n in range(n_step):
    for m in range(n_step):
        for i in range(n_walk):
            b2 += (norm_data[n,i]*norm_data[m,i] - norm_covar[n,m])**2
b2 /= (n_step*n_walk*n_walk)
lam = b2/d2
if lam > 1:
    lam = 1
elif lam < 0:
    lam = 0
print("\n OPTIMAL SHRINKAGE PARAMETER")
print(lam)

# apply shrinkage
boot_covar_shrunk = np.zeros((n_step,n_step))
for n in range(n_step):
    for m in range(n_step):
        if n == m:
            boot_covar_shrunk[n,m] = boot_covar[n,n] 
        else:
            boot_covar_shrunk[n,m] = (1-lam)*boot_covar[n,m]

print("\n BOOTSTRAP COVARIANCE WITH OPTIMAL SHRINKAGE")
print(boot_covar_shrunk.shape)
print("\n DIAGONAL")
print(np.array([boot_covar_shrunk[n,n] for n in range(printmax)]))
print("\n TOP ROW")
print(np.array([boot_covar_shrunk[0,n] for n in range(printmax)]))

# invert covariance matrix
boot_covar_shrunk_inv = np.linalg.inv(boot_covar_shrunk)

print("\n INVERSE BOOTSTRAP COVARIANCE WITH OPTIMAL SHRINKAGE")
print(boot_covar_shrunk_inv.shape)
print("\n DIAGONAL")
print(np.array([boot_covar_shrunk_inv[n,n] for n in range(printmax)]))
print("\n TOP ROW")
print(np.array([boot_covar_shrunk_inv[0,n] for n in range(printmax)]))

# do the fit!
Delta = 1/np.sum(boot_covar_shrunk_inv, axis=None)
delta_fit = np.sqrt(Delta)
fit = Delta * np.sum(sample_mean @ boot_covar_shrunk_inv)
chisq = (sample_mean - fit) @ boot_covar_shrunk_inv @ (sample_mean - fit)
dof = n_step-1

print("\n FIT RESULT")
print(fit, " +/- ", delta_fit)
print("dof = ", dof)
print("chi^2/dof = ", chisq/dof)
