### Fit GFMC results to a constant
import argparse
import numpy as np
import h5py
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import paper_plt

np.random.seed(0)

# fitting parameters
parser = argparse.ArgumentParser()
# data to fit
parser.add_argument('--database', type=str, required=True)
parser.add_argument('--dataset', type=str, default="Hammys")
# how many steps in to start fit
parser.add_argument('--start_fit', type=int, default=0)
# how many steps to skip in between samples to keep correlations managable
parser.add_argument('--n_skip', type=int, default=1)
# how many steps to average to keep correlations managable
parser.add_argument('--n_block', type=int, default=1)
# how many bootstrap samples
parser.add_argument('--n_boot', type=int, default=200)
# how often to print
parser.add_argument('--n_print', type=int, default=1)
# dtau for plotting
parser.add_argument('--dtau', type=float, default=0.2)
# plot height
parser.add_argument('--plot_scale', type=float, default=0.3*1e-5)
globals().update(vars(parser.parse_args()))

plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['xtick.labelsize'] = 6

if n_skip > 1 and n_block > 1:
    print("DON'T SKIP AND BLOCK")
    throw()

# read data
f = h5py.File(database, 'r')
dset = f[dataset]
print(dset.shape)
if dataset == "Rs":
    dset = np.mean(np.abs(dset), axis=(2,3))

# read weights
dset_Ws = f["Ws"]

n_walk_full = dset.shape[1]

if start_fit == -1:
    start_fit = (dset.shape[0] // 10) + 1

for n_tau_skip_exp in range(round(np.log(dset.shape[0]//n_walk_full+1)/np.log(2)), round(np.log(dset.shape[0])/np.log(2))-1):
    n_tau_skip = 2**n_tau_skip_exp
    print("\nTRYING N_TAU_SKIP = ", n_tau_skip)
    #full_data = np.real(dset[start_fit::n_tau_skip] * dset_Ws[start_fit::n_tau_skip])
    #full_Ws = np.real(dset_Ws[start_fit::n_tau_skip])

    n_step = (dset.shape[0] - start_fit) // n_tau_skip
    full_data = np.zeros((n_step, n_walk_full))
    full_Ws = np.zeros((n_step, n_walk_full))
    for tau in range(n_step):
        for k in range(n_tau_skip):
            full_data[tau] += np.real(dset[start_fit+tau*n_tau_skip+k] * dset_Ws[start_fit+tau*n_tau_skip+k])
            full_Ws[tau] += np.real(dset_Ws[start_fit+tau*n_tau_skip+k])
    
    if n_skip > 1:
        n_walk = n_walk_full // n_skip
    elif n_block > 1:
        n_walk = n_walk_full // n_block
    else :
        n_walk = n_walk_full
    
    # block data
    data = np.zeros((n_step,n_walk))
    Ws = np.zeros((n_step,n_walk))
    for i in range(n_walk):
        for k in range(n_block):
            data[:,i] += full_data[:,i*n_block+k]/n_block
            Ws[:,i] += full_Ws[:,i*n_block+k]/n_block
    
    # sparsen data
    if n_skip > 1:
        data = np.zeros((n_step,n_walk))
        Ws = np.zeros((n_step,n_walk))
        for i in range(n_walk):
            data[:,i] += full_data[:,i*n_skip]
            Ws[:,i] += full_Ws[:,i*n_skip]
    
    # sample mean for each step
    sample_mean = np.zeros((n_step))
    sample_mean_num = np.zeros((n_step))
    for n in range(n_step):
        sample_mean[n] = np.mean(data[n]) / np.mean(Ws[n])
        sample_mean_num[n] = np.mean(data[n])
    
    print("\n SAMPLE MEAN")
    print(np.size(sample_mean))
    print(sample_mean[0:n_step:n_print])
    
    # sample covariance
    sample_covar_num = np.zeros((n_step,n_step))
    for n in range(n_step):
        for m in range(n_step):
           sample_covar_num[n,m] = (np.mean(data[n]*data[m]) - np.mean(data[n])*np.mean(data[m])) * n_walk/(n_walk-1)
    
    #print("\n SAMPLE ERR")
    #print(sample_covar.shape)
    #print(np.array([np.sqrt(sample_covar[n,n]/n_walk) for n in range(0,n_step,n_print)]))
    
    # bootstrap ensemble means
    boot_ensemble = np.zeros((n_boot, n_step))
    for b in range(n_boot):
        inds = np.random.randint(n_walk, size=n_walk)
        for n in range(n_step):
            this_boot = data[n][inds]
            this_boot_Ws = Ws[n][inds]
            boot_ensemble[b,n] = np.mean(this_boot) / np.mean(this_boot_Ws)
    
    #print("\n BOOTSTRAP MEAN")
    #print(boot_ensemble.shape)
    #print(np.array([np.mean(boot_ensemble[:,n]) for n in range(0,n_step,n_print)]))
    
    # bootstrap variance
    boot_var = np.zeros((n_step))
    for n in range(n_step):
        boot_var[n] = (np.mean(boot_ensemble[:,n]**2) - np.mean(boot_ensemble[:,n])**2) * n_walk/(n_walk-1)
    
    print("\n BOOTSTRAP ERR")
    print(boot_var.shape)
    print(np.sqrt(boot_var[0:n_step:n_print]))
    
    # bootstrap covariance
    boot_covar = np.zeros((n_step,n_step))
    for n in range(n_step):
        for m in range(n_step):
           boot_covar[n,m] = (np.mean(boot_ensemble[:,n]*boot_ensemble[:,m]) - np.mean(boot_ensemble[:,n])*np.mean(boot_ensemble[:,m])) * n_walk/(n_walk-1)
    
    #print("\n BOOTSTRAP COVARIANCE")
    #print(boot_covar.shape)
    #print("\n DIAGONAL")
    #print(np.array([boot_covar[n,n] for n in range(0, n_step, n_print)]))
    #print("\n TOP ROW")
    #print(np.array([boot_covar[0,n] for n in range(0, n_step, n_print)]))
    
    # normalized sample mean and covariance
    norm_data = np.array([ (data[n,:] - sample_mean_num[n])/np.sqrt(sample_covar_num[n,n]) for n in range(n_step) ])
    norm_covar = np.array([ [ sample_covar_num[n,m]/np.sqrt(sample_covar_num[n,n]*sample_covar_num[m,m]) for m in range(n_step) ] for n in range(n_step) ])
    
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
    
    if lam >= 0.9:
        delta_fit = 0.0
        fit = np.mean(sample_mean)
        delta_fit = np.sqrt((sample_mean - fit) @ (sample_mean - fit))/np.sqrt(n_step)
        chisq = 0.0 
        dof = n_step-1
        chi_red = chisq/dof
        plot_scale = 1e6
        
        print("\n FIT RESULT")
        print("SINGULAR COVARIANCE MATRIX")
        print("n_start = ", start_fit)
        print(fit, " +/- ", delta_fit)
        print("dof = ", dof)
        print("chi^2/dof = ", chi_red)

        break
    else:
        # apply shrinkage
        boot_covar_shrunk = np.zeros((n_step,n_step))
        for n in range(n_step):
            for m in range(n_step):
                if n == m:
                    boot_covar_shrunk[n,m] = boot_covar[n,n]
                else:
                    boot_covar_shrunk[n,m] = (1-lam)*boot_covar[n,m]
        
        print("\n BOOTSTRAP COVARIANCE WITH OPTIMAL SHRINKAGE ERR")
        print(np.array([np.sqrt(boot_covar_shrunk[n,n]) for n in range(0, n_step, n_print)]))
        #print("\n TOP ROW")
        #print(np.array([boot_covar_shrunk[0,n] for n in range(0, n_step, n_print)]))
        
        # invert covariance matrix
        boot_covar_shrunk_inv = np.linalg.inv(boot_covar_shrunk)
        
        #print("\n INVERSE BOOTSTRAP COVARIANCE WITH OPTIMAL SHRINKAGE")
        #print(boot_covar_shrunk_inv.shape)
        #print("\n DIAGONAL")
        #print(np.array([boot_covar_shrunk_inv[n,n] for n in range(0, n_step, n_print)]))
        #print("\n TOP ROW")
        #print(np.array([boot_covar_shrunk_inv[0,n] for n in range(0, n_step, n_print)]))
        
        # do the fit!
        Delta = 1/np.sum(boot_covar_shrunk_inv, axis=None)
        delta_fit = np.sqrt(Delta)
        fit = Delta * np.sum(sample_mean @ boot_covar_shrunk_inv)
        chisq = (sample_mean - fit) @ boot_covar_shrunk_inv @ (sample_mean - fit)
        dof = n_step-1
        chi_red = chisq/dof
        
        print("\n FIT RESULT")
        print("n_start = ", start_fit)
        print(fit, " +/- ", delta_fit)
        print("dof = ", dof)
        print("chi^2/dof = ", chi_red)


with open(database[:-2]+'csv', 'w', newline='') as csvfile:
    fieldnames = ['mean', 'err', 'chi2dof']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'mean': fit, 'err': delta_fit, 'chi2dof': chisq/dof})

# plot H(tau)
plot_data = np.real(dset[:] * dset_Ws[:])
plot_Ws = np.real(dset_Ws[:])
plot_n_step = len(plot_Ws)
plot_tau = np.arange(plot_n_step) * dtau
        
plot_sample_mean = np.zeros((plot_n_step))
for n in range(plot_n_step):
    plot_sample_mean[n] = np.mean(plot_data[n]) / np.mean(plot_Ws[n])

plot_boot_ensemble = np.zeros((n_boot, plot_n_step))
for b in range(n_boot):
    inds = np.random.randint(n_walk, size=n_walk)
    for n in range(plot_n_step):
        this_boot = plot_data[n][inds]
        this_boot_Ws = plot_Ws[n][inds]
        plot_boot_ensemble[b,n] = np.mean(this_boot) / np.mean(this_boot_Ws)

plot_boot_var = np.zeros((plot_n_step))
for n in range(plot_n_step):
    plot_boot_var[n] = (np.mean(plot_boot_ensemble[:,n]**2) - np.mean(plot_boot_ensemble[:,n])**2) * n_walk/(n_walk-1)

plot_errs = np.sqrt(np.abs(plot_boot_var))

rect_start = start_fit * dtau

print("plot scale = ",plot_scale)

fig, ax = plt.subplots(1,1, figsize=(4,3))
ax.errorbar(plot_tau, plot_sample_mean, yerr=plot_errs, color='xkcd:forest green')
ax.set_ylim(fit - plot_scale, fit + plot_scale)
rect = patches.Rectangle((rect_start, fit - delta_fit), plot_tau[-1]-rect_start, 2*delta_fit, linewidth=0, facecolor='xkcd:blue', zorder=10, alpha=0.7)
ax.add_patch(rect)
ax.set_xlabel(r'$\tau \, m_Q$')
ax.set_ylabel(r'$\left< H(\tau) \right> / m_Q$')
fig.savefig(database[:-3]+'_EMP.pdf')
