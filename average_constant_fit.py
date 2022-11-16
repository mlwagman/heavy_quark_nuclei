### Fit GFMC results to a constant
import argparse
import numpy as np
import h5py
import csv

np.random.seed(0)

# fitting parameters
parser = argparse.ArgumentParser()
# data to fit
parser.add_argument('--database', type=str, required=True)
parser.add_argument('--dataset', type=str, default="Hammys")
# how many steps to skip in between samples to keep correlations managable
parser.add_argument('--n_skip', type=int, default=1)
# how many steps to skip in between samples to keep correlations managable
parser.add_argument('--n_tau_skip', type=int, default=1)
# how many steps to average to keep correlations managable
parser.add_argument('--n_block', type=int, default=1)
# how many bootstrap samples
parser.add_argument('--n_boot', type=int, default=200)
# how often to print
parser.add_argument('--n_print', type=int, default=10)
# how many fits to do
parser.add_argument('--n_fits', type=int, default=30)
# how many fits to do
parser.add_argument('--noshrink', action='store_true', default=False)
globals().update(vars(parser.parse_args()))

shrink = not noshrink
print("Shrinkage: ", shrink)

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
dset_Ws = np.ones_like(dset_Ws) 

fit_step = (dset.shape[0] // n_fits)

model_fits = np.zeros((n_fits))
model_errs = np.zeros((n_fits))
model_redchisq = np.zeros((n_fits))
model_weights = np.zeros((n_fits))

last_fit = 0.0

n_walk_full = dset.shape[1]

for n_tau_skip_exp in range((dset.shape[0] // n_walk_full) + 1, round(np.log(dset.shape[0])/np.log(2))-1):
    n_tau_skip = 2**n_tau_skip_exp
    print("\nTRYING N_TAU_SKIP = ", n_tau_skip)
    if (dset.shape[0] // n_tau_skip) < n_fits:
        n_fits = dset.shape[0] // n_tau_skip
        model_fits = np.zeros((n_fits))
        model_errs = np.zeros((n_fits))
        model_redchisq = np.zeros((n_fits))
        model_weights = np.zeros((n_fits))
    for fit_num in range(0, n_fits):
        start_fit = fit_num * fit_step
        #start_fit = (n_fits-1-fit_num) * fit_step
        full_data = np.real(dset[start_fit::n_tau_skip] * dset_Ws[start_fit::n_tau_skip])
        full_Ws = np.real(dset_Ws[start_fit::n_tau_skip])
        #full_data = np.real(dset[start_fit:] * dset_Ws[start_fit:])
        #full_Ws = np.real(dset_Ws[start_fit:])
        n_step = full_data.shape[0]
        
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
       
        if shrink: 
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
        else:
            lam = 0.0
    
        print("\n OPTIMAL SHRINKAGE PARAMETER")
        print(lam)
        
        fit = np.mean(sample_mean)
        delta_fit = np.sqrt((sample_mean - fit) @ (sample_mean - fit))
        chisq = (sample_mean - fit) @ (sample_mean - fit)
        if lam < 0.9:
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
        AIC_aug = chisq + 2 + 2*start_fit/n_tau_skip
            
        print("\nFIT RESULT ", fit_num+1, " / ", n_fits)
        print("n_start = ", start_fit)
        print(fit, " +/- ", delta_fit)
        print("dof = ", dof)
        print("chi^2/dof = ", chi_red)
        print("AIC = ", AIC_aug)
    
        model_fits[fit_num] = fit
        model_errs[fit_num] = delta_fit
        model_redchisq[fit_num] = chi_red 
        model_weights[fit_num] = np.exp(-0.5*AIC_aug)
    
    model_weights /= np.sum(model_weights)
    
    model_averaged_fit = np.sum( model_weights * model_fits )
    
    model_averaged_stat_sq = np.sum( model_weights * model_errs**2 )
    model_averaged_sys_sq = np.sum( model_weights * model_fits**2 ) - np.sum( model_weights * model_fits )**2
    model_averaged_err = np.sqrt( model_averaged_stat_sq + model_averaged_sys_sq )
    
    model_averaged_redchisq = np.sum( model_weights * model_redchisq )
            
    
    print("\nMODEL AVERAGING RESULT")
    print("N_TAU_SKIP = ", n_tau_skip)
    print("fits = ", model_fits)
    print("errs = ", model_errs)
    print("reduced chisq = ", model_redchisq)
    print("weights = ", model_weights)
    print("model averaged fit = ", model_averaged_fit)
    print("model averaged reduced chisq = ", model_averaged_redchisq)
    print("model averaged stat err = ", np.sqrt(model_averaged_stat_sq))
    print("model averaged sys err = ", np.sqrt(model_averaged_sys_sq))
    print("model averaged err = ", model_averaged_err)
    
    if abs(model_averaged_fit - last_fit) < model_averaged_err:
        print("\nRESULT ", model_averaged_fit, " +/- ", model_averaged_err, " AGREES WITH PREVIOUS N_TAU_SKIP ", last_fit, ", DONE")
        break
    else:
        print("\nRESULT ", model_averaged_fit, " +/- ", model_averaged_err, " DISAGREES WITH PREVIOUS N_TAU_SKIP ", last_fit, ", INCREASING")
        last_fit = model_averaged_fit
        n_tau_skip *= 2

with open(database[:-2]+'csv', 'w', newline='') as csvfile:
    fieldnames = ['mean', 'err', 'chi2dof']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'mean': model_averaged_fit, 'err': model_averaged_err, 'chi2dof': model_averaged_redchisq})

