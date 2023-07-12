### Fit GFMC results to a constant
import argparse
import numpy as np
import h5py
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import paper_plt
paper_plt.load_latex_config()

np.random.seed(0)

# fitting parameters
parser = argparse.ArgumentParser()
# data to fit
parser.add_argument('--database1', type=str, required=True)
parser.add_argument('--database2', type=str, required=True)
parser.add_argument('--dataset', type=str, default="Hammys")
# how many steps to skip in between samples to keep correlations managable
parser.add_argument('--n_skip', type=int, default=1)
# how many steps to average to keep correlations managable
parser.add_argument('--n_block', type=int, default=5)
# how many bootstrap samples
parser.add_argument('--n_boot', type=int, default=200)
# how often to print
parser.add_argument('--n_print', type=int, default=1)
# how many fits to do
parser.add_argument('--n_fits', type=int, default=30)
# stop increase n_tau_skip after agreement within how many sigma
parser.add_argument('--n_tau_tol', type=float, default=2.0)
# dtau for plotting
parser.add_argument('--dtau', type=float, default=0.4)
# plot height in sigma
parser.add_argument('--plot_scale', type=float, default=20)
parser.add_argument('--noshrink', action='store_true', default=False)
globals().update(vars(parser.parse_args()))

shrink = not noshrink
print("Shrinkage: ", shrink)

if n_skip > 1 and n_block > 1:
    print("DON'T SKIP AND BLOCK")
    throw()

# read data
f1 = h5py.File(database1, 'r')
f2 = h5py.File(database2, 'r')
dset1 = f1[dataset]
dset2 = f2[dataset]

dshape = dset1.shape
assert(dshape == dset2.shape)

n_step_full = dshape[0]
n_walk_full = dshape[1]

print(dshape)
if dataset == "Rs":
    dset1 = np.mean(np.abs(dset1), axis=(2,3))
    dset2 = np.mean(np.abs(dset2), axis=(2,3))

# read weights
dset1_Ws = f1["Ws"]
dset2_Ws = f2["Ws"]

model_fits = np.zeros((n_fits))
model_errs = np.zeros((n_fits))
model_redchisq = np.zeros((n_fits))
model_weights = np.zeros((n_fits))

last_fit = 1e6

min_dof = 3

for n_tau_skip_exp in range(round(np.log(dshape[0]//n_walk_full+1)/np.log(2)), round(np.log(dshape[0])/np.log(2))-1):
    n_tau_skip = 2*2**(n_tau_skip_exp+1)
    fit_step = ((dshape[0]-min_dof*n_tau_skip) // n_fits)
    print("\nTRYING N_TAU_SKIP = ", n_tau_skip)
    if ((dshape[0]-min_dof*n_tau_skip) // n_tau_skip) < n_fits:
        n_fits = (dshape[0]-min_dof*n_tau_skip) // n_tau_skip
        model_fits = np.zeros((n_fits))
        model_errs = np.zeros((n_fits))
        model_redchisq = np.zeros((n_fits))
        model_weights = np.zeros((n_fits))
    for fit_num in range(0, n_fits):
        start_fit = fit_num * fit_step
        #full_data = np.real(dset[start_fit::n_tau_skip] * dset_Ws[start_fit::n_tau_skip])
        #full_Ws = np.real(dset_Ws[start_fit::n_tau_skip])

        n_step = (dshape[0] - start_fit) // n_tau_skip
        full_data1 = np.zeros((n_step, n_walk_full))
        full_Ws1 = np.zeros((n_step, n_walk_full))
        full_data2 = np.zeros((n_step, n_walk_full))
        full_Ws2 = np.zeros((n_step, n_walk_full))
        for tau in range(n_step):
            for k in range(n_tau_skip):
                full_data1[tau] += np.real(dset1[start_fit+tau*n_tau_skip+k] * dset1_Ws[start_fit+tau*n_tau_skip+k])
                full_Ws1[tau] += np.real(dset1_Ws[start_fit+tau*n_tau_skip+k])
                full_data2[tau] += np.real(dset2[start_fit+tau*n_tau_skip+k] * dset2_Ws[start_fit+tau*n_tau_skip+k])
                full_Ws2[tau] += np.real(dset2_Ws[start_fit+tau*n_tau_skip+k])

        if n_skip > 1:
            n_walk = n_walk_full // n_skip
        elif n_block > 1:
            n_walk = n_walk_full // n_block
        else :
            n_walk = n_walk_full

        # block data
        data1 = np.zeros((n_step,n_walk))
        Ws1 = np.zeros((n_step,n_walk))
        data2 = np.zeros((n_step,n_walk))
        Ws2 = np.zeros((n_step,n_walk))
        for i in range(n_walk):
            for k in range(n_block):
                data1[:,i] += full_data1[:,i*n_block+k]/n_block
                Ws1[:,i] += full_Ws1[:,i*n_block+k]/n_block
                data2[:,i] += full_data2[:,i*n_block+k]/n_block
                Ws2[:,i] += full_Ws2[:,i*n_block+k]/n_block

        # sparsen data
        if n_skip > 1:
            data1 = np.zeros((n_step,n_walk))
            Ws1 = np.zeros((n_step,n_walk))
            data2 = np.zeros((n_step,n_walk))
            Ws2 = np.zeros((n_step,n_walk))
            for i in range(n_walk):
                data1[:,i] += full_data1[:,i*n_skip]
                Ws1[:,i] += full_Ws1[:,i*n_skip]
                data2[:,i] += full_data2[:,i*n_skip]
                Ws2[:,i] += full_Ws2[:,i*n_skip]

        # sample mean for each step
        sample_mean = np.zeros((n_step))
        sample_mean_num = np.zeros((n_step))
        for n in range(n_step):
            sample_mean[n] = np.mean(data1[n]) / np.mean(Ws1[n]) - np.mean(data2[n]) / np.mean(Ws2[n])
            sample_mean_num[n] = np.mean(data1[n])

        print("\n SAMPLE MEAN")
        print(np.size(sample_mean))
        print(sample_mean[0:n_step:n_print])

        # sample covariance
        sample_covar_num = np.zeros((n_step,n_step))
        for n in range(n_step):
            for m in range(n_step):
               sample_covar_num[n,m] = (np.mean(data1[n]*data1[m]) - np.mean(data1[n])*np.mean(data1[m])) * n_walk/(n_walk-1)

        #print("\n SAMPLE ERR")
        #print(sample_covar.shape)
        #print(np.array([np.sqrt(sample_covar[n,n]/n_walk) for n in range(0,n_step,n_print)]))

        # bootstrap ensemble means
        boot_ensemble = np.zeros((n_boot, n_step))
        for b in range(n_boot):
            inds = np.random.randint(n_walk, size=n_walk)
            for n in range(n_step):
                this_boot1 = data1[n][inds]
                this_boot_Ws1 = Ws1[n][inds]
                this_boot2 = data2[n][inds]
                this_boot_Ws2 = Ws2[n][inds]
                boot_ensemble[b,n] = np.mean(this_boot1) / np.mean(this_boot_Ws1) - np.mean(this_boot2) / np.mean(this_boot_Ws2)

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
        norm_data = np.array([ (data1[n,:] - sample_mean_num[n])/np.sqrt(sample_covar_num[n,n]) for n in range(n_step) ])
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

        lam *= 2
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

        print("\n BOOTSTRAP COVARIANCE WITH OPTIMAL SHRINKAGE ERR")
        print(np.array([np.sqrt(boot_covar_shrunk[n,n]) for n in range(0, n_step, n_print)]))

        # invert covariance matrix
        boot_covar_shrunk_inv = np.linalg.inv(boot_covar_shrunk)

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

    if abs(model_averaged_fit - last_fit) < n_tau_tol*model_averaged_err:
        print("\nRESULT ", model_averaged_fit, " +/- ", model_averaged_err, " AGREES WITH PREVIOUS N_TAU_SKIP ", last_fit, ", DONE")
        break
    else:
        print("\nRESULT ", model_averaged_fit, " +/- ", model_averaged_err, " DISAGREES WITH PREVIOUS N_TAU_SKIP ", last_fit, ", INCREASING")
        last_fit = model_averaged_fit
        n_tau_skip *= 2

outprefix = database1[:-3]+'_corrdiff'

with open(outprefix+'.csv', 'w', newline='') as csvfile:
    fieldnames = ['mean', 'err', 'chi2dof']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'mean': model_averaged_fit, 'err': model_averaged_err, 'chi2dof': model_averaged_redchisq})


# plot H(tau)
plot_data1 = np.real(dset1[:] * dset1_Ws[:])
plot_Ws1 = np.real(dset1_Ws[:])
plot_data2 = np.real(dset2[:] * dset2_Ws[:])
plot_Ws2 = np.real(dset2_Ws[:])
plot_n_step = len(plot_Ws1)
plot_tau = np.arange(plot_n_step) * dtau

plot_sample_mean = np.zeros((plot_n_step))
for n in range(plot_n_step):
    plot_sample_mean[n] = np.mean(plot_data1[n]) / np.mean(plot_Ws1[n]) - np.mean(plot_data2[n]) / np.mean(plot_Ws2[n])

plot_boot_ensemble = np.zeros((n_boot, plot_n_step))
for b in range(n_boot):
    inds = np.random.randint(n_walk, size=n_walk_full)
    for n in range(plot_n_step):
        this_boot1 = plot_data1[n][inds]
        this_boot_Ws1 = plot_Ws1[n][inds]
        this_boot2 = plot_data2[n][inds]
        this_boot_Ws2 = plot_Ws2[n][inds]
        plot_boot_ensemble[b,n] = np.mean(this_boot1) / np.mean(this_boot_Ws1) - np.mean(this_boot2) / np.mean(this_boot_Ws2)

plot_boot_var = np.zeros((plot_n_step))
for n in range(plot_n_step):
    plot_boot_var[n] = (np.mean(plot_boot_ensemble[:,n]**2) - np.mean(plot_boot_ensemble[:,n])**2) * n_walk_full/(n_walk_full-1)

plot_errs = np.sqrt(plot_boot_var)

highest_weight_fit_num = np.argmax(model_weights)
highest_weight_start_fit = highest_weight_fit_num * fit_step
rect_start = highest_weight_start_fit * dtau

fig, ax = plt.subplots(1,1, figsize=(4,3))
ax.errorbar(plot_tau, plot_sample_mean, yerr=plot_errs, color='xkcd:forest green')
ax.set_ylim(model_averaged_fit - plot_scale*model_averaged_err, model_averaged_fit + plot_scale*model_averaged_err)
rect = patches.Rectangle((rect_start, model_averaged_fit - model_averaged_err), plot_tau[-1]-rect_start, 2*model_averaged_err, linewidth=0, facecolor='xkcd:blue', zorder=10, alpha=0.7)
ax.add_patch(rect)
ax.set_xlabel(r'$\tau \, m_Q$')
ax.set_ylabel(r'$\left< H(\tau) \right> / m_Q$')
fig.savefig(outprefix+'_EMP.pdf')
