import numpy as np
import h5py
import sys
import argparse
import csv

sys.path.insert(0, './korr_dev')
from korr.drivers.korrelator import korrelator
from korr.bootstrap import make_bs_tensor, make_nbs_tensor
from korr.outlier import batched_median, make_boot_ci

# fitting parameters
parser = argparse.ArgumentParser()
# data to fit
parser.add_argument('--database', type=str, required=True)
parser.add_argument('--dataset', type=str, default="Hammys")
parser.add_argument('--n_block', type=int, default=5)
parser.add_argument('--n_skip', type=int, default=10)
parser.add_argument('--n_boot', type=int, default=50)
parser.add_argument('--dtau', type=float, default=0.4)
parser.add_argument('--thresh', type=float, default=0.05)
globals().update(vars(parser.parse_args()))

def bin_data(data, N_bin):
    n_meas = data.shape[0]
    bin_data = np.zeros( (n_meas//N_bin,)+data.shape[1:], dtype=complex )
    for n in range(n_meas//N_bin):
        for b in range(N_bin):
            bin_data[n] += data[n*N_bin+b]/N_bin
    return bin_data

f = h5py.File(database, 'r')
dset = f[dataset]

print(dset.shape)

r_string = ""
if dataset == "Rs":
    n_coord = dset.shape[2]
    CoM = np.zeros_like(dset)
    denom = np.mean(masses)
    num = np.zeros_like(dset)
    for r in range(n_coord):
        num[:,:,r,:] = masses[r]*dset[:,:,r,:]
    CoM = np.mean(num, axis=2, keepdims=True) / denom
    full_dset = np.zeros((n_step_full,n_walk_full))
    for r in which_Rs:
        full_dset += adl.norm_3vec(dset - CoM)[:,:,r]/len(which_Rs)
        r_string += str(r)
    dset = full_dset

# read weights
dset_Ws = np.transpose(f["Ws"])
dset = np.transpose(dset)

basename = database[:-3]
if dataset == "Rs":
    after_Rs = database.find("Rs") + 2
    basename = database[0:after_Rs]+r_string+database[after_Rs:-3]

Ct = dset_Ws
print("read Ct, shape = ", Ct.shape)

N_outer=n_boot
N_inner=n_boot

Ct = bin_data(Ct[:,:-1:n_skip], n_block)

print("blocked Ct, shape = ", Ct.shape)

N_samps = Ct.shape[0]
Nt = Ct.shape[1]
all_m = range(1,Nt//2)

tau_ac = 0
sub_dset = np.real(dset[tau_ac]*dset_Ws[tau_ac] - np.mean(dset[tau_ac]*dset_Ws[tau_ac]))
auto_corr = []
c0 = np.mean(sub_dset * sub_dset)
auto_corr.append(c0)
for i in range(1,N_samps//4):
     auto_corr.append(np.mean(sub_dset[i:] * sub_dset[:-i]))
littlec = np.asarray(auto_corr) / c0
last_point = N_samps//8
def tauint(t, littlec):
     return 1 + 2 * np.sum(littlec[1:t])
y = [tauint(i, littlec) for i in range(1, last_point)]
tauint0 = tauint(last_point, littlec)
print("integrated autocorrelation time = ", tauint0)

korr = korrelator(np.real(Ct), N_outer=N_outer, N_inner=N_inner, store_P=True, thresh=thresh)

Et = dset
Et = bin_data(Et[:,:-1:n_skip], n_block)

C3pt = np.zeros((N_samps, Nt//2, Nt//2))
for tau in range(Nt//2):
    for sigma in range(Nt//2):
        C3pt[:,sigma,tau] = np.real(Et[:,sigma+tau]*Ct[:,sigma+tau])

print("read C3pt, shape = ", C3pt.shape)

np.random.seed(17)

bs_tensor = make_bs_tensor(N_samps=N_samps, N_boots=N_inner)
bs_C3pt = np.einsum('bn,n...->b...', bs_tensor, C3pt)

nbs_tensor = make_nbs_tensor(N_samps=N_samps, N_outer=N_outer, N_inner=N_inner)
nbs_C3pt = np.einsum('oin,n...->oi...', nbs_tensor, C3pt)

med_H = np.zeros((Nt//2))
med_H_err = np.zeros((Nt//2))

for m in range(1,Nt//2+1):
    bs_P = korr.bs_Pl[m][:,0,:,0] # ~ bs, k, t, a
    nbs_P = korr.nbs_Pl[m][:,:,0,:,0] # ~ nbs, bs, k, t, a

    bs_H = np.real(np.einsum('bst,bs,bt->b', bs_C3pt[:,:m,:m], bs_P, np.conj(bs_P)))
    med_H[m-1] = batched_median(bs_H, thresh=thresh)

    nbs_H = np.real(np.einsum('bist,bis,bit->bi', nbs_C3pt[:,:,:m,:m], nbs_P, np.conj(nbs_P)))
    med_H_err[m-1] = make_boot_ci(np.median(nbs_H, axis=1))

print("H = ", med_H)
print("H_err = ", med_H_err)

if tauint0 > 0.5:
    med_H_err *= 2*tauint0

print("added autocorrelations")
print("integrated autocorrelation time = ", tauint0)
print("H_err = ", med_H_err)

final_E = korr.final_E[0] / (n_skip * dtau)
final_E_err = korr.final_E_err[0] / (n_skip * dtau)
print("\nFinal E = ", final_E, " +/- ", final_E_err)
print("Final <H> = ", med_H[-1], " +/- ", med_H_err[-1])
print("\n")

print("writing ", basename+'_Lanczos.csv')
with open(basename+'_Lanczos.csv', 'w', newline='') as csvfile:
    fieldnames = ['H_mean', 'H_err', 'E_mean', 'E_err']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'H_mean': med_H[-1], 'H_err': med_H_err[-1], 'E_mean': final_E, 'E_err': final_E_err })
