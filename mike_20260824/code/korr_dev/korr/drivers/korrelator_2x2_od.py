import numpy as np
import h5py

from ..util import isreal
from ..util import make_block_hankel, stack_dicts
from ..util import effective_mass, bs_effective_mass

from ..eig_util import eig

from ..bootstrap import make_bs_tensor, make_nbs_tensor

from ..outlier import batched_median, inner_median, final_iter_median, homogenize_analyses, make_boot_ci

from ..filter import filter_offdiag_analysis

from ..label import compute_label_dot, label_states_argmax, label_block_analysis

from ..bounds import get_res_bounds, get_offdiag_gap_bounds

# wrap main function for single iteration of off-diag RR
from .basic import run_2x2_offdiag_gpof as run_2x2_offdiag_gpof_basic
def run_2x2_offdiag_gpof(Cab, m, 
                         eig=eig, inv=np.linalg.inv, log=np.log,
                         real=np.real, abs=np.abs, isclose=np.isclose,

                         abs_B=False):
    analysis = run_2x2_offdiag_gpof_basic(Cab, m,
                                          eig=eig, inv=inv,
                                          real=real, abs=abs, isclose=isclose
                                          )

    # derived quantities
    ev = analysis['ev']
    Br, Bl = analysis['Br'], analysis['Bl']
    if abs_B:
        Br = abs(Br)
        Bl = abs(Bl)
    E = -log(ev)
    B = np.where(np.abs(Bl) < np.abs(Br), Bl, Br)

    return dict(
        **analysis,
        E = E,
        B=B,
    )


def run_all_2x2_offdiag_gpof(bs_Cab, 
                        eig=eig, inv=np.linalg.inv, log=np.log,
                        real=np.real, abs=np.abs, isclose=np.isclose,
                             abs_B=False):
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2

    all_m = range(1, m_max+1)
    analysis = {m : run_2x2_offdiag_gpof(bs_Cab, m=m, 
                                         eig=eig, inv=inv, log=log,
                                         real=real, abs=abs, isclose=isclose,
                                         abs_B=abs_B) for m in all_m}

    return analysis # ~ {m : {ev : ..., Zr : ...}}

# save RR results to HDF5 databas
def make_database(rr_analysis, database):
    with h5py.File(database+".h5", "w") as fout:
        for k in [
            # effective mass
            "eff_mass", "eff_mass_err", "eff_mass_boot",
            "theta_LR_k", "theta_LR_k_err", "theta_LR_k_boot",
            # final results
            "final_m", "final_E", "final_B", "final_Bl", "final_Br", 
            "final_Zl", "final_Zr",
            "final_E_err", "final_B_err", "final_Bl_err", "final_Br_err", 
            "final_Zl_err", "final_Zr_err",
            "final_E_boot", "final_B_boot", "final_Bl_boot", "final_Br_boot", 
            "final_Zl_boot", "final_Zr_boot",
            "final_res_lower", "final_res_upper",
            "final_res_lower_err", "final_res_upper_err",
            "final_res_lower_2sigmaerr", "final_res_upper_2sigmaerr",
            "final_res_lower_boot", "final_res_upper_boot",
            "final_gap_ll_lower", "final_gap_ll_upper",
            "final_gap_ll_lower_err", "final_gap_ll_upper_err",
            "final_gap_ll_lower_2sigmaerr", "final_gap_ll_upper_2sigmaerr",
            "final_gap_ll_lower_boot", "final_gap_ll_upper_boot",
            "final_gap_rr_lower", "final_gap_rr_upper",
            "final_gap_rr_lower_err", "final_gap_rr_upper_err",
            "final_gap_rr_lower_2sigmaerr", "final_gap_rr_upper_2sigmaerr",
            "final_gap_rr_lower_boot", "final_gap_rr_upper_boot",
            "final_srq_rr", "final_srq_ll",
            "final_srq_rr_err", "final_srq_ll_err",
            "final_srq_rr_boot", "final_srq_ll_boot",
            # filtering info
            "m_last",
            "eps_ZCW",
            "m_window",
            # m-dependent results
            "med_ev", "med_Zl", "med_Zr", "med_B", "med_Bl", "med_Br", 
            "med_E", "med_theta_LR_y",
            "med_res_lower", "med_res_upper",
            "med_gap_ll_lower", "med_gap_ll_upper",
            "med_gap_rr_lower", "med_gap_rr_upper",
            "med_ev_err", "med_Zl_err", "med_Zr_err", "med_B_err", "med_Bl_err", "med_Br_err", 
            "med_E_err", "med_theta_LR_y_err",
            "med_ev_boot", "med_Zl_boot", "med_Zr_boot", "med_B_boot", "med_Bl_boot", "med_Br_boot", 
            "med_E_boot", "med_theta_LR_y_boot",
            "med_res_lower_err", "med_res_upper_err",
            "med_res_lower_boot", "med_res_upper_boot",
            "med_gap_ll_lower_err", "med_gap_ll_upper_err",
            "med_gap_ll_lower_boot", "med_gap_ll_upper_boot",
            "med_gap_rr_lower_err", "med_gap_rr_upper_err",
            "med_gap_rr_lower_boot", "med_gap_rr_upper_boot",
            "med_srq_rr", "med_srq_ll",
            "med_srq_rr_err", "med_srq_ll_err",
            "med_srq_rr_boot", "med_srq_ll_boot",
        ]:
            fout.create_dataset(k, data=getattr(rr_analysis, k))

        if rr_analysis.store_P:
            all_m = np.array(list(rr_analysis.bs_Pl.keys()))
            for m in all_m:
                for k in ["bs_Pl", "bs_Pr", "nbs_Pl", "nbs_Pr"]:
                    fout.create_dataset(k+"/m"+str(m), data=getattr(rr_analysis, k)[m])

        if rr_analysis.store_boot:
            for k in [
                "bs_ev", "bs_Zl", "bs_Zr", "bs_Bl", "bs_Br", "bs_E",
                "nbs_ev", "nbs_Zl", "nbs_Zr", "nbs_Bl", "nbs_Br","nbs_E",
            ]:
                fout.create_dataset(k, data=getattr(rr_analysis, k))

class korrelator:
    def __init__(self,
        C, # correlator input ~ (Batch, t,[a,b])

        F_ZCW=10, # ZCW filtering hyperparameter
        eps_ZCW=None, # manually specify ZCW cutoff
        staggered=False,

        thresh=0.9, # fraction of boots that must contain physical state
        thresh_inner=None, # fraction of inner boots that must contain physical state, if different from thresh
        final_m=None, # manually specify iteration for final results
        m_window=3, # how many of last iterations to average over

        E_gap_bound=None, # first expected missing energy level for gap bound computation
        abs_B=False, # apply abs before median to B_k^{(m)}
        never_abs=False, # don't apply abs after median to B_k^{(m)}

        seed=17, # seed for bs tensors
        N_outer=200, # bootstrap confidence interval hyperparameter
        N_inner=200, # bootstrap median hyperparameter
        bs_C=None,  # manually pass outer bootstrap tensor
        nbs_C=None, # manually pass inner bootstrap tensor

        database=None, # database to save results
        store_boot=False, # save inner bootstrap Ritz values?
        store_P=False, # save Ritz rotator coefficients?

        progress=True, # display progress bar?
    ):
        # seed for reproducibility / correlations
        np.random.seed(seed)

        # check correlator dimensions are valid for scalar or block
        assert len(C.shape) == 4
        N_samps, Nt, bdim, _ = C.shape
        print(C.shape)
        print((N_samps, Nt, bdim, bdim))
        assert C.shape == (N_samps, Nt, bdim,bdim)
        assert bdim == 2 # special case

        self.N_samps = N_samps
        self.Nt = Nt
        self.bdim = bdim

        print("\n"+"=" * 100)
        print("Starting filtered Rayleigh-Ritz correlator analysis (2x2 off-diagonal special case)")

        print("\nObtaining Ritz values with RR = block Lanczos = block Prony = GPOF")
        print("\nNested bootstrap N_inner = ", N_inner, ", N_outer = ", N_outer)
        if eps_ZCW is not None:
            print("\nApplying ZCW filter with eps_ZCW = ", eps_ZCW)
        else:
            print("\nApplying ZCW filter with adaptive eps_ZCW, hyperparameter F_ZCW = ", F_ZCW)
        
        print("\nTaking median of last ", m_window, " iterations")
        print("\nBlock correlator shape = ", C.shape)
        print("=" * 100 + "\n")

        # make booted corrs (both levels of booting)
        if bs_C is None:
            bs_tensor = make_bs_tensor(N_samps=N_samps, N_boots=N_inner)
            bs_C = np.einsum('bn,n...->b...', bs_tensor, C)
        if nbs_C is None:
            nbs_tensor = make_nbs_tensor(N_samps=N_samps, N_outer=N_outer, N_inner=N_inner)
            nbs_C = np.einsum('oin,n...->oi...', nbs_tensor, C)
        assert bs_C.shape == (N_inner, Nt, bdim, bdim)
        assert nbs_C.shape == (N_outer, N_inner, Nt, bdim, bdim)

        print("bootstrap resampled data")
        print("bootstrap shape = ", bs_C.shape)
        print("nested bootstrap shape = ", nbs_C.shape)

        eff_mass = np.zeros((2, 2, Nt-1))
        eff_mass_err = np.zeros((2, 2, Nt-1))
        eff_mass_boot = np.zeros((2, 2, N_inner, Nt-1))
        for a in range(2):
            for b in range(2):
                eff_mass[a,b,:] = effective_mass(np.mean(C[:,:,a,b], axis=0))
                bs_eff_mass = bs_effective_mass(bs_C[:,:,a,b])
                eff_mass_boot[a,b,:,:] = bs_eff_mass
                for t in range(Nt-1):
                    eff_mass_err[a,b,t] = make_boot_ci(bs_eff_mass[:,t])
        # angle between L and R krylov vectors
        mean_C = np.mean(C, axis=0)
        theta_LR_k = 1/2*(mean_C[:,0,1] + mean_C[:,1,0])/np.sqrt(np.abs(mean_C[:,0,0]*mean_C[:,1,1]))
        bs_theta_LR_k = 1/2*(bs_C[:,:,0,1] + bs_C[:,:,1,0])/np.sqrt(np.abs(bs_C[:,:,0,0]*bs_C[:,:,1,1])) 
        theta_LR_k_boot = bs_theta_LR_k
        theta_LR_k_err = np.zeros((Nt,))
        for t in range(Nt):
            theta_LR_k_err[t] = make_boot_ci(bs_theta_LR_k[:,t])
        print("off-diagonal effective mass")
        print(eff_mass[0,1,:], " +/- ", eff_mass_err[0,1,:])
        print("theta_LR_k")
        print(theta_LR_k, " +/- ", theta_LR_k_err)

        # range of iterations
        all_m = range(1,Nt//2+1)
        if final_m is None:
            final_m = all_m[-2]
        print("final_m = ", final_m)

        # run RR = Lanczos = Prony = GPOF
        print("starting RR analysis")
        all_analyses = run_all_2x2_offdiag_gpof(bs_C, abs_B=abs_B)

        # find eps_ZCW
        fixed_eps_ZCW = None
        if eps_ZCW is None:
            # find last all-real iteration
            print("All-real counts", [np.count_nonzero(all_analyses[m]['is_real']) for m in all_m])
            herm_fac = (2 if staggered else 1)
            print("Thresholds", [bdim*m*N_inner*thresh//herm_fac for m in all_m])
            m_last = max([m if np.count_nonzero(all_analyses[m]['is_real']) > bdim*m*N_inner*thresh//herm_fac else 1 for m in all_m])
            print("m_last = ", m_last)
            ZCW_last = np.median( np.min(all_analyses[m_last]['ZCW'], axis=-1), axis=0)
            eps_ZCW = ZCW_last/F_ZCW
            print("adaptive eps_ZCW = ", eps_ZCW)
        else:
            fixed_eps_ZCW = eps_ZCW
            m_last = 1

        # Ritz value reality and ZCW filters
        print("applying reality and ZCW filters")
        all_filtered = {m : filter_offdiag_analysis(an, an['is_real'] & (an['ZCW'] > eps_ZCW)) for m,an in all_analyses.items()}
        #all_filtered = {m : filter_offdiag_analysis(an, (an['ZCW'] > eps_ZCW)) for m,an in all_analyses.items()}

        # compute Ritz value angles
        for m in all_m:
            all_filtered[m]['theta_LR_y'] = 1/np.sqrt(all_filtered[m]['ynorm_ll']*all_filtered[m]['ynorm_rr'])

        # remove projectors to save memory if not needed
        if not store_P:
            for m in all_m:
                del all_filtered[m]['Pl']
                del all_filtered[m]['Pr']

        # compute one-level bootstrap median
        print("taking median")
        if thresh_inner is None:
            thresh_inner = thresh
        all_filtered_median = {
            m : {k : batched_median(v, thresh=thresh_inner) for k,v in an.items()}
            for m,an in all_filtered.items()
        }
        # median Ps not useful for 3pts
        if store_P:
            for m in all_m:
                del all_filtered_median[m]['Pl']
                del all_filtered_median[m]['Pr']

        # compute two-sided bounds
        print("E_gap_bound = ", E_gap_bound)
        for m in all_m:
            get_res_bounds(all_filtered_median[m])
            get_offdiag_gap_bounds(all_filtered_median[m], E_gap_bound=E_gap_bound, never_abs=never_abs)

        print("\nFinal iteration m =", final_m)
        print("\nZ_final = ", all_filtered_median[final_m]['Zl'])
        print("\nE_final = ", all_filtered_median[final_m]['E'])
        print("\nB_final = ", all_filtered_median[final_m]['B'])

        # run nested bootstrap RR
        print("starting nested bootstrap RR analysis")
        all_nbs_filtered = []
        # split over outer bootstrap
        if progress:
            from tqdm import tqdm
        for b in (tqdm(range(N_outer)) if progress else range(N_outer)):
            # RR for each outer
            nbs_analyses = run_all_2x2_offdiag_gpof(nbs_C[b])
            # find eps_ZCW
            if fixed_eps_ZCW is None:
                nbs_m_last = max([m if np.count_nonzero(nbs_analyses[m]['is_real']) > bdim*m*N_inner*thresh//herm_fac else 1 for m in all_m])
                ZCW_last = np.median( np.min(nbs_analyses[nbs_m_last]['ZCW'], axis=-1), axis=0)
                nbs_eps_ZCW = ZCW_last/F_ZCW
            else:
                nbs_eps_ZCW = fixed_eps_ZCW
                nbs_m_last = 1

            # Ritz value reality and ZCW filters
            nbs_filtered = {m : filter_offdiag_analysis(an, an['is_real'] & (an['ZCW'] > nbs_eps_ZCW))
            #nbs_filtered = {m : filter_offdiag_analysis(an, (an['ZCW'] > nbs_eps_ZCW))
                            for m,an in nbs_analyses.items()}

            # remove projectors to save memory if not needed
            if not store_P:
                for m in all_m:
                    del(nbs_filtered[m]['Pl'])
                    del(nbs_filtered[m]['Pr'])

            # accumulate
            all_nbs_filtered.append(nbs_filtered)


        # build combined nested bootstrap dictionary
        all_nbs_filtered = {m : [d[m] for d in all_nbs_filtered] for m in all_m} # group by m
        all_nbs_filtered = {m : stack_dicts(v) for m,v in all_nbs_filtered.items()}
        # compute Ritz value angles
        for m in all_m:
            all_nbs_filtered[m]['theta_LR_y'] = 1/(all_nbs_filtered[m]['ynorm_ll']*all_nbs_filtered[m]['ynorm_rr'])

        print("taking inner median")
        all_nbs_filtered_median = {
            m : {k : inner_median(v, thresh=thresh, thresh_inner=thresh_inner) for k,v in an.items()}
            for m,an in all_nbs_filtered.items()
        }
        # median Ps not useful for 3pts
        if store_P:
            for m in all_m:
                del all_nbs_filtered_median[m]['Pl']
                del all_nbs_filtered_median[m]['Pr']

        # compute bounds
        for m in all_m:
            get_res_bounds(all_nbs_filtered_median[m])
            get_offdiag_gap_bounds(all_nbs_filtered_median[m], E_gap_bound=E_gap_bound)

        # pad nans to homogenize data arrays
        N_states = max([np.count_nonzero(~np.isnan(all_filtered_median[m]['E'])) for m in all_m])
        print("N_states = ", N_states)
        # outer boot median
        all_filtered_median = homogenize_analyses(all_filtered_median, N_states=N_states, state_ax=0)
        # inner boot median
        all_nbs_filtered_median = homogenize_analyses(all_nbs_filtered_median, N_states=N_states, state_ax=1)
        # outer boot
        all_filtered = homogenize_analyses(all_filtered, N_states=N_states, state_ax=1)
        # inner boot
        all_nbs_filtered = homogenize_analyses(all_nbs_filtered, N_states=N_states, state_ax=2)

        # center bound windows on E
        lower_keys = ['res_lower']
        upper_keys = ['res_upper']
        all_filtered_median = {
            m : {k : an['E'] if (k in lower_keys or k in upper_keys) else v for k,v in an.items()}
            for m,an in all_filtered_median.items()
        }
        # center gap bound windows on SRQ
        lower_ll_keys = ['gap_ll_lower']
        upper_ll_keys = ['gap_ll_upper']
        all_filtered_median = {
            m : {k : -np.log(an['srq_ll']) if (k in lower_ll_keys or k in upper_ll_keys) else v for k,v in an.items()}
            for m,an in all_filtered_median.items()
        }
        lower_rr_keys = ['gap_rr_lower']
        upper_rr_keys = ['gap_rr_upper']
        all_filtered_median = {
            m : {k : -np.log(an['srq_rr']) if (k in lower_rr_keys or k in upper_rr_keys) else v for k,v in an.items()}
            for m,an in all_filtered_median.items()
        }

        # take median of last few iterations
        final_filtered_median = final_iter_median(all_filtered_median, N_states=N_states, m_window=m_window)
        final_nbs_filtered_median = final_iter_median(all_nbs_filtered_median, N_states=N_states, m_window=m_window)

        # compute confidence intervals for all iterations
        print("computing confidence intervals")
        all_filtered_median_err = {
            m : {k : make_boot_ci(np.real(v), side="lower", cv=an['E']) if (k in lower_keys or k in lower_ll_keys or k in lower_rr_keys)
                     else
                     make_boot_ci(np.real(v), side="lower", cv=(-np.log(an['srq_ll'])) if k in lower_ll_keys
                     else
                     make_boot_ci(np.real(v), side="lower", cv=(-np.log(an['srq_rr'])) if k in lower_rr_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=an['E']) if k in upper_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=(-np.log(an['srq_ll'])) if k in upper_ll_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=(-np.log(an['srq_rr'])) if k in upper_rr_keys
                     else
                     make_boot_ci(np.real(v)) for k,v in an.items()}
        for m,an in all_nbs_filtered_median.items()} 

        # compute confidence intervals for final iteration
        final_filtered_median_err = {k : make_boot_ci(np.real(v), side="lower", cv=final_filtered_median['E']) if k in lower_keys
                     else
                     make_boot_ci(np.real(v), side="lower", cv=(-np.log(final_filtered_median['srq_ll'])) if k in lower_ll_keys
                     else
                     make_boot_ci(np.real(v), side="lower", cv=(-np.log(final_filtered_median['srq_rr'])) if k in lower_rr_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=final_filtered_median['E']) if k in upper_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=(-np.log(final_filtered_median['srq_ll'])) if k in upper_ll_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=(-np.log(final_filtered_median['srq_rr'])) if k in upper_rr_keys
                     else
                     make_boot_ci(np.real(v)) for k,v in final_nbs_filtered_median.items()}

        # compute 2sigma confidence intervals for final iteration
        final_filtered_median_2sigmaerr = {k : make_boot_ci(np.real(v), sigma=2, side="lower", cv=final_filtered_median['E']) if k in lower_keys
                     else
                     make_boot_ci(np.real(v), sigma=2, side="lower", cv=(-np.log(final_filtered_median['srq_ll'])) if k in lower_ll_keys
                     else
                     make_boot_ci(np.real(v), sigma=2, side="lower", cv=(-np.log(final_filtered_median['srq_rr'])) if k in lower_rr_keys
                     else
                     make_boot_ci(np.real(v), sigma=2, side="upper", cv=final_filtered_median['E']) if k in upper_keys
                     else
                     make_boot_ci(np.real(v), sigma=2, side="upper", cv=(-np.log(final_filtered_median['srq_ll'])) if k in upper_ll_keys
                     else
                     make_boot_ci(np.real(v), sigma=2, side="upper", cv=(-np.log(final_filtered_median['srq_rr'])) if k in upper_rr_keys
                     else
                     make_boot_ci(np.real(v), sigma=2) for k,v in final_nbs_filtered_median.items()}


        print("\nFinal iteration m in [", final_m-m_window+1, ", ", final_m, "]")
        print("\nZ_final = ", final_filtered_median['Zl'], " +/- ", final_filtered_median_err['Zl'])
        print("\nE_final = ", final_filtered_median['E'], " +/- ", final_filtered_median_err['E'])
        print("\nB_final = ", final_filtered_median['B'], " +/- ", final_filtered_median_err['B'])
        print("\nres_lower_final 1sigma = ", final_filtered_median['res_lower'] - final_filtered_median_err['res_lower'])
        print("res_upper_final 1sigma = ", final_filtered_median['res_upper'] + final_filtered_median_err['res_upper'])
        print("\ngap_ll_lower_final 1sigma = ", final_filtered_median['gap_ll_lower'] - final_filtered_median_err['gap_ll_lower'])
        print("gap_ll_upper_final 1sigma = ", final_filtered_median['gap_ll_upper'] + final_filtered_median_err['gap_ll_upper'])
        print("\ngap_rr_lower_final 1sigma = ", final_filtered_median['gap_rr_lower'] - final_filtered_median_err['gap_rr_lower'])
        print("gap_rr_upper_final 1sigma = ", final_filtered_median['gap_rr_upper'] + final_filtered_median_err['gap_rr_upper'])

        print("\nres_lower_final 2sigma = ", final_filtered_median['res_lower'] - final_filtered_median_2sigmaerr['res_lower'])
        print("res_upper_final 2sigma = ", final_filtered_median['res_upper'] + final_filtered_median_2sigmaerr['res_upper'])
        print("\ngap_ll_lower_final 2sigma = ", final_filtered_median['gap_ll_lower'] - final_filtered_median_2sigmaerr['gap_ll_lower'])
        print("gap_ll_upper_final 2sigma = ", final_filtered_median['gap_ll_upper'] + final_filtered_median_2sigmaerr['gap_ll_upper'])
        print("\ngap_rr_lower_final 2sigma = ", final_filtered_median['gap_rr_lower'] - final_filtered_median_2sigmaerr['gap_rr_lower'])
        print("gap_rr_upper_final 2sigma = ", final_filtered_median['gap_rr_upper'] + final_filtered_median_2sigmaerr['gap_rr_upper'])

        # store metadata about storage
        self.store_boot=store_boot
        self.store_P=store_P
        self.database=database
        # store eff mass
        self.eff_mass=eff_mass
        self.eff_mass_err=eff_mass_err
        self.eff_mass_boot=eff_mass_boot
        self.theta_LR_k=theta_LR_k
        self.theta_LR_k_err=theta_LR_k_err
        self.theta_LR_k_boot=theta_LR_k_boot
        # store filtering info
        self.m_last=m_last
        self.eps_ZCW=eps_ZCW
        # store final results
        self.final_m=final_m
        self.m_window=m_window
        # final E
        self.final_E=final_filtered_median['E']
        self.final_E_err=final_filtered_median_err['E']
        self.final_E_boot=final_nbs_filtered_median['E']
        # final B
        self.final_B=final_filtered_median['B']
        self.final_B_err=final_filtered_median_err['B']
        self.final_B_boot=final_nbs_filtered_median['B']
        self.final_Bl=final_filtered_median['Bl']
        self.final_Bl_err=final_filtered_median_err['Bl']
        self.final_Bl_boot=final_nbs_filtered_median['Bl']
        self.final_Br=final_filtered_median['Br']
        self.final_Br_err=final_filtered_median_err['Br']
        self.final_Br_boot=final_nbs_filtered_median['Br']
        # final Z
        self.final_Zl=final_filtered_median['Zl']
        self.final_Zl_err=final_filtered_median_err['Zl']
        self.final_Zl_boot=final_nbs_filtered_median['Zl']
        self.final_Zr=final_filtered_median['Zr']
        self.final_Zr_err=final_filtered_median_err['Zr']
        self.final_Zr_boot=final_nbs_filtered_median['Zr']
        # final res bound
        self.final_res_lower=final_filtered_median['res_lower']
        self.final_res_lower_err=final_filtered_median_err['res_lower']
        self.final_res_lower_2sigmaerr=final_filtered_median_2sigmaerr['res_lower']
        self.final_res_lower_boot=final_nbs_filtered_median['res_lower']
        self.final_res_upper=final_filtered_median['res_upper']
        self.final_res_upper_err=final_filtered_median_err['res_upper']
        self.final_res_upper_2sigmaerr=final_filtered_median_2sigmaerr['res_upper']
        self.final_res_upper_boot=final_nbs_filtered_median['res_upper']
        # final gap bound
        self.final_gap_rr_lower=final_filtered_median['gap_rr_lower']
        self.final_gap_rr_lower_err=final_filtered_median_err['gap_rr_lower']
        self.final_gap_rr_lower_2sigmaerr=final_filtered_median_2sigmaerr['gap_rr_lower']
        self.final_gap_rr_lower_boot=final_nbs_filtered_median['gap_rr_lower']
        self.final_gap_rr_upper=final_filtered_median['gap_rr_upper']
        self.final_gap_rr_upper_2sigmaerr=final_filtered_median_2sigmaerr['gap_rr_upper']
        self.final_gap_rr_upper_err=final_filtered_median_err['gap_rr_upper']
        self.final_gap_rr_upper_boot=final_nbs_filtered_median['gap_rr_upper']
        self.final_gap_ll_lower=final_filtered_median['gap_ll_lower']
        self.final_gap_ll_lower_err=final_filtered_median_err['gap_ll_lower']
        self.final_gap_ll_lower_2sigmaerr=final_filtered_median_2sigmaerr['gap_ll_lower']
        self.final_gap_ll_lower_boot=final_nbs_filtered_median['gap_ll_lower']
        self.final_gap_ll_upper=final_filtered_median['gap_ll_upper']
        self.final_gap_ll_upper_2sigmaerr=final_filtered_median_2sigmaerr['gap_ll_upper']
        self.final_gap_ll_upper_err=final_filtered_median_err['gap_ll_upper']
        self.final_gap_ll_upper_boot=final_nbs_filtered_median['gap_ll_upper']
        # final srq
        self.final_srq_rr=final_filtered_median['srq_rr']
        self.final_srq_ll=final_filtered_median['srq_ll']
        self.final_srq_rr_err=final_filtered_median_err['srq_rr']
        self.final_srq_ll_err=final_filtered_median_err['srq_ll']
        self.final_srq_rr_boot=final_nbs_filtered_median['srq_rr']
        self.final_srq_ll_boot=final_nbs_filtered_median['srq_ll']
        # store bootstrap median results
        self.med_ev=np.array([all_filtered_median[m]['ev'] for m in all_m])
        self.med_theta_LR_y=np.array([all_filtered_median[m]['theta_LR_y'] for m in all_m])
        self.med_Zl=np.array([all_filtered_median[m]['Zl'] for m in all_m])
        self.med_Zr=np.array([all_filtered_median[m]['Zr'] for m in all_m])
        self.med_B=np.array([all_filtered_median[m]['B'] for m in all_m])
        self.med_Bl=np.array([all_filtered_median[m]['Bl'] for m in all_m])
        self.med_Br=np.array([all_filtered_median[m]['Br'] for m in all_m])
        self.med_E=np.array([all_filtered_median[m]['E'] for m in all_m])
        self.med_res_lower=np.array([all_filtered_median[m]['res_lower'] for m in all_m])
        self.med_res_upper=np.array([all_filtered_median[m]['res_upper'] for m in all_m])
        self.med_gap_ll_lower=np.array([all_filtered_median[m]['gap_ll_lower'] for m in all_m])
        self.med_gap_ll_upper=np.array([all_filtered_median[m]['gap_ll_upper'] for m in all_m])
        self.med_gap_ll_lower=np.array([all_filtered_median[m]['gap_ll_lower'] for m in all_m])
        self.med_gap_ll_upper=np.array([all_filtered_median[m]['gap_ll_upper'] for m in all_m])
        self.med_gap_rr_lower=np.array([all_filtered_median[m]['gap_rr_lower'] for m in all_m])
        self.med_gap_rr_upper=np.array([all_filtered_median[m]['gap_rr_upper'] for m in all_m])
        self.med_gap_rr_lower=np.array([all_filtered_median[m]['gap_rr_lower'] for m in all_m])
        self.med_gap_rr_upper=np.array([all_filtered_median[m]['gap_rr_upper'] for m in all_m])
        self.med_ev_err=np.array([all_filtered_median_err[m]['ev'] for m in all_m])
        self.med_theta_LR_y_err=np.array([all_filtered_median_err[m]['theta_LR_y'] for m in all_m])
        self.med_Zl_err=np.array([all_filtered_median_err[m]['Zl'] for m in all_m])
        self.med_Zr_err=np.array([all_filtered_median_err[m]['Zr'] for m in all_m])
        self.med_B_err=np.array([all_filtered_median_err[m]['B'] for m in all_m])
        self.med_Bl_err=np.array([all_filtered_median_err[m]['Bl'] for m in all_m])
        self.med_Br_err=np.array([all_filtered_median_err[m]['Br'] for m in all_m])
        self.med_E_err=np.array([all_filtered_median_err[m]['E'] for m in all_m])
        self.med_res_lower_err=np.array([all_filtered_median_err[m]['res_lower'] for m in all_m])
        self.med_res_upper_err=np.array([all_filtered_median_err[m]['res_upper'] for m in all_m])
        self.med_gap_ll_lower_err=np.array([all_filtered_median_err[m]['gap_ll_lower'] for m in all_m])
        self.med_gap_ll_upper_err=np.array([all_filtered_median_err[m]['gap_ll_upper'] for m in all_m])
        self.med_gap_rr_lower_err=np.array([all_filtered_median_err[m]['gap_rr_lower'] for m in all_m])
        self.med_gap_rr_upper_err=np.array([all_filtered_median_err[m]['gap_rr_upper'] for m in all_m])
        self.med_ev_boot=np.array([all_nbs_filtered_median[m]['ev'] for m in all_m])
        self.med_theta_LR_y_boot=np.array([all_nbs_filtered_median[m]['theta_LR_y'] for m in all_m])
        self.med_Zl_boot=np.array([all_nbs_filtered_median[m]['Zl'] for m in all_m])
        self.med_Zr_boot=np.array([all_nbs_filtered_median[m]['Zr'] for m in all_m])
        self.med_B_boot=np.array([all_nbs_filtered_median[m]['B'] for m in all_m])
        self.med_Bl_boot=np.array([all_nbs_filtered_median[m]['Bl'] for m in all_m])
        self.med_Br_boot=np.array([all_nbs_filtered_median[m]['Br'] for m in all_m])
        self.med_E_boot=np.array([all_nbs_filtered_median[m]['E'] for m in all_m])
        self.med_res_lower_boot=np.array([all_nbs_filtered_median[m]['res_lower'] for m in all_m])
        self.med_res_upper_boot=np.array([all_nbs_filtered_median[m]['res_upper'] for m in all_m])
        self.med_gap_ll_lower_boot=np.array([all_nbs_filtered_median[m]['gap_ll_lower'] for m in all_m])
        self.med_gap_ll_upper_boot=np.array([all_nbs_filtered_median[m]['gap_ll_upper'] for m in all_m])
        self.med_gap_rr_lower_boot=np.array([all_nbs_filtered_median[m]['gap_rr_lower'] for m in all_m])
        self.med_gap_rr_upper_boot=np.array([all_nbs_filtered_median[m]['gap_rr_upper'] for m in all_m])
        self.med_srq_rr=np.array([all_filtered_median[m]['srq_rr'] for m in all_m])
        self.med_srq_rr_err=np.array([all_filtered_median_err[m]['srq_rr'] for m in all_m])
        self.med_srq_rr_boot=np.array([all_nbs_filtered_median[m]['srq_rr'] for m in all_m])
        self.med_srq_ll=np.array([all_filtered_median[m]['srq_ll'] for m in all_m])
        self.med_srq_ll_err=np.array([all_filtered_median_err[m]['srq_ll'] for m in all_m])
        self.med_srq_ll_boot=np.array([all_nbs_filtered_median[m]['srq_ll'] for m in all_m])
        if store_boot:
            # store one-level bootstrap results
            self.bs_ev=np.array([all_filtered[m]['ev'] for m in all_m])
            self.bs_Zl=np.array([all_filtered[m]['Zl'] for m in all_m])
            self.bs_Zr=np.array([all_filtered[m]['Zr'] for m in all_m])
            self.bs_Bl=np.array([all_filtered[m]['Bl'] for m in all_m])
            self.bs_Br=np.array([all_filtered[m]['Br'] for m in all_m])
            self.bs_E=np.array([all_filtered[m]['E'] for m in all_m])
            # store inner bootstrap median results
            self.nbs_ev=np.array([all_nbs_filtered[m]['ev'] for m in all_m])
            self.nbs_Zl=np.array([all_nbs_filtered[m]['Zl'] for m in all_m])
            self.nbs_Zr=np.array([all_nbs_filtered[m]['Zr'] for m in all_m])
            self.nbs_Bl=np.array([all_nbs_filtered[m]['Bl'] for m in all_m])
            self.nbs_Br=np.array([all_nbs_filtered[m]['Br'] for m in all_m])
            self.nbs_E=np.array([all_nbs_filtered[m]['E'] for m in all_m])
        # store Ritz coefficients if desired
        if store_P:
            self.bs_Pl={ m: all_filtered[m]['Pl'] for m in all_m }
            self.nbs_Pl={ m: all_nbs_filtered[m]['Pl'] for m in all_m }
            self.bs_Pr={ m: all_filtered[m]['Pr'] for m in all_m }
            self.nbs_Pr={ m: all_nbs_filtered[m]['Pr'] for m in all_m }

        # save results to database if desired
        if database is not None:
            make_database(self, database)
