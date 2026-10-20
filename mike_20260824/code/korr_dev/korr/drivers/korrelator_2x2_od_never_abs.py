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


# wrap main function for single iteration of off-diag RR
from .basic import run_2x2_offdiag_gpof as run_2x2_offdiag_gpof_basic
def run_2x2_offdiag_gpof(Cab, m, log=np.log):
    analysis = run_2x2_offdiag_gpof_basic(Cab, m)

    # derived quantities
    ev = analysis['ev']
    Br, Bl = analysis['Br'], analysis['Bl']
    E = -log(ev)
    B = np.min(np.stack([Br, Bl]), axis=0)

    # residual bounds
    res_lower = -log(ev + np.sqrt(np.abs(B)))
    res_upper = -log(ev - np.sqrt(np.abs(B)))

    return dict(
        **analysis,
        E = E,
        res_lower=res_lower, res_upper=res_upper,
        B=B,
    )


def run_all_2x2_offdiag_gpof(bs_Cab):
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2

    all_m = range(1, m_max+1)
    analysis = {m : run_2x2_offdiag_gpof(bs_Cab, m=m) for m in all_m}

    return analysis # ~ {m : {ev : ..., Zr : ...}}

# compute gap bounds and add to dictionary
def get_gap_bounds(analysis, E_gap_bound=None):
    ev = analysis['ev']
    B = analysis['B']
    gap = np.zeros(ev.shape)
    if E_gap_bound is not None:
        gap = np.transpose(np.array([np.abs(ev[...,k] - np.exp(-E_gap_bound))
               for k in range(ev.shape[-1])]))
    elif ev.shape[-1] > 1:
        gap = np.zeros(ev.shape)
        for k in range(ev.shape[-1]):
            diffs = np.abs(ev - ev[...,k][...,None])
            gap[...,k] = np.sort(diffs, axis=-1)[...,1]
    gap_lower = -np.log(ev + B/gap)
    gap_upper = -np.log(ev - B/gap)

    # set gap bound to residual bound if gap zero
    gap_lower = np.where(~np.isfinite(gap_lower), analysis['res_lower'], gap_lower)
    gap_upper = np.where(~np.isfinite(gap_upper), analysis['res_upper'], gap_upper)

    analysis['gap_lower'] = gap_lower
    analysis['gap_upper'] = gap_upper

# save RR results to HDF5 database
def make_database(rr_analysis, database):
    with h5py.File(database+".h5", "w") as fout:
        for k in [
            # effective mass
            "eff_mass", "eff_mass_err", "eff_mass_boot",
            # final results
            "final_m", "final_E", "final_B", "final_Z",
            "final_E_err", "final_B_err", "final_Z_err",
            "final_E_boot", "final_B_boot", "final_Z_boot",
            "final_res_lower", "final_res_upper",
            "final_gap_lower", "final_gap_upper",
            "final_res_lower_err", "final_res_upper_err",
            "final_gap_lower_err", "final_gap_upper_err",
            "final_res_lower_boot", "final_res_upper_boot",
            "final_gap_lower_boot", "final_gap_upper_boot",
            # filtering info
            "m_last",
            "eps_ZCW",
            "m_window",
            # m-dependent results
            "med_ev", "med_Z", "med_B", "med_E",
            "med_res_lower", "med_res_upper",
            "med_gap_lower", "med_gap_upper",
            "med_ev_err", "med_Z_err", "med_B_err", "med_E_err",
            "med_ev_boot", "med_Z_boot", "med_B_boot", "med_E_boot",
            "med_res_lower_err", "med_res_upper_err",
            "med_gap_lower_err", "med_gap_upper_err",
            "med_res_lower_boot", "med_res_upper_boot",
            "med_gap_lower_boot", "med_gap_upper_boot",
        ]:
            fout.create_dataset(k, data=getattr(rr_analysis, k))

        if rr_analysis.store_P:
            all_m = np.array(list(rr_analysis.bs_Pl.keys()))
            for m in all_m:
                for k in ["bs_Pl", "nbs_Pl"]:
                    fout.create_dataset(k+"/m"+str(m), data=getattr(rr_analysis, k)[m])

        if rr_analysis.store_boot:
            for k in [
                "bs_ev", "bs_Z", "bs_B", "bs_E",
                "bs_res_lower", "bs_res_upper",
                "bs_gap_lower", "bs_gap_upper",
                "nbs_ev", "nbs_Z", "nbs_B", "nbs_E",
                "nbs_res_lower", "nbs_res_upper",
                "nbs_gap_lower", "nbs_gap_upper",
            ]:
                fout.create_dataset(k, data=getattr(rr_analysis, k))

class korrelator:
    def __init__(self,
        C, # correlator input ~ (Batch, t,[a,b])

        F_ZCW=10, # ZCW filtering hyperparameter
        eps_ZCW=None, # manually specify ZCW cutoff
        ignore_is_norm=False, # ignore norm trick?
        staggered=False,

        thresh=0.9, # fraction of boots that must contain physical state
        thresh_inner=None, # fraction of inner boots that must contain physical state, if different from thresh
        final_m=None, # manually specify iteration for final results
        m_window=3, # how many of last iterations to average over

        E_gap_bound=None, # first expected missing energy level for gap bound computation
        abs_B=False, # apply abs before median to B_k^{(m)}

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

        eff_mass = np.zeros((bdim, Nt-1))
        eff_mass_err = np.zeros((bdim, Nt-1))
        eff_mass_boot = np.zeros((bdim, N_inner, Nt-1))
        for a in range(bdim):
            eff_mass[a,:] = effective_mass(np.mean(C[:,:,a,a], axis=0))
            bs_eff_mass = bs_effective_mass(bs_C[:,:,a,a])
            eff_mass_boot[a,:,:] = bs_eff_mass
            for t in range(Nt-1):
                eff_mass_err[a,t] = make_boot_ci(bs_eff_mass[:,t])
            print("operator a=",a," effective mass")
            print(eff_mass[a,:], " +/- ", eff_mass_err[a,:])

        # range of iterations
        all_m = range(1,Nt//2+1)
        if final_m is None:
            final_m = all_m[-2]
        print("final_m = ", final_m)

        # run RR = Lanczos = Prony = GPOF
        print("starting RR analysis")
        all_analyses = run_all_2x2_offdiag_gpof(bs_C)

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

        # Hermitian subspace and ZCW filters
        print("applying HS and ZCW filters")
        all_filtered = {m : filter_offdiag_analysis(an, an['is_real'] & (an['ZCW'] > eps_ZCW)) for m,an in all_analyses.items()}

        # remove projectors to save memory if not needed
        if not store_P:
            for m in all_m:
                del all_filtered[m]['Pl']
                del all_filtered[m]['Pr']

        # compute gap bounds
        print("E_gap_bound = ", E_gap_bound)
        for m in all_m:
            get_gap_bounds(all_filtered[m], E_gap_bound=E_gap_bound)

        # compute one-level bootstrap median
        print("taking median")
        if thresh_inner is None:
            thresh_inner = thresh
        all_filtered_median = {
            m : {k : batched_median(v, thresh=thresh_inner) for k,v in an.items()}
            for m,an in all_filtered.items()
        }

        print("\nFinal iteration m =", final_m)
        print("Z_final = ", all_filtered_median[final_m]['Zl'])
        print("E_final = ", all_filtered_median[final_m]['E'])
        print("res_lower_final = ", all_filtered_median[final_m]['res_lower'])
        print("res_upper_final = ", all_filtered_median[final_m]['res_upper'])
        print("gap_lower_final = ", all_filtered_median[final_m]['gap_lower'])
        print("gap_upper_final = ", all_filtered_median[final_m]['gap_upper'])

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

            # Hermitian subspace and ZCW filters
            nbs_filtered = {m : filter_offdiag_analysis(an, an['is_real'] & (an['ZCW'] > nbs_eps_ZCW))
                            for m,an in nbs_analyses.items()}

            # remove projectors to save memory if not needed
            if not store_P:
                for m in all_m:
                    del(nbs_filtered[m]['Pl'])
                    del(nbs_filtered[m]['Pr'])

            # compute gap bounds
            for m in all_m:
                get_gap_bounds(nbs_filtered[m], E_gap_bound=E_gap_bound)

            # accumulate
            all_nbs_filtered.append(nbs_filtered)


        # build combined nested bootstrap dictionary
        all_nbs_filtered = {m : [d[m] for d in all_nbs_filtered] for m in all_m} # group by m
        all_nbs_filtered = {m : stack_dicts(v) for m,v in all_nbs_filtered.items()}

        print("taking inner median")
        all_nbs_filtered_median = {
            m : {k : inner_median(v, thresh=thresh, thresh_inner=thresh_inner) for k,v in an.items()}
            for m,an in all_nbs_filtered.items()
        }

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

        # take median of last few iterations
        final_filtered_median = final_iter_median(all_filtered_median, N_states=N_states, m_window=m_window)
        final_nbs_filtered_median = final_iter_median(all_nbs_filtered_median, N_states=N_states, m_window=m_window)

        # no negative median bound windows
        final_filtered_median['res_upper'] = np.max(np.array([final_filtered_median['res_upper'], final_filtered_median['E']]), axis=0)
        final_filtered_median['res_lower'] = np.min([final_filtered_median['res_lower'], final_filtered_median['E']], axis=0)
        final_filtered_median['gap_upper'] = np.max([final_filtered_median['gap_upper'], final_filtered_median['E']], axis=0)
        final_filtered_median['gap_lower'] = np.min([final_filtered_median['gap_lower'], final_filtered_median['E']], axis=0)

        print("computing confidence intervals")
        all_filtered_median_err = {
            m : {k : make_boot_ci(np.real(v)) for k,v in an.items()}
            for m,an in all_nbs_filtered_median.items()
        }
        final_filtered_median_err = {k : make_boot_ci(np.real(v)) for k,v in final_nbs_filtered_median.items()}

        print("\nFinal iteration m in [", final_m-m_window+1, ", ", final_m, "]")
        print("\nZ_final = ", final_filtered_median['Zl'], " +/- ", final_filtered_median_err['Zl'])
        print("\nB_final = ", final_filtered_median['Bl'], " +/- ", final_filtered_median_err['Bl'])
        print("\nE_final = ", final_filtered_median['E'], " +/- ", final_filtered_median_err['E'])
        print("\nres_lower_final = ", final_filtered_median['res_lower'] - final_filtered_median_err['res_lower'])
        print("res_upper_final = ", final_filtered_median['res_upper'] + final_filtered_median_err['res_upper'])
        print("\ngap_lower_final = ", final_filtered_median['gap_lower'] - final_filtered_median_err['gap_lower'])
        print("gap_upper_final = ", final_filtered_median['gap_upper'] + final_filtered_median_err['gap_upper'])

        # store metadata about storage
        self.store_boot=store_boot
        self.store_P=store_P
        self.database=database
        # store ff mass
        self.eff_mass=eff_mass
        self.eff_mass_err=eff_mass_err
        self.eff_mass_boot=eff_mass_boot
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
        self.final_B=final_filtered_median['Bl']
        self.final_B_err=final_filtered_median_err['Bl']
        self.final_B_boot=final_nbs_filtered_median['Bl']
        # final Z
        self.final_Z=final_filtered_median['Zl']
        self.final_Z_err=final_filtered_median_err['Zl']
        self.final_Z_boot=final_nbs_filtered_median['Zl']
        # final res bound
        self.final_res_lower=final_filtered_median['res_lower']
        self.final_res_lower_err=final_filtered_median_err['res_lower']
        self.final_res_lower_boot=final_nbs_filtered_median['res_lower']
        self.final_res_upper=final_filtered_median['res_upper']
        self.final_res_upper_err=final_filtered_median_err['res_upper']
        self.final_res_upper_boot=final_nbs_filtered_median['res_upper']
        # final gap bound
        self.final_gap_lower=final_filtered_median['gap_lower']
        self.final_gap_lower_err=final_filtered_median_err['gap_lower']
        self.final_gap_lower_boot=final_nbs_filtered_median['gap_lower']
        self.final_gap_upper=final_filtered_median['gap_upper']
        self.final_gap_upper_err=final_filtered_median_err['gap_upper']
        self.final_gap_upper_boot=final_nbs_filtered_median['gap_upper']
        # store bootstrap median results
        self.med_ev=np.array([all_filtered_median[m]['ev'] for m in all_m])
        self.med_Z=np.array([all_filtered_median[m]['Zl'] for m in all_m])
        self.med_B=np.array([all_filtered_median[m]['Bl'] for m in all_m])
        self.med_E=np.array([all_filtered_median[m]['E'] for m in all_m])
        self.med_res_lower=np.array([all_filtered_median[m]['res_lower'] for m in all_m])
        self.med_res_upper=np.array([all_filtered_median[m]['res_upper'] for m in all_m])
        self.med_gap_lower=np.array([all_filtered_median[m]['gap_lower'] for m in all_m])
        self.med_gap_upper=np.array([all_filtered_median[m]['gap_upper'] for m in all_m])
        self.med_ev_err=np.array([all_filtered_median_err[m]['ev'] for m in all_m])
        self.med_Z_err=np.array([all_filtered_median_err[m]['Zl'] for m in all_m])
        self.med_B_err=np.array([all_filtered_median_err[m]['Bl'] for m in all_m])
        self.med_E_err=np.array([all_filtered_median_err[m]['E'] for m in all_m])
        self.med_res_lower_err=np.array([all_filtered_median_err[m]['res_lower'] for m in all_m])
        self.med_res_upper_err=np.array([all_filtered_median_err[m]['res_upper'] for m in all_m])
        self.med_gap_lower_err=np.array([all_filtered_median_err[m]['gap_lower'] for m in all_m])
        self.med_gap_upper_err=np.array([all_filtered_median_err[m]['gap_upper'] for m in all_m])
        self.med_ev_boot=np.array([all_nbs_filtered_median[m]['ev'] for m in all_m])
        self.med_Z_boot=np.array([all_nbs_filtered_median[m]['Zl'] for m in all_m])
        self.med_B_boot=np.array([all_nbs_filtered_median[m]['Bl'] for m in all_m])
        self.med_E_boot=np.array([all_nbs_filtered_median[m]['E'] for m in all_m])
        self.med_res_lower_boot=np.array([all_nbs_filtered_median[m]['res_lower'] for m in all_m])
        self.med_res_upper_boot=np.array([all_nbs_filtered_median[m]['res_upper'] for m in all_m])
        self.med_gap_lower_boot=np.array([all_nbs_filtered_median[m]['gap_lower'] for m in all_m])
        self.med_gap_upper_boot=np.array([all_nbs_filtered_median[m]['gap_upper'] for m in all_m])
        if store_boot:
            # store one-level bootstrap results
            self.bs_ev=np.array([all_filtered[m]['ev'] for m in all_m])
            self.bs_Z=np.array([all_filtered[m]['Zl'] for m in all_m])
            self.bs_B=np.array([all_filtered[m]['Bl'] for m in all_m])
            self.bs_E=np.array([all_filtered[m]['E'] for m in all_m])
            self.bs_res_lower=np.array([all_filtered[m]['res_lower'] for m in all_m])
            self.bs_res_upper=np.array([all_filtered[m]['res_upper'] for m in all_m])
            self.bs_gap_lower=np.array([all_filtered[m]['gap_lower'] for m in all_m])
            self.bs_gap_upper=np.array([all_filtered[m]['gap_upper'] for m in all_m])
            # store inner bootstrap median results
            self.nbs_ev=np.array([all_nbs_filtered[m]['ev'] for m in all_m])
            self.nbs_Z=np.array([all_nbs_filtered[m]['Zl'] for m in all_m])
            self.nbs_B=np.array([all_nbs_filtered[m]['Bl'] for m in all_m])
            self.nbs_E=np.array([all_nbs_filtered[m]['E'] for m in all_m])
            self.nbs_res_lower=np.array([all_nbs_filtered[m]['res_lower'] for m in all_m])
            self.nbs_res_upper=np.array([all_nbs_filtered[m]['res_upper'] for m in all_m])
            self.nbs_gap_lower=np.array([all_nbs_filtered[m]['gap_lower'] for m in all_m])
            self.nbs_gap_upper=np.array([all_nbs_filtered[m]['gap_upper'] for m in all_m])
        # store Ritz coefficients if desired
        if store_P:
            self.bs_Pl={ m: all_filtered[m]['Pl'] for m in all_m }
            self.nbs_Pl={ m: all_nbs_filtered[m]['Pl'] for m in all_m }
            self.bs_Pr={ m: all_filtered[m]['Pr'] for m in all_m }
            self.nbs_Pr={ m: all_nbs_filtered[m]['Pr'] for m in all_m }

        # save results to database if desired
        if database is not None:
            make_database(self, database)
