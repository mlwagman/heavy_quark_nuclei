import numpy as np
import h5py

from ..util import isreal
from ..util import make_block_hankel, stack_dicts
from ..util import effective_mass, bs_effective_mass

from ..eig_util import eig

from ..bootstrap import make_bs_tensor, make_nbs_tensor

from ..outlier import batched_median, inner_median, make_boot_ci

from ..filter import filter_block_analysis

from ..label import compute_label_dot, label_states_argmax, label_block_analysis

from .korrelator import run_all_block_rgpof

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
            # label info
            "m_label",
            # m-dependent results
            "med_ev", "med_Z", "med_B", "med_E",
            "med_res_lower", "med_res_upper",
            "med_gap_lower", "med_gap_upper",
            "med_ev_err", "med_Z_err", "med_B_err", "med_E_err",
            "med_res_lower_err", "med_res_upper_err",
            "med_gap_lower_err", "med_gap_upper_err",
            "med_ev_boot", "med_Z_boot", "med_B_boot", "med_E_boot",
            "med_res_lower_boot", "med_res_upper_boot",
            "med_gap_lower_boot", "med_gap_upper_boot",
            # correlated difference quantites
            "eff_mass_diff", "eff_mass_diff_err",
            "final_E_diff", "final_E_diff_err", "final_E_diff_boot",
            "final_gap_lower_diff", "final_gap_upper_diff",
            "final_gap_lower_diff_err", "final_gap_upper_diff_err",
            "final_gap_lower_diff_boot", "final_gap_upper_diff_boot",
            "med_E_diff", "med_E_diff_err",
            "med_gap_lower_diff", "med_gap_upper_diff",
            "med_gap_lower_diff_err", "med_gap_upper_diff_err",
            "med_E_diff_boot",
            "med_gap_lower_diff_boot", "med_gap_upper_diff_boot",
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
                # correlated difference quantities
                "bs_E_diff", 
                "bs_gap_lower_diff", "bs_gap_upper_diff",
                "nbs_E_diff", 
                "nbs_gap_lower_diff", "nbs_gap_upper_diff"
            ]:
                fout.create_dataset(k, data=getattr(rr_analysis, k))

class korrelated_difference:
    def __init__(self,
        C, # correlator input ~ (Batch, t,[a,b])
        C_2, # correlator input ~ (Batch, t,[a,b])
        coefficient, # compute E - coefficient*E2

        herm_thresh=0.99, # for adaptive m_label choice
        m_label_override=None, # overrides adaptive choice

        thresh=0.95, # fraction of boots that must contain physical state
        thresh_inner=None, # fraction of inner boots that must contain physical state, if different from thresh

        N_outer=200, # bootstrap confidence interval hyperparameter
        N_inner=200, # bootstrap median hyperparameter
        bs_C=None,  # manually pass outer bootstrap tensor
        nbs_C=None, # manually pass inner bootstrap tensor
        bs_C_2=None,  # manually pass outer bootstrap tensor
        nbs_C_2=None, # manually pass inner bootstrap tensor

        E_gap_bound=None, # first expected missing energy level for gap bound computation
        E_gap_bound_2=None, # first expected missing energy level for gap bound computation
        abs_B=True, # apply abs before median to B_k^{(m)}
        ignore_is_norm=False, # ignore norm trick?
        final_m=None, # manually specify iteration for final results

        database=None, # database to save results
        store_boot=False, # save inner bootstrap Ritz values?
        store_P=False, # save Ritz rotator coefficients?

        progress=True, # display progress bar?

    ):
        # seed for reproducibility / correlations
        np.random.seed(17)

        # check correlator dimensions are valid for scalar or block
        assert len(C.shape) in (2, 4)
        self.N_samps = N_samps = C.shape[0]
        self.Nt = Nt = C.shape[1]
        # blockify
        if len(C.shape) == 2:
            C = C[..., None,None] # it -> itab
            self.bdim = bdim = 1
        else:
            self.bdim = bdim = C.shape[-1]
        assert C.shape == (N_samps, Nt, bdim, bdim)

        assert len(C_2.shape) in (2, 4)
        # blockify
        if len(C.shape) == 2:
            C_2 = C_2[..., None,None] # it -> itab
            self.bdim2 = bdim2 = 1
        else:
            self.bdim2 = bdim2 = C_2.shape[-1]
        print(C_2.shape)
        print((N_samps, Nt, bdim2, bdim2))
        assert C_2.shape == (N_samps, Nt, bdim2, bdim2)

        print("\n"+"=" * 100)
        print("Starting labeled Rayleigh-Ritz correlator analysis")
        print("\nObtaining Ritz values with RR = block Lanczos = block Prony = GPOF")
        if m_label_override is not None:
            print("\nLabeling with m_label = ", m_label_override)
        else:
            print("\nLabeling with adaptive m_label, hyperparameter herm_thresh = ", herm_thresh)
        print("\nBlock correlator shape = ", C.shape)
        print("\nBlock correlator 2 shape = ", C_2.shape)
        print("=" * 100 + "\n")

        # make booted corrs (both levels of booting)
        if bs_C is None:
            bs_tensor = make_bs_tensor(N_samps=N_samps, N_boots=N_inner)
            bs_C = np.einsum('bn,n...->b...', bs_tensor, C)
        elif bdim == 1:
            N_inner = bs_C.shape[0]
            bs_C = bs_C[..., None,None] # it -> itab
        if nbs_C is None:
            nbs_tensor = make_nbs_tensor(N_samps=N_samps, N_outer=N_outer, N_inner=N_inner)
            nbs_C = np.einsum('oin,n...->oi...', nbs_tensor, C)
        elif bdim == 1:
            N_outer = nbs_C.shape[0]
            nbs_C = nbs_C[..., None,None] # it -> itab
        assert bs_C.shape == (N_inner, Nt, bdim, bdim)
        assert nbs_C.shape == (N_outer, N_inner, Nt, bdim, bdim)


        # same seed for correlated difference
        np.random.seed(17)

        if bs_C_2 is None:
            bs_tensor = make_bs_tensor(N_samps=N_samps, N_boots=N_inner)
            bs_C_2 = np.einsum('bn,n...->b...', bs_tensor, C_2)
        elif bdim == 1:
            N_inner = bs_C_2.shape[0]
            bs_C_2 = bs_C_2[..., None,None] # it -> itab
        if nbs_C_2 is None:
            nbs_tensor = make_nbs_tensor(N_samps=N_samps, N_outer=N_outer, N_inner=N_inner)
            nbs_C_2 = np.einsum('oin,n...->oi...', nbs_tensor, C_2)
        elif bdim == 1:
            N_outer = nbs_C_2.shape[0]
            nbs_C_2 = nbs_C_2[..., None,None] # it -> itab
        assert bs_C_2.shape == (N_inner, Nt, bdim2, bdim2)
        assert nbs_C_2.shape == (N_outer, N_inner, Nt, bdim2, bdim2)

        print("bootstrap resampled data")
        print("bootstrap shape = ", bs_C.shape)
        print("bootstrap 2 shape = ", bs_C_2.shape)
        print("nested bootstrap shape = ", nbs_C.shape)
        print("nested bootstrap 2 shape = ", nbs_C_2.shape)

        eff_mass = np.zeros((bdim, Nt-1))
        eff_mass_err = np.zeros((bdim, Nt-1))
        eff_mass_boot = np.zeros((bdim, N_inner, Nt-1))
        for a in range(bdim):
            eff_mass[a,:] = effective_mass(np.mean(C[:,:,a,a], axis=0))
            bs_eff_mass = bs_effective_mass(bs_C[:,:,a,a])
            eff_mass_boot[a,:,:] = bs_eff_mass
            for t in range(Nt-1):
                eff_mass_err[a,t] = make_boot_ci(bs_eff_mass[:,t])

        eff_mass_2 = np.zeros((bdim, Nt-1))
        eff_mass_2_err = np.zeros((bdim, Nt-1))
        eff_mass_2_boot = np.zeros((bdim, N_inner, Nt-1))
        for a in range(bdim):
            eff_mass_2[a,:] = effective_mass(np.mean(C_2[:,:,0,0], axis=0))
            bs_eff_mass_2 = bs_effective_mass(bs_C_2[:,:,0,0])
            eff_mass_2_boot[a,:,:] = bs_eff_mass_2
            for t in range(Nt-1):
                eff_mass_2_err[a,t] = make_boot_ci(bs_eff_mass_2[:,t])

        eff_mass_diff = np.zeros((bdim, Nt-1))
        eff_mass_diff_err = np.zeros((bdim, Nt-1))
        eff_mass_diff_boot = np.zeros((bdim, N_inner, Nt-1))
        for a in range(bdim):
            eff_mass_diff[a,:] = effective_mass(np.mean(C[:,:,a,a], axis=0))
            bs_eff_mass_diff = bs_effective_mass(bs_C[:,:,a,a])
            eff_mass_diff[a,:] -= coefficient * effective_mass(np.mean(C_2[:,:,0,0], axis=0))
            bs_eff_mass_diff -= coefficient * bs_effective_mass(bs_C_2[:,:,0,0])
            eff_mass_diff_boot[a,:,:] = bs_eff_mass_diff
            for t in range(Nt-1):
                eff_mass_diff_err[a,t] = make_boot_ci(bs_eff_mass_diff[:,t])
            print("operator a=",a," effective mass correlated difference")
            print(eff_mass_diff[a,:], " +/- ", eff_mass_diff_err[a,:])

        # range of iterations
        all_m = range(1,Nt//2+1)
        if final_m == None:
            final_m = all_m[-2]

        # run RR = Lanczos = Prony = GPOF
        print("starting RR analysis")
        all_analyses = run_all_block_rgpof(bs_C, 
                                           abs_B=abs_B, 
                                           ignore_is_norm=ignore_is_norm,
                                           E_gap_bound=E_gap_bound)
        all_analyses_2 = run_all_block_rgpof(bs_C_2, 
                                           abs_B=abs_B, 
                                           ignore_is_norm=ignore_is_norm,
                                           E_gap_bound=E_gap_bound_2)

        # find m_label for bs
        def _find_max_herm_m(all_analyses):
            all_m = np.array(list(all_analyses.keys()))
            is_herm = np.array([an['is_herm'].mean() > herm_thresh for an in all_analyses.values()])
            return np.max(all_m[is_herm])

        m_label = _find_max_herm_m(all_analyses) if m_label_override is None else m_label_override
        m_label_2 = _find_max_herm_m(all_analyses_2) if m_label_override is None else m_label_override
        print("m_label =", m_label)
        print("m_label_2 =", m_label_2)

        # Hermitian subspace filter
        print("applying HS filter")
        all_filtered = {m : filter_block_analysis(an, an['is_herm']) for m,an in all_analyses.items()}
        all_filtered_2 = {m : filter_block_analysis(an, an['is_herm']) for m,an in all_analyses_2.items()}

        # label states based on overlap with iteration m_label
        print("labeling bs analyses")
        Pl_label = all_analyses[m_label]['Pl']
        all_labeled = {
            m : label_block_analysis(an, label_states_argmax(compute_label_dot(bs_C, Pl_label, an['Pr'])))
            for m,an in all_filtered.items()
        }
        Pl_label = all_analyses_2[m_label_2]['Pl']
        all_labeled_2 = {
            m : label_block_analysis(an, label_states_argmax(compute_label_dot(bs_C_2, Pl_label, an['Pr'])))
            for m,an in all_filtered_2.items()
        }

        # remove projectors to save memory if not needed
        if not store_P:
            for m in all_m:
                del(all_labeled[m]['Pl']) 
                del(all_labeled[m]['Pr'])
                del(all_labeled_2[m]['Pl']) 
                del(all_labeled_2[m]['Pr'])

        # compute correlated difference
        for m in all_m:
            all_labeled[m]['E_diff'] = all_labeled[m]['E'] - coefficient*all_labeled_2[m]['E'][:,0][:,None]
            all_labeled[m]['gap_lower_diff'] = all_labeled[m]['gap_lower'] - all_labeled[m]['E'] + all_labeled[m]['E_diff']
            all_labeled[m]['gap_upper_diff'] = -1*( all_labeled_2[m]['gap_lower'][:,0][:,None] - all_labeled_2[m]['E'][:,0][:,None] ) + all_labeled[m]['E_diff']

        # compute one-level bootstrap median
        print("taking median")
        if thresh_inner == None:
            thresh_inner = thresh
        all_labeled_median = {
            m : {k : batched_median(v, thresh=thresh_inner) for k,v in an.items()}
            for m,an in all_labeled.items()
        }

        print("Z_final = ", all_labeled_median[final_m]['Zl'])
        print("E_final = ", all_labeled_median[final_m]['E'])
        print("res_lower_final = ", all_labeled_median[final_m]['res_lower'])
        print("res_upper_final = ", all_labeled_median[final_m]['res_upper'])
        print("gap_lower_final = ", all_labeled_median[final_m]['gap_lower'])
        print("gap_upper_final = ", all_labeled_median[final_m]['gap_upper'])
        print("E_diff_final = ", all_labeled_median[final_m]['E_diff'])
        print("gap_lower_diff_final = ", all_labeled_median[final_m]['gap_lower_diff'])
        print("gap_upper_diff_final = ", all_labeled_median[final_m]['gap_upper_diff'])


        # run nested bootstrap RR
        print("starting nested bootstrap RR analysis")
        all_nbs_labeled = []
        all_nbs_labeled_2 = []
        # split over outer bootstrap
        if progress:
            from tqdm import tqdm
        for b in (tqdm(range(N_outer)) if progress else range(N_outer)):
            # RR for each outer 
            nbs_analyses = run_all_block_rgpof(nbs_C[b], 
                                           abs_B=abs_B, 
                                           ignore_is_norm=ignore_is_norm,
                                           E_gap_bound=E_gap_bound)
            nbs_analyses_2 = run_all_block_rgpof(nbs_C_2[b], 
                                           abs_B=abs_B, 
                                           ignore_is_norm=ignore_is_norm,
                                           E_gap_bound=E_gap_bound_2)

            nbs_m_label = _find_max_herm_m(nbs_analyses) if m_label_override is None else m_label_override
            if nbs_m_label != m_label:
                print("outer boot ", b, " has nbs_m_label = ", nbs_m_label)
            nbs_m_label_2 = _find_max_herm_m(nbs_analyses_2) if m_label_override is None else m_label_override
            if nbs_m_label_2 != m_label_2:
                print("outer boot ", b, " has nbs_m_label_2 = ", nbs_m_label_2)

            # Hermitian subspace filter
            nbs_filtered = {m : filter_block_analysis(an, an['is_herm']) for m,an in nbs_analyses.items()}
            nbs_filtered_2 = {m : filter_block_analysis(an, an['is_herm']) for m,an in nbs_analyses_2.items()}

            # label states based on overlap with iteration m_label
            Pl_label = nbs_analyses[nbs_m_label]['Pl']
            nbs_labeled = {
                m : label_block_analysis(an, label_states_argmax(compute_label_dot(nbs_C[b], Pl_label, an['Pr'])))
                for m,an in nbs_filtered.items()
            }
            Pl_label = nbs_analyses_2[nbs_m_label_2]['Pl']
            nbs_labeled_2 = {
                m : label_block_analysis(an, label_states_argmax(compute_label_dot(nbs_C_2[b], Pl_label, an['Pr'])))
                for m,an in nbs_filtered_2.items()
            }

            # remove projectors to save memory if not needed
            if not store_P:
                for m in all_m:
                    del(nbs_labeled[m]['Pl'])
                    del(nbs_labeled[m]['Pr'])
                    del(nbs_labeled_2[m]['Pl'])
                    del(nbs_labeled_2[m]['Pr'])

            # compute correlated difference
            for m in all_m:
                nbs_labeled[m]['E_diff'] = nbs_labeled[m]['E'] - coefficient*nbs_labeled_2[m]['E'][:,0][:,None]
                nbs_labeled[m]['gap_lower_diff'] = nbs_labeled[m]['gap_lower'] - nbs_labeled[m]['E'] + nbs_labeled[m]['E_diff']
                nbs_labeled[m]['gap_upper_diff'] = -1*( nbs_labeled_2[m]['gap_lower'][:,0][:,None] - nbs_labeled_2[m]['E'][:,0][:,None] ) + nbs_labeled[m]['E_diff']

            # accumulate
            all_nbs_labeled.append(nbs_labeled)

        # build combined nested bootstrap dictionary
        all_nbs_labeled = {m : [d[m] for d in all_nbs_labeled] for m in all_m} # group by m
        all_nbs_labeled = {m : stack_dicts(v) for m,v in all_nbs_labeled.items()}

        print("taking inner median")
        all_nbs_labeled_median = {
            m : {k : inner_median(v, thresh=thresh, thresh_inner=thresh_inner) for k,v in an.items()}
            for m,an in all_nbs_labeled.items()
        }

        print("computing confidence intervals")
        all_labeled_median_err = {
            m : {k : make_boot_ci(np.real(v)) for k,v in an.items()}
            for m,an in all_nbs_labeled_median.items()
        }

        print("\nZ_final = ", all_labeled_median[final_m]['Zl'], " +/- ", all_labeled_median_err[final_m]['Zl'])
        print("\nE_final = ", all_labeled_median[final_m]['E'], " +/- ", all_labeled_median_err[final_m]['E'])
        print("\nres_lower_final = ", all_labeled_median[final_m]['res_lower'], " +/- ", all_labeled_median_err[final_m]['res_lower'])
        print("res_upper_final = ", all_labeled_median[final_m]['res_upper'], " +/- ", all_labeled_median_err[final_m]['res_upper'])
        print("\ngap_lower_final = ", all_labeled_median[final_m]['gap_lower'], " +/- ", all_labeled_median_err[final_m]['gap_lower'])
        print("gap_upper_final = ", all_labeled_median[final_m]['gap_upper'], " +/- ", all_labeled_median_err[final_m]['gap_upper'])

        print("\nE_diff_final = ", all_labeled_median[final_m]['E_diff'], " +/- ", all_labeled_median_err[final_m]['E_diff'])
        print("\ngap_lower_diff_final = ", all_labeled_median[final_m]['gap_lower_diff'], " +/- ", all_labeled_median_err[final_m]['gap_lower_diff'])
        print("gap_upper_diff_final = ", all_labeled_median[final_m]['gap_upper_diff'], " +/- ", all_labeled_median_err[final_m]['gap_upper_diff'])

        # store metadata about storage
        self.store_boot=store_boot
        self.store_P=store_P
        self.database=database
        # store eff mass
        self.eff_mass=eff_mass
        self.eff_mass_err=eff_mass_err
        self.eff_mass_diff=eff_mass_diff
        self.eff_mass_diff_err=eff_mass_diff_err
        self.eff_mass_boot=eff_mass_boot
        # store filtering info
        self.m_label=m_label
        # store final results
        self.final_m=final_m
        # final E
        self.final_E=all_labeled_median[final_m]['E']
        self.final_E_err=all_labeled_median_err[final_m]['E']
        self.final_E_boot=all_nbs_labeled_median[final_m]['E']
        # final B
        self.final_B=all_labeled_median[final_m]['Bl']
        self.final_B_err=all_labeled_median_err[final_m]['Bl']
        self.final_B_boot=all_nbs_labeled_median[final_m]['E']
        # final Z
        self.final_Z=all_labeled_median[final_m]['Zl']
        self.final_Z_err=all_labeled_median_err[final_m]['Zl']
        self.final_Z_boot=all_nbs_labeled_median[final_m]['Zl']
        # final res bound
        self.final_res_lower=all_labeled_median[final_m]['res_lower']
        self.final_res_lower_err=all_labeled_median_err[final_m]['res_lower']
        self.final_res_lower_boot=all_nbs_labeled_median[final_m]['res_lower']
        self.final_res_upper=all_labeled_median[final_m]['res_upper']
        self.final_res_upper_err=all_labeled_median_err[final_m]['res_upper']
        self.final_res_upper_boot=all_nbs_labeled_median[final_m]['res_upper']
        # final gap bound
        self.final_gap_lower=all_labeled_median[final_m]['gap_lower']
        self.final_gap_lower_err=all_labeled_median_err[final_m]['gap_lower']
        self.final_gap_lower_boot=all_nbs_labeled_median[final_m]['gap_lower']
        self.final_gap_upper=all_labeled_median[final_m]['gap_upper']
        self.final_gap_upper_err=all_labeled_median_err[final_m]['gap_upper']
        self.final_gap_upper_boot=all_nbs_labeled_median[final_m]['gap_upper']
        # correlated difference
        self.final_E_diff=all_labeled_median[final_m]['E_diff']
        self.final_E_diff_err=all_labeled_median_err[final_m]['E_diff']
        self.final_E_diff_boot=all_nbs_labeled_median[final_m]['E_diff']
        self.final_gap_lower_diff=all_labeled_median[final_m]['gap_lower_diff']
        self.final_gap_lower_diff_err=all_labeled_median_err[final_m]['gap_lower_diff']
        self.final_gap_lower_diff_boot=all_nbs_labeled_median[final_m]['gap_lower_diff']
        self.final_gap_upper_diff=all_labeled_median[final_m]['gap_upper_diff']
        self.final_gap_upper_diff_err=all_labeled_median_err[final_m]['gap_upper_diff']
        self.final_gap_upper_diff_boot=all_nbs_labeled_median[final_m]['gap_upper_diff']
        # store bootstrap median results
        self.med_ev=np.array([all_labeled_median[m]['ev'] for m in all_m])
        self.med_Z=np.array([all_labeled_median[m]['Zl'] for m in all_m])
        self.med_B=np.array([all_labeled_median[m]['Bl'] for m in all_m])
        self.med_E=np.array([all_labeled_median[m]['E'] for m in all_m])
        self.med_res_lower=np.array([all_labeled_median[m]['res_lower'] for m in all_m])
        self.med_res_upper=np.array([all_labeled_median[m]['res_upper'] for m in all_m])
        self.med_gap_lower=np.array([all_labeled_median[m]['gap_lower'] for m in all_m])
        self.med_gap_upper=np.array([all_labeled_median[m]['gap_upper'] for m in all_m])
        self.med_ev_err=np.array([all_labeled_median_err[m]['ev'] for m in all_m])
        self.med_Z_err=np.array([all_labeled_median_err[m]['Zl'] for m in all_m])
        self.med_B_err=np.array([all_labeled_median_err[m]['Bl'] for m in all_m])
        self.med_E_err=np.array([all_labeled_median_err[m]['E'] for m in all_m])
        self.med_res_lower_err=np.array([all_labeled_median_err[m]['res_lower'] for m in all_m])
        self.med_res_upper_err=np.array([all_labeled_median_err[m]['res_upper'] for m in all_m])
        self.med_gap_lower_err=np.array([all_labeled_median_err[m]['gap_lower'] for m in all_m])
        self.med_gap_upper_err=np.array([all_labeled_median_err[m]['gap_upper'] for m in all_m])
        self.med_ev_boot=np.array([all_nbs_labeled_median[m]['ev'] for m in all_m])
        self.med_Z_boot=np.array([all_nbs_labeled_median[m]['Zl'] for m in all_m])
        self.med_B_boot=np.array([all_nbs_labeled_median[m]['Bl'] for m in all_m])
        self.med_E_boot=np.array([all_nbs_labeled_median[m]['E'] for m in all_m])
        self.med_res_lower_boot=np.array([all_nbs_labeled_median[m]['res_lower'] for m in all_m])
        self.med_res_upper_boot=np.array([all_nbs_labeled_median[m]['res_upper'] for m in all_m])
        self.med_gap_lower_boot=np.array([all_nbs_labeled_median[m]['gap_lower'] for m in all_m])
        self.med_gap_upper_boot=np.array([all_nbs_labeled_median[m]['gap_upper'] for m in all_m])
        # correlated difference
        self.med_E_diff=np.array([all_labeled_median[m]['E_diff'] for m in all_m])
        self.med_E_diff_err=np.array([all_labeled_median_err[m]['E_diff'] for m in all_m])
        self.med_gap_lower_diff=np.array([all_labeled_median[m]['gap_lower_diff'] for m in all_m])
        self.med_gap_upper_diff=np.array([all_labeled_median[m]['gap_upper_diff'] for m in all_m])
        self.med_gap_lower_diff_err=np.array([all_labeled_median_err[m]['gap_lower_diff'] for m in all_m])
        self.med_gap_upper_diff_err=np.array([all_labeled_median_err[m]['gap_upper_diff'] for m in all_m])
        self.med_E_diff_boot=np.array([all_nbs_labeled_median[m]['E_diff'] for m in all_m])
        self.med_gap_lower_diff_boot=np.array([all_nbs_labeled_median[m]['gap_lower_diff'] for m in all_m])
        self.med_gap_upper_diff_boot=np.array([all_nbs_labeled_median[m]['gap_upper_diff'] for m in all_m])
        if store_boot:
            # store one-level bootstrap results
            self.bs_ev=np.array([all_labeled[m]['ev'] for m in all_m])
            self.bs_Z=np.array([all_labeled[m]['Zl'] for m in all_m])
            self.bs_B=np.array([all_labeled[m]['Bl'] for m in all_m])
            self.bs_E=np.array([all_labeled[m]['E'] for m in all_m])
            self.bs_res_lower=np.array([all_labeled[m]['res_lower'] for m in all_m])
            self.bs_res_upper=np.array([all_labeled[m]['res_upper'] for m in all_m])
            self.bs_gap_lower=np.array([all_labeled[m]['gap_lower'] for m in all_m])
            self.bs_gap_upper=np.array([all_labeled[m]['gap_upper'] for m in all_m])
            # store inner bootstrap median results
            self.nbs_ev=np.array([all_nbs_labeled[m]['ev'] for m in all_m])
            self.nbs_Z=np.array([all_nbs_labeled[m]['Zl'] for m in all_m])
            self.nbs_B=np.array([all_nbs_labeled[m]['Bl'] for m in all_m])
            self.nbs_E=np.array([all_nbs_labeled[m]['E'] for m in all_m])
            self.nbs_res_lower=np.array([all_nbs_labeled[m]['res_lower'] for m in all_m])
            self.nbs_res_upper=np.array([all_nbs_labeled[m]['res_upper'] for m in all_m])
            self.nbs_gap_lower=np.array([all_nbs_labeled[m]['gap_lower'] for m in all_m])
            self.nbs_gap_upper=np.array([all_nbs_labeled[m]['gap_upper'] for m in all_m])
            # correlated difference
            self.bs_E_diff=np.array([all_labeled[m]['E_diff'] for m in all_m])
            self.bs_gap_lower_diff=np.array([all_labeled[m]['gap_lower_diff'] for m in all_m])
            self.bs_gap_upper_diff=np.array([all_labeled[m]['gap_upper_diff'] for m in all_m])
            self.nbs_E_diff=np.array([all_nbs_labeled[m]['E_diff'] for m in all_m])
            self.nbs_gap_lower_diff=np.array([all_nbs_labeled[m]['gap_lower_diff'] for m in all_m])
            self.nbs_gap_upper_diff=np.array([all_nbs_labeled[m]['gap_upper_diff'] for m in all_m])
        # store Ritz coefficients if desired
        if store_P:
            self.bs_Pl={ m: all_labeled[m]['Pl'] for m in all_m }
            self.nbs_Pl={ m: all_nbs_labeled[m]['Pl'] for m in all_m }

        # save results to database if desired
        if database is not None:
            make_database(self, database)
