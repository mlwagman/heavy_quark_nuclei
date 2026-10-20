import numpy as np
import h5py

from ..util import isreal
from ..util import make_block_hankel, stack_dicts
from ..util import effective_mass, bs_effective_mass

from ..eig_util import eig

from ..bootstrap import make_bs_tensor, make_nbs_tensor

from ..rgpof import diagonalize_rgpof, make_block_rgpof_ritz_coeffs, make_block_rgpof_overlaps

from ..orr import make_block_rr_bounds_PrPl

from ..outlier import batched_median, inner_median, make_boot_ci

from ..filter import filter_block_analysis, compute_block_ZCW

from ..label import compute_label_dot, label_states_argmax, label_block_analysis

# main function for single iteration of RR = Lanczos = Prony = GPOF
def run_block_rgpof(bs_Cab, m,
    eig=eig, inv=np.linalg.inv,
    real=np.real, abs=np.abs, isclose=np.isclose,
    abs_B=True, ignore_is_norm=False, E_gap_bound=None
):
    *batch, Nt, bdim,_ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim,bdim)

    Gab = make_block_hankel(bs_Cab, 0)[..., :m, :, :m, :]
    Mab = make_block_hankel(bs_Cab, 1)[..., :m, :, :m, :]
    d = m*bdim

    G,M = Gab.reshape(*batch, d,d), Mab.reshape(*batch, d,d)
    ev, evec = diagonalize_rgpof(G, M, eig=eig, inv=inv)

    Zr, Zl, Nk, is_norm = make_block_rgpof_overlaps(evec, bs_Cab,
        inv=inv, abs=abs, real=real, isclose=isclose)
    ZZ = Zr.conj() * Zl # diagonal amplitude Zr*_ka Zl_ka
    is_herm = isreal(ev)  & isreal(ZZ).all(-1)
    if not ignore_is_norm:
        is_herm = is_herm & is_norm & (ZZ.real > 0).all(-1)

    ZCW = compute_block_ZCW(Zr, Zl, bs_Cab)

    Pr, Pl = make_block_rgpof_ritz_coeffs(evec, Nk, Gab)

    if 2*m < Nt:
        Br, Bl = make_block_rr_bounds_PrPl(bs_Cab, ev, Pr, Pl)
        if abs_B:
            Br = abs(Br)
            Bl = abs(Bl)
    else:
        Br = np.nan * np.zeros_like(ev)
        Bl = np.nan * np.zeros_like(ev)

    # derived quantities
    E = -np.log(ev)
    B = np.min(np.array([np.abs(Br), np.abs(Bl)]), axis=0)

    # residual bounds
    res_lower = -np.log(ev + np.sqrt(B))
    res_upper = -np.log(ev - np.sqrt(B))

    # gap bounds
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


    return dict(
        ev=ev, #evec=evec,
        Zr=Zr, Zl=Zl,
        is_herm=is_herm,
        ZCW=ZCW,
        Br=Br, Bl=Bl,
        Pr=Pr, Pl=Pl,
        E = E,
        res_lower=res_lower, res_upper=res_upper,
        gap_lower=gap_lower, gap_upper=gap_upper
    )

def run_all_block_rgpof(bs_Cab, 
                        abs_B=True, ignore_is_norm=False, E_gap_bound=None):
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2

    all_m = range(1, m_max+1)

    analysis = {}
    for m in all_m:
        danalysis = run_block_rgpof(bs_Cab, m=m, 
                                    abs_B=abs_B, 
                                    ignore_is_norm=ignore_is_norm,
                                    E_gap_bound=E_gap_bound)
        analysis[m] = danalysis

    return analysis # ~ {m : {ev : ..., Zr : ...}}

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
        filter_herm=True, # apply Hermitian subspace filter

        thresh=0.95, # fraction of boots that must contain physical state
        thresh_inner=None, # fraction of inner boots that must contain physical state, if different from thresh

        seed=17, # seed for bs tensors
        N_outer=200, # bootstrap confidence interval hyperparameter
        N_inner=200, # bootstrap median hyperparameter
        bs_C=None,  # manually pass outer bootstrap tensor
        nbs_C=None, # manually pass inner bootstrap tensor

        E_gap_bound=None, # first expected missing energy level for gap bound computation
        abs_B=True, # apply abs before median to B_k^{(m)}
        ignore_is_norm=False, # ignore norm trick?
        final_m=None, # manually specify iteration for final results

        database=None, # database to save results
        store_boot=False, # save inner bootstrap Ritz values?
        store_P=False, # save Ritz rotator coefficients?

        progress=True, # display progress bar?
    ):
        # seed for reproducibility / correlations
        np.random.seed(seed)

        # check correlator dimensions are valid for scalar or block
        assert len(C.shape) in (2, 4)
        if len(C.shape) == 2:
            C = C[..., None,None] # it -> itab
        N_samps, Nt, bdim, _ = C.shape
        print(C.shape)
        print((N_samps, Nt, bdim, bdim))
        assert C.shape == (N_samps, Nt, bdim,bdim)
        self.N_samps = N_samps
        self.Nt = Nt
        self.bdim = bdim

        print("\n"+"=" * 100)
        print("Starting filtered Rayleigh-Ritz correlator analysis")
        print("\nObtaining Ritz values with RR = block Lanczos = block Prony = GPOF")
        if eps_ZCW is not None:
            print("\nApplying ZCW filter with eps_ZCW = ", eps_ZCW)
        else:
            print("\nApplying ZCW filter with adaptive eps_ZCW, hyperparameter F_ZCW = ", F_ZCW)
        print("\nBlock correlator shape = ", C.shape)
        print("=" * 100 + "\n")

        # make booted corrs (both levels of booting)
        if bs_C is None:
            bs_tensor = make_bs_tensor(N_samps=N_samps, N_boots=N_inner)
            bs_C = np.einsum('bn,n...->b...', bs_tensor, C)
        elif bdim == 1:
            N_inner = bs_C.shape[0]
            # blockify
            if len(bs_C.shape) == 2:
                bs_C = bs_C[..., None,None] # it -> itab
        if nbs_C is None:
            nbs_tensor = make_nbs_tensor(N_samps=N_samps, N_outer=N_outer, N_inner=N_inner)
            nbs_C = np.einsum('oin,n...->oi...', nbs_tensor, C)
        elif bdim == 1:
            N_outer = nbs_C.shape[0]
            # blockify
            if len(nbs_C.shape) == 3:
                nbs_C = nbs_C[..., None,None] # oit -> oitab
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
        if final_m == None:
            final_m = all_m[-2]

        print("final_m = ", final_m)

        # run RR = Lanczos = Prony = GPOF
        print("starting RR analysis")
        all_analyses = run_all_block_rgpof(bs_C, 
                                           abs_B=abs_B, 
                                           ignore_is_norm=ignore_is_norm,
                                           E_gap_bound=E_gap_bound)

        # find m_label for bs
        def _find_max_herm_m(all_analyses):
            all_m = np.array(list(all_analyses.keys()))
            is_herm = np.array([an['is_herm'].mean() > thresh for an in all_analyses.values()])
            return np.max(all_m[is_herm])
        m_last = _find_max_herm_m(all_analyses) 
        print("m_last = ", m_last)
        print("E[m_last] = ", np.nanmedian(all_analyses[m_last]['E'],axis=0))

        # find eps_ZCW 
        fixed_eps_ZCW = None
        if eps_ZCW is None:
            ZCW_last = np.median( np.min(all_analyses[m_last]['ZCW'], axis=-1), axis=0)
            eps_ZCW = ZCW_last/F_ZCW
            print("adaptive eps_ZCW = ", eps_ZCW)
        else:
            fixed_eps_ZCW = eps_ZCW

        # Hermitian subspace and ZCW filters
        print("applying HS and ZCW filters")
        all_filtered = {m : filter_block_analysis(an, an['is_herm'] & (an['ZCW'] > eps_ZCW)) for m,an in all_analyses.items()}

        # remove projectors to save memory if not needed
        if not store_P:
            for m in all_m:
                del(all_filtered[m]['Pl']) 
                del(all_filtered[m]['Pr'])

        # compute one-level bootstrap median
        print("taking median")
        if thresh_inner == None:
            thresh_inner = thresh
        all_filtered_median = {
            m : {k : batched_median(v, thresh=thresh_inner) for k,v in an.items()}
            for m,an in all_filtered.items()
        }

        #N_states = np.count_nonzero(~np.isnan(all_filtered_median[final_m]['E']))
        N_states = max([np.count_nonzero(~np.isnan(all_filtered_median[m]['E'])) for m in all_m])
        print("N_states = ", N_states)
        for an in all_filtered_median.values():
            for key,val in an.items():
                an[key] = val[0:N_states]
                if an[key].shape[0] < N_states:
                    ones_T = np.einsum('k,...->k...', np.ones((N_states-an[key].shape[0])), an[key][-1])
                    an[key] = np.concatenate([an[key], ones_T])

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
            nbs_analyses = run_all_block_rgpof(nbs_C[b], 
                                           abs_B=abs_B, 
                                           ignore_is_norm=ignore_is_norm,
                                           E_gap_bound=E_gap_bound)

            # find eps_ZCW 
            if fixed_eps_ZCW is None:
                nbs_m_last = _find_max_herm_m(nbs_analyses) 
                ZCW_last = np.median( np.min(nbs_analyses[nbs_m_last]['ZCW'], axis=-1), axis=0)
                eps_ZCW = ZCW_last/F_ZCW

            # Hermitian subspace and ZCW filters
            nbs_filtered = {m : filter_block_analysis(an, an['is_herm'] & (an['ZCW'] > eps_ZCW)) for m,an in nbs_analyses.items()}


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

        print("taking inner median")
        all_nbs_filtered_median = {

            m : {k : inner_median(v, thresh=thresh, thresh_inner=thresh_inner) for k,v in an.items()}
            for m,an in all_nbs_filtered.items()
        }

        for m,an in all_nbs_filtered_median.items():
            for key,val in an.items():
                an[key] = val[:,0:N_states]
                if an[key].shape[1] < N_states:
                    val_T = np.einsum('bk...->kb...', an[key])
                    ones_T = np.einsum('bk...,b...->kb...', np.ones((an[key].shape[0],N_states-an[key].shape[1])), an[key][:,-1])
                    an[key] = np.einsum('kb...->bk...', np.vstack([val_T, ones_T]))

        print("computing confidence intervals")
        all_filtered_median_err = {
            m : {k : make_boot_ci(np.real(v)) for k,v in an.items()}
            for m,an in all_nbs_filtered_median.items()
        }

        print("\nZ_final = ", all_filtered_median[final_m]['Zl'], " +/- ", all_filtered_median_err[final_m]['Zl'])
        print("\nE_final = ", all_filtered_median[final_m]['E'], " +/- ", all_filtered_median_err[final_m]['E'])
        print("\nres_lower_final = ", all_filtered_median[final_m]['res_lower'], " +/- ", all_filtered_median_err[final_m]['res_lower'])
        print("res_upper_final = ", all_filtered_median[final_m]['res_upper'], " +/- ", all_filtered_median_err[final_m]['res_upper'])
        print("\ngap_lower_final = ", all_filtered_median[final_m]['gap_lower'], " +/- ", all_filtered_median_err[final_m]['gap_lower'])
        print("gap_upper_final = ", all_filtered_median[final_m]['gap_upper'], " +/- ", all_filtered_median_err[final_m]['gap_upper'])

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
        # final E
        self.final_E=all_filtered_median[final_m]['E']
        self.final_E_err=all_filtered_median_err[final_m]['E']
        self.final_E_boot=all_nbs_filtered_median[final_m]['E']
        # final B
        self.final_B=all_filtered_median[final_m]['Bl']
        self.final_B_err=all_filtered_median_err[final_m]['Bl']
        self.final_B_boot=all_nbs_filtered_median[final_m]['E']
        # final Z
        self.final_Z=all_filtered_median[final_m]['Zl']
        self.final_Z_err=all_filtered_median_err[final_m]['Zl']
        self.final_Z_boot=all_nbs_filtered_median[final_m]['Zl']
        # final res bound
        self.final_res_lower=all_filtered_median[final_m]['res_lower']
        self.final_res_lower_err=all_filtered_median_err[final_m]['res_lower']
        self.final_res_lower_boot=all_nbs_filtered_median[final_m]['res_lower']
        self.final_res_upper=all_filtered_median[final_m]['res_upper']
        self.final_res_upper_err=all_filtered_median_err[final_m]['res_upper']
        self.final_res_upper_boot=all_nbs_filtered_median[final_m]['res_upper']
        # final gap bound
        self.final_gap_lower=all_filtered_median[final_m]['gap_lower']
        self.final_gap_lower_err=all_filtered_median_err[final_m]['gap_lower']
        self.final_gap_lower_boot=all_nbs_filtered_median[final_m]['gap_lower']
        self.final_gap_upper=all_filtered_median[final_m]['gap_upper']
        self.final_gap_upper_err=all_filtered_median_err[final_m]['gap_upper']
        self.final_gap_upper_boot=all_nbs_filtered_median[final_m]['gap_upper']
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

        # save results to database if desired
        if database is not None:
            make_database(self, database)
