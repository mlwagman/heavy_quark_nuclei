import numpy as np
import h5py

from ..util import isreal
from ..util import make_block_hankel, stack_dicts
from ..util import effective_mass, bs_effective_mass

from ..eig_util import eig

from ..bootstrap import make_bs_tensor, make_nbs_tensor

from ..rgpof import diagonalize_rgpof, make_block_rgpof_ritz_coeffs, make_block_rgpof_overlaps

from ..orr import make_block_rr_bounds_PrPl

from ..outlier import batched_median, inner_median, final_iter_median, homogenize_analyses, make_boot_ci

from ..filter import filter_block_analysis, compute_block_ZCW

from ..label import compute_label_dot, label_states_argmax, label_block_analysis

from ..bounds import get_res_bounds, get_gap_bounds

# main function for single iteration of RR = Lanczos = Prony = GPOF
def run_block_rgpof(bs_Cab, m,
    eig=eig, inv=np.linalg.inv, log=np.log,
    real=np.real, abs=np.abs, isclose=np.isclose,
    abs_B=False, ignore_is_norm=False
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
    is_herm = isreal(ev) & isreal(ZZ).all(-1)
    is_herm = is_herm & (real(ev) > 0)
    is_herm = is_herm & (real(ev) < 1)
    is_herm = is_herm & (real(ZZ) > 0).all(-1)
    if not ignore_is_norm:
        is_herm = is_herm & is_norm

    ZCW = compute_block_ZCW(Zr, Zl, bs_Cab, inv=inv, abs=abs)

    Pr, Pl = make_block_rgpof_ritz_coeffs(evec, Nk, Gab, inv=inv)

    if 2*m < Nt:
        Br, Bl = make_block_rr_bounds_PrPl(bs_Cab, ev, Pr, Pl)
        if abs_B:
            Br = abs(Br)
            Bl = abs(Bl)
    else:
        Br = np.nan * np.zeros_like(ev)
        Bl = np.nan * np.zeros_like(ev)

    # derived quantities
    E = -log(ev)
    B = np.where(np.abs(Bl) < np.abs(Br), Bl, Br)

    return dict(
        ev=ev, #evec=evec,
        Zr=Zr, Zl=Zl,
        is_herm=is_herm,
        ZCW=ZCW,
        Br=Br, Bl=Bl,
        Pr=Pr, Pl=Pl,
        E = E, B=B
    )

def run_all_block_rgpof(bs_Cab,
                        eig=eig, inv=np.linalg.inv, log=np.log,
                        real=np.real, abs=np.abs, isclose=np.isclose,
                        abs_B=False, ignore_is_norm=False, progress=False):
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2

    all_m = range(1, m_max+1)

    analysis = {}
    if progress:
        from tqdm import tqdm
    for m in (tqdm(all_m) if progress else all_m):
        danalysis = run_block_rgpof(bs_Cab, m=m,
                                    abs_B=abs_B,
                                    eig=eig, inv=inv, log=log,
                                    real=real, abs=abs, isclose=isclose,
                                    ignore_is_norm=ignore_is_norm)
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
            "final_res_lower_2sigmaerr", "final_res_upper_2sigmaerr",
            "final_gap_lower_2sigmaerr", "final_gap_upper_2sigmaerr",
            "final_res_lower_boot", "final_res_upper_boot",
            "final_gap_lower_boot", "final_gap_upper_boot",
            # filtering info
            "m_last",
            "eps_ZCW",
            "m_window",
            # gap bound info
            "E_gap_bound",
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
                for k in ["bs_Pl", "nbs_Pl", "bs_Pr", "nbs_Pr"]:
                    fout.create_dataset(k+"/m"+str(m), data=getattr(rr_analysis, k)[m])

        if rr_analysis.store_boot:
            for k in [
                "bs_ev", "bs_Z", "bs_B", "bs_E",
                "nbs_ev", "nbs_Z", "nbs_B", "nbs_E",
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
        m_window=3, # how many of lst iterations to average over
        do_labeling=True, # label based on last non-spurious iteration instead of filter-and-sort
        m_label_override=None, # manually specificy m_label choice

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
        assert len(C.shape) in (2, 4)
        if len(C.shape) == 2:
            C = C[..., None,None] # it -> itab
        N_samps, Nt, bdim, _ = C.shape
        assert C.shape == (N_samps, Nt, bdim,bdim)
        self.N_samps = N_samps
        self.Nt = Nt
        self.bdim = bdim


        print("\n"+"=" * 100)
        print("Starting filtered Rayleigh-Ritz correlator analysis")
        print("\nObtaining Ritz values with RR = block Lanczos = block Prony = GPOF")
        print("\nNested bootstrap N_inner = ", N_inner, ", N_outer = ", N_outer)
        if eps_ZCW is not None:
            print("\nApplying ZCW filter with eps_ZCW = ", eps_ZCW)
        else:
            print("\nApplying ZCW filter with adaptive eps_ZCW, hyperparameter F_ZCW = ", F_ZCW)
        if do_labeling and m_label_override is not None:
            print("\nLabeling based on last iteration = ", m_label_override)
        elif do_labeling:
            print("\nLabeling states based on last non-spurious iteration")
        print("\nTaking median of last ", m_window, " iterations")
        print("\nBlock correlator shape = ", C.shape)
        print("=" * 100 + "\n")
        
        # perform random change of basis to avoid starting in orthogonal basis
        rand_gauss = np.random.randn(bdim, bdim)
        # o(N)
        rand_oN = (rand_gauss - rand_gauss.transpose()) / 2
        ev, evec = np.linalg.eig(rand_oN)
        evec_inv = np.linalg.inv(evec)
        # O(N)
        rand_ON = np.real(evec @ np.diag(np.exp(ev)) @ evec_inv)
        rand_ON_inv = rand_ON.transpose()
        # check orthogonality 
        assert np.allclose(np.linalg.inv(rand_ON), rand_ON.transpose() )
        C = rand_ON_inv @ C @ rand_ON
        if bdim > 1:
            print(f"applied random SO({bdim}) change-of-basis")

        np.random.seed(seed)
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
        C_phys = rand_ON @ C @ rand_ON_inv
        bs_C_phys = rand_ON @ bs_C @ rand_ON_inv
        for a in range(bdim):
            eff_mass[a,:] = effective_mass(np.mean(C_phys[:,:,a,a], axis=0))
            bs_eff_mass = bs_effective_mass(bs_C_phys[:,:,a,a])
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
                                           ignore_is_norm=ignore_is_norm)

        # find eps_ZCW
        fixed_eps_ZCW = None
        if eps_ZCW is None:
            # find last hermitian iteration
            print("Hermitian counts", [np.count_nonzero(all_analyses[m]['is_herm']) for m in all_m])
            herm_fac = 1
            if staggered:
                herm_fac = 2
            print("Thresholds", [bdim*m*N_inner*thresh//herm_fac for m in all_m])
            m_last = max([m if np.count_nonzero(all_analyses[m]['is_herm']) > bdim*m*N_inner*thresh//herm_fac else 1 for m in all_m])
            print("m_last = ", m_last)
            ZCW_last = np.median( np.min(all_analyses[m_last]['ZCW'], axis=-1), axis=0)
            eps_ZCW = ZCW_last/F_ZCW
            print("adaptive eps_ZCW = ", eps_ZCW)
        else:
            fixed_eps_ZCW = eps_ZCW
            m_last = 1

        # Hermitian subspace and ZCW filters
        print("applying HS and ZCW filters")
        all_filtered = {m : filter_block_analysis(an, an['is_herm'] & (an['ZCW'] > eps_ZCW)) for m,an in all_analyses.items()}

        # label states based on overlap with iteration m_label
        m_label = max([m if np.count_nonzero(~np.isnan(all_filtered[m]['ev'])) > bdim*m*N_inner*thresh else 0 for m in all_m])
        # override
        if m_label_override is not None:
            m_label = m_label_override
        print("last iteration with no spurious states = ", m_label)
        if m_label > 0:
            print("E[m_label] = ", np.nanmedian(all_analyses[m_label]['E'],axis=0))

        print("labeling bs analyses")
        for m in all_m:
            if m <= m_label or m_label == 0 or not do_labeling:
                continue
            # construct label vector
            Pl_label = all_filtered[m_label]['Pl']
            # apply labeling
            all_filtered[m] = label_block_analysis(all_filtered[m], label_states_argmax(compute_label_dot(bs_C, Pl_label, all_filtered[m]['Pr'])) )

        if m_label > 0:
            print("E[m_label] = ", np.nanmedian(all_analyses[m_label]['E'],axis=0))

        
        # convert basis-dependent outputs back to original basis
        for m in all_m:
            Zr = all_filtered[m]['Zr']
            Zr = Zr @ rand_ON_inv
            Zl = all_filtered[m]['Zl']
            Zl = Zl @ rand_ON_inv
            # correct phase in original basis
            phase = Zr[...,:,0] / abs(Zr[...,:,0]) # ~ *k
            Zr = (1/phase)[...,:,None] * Zr
            Zl = phase[...,:,None] * Zl
            all_filtered[m]['Zl'] = Zl
            all_filtered[m]['Zr'] = Zr
            if store_P:
                # correct basis and phase of Ritz rotators
                Pr = all_filtered[m]['Pr']
                Pl = all_filtered[m]['Pl']
                all_filtered[m]['Pr'] = np.einsum('...abk,bc,...k->...ack', Pr, rand_ON_inv, 1/phase)
                all_filtered[m]['Pl'] = np.einsum('...kab,bc,...k->...kac', Pl, rand_ON_inv, phase)
            # otherwise remove projectors to save memory
            else:
                del all_filtered[m]['Pl']
                del all_filtered[m]['Pr']


        # compute one-level bootstrap median
        print("taking median")
        if thresh_inner == None:
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
            get_gap_bounds(all_filtered_median[m], E_gap_bound=E_gap_bound, never_abs=never_abs)

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
            nbs_analyses = run_all_block_rgpof(nbs_C[b],
                                           abs_B=abs_B,
                                           ignore_is_norm=ignore_is_norm)
            # find eps_ZCW
            if fixed_eps_ZCW is None:
                nbs_m_last = max([m if np.count_nonzero(nbs_analyses[m]['is_herm']) > bdim*m*N_inner*thresh//herm_fac else 1 for m in all_m])
                ZCW_last = np.median( np.min(nbs_analyses[nbs_m_last]['ZCW'], axis=-1), axis=0)
                nbs_eps_ZCW = ZCW_last/F_ZCW
            else:
                nbs_eps_ZCW = fixed_eps_ZCW
                nbs_m_last = 1

            # Hermitian subspace and ZCW filters
            nbs_filtered = {m : filter_block_analysis(an, an['is_herm'] & (an['ZCW'] > nbs_eps_ZCW)) for m,an in nbs_analyses.items()}


            # label states based on overlap with iteration m_label
            nbs_m_label = max([m if np.count_nonzero(~np.isnan(nbs_filtered[m]['ev'])) > bdim*m*N_inner*thresh else 0 for m in all_m])
            if m_label_override is not None:
                nbs_m_label = m_label_override
            for m in all_m:
                if m <= nbs_m_label or nbs_m_label == 0 or not do_labeling:
                    continue
                # construct label vector
                Pl_label = nbs_filtered[nbs_m_label]['Pl']
                # apply labeling
                nbs_filtered[m] = label_block_analysis(nbs_filtered[m], label_states_argmax(compute_label_dot(nbs_C[b], Pl_label, nbs_filtered[m]['Pr'])) )

            # convert basis-dependent outputs back to original basis
            for m in all_m:
                Zr = nbs_filtered[m]['Zr']
                Zr = Zr @ rand_ON_inv
                Zl = nbs_filtered[m]['Zl']
                Zl = Zl @ rand_ON_inv
                # correct phase in original basis
                phase = Zr[...,:,0] / abs(Zr[...,:,0]) # ~ *k
                Zr = (1/phase)[...,:,None] * Zr
                Zl = phase[...,:,None] * Zl
                nbs_filtered[m]['Zl'] = Zl
                nbs_filtered[m]['Zr'] = Zr
                if store_P:
                    # correct basis and phase of Ritz rotators
                    Pr = nbs_filtered[m]['Pr']
                    Pl = nbs_filtered[m]['Pl']
                    nbs_filtered[m]['Pr'] = np.einsum('...abk,bc,...k->...ack', Pr, rand_ON_inv, 1/phase)
                    nbs_filtered[m]['Pl'] = np.einsum('...kab,bc,...k->...kac', Pl, rand_ON_inv, phase)
                # otherwise remove projectors to save memory
                else:
                    del nbs_filtered[m]['Pl']
                    del nbs_filtered[m]['Pr']

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
        # median Ps not useful for 3pts
        if store_P:
            for m in all_m:
                del all_nbs_filtered_median[m]['Pl']
                del all_nbs_filtered_median[m]['Pr']

        # compute bounds
        for m in all_m:
            get_res_bounds(all_nbs_filtered_median[m])
            get_gap_bounds(all_nbs_filtered_median[m], E_gap_bound=E_gap_bound)

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
        lower_keys = ['res_lower', 'gap_lower']
        upper_keys = ['res_upper', 'gap_upper']
        all_filtered_median = {
            m : {k : an['E'] if (k in lower_keys or k in upper_keys) else v for k,v in an.items()}
            for m,an in all_filtered_median.items()
        }

        # take median of last few iterations
        final_filtered_median = final_iter_median(all_filtered_median, N_states=N_states, m_window=m_window, final_m=final_m)
        final_nbs_filtered_median = final_iter_median(all_nbs_filtered_median, N_states=N_states, m_window=m_window, final_m=final_m)

        # compute confidence intervals for all iterations
        print("computing confidence intervals")
        all_filtered_median_err = {
            m : {k : make_boot_ci(np.real(v), side="lower", cv=an['E']) if k in lower_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=an['E']) if k in upper_keys
                     else
                     make_boot_ci(np.real(v)) for k,v in an.items()}
        for m,an in all_nbs_filtered_median.items()} 

        # compute confidence intervals for final iteration
        final_filtered_median_err = {k : make_boot_ci(np.real(v), side="lower", cv=final_filtered_median['E']) if k in lower_keys
                     else
                     make_boot_ci(np.real(v), side="upper", cv=final_filtered_median['E']) if k in upper_keys
                     else
                     make_boot_ci(np.real(v)) for k,v in final_nbs_filtered_median.items()}

        # compute 2sigma confidence intervals for final iteration
        final_filtered_median_2sigmaerr = {k : make_boot_ci(np.real(v), sigma=2, side="lower", cv=final_filtered_median['E']) if k in lower_keys
                     else
                     make_boot_ci(np.real(v), sigma=2, side="upper", cv=final_filtered_median['E']) if k in upper_keys
                     else
                     make_boot_ci(np.real(v), sigma=2) for k,v in final_nbs_filtered_median.items()}


        print("\nFinal iteration m in [", final_m-m_window+1, ", ", final_m, "]")
        print("\nZ_final = ", final_filtered_median['Zl'], " +/- ", final_filtered_median_err['Zl'])
        print("\nE_final = ", final_filtered_median['E'], " +/- ", final_filtered_median_err['E'])
        print("\nB_final = ", final_filtered_median['B'], " +/- ", final_filtered_median_err['B'])
        print("\nres_lower_final 1sigma = ", final_filtered_median['res_lower'] - final_filtered_median_err['res_lower'])
        print("res_upper_final 1sigma = ", final_filtered_median['res_upper'] + final_filtered_median_err['res_upper'])
        print("\ngap_lower_final 1sigma = ", final_filtered_median['gap_lower'] - final_filtered_median_err['gap_lower'])
        print("gap_upper_final 1sigma = ", final_filtered_median['gap_upper'] + final_filtered_median_err['gap_upper'])

        print("\nres_lower_final 2sigma = ", final_filtered_median['res_lower'] - final_filtered_median_2sigmaerr['res_lower'])
        print("res_upper_final 2sigma = ", final_filtered_median['res_upper'] + final_filtered_median_2sigmaerr['res_upper'])
        print("\ngap_lower_final 2sigma = ", final_filtered_median['gap_lower'] - final_filtered_median_2sigmaerr['gap_lower'])
        print("gap_upper_final 2sigma = ", final_filtered_median['gap_upper'] + final_filtered_median_2sigmaerr['gap_upper'])

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
        # store gap bound if set manually
        if E_gap_bound is not None:
            self.E_gap_bound=E_gap_bound
        else:
            self.E_gap_bound=0
        # final E
        self.final_E=final_filtered_median['E']
        self.final_E_err=final_filtered_median_err['E']
        self.final_E_boot=final_nbs_filtered_median['E']
        # final B
        self.final_B=final_filtered_median['B']
        self.final_B_err=final_filtered_median_err['B']
        self.final_B_boot=final_nbs_filtered_median['B']
        # final Z
        self.final_Z=final_filtered_median['Zl']
        self.final_Z_err=final_filtered_median_err['Zl']
        self.final_Z_boot=final_nbs_filtered_median['Zl']
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
        self.final_gap_lower=final_filtered_median['gap_lower']
        self.final_gap_lower_err=final_filtered_median_err['gap_lower']
        self.final_gap_lower_2sigmaerr=final_filtered_median_2sigmaerr['gap_lower']
        self.final_gap_lower_boot=final_nbs_filtered_median['gap_lower']
        self.final_gap_upper=final_filtered_median['gap_upper']
        self.final_gap_upper_err=final_filtered_median_err['gap_upper']
        self.final_gap_upper_2sigmaerr=final_filtered_median_2sigmaerr['gap_upper']
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
        self.med_B_err=np.array([all_filtered_median_err[m]['B'] for m in all_m])
        self.med_E_err=np.array([all_filtered_median_err[m]['E'] for m in all_m])
        self.med_res_lower_err=np.array([all_filtered_median_err[m]['res_lower'] for m in all_m])
        self.med_res_upper_err=np.array([all_filtered_median_err[m]['res_upper'] for m in all_m])
        self.med_gap_lower_err=np.array([all_filtered_median_err[m]['gap_lower'] for m in all_m])
        self.med_gap_upper_err=np.array([all_filtered_median_err[m]['gap_upper'] for m in all_m])
        self.med_ev_boot=np.array([all_nbs_filtered_median[m]['ev'] for m in all_m])
        self.med_Z_boot=np.array([all_nbs_filtered_median[m]['Zl'] for m in all_m])
        self.med_B_boot=np.array([all_nbs_filtered_median[m]['B'] for m in all_m])
        self.med_E_boot=np.array([all_nbs_filtered_median[m]['E'] for m in all_m])
        self.med_res_lower_boot=np.array([all_nbs_filtered_median[m]['res_lower'] for m in all_m])
        self.med_res_upper_boot=np.array([all_nbs_filtered_median[m]['res_upper'] for m in all_m])
        self.med_gap_lower_boot=np.array([all_nbs_filtered_median[m]['gap_lower'] for m in all_m])
        self.med_gap_upper_boot=np.array([all_nbs_filtered_median[m]['gap_upper'] for m in all_m])
        if store_boot:
            # store one-level bootstrap results
            self.bs_ev=np.array([all_filtered[m]['ev'] for m in all_m])
            self.bs_Z=np.array([all_filtered[m]['Zl'] for m in all_m])
            self.bs_B=np.array([all_filtered[m]['B'] for m in all_m])
            self.bs_E=np.array([all_filtered[m]['E'] for m in all_m])
            # store inner bootstrap median results
            self.nbs_ev=np.array([all_nbs_filtered[m]['ev'] for m in all_m])
            self.nbs_Z=np.array([all_nbs_filtered[m]['Zl'] for m in all_m])
            self.nbs_B=np.array([all_nbs_filtered[m]['B'] for m in all_m])
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
