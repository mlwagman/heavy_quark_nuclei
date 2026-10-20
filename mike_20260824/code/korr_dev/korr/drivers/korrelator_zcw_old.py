import numpy as np
import h5py

from ..util import isreal
from ..util import make_block_hankel
from ..util import effective_mass, bs_effective_mass

from ..bootstrap import make_bs_tensor, make_nbs_tensor

from ..rgpof import diagonalize_block_rgpof
from ..rgpof import make_block_rgpof_ritz_coeffs
from ..rgpof import make_block_rgpof_overlaps

from ..orr import make_block_rr_bounds_PrPl

from ..filter import compute_block_ZCW
from ..filter import apply_ZCW_test

from ..outlier import get_boot_err, get_final_nested_median_err
from ..outlier import apply_median, apply_nested_median
from ..outlier import get_bounds_median, get_bounds_nested_median

from ..bounds import get_residual_bounds, get_gap_bounds

# main function for single iteration of RR = Lanczos = Prony = GPOF
def run_block_rr(bs_Cab, m, store_P=False, method="rgpof", abs_B=True, staggered=False):
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    Gab = make_block_hankel(bs_Cab, 0)[..., :m, :, :m, :]
    Mab = make_block_hankel(bs_Cab, 1)[..., :m, :, :m, :]
    d = m*bdim

    if method == "rgpof":
        ev, evec = diagonalize_block_rgpof(Gab, Mab)

        Zr, Zl, Nk, is_norm = make_block_rgpof_overlaps(evec, bs_Cab)
        ZZ = Zr.conj() * Zl # diagonal amplitude Zr*_ka Zl_ka
        is_herm = isreal(ev)  & isreal(ZZ).all(-1)
        if not staggered:
            is_herm = is_herm & is_norm & (ZZ.real > 0).all(-1)

        ZCW = compute_block_ZCW(Zr, Zl, bs_Cab)

        Pr, Pl = make_block_rgpof_ritz_coeffs(evec, Nk, Gab)

    else:
        print("unsupported RR method")
        throw()

    if 2*m < Nt:
        if abs_B:
            Br, Bl = np.abs(make_block_rr_bounds_PrPl(bs_Cab, ev, Pr, Pl))
        else:
            Br, Bl = np.real(make_block_rr_bounds_PrPl(bs_Cab, ev, Pr, Pl))
    else:
        Br = np.nan * np.zeros_like(ev)
        Bl = np.nan * np.zeros_like(ev)

    if store_P:
        return dict(
            ev=ev, #evec=evec,
            Zr=Zr, Zl=Zl,
            is_herm=is_herm,
            ZCW=ZCW,
            Br=Br, Bl=Bl,
            Pr=Pr, Pl=Pl,
        )
    else:
        return dict(
            ev=ev, #evec=evec,
            Zr=Zr, Zl=Zl,
            is_herm=is_herm,
            ZCW=ZCW,
            Br=Br, Bl=Bl,
        )

# main function to run RR for all iterations
def run_all_block_rr(bs_Cab, progress=True, store_P=False, method="rgpof", abs_B=True, staggered=False):
    *batch, Nt, bdim, _ = bs_Cab.shape
    assert bs_Cab.shape == (*batch, Nt, bdim, bdim)
    m_max = Nt//2

    # enable progress bar
    if progress:
        from tqdm import tqdm

    analysis = {}
    all_m = range(1, m_max+1)
    if len(bs_Cab.shape) == 5:
        # split nested bootstrap analysis
        N_outer, N_inner = bs_Cab.shape[0], bs_Cab.shape[1]
        for b in tqdm(range(N_outer)) if progress else range(N_outer):
            for m in all_m:
                danalysis = run_block_rr(bs_Cab[b], m, store_P=store_P, method=method, abs_B=abs_B, staggered=staggered)
                for k,v in danalysis.items():
                    if k not in analysis: analysis[k] = {}
                    if m not in analysis[k]:
                        analysis[k][m] = v[None,...]
                    else:
                        analysis[k][m] = np.concatenate([analysis[k][m], v[None,...]])
    else:
        for m in tqdm(all_m) if progress else all_m:
            danalysis = run_block_rr(bs_Cab, m, store_P=store_P, method=method, abs_B=abs_B, staggered=staggered)
            for k,v in danalysis.items():
                if k not in analysis: analysis[k] = {}
                analysis[k][m] = v
    return analysis

# save RR results to HDF5 database
def make_database(rr_analysis, database):
    output_f = h5py.File(database+".h5", "w")
    output_f.create_dataset("eff_mass", data=rr_analysis.eff_mass)
    output_f.create_dataset("eff_mass_err", data=rr_analysis.eff_mass_err)
    output_f.create_dataset("eff_mass_boot", data=rr_analysis.eff_mass_boot)
    # store final results
    output_f.create_dataset("final_m", data=rr_analysis.final_m)
    output_f.create_dataset("final_E", data=rr_analysis.final_E)
    output_f.create_dataset("final_B", data=rr_analysis.final_B)
    output_f.create_dataset("final_Z", data=rr_analysis.final_Z)
    output_f.create_dataset("final_E_err", data=rr_analysis.final_E_err)
    output_f.create_dataset("final_B_err", data=rr_analysis.final_B_err)
    output_f.create_dataset("final_Z_err", data=rr_analysis.final_Z_err)
    output_f.create_dataset("final_E_boot", data=rr_analysis.final_E_boot)
    output_f.create_dataset("final_B_boot", data=rr_analysis.final_B_boot)
    output_f.create_dataset("final_Z_boot", data=rr_analysis.final_Z_boot)
    output_f.create_dataset("final_res_lower", data=rr_analysis.final_res_lower)
    output_f.create_dataset("final_res_upper", data=rr_analysis.final_res_upper)
    output_f.create_dataset("final_gap_lower", data=rr_analysis.final_gap_lower)
    output_f.create_dataset("final_gap_upper", data=rr_analysis.final_gap_upper)
    output_f.create_dataset("final_res_lower_err", data=rr_analysis.final_res_lower_err)
    output_f.create_dataset("final_res_upper_err", data=rr_analysis.final_res_upper_err)
    output_f.create_dataset("final_gap_lower_err", data=rr_analysis.final_gap_lower_err)
    output_f.create_dataset("final_gap_upper_err", data=rr_analysis.final_gap_upper_err)
    output_f.create_dataset("final_res_lower_boot", data=rr_analysis.final_res_lower_boot)
    output_f.create_dataset("final_res_upper_boot", data=rr_analysis.final_res_upper_boot)
    output_f.create_dataset("final_gap_lower_boot", data=rr_analysis.final_gap_lower_boot)
    output_f.create_dataset("final_gap_upper_boot", data=rr_analysis.final_gap_upper_boot)
    # store ZCW test info
    output_f.create_dataset("med_m_last", data=rr_analysis.med_m_last)
    output_f.create_dataset("med_eps_ZCW", data=rr_analysis.med_eps_ZCW)
    output_f.create_dataset("med_m_last_err", data=rr_analysis.med_m_last_err)
    output_f.create_dataset("med_eps_ZCW_err", data=rr_analysis.med_eps_ZCW_err)
    # store outer bootstrap results
    all_m = rr_analysis.med_ev.keys()
    N_outer = len(rr_analysis.med_ev[list(all_m)[0]])
    for m in all_m:
        for n in range(len(rr_analysis.med_ev[m])):
            output_f.create_dataset("med_ev/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.med_ev[m][n])
            output_f.create_dataset("med_Z/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.med_Z[m][n])
            output_f.create_dataset("med_B/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.med_B[m][n])
            output_f.create_dataset("med_res_lower/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.med_res_lower[m][n])
            output_f.create_dataset("med_res_upper/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.med_res_lower[m][n])
            output_f.create_dataset("med_gap_lower/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.med_res_lower[m][n])
            output_f.create_dataset("med_gap_upper/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.med_res_lower[m][n])
        for n in range(len(rr_analysis.nested_med_ev[m])):
            output_f.create_dataset("nested_med_ev/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.nested_med_ev[m][n])
            output_f.create_dataset("nested_med_Z/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.nested_med_Z[m][n])
            output_f.create_dataset("nested_med_B/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.nested_med_B[m][n])
            output_f.create_dataset("nested_med_res_lower/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.nested_med_res_lower[m][n])
            output_f.create_dataset("nested_med_res_upper/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.nested_med_res_lower[m][n])
            output_f.create_dataset("nested_med_gap_lower/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.nested_med_res_lower[m][n])
            output_f.create_dataset("nested_med_gap_upper/m"+str(m)+"/state"+str(n),
                                    data=rr_analysis.nested_med_res_lower[m][n])

    # store inner bootstrap if desired
    if rr_analysis.store_inner_boot:
        for m in all_m:
            for b in range(len(rr_analysis.filtered_outer_ev[m])):
                output_f.create_dataset("filtered_outer_ev/m"+str(m)+"/b"+str(b),
                                data=rr_analysis.filtered_outer_ev[m][b])
                output_f.create_dataset("filtered_outer_Z/m"+str(m)+"/b"+str(b),
                                data=rr_analysis.filtered_outer_Z[m][b])
                output_f.create_dataset("filtered_outer_B/m"+str(m)+"/b"+str(b),
                                data=rr_analysis.filtered_outer_B[m][b])
            for b in range(len(rr_analysis.filtered_inner_ev[m])):
                for ib in range(len(rr_analysis.filtered_inner_ev[m][b])):
                    output_f.create_dataset("filtered_inner_ev/m"+str(m)+"/b"+str(b)+"/ib"+str(ib),
                                    data=rr_analysis.filtered_inner_ev[m][b][ib])
                    output_f.create_dataset("filtered_inner_Z/m"+str(m)+"/b"+str(b)+"/ib"+str(ib),
                                    data=rr_analysis.filtered_inner_Z[m][b][ib])
                    output_f.create_dataset("filtered_inner_B/m"+str(m)+"/b"+str(b)+"/ib"+str(ib),
                                    data=rr_analysis.filtered_inner_B[m][b][ib])

    # store Ritz rotators if desired
    if rr_analysis.store_P:
        for m in all_m:
            for b in range(len(rr_analysis.filtered_outer_P[m])):
                output_f.create_dataset("filtered_outer_Pl/m"+str(m)+"/b"+str(b),
                                data=rr_analysis.filtered_outer_P[m][b])
            for b in range(len(rr_analysis.filtered_inner_P[m])):
                for ib in range(len(rr_analysis.filtered_inner_P[m][b])):
                    output_f.create_dataset("filtered_inner_Pl/m"+str(m)+"/b"+str(b)+"/ib"+str(ib),
                                    data=rr_analysis.filtered_inner_P[m][b][ib])

    # store unfiltered results if desired
    if rr_analysis.store_unfiltered:
        for m in all_m:
            output_f.create_dataset("unfiltered_ev/m"+str(m),
                                    data=rr_analysis.unfiltered_ev[m])
            output_f.create_dataset("unfiltered_Zr/m"+str(m),
                                    data=rr_analysis.unfiltered_Zr[m])
            output_f.create_dataset("unfiltered_Zl/m"+str(m),
                                    data=rr_analysis.unfiltered_Zl[m])
            output_f.create_dataset("unfiltered_Br/m"+str(m),
                                    data=rr_analysis.unfiltered_Br[m])
            output_f.create_dataset("unfiltered_Bl/m"+str(m),
                                    data=rr_analysis.unfiltered_Bl[m])
            if rr_analysis.store_P:
                output_f.create_dataset("unfiltered_Pr/m"+str(m),
                                    data=rr_analysis.unfiltered_Pr[m])
                output_f.create_dataset("unfiltered_Pl/m"+str(m),
                                    data=rr_analysis.unfiltered_Pl[m])

    output_f.close()

class korrelator:
    def __init__(self,
        C, # correlator input ~ (Batch, t,[a,b])

        F_ZCW=10, # ZCW filtering hyperparameter
        eps_ZCW=None, # manually specify ZCW cutoff
        filter_herm=True, # apply Hermitian subspace filter

        Q=0.05, # state ID hyperparameter
        N_outer=200, # bootstrap confidence interval hyperparameter
        N_inner=200, # bootstrap median hyperparameter
        bs_C=None,  # manually pass outer bootstrap tensor
        nbs_C=None, # manually pass inner bootstrap tensor

        abs_B=True, # apply abs before median to B_k^{(m)}

        final_m=None, # manually specify iteration for final results
        method="rgpof", # method for calculating Ritz values

        staggered=False,
        true_E=None,

        database=None, # database to save results
        store_unfiltered=False, # save unfiltered Ritz values?
        store_inner_boot=False, # save inner bootstrap Ritz values?
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
        print(C.shape)
        print((N_samps, Nt, bdim, bdim))
        assert C.shape == (N_samps, Nt, bdim, bdim)

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
            bs_C = bs_C[..., None,None] # it -> itab
        if nbs_C is None:
            nbs_tensor = make_nbs_tensor(N_samps=N_samps, N_outer=N_outer, N_inner=N_inner)
            nbs_C = np.einsum('oin,n...->oi...', nbs_tensor, C)
        elif bdim == 1:
            N_outer = nbs_C.shape[0]
            nbs_C = nbs_C[..., None,None] # it -> itab
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
                eff_mass_err[a,t] = get_boot_err(bs_eff_mass[:,t])
            print("operator a=",a," effective mass")
            print(eff_mass[a,:], " +/- ", eff_mass_err[a,:])

        # range of iterations
        all_m = range(1,Nt//2+1)
        if final_m == None:
            final_m = all_m[-2]

        # run RR = Lanczos = Prony = GPOF
        print("starting RR analysis")
        analysis = run_all_block_rr(bs_C, store_P=store_P, progress=progress, method=method, abs_B=abs_B, staggered=staggered)

        # apply ZCW filter
        print("applying ZCW filter")
        filtered_analysis = apply_ZCW_test(analysis, F_ZCW=F_ZCW,
                                      store_P=store_P, eps_ZCW=eps_ZCW, filter_herm=filter_herm, staggered=staggered)

        med_eps_ZCW = np.median(filtered_analysis["eps_ZCW"])
        med_m_last = np.median(filtered_analysis["m_last"])
        med_m_last = np.floor(med_m_last).astype(int)
        print("eps_ZCW = ", med_eps_ZCW)
        print("m_last = ", med_m_last)

        # boostrap median
        med_ev = apply_median(filtered_analysis["ev"], Q=Q)
        med_Z = apply_median(filtered_analysis["Z"], Q=Q)
        med_B = apply_median(filtered_analysis["B"], Q=Q)
        med_res_lower, med_res_upper, med_gap_lower, med_gap_upper = get_bounds_median(filtered_analysis["ev"], filtered_analysis["B"], Q=Q, true_E=true_E)
        print("med_ev[med_m_last] = ", med_ev[med_m_last])

        # bootstrap median final results
        # last iteration where residual calculabe unless otherwise specified
        print("final_m = ", final_m)
        print("med_ev[final_m] = ", med_ev[final_m])
        N_states = len(med_ev[final_m])
        print("N_states = ", N_states)
        if staggered:
            k_osc = []
            for k in range(N_states):
                if med_ev[final_m][k] < 0:
                    k_osc.append(k)
            N_osc = len(k_osc)
            print("ignoring ", N_osc, " oscillating states with energies ", -1*np.log(np.abs(med_ev[final_m][N_states-N_osc:N_states])))
            N_states -= N_osc

        final_ev = np.array([med_ev[final_m][k] for k in range(N_states)])
        final_E = -np.log(final_ev)
        final_Z = np.array([med_Z[final_m][k] for k in range(N_states)])
        final_B = np.array([med_B[final_m][k] for k in range(N_states)])
        final_res_lower = np.array([med_res_lower[final_m][k] for k in range(N_states)])
        final_res_upper = np.array([med_res_upper[final_m][k] for k in range(N_states)])
        N_states_gap = len(med_gap_lower[final_m])
        if N_states_gap > N_states:
            N_states_gap = N_states
        final_gap_lower = np.array([med_gap_lower[final_m][k] for k in range(N_states_gap)])
        final_gap_upper = np.array([med_gap_upper[final_m][k] for k in range(N_states_gap)])
        print("\nE_final = ", final_E)
        print("Z_final = ", final_Z)
        print("B_final = ", final_B)
        print("res_lower_final = ", final_res_lower)
        print("res_upper_final = ", final_res_upper)
        print("gap_lower_final = ", final_gap_lower)
        print("gap_upper_final = ", final_gap_upper)


        # run nested bootstrap RR
        print("starting nested bootstrap RR analysis")
        nested_analysis = run_all_block_rr(nbs_C, store_P=store_P, method=method, abs_B=abs_B, staggered=staggered)

        # apply ZCW test
        print("applying ZCW filter")
        filtered_nested_analysis = apply_ZCW_test(nested_analysis, F_ZCW=F_ZCW,
                                             store_P=store_P, eps_ZCW=eps_ZCW, filter_herm=filter_herm, staggered=staggered)

        nested_med_eps_ZCW = filtered_nested_analysis["eps_ZCW"]
        nested_med_m_last = filtered_nested_analysis["m_last"]
        med_eps_ZCW_err = get_boot_err(nested_med_eps_ZCW)
        med_m_last_err = get_boot_err(nested_med_m_last)
        print("eps_ZCW = ", med_eps_ZCW, " +/- ", med_eps_ZCW_err)
        print("m_last = ", med_m_last, " +/- ", med_m_last_err)

        # nested bootstrap median
        nested_med_ev = apply_nested_median(filtered_nested_analysis["ev"], Q=Q)
        nested_med_Z = apply_nested_median(filtered_nested_analysis["Z"], Q=Q)
        nested_med_B = apply_nested_median(filtered_nested_analysis["B"], Q=Q)
        nested_med_E = {m: [-np.log(nested_med_ev[m][n])
                            for n in range(len(nested_med_ev[m]))] for m in all_m}

        nested_med_res_lower, nested_med_res_upper, nested_med_gap_lower, nested_med_gap_upper = get_bounds_nested_median(filtered_nested_analysis["ev"], filtered_nested_analysis["B"], Q=Q, true_E=true_E)

        # nested bootstrap uncertainties
        nested_med_final_m = np.zeros((N_outer,N_states), dtype=int)
        for k in range(N_states):
            nested_med_final_m[:,k] = final_m
        final_E_err, final_E_boot = get_final_nested_median_err(nested_med_E, nested_med_final_m, N_states)
        # ignore states whose median does not exist in any outer boot
        for n in range(N_states):
            if np.isnan(final_E_err[n]):
                N_states = n
                if N_states_gap > N_states:
                    N_states_gap = N_states
                final_E = final_E[:N_states]
                final_Z = final_Z[:N_states]
                final_B = final_B[:N_states]
                final_res_lower = final_res_lower[:N_states]
                final_res_upper = final_res_upper[:N_states]
                final_gap_lower = final_gap_lower[:N_states_gap]
                final_gap_upper = final_gap_upper[:N_states_gap]
                final_E_err, final_E_boot = get_final_nested_median_err(nested_med_E, nested_med_final_m, N_states)
                break
        print("N_states = ", N_states)
        final_Z_err, final_Z_boot = get_final_nested_median_err(nested_med_Z, nested_med_final_m, N_states)
        final_B_err, final_B_boot = get_final_nested_median_err(nested_med_B, nested_med_final_m, N_states)
        final_res_lower_err, final_res_lower_boot = get_final_nested_median_err(nested_med_res_lower, nested_med_final_m, N_states)
        final_res_upper_err, final_res_upper_boot = get_final_nested_median_err(nested_med_res_upper, nested_med_final_m, N_states)
        final_gap_lower_err, final_gap_lower_boot = get_final_nested_median_err(nested_med_gap_lower, nested_med_final_m, N_states_gap)
        final_gap_upper_err, final_gap_upper_boot = get_final_nested_median_err(nested_med_gap_upper, nested_med_final_m, N_states_gap)

        # min res bootstrap median errors
        print("\nE_final = ", final_E, " +/- ", final_E_err)
        print("Z_final = ", final_Z, " +/- ", final_Z_err)
        print("B_final = ", final_B, " +/- ", final_B_err)
        print("res_lower_final = ", final_res_lower, " +/- ", final_res_lower_err)
        print("res_upper_final = ", final_res_upper, " +/- ", final_res_upper_err)
        print("gap_lower_final = ", final_gap_lower, " +/- ", final_gap_lower_err)
        print("gap_upper_final = ", final_gap_upper, " +/- ", final_gap_upper_err)

        # store metadata about storage
        self.store_unfiltered=store_unfiltered
        self.store_inner_boot=store_inner_boot
        self.store_P=store_P
        self.database=database
        self.eff_mass=eff_mass
        self.eff_mass_err=eff_mass_err
        self.eff_mass_boot=eff_mass_boot
        # store final results
        self.final_m=final_m
        self.final_E=final_E
        self.final_Z=final_Z
        self.final_B=final_B
        self.final_E_err=final_E_err
        self.final_Z_err=final_Z_err
        self.final_B_err=final_B_err
        self.final_E_boot=final_E_boot
        self.final_Z_boot=final_Z_boot
        self.final_B_boot=final_B_boot
        # store final bound results
        self.final_res_lower=final_res_lower
        self.final_res_upper=final_res_upper
        self.final_gap_lower=final_gap_lower
        self.final_gap_upper=final_gap_upper
        self.final_res_lower_err=final_res_lower_err
        self.final_res_upper_err=final_res_upper_err
        self.final_gap_lower_err=final_gap_lower_err
        self.final_gap_upper_err=final_gap_upper_err
        self.final_res_lower_boot=final_res_lower_boot
        self.final_res_upper_boot=final_res_upper_boot
        self.final_gap_lower_boot=final_gap_lower_boot
        self.final_gap_upper_boot=final_gap_upper_boot
        # store ZCW info
        self.med_eps_ZCW=med_eps_ZCW
        self.med_eps_ZCW_err=med_eps_ZCW_err
        self.med_m_last=med_m_last
        self.med_m_last_err=med_m_last_err
        # store outer bootstrap results
        self.med_ev=med_ev
        self.med_Z=med_Z
        self.med_B=med_B
        self.med_res_lower=med_res_lower
        self.med_res_upper=med_res_upper
        self.med_gap_lower=med_gap_lower
        self.med_gap_upper=med_gap_upper
        # store inner bootstrap results
        self.nested_med_ev=nested_med_ev
        self.nested_med_Z=nested_med_Z
        self.nested_med_B=nested_med_B
        self.nested_med_res_lower=med_res_lower
        self.nested_med_res_upper=med_res_upper
        self.nested_med_gap_lower=med_gap_lower
        self.nested_med_gap_upper=med_gap_upper
        # store inner bootstrap if desired
        if store_inner_boot:
            self.filtered_outer_ev = filtered_analysis["ev"]
            self.filtered_outer_Z = filtered_analysis["Z"]
            self.filtered_outer_B = filtered_analysis["B"]
            self.filtered_inner_ev = filtered_nested_analysis["ev"]
            self.filtered_inner_Z = filtered_nested_analysis["Z"]
            self.filtered_inner_B = filtered_nested_analysis["B"]
        # store Ritz rotators if desired
        if store_P:
            self.filtered_outer_P = filtered_analysis["P"]
            self.filtered_inner_P = filtered_nested_analysis["P"]
        # store unfiltered results if desired
        if store_unfiltered:
            self.unfiltered_ev = nested_analysis["ev"]
            self.unfiltered_Zr = nested_analysis["Zr"]
            self.unfiltered_Zl = nested_analysis["Zl"]
            self.unfiltered_Br = nested_analysis["Br"]
            self.unfiltered_Bl = nested_analysis["Bl"]
            if store_P:
                self.unfiltered_Pr = nested_analysis["Pr"]
                self.unfiltered_Pl = nested_analysis["Pl"]

        # show two-sided bounds
        print("\nAlternative residual bound calculation")
        get_residual_bounds(self)

        # save results to database if desired
        if database is not None:
            make_database(self, database)
