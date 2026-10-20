r"""Autocorrelation along the walker index, and a binning plateau for the GFMC errors.

Why this matters
----------------
The walker index is **not** an independent-sample index by construction: ``gfmc_Rs`` starts from
``adl.metropolis(..., n_step=n_walkers, n_skip=n_skip)`` (generator line 634), i.e. the walkers are
successive configurations of one Metropolis chain, stored every ``n_skip`` sweeps (default 100).
GFMC then evolves them, and with ``--resampling`` unset (the default) no branching mixes them, so
the Metropolis chain is the only correlation source -- but its strength has to be measured.

Ben's ``Lanczos_gfmc.py:126`` measures the integrated autocorrelation time and inflates the error
by ``2*tauint0`` when it exceeds 0.5.  ``Lanczos_twoleg.py`` dropped that step and relies on
``n_block=5`` binning alone, so every error bar in the two-leg pipeline -- and in
``gfmc_me_korr.py``, which follows the same convention -- is only correct if the binning is
sufficient.  This script checks it two ways:

1. **Integrated autocorrelation time** with Madras-Sokal automatic windowing, on the observables
   that actually enter the analysis: the two-point weights, the three-point correlator, and the
   per-walker ratio that becomes the form factor.  ``tau_int = 0.5`` means independent.
2. **Binning plateau**: the bootstrap error of the ratio estimator as a function of bin size.  If
   there is autocorrelation the error rises with bin size and then flattens; the plateau value is
   the honest error and ``sqrt(2 tau_int)`` is the inflation factor.  If it is flat from the start,
   the walkers are independent and ``n_block`` does not matter.

Usage:
    python walker_autocorr.py <currents_3pt_*.h5> [...] [--n_boot 400]
"""

import argparse
import os

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('files', nargs='+')
p.add_argument('--n_boot', type=int, default=400)
p.add_argument('--seed', type=int, default=17)
p.add_argument('--bins', type=int, nargs='+', default=[1, 2, 5, 10, 20, 25, 50],
               help='bin sizes for the plateau study (n_block in the analysis scripts)')
args = p.parse_args()


def tau_int(x, c_max=None):
    r"""Integrated autocorrelation time with Madras-Sokal automatic windowing.

    ``tau_int(W) = 0.5 + sum_{k=1..W} c(k)``, truncated at the first ``W`` with ``W >= 4
    tau_int(W)`` or at the first non-positive ``c(k)``.  Normalized so that an independent series
    gives 0.5, which is the convention ``Lanczos_gfmc.py`` compares against.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    v = float((x * x).mean())
    if v <= 0 or n < 8:
        return 0.5, 0
    c_max = c_max or max(4, n // 4)
    t = 0.5
    for k in range(1, c_max):
        ck = float((x[:-k] * x[k:]).mean()) / v
        if ck <= 0:
            return t, k - 1
        t += ck
        if k >= 4 * t:
            return t, k
    return t, c_max - 1


def binned_ratio_err(num, den, bin_size, n_boot, rng):
    """Bootstrap error of ``<num>/<den>`` after averaging consecutive walkers in blocks.

    Binning consecutive walkers is what removes autocorrelation; resampling the *bins* then gives
    an error that stops being an underestimate once the bin exceeds the correlation length.
    """
    n = len(num) // bin_size
    if n < 4:
        return np.nan
    nb = num[:n * bin_size].reshape(n, bin_size).mean(axis=1)
    db = den[:n * bin_size].reshape(n, bin_size).mean(axis=1)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = nb[idx].mean(axis=1) / db[idx].mean(axis=1)
    return float(draws.std())


def report(fn, rng):
    with h5py.File(fn, 'r') as f:
        Ws2 = np.real(f['Ws2_all'][:])          # (N_tau, N_w)
        C3 = np.real(f['C3_N'][:])              # (N_w, N_tau)
        tau = f['tau_values'][:]
        q2 = float(f['q_squared'][()])
        n_coord = len(f['masses'][:])
    n_tau, n_w = Ws2.shape
    print(f"\n{'=' * 78}\n{os.path.basename(fn)[:76]}\n{'=' * 78}")
    print(f"  Q = {np.sqrt(q2):.4g}   N_tau = {n_tau}   N_walkers = {n_w}   N_coord = {n_coord}")

    # --- 1. integrated autocorrelation time along the walker index ---------------------------
    print("\n  [1] tau_int along the WALKER index (0.5 = independent; "
          "error inflation = sqrt(2 tau_int))")
    print("      tau_idx  tau       Ws2            C3_N           ratio C3/Ws2")
    probe = sorted({0, 1, n_tau // 4, n_tau // 2, 3 * n_tau // 4, n_tau - 1})
    worst = 0.5
    for i_t in probe:
        w, c = Ws2[i_t], C3[:, i_t]
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(np.abs(w) > 1e-300, c / w, np.nan)
        r = r[np.isfinite(r)]
        t_w, _ = tau_int(w)
        t_c, _ = tau_int(c)
        t_r, _ = tau_int(r) if len(r) > 8 else (np.nan, 0)
        worst = max(worst, t_w, t_c, t_r if np.isfinite(t_r) else 0.5)
        print(f"      {i_t:<8d} {tau[i_t]:<9.3g} {t_w:<14.3f} {t_c:<14.3f} {t_r:.3f}")
    print(f"      worst tau_int = {worst:.3f}  =>  error inflation sqrt(2 tau_int) = "
          f"{np.sqrt(2 * worst):.3f}x")

    # --- 2. binning plateau on the ratio estimator ------------------------------------------
    # The form factor at a given tau is <C3>/<Ws2>; its bootstrap error must stop growing once
    # the bin size exceeds the correlation length.
    print("\n  [2] binning plateau: bootstrap error of <C3_N>/<Ws2> vs bin size")
    i_mid = n_tau // 2
    for i_t in sorted({1, i_mid, n_tau - 1}):
        errs = [(b, binned_ratio_err(C3[:, i_t], Ws2[i_t], b, args.n_boot, rng))
                for b in args.bins]
        errs = [(b, e) for b, e in errs if np.isfinite(e)]
        if not errs:
            continue
        base = errs[0][1]
        if base <= 0:
            # Q=0: J_N = N_coord * 1, so C3/Ws2 is identically N_coord for every walker and the
            # ratio has exactly zero variance. Nothing to plateau.
            print(f"      tau={tau[i_t]:<8.3g} error identically 0 "
                  f"(C3/Ws2 = N_coord per walker; Q=0)")
            continue
        cells = "  ".join(f"B={b}:{e:.2e}({e / base:.2f}x)" for b, e in errs)
        print(f"      tau={tau[i_t]:<8.3g} {cells}")
    print("      (flat in B => walkers independent, n_block irrelevant; rising then flat =>"
          " the plateau is the honest error)")
    return worst


def main():
    rng = np.random.default_rng(args.seed)
    worst = [report(fn, rng) for fn in args.files]
    print(f"\n{'=' * 78}")
    print(f"worst tau_int over all files = {max(worst):.3f}  "
          f"(inflation {np.sqrt(2 * max(worst)):.3f}x)")
    if max(worst) < 0.75:
        print("=> walkers are effectively independent; the quoted errors need no inflation.")
    else:
        print("=> NON-TRIVIAL autocorrelation: inflate errors by sqrt(2 tau_int), or bin harder.")


main()
