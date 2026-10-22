r"""Where do the GFMC weight spikes come from: discretization, or unbranched importance sampling?

The question
------------
A single walker carrying 24% of the total weight (Nc=3 baryon, walker 223 at 170x the cell median)
either is a :math:`d\tau` artifact, or is intrinsic to the sampling.  The two have completely
different fixes -- change :math:`d\tau`, versus add population control -- so it is worth settling.

The structural expectation
--------------------------
``--resampling`` defaults to ``None``, so **there is no branching**: each walker's weight is a free
running product over steps,

.. math::  W(\tau) \;=\; \prod_i e^{-d\tau\,(V(R_i) - E_T)} \times (\text{importance ratios}),

so :math:`\log W` is a *sum* of many roughly independent increments.  By the CLT :math:`\log W` is
asymptotically Gaussian -- :math:`W` is **log-normal** -- with

.. math::  \sigma^2(\tau) \equiv {\rm Var}[\log W] \;\simeq\; 2\lambda\,\tau ,

growing without bound.  This is the textbook reason diffusion Monte Carlo branches.  Three
consequences are testable here:

1. :math:`{\rm CV} = \sqrt{e^{\sigma^2}-1}`, so CV grows **exponentially** in :math:`\tau` -- which
   is what the earlier SNR study measured (λ per step 0.022-0.080).
2. The sample mean of a log-normal is dominated by its upper tail, and the **Kish effective sample
   size** :math:`N_{\rm eff} = (\sum_w |w|)^2 / \sum_w w^2` collapses as
   :math:`N_{\rm eff}/N \simeq e^{-\sigma^2}`.  This, not CV, is the number of walkers actually
   contributing to a Hankel entry.
3. To leading order :math:`\sigma^2` is a property of the *continuum* diffusion over total time
   :math:`\tau`.  It should therefore scale with the physical time and the energy scale, and be
   **insensitive to** :math:`d\tau` at fixed :math:`\tau` -- discretization enters the bias, not the
   variance growth.  If that holds, the spikes are intrinsic and no choice of :math:`d\tau` removes
   them.

What this script checks
-----------------------
* is :math:`\log|W|` actually Gaussian (skew / excess kurtosis / Shapiro), i.e. is log-normal the
  right model;
* does the observed extreme **exceed** the log-normal prediction for the max of N draws (if so,
  something worse than log-normal is going on and population control is not the whole story);
* :math:`N_{\rm eff}(k)` versus the Hankel reach :math:`k = 2m-1`, which is the natural replacement
  for the failed SNR criterion -- the ``2m << sqrt(N_samps)`` rule should arguably read
  :math:`N_{\rm eff}`, not :math:`N_{\rm samps}`;
* whether :math:`\lambda` per unit *physical* :math:`\tau`, scaled by :math:`|E_0|`, is common
  across runs with very different :math:`d\tau` -- the discretization-versus-physics discriminator.

Usage:  python3.12 weight_tails.py
"""

import glob
import os

import h5py
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# label, dir, dtau_iMev, count_skip, |E_0| (exact or best reference)
SETS = [
    ('coul_meson_a0.3', 'mike_20260824/out/coul_meson_a0.3', 0.5, 5, 0.0400),
    ('coul_meson_a0.3_1000w', 'nw1000/coul_meson_a0.3', 0.5, 5, 0.0400),
    ('coul_meson_a0.75', 'atomic_units/coul_meson_a0.75', 0.08, 5, 0.2500),
    ('coul_baryon_a0.75', 'atomic_units/coul_baryon_a0.75', 0.048, 8, 0.267838),
    ('mrs_lo_meson_a0.3', 'mrs_korr/mrs_lo_meson_a0.3', 0.5, 5, 0.028033),
]


def q0_file(d):
    hits = [f for f in glob.glob(os.path.join(HERE, d, 'currents_3pt_*Qmag0.0.h5'))
            if 'korr' not in os.path.basename(f)]
    return hits[0] if hits else None


def n_eff(w):
    """Kish effective sample size: how many walkers actually carry the mean."""
    s1 = np.abs(w).sum()
    s2 = (w ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def main():
    print("=" * 100)
    print("PART 1 -- is log|W| Gaussian (is W log-normal), and does the tail exceed log-normal?")
    print("=" * 100)
    print("  Using the leg-1 row i=0 of Ws2_offdiag, i.e. the longest leg-2 evolution,")
    print("  at the largest stored time -- where the weight spread is widest.\n")
    rows = []
    for label, d, dtau, skip, E0 in SETS:
        fn = q0_file(d)
        if fn is None:
            print(f"  skip {label}")
            continue
        with h5py.File(fn, 'r') as f:
            od = np.real(f['Ws2_offdiag'][:])
            tau = f['tau_values'][:]
        n_tau, _, n_w = od.shape
        D = float(tau[1] - tau[0])
        w = od[0, n_tau - 1]                       # (N_w,)
        pos = w[w > 0]
        lw = np.log(pos)
        sig = float(lw.std(ddof=1))
        skew = float(stats.skew(lw))
        kurt = float(stats.kurtosis(lw))           # excess
        # Shapiro on a subsample (it is unreliable above ~5000 points)
        sh_p = float(stats.shapiro(lw[:500]).pvalue) if len(lw) >= 20 else np.nan
        # log-normal prediction for the max of N draws: exp(sigma * z_{1-1/N})
        z = stats.norm.ppf(1.0 - 1.0 / len(pos))
        pred_max_over_med = float(np.exp(sig * z))
        obs_max_over_med = float(np.abs(w).max() / np.median(np.abs(w)))
        print(f"--- {label}   (N_w={n_w}, dtau={dtau}, D={D:.3f}, tau_max={tau[-1]:.2f})")
        print(f"    log|W|:  sigma={sig:.3f}  skew={skew:+.3f}  excess kurt={kurt:+.3f}  "
              f"Shapiro p={sh_p:.3f}" + ("  (Gaussian)" if sh_p > 0.05 else "  (non-Gaussian)"))
        print(f"    max/median:  observed {obs_max_over_med:7.1f}x   "
              f"log-normal predicts {pred_max_over_med:7.1f}x   "
              f"ratio {obs_max_over_med / pred_max_over_med:5.2f}")
        print(f"    top walker carries {100 * np.abs(w).max() / np.abs(w).sum():.1f}% of sum|w|; "
              f"N_eff = {n_eff(w):.0f} of {n_w} ({100 * n_eff(w) / n_w:.1f}%)")
        rows.append((label, dtau, D, E0, sig, n_tau, od, tau, n_w))

    print("\n" + "=" * 100)
    print("PART 2 -- N_eff(k) vs the Hankel reach k = 2m-1")
    print("=" * 100)
    print("  The '2m << sqrt(N_samps)' rule arguably should read N_eff, not N_samps: N_eff is how")
    print("  many walkers actually carry the Hankel entry at that time.\n")
    for label, dtau, D, E0, sig, n_tau, od, tau, n_w in rows:
        n_k = 2 * (n_tau - 1) + 1
        c2 = np.zeros((n_k, n_w))
        for k in range(n_k):
            cells = [(i, k - i) for i in range(n_tau) if 0 <= k - i < n_tau]
            c2[k] = np.mean([od[i, j] for i, j in cells], axis=0)
        print(f"--- {label}")
        print(f"    {'k':>4} {'m=(k+1)/2':>10} {'N_eff':>9} {'N_eff/N':>9} "
              f"{'sqrt(N_eff)':>12} {'2m':>5} {'2m<=sqrt(N_eff)?':>18}")
        for k in range(1, n_k, max(1, n_k // 8)):
            ne = n_eff(c2[k])
            m = (k + 1) // 2
            ok = 'yes' if 2 * m <= np.sqrt(ne) else 'NO'
            print(f"    {k:>4} {m:>10} {ne:>9.0f} {100 * ne / n_w:>8.1f}% "
                  f"{np.sqrt(ne):>12.1f} {2 * m:>5} {ok:>18}")

    print("\n" + "=" * 100)
    print("PART 3 -- is the variance growth set by physics or by discretization?")
    print("=" * 100)
    print("  sigma^2(tau) ~ 2 lambda tau is a continuum property, so lambda per unit *physical*")
    print("  tau, in units of |E_0|, should be comparable across runs with different dtau.")
    print("  If instead it tracked dtau, the spikes would be a discretization artifact.\n")
    print(f"  {'system':>24} {'dtau':>7} {'D':>7} {'|E_0|':>9} {'sigma':>7} "
          f"{'lam_tau=sig^2/(2 tau)':>22} {'/|E_0|':>9} {'dtau*|E_0|':>11}")
    for label, dtau, D, E0, sig, n_tau, od, tau, n_w in rows:
        tau_tot = float(tau[-1])                   # leg-1 at 0, leg-2 to the end
        lam_tau = sig ** 2 / (2 * tau_tot) if tau_tot > 0 else np.nan
        print(f"  {label:>24} {dtau:>7} {D:>7.3f} {E0:>9.5f} {sig:>7.3f} "
              f"{lam_tau:>22.5f} {lam_tau / E0:>9.3f} {dtau * E0:>11.5f}")
    print("\n  Read the last two columns together: runs that share dtau*|E_0| are the *same*")
    print("  calculation under the dtau ~ 1/alpha^2 scaling rule, so agreement there is a")
    print("  consistency check rather than evidence. Disagreement in lam_tau/|E_0| across")
    print("  *different* systems is expected (it depends on the potential's fluctuations), but")
    print("  a common value across different dtau at matched physics points to the continuum.")


main()
