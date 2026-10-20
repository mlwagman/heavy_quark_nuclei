r"""Is there an SNR criterion on the GFMC weights that adaptively sets the Ritz cap m_max?

Motivation
----------
``Lanczos_twoleg.py``'s ``--max_tau_pts`` help states the rule *"keep 2*N_ritz_max <<
sqrt(N_samps)"* but nothing enforces it, and it ignores the single most important fact about GFMC
data: the weight variance grows exponentially, so how far the Ritz analysis can be pushed depends
on the *system*, not just on the sample count.  Measured growth rates span an order of magnitude
(Coulomb meson :math:`c \approx 0.04`, Nc=3 baryon :math:`c \approx 0.27`), and the observed
breakdown tracked them: the meson survived to m=12, the baryon died past m=9.

The structural fact that turns this into a criterion: the Rayleigh-Ritz pencil at iteration
:math:`m` is built from :math:`H^{(0)}_{st} = C(s{+}t)` and :math:`H^{(1)}_{st} = C(s{+}t{+}1)` for
:math:`s,t < m`, so it reaches two-point time index

.. math::  k_{\max}(m) = 2m - 1 .

So the question is simply: at what signal-to-noise on :math:`C_2(k_{\max})` does the extraction
stop being trustworthy?

.. math::
    {\rm SNR}(k) \;=\; \frac{|\langle C_2(k)\rangle|}{\sigma(C_2(k))/\sqrt{N_{\rm samps}}}
                 \;=\; \frac{\sqrt{N_{\rm samps}}}{{\rm CV}(k)},
    \qquad {\rm CV}(k) = \frac{\sigma(C_2(k))}{\langle C_2(k)\rangle} .

Two properties make this the right variable:

* **It is shift-invariant.**  The :math:`e^{-E_{\rm shift} k \Delta}` factor is a constant at fixed
  :math:`k`, so it cancels in CV.  The criterion is therefore a property of the raw weights, not of
  the analysis gauge -- the same invariance the matrix element itself has.
* **It is exponential in** :math:`k`, so :math:`{\rm SNR}(k) \approx \sqrt{N}\,{\rm CV}_0^{-1}
  e^{-\lambda k}` and the cap has a closed form (reported below), which also lets it be
  *predicted* for a planned run rather than only measured after the fact.

Ground truth, for free
----------------------
At :math:`Q=0` the number operator is :math:`J_N = N_{\rm coord}\mathbb{1}`, so
:math:`J^{(m)} = N_{\rm coord}` **exactly, at every m, for every coupling**.  Any deviation is the
extraction breaking down, not physics.  That gives a per-:math:`m` error measurement with no
reference calculation needed, in every single run -- which is what this study calibrates the SNR
threshold against.

What it does
------------
For each :math:`Q=0` file: measure CV(k) and SNR(k) on the same step-:math:`\Delta` two-point
correlator the analysis consumes, fit :math:`\lambda`, run the production extraction once at a high
cap to harvest :math:`J^{(m)}` for every :math:`m`, and tabulate the charge-check error against
SNR(2m-1).  Pooling over systems then asks whether one SNR threshold separates the good :math:`m`
from the bad across all of them.

Usage:
    python3.12 mmax_snr_study.py                     # the built-in dataset list
    python3.12 mmax_snr_study.py --files A.h5 B.h5 --labels A B
"""

import argparse
import glob
import os
import re
import subprocess
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, glob for the Q=0 file).  Spans growth rate c ~ 0.028 .. 0.27 and N_walkers 500 / 1000,
# which is the range over which a single threshold has to work if it is to be useful.
DEFAULT = [
    ('coul_meson_a0.3_500w', 'mike_20260824/out/coul_meson_a0.3/currents_3pt_*Qmag0.0.h5'),
    ('coul_meson_a0.3_1000w', 'nw1000/coul_meson_a0.3/currents_3pt_*Qmag0.0.h5'),
    ('coul_meson_a0.75', 'atomic_units/coul_meson_a0.75/currents_3pt_*Qmag0.0.h5'),
    ('coul_baryon_a0.75', 'atomic_units/coul_baryon_a0.75/currents_3pt_*Qmag0.0.h5'),
    ('mrs_lo_meson_a0.3', 'mrs_korr/mrs_lo_meson_a0.3/currents_3pt_*Qmag0.0.h5'),
]

p = argparse.ArgumentParser()
p.add_argument('--files', nargs='+', default=None)
p.add_argument('--labels', nargs='+', default=None)
p.add_argument('--n_block', type=int, default=1, help='walkers per sample (1 = all independent)')
p.add_argument('--m_max_probe', type=int, default=20, help='cap used to harvest J^(m) for all m')
p.add_argument('--tol', type=float, default=0.005,
               help='fractional charge-check deviation that counts as "still good" (secondary; '
                    'the primary classifier is the pull below)')
p.add_argument('--n_sigma', type=float, default=3.0,
               help='an iteration counts as BROKEN if its error is non-finite, or if the exact '
                    'Q=0 charge check misses N_coord by more than this many of its OWN sigma. '
                    'Comparing to a fixed fractional tolerance instead would mostly flag '
                    'statistical scatter in J^(m), not breakdown.')
p.add_argument('--py', default='/usr/local/bin/python3.12')
args = p.parse_args()


def bin_walkers(x, n_block):
    n_w = x.shape[-1]
    n_s = n_w // n_block
    head = x[..., :n_s * n_block].reshape(*x.shape[:-1], n_s, n_block).mean(axis=-1)
    return np.moveaxis(head, -1, 0)


def antidiagonal_c2(ws2_od):
    r"""``C_2(k\Delta)`` for ``k = 0 .. 2(N_tau-1)``, the grid the two-time analysis uses."""
    n_tau, _, n_w = ws2_od.shape
    n_k = 2 * (n_tau - 1) + 1
    c2 = np.zeros((n_k, n_w))
    for k in range(n_k):
        cells = [(i, k - i) for i in range(n_tau) if 0 <= k - i < n_tau]
        c2[k] = np.mean([ws2_od[i, j] for i, j in cells], axis=0)
    return c2


def snr_profile(fn, n_block):
    """``(k, CV, SNR, lambda, N_samps, N_coord, c)`` for one file."""
    with h5py.File(fn, 'r') as f:
        if 'Ws2_offdiag' not in f:
            return None
        ws2_od = np.real(f['Ws2_offdiag'][:])
        tau = f['tau_values'][:]
        n_coord = len(f['masses'][:])
    dt = float(tau[1] - tau[0])
    c2_w = antidiagonal_c2(ws2_od)                      # (Nk, N_walkers)
    c2 = bin_walkers(c2_w, n_block)                     # (N_samps, Nk)
    n_samps = c2.shape[0]
    mean = c2.mean(axis=0)
    std = c2.std(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.abs(std / mean)
        snr = np.sqrt(n_samps) / cv                     # SNR of the *mean*; shift-invariant
    k = np.arange(len(mean))
    # growth rate of the correlator itself, for context (this is Ben's `c`)
    n_fit = min(11, len(mean))
    c_growth = float(np.polyfit(k[:n_fit] * dt, np.log(np.maximum(mean[:n_fit], 1e-300)), 1)[0])
    # exponential growth of the noise-to-signal: CV(k) ~ CV_0 exp(lambda k)
    good = np.isfinite(cv) & (cv > 0) & (k > 0)
    lam = float(np.polyfit(k[good], np.log(cv[good]), 1)[0]) if good.sum() > 3 else np.nan
    return dict(k=k, cv=cv, snr=snr, lam=lam, n_samps=n_samps, n_coord=n_coord,
                c=c_growth, dt=dt, n_k=len(mean))


def harvest_jm(fn, n_block, m_max):
    """``{m: (J, err)}`` from one production run at Q=0 -- the exact-answer ground truth."""
    cmd = [args.py, os.path.join(HERE, 'gfmc_me_korr.py'), '--quiet',
           '--currents_file', fn, '--n_block', str(n_block), '--m_max', str(m_max),
           '--out_name', '/tmp/mmax_snr_probe_korr_ME.csv']
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    out = {}
    for line in r.stdout.splitlines():
        mm = re.match(r'\s*m=\s*(\d+)\s+([-+0-9.eE]+)\s+\+/-\s+(\S+)', line)
        if mm:
            m = int(mm.group(1))
            val = float(mm.group(2))
            try:
                err = float(mm.group(3))
            except ValueError:
                err = np.nan
            out[m] = (val, err)
    if not out:
        print(f"    (no J^(m) harvested; stderr tail: {r.stderr[-200:]!s})")
    return out


def main():
    if args.files:
        datasets = list(zip(args.labels or [os.path.basename(f) for f in args.files], args.files))
    else:
        datasets = []
        for label, pat in DEFAULT:
            hits = [f for f in glob.glob(os.path.join(HERE, pat))
                    if 'korr' not in os.path.basename(f)]
            if hits:
                datasets.append((label, hits[0]))
            else:
                print(f"skip {label}: no match for {pat}")
    if not datasets:
        sys.exit("no datasets found")

    rows = []       # (label, m, snr_at_2m_1, dev, err_finite)
    print("=" * 96)
    print("PART 1 -- weight SNR profile and the exact Q=0 charge check, per system")
    print("=" * 96)
    for label, fn in datasets:
        prof = snr_profile(fn, args.n_block)
        if prof is None:
            print(f"\n{label}: no Ws2_offdiag, skipping")
            continue
        print(f"\n--- {label}")
        print(f"    N_samps={prof['n_samps']}  N_coord={prof['n_coord']}  "
              f"Nk={prof['n_k']}  dt={prof['dt']:.4f}")
        print(f"    correlator growth c = {prof['c']:+.6f} / tau   "
              f"noise growth lambda = {prof['lam']:+.6f} / step")
        # korr refuses m_max > Nt//2 (its own message: "the Hankel matrices of iteration m need
        # C(t) through t=2m-1"), which is the same structural fact this study is built on.
        m_probe = min(args.m_max_probe, prof['n_k'] // 2)
        jm = harvest_jm(fn, args.n_block, m_probe)
        if not jm:
            continue
        print(f"    probe m_max = {m_probe} (capped by Nk//2 = {prof['n_k'] // 2})")
        print(f"    {'m':>3} {'k=2m-1':>7} {'SNR(k)':>10} {'CV(k)':>9} "
              f"{'J^(m)':>11} {'err':>10} {'|dev|':>9} {'pull':>7}")
        for m in sorted(jm):
            k = 2 * m - 1
            if k >= prof['n_k']:
                continue
            val, err = jm[m]
            dev_abs = abs(val - prof['n_coord'])
            dev = dev_abs / prof['n_coord']
            # The deviation must be judged against J^(m)'s own error: at large m the charge check
            # scatters by its statistical error, and calling that "breakdown" would just be
            # flagging noise.  Breakdown is a non-finite error or a pull far outside it.
            #
            # m=1 is the exception that must not be mis-flagged: there the Ritz projector is
            # trivial, J^(1) reproduces N_coord identically and the error is *exactly* zero, so
            # dev/err is 0/0.  That is the most perfect iteration, not a broken one.
            exact = np.isfinite(val) and dev_abs == 0.0
            fin = np.isfinite(err) and np.isfinite(val) and err > 0
            if exact:
                pull, broken = 0.0, False
            elif fin:
                pull = dev_abs / err
                broken = pull > args.n_sigma
            else:
                pull, broken = np.inf, True
            flag = '   <-- BROKEN' if broken else ''
            ps = f"{pull:>7.2f}" if np.isfinite(pull) else f"{'inf':>7}"
            print(f"    {m:>3} {k:>7} {prof['snr'][k]:>10.2f} {prof['cv'][k]:>9.4f} "
                  f"{val:>11.6f} {err:>10.6f} {dev:>9.5f} {ps}{flag}")
            rows.append((label, m, float(prof['snr'][k]), float(dev), not broken))

    if not rows:
        sys.exit("\nno rows collected")

    # ---------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("PART 2 -- is there ONE SNR threshold that separates good m from bad, across systems?")
    print("=" * 96)
    good = [r for r in rows if r[4]]
    bad = [r for r in rows if not r[4]]
    print(f"  {len(good)} usable (m, system) pairs, {len(bad)} broken   "
          f"(broken = non-finite error, or Q=0 pull > {args.n_sigma})")
    if good:
        print(f"  SNR over GOOD pairs: min {min(r[2] for r in good):8.2f}   "
              f"median {np.median([r[2] for r in good]):8.2f}")
    if bad:
        print(f"  SNR over BAD  pairs: max {max(r[2] for r in bad):8.2f}   "
              f"median {np.median([r[2] for r in bad]):8.2f}")

    print("\n  threshold scan: classify m as usable iff SNR(2m-1) >= S_min")
    print(f"    {'S_min':>8} {'kept_good':>10} {'kept_bad':>9} {'lost_good':>10} "
          f"{'purity':>8} {'recall':>8}")
    best = None
    for s_min in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300]:
        kg = sum(1 for r in good if r[2] >= s_min)
        kb = sum(1 for r in bad if r[2] >= s_min)
        lg = len(good) - kg
        purity = kg / max(kg + kb, 1)
        recall = kg / max(len(good), 1)
        print(f"    {s_min:>8} {kg:>10} {kb:>9} {lg:>10} {purity:>8.3f} {recall:>8.3f}")
        # prefer admitting no bad m at all, then as many good m as possible
        score = (kb == 0, kg)
        if best is None or score > best[0]:
            best = (score, s_min, kg, kb, lg)
    print(f"\n  -> tightest threshold admitting no bad m: S_min = {best[1]} "
          f"(keeps {best[2]}/{len(good)} good, admits {best[3]} bad)")

    print("\n" + "=" * 96)
    print("PART 3 -- the resulting adaptive rule, per system")
    print("=" * 96)
    print("  m_max = max{ m : SNR(2m-1) >= S_min },  SNR(k) = sqrt(N_samps)/CV(k)")
    print("  closed form from the exponential fit, for planning a run before taking the data:")
    print("      m_max ~ [ log( sqrt(N_samps) / (CV_0 S_min) ) / lambda + 1 ] / 2")
    print(f"\n  with S_min = {best[1]}:")
    print(f"    {'system':>24} {'lambda':>9} {'m_max(SNR)':>11} {'m_max(obs)':>11} "
          f"{'2m<<sqrt(N) rule':>18}")
    for label, fn in datasets:
        prof = snr_profile(fn, args.n_block)
        if prof is None:
            continue
        ok_m = [m for m in range(1, args.m_max_probe + 1)
                if 2 * m - 1 < prof['n_k'] and prof['snr'][2 * m - 1] >= best[1]]
        m_snr = max(ok_m) if ok_m else 0
        # largest m with NO broken iteration at or below it -- a cap has to be contiguous
        usable = {r[1] for r in rows if r[0] == label and r[4]}
        broken = {r[1] for r in rows if r[0] == label and not r[4]}
        m_obs = 0
        for m in sorted(usable | broken):
            if m in broken:
                break
            m_obs = m
        m_old = int(np.sqrt(prof['n_samps']) / 2)        # the "2m << sqrt(N)" reading, generously
        print(f"    {label:>24} {prof['lam']:>9.5f} {m_snr:>11} {m_obs:>11} {m_old:>18}")
    print("\n  m_max(obs) is the largest m whose exact Q=0 charge check still passes -- the")
    print("  ground truth the criterion has to reproduce.")


main()
