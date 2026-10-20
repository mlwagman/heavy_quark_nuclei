r"""Ground-state matrix elements from two-leg GFMC data, via korr's Ritz rotators.

This is the "hydrogen atom" case of the FRR/LME three-point workflow: a Coulomb bound state whose
energies, wavefunction and form factor are all known in closed form, measured by GFMC and then
analysed exactly as a lattice three-point function.  It is written to stand alone (one HDF5 file
in, one CSV out) so that it can become a korr tutorial or regression test.

What is being computed
----------------------
The two-leg GFMC estimator produces

.. math::
    C_J(\sigma,\tau) \;=\; \langle\psi|\,e^{-H\sigma}\,J\,e^{-H\tau}\,|\psi\rangle
    \;=\;\sum_{f,i} Z'_f\,J_{fi}\,Z_i\;e^{-E_f\sigma - E_i\tau},

with :math:`\tau` the source-to-current separation (leg 1, where the current is evaluated) and
:math:`\sigma` the current-to-sink separation (leg 2).  korr's convention is the same
(:func:`korr.synth.make_3pt`), so the generator's ``C3_N_offdiag[i, j]`` -- leg-1 index first --
maps to ``C3pt[sigma, tau] = C3_N_offdiag[tau, sigma]``.  The partner two-point function is
``Ws2_offdiag[i, j] = C_2((t_i + t_j)\Delta)``, a genuine Hankel matrix on the grid step
:math:`\Delta`.

Why the two-time data and not the diagonal
------------------------------------------
``Lanczos_twoleg.py:213`` builds the three-point Hankel as ``C3pt[:, s, t] = C3_binned[:, s + t]``
from the *diagonal* array ``C3_N[:, k] = C_J(k\Delta, k\Delta)``.  Its two-point partner,
``Ws2_all[k] = C_2(2k\Delta)``, is a valid Hankel in step :math:`2\Delta` -- so the rotators live
on that grid and the entry needed is :math:`C_J(2s\Delta, 2t\Delta)`, not
:math:`C_J((s{+}t)\Delta, (s{+}t)\Delta)`.  Spectrally the supplied entry carries
:math:`e^{-(E_f+E_i)(s+t)\Delta}` where :math:`e^{-2E_f s\Delta - 2E_i t\Delta}` is required: equal
only for :math:`f = i`.  The diagonal spectral terms survive; the :math:`f \neq i` interference
terms -- what the Ritz projector exists to cancel -- do not.  At :math:`Q = 0` the number operator
:math:`J_N = \sum_k e^{iq\cdot x_k} = N_{\rm coord}\mathbb{1}` commutes with :math:`H`, so
:math:`J_{fi}\propto\delta_{fi}` and the defect is exactly inert; it switches on at
:math:`Q \neq 0`, which is where the form-factor shape lives.  ``--hankel legacy`` reproduces that
construction through this same korr backend, so the Hankel choice can be varied with nothing else
changing.

The shift is a gauge choice
---------------------------
GFMC weights grow, and the Rayleigh-Ritz pencil wants a decaying correlator, so both correlators
are multiplied by :math:`e^{-E_{\rm shift} t_{\rm tot}}` -- :math:`t_{\rm tot} = k\Delta` for the
two-point function and :math:`(\sigma + \tau)\Delta` for the three-point one.  Because the factor
is *the same function of the total time* on both sides, it cancels identically in
:math:`J = P^L C^{\rm 3pt} P^R`: it rescales the Ritz values and the rotator normalization in
compensating ways.  ``--check_shift_invariance`` verifies this on the data rather than trusting it,
and it is the cleanest available test that the normalization convention is being respected.
(``Lanczos_twoleg.py`` needed ``E_shift = c/2`` because its ``tau_values`` label half the total
time; here the grid is the total time throughout, so no factor of two appears -- and the
``/2.0`` energy fix-up at its line 173 is unnecessary.)

Usage
-----
    python3.12 gfmc_me_korr.py --currents_file currents_3pt_*.h5 [options]
    python3.12 gfmc_me_korr.py --self_test        # synthetic closed-form check, no data needed

Requires korr (python >= 3.12).  The GFMC generator itself runs under a different interpreter
(it needs jax); the HDF5 file is the interface.
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------------------------
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--currents_file', help='a single currents_3pt_*.h5')
p.add_argument('--dir', help='analyse a whole output directory and write prod_ME.csv; a drop-in '
                            'replacement for prod_extract_lanczos.py (which drives korr_dev)')
p.add_argument('--prod_name', default='prod_ME.csv', help='--dir output name')
p.add_argument('--out_suffix', default='_korr_ME',
               help='suffix for the per-Q CSV/boot files, so that two_time and legacy runs over '
                    'the same directory do not overwrite each other')
p.add_argument('--calibrate_m_max', action='store_true', default=True,
               help='with --dir, pick m_max as the largest value still passing the exact Q=0 '
                    'charge check (F(0) == N_coord). On by default; --m_max overrides it.')
p.add_argument('--no_calibrate_m_max', dest='calibrate_m_max', action='store_false')
p.add_argument('--m_max_ladder', type=int, nargs='+', default=None,
               help='candidate m_max values for the calibration, ascending')
p.add_argument('--charge_tol', type=float, default=0.0025,
               help='VETO threshold: reject an m_max whose exact Q=0 charge check misses N_coord '
                    'by more than this fraction. 0.0025 separates the stable region from the '
                    'broken one on both systems measured (MRS meson: 0.006-0.145%% stable vs '
                    '0.46-0.55%% broken; Nc=3 baryon after spike screening: all <= 0.149%%). '
                    'Calibrated on two systems only -- re-check it on a new one.')
p.add_argument('--spike_leverage', type=float, default=0.0,
               help='drop walkers whose leverage max_ij |w|/median_w|w| exceeds this (0 = off). '
                    'A single spiking walker biases the Hankel means directly and binning cannot '
                    'remove it. With --dir the drop set is fixed once from the Q=0 file so the '
                    'cross-Q bootstrap ensembles stay aligned.')
p.add_argument('--current', default='C3_N',
               help="which current; only C3_N has two-time data stored")
p.add_argument('--n_block', type=int, default=5, help='walkers averaged per sample')
p.add_argument('--n_boot', type=int, default=200, help='outer = inner bootstrap draws')
p.add_argument('--seed', type=int, default=17)
p.add_argument('--hankel', choices=['two_time', 'legacy'], default='two_time',
               help="'two_time' uses C3_N_offdiag (correct); 'legacy' reproduces "
                    "Lanczos_twoleg.py's C3_N[s+t] through this same backend")
p.add_argument('--extra_decay_rel', type=float, default=0.1875,
               help='shifted decay rate as a fraction of the measured growth rate |c|; the '
                    'answer must not depend on it (see --check_shift_invariance)')
p.add_argument('--extra_decay', type=float, default=None,
               help='absolute override for the above. A fixed absolute value is not '
                    'scale-covariant: c ~ alpha^2, so at small alpha it underflows the '
                    'shifted correlator. Prefer the relative form.')
p.add_argument('--n_fit_growth', type=int, default=11,
               help='early total-time points used to measure the growth rate c')
p.add_argument('--m_window', type=int, default=3)
p.add_argument('--final_m', type=int, default=None)
p.add_argument('--m_max', type=int, default=None)
p.add_argument('--max_tau_pts', type=int, default=None,
               help='truncate the tau grid to this many points before anything else')
p.add_argument('--check_shift_invariance', action='store_true',
               help='rerun with a different E_shift and confirm J is unchanged')
p.add_argument('--out_name', default=None, help='output CSV (default <file>_korr_ME.csv)')
p.add_argument('--self_test', action='store_true',
               help='run the synthetic closed-form check and exit')
p.add_argument('--quiet', action='store_true',
               help="suppress korr's full spectroscopy report and its internal RuntimeWarnings; "
                    "the matrix-element summary is still printed")
args = p.parse_args()

if args.quiet:
    # korr's internals divide by zero at degenerate iterations and recover; the RuntimeWarnings
    # are noise, not diagnostics, and they bury the result.
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

from korr.analysis import Config, RitzRotators, matrix_elements, spectroscopy  # noqa: E402

# summarize=False suppresses the full spectroscopy report; the matrix element is the deliverable
# here and the spectrum is summarized in two lines below.
_SUMMARIZE = not args.quiet


def say(*a):
    print(*a)


# --------------------------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------------------------
def bin_walkers(x, n_block):
    """Average blocks of ``n_block`` along the LAST axis, returning it FIRST.

    ``x`` is ``(..., N_walkers)`` and the result is ``(N_samps, ...)`` with
    ``N_samps = N_walkers // n_block``, the sample axis korr expects to lead.
    """
    n_w = x.shape[-1]
    n_s = n_w // n_block
    if n_s < 2:
        raise ValueError(f"{n_w} walkers / n_block={n_block} leaves {n_s} samples; need >= 2")
    head = x[..., :n_s * n_block].reshape(*x.shape[:-1], n_s, n_block).mean(axis=-1)
    return np.moveaxis(head, -1, 0)


def walker_leverage(ws2_od):
    r"""Per-walker spike score: :math:`\max_{ij} |w_{ij}| / {\rm median}_w |w_{ij}|`.

    GFMC importance weights occasionally blow up on a single walker, and because the Hankel
    entries are sample *means* such a walker biases them directly.  Binning cannot help -- it
    leaves the grand mean exactly invariant -- so the spike has to be dealt with in the data.
    (``Lanczos_twoleg.py`` handled the same phenomenon by interpolating whole time slices whose CV
    exceeded ``--spike_thresh``; this is the per-walker version, which does not touch the
    unaffected walkers at a contaminated time.)

    The score is deliberately a max over cells rather than an average: a walker that is fine
    almost everywhere and catastrophic in one cell is exactly the case that matters.  Measured
    scores separate cleanly -- the Nc=3 baryon has one walker at 170x with the next at 6x, while
    the meson ensembles top out around 16-37x with a smooth tail and no outlier.
    """
    lev = np.zeros(ws2_od.shape[2])
    for i in range(ws2_od.shape[0]):
        for j in range(ws2_od.shape[1]):
            med = np.median(np.abs(ws2_od[i, j]))
            if med > 0:
                lev = np.maximum(lev, np.abs(ws2_od[i, j]) / med)
    return lev


def spike_walkers(fn, thresh):
    """``(indices, leverage)`` of walkers above the leverage threshold in one file."""
    import h5py
    with h5py.File(fn, 'r') as f:
        if 'Ws2_offdiag' not in f:
            return np.array([], dtype=int), None
        ws2_od = np.real(f['Ws2_offdiag'][:])
    lev = walker_leverage(ws2_od)
    return np.where(lev > thresh)[0], lev


def load(fn, current, max_tau_pts=None, drop_walkers=None):
    """Read the generator's output.  Returns a dict of real arrays plus metadata."""
    import h5py
    with h5py.File(fn, 'r') as f:
        need = (f'{current}_offdiag', 'Ws2_offdiag')
        if args.hankel == 'two_time':
            for k in need:
                if k not in f:
                    raise SystemExit(
                        f"{fn}: no '{k}'.\nOnly heavy_quark_nuclei_gfmc_boosted_FV_currents.py "
                        f"(Coulomb) stores two-time data, and only for C3_N. Use --hankel legacy "
                        f"to run the diagonal construction instead.")
        out = {
            'tau': np.asarray(f['tau_values'][:], dtype=float),
            'q2': float(f['q_squared'][()]),
            'masses': np.asarray(f['masses'][:], dtype=float),
            'Ws2_all': np.real(f['Ws2_all'][:]),                       # (N_tau, N_w)
            'C3_diag': np.real(f[current][:]),                         # (N_w, N_tau)
        }
        if f'{current}_offdiag' in f:
            out['C3_od'] = np.real(f[f'{current}_offdiag'][:])         # (N_tau, N_tau, N_w)
        if 'Ws2_offdiag' in f:
            out['Ws2_od'] = np.real(f['Ws2_offdiag'][:])               # (N_tau, N_tau, N_w)
    if out['C3_diag'].ndim != 2:
        raise SystemExit(f"{current} has shape {out['C3_diag'].shape}; this script handles the "
                         "scalar currents (C3_N, C3_S, C3_V0) only")
    if max_tau_pts is not None:
        n = min(max_tau_pts, len(out['tau']))
        out['tau'] = out['tau'][:n]
        out['Ws2_all'] = out['Ws2_all'][:n]
        out['C3_diag'] = out['C3_diag'][:, :n]
        for k in ('C3_od', 'Ws2_od'):
            if k in out:
                out[k] = out[k][:n, :n]

    if drop_walkers is not None and len(drop_walkers):
        n_w = out['Ws2_all'].shape[1]
        keep = np.setdiff1d(np.arange(n_w), np.asarray(drop_walkers, dtype=int))
        out['Ws2_all'] = out['Ws2_all'][:, keep]
        out['C3_diag'] = out['C3_diag'][keep, :]
        for k in ('C3_od', 'Ws2_od'):
            if k in out:
                out[k] = out[k][:, :, keep]
        out['n_dropped'] = int(n_w - len(keep))
    return out


def antidiagonal_c2(Ws2_od):
    r"""``C_2(k\Delta)`` for ``k = 0 .. 2(N_tau-1)`` from the two-time weights.

    ``Ws2_od[i, j]`` is :math:`C_2((t_i + t_j)\Delta)` and so depends on :math:`i + j` alone;
    averaging each antidiagonal is the minimum-variance estimator of that single number, and the
    spread within an antidiagonal is pure GFMC noise (``audit_c3_two_time.py`` check 2 measures
    it).  The resulting grid has step :math:`\Delta` and reaches :math:`2(N_\tau-1)`, exactly the
    range the three-point Hankel needs: iteration :math:`m` touches two-point index
    :math:`2m-1 \le 2N_\tau - 2`.
    """
    n_tau, _, n_w = Ws2_od.shape
    n_k = 2 * (n_tau - 1) + 1
    c2 = np.zeros((n_k, n_w))
    for k in range(n_k):
        cells = [(i, k - i) for i in range(n_tau) if 0 <= k - i < n_tau]
        c2[k] = np.mean([Ws2_od[i, j] for i, j in cells], axis=0)
    return c2


def growth_rate(c2_mean, dt, n_fit):
    r"""``c = d log <C_2> / d t_tot`` from the first ``n_fit`` points of the total-time grid."""
    n = min(n_fit, len(c2_mean))
    t = np.arange(n) * dt
    return float(np.polyfit(t, np.log(np.maximum(c2_mean[:n], 1e-300)), 1)[0])


def build(d, hankel, extra_decay_rel, extra_decay, n_block, n_fit_growth, verbose=True):
    r"""Assemble ``(C2, C3pt, info)`` on a single consistent grid, shifted and binned.

    ``C2`` is ``(N_samps, Nt)`` on a grid of step ``dt_unit`` in the *total* imaginary time;
    ``C3pt`` is ``(N_samps, Nsigma, Ntau)`` in korr's :math:`(\sigma,\tau)` layout, carrying the
    matching shift :math:`e^{-E_{\rm shift}(\sigma+\tau)\,dt_{\rm unit}}`.
    """
    dt_grid = float(d['tau'][1] - d['tau'][0])
    n_tau = len(d['tau'])

    if hankel == 'two_time':
        # the honest object: both legs free, one grid of step dt_grid in the total time
        c2_w = antidiagonal_c2(d['Ws2_od'])                    # (Nt, N_w), index k = t_tot/dt
        c3_w = np.swapaxes(d['C3_od'], 0, 1)                   # (sigma, tau, N_w)
        dt_unit = dt_grid
    else:
        # Lanczos_twoleg.py's construction: the diagonal array indexed by sigma+tau, with the
        # symmetric-point 2pt alongside. Both live on a grid of step 2*dt_grid in the total time.
        m_cap = (n_tau + 1) // 2
        c2_w = d['Ws2_all']                                    # (N_tau, N_w), index k -> 2k*dt
        c3_w = np.zeros((m_cap, m_cap, d['C3_diag'].shape[0]))
        for s in range(m_cap):
            for t in range(m_cap):
                c3_w[s, t] = d['C3_diag'][:, s + t]            # C_J((s+t)dt, (s+t)dt)  <- the bug
        dt_unit = 2.0 * dt_grid

    c2_mean = c2_w.mean(axis=-1)
    c = growth_rate(c2_mean, dt_unit, n_fit_growth)
    if extra_decay is not None:
        extra, src = extra_decay, f'absolute {extra_decay:.6g}'
    else:
        extra = extra_decay_rel * abs(c)
        src = f'{extra_decay_rel} * |c| = {extra:.6g}'
    e_shift = c + extra

    k = np.arange(c2_w.shape[0])
    shift_2pt = np.exp(-e_shift * k * dt_unit)
    n_sig, n_tau3 = c3_w.shape[0], c3_w.shape[1]
    shift_3pt = np.exp(-e_shift * (np.arange(n_sig)[:, None]
                                   + np.arange(n_tau3)[None, :]) * dt_unit)

    C2 = bin_walkers(c2_w * shift_2pt[:, None], n_block)                 # (N_samps, Nt)
    C3 = bin_walkers(c3_w * shift_3pt[..., None], n_block)               # (N_samps, Nsig, Ntau)

    if verbose:
        say(f"  hankel={hankel}  grid step in total time dt_unit={dt_unit:.4f} "
            f"(tau grid step {dt_grid:.4f})")
        say(f"  2pt: {C2.shape} (N_samps, Nt)   3pt: {C3.shape} (N_samps, Nsigma, Ntau)")
        say(f"  growth rate c = {c:.6g}  =>  E_shift = c + {src} = {e_shift:.6g}  "
            f"(shifted decay rate {extra:.6g})")
    info = {'dt_unit': dt_unit, 'dt_grid': dt_grid, 'c': c, 'E_shift': e_shift, 'extra': extra}
    return C2, C3, info


# --------------------------------------------------------------------------------------------
# the analysis
# --------------------------------------------------------------------------------------------
def run_korr(C2, C3, cfg, label='', verbose=True):
    """Spectroscopy -> Ritz rotators -> the FRR sandwich.  Returns ``(me_result, spec)``.

    The rotators come from one analysis of the same two-point function on both sides: the GFMC
    source and sink are the same trial state, so the primed and unprimed spectra of FRR section
    1.9 coincide.  Taking both from one ``SpectrumResult`` also makes the resampling fingerprints
    agree by construction, which is what ``matrix_elements`` checks -- no
    ``np.random.seed(17)``-and-hope, and no need for ``assume_aligned``.
    """
    spec = spectroscopy(C2, cfg)
    rot_L = RitzRotators.from_result(spec, side='L')      # final (sink) state
    rot_R = RitzRotators.from_result(spec, side='R')      # initial (source) state
    me = matrix_elements(C3, rot_L, rot_R, cfg)
    if verbose:
        say(f"\n  --- korr {label} ---")
        say(f"  iterations m: {me.all_m}   final_m={me.metadata.get('final_m')}  "
            f"m_window={cfg.m_window}")
        say(f"  alignment: {me.metadata.get('alignment')}")
    return me, spec


def ground_state(me):
    """``(J_00, err)`` -- the ground-state diagonal matrix element and its 1-sigma half-width."""
    return float(me.matrix[0, 0]), float(np.asarray(me.errors['final']['me'])[0, 0])


def shifted_energy(spec, info):
    r"""Ground-state energy relative to the trial energy, undoing the shift.

    The pencil measures the decay rate of the *shifted* correlator in units of ``dt_unit``; the
    physical rate is that over ``dt_unit`` minus ``E_shift``.  Since the GFMC weights already
    carry the trial-energy subtraction, what comes out is :math:`E_0 - E_T`, which cancels in the
    form-factor ratio.
    """
    e = np.atleast_1d(np.asarray(spec.final['energies'], dtype=float))
    err = np.atleast_1d(np.asarray(spec.errors['final']['energies'], dtype=float))
    return e / info['dt_unit'] - info['E_shift'], err / info['dt_unit']


def analyse(fn, m_max=None, out_name=None, verbose=True, drop_walkers=None):
    m_max = args.m_max if m_max is None else m_max
    d = load(fn, args.current, args.max_tau_pts, drop_walkers)
    n_coord = len(d['masses'])
    if verbose:
        say(f"\n{'=' * 78}\n{os.path.basename(fn)}\n{'=' * 78}")
        say(f"  Q^2 = {d['q2']:.6g}  (Q = {np.sqrt(d['q2']):.6g})   N_coord = {n_coord}   "
            f"N_tau = {len(d['tau'])}   N_walkers = {d['Ws2_all'].shape[1]}"
            + (f"  ({d['n_dropped']} spiking walker(s) dropped)" if d.get('n_dropped') else ""))

    C2, C3, info = build(d, args.hankel, args.extra_decay_rel, args.extra_decay,
                         args.n_block, args.n_fit_growth, verbose=verbose)
    cfg = Config(n_outer=args.n_boot, n_inner=args.n_boot, seed=args.seed,
                 m_window=args.m_window, final_m=args.final_m, m_max=m_max,
                 ritz_coeffs=True, progress=False, summarize=_SUMMARIZE and verbose)
    me, spec = run_korr(C2, C3, cfg, label=args.hankel, verbose=verbose)

    F, F_err = ground_state(me)
    E, E_err = shifted_energy(spec, info)

    # the tau=0 (VMC) value, for continuity with Lanczos_twoleg.py's F_m1 column
    w0 = d['Ws2_all'][0]
    F_m1 = float((d['C3_diag'][:, 0] * w0).sum() / w0.sum())

    stack = me.stack_by_m()[:, 0, 0]
    stack_err = me.stack_by_m(source='err')[:, 0, 0]
    if verbose:
        say("\n  J^(m) vs m (ground state):")
        for m, v, e in zip(me.all_m, stack, stack_err):
            say(f"    m={m:3d}   {v:+.6f} +/- {e:.6f}")
        say(f"\n  E_0 - E_T   = {E[0]:+.6f} +/- {E_err[0]:.6f}"
            + (f"   (E_1 - E_T = {E[1]:+.6f})" if len(E) > 1 else ""))
        say(f"  F (VMC, tau=0)  = {F_m1:.6f}")
        say(f"  F (Ritz, final) = {F:.6f} +/- {F_err:.6f}")
        if abs(d['q2']) < 1e-12:
            say(f"  Q=0 charge check: F = {F:.6f}, must equal N_coord = {n_coord}   "
                f"(deviation {F - n_coord:+.2e}, "
                f"{abs(F - n_coord) / max(F_err, 1e-30):.2f} sigma)")

    if args.check_shift_invariance:
        alt_rel = args.extra_decay_rel * 3.0
        say(f"\n  --- shift invariance: extra_decay_rel {args.extra_decay_rel} -> {alt_rel} ---")
        C2b, C3b, info_b = build(d, args.hankel, alt_rel, None, args.n_block,
                                 args.n_fit_growth, verbose=False)
        me_b, spec_b = run_korr(C2b, C3b, cfg, verbose=False)
        Fb, Fb_err = ground_state(me_b)
        Eb, _ = shifted_energy(spec_b, info_b)
        say(f"  E_shift {info['E_shift']:.6g} -> {info_b['E_shift']:.6g}")
        say(f"  F  {F:.6f} -> {Fb:.6f}   (change {Fb - F:+.3e}, "
            f"{abs(Fb - F) / max(F_err, 1e-30):.3f} sigma of the stat error)")
        say(f"  E  {E[0]:+.6f} -> {Eb[0]:+.6f}   (change {Eb[0] - E[0]:+.3e})")
        say("  The shift is a gauge choice: it must cancel in P^L C3 P^R.")

    out = out_name or args.out_name or (fn[:-3] + args.out_suffix + '.csv')
    label = args.current if args.hankel == 'two_time' else args.current + '_legacy'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['current', 'F_m1', 'F_final', 'F_err'])
        w.writeheader()
        w.writerow({'current': label, 'F_m1': F_m1, 'F_final': F, 'F_err': F_err})
    if verbose:
        say(f"\n  wrote {out}")

    # The outer bootstrap draws, for a *correlated* fit across Q.  Every Q uses the same
    # Resampling plan (it is derived from the config seed and n_cfg, both Q-independent), so draw
    # b resamples the same walkers at every Q and the arrays are aligned sample-for-sample.  That
    # is what lets fit_ff.py --boot build the covariance across Q instead of assuming it diagonal
    # -- which matters here because every Q run shares the GFMC seed, so the points are strongly
    # correlated and a diagonal chi^2 is meaningless (it comes out ~0.02).
    boot = np.real(np.asarray(me.final_boot['me'])[:, 0, 0])
    np.save(out[:-4] + '_boot.npy', boot)
    if verbose:
        say(f"  wrote {out[:-4]}_boot.npy  ({boot.size} aligned outer draws, "
            f"std {boot.std():.6f})")
    return {'Q': np.sqrt(d['q2']), 'q2': d['q2'], 'F': F, 'F_err': F_err, 'F_m1': F_m1,
            'E': E[0], 'E_err': E_err[0], 'N_coord': n_coord, 'label': label,
            'm_max': m_max, 'stack': stack, 'stack_err': stack_err, 'all_m': list(me.all_m)}


# --------------------------------------------------------------------------------------------
# directory driver: the korr_dev -> korr swap for the whole pipeline
# --------------------------------------------------------------------------------------------
def currents_files(d):
    """The generator's three-point files, excluding the korrelator databases written beside them."""
    return sorted(f for f in glob.glob(os.path.join(d, 'currents_3pt_*.h5'))
                  if 'korr' not in os.path.basename(f))


def q_of(fn):
    m = re.search(r'Qmag([0-9.]+)\.h5$', os.path.basename(fn))
    return float(m.group(1)) if m else None


def calibrate_m_max(fn_q0, ladder, tol, drop_walkers=None):
    r"""Pick the largest Ritz iteration cap that still passes the exact :math:`Q=0` charge check.

    At :math:`Q = 0` the number operator is :math:`J_N = N_{\rm coord}\mathbb{1}`, so
    :math:`J^{(m)} = N_{\rm coord}` **exactly**, at every :math:`m`, for every coupling -- a known
    answer available for free in every run.  Any deviation is the analysis breaking down, not
    physics, which makes it the natural calibrator for ``m_max``.

    This matters because nothing else enforces the rule ``Lanczos_twoleg.py``'s own ``--max_tau_pts``
    help states -- *"keep 2*N_ritz_max << sqrt(N_samps)"*.  korr picks ``m_max`` from the number of
    time slices, and ``final_m``/``m_window`` then average the **last** few iterations, which can
    sit entirely inside the region where the bootstrap has already gone singular.  On the Nc=3
    baryon that returned F(0) = 2.93 where the data was exactly 3.00.
    """
    import tempfile
    say(f"\n{'=' * 78}\ncalibrating m_max on the exact Q=0 charge check "
        f"(tol {100 * tol:.2f}% of N_coord)\n{'=' * 78}")
    best, rows = None, []
    tmp = os.path.join(tempfile.mkdtemp(prefix='korr_mmax_'), 'probe_korr_ME.csv')
    for mm in ladder:
        try:
            r = analyse(fn_q0, m_max=mm, out_name=tmp, verbose=False,
                        drop_walkers=drop_walkers)
        except Exception as e:                      # a cap past the conditioning limit can throw
            say(f"  m_max={mm:<3d}  failed ({e!s:.60})")
            continue
        dev = abs(r['F'] - r['N_coord']) / r['N_coord']
        ok = np.isfinite(r['F_err']) and dev <= tol
        rows.append((mm, r['F'], r['F_err'], dev, ok))
        say(f"  m_max={mm:<3d}  F(0) = {r['F']:.6f} +/- {r['F_err']:.6f}   "
            f"deviation {100 * dev:+.3f}%   {'PASS' if ok else 'fail'}")

    passing = [r for r in rows if r[4]]
    if not passing:
        say("  WARNING: no m_max passed the veto; falling back to the smallest tried")
        return ladder[0]

    # The charge check is a VETO, not an objective to minimize.
    #
    # Structurally, J_N = N_coord * 1 at Q=0 has no off-diagonal elements, so J^(m) = N_coord
    # tests only that P^L C2 P^R = 1 -- the rotator normalization.  It is therefore *blind to
    # excited-state contamination*, which is what sets the error at Q != 0.  Minimizing it is
    # meaningless: two earlier rules here both failed badly for that reason, one parking m_max at
    # the tolerance edge and one (reacting to any rise, including ties differing in the 16th
    # decimal) collapsing to m_max=3 while discarding m_max=12 at 0.011%.
    #
    # What the check *does* detect is numerical breakdown of the pencil, which shows up in the
    # normalization.  Measured on the MRS meson, the deviation is 0.006-0.145% across the whole
    # region where <r^2> is stable (m_max 6-12, spanning 26.75-27.24 against the Numerov 26.89)
    # and steps up 3-4x to 0.46-0.55% exactly where <r^2> breaks away (+5.1% at 16, +8.2% at 20).
    # So: reject what it rejects, take the largest survivor, and quote the spread over the
    # surviving range as the rank systematic rather than pretending one m_max is the answer.
    best = passing[-1][0]
    rejected = [r[0] for r in rows if not r[4]]
    say(f"  veto rejects m_max in {rejected}" if rejected else "  veto rejects nothing")
    say(f"  -> using m_max = {best} (largest survivor; charge check "
        f"{100 * passing[-1][3]:+.3f}%)")
    say(f"  surviving range m_max {passing[0][0]}..{best}: scan it and quote the spread of the "
        f"final answer as the rank systematic")
    return best


def run_dir(d):
    """Analyse a whole output directory and write ``prod_ME.csv`` in the reference schema.

    A drop-in replacement for ``prod_extract_lanczos.py``, which drives ``Lanczos_twoleg.py`` and
    hence the vendored ``korr_dev``.  Differences that matter:

    * the two-time Hankel (``--hankel two_time``) instead of the ``C3_N[s+t]`` diagonal;
    * modern ``korr`` instead of ``korr_dev`` -- whose ZCW filter returns ``N_states = 0`` on the
      MRS correlator, and which takes hours per file at ``--max_tau_pts 40``;
    * ``m_max`` calibrated on the Q=0 charge check rather than left to the time extent;
    * aligned bootstrap draws saved per Q, so ``fit_ff.py`` can fit with the covariance.

    The output columns (``Q, channel, F, F_err``) are unchanged, so ``run/fit.py`` and the rest of
    the downstream chain work against it untouched.
    """
    files = currents_files(d)
    if not files:
        raise SystemExit(f"no currents_3pt_*.h5 in {d}")
    by_q = {q_of(f): f for f in files if q_of(f) is not None}
    say(f"{len(by_q)} Q values in {d}: " + "  ".join(f"{q:g}" for q in sorted(by_q)))

    # One drop set for every Q.  It must be identical across Q or the aligned bootstrap replicas
    # that fit_ff.py relies on no longer describe the same ensemble.
    drop = None
    if args.spike_leverage > 0:
        if 0.0 not in by_q:
            say("  no Q=0 file; cannot fix a spike drop set, leaving all walkers in")
        else:
            drop, lev = spike_walkers(by_q[0.0], args.spike_leverage)
            if lev is None:
                say("  no Ws2_offdiag; spike screening skipped")
                drop = None
            else:
                srt = np.sort(lev)[::-1]
                say(f"  walker leverage: worst {srt[0]:.0f}x, next {srt[1]:.0f}x, "
                    f"{srt[2]:.0f}x   threshold {args.spike_leverage:g}x")
                say(f"  dropping {len(drop)} of {len(lev)} walkers: {list(map(int, drop))}")

    m_max = args.m_max
    if m_max is None and args.calibrate_m_max:
        if 0.0 not in by_q:
            say("  no Q=0 file; cannot calibrate m_max, leaving it automatic")
        else:
            ladder = args.m_max_ladder or [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
            m_max = calibrate_m_max(by_q[0.0], ladder, args.charge_tol, drop)

    rows, failed = [], []
    for q in sorted(by_q):
        try:
            r = analyse(by_q[q], m_max=m_max, verbose=False, drop_walkers=drop)
        except Exception as e:
            say(f"  FAILED Q={q}: {e!s:.100}")
            failed.append(q)
            continue
        rows.append({'Q': q, 'channel': r['label'], 'F': r['F'], 'F_err': r['F_err']})
        say(f"  Q={q:<6g} F = {r['F']:.6f} +/- {r['F_err']:.6f}   "
            f"E_0-E_T = {r['E']:+.6f} +/- {r['E_err']:.6f}")

    out = os.path.join(d, args.prod_name)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['Q', 'channel', 'F', 'F_err'])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['Q'], r['channel'])):
            w.writerow(r)
    say(f"\nwrote {out} ({len(rows)} rows, {len(failed)} failed)"
        + (f"  FAILED: {failed}" if failed else ""))
    return rows


# --------------------------------------------------------------------------------------------
# synthetic self-test: the whole chain on data whose answer is known in closed form
# --------------------------------------------------------------------------------------------
def self_test():
    r"""Three checks on synthetic two-leg data, no GFMC needed.

    1. On a noiseless two-state system korr returns the input :math:`J_{fi}` once the Krylov space
       is complete -- the machinery and the index conventions are right.
    2. The same data fed through the ``legacy`` Hankel construction does *not*, and the error is
       controlled by the off-diagonal :math:`J_{01}`: setting :math:`J_{01} = 0` (a current that
       commutes with :math:`H`, the :math:`Q=0` case) makes the legacy construction exact again.
       This is the bug, isolated, with no statistics involved.
    3. The uniform shift cancels in the sandwich.
    """
    from korr.synth import make_2pt, make_3pt
    say("=" * 78)
    say("SELF TEST: synthetic two-leg data, closed-form answer")
    say("=" * 78)
    n_tau, Z, E = 12, np.array([1.3, 0.8]), np.array([0.25, 0.9])
    ok = True

    for J01 in (0.4, 0.0):
        J = np.array([[2.0, J01], [J01, 0.5]])
        tag = (f"J_01 = {J01}  "
               + ("(a current that does NOT commute with H: the Q != 0 case)" if J01
                  else "(J diagonal <-> [J,H] = 0: the Q = 0 case)"))
        say(f"\n--- {tag}")

        # the true two-time object on a grid of step 1, and its 2pt partner
        C3_true = make_3pt(Z, Z, E, E, J, n_tau)                 # (sigma, tau)
        C2_true = make_2pt(Z, E, 2 * (n_tau - 1) + 1)            # total-time grid

        # m_max = 2 = the number of states: a noiseless two-state Hankel has rank 2 exactly, so
        # m > 2 is genuinely singular rather than ill-conditioned.
        cfg = Config(n_outer=1, n_inner=1, seed=args.seed, m_window=1, final_m=2, m_max=2,
                     ritz_coeffs=True, progress=False, summarize=_SUMMARIZE)
        me, _ = run_korr(C2_true[None], C3_true[None], cfg, verbose=False)
        got = float(me.matrix[0, 0])
        err = abs(got - J[0, 0])
        say(f"  two_time Hankel : J_00 = {got:+.9f}   target {J[0, 0]:+.9f}   |err| = {err:.2e}")
        ok &= err < 1e-6

        # the legacy construction: C3[s,t] <- C_J((s+t)dt,(s+t)dt) with the symmetric-point 2pt
        m_cap = (n_tau + 1) // 2
        diag = np.array([C3_true[k, k] for k in range(n_tau)])
        C3_leg = np.array([[diag[s + t] for t in range(m_cap)] for s in range(m_cap)])
        C2_leg = np.array([C2_true[2 * k] for k in range(m_cap + 1)])
        me_l, _ = run_korr(C2_leg[None], C3_leg[None], cfg, verbose=False)
        got_l = float(me_l.matrix[0, 0])
        err_l = abs(got_l - J[0, 0])
        say(f"  legacy   Hankel : J_00 = {got_l:+.9f}   target {J[0, 0]:+.9f}   |err| = {err_l:.2e}")
        if J01:
            say(f"    -> off by {100 * err_l / J[0, 0]:.2f}% with NO noise: the f != i terms do "
                f"not cancel.")
            ok &= err_l > 1e-3
        else:
            say("    -> exact, because a diagonal J has no f != i terms to get wrong.")
            ok &= err_l < 1e-6

        if J01:   # check 3, on the correct construction
            S = 0.37
            k = np.arange(len(C2_true))
            sig = np.arange(n_tau)
            C2_s = C2_true * np.exp(-S * k)
            C3_s = C3_true * np.exp(-S * (sig[:, None] + sig[None, :]))
            me_s, _ = run_korr(C2_s[None], C3_s[None], cfg, verbose=False)
            d = abs(float(me_s.matrix[0, 0]) - got)
            say(f"  shift invariance: E_shift 0 -> {S}  changes J_00 by {d:.2e}")
            ok &= d < 1e-6

    say("\n" + "=" * 78)
    say("SELF TEST " + ("PASSED" if ok else "FAILED"))
    say("=" * 78)
    return 0 if ok else 1


if __name__ == '__main__':
    if args.self_test:
        sys.exit(self_test())
    if args.dir:
        run_dir(args.dir)
    elif args.currents_file:
        analyse(args.currents_file)
    else:
        p.error('one of --currents_file, --dir or --self_test is required')
