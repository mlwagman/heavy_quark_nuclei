r"""How should the extraction cutoffs scale with dtau?

The Rayleigh-Ritz pencil at iteration :math:`m` uses :math:`C(s+t)` and :math:`C(s+t+1)` for
:math:`s,t<m`, so it reaches two-point index :math:`2m-1` -- i.e. **physical** time

.. math::  t_{\rm reach}(m) \;=\; (2m-1)\,\Delta, \qquad \Delta = \texttt{count\_skip}\times d\tau .

Everything that limits :math:`m` from above is a function of :math:`t_{\rm reach}`, not of the
number of grid points:

* statistics -- :math:`N_{\rm eff}` degrades as :math:`e^{-\sigma^2}` with
  :math:`\sigma^2 \simeq 2\lambda_\tau t` and :math:`\lambda_\tau \approx 0.05|E_0|`
  (``weight_tails.py``), a continuum property measured to be :math:`d\tau`-independent;
* spectral information -- set by the window length against the gap.

So refining :math:`d\tau` at fixed ``count_skip`` shrinks :math:`\Delta`, adds grid points that are
increasingly *linearly dependent*, and moves the usable :math:`m_{\max}` for purely numerical
reasons.  A :math:`d\tau\to0` extrapolation done that way is partly measuring its own extraction.

**Hypothesis under test:** hold :math:`\Delta` fixed (``count_skip`` :math:`\propto 1/d\tau`) and the
extraction is invariant -- same :math:`N_\tau`, same :math:`t_{\rm reach}(m)`, same
:math:`N_{\rm eff}(k)`, same plateau in :math:`m` -- leaving only the Trotter error to extrapolate.
Configs A/B/C hold :math:`\Delta = 2.5`; config D is the counter-example at fixed ``count_skip``.

Predictions, if the hypothesis holds:

====================  =====================================  ============================
quantity              fixed :math:`\Delta` (A, B, C)          fixed ``count_skip`` (D)
====================  =====================================  ============================
:math:`N_\tau`        identical                               doubles
:math:`N_{\rm eff}(k)` identical as a function of k           identical vs *physical time*
plateau in m          identical                               extends to ~2x larger m
:math:`t_{\rm reach}` identical                               halves at fixed m
====================  =====================================  ============================

The last row is the operational point: at fixed ``count_skip``, the *same* :math:`m_{\max}` probes
only half as much physical time, so a cutoff quoted as an :math:`m` is not transferable between
:math:`d\tau` values -- it has to be quoted as a :math:`t_{\rm reach}`, or :math:`\Delta` has to be
held fixed.

Usage:  python3.12 dtau_cutoff_scaling.py
"""

import glob
import os
import re
import subprocess

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = '/usr/local/bin/python3.12'

CFGS = [('A', 0.5, 5), ('B', 0.25, 10), ('C', 0.125, 20), ('D', 0.25, 5)]
E0_EXACT = -0.04
R2_EXACT = 18.75


def cfg_dir(tag, dtau, skip):
    return os.path.join(HERE, 'dtau_scan', f'cfg{tag}_dtau{dtau}_skip{skip}')


def q_files(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, 'currents_3pt_*.h5'))):
        if 'korr' in os.path.basename(f):
            continue
        m = re.search(r'Qmag([0-9.]+)\.h5$', os.path.basename(f))
        if m:
            out[float(m.group(1))] = f
    return out


def n_eff(w):
    s1, s2 = np.abs(w).sum(), (w ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def antidiag_c2(od):
    n_tau, _, n_w = od.shape
    n_k = 2 * (n_tau - 1) + 1
    c2 = np.zeros((n_k, n_w))
    for k in range(n_k):
        cells = [(i, k - i) for i in range(n_tau) if 0 <= k - i < n_tau]
        c2[k] = np.mean([od[i, j] for i, j in cells], axis=0)
    return c2


def jm_table(fn, m_max, n_block=1, spike=50.0):
    """``{m: (J, err)}`` from the production extraction at Q=0."""
    cmd = [PY, os.path.join(HERE, 'gfmc_me_korr.py'), '--quiet', '--currents_file', fn,
           '--n_block', str(n_block), '--m_max', str(m_max),
           '--spike_leverage', str(spike),
           '--out_name', '/tmp/dtau_probe_korr_ME.csv']
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    out, e0 = {}, None
    for line in r.stdout.splitlines():
        mm = re.match(r'\s*m=\s*(\d+)\s+([-+0-9.eE]+)\s+\+/-\s+(\S+)', line)
        if mm:
            try:
                out[int(mm.group(1))] = (float(mm.group(2)), float(mm.group(3)))
            except ValueError:
                out[int(mm.group(1))] = (float(mm.group(2)), np.nan)
        if 'E_0 - E_T' in line:
            e0 = line.split('=')[1].split('+/-')[0].strip()
    return out, e0


def main():
    print("=" * 104)
    print("PART 1 -- grid geometry: what each config actually probes")
    print("=" * 104)
    print(f"  {'cfg':>4} {'dtau':>7} {'skip':>5} {'Delta':>7} {'N_tau':>6} {'tau_max':>8} "
          f"{'dtau*|E0|':>10} {'t_reach(m=6)':>13} {'t_reach(m=12)':>14}")
    live = []
    for tag, dtau, skip in CFGS:
        d = cfg_dir(tag, dtau, skip)
        qs = q_files(d)
        if not qs:
            print(f"  {tag:>4}   (not generated yet)")
            continue
        with h5py.File(qs[sorted(qs)[0]], 'r') as f:
            tau = f['tau_values'][:]
        D = float(tau[1] - tau[0])
        print(f"  {tag:>4} {dtau:>7} {skip:>5} {D:>7.3f} {len(tau):>6} {tau[-1]:>8.2f} "
              f"{dtau * abs(E0_EXACT):>10.4f} {11 * D:>13.2f} {23 * D:>14.2f}")
        live.append((tag, dtau, skip, D, qs, tau))

    if not live:
        print("\nnothing generated yet; rerun once run_dtau_scan.sh has produced output")
        return

    print("\n" + "=" * 104)
    print("PART 2 -- N_eff: identical vs k (grid index), or identical vs physical time?")
    print("=" * 104)
    print("  If the limit is physical, N_eff should collapse onto one curve when plotted against")
    print("  t = k*Delta, NOT against k.  That is the test of which variable the cutoff lives in.\n")
    for tag, dtau, skip, D, qs, tau in live:
        if 0.0 not in qs:
            continue
        with h5py.File(qs[0.0], 'r') as f:
            if 'Ws2_offdiag' not in f:
                continue
            od = np.real(f['Ws2_offdiag'][:])
        c2 = antidiag_c2(od)
        n_w = od.shape[2]
        print(f"--- cfg {tag} (dtau={dtau}, Delta={D:.3f})")
        print(f"    {'k':>4} {'t=k*Delta':>11} {'N_eff':>8} {'N_eff/N':>9}")
        for k in range(1, c2.shape[0], max(1, c2.shape[0] // 6)):
            ne = n_eff(c2[k])
            print(f"    {k:>4} {k * D:>11.2f} {ne:>8.0f} {100 * ne / n_w:>8.1f}%")

    print("\n" + "=" * 104)
    print("PART 3 -- where does J^(m) stop being flat, in m and in t_reach?")
    print("=" * 104)
    print(f"  Q=0, so J^(m) = N_coord = 2 exactly at every m. Deviation is pure extraction error.")
    print("  The question is whether the breakdown sits at a common m or a common t_reach.\n")
    for tag, dtau, skip, D, qs, tau in live:
        if 0.0 not in qs:
            continue
        m_cap = (2 * (len(tau) - 1) + 1) // 2
        jm, e0 = jm_table(qs[0.0], m_cap)
        if not jm:
            print(f"--- cfg {tag}: no J^(m) harvested")
            continue
        print(f"--- cfg {tag} (dtau={dtau}, Delta={D:.3f})   E_0 = {e0}  (exact {E0_EXACT})")
        print(f"    {'m':>3} {'t_reach':>9} {'J^(m)':>11} {'err':>10} {'|dev|':>9} {'pull':>7}")
        for m in sorted(jm):
            v, e = jm[m]
            dev = abs(v - 2.0) / 2.0
            pull = abs(v - 2.0) / e if (np.isfinite(e) and e > 0) else np.inf
            ps = f"{pull:>7.2f}" if np.isfinite(pull) else f"{'inf':>7}"
            bad = '  <-- broken' if (not np.isfinite(e)) or pull > 3 else ''
            print(f"    {m:>3} {(2 * m - 1) * D:>9.2f} {v:>11.6f} {e:>10.6f} {dev:>9.5f} {ps}{bad}")

    print("\n" + "=" * 104)
    print("CONCLUSION to read off")
    print("=" * 104)
    print("  * If A/B/C agree row-for-row in Parts 2 and 3 while D differs at the same m but")
    print("    agrees with them at the same t_reach, then the cutoff is physical: hold")
    print("    Delta = count_skip*dtau fixed in a dtau -> 0 scan, and the extraction is invariant.")
    print("  * If instead A/B/C drift, the cutoff has its own dtau dependence and has to be")
    print("    recalibrated per dtau -- which would need folding into the extrapolation's")
    print("    systematic rather than ignored.")


main()
