"""Audit of the two-leg GFMC 3pt correlator: is C_J(sigma,tau) what the extraction thinks?

The claim under test concerns Lanczos_twoleg.py:213,

    C3pt[:, si, ti] = C3_binned[:, si + ti]

where C3_N[:, k] is the *symmetric-point* two-leg correlator C_J(k D, k D) (generator line 906:
leg-1 time and leg-2 time are both `t`), not a one-leg correlator.  The 2pt fed alongside it,
Ct[k] = Ws2_all[k] = C2(2 k D), is a valid Hankel in step 2D -- which the extraction itself
acknowledges by dividing the Ritz rate by 2.  On that grid the (si,ti) entry of the 3pt Hankel
must be C_J(2 si D, 2 ti D), but the line above supplies C_J((si+ti) D, (si+ti) D).  Spectrally,

    needed    sum_fi Z'_f Z_i J_fi exp(-2 E_f si D - 2 E_i ti D)
    supplied  sum_fi Z'_f Z_i J_fi exp(-(E_f + E_i)(si+ti) D)

which agree only for f = i.  The diagonal spectral terms are right; the f != i interference
terms -- exactly what the Ritz projector exists to cancel -- are not.

At Q = 0 the number operator J_N = sum_k exp(i q.x_k) = N_coord * 1 commutes with H, so
J_fi ~ delta_fi, every off-diagonal term vanishes and the defect is *exactly inert*.  That is why
the F(0) = N_coord check passes.  It switches on at Q != 0, which is where the form-factor shape
and hence <r^2> live.

The generator already stores the correct object -- C3_N_offdiag[i,j] = C_J(t_i D, t_j D) and
Ws2_offdiag[i,j] = C2((t_i + t_j) D), written at lines 848/908/1148-1149 and read by nothing in
the bundle.  This script validates that data and then measures the defect.

Usage:
    python audit_c3_two_time.py <currents_3pt_*.h5> [...] [--plot_dir audit_plots]
"""
import argparse
import os

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('files', nargs='+')
p.add_argument('--n_boot', type=int, default=400)
p.add_argument('--plot_dir', default='audit_plots')
p.add_argument('--seed', type=int, default=17)
args = p.parse_args()


def boot_mean(x, n_boot, rng):
    """Bootstrap over the walker axis (last).  Returns (mean, err)."""
    n = x.shape[-1]
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.array([x[..., i].mean(axis=-1) for i in idx])
    return x.mean(axis=-1), draws.std(axis=0)


def ratio_boot(num, den, n_boot, rng):
    """Bootstrap of <num>/<den>, correlated (same walker resample for both)."""
    n = num.shape[-1]
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = np.array([num[..., i].mean(axis=-1) / den[..., i].mean(axis=-1) for i in idx])
    return num.mean(axis=-1) / den.mean(axis=-1), draws.std(axis=0)


def report(fn, rng):
    with h5py.File(fn, 'r') as f:
        missing = [k for k in ('C3_N_offdiag', 'Ws2_offdiag') if k not in f]
        if missing:
            print("  SKIP: no two-time data in this file (%s). Only the Coulomb generator "
                  "stores it." % ', '.join(missing))
            return None
        C3_od = np.real(f['C3_N_offdiag'][:])     # (N_tau, N_tau, N_w)  [leg1, leg2, walker]
        Ws2_od = np.real(f['Ws2_offdiag'][:])     # (N_tau, N_tau, N_w)
        C3_diag = np.real(f['C3_N'][:])           # (N_w, N_tau)
        Ws2_all = np.real(f['Ws2_all'][:])        # (N_tau, N_w)
        tau = f['tau_values'][:]
        q2 = float(f['q_squared'][()])
        masses = f['masses'][:]
    N_tau, _, N_w = C3_od.shape
    D = float(tau[1] - tau[0])
    Q = np.sqrt(q2)
    N_coord = len(masses)
    print("  N_tau=%d  N_walkers=%d  D=%.4f  Q=%.4f  Q^2=%.5f  N_coord=%d"
          % (N_tau, N_w, D, Q, q2, N_coord))

    out = {'Q': Q, 'q2': q2, 'N_coord': N_coord, 'N_tau': N_tau, 'D': D}

    # -- 1. the off-diagonal data reduces to the diagonal data -------------------------------
    # Ws2_all[k] must be Ws2_offdiag[k,k] and C3_N[:,k] must be C3_N_offdiag[k,k,:], walker by
    # walker: the diagonal arrays are the symmetric point of the two-time ones.
    d_w = np.abs(Ws2_od[np.arange(N_tau), np.arange(N_tau)] - Ws2_all).max()
    d_c = np.abs(C3_od[np.arange(N_tau), np.arange(N_tau)] - C3_diag.T).max()
    s_w = np.abs(Ws2_all).max()
    print("  [1] diagonal consistency (walker-by-walker, exact up to storage):")
    print("      max|Ws2_offdiag[k,k] - Ws2_all[k]|  = %.3e   (scale %.3e)" % (d_w, s_w))
    print("      max|C3_N_offdiag[k,k] - C3_N[:,k]|  = %.3e" % d_c)
    out['diag_ok'] = bool(d_w <= 1e-10 * max(s_w, 1.0) and d_c <= 1e-10 * max(s_w, 1.0))

    # -- 2. Ws2_offdiag is a genuine 2pt function: depends on i+j only -----------------------
    # C2((t_i + t_j) D) cannot know how the total time was split.  Any dependence on i-j at
    # fixed i+j is pure GFMC noise, so this bounds the noise floor for check 4.
    print("  [2] Ws2_offdiag[i,j] depends only on i+j (it is C2 of the total time):")
    worst = 0.0
    for k in range(1, min(2 * N_tau - 1, 2 * (N_tau - 1) + 1)):
        cells = [(i, k - i) for i in range(N_tau) if 0 <= k - i < N_tau]
        if len(cells) < 2:
            continue
        vals = np.array([Ws2_od[i, j].mean() for i, j in cells])
        errs = np.array([boot_mean(Ws2_od[i, j], args.n_boot, rng)[1] for i, j in cells])
        spread = (vals.max() - vals.min()) / abs(vals.mean())
        pull = (vals.max() - vals.min()) / max(np.sqrt((errs ** 2).sum()), 1e-300)
        worst = max(worst, pull)
        if k <= 6 or k % 4 == 0:
            print("      i+j=%2d (%d cells): rel spread %7.2e   pull %5.2f sigma"
                  % (k, len(cells), spread, pull))
    print("      worst antidiagonal pull: %.2f sigma" % worst)
    out['ws2_antidiag_pull'] = float(worst)

    # -- 3. sigma/tau orientation --------------------------------------------------------------
    # korr's convention is sigma = current-to-sink, tau = source-to-current.  Here index 0 is the
    # leg-1 (source-to-current) time and index 1 the leg-2 (current-to-sink) time, so
    # C3pt[sigma,tau] = C3_N_offdiag[tau,sigma].  For a Hermitian J with the same state either
    # side the matrix is physically symmetric, so the asymmetry measured here is the statistical
    # cost of getting the transpose wrong -- not a bias.
    sym = C3_od + np.swapaxes(C3_od, 0, 1)
    asym = C3_od - np.swapaxes(C3_od, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_asym = np.abs(asym.mean(axis=-1)) / np.abs(sym.mean(axis=-1))
    iu = np.triu_indices(N_tau, 1)
    finite = np.isfinite(rel_asym[iu])
    print("  [3] transpose asymmetry |C3[i,j]-C3[j,i]| / |C3[i,j]+C3[j,i]|:")
    print("      median %.3e   max %.3e   (physically symmetric; this is noise)"
          % (np.median(rel_asym[iu][finite]), np.nanmax(rel_asym[iu][finite])))
    out['asym_median'] = float(np.median(rel_asym[iu][finite]))

    # -- 4. THE DEFECT -----------------------------------------------------------------------
    # Build the effective form factor two ways on the step-2D grid the rotators live on:
    #   A (as extracted)  <C3_N[s+t]>      / <Ws2_all[s+t]>        = C_J((s+t)D,(s+t)D) / C2(2(s+t)D)
    #   B (correct)       <C3_od[2s,2t]>   / <Ws2_od[2s,2t]>       = C_J(2sD,2tD)       / C2(2(s+t)D)
    # Both carry the same total time, so the same C2 normalizes both and the comparison is clean.
    # They coincide identically at s=t; the off-diagonal difference is the bug.
    m_max = (N_tau - 1) // 2 + 1
    A = np.zeros((m_max, m_max))
    A_e = np.zeros((m_max, m_max))
    B = np.zeros((m_max, m_max))
    B_e = np.zeros((m_max, m_max))
    for s in range(m_max):
        for t in range(m_max):
            A[s, t], A_e[s, t] = ratio_boot(C3_diag[:, s + t], Ws2_all[s + t], args.n_boot, rng)
            B[s, t], B_e[s, t] = ratio_boot(C3_od[2 * s, 2 * t], Ws2_od[2 * s, 2 * t],
                                            args.n_boot, rng)
    diff = A - B
    pull = diff / np.sqrt(A_e ** 2 + B_e ** 2 + 1e-300)
    print("  [4] effective form factor, as-extracted (A) vs two-time (B), %dx%d block:"
          % (m_max, m_max))
    print("      s  t    A (used)        B (correct)      A-B           pull")
    for s in range(m_max):
        for t in range(m_max):
            if s == t or abs(s - t) == m_max - 1 or (s == 0 and t <= 3):
                tag = '  <- diagonal, must agree exactly' if s == t else ''
                print("      %d  %d  %+.6f(%.6f) %+.6f(%.6f) %+.3e  %6.2f%s"
                      % (s, t, A[s, t], A_e[s, t], B[s, t], B_e[s, t], diff[s, t],
                         pull[s, t], tag))
    offd = ~np.eye(m_max, dtype=bool)
    print("      diagonal   : max|A-B| = %.3e  (identical by construction)"
          % np.abs(diff[np.eye(m_max, dtype=bool)]).max())
    if offd.any():
        print("      off-diagonal: max|A-B| = %.3e   max pull = %.2f sigma   median pull = %.2f"
              % (np.abs(diff[offd]).max(), np.abs(pull[offd]).max(),
                 np.median(np.abs(pull[offd]))))
    out['A'], out['B'], out['A_e'], out['B_e'] = A, B, A_e, B_e
    out['offdiag_max_absdiff'] = float(np.abs(diff[offd]).max()) if offd.any() else 0.0
    out['offdiag_max_pull'] = float(np.abs(pull[offd]).max()) if offd.any() else 0.0

    # growth of the defect with |s-t|: the signature of an E_f != E_i interference term
    print("      |s-t| dependence of the defect (the f!=i interference signature):")
    for d in range(m_max):
        cells = [(s, t) for s in range(m_max) for t in range(m_max) if abs(s - t) == d]
        if not cells:
            continue
        v = np.array([abs(diff[s, t]) for s, t in cells])
        pl = np.array([abs(pull[s, t]) for s, t in cells])
        print("        |s-t|=%d : mean|A-B| = %.3e   mean pull = %5.2f" % (d, v.mean(), pl.mean()))
    out['defect_by_lag'] = [float(np.mean([abs(diff[s, t]) for s in range(m_max)
                                           for t in range(m_max) if abs(s - t) == d]))
                            for d in range(m_max)]

    # -- 5. uncancelled CM recoil ------------------------------------------------------------
    # The junction weight carries r01 = conj(f_R_q)/f_R_0, so the sink trial state is psi_q while
    # the source is psi_0, and only one 2pt function is stored ("norm is the same for zero and
    # finite momentum", generator line 839).  FRR divides out no Z, so a real Z' != Z biases F(Q).
    # If the conjugation does *not* fully cancel the CM phase, C3 decays faster than C2 along the
    # leg-2 direction by the recoil rate Q^2/(2M).
    M = float(np.abs(masses).sum())
    recoil = q2 / (2 * M)
    i1 = 0
    g2, g3 = [], []
    for i2 in range(1, N_tau):
        r2 = Ws2_od[i1, i2].mean() / Ws2_od[i1, i2 - 1].mean()
        r3 = C3_od[i1, i2].mean() / C3_od[i1, i2 - 1].mean()
        if r2 > 0 and r3 > 0:
            g2.append(-np.log(r2) / D)
            g3.append(-np.log(r3) / D)
    n_use = max(1, min(5, len(g2)))
    rate2, rate3 = float(np.mean(g2[:n_use])), float(np.mean(g3[:n_use]))
    print("  [5] decay rate along the leg-2 (sink) direction at leg-1 index 0, first %d steps:"
          % n_use)
    print("      2pt  d(-log C2)/dtau  = %+.6f" % rate2)
    print("      3pt  d(-log C3)/dtau  = %+.6f" % rate3)
    print("      difference            = %+.6f    vs Q^2/(2M) = %.6f  (M=%.1f)"
          % (rate3 - rate2, recoil, M))
    if recoil > 0:
        print("      => %.1f%% of the recoil rate is left uncancelled"
              % (100 * (rate3 - rate2) / recoil))
    out['rate2'], out['rate3'], out['recoil'] = rate2, rate3, recoil
    return out


def main():
    rng = np.random.default_rng(args.seed)
    results = []
    for fn in args.files:
        print("\n" + "=" * 78)
        print(os.path.basename(fn))
        print("=" * 78)
        r = report(fn, rng)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        results.sort(key=lambda r: r['Q'])
        print("\n" + "=" * 78)
        print("SUMMARY: the defect vs Q  (it must vanish at Q=0, where J_N commutes with H)")
        print("=" * 78)
        print("    Q      Q^2       max|A-B| off-diag   max pull   recoil left uncancelled")
        for r in results:
            rec = ('%+.6f (%.0f%% of %.6f)'
                   % (r['rate3'] - r['rate2'],
                      100 * (r['rate3'] - r['rate2']) / r['recoil'], r['recoil'])
                   if r['recoil'] > 0 else 'n/a (Q=0)')
            print("  %6.3f  %8.5f   %.6e        %6.2f     %s"
                  % (r['Q'], r['q2'], r['offdiag_max_absdiff'], r['offdiag_max_pull'], rec))
        print("\n  Interpretation: a defect that is zero at Q=0 and grows with Q^2 and with "
              "|s-t|\n  is the f!=i interference term, i.e. the Hankel-index bug -- not noise.")

        try:
            plot(results)
        except Exception as e:                                    # plotting is not the deliverable
            print("\n  (plot skipped: %s)" % e)


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(args.plot_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    q2 = [r['q2'] for r in results]
    ax[0].plot(q2, [r['offdiag_max_absdiff'] for r in results], 'o-')
    ax[0].set_xlabel(r'$Q^2$')
    ax[0].set_ylabel(r'max$|A-B|$ off-diagonal')
    ax[0].set_title('Hankel-index defect vs $Q^2$\n(zero at $Q=0$: $J_N$ commutes with $H$)')
    ax[0].grid(alpha=.3)
    for r in results:
        lag = r['defect_by_lag']
        ax[1].plot(range(len(lag)), lag, 'o-', label='Q=%.3f' % r['Q'])
    ax[1].set_xlabel(r'$|s-t|$')
    ax[1].set_ylabel(r'mean$|A-B|$')
    ax[1].set_title('Defect grows with the time-index lag')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    path = os.path.join(args.plot_dir, 'c3_two_time_defect.png')
    fig.savefig(path, dpi=140)
    print("\n  wrote %s" % path)


main()
