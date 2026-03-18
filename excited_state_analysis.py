"""
Two-state fit and summation method for extracting form factors
from GFMC 3-point function data with excited state contamination.

Usage:
    python excited_state_analysis.py <currents_3pt_h5_file>
"""
import sys
import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def two_state_model(tau, F0, A, dE):
    """R(tau) = F0 + A * exp(-dE * tau)"""
    return F0 + A * np.exp(-dE * tau)

def two_state_fit(tau, F, F_err, label="", tau_min=0):
    """Fit F(tau) = F0 + A*exp(-dE*tau) to extract ground state F0."""
    mask = tau >= tau_min
    t, y, yerr = tau[mask], F[mask], F_err[mask]

    # Initial guesses from data
    F0_guess = y[-1]
    A_guess = y[0] - y[-1]
    dE_guess = 0.02

    try:
        popt, pcov = curve_fit(
            two_state_model, t, y, p0=[F0_guess, A_guess, dE_guess],
            sigma=yerr, absolute_sigma=True,
            bounds=([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 1.0]),
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        F0, A, dE = popt
        F0_err, A_err, dE_err = perr

        # chi2
        resid = (y - two_state_model(t, *popt)) / yerr
        chi2 = np.sum(resid**2)
        ndof = len(t) - 3
        chi2_dof = chi2 / ndof if ndof > 0 else np.inf

        print(f"  {label}:")
        print(f"    F0 = {F0:.6f} +/- {F0_err:.6f}")
        print(f"    A  = {A:.6f} +/- {A_err:.6f}")
        print(f"    dE = {dE:.6f} +/- {dE_err:.6f}")
        print(f"    chi2/dof = {chi2:.1f}/{ndof} = {chi2_dof:.2f}")
        return popt, perr, chi2_dof
    except Exception as e:
        print(f"  {label}: fit failed — {e}")
        return None, None, None

def summation_method(tau, F, F_err, label=""):
    """
    Summation method: S(tau) = sum_{t=0}^{tau} R(t) * dtau
    At large tau: S(tau) = const + F0 * tau
    Slope gives F0 with excited states suppressed by extra exp(-dE*tau).
    """
    dtau = tau[1] - tau[0]

    # Cumulative sum (trapezoidal)
    S = np.cumsum(F) * dtau
    # Error propagation for cumulative sum
    S_err = np.sqrt(np.cumsum(F_err**2)) * dtau

    # Fit linear model S = c + F0 * tau at large tau
    # Use upper half of data where excited states are most suppressed
    tau_half = tau[len(tau)//2:]
    S_half = S[len(tau)//2:]
    S_err_half = S_err[len(tau)//2:]

    # Weighted linear fit
    def linear(t, c, F0):
        return c + F0 * t

    try:
        popt, pcov = curve_fit(linear, tau_half, S_half,
                               sigma=S_err_half, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        c, F0 = popt
        c_err, F0_err = perr

        resid = (S_half - linear(tau_half, *popt)) / S_err_half
        chi2 = np.sum(resid**2)
        ndof = len(tau_half) - 2
        chi2_dof = chi2 / ndof if ndof > 0 else np.inf

        print(f"  {label}:")
        print(f"    F0 (slope) = {F0:.6f} +/- {F0_err:.6f}")
        print(f"    const      = {c:.6f} +/- {c_err:.6f}")
        print(f"    chi2/dof   = {chi2:.1f}/{ndof} = {chi2_dof:.2f}")
        return S, S_err, popt, perr, chi2_dof
    except Exception as e:
        print(f"  {label}: fit failed — {e}")
        return S, S_err, None, None, None


def main(h5_file):
    f = h5py.File(h5_file, 'r')
    tau = np.array(f['tau_values'])
    N_coord = len(f['masses'])
    q2 = float(np.array(f['q_squared']))

    channels = {}
    for ch in ['N', 'S', 'V0']:
        key = f'F_{ch}'
        channels[ch] = {
            'F': np.array(f[key]),
            'F_err': np.array(f[f'{key}_err']),
        }
    f.close()

    dtau = tau[1] - tau[0]
    print(f"File: {h5_file}")
    print(f"N_coord={N_coord}, q^2={q2:.6f}, dtau={dtau}, tau_max={tau[-1]}")
    print(f"N_tau = {len(tau)}")
    print()

    # ==================== TWO-STATE FITS ====================
    print("=" * 60)
    print("TWO-STATE FIT: R(tau) = F0 + A*exp(-dE*tau)")
    print("=" * 60)

    two_state_results = {}
    for ch in ['N', 'S', 'V0']:
        popt, perr, chi2 = two_state_fit(
            tau, channels[ch]['F'], channels[ch]['F_err'],
            label=f"F_{ch}", tau_min=0
        )
        two_state_results[ch] = (popt, perr, chi2)
    print()

    # ==================== SUMMATION METHOD ====================
    print("=" * 60)
    print("SUMMATION METHOD: S(tau) = const + F0*tau  (slope = F0)")
    print(f"Linear fit region: tau in [{tau[len(tau)//2]:.0f}, {tau[-1]:.0f}]")
    print("=" * 60)

    summ_results = {}
    for ch in ['N', 'S', 'V0']:
        S, S_err, popt, perr, chi2 = summation_method(
            tau, channels[ch]['F'], channels[ch]['F_err'],
            label=f"F_{ch}"
        )
        summ_results[ch] = (S, S_err, popt, perr, chi2)
    print()

    # ==================== COMPARISON ====================
    print("=" * 60)
    print(f"COMPARISON (N_coord={N_coord}, expect F/N_coord -> 1 at q->0)")
    print("=" * 60)
    print(f"{'Channel':<8} {'Plateau':<20} {'Two-state':<20} {'Summation':<20}")
    print("-" * 68)
    for ch in ['N', 'S', 'V0']:
        # Plateau: last 5 points average
        F_last = channels[ch]['F'][-5:]
        F_err_last = channels[ch]['F_err'][-5:]
        w = 1.0 / F_err_last**2
        F_plat = np.sum(w * F_last) / np.sum(w)
        F_plat_err = 1.0 / np.sqrt(np.sum(w))

        ts = two_state_results[ch]
        if ts[0] is not None:
            ts_str = f"{ts[0][0]/N_coord:.6f}({ts[1][0]/N_coord*1e6:.0f})"
        else:
            ts_str = "FAILED"

        sm = summ_results[ch]
        if sm[2] is not None:
            sm_str = f"{sm[2][1]/N_coord:.6f}({sm[3][1]/N_coord*1e6:.0f})"
        else:
            sm_str = "FAILED"

        print(f"F_{ch}/N  {F_plat/N_coord:.6f}({F_plat_err/N_coord*1e6:.0f})   "
              f"{ts_str:<20} {sm_str:<20}")

    # ==================== PLOTS ====================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    colors = {'N': 'C0', 'S': 'C1', 'V0': 'C2'}
    labels = {'N': r'$F_N$ (phase sum)', 'S': r'$F_S$', 'V0': r'$F_{V_0}$'}

    for i, ch in enumerate(['N', 'S', 'V0']):
        F = channels[ch]['F']
        F_err = channels[ch]['F_err']
        col = colors[ch]

        # Top row: two-state fit
        ax = axes[0, i]
        ax.errorbar(tau, F / N_coord, yerr=F_err / N_coord,
                     fmt='.', color=col, ms=4, alpha=0.7, label='Data')

        ts = two_state_results[ch]
        if ts[0] is not None:
            tau_fine = np.linspace(tau[0], tau[-1], 200)
            fit_curve = two_state_model(tau_fine, *ts[0])
            ax.plot(tau_fine, fit_curve / N_coord, 'k-', lw=1.5, label='Two-state fit')
            ax.axhline(ts[0][0] / N_coord, color='r', ls='--', lw=1,
                       label=f'$F_0/N$ = {ts[0][0]/N_coord:.4f}({ts[1][0]/N_coord*1e4:.1f})')
            ax.set_title(f'{labels[ch]} — Two-state fit\n'
                         f'$\\Delta E$ = {ts[0][2]:.4f}, $\\chi^2$/dof = {ts[2]:.2f}')
        else:
            ax.set_title(f'{labels[ch]} — Two-state fit FAILED')

        ax.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
        ax.set_ylabel(r'$F / N_\mathrm{coord}$')
        ax.legend(fontsize=8)
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5)

        # Bottom row: summation method
        ax2 = axes[1, i]
        sm = summ_results[ch]
        S, S_err = sm[0], sm[1]
        ax2.errorbar(tau, S / N_coord, yerr=S_err / N_coord,
                     fmt='.', color=col, ms=4, alpha=0.7, label='$S(\\tau)$')

        if sm[2] is not None:
            tau_fit = tau[len(tau)//2:]
            fit_line = sm[2][0] + sm[2][1] * tau_fit
            ax2.plot(tau_fit, fit_line / N_coord, 'k-', lw=1.5, label='Linear fit')
            ax2.set_title(f'{labels[ch]} — Summation method\n'
                         f'slope/$N$ = {sm[2][1]/N_coord:.4f}({sm[3][1]/N_coord*1e4:.1f}), '
                         f'$\\chi^2$/dof = {sm[4]:.2f}')
        else:
            ax2.set_title(f'{labels[ch]} — Summation FAILED')

        ax2.set_xlabel(r'$\tau$ [MeV$^{-1}$]')
        ax2.set_ylabel(r'$S(\tau) / N_\mathrm{coord}$')
        ax2.legend(fontsize=8)

    plt.tight_layout()
    outfile = h5_file.replace('.h5', '_excited_state_analysis.png')
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved to {outfile}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python excited_state_analysis.py <currents_3pt_h5_file>")
        sys.exit(1)
    main(sys.argv[1])
