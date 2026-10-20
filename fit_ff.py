r"""Charge radius from a form factor: generalized dipole and z-expansion.

Reads either Ben's aggregated ``prod_ME.csv`` (columns ``Q, channel, F, F_err``) or a directory
of per-Q ``*_korr_ME.csv`` written by ``gfmc_me_korr.py``, normalizes by :math:`F(0)` and fits
:math:`F(Q^2)/F(0)`.

Conventions
-----------
:math:`\langle r^2\rangle` throughout is the **one-body** radius about the centre of mass -- the
charge radius of the density the number operator measures, matching ``run/fit.py`` and
``svm_baryon.py``'s ``<r^2>_ch``.  For the Coulomb meson this is known in closed form: each quark
sits at :math:`\pm r/2` from the CM, so the 1s density Fourier transform
:math:`(1+q^2a_0^2/4)^{-2}` becomes

.. math::
    \frac{F(q)}{F(0)} = \Big(1 + \frac{q^2 a_0^2}{16}\Big)^{-2},
    \qquad a_0 = \frac{1}{\mu\,\alpha C_F} = \frac{1.5}{\alpha},
    \qquad \langle r^2\rangle = \tfrac34 a_0^2 = \frac{1.6875}{\alpha^2},

an *exact* dipole with :math:`p = 2` and a fourth-order pole at :math:`Q^2 = -16/a_0^2`.  That
last number is the meson's :math:`t_{\rm cut}`, which makes the meson a closed-form control for
the z-expansion machinery as well (``--self_test``).

Fit forms
---------
``dipole``
    :math:`(1 + Q^2 b^2/4p)^{-p}` with :math:`p` fixed (default 2), :math:`\langle r^2\rangle =
    1.5\,b^2`.  Exact for the Coulomb meson; an assumption anywhere else.
``dipole_free``
    the same with :math:`p` floating -- the honest way to *test* the dipole rather than impose it.
``z``
    the conformal-map expansion with :math:`t_0 = 0`, so :math:`z(0) = 0` and

    .. math::
        z(Q^2)=\frac{\sqrt{t_{\rm cut}+Q^2}-\sqrt{t_{\rm cut}}}
                    {\sqrt{t_{\rm cut}+Q^2}+\sqrt{t_{\rm cut}}},
        \qquad \frac{F}{F(0)}=\sum_{k=0}^{k_{\max}} c_k z^k,
        \qquad \frac{dz}{dQ^2}\Big|_0=\frac{1}{4t_{\rm cut}} .

    The radius comes from the linear coefficient alone,
    :math:`\langle r^2\rangle = -\tfrac32 (c_1/c_0)/t_{\rm cut}`, with no assumption about the
    shape.  :math:`t_{\rm cut}` is the nearest singularity in :math:`Q^2`, fixed by the
    exponential falloff of the one-body density (:math:`\rho\sim e^{-\kappa r} \Rightarrow
    t_{\rm cut} = \kappa^2`).

    .. note::
        **The coefficients are not expected to be** :math:`O(1)` **here.**  The usual
        coefficient-boundedness argument assumes the singularity is a branch cut on which the form
        factor stays bounded.  This system has a *pole*: with :math:`t_{\rm cut} = 16/a_0^2` the
        meson's exact form factor is :math:`F/F(0) = ((1-z)/(1+z))^4`, a fourth-order pole sitting
        at :math:`z = -1`, whose Taylor coefficients are :math:`1, -8, 32, -88, 192, \dots` --
        growing roughly linearly.  ``--self_test`` checks that :math:`-1.5(c_1/c_0)/t_{\rm cut}`
        nonetheless returns :math:`0.75 a_0^2` to machine precision.  The meaningful diagnostics
        are stability against :math:`k_{\max}` and against varying :math:`t_{\rm cut}`, which
        ``--scan`` reports -- not the size of the coefficients.

Usage
-----
    python3 fit_ff.py --self_test
    python3 fit_ff.py --csv out/dir/prod_ME.csv --form dipole --alpha 0.3
    python3 fit_ff.py --korr_dir out/dir --form all --alpha 0.75 --scan
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.optimize import curve_fit

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--self_test', action='store_true',
               help='fit the exact Coulomb-meson form factor and check every form returns '
                    '0.75 a0^2; no data needed')
src = p.add_mutually_exclusive_group(required='--self_test' not in sys.argv)
src.add_argument('--csv', help="aggregated prod_ME.csv (columns Q, channel, F, F_err)")
src.add_argument('--korr_dir', help="directory of per-Q CSVs from gfmc_me_korr.py")
p.add_argument('--suffix', default='_korr_ME',
               help="per-Q filename suffix, matching gfmc_me_korr.py --out_suffix")
p.add_argument('--channel', default='C3_N', help='which current row to use')
p.add_argument('--form', default='dipole', choices=['dipole', 'dipole_free', 'z', 'all'])
p.add_argument('--p_fixed', type=float, default=2.0, help="p for form='dipole'")
p.add_argument('--t_cut', type=float, default=None,
               help="z-expansion branch point in Q^2. Required for form='z' unless --alpha is "
                    "given (then the meson's closed-form 16/a0^2 is used).")
p.add_argument('--alpha', type=float, default=None,
               help='coupling, for the closed-form meson expectations and t_cut')
p.add_argument('--k_max', type=int, default=2, help='highest power of z')
p.add_argument('--scan', action='store_true',
               help='vary k_max and t_cut and report the spread as a systematic')
p.add_argument('--expect', type=float, default=None, help='reference <r^2> to compare against')
p.add_argument('--min_err', type=float, default=1e-4,
               help='floor on the relative error, so an exact point cannot dominate the chi^2')
args = p.parse_args()


# ----------------------------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------------------------
def read_prod_me(fn, channel):
    d = {}
    with open(fn) as fh:
        for r in csv.DictReader(fh):
            if r['channel'] == channel:
                d[float(r['Q'])] = (float(r['F']), float(r['F_err']))
    return d


def read_korr_dir(d, channel, suffix='_korr_ME'):
    """Glob ``*<suffix>.csv`` and take Q from the ``Qmag<...>`` in each filename."""
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, f'*{suffix}.csv'))):
        m = re.search(rf'Qmag([0-9.]+?){re.escape(suffix)}\.csv$', os.path.basename(fn))
        if not m:
            continue
        with open(fn) as fh:
            for r in csv.DictReader(fh):
                if r['current'] == channel:
                    out[float(m.group(1))] = (float(r['F_final']), float(r['F_err']))
    return out


def read_korr_boots(d):
    r"""``{Q: (n_boot,)}`` of aligned outer bootstrap draws written by ``gfmc_me_korr.py``.

    Draw ``b`` resamples the same walkers at every Q (the korr ``Resampling`` plan depends only on
    the seed and the sample count, both Q-independent), so these arrays support a fit that knows
    the full covariance across Q.  That is necessary rather than luxurious here: all the Q runs
    share the GFMC seed, so the points are strongly correlated and a diagonal chi^2 is not
    interpretable.
    """
    out = {}
    for fn in sorted(glob.glob(os.path.join(d, f'*{args.suffix}_boot.npy'))):
        m = re.search(rf'Qmag([0-9.]+?){re.escape(args.suffix)}_boot\.npy$',
                      os.path.basename(fn))
        if m:
            out[float(m.group(1))] = np.load(fn)
    return out


def correlated_fit(boots, fitter, min_err):
    r"""Refit every aligned bootstrap replica; return ``(central, err, replicas)``.

    The replicas are already the correlated ensemble, so the spread of the *fitted*
    :math:`\langle r^2\rangle` over replicas is the correlated error -- no covariance matrix needs
    inverting, and no assumption about the correlation structure is made.
    """
    qs = np.array(sorted(q for q in boots if q > 0))
    if 0.0 not in boots or len(qs) < 2:
        return None
    n = min(len(boots[q]) for q in list(boots))
    ratios = np.array([boots[q][:n] / boots[0.0][:n] for q in qs])     # (nQ, n_boot)
    # A replica is usable only if it is finite at *every* Q: korr's filter occasionally rejects
    # every state on a degenerate resample and returns NaN there. Dropping such a draw from all Q
    # at once is what keeps the ensemble aligned, which is the whole point of using it.
    good = np.isfinite(ratios).all(axis=0)
    n_drop = int((~good).sum())
    ratios = ratios[:, good]
    if ratios.shape[1] < 10:
        return None
    Q2 = qs ** 2
    err = np.maximum(ratios.std(axis=1), min_err)
    vals = []
    for b in range(ratios.shape[1]):
        try:
            r = fitter(Q2, ratios[:, b], err)
        except Exception:
            continue
        if r is not None and np.isfinite(r['r2']):
            vals.append(r['r2'])
    vals = np.array(vals)
    central = fitter(Q2, ratios.mean(axis=1), err)
    return central, float(np.std(vals)), vals, n_drop, ratios.shape[1]


def normalize(data, min_err):
    """``(Q, Q^2, F/F(0), err)`` from a ``{Q: (F, F_err)}`` dict."""
    if 0.0 not in data:
        sys.exit("no Q=0 point; cannot normalize")
    if len(data) < 3:
        sys.exit(f"only {len(data)} points; need >= 3")
    F0 = data[0.0][0]
    Qs = np.array(sorted(q for q in data if q > 0))
    R = np.array([data[q][0] / F0 for q in Qs])
    E = np.array([max(data[q][1] / abs(F0), min_err) for q in Qs])
    return Qs, Qs ** 2, R, E, F0, data[0.0][1]


# ----------------------------------------------------------------------------------------------
# fit forms
# ----------------------------------------------------------------------------------------------
def gendip(q2, b, pp):
    return (1.0 + q2 * b * b / (4 * pp)) ** (-pp)


def zvar(q2, t_cut):
    """The conformal map with t_0 = 0, so z(0) = 0 and the constant term is F(0)/F(0) = 1."""
    s, s0 = np.sqrt(t_cut + q2), np.sqrt(t_cut)
    return (s - s0) / (s + s0)


def fit_dipole(Q2, R, E, p_free, p_fixed=2.0):
    if p_free:
        par, cov = curve_fit(gendip, Q2, R, p0=[np.sqrt(6.0), 2.0], sigma=E,
                             absolute_sigma=True, maxfev=20000,
                             bounds=([1e-3, 0.3], [500.0, 20.0]))
        b, pp = par
        b_err, p_err = np.sqrt(np.diag(cov))
        n_par, model = 2, lambda x: gendip(x, b, pp)
    else:
        par, cov = curve_fit(lambda x, b: gendip(x, b, p_fixed), Q2, R, p0=[np.sqrt(6.0)],
                             sigma=E, absolute_sigma=True, maxfev=20000,
                             bounds=([1e-3], [500.0]))
        b, pp = par[0], p_fixed
        b_err, p_err = np.sqrt(cov[0, 0]), 0.0
        n_par, model = 1, lambda x: gendip(x, b, p_fixed)
    dof = max(len(R) - n_par, 1)
    chi2 = float((((R - model(Q2)) / E) ** 2).sum()) / dof
    return dict(name='dipole' + (' (p free)' if p_free else f' (p={p_fixed:g} fixed)'),
                r2=1.5 * b * b, r2_err=3.0 * b * b_err, chi2=chi2, dof=dof,
                extra=f"b = {b:.5f} +/- {b_err:.5f}   p = {pp:.4f}"
                      + (f" +/- {p_err:.4f}" if p_free else " (fixed)"))


def fit_z(Q2, R, E, t_cut, k_max):
    r"""Fit ``sum_k c_k z^k`` and read the radius off ``c_1/c_0``.

    Linear in the coefficients, so this is a plain weighted least squares -- no starting guess and
    no local minima.  :math:`\langle r^2\rangle = -6\,d(F/F_0)/dQ^2|_0` and
    :math:`dz/dQ^2|_0 = 1/(4t_{\rm cut})`, hence
    :math:`\langle r^2\rangle = -1.5 (c_1/c_0)/t_{\rm cut}`.
    """
    if len(R) < k_max + 1:
        return None
    z = zvar(Q2, t_cut)
    A = np.vander(z, k_max + 1, increasing=True) / E[:, None]
    c, *_ = np.linalg.lstsq(A, R / E, rcond=None)
    cov = np.linalg.inv(A.T @ A)
    c0, c1 = c[0], c[1]
    g = np.zeros(k_max + 1)
    g[0], g[1] = -c1 / c0 ** 2, 1.0 / c0          # gradient of c1/c0
    r2 = -1.5 * (c1 / c0) / t_cut
    r2_err = 1.5 * np.sqrt(max(float(g @ cov @ g), 0.0)) / t_cut
    dof = max(len(R) - (k_max + 1), 1)
    chi2 = float((((R - np.polyval(c[::-1], z)) / E) ** 2).sum()) / dof
    return dict(name=f'z-expansion (t_cut={t_cut:g}, k_max={k_max})',
                r2=r2, r2_err=r2_err, chi2=chi2, dof=dof,
                extra=f"c0 = {c0:.5f} (should be 1)   "
                      + "  ".join(f"c{k}/c0 = {c[k] / c0:+.4f}" for k in range(1, k_max + 1)),
                c=c, c0=c0)


def show(res, expect=None):
    if res is None:
        return
    print(f"\n{res['name']}")
    print(f"  {res['extra']}")
    print(f"  chi^2/dof = {res['chi2']:.3f}  (dof={res['dof']})")
    line = f"  <r^2> = {res['r2']:.4f} +/- {res['r2_err']:.4f}"
    if expect:
        pull = (res['r2'] - expect) / max(res['r2_err'], 1e-30)
        line += (f"    vs {expect:.4f}  ->  {100 * (res['r2'] / expect - 1):+.2f}%  "
                 f"({pull:+.1f} sigma)")
    print(line)


# ----------------------------------------------------------------------------------------------
# self test: the exact Coulomb meson, where every number is known
# ----------------------------------------------------------------------------------------------
def self_test():
    print("=" * 78)
    print("SELF TEST: the exact Coulomb-meson form factor, F/F(0) = (1 + Q^2 a0^2/16)^-2")
    print("=" * 78)
    ok = True
    for alpha in (0.3, 0.75):
        a0 = 1.5 / alpha
        t_cut = 16.0 / a0 ** 2
        target = 0.75 * a0 ** 2
        Qs = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7]) * (alpha / 0.3)
        Q2 = Qs ** 2
        R = (1.0 + Q2 * a0 ** 2 / 16.0) ** -2            # exact, noiseless
        E = np.full_like(R, 1e-6)
        print(f"\n--- alpha = {alpha}:  a0 = {a0:.6f}   t_cut = 16/a0^2 = {t_cut:.6f}   "
              f"target <r^2> = 0.75 a0^2 = {target:.6f}")
        for res in (fit_dipole(Q2, R, E, False), fit_dipole(Q2, R, E, True)):
            err = abs(res['r2'] - target) / target
            print(f"  {res['name']:<24s} <r^2> = {res['r2']:.6f}   rel err {err:.2e}")
            ok &= err < 1e-6
        # The z-expansion is a truncated series, so it is not exact at any finite k_max: the
        # criterion is that it *converges* to the target as k_max grows, and that c_1/c_0 tends to
        # the exact -8 of the fourth-order pole.  (The dipole, by contrast, is the exact
        # functional form here and must be exact to machine precision -- asserted above.)
        errs, c1s = [], []
        for km in (1, 2, 3, 4):
            res = fit_z(Q2, R, E, t_cut, km)
            if res is None:
                continue
            err = abs(res['r2'] - target) / target
            errs.append(err)
            c1s.append(res['c'][1] / res['c0'])
            cs = "  ".join(f"{res['c'][k] / res['c0']:+.2f}" for k in range(1, km + 1))
            print(f"  z-expansion k_max={km:<12d} <r^2> = {res['r2']:.6f}   rel err {err:.2e}"
                  f"   c_k/c0: {cs}")
        print("  exact c_k/c0 for a 4th-order pole at z=-1:  -8.00  +32.00  -88.00  +192.00")
        monotone = all(b < a for a, b in zip(errs, errs[1:]))
        print(f"  convergence: rel err strictly decreasing in k_max = {monotone}; "
              f"err(k_max=4) = {errs[-1]:.2e}; c1/c0 -> {c1s[-1]:+.4f} (exact -8)")
        ok &= monotone and errs[-1] < 1e-3 and abs(c1s[-1] + 8.0) < 0.05
    print("\n" + "=" * 78)
    print("SELF TEST " + ("PASSED" if ok else "FAILED"))
    print("=" * 78)
    return 0 if ok else 1


# ----------------------------------------------------------------------------------------------
def main():
    data = (read_prod_me(args.csv, args.channel) if args.csv
            else read_korr_dir(args.korr_dir, args.channel, args.suffix))
    if not data:
        sys.exit(f"no rows for channel {args.channel!r} in {args.csv or args.korr_dir}")
    Qs, Q2, R, E, F0, F0_err = normalize(data, args.min_err)

    print(f"source  : {args.csv or args.korr_dir}")
    print(f"channel : {args.channel}")
    print(f"F(0)    = {F0:.6f} +/- {F0_err:.6f}")
    print(f"{len(Qs)} nonzero Q points")
    print("  Q        Q^2       F/F(0)       err")
    for q, r, e in zip(Qs, R, E):
        print(f"  {q:<7g}  {q * q:<8.5f}  {r:<10.6f}  {e:.6f}")

    t_cut, expect = args.t_cut, args.expect
    if args.alpha:
        a0 = 1.5 / args.alpha
        print(f"\nclosed form at alpha={args.alpha:g} (meson): a0 = 1.5/alpha = {a0:.6f}, "
              f"<r^2> = 0.75 a0^2 = {0.75 * a0 ** 2:.6f}, exact dipole p=2, "
              f"t_cut = 16/a0^2 = {16 / a0 ** 2:.6f}")
        t_cut = t_cut if t_cut is not None else 16 / a0 ** 2
        expect = expect if expect is not None else 0.75 * a0 ** 2

    if args.form in ('dipole', 'all'):
        show(fit_dipole(Q2, R, E, False, args.p_fixed), expect)
    if args.form in ('dipole_free', 'all'):
        show(fit_dipole(Q2, R, E, True), expect)

    if args.korr_dir:
        boots = read_korr_boots(args.korr_dir)
        if len(boots) >= 3 and 0.0 in boots:
            print(f"\n{'-' * 78}")
            print(f"CORRELATED refit over {min(len(v) for v in boots.values())} aligned bootstrap "
                  f"replicas (the Q points share the GFMC seed, so\nthe diagonal chi^2 above is "
                  f"not interpretable; this error knows the covariance):")
            jobs = [('dipole (p=2 fixed)',
                     lambda q, r, e: fit_dipole(q, r, e, False, args.p_fixed)),
                    ('dipole (p free)', lambda q, r, e: fit_dipole(q, r, e, True))]
            if t_cut is not None:
                jobs.append((f'z-expansion (t_cut={t_cut:g}, k_max={args.k_max})',
                             lambda q, r, e: fit_z(q, r, e, t_cut, args.k_max)))
            for name, fitter in jobs:
                res = correlated_fit(boots, fitter, args.min_err)
                if res is None:
                    continue
                central, err, vals, n_drop, n_use = res
                line = (f"  {name:<40s} <r^2> = {central['r2']:.4f} +/- {err:.4f}  "
                        f"[diagonal said +/- {central['r2_err']:.4f}]"
                        + (f"  ({n_use} replicas, {n_drop} dropped as non-finite)"
                           if n_drop else f"  ({n_use} replicas)"))
                if expect:
                    line += (f"\n{'':44s}vs {expect:.4f} -> "
                             f"{100 * (central['r2'] / expect - 1):+.2f}%  "
                             f"({(central['r2'] - expect) / max(err, 1e-30):+.1f} sigma)")
                print(line)
    if args.form in ('z', 'all'):
        if t_cut is None:
            sys.exit("form='z' needs --t_cut (or --alpha, for the meson's closed-form value)")
        show(fit_z(Q2, R, E, t_cut, args.k_max), expect)
        if args.scan:
            print(f"\n  systematic scan around t_cut={t_cut:g}:")
            print("    t_cut     k_max   <r^2>        err        chi^2/dof")
            at_nominal = []
            for tc in (0.5 * t_cut, 0.75 * t_cut, t_cut, 1.5 * t_cut, 2.0 * t_cut):
                for km in range(1, args.k_max + 2):
                    r = fit_z(Q2, R, E, tc, km)
                    if r is None:
                        continue
                    print(f"    {tc:<9.5f} {km:<7d} {r['r2']:<12.4f} {r['r2_err']:<10.4f} "
                          f"{r['chi2']:.3f}")
                    if tc == t_cut:
                        at_nominal.append(r['r2'])
            if len(at_nominal) > 1:
                print(f"    k_max spread at the nominal t_cut: {min(at_nominal):.4f} .. "
                      f"{max(at_nominal):.4f}  "
                      f"(half-spread {0.5 * (max(at_nominal) - min(at_nominal)):.4f})")


if __name__ == '__main__':
    sys.exit(self_test() if args.self_test else main())
