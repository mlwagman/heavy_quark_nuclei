#!/usr/bin/env python3
r"""Does the charge radius of an N-quark Coulomb baryon lie on one curve in N?

All at :math:`\alpha = 1`, where Coulomb scaling makes everything else a rescale.  Inside the
totally antisymmetric :math:`N = N_c` colour singlet every pair sits at :math:`-\kappa_N/r` with

.. math::
    \kappa_N = \frac{\alpha C_F}{N_c - 1} = \frac{N_c + 1}{2 N_c} ,

which is the generator's ``'OA'`` channel.  At :math:`N_c = 2` that is :math:`3/4` -- and the
:math:`q\bar q` singlet coupling :math:`\alpha C_F` is **also** :math:`3/4`, so at :math:`N_c=2`
the diquark and the meson are the same Coulomb problem exactly.  (The :math:`2/3` vs :math:`4/3`
split at :math:`N_c=3` closes as :math:`N_c \to 2`.)  So the :math:`N=2` end of the curve is
closed form,

.. math::
    a_0 = \frac{2}{\kappa}, \qquad \langle r^2\rangle_{\rm ch} = \tfrac34 a_0^2
        = \frac{3}{\kappa^2} ,

and it anchors the large-:math:`N_c` extrapolation from below exactly as the SVM anchors
:math:`N = 3`.

The coupling varies only mildly along the sequence (:math:`\kappa = 0.75, 0.667, 0.625, 0.6`),
so divide it out: :math:`R_N = \langle r^2\rangle_{\rm ch}\,\kappa_N^2` is pure :math:`N`
dependence, and :math:`1/\sqrt{R_N}` -- the inverse rms charge radius in units of
:math:`\kappa` -- comes out **linear in N**.  That is the large-:math:`N_c` statement: the
baryon shrinks like :math:`1/N_c`.

Usage:  nc_scaling_curve.py
"""
import numpy as np

# pair coupling inside the antisymmetric N = Nc singlet, at alpha = 1
K = {N: (N + 1) / (2 * N) for N in (2, 3, 4, 5, 6)}

# exact / variational references
REF = {2: 3.0 / K[2] ** 2,        # closed form, 16/3
       3: 3.8260983,              # svm_nbody.py, = the published Coulomb baryon 3.826
       4: 2.7191553,              # svm_nbody.py, ~0.7% (virial 1.001)
       5: 2.015}                  # svm_nbody.py, NOT converged (virial 1.006..1.047): indicative
TAG = {2: 'exact 3/kappa^2', 3: 'SVM', 4: 'SVM', 5: 'SVM approx'}

# GFMC at alpha = 1 (run/19_nc_alpha1.sh).  N = 2 is also measured independently by the alpha=0.1
# fo_meson_lo run, 164.1 +/- 3.1 at kappa = 4/30, carried over by <r^2> -> <r^2>(k_old/k_new)^2.
k_old = 4.0 / 3.0 * 0.1
GFMC = {2: (164.1 * (k_old / K[2]) ** 2, 3.1 * (k_old / K[2]) ** 2, 'fo_meson_lo rescaled'),
        3: (3.897, 0.063, 'nc3_baryon'),
        4: (2.696, 0.034, 'nc4_baryon')}

print(f"alpha = 1.  <r^2>_ch is the one-body charge radius about the CM.\n")
print(f"{'N':>2} {'kappa':>7} {'ref':>10} {'source':>16} {'GFMC':>18} {'pull':>8}"
      f" {'R = <r^2>k^2':>13} {'1/sqrt(R)':>10}")
for N in sorted(REF):
    k, ref = K[N], REF[N]
    R = ref * k * k
    g = GFMC.get(N)
    gs = f'{g[0]:9.3f} +/- {g[1]:<5.3f}' if g else f'{"(running)":>18}'
    ps = f'{100 * (g[0] / ref - 1):+7.2f}%' if g else f'{"":>8}'
    print(f'{N:2d} {k:7.4f} {ref:10.5f} {TAG[N]:>16} {gs} {ps} {R:13.5f} {1 / np.sqrt(R):10.4f}')

Ns = np.array(sorted(REF), float)
y = np.array([1 / np.sqrt(REF[int(n)] * K[int(n)] ** 2) for n in Ns])
a, b = np.polyfit(Ns, y, 1)
res = 100 * ((a * Ns + b) / y - 1)
print(f'\n  1/(kappa sqrt(<r^2>_ch)) = {b:+.4f} {a:+.4f} N'
      f'   residuals: ' + '  '.join(f'{r:+.2f}%' for r in res))
print(f'  => sqrt(<r^2>_ch) kappa = 1/({b:.4f} + {a:.4f} N): the charge radius falls like 1/N,')
print(f'     and since kappa -> 1/2, the radius itself goes as ~{2 / a:.1f}/N at large N.')
for N in (6, 8, 10):
    kk = (N + 1) / (2 * N)
    print(f'     extrapolation: N={N:2d}  <r^2>_ch = {1 / ((a * N + b) ** 2 * kk ** 2):7.4f}'
          f'   rms = {1 / ((a * N + b) * kk):6.4f}')
print('\n  Caveat: the N=5 reference is NOT converged and is shown only to place the point;')
print('  the fit above uses it, so re-run when the GFMC N=5 number lands and judge against that.')
