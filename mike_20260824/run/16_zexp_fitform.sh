#!/bin/bash
# Does the fit ansatz explain the ~+2% NLO excess in the MRS radii?
#
# Every pull below is FIT-TO-FIT: --ref_ff hands the extraction the EXACT F(Q)/F(0) on the very
# same Q grid, and the identical fit is applied to it, so the quoted target already contains
# whatever bias the ansatz carries.  The reference form factors come from Ben's variational
# solver on his branch (svm_baryon.py --potential mrs, N=2 and N=3), run at our exact Q values:
# it reproduces the published E0 and <r^2> to every printed digit (56.692063 / 114.2523 at LO,
# 52.071288 / 103.14328 at NLO), so these are the same references the table was judged against.
#
# t_cut is the branch point in Q^2, fixed by the exponential falloff of the ONE-BODY density:
#   meson    x = r/2,            |psi|^2 ~ e^{-2 kappa r}   =>  t_cut = 16 kappa^2 = 16|E0|
#   baryon   x = -2y/3,          |psi|^2 ~ e^{-2 kappa_2 y} =>  t_cut =  9 kappa_2^2 = 12|E_sep|
# with kappa = sqrt(2 mu |E|), mu = 1/2 (meson) and 2/3 (quark vs. the remaining pair), and
# E_sep = E0(qqq) - E0(qq in 3bar).  The 3bar diquark energies come from the same radial solve
# with the color factor set to 1 instead of Nc-1:
#   LO   E0(qqq) = -0.020959979   E0(qq) = -0.005324798   =>  t_cut = 0.1876222
#   NLO  E0(qqq) = -0.023149129   E0(qq) = -0.005863789   =>  t_cut = 0.2074241
#   LO   meson E0 = -0.017505892  => t_cut = 0.2800943
#   NLO  meson E0 = -0.019235763  => t_cut = 0.3077722
# Reproducing the reference form factors (svm_baryon.py --potential mrs lives on Ben's branch
# origin/fix/3pt-hankel-offdiag, not in this bundle; it needs numpy>=2 for np.trapezoid, or a
# np.trapz shim on the jax interpreter, and the MRS generator in the working directory):
#
#   git show origin/fix/3pt-hankel-offdiag:svm_baryon.py > svm_baryon_mrs.py
#   python svm_baryon_mrs.py --alpha 0.1 --N 2 --potential mrs --OLO LO  --mu 0.0362001 \
#       --mufac 1 --ascale 0.5796  --nbasis 250 --qs 0.05,0.1,0.15,0.2,0.3,0.4
#   python svm_baryon_mrs.py --alpha 0.1 --N 3 --symmetrize --potential mrs --OLO LO \
#       --mu 0.0362001 --mufac 1 --ascale 0.5796 --nbasis 600 --qs 0.04,0.08,0.12,0.16,0.22,0.28
#
# with --ascale = sqrt(r2_meson/r2_Coulomb) = sqrt(r2_meson*16*alpha^2/27): 0.5796 at LO and
# 0.55548 at NLO.  Those runs return E0 and <r^2> identical to every published digit
# (56.692063 / 114.2523 at LO, 52.071288 / 103.14328 at NLO), which is the check that these are
# the same references the campaign table was judged against.  The t_cut inputs come from
# ../../mrs_two_body.py.
set -eu
. "$(dirname "$0")/00_env.sh"

PY312=${PY312:-/usr/local/bin/python3.12}
GEVP="$ROOT/../gfmc_me_gevp.py"
O="$OUT/refmass_a0.1"
A="$O/analysis"
mkdir -p "$A"
MS=${MS:-"4 6 8 10 12 14 16"}

FF_MESON_LO="0.05:0.976760,0.1:0.911376,0.15:0.815224,0.2:0.702806,0.3:0.479754,0.4:0.305037"
FF_MESON_NLO="0.05:0.978621,0.1:0.918118,0.15:0.828143,0.2:0.721327,0.3:0.503541,0.4:0.326509"
FF_BARYON_LO="0.04:0.970108,0.08:0.886924,0.12:0.767227,0.16:0.631715,0.22:0.437099,0.28:0.282113"
FF_BARYON_NLO="0.04:0.972960,0.08:0.897128,0.12:0.786384,0.16:0.658417,0.22:0.468877,0.28:0.311722"

run () {   # name system ref t_cut ref_ff
  name=$1; system=$2; ref=$3; tc=$4; ff=$5
  d="$O/$name"
  [ -d "$d" ] || { echo "-- $name: no run dir, skipping"; return; }
  for spec in "dipole|" "z|--t_cut $tc --k_max 2" "z|--t_cut $tc --k_max 3" "z|--t_cut $tc --k_max 4"; do
    form=${spec%%|*}; extra=${spec#*|}
    tag=$form$(echo "$extra" | sed -n 's/.*--k_max \([0-9]*\).*/_k\1/p')
    echo
    echo "=============== $name   fit=$tag   (reference $ref) ==============="
    "$PY312" -I "$GEVP" --dir "$d" --system "$system" --ref "$ref" --ref_ff "$ff" \
        --fit "$form" $extra --spike_leverage 100 --gevp_cut 1e-2 --n_boot 200 --m $MS \
        2>&1 | tee "$A/${name}_${tag}.txt" | grep -vE "^\s*$|^  F\(0\) must|^  deviation|^  and judge"
  done
}

# RESULT, m = 16, every pull fit-to-fit:
#
#   config            fit      GFMC <r^2>        target   fit-form bias   pull
#   mrs_meson_lo      dipole   56.90 +/- 1.04    59.114        +4.27%     -2.1s
#   mrs_meson_lo      z k=3    55.66 +/- 1.08    56.344        -0.61%     -0.6s
#   mrs_meson_lo      z k=4    55.93 +/- 1.16    56.677        -0.03%     -0.6s
#   mrs_meson_nlo     dipole   53.64 +/- 1.10    54.600        +4.86%     -0.9s
#   mrs_meson_nlo     z k=3    52.61 +/- 1.15    51.826        -0.47%     +0.7s
#   mrs_meson_nlo     z k=4    52.81 +/- 1.17    52.063        -0.02%     +0.6s
#   mrs_baryon_lo     dipole  114.6  +/- 1.75   114.181        -0.06%     +0.3s
#   mrs_baryon_lo     z k=3   114.3  +/- 1.72   113.456        -0.70%     +0.5s
#   mrs_baryon_lo     z k=4   115.0  +/- 1.82   114.212        -0.04%     +0.4s
#   mrs_baryon_nlo    dipole  105.3  +/- 1.42   103.091        -0.05%     +1.5s
#   mrs_baryon_nlo    z k=3   105.1  +/- 1.39   102.609        -0.52%     +1.8s
#   mrs_baryon_nlo    z k=4   105.3  +/- 1.41   103.120        -0.02%     +1.6s
#
# ANSWER: the z-expansion does NOT improve the NLO tension.
#
#   * For the BARYON the free-p dipole was already faithful -- applied to the exact F it
#     returns the exact radius to -0.06% at LO and -0.05% at NLO -- so there was no fit-form
#     bias to remove, and z k=4 reproduces it digit for digit (115.0 vs 114.6, 105.3 vs 105.3).
#     The NLO pull goes +1.5s -> +1.6s.  Ben's own table says the same thing from the other
#     side: his r2_z and r2_dipole agree to ~0.1% at every coupling.
#   * For the MESON the z-expansion is a genuine improvement, but of the TARGET, not the
#     tension: the p=2 dipole is exact only for Coulomb and over this Q grid it biases the
#     radius by +4.3% (LO) and +4.9% (NLO), so the dipole number was being compared against
#     a target 4-5% too large.  z k=4 removes that (bias <= 0.03%) and the LO/NLO split is
#     unchanged: -1.3% and +1.4% against the true values.
#   * k_max = 2 -- Ben's "three-term" expansion is K=3 in his counting, i.e. z^1..z^3 = our
#     k_max 3; k_max 2 is NOT enough here, biasing low by 5.5-6.5% on the exact F.  Use 3 or 4.
#   * The cleanest statement needs no fit at all.  (1 - F)/(1 - F_exact) at fixed Q tends to
#     the radius ratio as Q -> 0, and at the lowest Q it reads
#         meson  LO  0.987 +/- 0.019    baryon LO  1.006 +/- 0.015
#         meson  NLO 1.013 +/- 0.021    baryon NLO 1.021 +/- 0.015
#     Both NLO channels high, both LO channels at zero, and never more than 1.4 sigma.  The
#     NLO excess is in the data, not in the ansatz.  With the 1.15x error inflation measured
#     from the 16-seed replica study, the baryon NLO is +1.2 sigma.
#
# The fit itself is F/F(0) = 1 + sum_{k=1..k_max} c_k z^k with the constant FIXED to 1 (same
# convention as Ben's paper_korr.py zfit).  Letting c_0 float costs ~1% of bias at k_max = 3,
# which is the size of the effect under study.

run mrs_meson_lo   meson  56.692063 0.2800943 "$FF_MESON_LO"
run mrs_meson_nlo  meson  52.071288 0.3077722 "$FF_MESON_NLO"
run mrs_baryon_lo  baryon 114.2523  0.1876222 "$FF_BARYON_LO"
run mrs_baryon_nlo baryon 103.14328 0.2074241 "$FF_BARYON_NLO"
