#!/bin/bash
# Nc scaling.  Read this header before launching anything: the meson half of the scan is
# partly redundant, and which Nc trajectory to walk is a physics choice, not a default.
#
# ---------------------------------------------------------------------------------------------
# 1.  THE LO MESON Nc SCAN IS EXACTLY REDUNDANT WITH THE alpha SCAN
# ---------------------------------------------------------------------------------------------
# The meson potential is V(r) = -CF * VB_MRS(alpha_s(r); Nc, nf) / r.  Measured directly
# (scratch script vshape.py, reproduced below): in units of Lambda_1loop, and at FIXED
# mu_cut/Lambda_1loop, the ratio V_Nc(x)/V_3(x) at x = r*Lambda_1loop is
#
#     Nc = 4, 5, 6, 8   ->   flat to ALL printed digits, spread +0.00%
#
# so at LO the entire Nc dependence is one overall constant, CF/beta0 up to convention: every
# dimensionless observable is a function of that single number and an LO meson Nc scan carries
# no information an alpha_s scan does not already have.  Keep it anyway as a NULL TEST -- it
# must reproduce the alpha-scan curve exactly, and any deviation is a bug in the Nc plumbing.
#
# At NLO the collapse genuinely fails.  With Ben's mu_cut = 3.4 Lambda_2loop:
#
#     x = r*Lambda_1loop    0.003    0.01    0.03     0.1     0.3       1       3   spread
#     LO   Nc=4            0.9921  0.9924  0.9945  1.0000  0.9982  0.9970  0.9969   +0.79%
#     LO   Nc=8            0.9780  0.9788  0.9848  1.0000  0.9763  0.9616  0.9599   +4.01%
#     NLO  Nc=4            0.9710  0.9671  0.9719  1.0000  1.0037  1.0003  0.9999   +3.67%
#     NLO  Nc=5            0.9494  0.9420  0.9493  1.0000  1.0099  1.0031  1.0022   +6.79%
#     NLO  Nc=6            0.9327  0.9222  0.9308  1.0000  1.0179  1.0077  1.0063   +9.58%
#     NLO  Nc=8            0.9101  0.8940  0.9032  1.0000  1.0377  1.0211  1.0186  +14.36%
#
# (the LO rows are nonzero ONLY because mu_cut = 3.4 Lambda_2loop and Lambda_2loop/Lambda_1loop
# falls 3.2204 -> 1.9386 from Nc=3 to Nc=8, i.e. that convention makes the IR cutoff itself Nc
# dependent; hold mu_cut/Lambda_1loop fixed instead and the LO rows are exactly 1.0000.)
# The NLO drift lives in beta1, aa1 = 31Nc/9, aa2 ~ Nc^2 and the Nc(Nc +- 2)/2 singlet term, and
# the short- and long-distance ends move in OPPOSITE directions, so no coupling rescale can
# absorb it.  Note also that at fixed mu_cut/Lambda_1loop the NLO potential goes pathological as
# Nc grows (it changes sign near x = 0.3 for Nc >= 5), which is presumably why the cutoff is
# tied to the two-loop scale in the first place.
#
# => run the MESON scan at NLO for physics and at LO as a null test.
#
# ---------------------------------------------------------------------------------------------
# 2.  THE BARYON IS WHERE THE Nc PHYSICS IS, AND IT IS NOT alpha SCALING AT ANY ORDER
# ---------------------------------------------------------------------------------------------
# A colour-singlet baryon has N_coord = Nc quarks, hence Nc(Nc-1)/2 pairs each at -CF/(Nc-1),
# so the total attraction grows ~Nc^2/4 while the many-body structure changes outright.  No
# two-body alpha scan reaches it.
#
# The generator now supports it: the N_coord in (2,3) asserts are lifted in
# heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py and in
# afdmc_lib_col/lib_col.py:make_pairwise_singlet_potential, N_coord > 3 requires
# --no_color_vectors (the explicit colour-vector build hardcodes the Nc=3 epsilon tensor), and
# gfmc_me_gevp.py reads N from the Ncoord<N> in the filename.  VERIFIED: an Nc=4, N_coord=4 run
# assigns all six pairs to 'OA', completes end to end, and returns F(0) = 4.0000000000 exactly.
#
# Still missing before this is a measurement rather than a pipeline:
#   * spoila retuning per (Nc, order) -- the trial a0 = spoila*2/VB(Rstar) has no (Nc-1) factor
#     for N_coord > 2, so the widths are wrong by O(1).  Use the existing recipe:
#     radius-match via 15_vmc_spoila_scan.sh, which the VMC study showed IS the variational
#     optimum.
#   * dtau per (Nc, order) -- the bound-state scale moves fast with Nc (Lambda_1loop/mQ goes
#     3.3e-3 -> 5.7e-2 from Nc=3 to Nc=6 at fixed alpha_s(mQ) = 0.1), so dtau must be rescaled
#     or the Trotter error is not controlled.
#   * AN EXACT REFERENCE.  This is the long pole.  svm_baryon.py handles N in (2,3) only; a
#     general-N correlated-Gaussian solve needs the Jacobi/Lambda/W matrices written for
#     arbitrary N.  Without it there is nothing to compare an Nc=4 radius against.
#
# ---------------------------------------------------------------------------------------------
# 3.  THE TRAJECTORY IS AN OPEN CHOICE -- DO NOT HARDCODE ONE
# ---------------------------------------------------------------------------------------------
# At fixed alpha_s(mQ) = 0.1 the coupling strength CF*alpha ~ Nc*alpha/2 GROWS with Nc, and
# Lambda/mQ grows with it (beta0 ~ 11Nc/3), so the scan walks toward strong coupling and loses
# the mQ >> bound-state-scale hierarchy.  The large-Nc limit is normally taken at fixed 't Hooft
# coupling, alpha_s*Nc = const (alpha_s = 0.3/Nc here, matching 0.1 at Nc=3).  Which one is
# wanted depends on whether Nc or the dark coupling is the model input.
#
# mu_cut = 3.4*Lambda_2loop at alpha_s(mQ) = 0.1, from ../../mrs_two_body.py:
#   Nc=3  Lambda_1loop 0.003306098  Lambda_2loop 0.010647099  mu_cut 0.036200136
#   Nc=4  Lambda_1loop 0.01378754   Lambda_2loop 0.039046485  mu_cut 0.13275805
#   Nc=5  Lambda_1loop 0.032478012  Lambda_2loop 0.082348501  mu_cut 0.2799849
#   Nc=6  Lambda_1loop 0.057498678  Lambda_2loop 0.13206518   mu_cut 0.4490216
# Recompute these for any other alpha_s: mrs_two_body.py --Nc <Nc> --alpha_s <a> prints them.
# ---------------------------------------------------------------------------------------------
# 4.  DECIDED: FIXED-ORDER COULOMB AT alpha = 1, AND RESCALE
# ---------------------------------------------------------------------------------------------
# Pure Coulomb scales exactly -- E0 ~ alpha^2, <r^2> ~ 1/alpha^2, F a function of q/alpha -- so
# one set of runs at alpha = 1 covers every coupling and BOTH trajectories in section 3 by
# rescaling.  That retires the trajectory question instead of answering it.  Use the FO
# generator (heavy_quark_nuclei_gfmc_boosted_FV_currents.py --alpha 1), not the MRS one.
#
# Reference: ../../svm_nbody.py, a general-N correlated-Gaussian solve (Ben's svm_baryon.py
# matrix elements, with the Jacobi coordinates generated for arbitrary N instead of tabulated
# for N in (2,3)).  Validated: N=2 gives E0 = -0.44443853 and <r^2>_ch = 1.6875501 against the
# closed form -4/9 and 27/16, with F(q) matching the exact dipole to 1e-4; N=3 gives
# <r^2>_ch = 3.8260983, exactly the published Coulomb baryon 3.826.  <K>/|E0| = 1 by the virial
# is the convergence monitor.
#
#   Nc   pairs   E0          gap      <r^2>_ch     status
#   3      3     -0.476326   0.2210   3.8260983    converged, matches 3.826
#   4      6     -1.08897    0.3739   2.72(2)      converged to ~0.7% (virial 1.001)
#   5     10     -2.0539     0.5801   2.015(?)     NOT CONVERGED -- virial 1.006..1.096 across
#                                                  seeds; random-width sampling is not enough at
#                                                  nj = 4, needs real SVM one-at-a-time selection
#
# Trial widths: a0(spoila=1) = 2/VB = 2(Nc-1)/(alpha CF) is exactly the Coulomb PAIR Bohr radius
# (3.0, 3.2, 10/3 at Nc = 3, 4, 5 and alpha = 1), and the product-of-pairs trial at that width
# already has nearly the right N-body radius, so spoila ~ 1 for every Nc.  Measured from the
# tau = 0 metropolis samples, c = <r^2>_ch(trial)/a0^2:
#
#   Nc   c                 spoila = sqrt(target/c)/a0
#   3    0.4232 +/- 0.0051   1.002226    (the campaign's 1.01131 at alpha=0.1, as it must be --
#                                         spoila is alpha independent; 0.9% apart, which is the
#                                         precision of c)
#   4    0.2675 +/- 0.0051   0.996484
#   5    0.1874 +/- 0.0037   0.983718
#
# dtau and n_step scaled off the validated Nc=3 setup (dtau 0.027, n_step 200, count_skip 10 at
# alpha=1) holding dtau*|E0| and gap*tau_max fixed, and Q scaled to a common q*sqrt(<r^2>) grid
# so every Nc is compared at the same place on the form factor:
#
#   Nc   dtau       n_step  count_skip  Q values
#   3    0.027      200     10          0  0.2   0.4   0.6   0.8   1.2   1.6
#   4    0.011824   270     14          0  0.24  0.47  0.71  0.95  1.42  1.90
#   5    0.006262   330     17          0  0.27  0.55  0.83  1.11  1.66  2.21
#
# The Nc=3 row is a NULL TEST: rescaled by alpha = 0.1 it must reproduce the campaign's
# fo_baryon_lo numbers (389.2 +/- 5.9 against 384), since Coulomb scaling is exact.
#
# STILL OPEN: cost.  N_coord enters the gradient/Laplacian through triply nested N_coord_local
# loops, so Nc=4 is several times the per-walker cost of Nc=3 on top of twice the pairs -- a
# 200-walker, 270-step Nc=4 job runs >10 minutes here.  Size the campaign off a timing probe
# before launching 5000 walkers x 7 Q.
set -eu
echo "Design document: parameters are fixed (section 4) but the campaign is not sized yet."
echo "Next: timing probe at Nc=4, then launch Nc=3 (null test) and Nc=4 baryons at alpha=1."
exit 1
