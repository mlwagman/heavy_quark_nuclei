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
set -eu
echo "This is a design document, not a runnable campaign yet -- see section 3."
echo "Blocking items: trajectory choice, spoila/dtau retuning, and a general-N reference solver."
exit 1
