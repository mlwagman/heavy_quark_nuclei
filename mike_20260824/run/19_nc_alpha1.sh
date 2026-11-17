#!/bin/bash
# Nc scan: fixed-order Coulomb baryons at alpha = 1, N_coord = Nc.
#
# Coulomb scales exactly, so these runs rescale to any coupling: <r^2> -> <r^2>/alpha^2,
# E0 -> E0*alpha^2, F at q -> F at q*alpha.  See 17_nc_scan.sh for why alpha = 1 is the right
# place to work and where every number below comes from.
#
# Exact references from ../../svm_nbody.py (general-N correlated Gaussian), on these Q grids:
#   Nc=3  <r^2>_ch = 3.8260983   F/F0 = 0.974918 0.904503 0.801393 0.681613 0.447258 0.268560
#   Nc=4  <r^2>_ch = 2.7191553   F/F0 = 0.974332 0.906044 0.801557 0.679270 0.443208 0.258455
# The grids are matched in q*sqrt(<r^2>), which is why the two F/F0 rows nearly coincide --
# every Nc is probed at the same place on its own form factor.
#
# Nc=3 is a NULL TEST: rescaled by alpha=0.1 it must reproduce fo_baryon_lo, 389.2 +/- 5.9
# against 384, since Coulomb scaling is exact and that run used the same generator.
#
# nc2_baryon_s{1,2} are seed replicas of nc2_baryon.  The FO filename tag carries no seed, so a
# replica needs its own outdir, hence its own config name; the trailing --seed overrides the one
# in COMMON because argparse takes the last occurrence.  They exist because BOTH exact-trial
# two-body runs (fo_meson_lo and nc2_baryon) came in ~2.5% low on the radius at seed 0, and when
# the trial IS the eigenstate the Gram is numerically rank-1, so m cannot project the bias away.
# Two more seeds say whether that is a seed fluctuation or a systematic of the exact-trial case.
#
# Nc=2 is the free closed-form anchor at the OTHER end.  Inside the totally antisymmetric
# N = Nc singlet each pair carries kappa = alpha CF/(Nc-1) = (Nc+1)/(2Nc), which at Nc=2 is 3/4
# -- and the qqbar singlet coupling alpha CF is ALSO 3/4, so at Nc=2 the diquark and the meson
# are literally the same Coulomb problem (the 2/3 vs 4/3 split at Nc=3 closes at Nc=2).  So the
# N=2 baryon has a closed form, a0 = 2/kappa = 8/3 and <r^2>_ch = 3/kappa^2 = 16/3, and running
# it through the BARYON colour path (--masses 1 1 picks 'OA', not 'OSing') tests the colour
# bookkeeping against an exact answer.  Its trial is the exact 1s state at spoila = 1, so expect
# the Gram to go numerically rank-1 and m to cap early, exactly as fo_meson_lo did.
#
# Nc=5 runs WITHOUT an exact reference, by decision: svm_nbody.py does not converge at nj = 4
# (virial 1.006..1.096 across seeds) and building a proper one-at-a-time SVM is not worth it.
# Anchoring the method on Nc=3, where it reproduces 3.826 and passes the alpha-rescaling null
# test to 0.06 sigma, plus the m plateau and gfmc_walker_diag.py at every Nc, is the standard
# instead.  Quote the dipole and z k=4 fits together: their spread IS the ansatz systematic
# when no like-for-like target can be computed.
#
# Nc=6 parameters are EXTRAPOLATED, not measured, because there is no reference:
#   E0 / [(N(N-1)/2) kappa^2] = -0.03651 - 0.10685 N   (linear to 0.08% over N = 2..5)
#   gap/|E0|                  = +1.57505/N - 0.04529   (to ~4%)
#   1/(kappa sqrt(<r^2>_ch))  = +0.1765 + 0.1992 N     (to ~2%; see nc_scaling_curve.py)
# give E0 = -3.459, gap = 0.751, <r^2>_ch = 1.561 at Nc=6, hence dtau = 0.027|E0(3)|/|E0| and
# tau_max = 1.1934/gap.  spoila = sqrt(1.5614/c)/a0 with c = 0.1281 +/- 0.0022 measured from the
# tau = 0 Metropolis samples (a0 = 2/kappa = 3.428571); it is insensitive, a 3% error on the
# target radius moves spoila 1.5%.  F(0) = 6.0000000000 exactly on a smoke test.
#
# MEMORY, not CPU, is the constraint and it is a whole-machine property.  Peak RSS per job is
# 0.76 GB at N_coord=2 and 2.37 GB at N_coord=5, growing roughly as N^2 (the XLA graph, since
# the (Nt, Nt, n_walkers) arrays do not scale with N_coord), so expect ~3 GB at Nc=6 and ~4 GB
# at Nc=7.  Three concurrent Nc=5 jobs were reaped by the harness for OOM, and the survivors
# ran at 9.6% CPU EFFICIENCY -- 36 min of CPU in 6.3 h elapsed -- purely from paging, which is
# why the Nc=5 wall times came out 4-9 h per Q against a ~1 h CPU cost.  RUN ONE AT A TIME, and
# check free+inactive before launching: quitting a browser took this box from 5.9 to 7.8 GB
# reclaimable and drained swap from 6.7 to 2.7 GB, after which jobs ran at 100% efficiency.
#
# NW and QS are the two cost dials.  A 200-walker, 270-step Nc=4 job takes 14m27s here, so size
# the tranche before raising them: N_coord enters the gradient and Laplacian through triply
# nested loops, on top of Nc(Nc-1)/2 pairs.  --f0_analytic in the extractor means the Q=0 run is
# optional (F(0) = N is an identity for the consistent-Gram GEVP), so Q=0 is NOT in the default
# grid.
# RESULTS (12/12 jobs clean, 5000 walkers, m = 20 for Nc=3 and m = 16 for Nc=4 -- the rank is
# capped by the stored grid, n_step/count_skip):
#
#   config       fit      GFMC <r^2>_ch     target   fit-form bias   pull
#   nc3_baryon   dipole   3.895 +/- 0.064    3.820       -0.15%      +1.1s
#   nc3_baryon   z k=4    3.897 +/- 0.068    3.825       -0.03%      +1.0s
#   nc4_baryon   dipole   2.696 +/- 0.034    2.709       -0.37%      -0.4s
#   nc4_baryon   z k=4    2.692 +/- 0.033    2.718       -0.05%      -0.8s
#
# NULL TEST PASSED.  Nc=3 at alpha=1 gives 3.897 +/- 0.063; the independent alpha=0.1 campaign
# gave 389.2 +/- 5.9, which rescales to 3.892 +/- 0.059 -- 0.06 sigma apart, across 10x in alpha
# and 100x in dtau, with different seeds.  Coulomb scaling is confirmed end to end, so these runs
# really do stand in for every coupling.
#
# m plateau: Nc=3 drifts 3.920 (m=4) -> 3.895 (m=20), 0.6% over the whole range against a 1.6%
# error, monotone and flattening; Nc=4 2.707 -> 2.696, 0.4%.  Both fits agree with each other,
# so the residual +2% at Nc=3 (reproduced from the alpha=0.1 run) is not the ansatz.
#
# Walker state-ID diagnostics (../../gfmc_walker_diag.py, written into analysis/):
#   N_eff/N 0.90-0.97, top1 0.06-0.08% (flat 0.02%), top10 0.41-0.57% (flat 0.20%),
#   NO negative Ritz weights anywhere, |dJ(top1)| <= 0.09 sigma, and the |d_w| histogram spans
#   about one decade with no detached bin.  These are much healthier than the alpha=0.1 runs --
#   note the RAW w_ij leverage is still 22-24x here, so raw leverage overstates the problem and
#   the Ritz-projected weight is the thing to look at.
#
# Nc >= 5 needs no SVM: anchor on Nc=3 and carry the m plateau and these diagnostics forward.
set -eu
. "$(dirname "$0")/00_env.sh"

FO=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
NW=${NW:-5000}
POOL=${POOL:-2}
NICE=${NICE:-5}      # this box is shared with other sessions and other projects
O="$OUT/nc_alpha1"
LOGS="$O/logs"
mkdir -p "$LOGS"

COMMON="--nf 0 --n_walkers $NW --no_color_vectors --wavefunction compact --potential full \
--color 1x1 --Q_dir 1 0 0 --seed 0 --alpha 1 --n_skip 50"

# name|Nc|Q list|config-specific args
JOBS=$(cat <<'SPEC'
nc2_baryon|2|0.17 0.34 0.51 0.68 1.02 1.36|--spoila 1 --n_step 124 --dtau_iMev 0.091455 --count_skip 6
nc2_baryon_s1|2|0.17 0.34 0.51 0.68 1.02 1.36|--spoila 1 --n_step 124 --dtau_iMev 0.091455 --count_skip 6 --seed 1
nc2_baryon_s2|2|0.17 0.34 0.51 0.68 1.02 1.36|--spoila 1 --n_step 124 --dtau_iMev 0.091455 --count_skip 6 --seed 2
nc3_baryon|3|0.2 0.4 0.6 0.8 1.2 1.6|--spoila 1.002226 --n_step 200 --dtau_iMev 0.027 --count_skip 10
nc4_baryon|4|0.24 0.47 0.71 0.95 1.42 1.90|--spoila 0.996484 --n_step 270 --dtau_iMev 0.011824 --count_skip 14
nc5_baryon|5|0.27 0.55 0.83 1.11 1.66 2.21|--spoila 0.983718 --n_step 330 --dtau_iMev 0.006262 --count_skip 17
nc6_baryon|6|0.31 0.62 0.94 1.26 1.88 2.51|--spoila 1.018416 --n_step 430 --dtau_iMev 0.003719 --count_skip 22
SPEC
)

# Namespaced by ONLY, because two invocations running at once would otherwise both write and
# both read the SAME tasks file and the same worker, and each xargs would pick up whatever the
# other had just written.  That happened: an nc2 stream and an nc6 stream launched together
# both ended up running nc6 Q=0.31, duplicating 30 minutes of work and doubling the memory.
TASKS="$O/tasks${ONLY:+_$ONLY}.tsv"
WORKER="$O/worker${ONLY:+_$ONLY}.sh"
: > "$TASKS"
# ONLY=nc5_baryon runs one config without re-running the finished ones
while IFS='|' read -r name nc qs extra; do
  [ -z "$name" ] && continue
  if [ -n "${ONLY:-}" ] && [ "$name" != "$ONLY" ]; then continue; fi
  ms=$(for i in $(seq "$nc"); do printf '1 '; done)
  for q in ${QS:-$qs}; do
    printf '%s\t%s\t%s\n' "$name" "$q" \
      "$COMMON --Nc $nc --N_coord $nc --masses $ms $extra --Q_mag $q --outdir $O/$name/" \
      >> "$TASKS"
  done
done <<< "$JOBS"

n=$(wc -l < "$TASKS" | tr -d ' ')
echo "Nc scan at alpha = 1: $n jobs, $NW walkers, pool $POOL -> $O"
echo "References: Nc=3 <r^2>_ch = 3.8260983, Nc=4 <r^2>_ch = 2.7191553 (svm_nbody.py)"

# A worker script file rather than an inline bash -c: BSD xargs caps the -I replacement at
# -S bytes (255 by default), and "command line cannot be assembled, too long" is what it says
# when a task line plus an inline script exceeds that.  Same pattern as 10/12/13/15.
cat > "$WORKER" <<'WEOF'
#!/bin/bash
name=$(printf '%s' "$1" | cut -f1); q=$(printf '%s' "$1" | cut -f2)
cmd=$(printf '%s' "$1" | cut -f3)
mkdir -p "$OUT/nc_alpha1/$name"
cd "$CODE"
if nice -n "${NICE:-5}" "$PY" "$CODE/$FO" $cmd > "$OUT/nc_alpha1/logs/${name}_Q${q}.log" 2>&1
then echo "ok   $name Q=$q"
else echo "FAIL $name Q=$q  -- see logs/${name}_Q${q}.log"
fi
WEOF
chmod +x "$WORKER"
export PY CODE FO OUT NICE

# xargs rather than wait -n: bash 3.2 here.  -S raises the -I replacement buffer.
tr '\n' '\0' < "$TASKS" | xargs -0 -S 65536 -P "$POOL" -I{} "$WORKER" "{}"
echo "done; resume anything missing with: 14_resume.sh \"$TASKS\""
