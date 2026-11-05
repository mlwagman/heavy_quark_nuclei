#!/bin/bash
# Tranche 3 of the alpha_s(mQ)=0.1 campaign: the FO baryon at a trial width matched to the
# CORRECTED reference, plus the independent-seed replica test for the error bars.
#
# Why the FO baryon is being run a third time.
#   run 1  spoila 0.713489  trial <r^2>_ch = 191   (meson relation on a 3-body trial)
#   run 2  spoila 0.9566    trial <r^2>_ch = 345   (3-body c = 0.4172, but target was wrong)
#   run 3  spoila 1.01131   trial <r^2>_ch = 384   (target corrected)
# `svm_baryon.py --alpha 0.1` is NOT converged: for pure Coulomb with pair coupling
# kappa = 2 alpha/3 both E0/alpha^2 and alpha^2 <r^2> are strictly alpha-independent, and
#     alpha      0.5        0.3        0.1
#     E0/a^2   -0.47624   -0.47597   -0.47300
#     a^2<r^2>   3.8275     3.8529     3.4362
# so its radius is 11% low at alpha=0.1 (and E0 0.7% high) while alpha=0.5 and 0.3 agree to
# 0.66%.  The correct target is alpha^2<r^2> = 3.84, i.e. <r^2>_ch = 384 at alpha = 0.1.
# Our own GFMC <H> confirms the scaling independently: it plateaus at -0.004761(5) against the
# scaling value -0.0047624, agreeing to 0.03%, so the generator is right and SVM is the outlier.
# With spoila 0.9566 the GEVP radius was still climbing at m=16 (349.1 -> 362.9 over m=2..16,
# +0.35% per rank, -5.5% from 384) -- under-converged, not biased, because the trial was 10%
# compact.  a0 = sqrt(384/0.4172) = 30.339, so spoila = 30.339/30 = 1.01131.
#
# The replica test answers whether the within-run block bootstrap understates the error.  The
# block-size scan is already saturated (+/-1.14, 0.96, 1.02, 1.12, 0.99, 1.02 for 500 down to 10
# blocks, i.e. no trend), which rules out short-range walker correlation -- but a within-run
# resample is blind to anything global: one Metropolis chain seeds all the walkers, and the weight
# normalisation and E_shift fit are shared.  Independent runs see those.  8 seeds x {Q=0, Q=0.1}
# at 1000 walkers costs about 1.6 full runs; the output is scatter/error, which multiplies every
# error bar quoted so far.
#
# NOTE the first attempt at this test set OMP/MKL/OPENBLAS threads but not XLA_FLAGS, so each job
# took a full XLA thread pool and load hit 33 on 6 cores with zero completions.  00_env.sh sets
# XLA_FLAGS, so sourcing it (as here) is what makes the pool size mean anything.
set -eu
. "$(dirname "$0")/00_env.sh"
cd "$CODE"

POOL=${POOL:-5}
MU=0.0362001
O="$OUT/refmass_a0.1"
LOGS="$O/logs"
R="$O/replica_mrs_meson_lo"
mkdir -p "$LOGS" "$R"

FO=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
MRS=heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py
BASE="--nf 0 --Nc 3 --no_color_vectors --wavefunction compact --potential full \
--color 1x1 --Q_dir 1 0 0"

TASKS="$O/tasks3.tsv"
: > "$TASKS"

# 1. FO baryon, corrected trial width, fresh seed
mkdir -p "$O/fo_baryon_lo_sp1.011"
for q in 0.0 0.02 0.04 0.06 0.08 0.12 0.16; do
  printf 'fo_baryon_lo_sp1.011\t%s\t%s\t21\t5000\t%s\t\n' "$FO" "$q" \
    "--N_coord 3 --masses 1 1 1 --alpha 0.1 --spoila 1.01131 --n_step 200 --dtau_iMev 2.7 --count_skip 10" >> "$TASKS"
done

# 2. replica test: 8 independent seeds, two Q, 1000 walkers each
for seed in 101 102 103 104 105 106 107 108; do
  for q in 0.0 0.1; do
    mkdir -p "$R/s$seed"
    printf 'replica_mrs_meson_lo/s%s\t%s\t%s\t%s\t1000\t%s\t%s\n' "$seed" "$MRS" "$q" "$seed" \
      "--N_coord 2 --alpha_s 0.1 --OLO LO --spoila 0.611170 --n_step 200 --dtau_iMev 0.632466 --count_skip 10" \
      "--mQ 1.0 --mu $MU --mufac 1.0 --Rstar 0.35" >> "$TASKS"
  done
done

cat > "$O/worker3.sh" <<'WORKER'
#!/bin/bash
IFS=$'\t' read -r name script q seed nw cargs extra <<< "$1"
log="$LOGS/$(echo "$name" | tr / _)_Q${q}.log"
if "$PY" "$script" $BASE $cargs $extra --n_walkers "$nw" --seed "$seed" --Q_mag "$q" \
      --outdir "$O/$name/" > "$log" 2>&1; then
  echo "  done  $name Q=$q"
else
  echo "  FAIL  $name Q=$q  (see $log)"
fi
WORKER
chmod +x "$O/worker3.sh"
export BASE LOGS O PY

n=$(wc -l < "$TASKS" | tr -d ' ')
echo "tranche 3: $n runs (7 FO baryon at spoila 1.01131, 16 replica), pool $POOL"
tr '\n' '\0' < "$TASKS" | xargs -0 -P "$POOL" -I{} "$O/worker3.sh" "{}"
echo "all $n runs finished"
