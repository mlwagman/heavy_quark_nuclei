#!/bin/bash
# Second tranche of the alpha_s(mQ)=0.1 reference-mass campaign, fixing two mistakes in
# run/10_refmass_a0.1.sh.
#
# 1. BARYON spoila was wrong. I set it from the MESON relation
#        <r^2>_ch = (3/4) a0^2   ->   spoila = sqrt(4 <r^2>_ch/3) / a0(spoila=1),
#    which is the 1s two-body result and does not hold for the compact three-body trial
#    prod_{i<j} exp(-r_ij/a0).  That trial scales purely, <r^2>_ch = c a0^2, and c is measured
#    from the first FO-baryon run (a0 = 21.4047 gave <r^2>_ch = 191.14 +/- 2.45):
#        c = 0.4172 +/- 0.0054        (not 0.75)
#    so spoila = sqrt(target/c) / a0(spoila=1).  The FO baryon was therefore 1.34x too small,
#    putting the trial 44% below the SVM radius (191 against 343.62) -- which is why its GEVP
#    radius climbed 197.7 -> 236.6 over m = 2..12 with no plateau instead of converging.
#    The MRS baryons came out nearly right by accident (0.95x) because using the meson radius
#    with the meson relation on the baryon's a0 partly cancels; they are corrected anyway.
#
#         config            spoila was   now
#         FO  baryon LO     0.713489     0.9566
#         MRS baryon LO     0.611170     0.5817
#         MRS baryon NLO    0.582644     0.5497
#
# 2. ONE SEED FOR EVERYTHING.  Tranche 1 used --seed 0 for all 42 runs so the F(Q)/F(0) ratios
#    stay correlated, but that also means a single bad weight draw contaminates every config, and
#    seed 0 is a bad draw: worst walker leverage 3607x with 21 Ws2 cells above 5% error, against
#    147x and zero such cells at seed 1.  The seed is now fixed per config (still constant across
#    Q within a config, which is what the correlated ratio needs).
#
# Not changed: dtau.  Cutting it 3x removes the weight tails outright (same seed 0, dtau 4.5 ->
# 1.5 takes the worst leverage 3607x -> 12x, because the propagator is the truncated
# 1 - (dtau/2)V + (dtau^2/8)V^2 and a walker at small r gets a wildly wrong weight), but it costs
# 3x the steps, and coarse dtau + spike screening is already validated against the exact answer:
# <r^2> = 164.2 +/- 3.1 against 168.75, matching the ensemble's own radius to 0.3%.  Finer dtau is
# the recipe for the next campaign, not a reason to redo this one.
#
# fo_meson_lo from tranche 1 is kept as is (7/7, validated).  mrs_meson_lo keeps seed 0 for its
# three remaining Q so it matches the four already done.
set -eu
. "$(dirname "$0")/00_env.sh"
cd "$CODE"

POOL=${POOL:-5}
NW=${NW:-5000}
MU=0.0362001
O="$OUT/refmass_a0.1"
LOGS="$O/logs"
mkdir -p "$LOGS"

FO=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
MRS=heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py

COMMON="--nf 0 --Nc 3 --n_walkers $NW --no_color_vectors --wavefunction compact \
--potential full --color 1x1 --Q_dir 1 0 0"

# name|FO/MRS|seed|Q list|config args
JOBS=$(cat <<'SPEC'
mrs_meson_lo|MRS|0|0.2 0.3 0.4|--N_coord 2 --alpha_s 0.1 --OLO LO --spoila 0.611170 --n_step 200 --dtau_iMev 0.632466 --count_skip 10
fo_baryon_lo|FO|11|0.0 0.02 0.04 0.06 0.08 0.12 0.16|--N_coord 3 --masses 1 1 1 --alpha 0.1 --spoila 0.9566 --n_step 200 --dtau_iMev 2.7 --count_skip 10
mrs_meson_nlo|MRS|13|0.0 0.05 0.1 0.15 0.2 0.3 0.4|--N_coord 2 --alpha_s 0.1 --OLO NLO --spoila 0.582644 --n_step 200 --dtau_iMev 0.580922 --count_skip 10
mrs_baryon_lo|MRS|14|0.0 0.04 0.08 0.12 0.16 0.22 0.28|--N_coord 3 --alpha_s 0.1 --OLO LO --spoila 0.5817 --n_step 200 --dtau_iMev 0.632466 --count_skip 10
mrs_baryon_nlo|MRS|15|0.0 0.04 0.08 0.12 0.16 0.22 0.28|--N_coord 3 --alpha_s 0.1 --OLO NLO --spoila 0.5497 --n_step 200 --dtau_iMev 0.580922 --count_skip 10
SPEC
)

TASKS="$O/tasks2.tsv"
: > "$TASKS"
n=0
while IFS='|' read -r name which seed qs cargs; do
  [ -z "$name" ] && continue
  case "$which" in FO) script=$FO ;; MRS) script=$MRS ;; esac
  extra=""
  [ "$which" = "MRS" ] && extra="--mQ 1.0 --mu $MU --mufac 1.0 --Rstar 0.35"
  mkdir -p "$O/$name"
  for q in $qs; do
    n=$((n+1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$script" "$q" "$seed" "$cargs" "$extra" >> "$TASKS"
  done
done <<< "$JOBS"

cat > "$O/worker2.sh" <<'WORKER'
#!/bin/bash
IFS=$'\t' read -r name script q seed cargs extra <<< "$1"
log="$LOGS/${name}_Q${q}.log"
if "$PY" "$script" $COMMON $cargs $extra --seed "$seed" --Q_mag "$q" \
      --outdir "$O/$name/" > "$log" 2>&1; then
  echo "  done  $name Q=$q"
else
  echo "  FAIL  $name Q=$q  (see $log)"
fi
WORKER
chmod +x "$O/worker2.sh"
export COMMON LOGS O PY

echo "tranche 2: corrected baryon spoila, per-config seeds; $n runs, pool $POOL, $NW walkers"
tr '\n' '\0' < "$TASKS" | xargs -0 -P "$POOL" -I{} "$O/worker2.sh" "{}"
echo "all $n runs finished"
