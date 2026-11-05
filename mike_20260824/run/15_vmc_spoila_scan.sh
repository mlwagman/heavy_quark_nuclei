#!/bin/bash
# VMC-style scan of the trial width: minimise <H> at tau = 0 over spoila, and compare the
# variational optimum against the radius-matched spoila the production runs use.
#
# WHY tau=0 IS A PURE VMC QUANTITY.  The generator stores Hammys[t, walker] = local energy
# E_L(R) = (H psi)(R)/psi(R) and Ws_leg1[t, walker] = the GFMC weight.  At t = 0 no propagation
# has happened: Ws_leg1[0] == 1 exactly (checked, min = max = 1+0j), so the plain walker mean of
# Hammys[0] is <psi|H|psi>/<psi|psi> for the Metropolis-sampled |psi|^2.  Validated against the
# one config with an exact answer: the FO meson at spoila = 1 IS the exact 1s state, and there
# Hammys[0] has std = 1.28e-18 over 5000 walkers (relative 2.9e-16) with mean -0.0044444444444,
# i.e. a zero-variance local energy -- the signature of an exact eigenstate.  That config is kept
# in the scan below as a control: its minimum must land at spoila = 1.000.
#
# COST.  Metropolis dominates: n_therm + n_skip*n_walkers = 500*n_skip + n_skip*n_walkers steps at
# ~1000 it/s, so the stock --n_skip 100 costs 150k steps (~150 s) for a single point.  --n_skip 10
# cuts that 10x.  The price is correlated walkers, which inflates nothing but the error bar -- and
# for LOCATING A MINIMUM that is the right trade, especially with --seed held fixed across the grid
# so the <H>(spoila) curve is correlated point to point and far smoother than its absolute errors.
# --n_step 1 because only the t = 0 slice is read.
set -eu
. "$(dirname "$0")/00_env.sh"
cd "$CODE"

POOL=${POOL:-2}
NW=${NW:-2000}
NSKIP=${NSKIP:-10}
SEED=${SEED:-7}
MU=0.0362001
S="$OUT/refmass_a0.1/vmc_scan"
mkdir -p "$S"

FO=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
MRS=heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py
BASE="--nf 0 --Nc 3 --no_color_vectors --wavefunction compact --potential full \
--color 1x1 --Q_dir 1 0 0 --Q_mag 0.0"

# name|script|spoila_production|config args|extra args
JOBS=$(cat <<'SPEC'
fo_meson_lo|FO|1.0|--N_coord 2 --masses 1 -1 --alpha 0.1 --dtau_iMev 4.5|
fo_baryon_lo|FO|1.01131|--N_coord 3 --masses 1 1 1 --alpha 0.1 --dtau_iMev 2.7|
mrs_meson_lo|MRS|0.61117|--N_coord 2 --alpha_s 0.1 --OLO LO --dtau_iMev 0.632466|--mQ 1.0 --mu MUV --mufac 1.0 --Rstar 0.35
mrs_meson_nlo|MRS|0.582644|--N_coord 2 --alpha_s 0.1 --OLO NLO --dtau_iMev 0.580922|--mQ 1.0 --mu MUV --mufac 1.0 --Rstar 0.35
mrs_baryon_lo|MRS|0.5817|--N_coord 3 --alpha_s 0.1 --OLO LO --dtau_iMev 0.632466|--mQ 1.0 --mu MUV --mufac 1.0 --Rstar 0.35
mrs_baryon_nlo|MRS|0.5497|--N_coord 3 --alpha_s 0.1 --OLO NLO --dtau_iMev 0.580922|--mQ 1.0 --mu MUV --mufac 1.0 --Rstar 0.35
SPEC
)

# ratios of the production spoila to probe
RATIOS=${RATIOS:-"0.70 0.80 0.90 0.95 1.00 1.05 1.10 1.20 1.35 1.50"}

TASKS="$S/tasks.tsv"
: > "$TASKS"
while IFS='|' read -r name which sp0 cargs extra; do
  [ -z "$name" ] && continue
  case "$which" in FO) script=$FO ;; MRS) script=$MRS ;; esac
  extra=${extra//MUV/$MU}
  mkdir -p "$S/$name"
  for r in $RATIOS; do
    sp=$(awk -v a="$sp0" -v b="$r" 'BEGIN{printf "%.6f", a*b}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$script" "$sp" "$cargs" "$extra" >> "$TASKS"
  done
done <<< "$JOBS"

cat > "$S/worker.sh" <<'WORKER'
#!/bin/bash
IFS=$'\t' read -r name script sp cargs extra <<< "$1"
log="$S/$name/vmc_spoila${sp}.log"
if "$PY" "$script" $BASE $cargs $extra --spoila "$sp" --n_step 1 --count_skip 1 \
      --n_skip "$NSKIP" --n_walkers "$NW" --seed "$SEED" --outdir "$S/$name/" \
      > "$log" 2>&1; then
  echo "  done  $name spoila=$sp"
else
  echo "  FAIL  $name spoila=$sp  (see $log)"
fi
WORKER
chmod +x "$S/worker.sh"
export BASE S PY NSKIP NW SEED

n=$(wc -l < "$TASKS" | tr -d ' ')
echo "VMC spoila scan: $n points, pool $POOL, $NW walkers, n_skip $NSKIP, seed $SEED"
tr '\n' '\0' < "$TASKS" | xargs -0 -P "$POOL" -I{} "$S/worker.sh" "{}"
echo "scan complete: all $n points finished"
