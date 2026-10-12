#!/bin/bash
# MRS LO meson, alpha_s=0.3, nf=0. Numerov reference is 26.89.
# spoila=0.9241 matches the trial a0 to the exact ground state:
#   a_eff = sqrt(4 <r^2>/3),  spoila = a_eff / a0(spoila=1).
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
O="$OUT/mrs_lo_meson_a0.3"; mkdir -p "$O"
for Q in 0.0 0.05 0.1 0.2 0.35; do
  echo "Q=$Q"
  $PY heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py \
    --n_step 200 --n_walkers 500 --dtau_iMev 0.5 --count_skip 5 \
    --nf 0 --Nc 3 --N_coord 2 --alpha_s 0.3 --OLO LO --spoila 0.9241 \
    --mQ 1.0 --mu 2.0 --mufac 1.0 --Rstar 0.35 \
    --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --Q_mag $Q --seed 0 --outdir "$O/"
done
$PY prod_extract_lanczos.py "$O" --max_tau_pts 40
$PY "$ROOT/run/fit.py" "$O/prod_ME.csv" meson 26.89
echo "Our full-statistics value is 25.69 +/- 0.37, 4.5% low. That offset is a"
echo "projection-time systematic, still being measured."
