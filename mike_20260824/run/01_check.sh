#!/bin/bash
# Coulomb meson, alpha=0.3. Closed form <r^2> = 27/(16 alpha^2) = 18.75.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
O="$OUT/coul_meson_a0.3"; mkdir -p "$O"
for Q in 0.0 0.05 0.1 0.2 0.35; do
  echo "Q=$Q"
  $PY heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
    --n_step 60 --n_walkers 500 --dtau_iMev 0.5 --count_skip 5 \
    --nf 0 --Nc 3 --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila 1 \
    --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --Q_mag $Q --seed 0 --outdir "$O/"
done
$PY prod_extract_lanczos.py "$O" --max_tau_pts 40
$PY "$ROOT/run/fit.py" "$O/prod_ME.csv" meson 18.75
