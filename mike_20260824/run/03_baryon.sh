#!/bin/bash
# MRS LO baryon, alpha_s=0.3. No exact 3-body solver for the MRS potential,
# so GFMC is the only route. Our full-statistics value is 59.74 +/- 1.10.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
O="$OUT/mrs_lo_baryon_a0.3"; mkdir -p "$O"
for Q in 0.0 0.05 0.1 0.2 0.35; do
  echo "Q=$Q"
  $PY heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py \
    --n_step 300 --n_walkers 500 --dtau_iMev 0.3 --count_skip 10 \
    --nf 0 --Nc 3 --N_coord 3 --alpha_s 0.3 --OLO LO --spoila 0.9250 \
    --mQ 1.0 --mu 2.0 --mufac 1.0 --Rstar 0.35 \
    --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --Q_mag $Q --seed 0 --outdir "$O/"
done
$PY prod_extract_lanczos.py "$O" --max_tau_pts 40
$PY "$ROOT/run/fit.py" "$O/prod_ME.csv" baryon 59.74
