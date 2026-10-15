#!/bin/bash
# MRS LO baryon, alpha_s=0.3. No independent 3-body solver for MRS.
# Ours with 1000 walkers: 55.8 +/- 1.3.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
O="$OUT/mrs_lo_baryon_a0.3"; mkdir -p "$O"
for Q in $QS; do
  echo "Q=$Q"
  $PY heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py \
    --n_step 300 --n_walkers 500 --dtau_iMev 0.3 --count_skip 10 \
    --nf 0 --Nc 3 --N_coord 3 --alpha_s 0.3 --OLO LO --spoila 0.8999 \
    --mQ 1.0 --mu 2.0 --mufac 1.0 --Rstar 0.35 \
    --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --Q_mag $Q --seed 3 --outdir "$O/" > "$O/Q$Q.log" 2>&1
done
extract "$O" baryon
