#!/bin/bash
# MRS LO meson, alpha_s=0.3, nf=0. Ref 26.89 from Numerov (06_crosschecks.sh).
# spoila=0.9241 sets the trial size to the Numerov ground state.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
O="$OUT/mrs_lo_meson_a0.3"; mkdir -p "$O"
for Q in $QS; do
  echo "Q=$Q"
  $PY heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py \
    --n_step 200 --n_walkers 500 --dtau_iMev 0.5 --count_skip 5 \
    --nf 0 --Nc 3 --N_coord 2 --alpha_s 0.3 --OLO LO --spoila 0.9241 \
    --mQ 1.0 --mu 2.0 --mufac 1.0 --Rstar 0.35 \
    --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --Q_mag $Q --seed 2 --outdir "$O/" > "$O/Q$Q.log" 2>&1
done
extract "$O" meson 26.89
