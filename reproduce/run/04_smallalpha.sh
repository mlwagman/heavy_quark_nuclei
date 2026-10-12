#!/bin/bash
# Coulomb is alpha-independent in units of a0 = 1.5/alpha. Scaling
# dtau by k^2 and Q by 1/k with k = 0.3/alpha makes the run at alpha the
# same calculation as the one at 0.3, so alpha^2 <r^2> = 27/16 = 1.6875
# at every alpha.
#
# This failed before 2026-08-24. Lanczos_twoleg.py added a fixed absolute
# 0.015 to E_shift while the physical rate scales as alpha^2, so below
# alpha ~ 0.05 the shifted correlator underflowed and Korrelator returned
# a singular matrix. It now scales with the measured rate.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
for AL in 0.3 0.001; do
  DT=$($PY -c "print(0.5*(0.3/$AL)**2)")
  O="$OUT/scal_a$AL"; mkdir -p "$O"
  echo "alpha=$AL dtau=$DT"
  for QR in 0.0 0.05 0.1 0.2 0.35; do
    Q=$($PY -c "print(f'{$QR*$AL/0.3:.10g}')")
    $PY heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
      --n_step 60 --n_walkers 500 --dtau_iMev $DT --count_skip 5 \
      --nf 0 --Nc 3 --N_coord 2 --masses 1 -1 --alpha $AL --spoila 1 \
      --wavefunction compact --potential full --color 1x1 \
      --Q_dir 1 0 0 --Q_mag $Q --seed 0 --outdir "$O/"
  done
  $PY prod_extract_lanczos.py "$O" --max_tau_pts 40
  $PY "$ROOT/run/fit.py" "$O/prod_ME.csv" meson $($PY -c "print(1.6875/$AL**2)")
done
