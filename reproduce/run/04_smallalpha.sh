#!/bin/bash
# Coulomb is alpha-independent in units of a0 = 1.5/alpha. Scaling dtau by
# k^2 and Q by 1/k, k = 0.3/alpha, makes the run at alpha the same calculation
# as the one at 0.3, so alpha^2 <r^2> = 27/16 at every alpha.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
SEED=4
for AL in 0.3 0.001; do
  DT=$($PY -c "print(0.5*(0.3/$AL)**2)")
  O="$OUT/scal_a$AL"; mkdir -p "$O"
  echo "alpha=$AL dtau=$DT"
  for QR in $QS; do
    Q=$($PY -c "print(f'{$QR*$AL/0.3:.10g}')")
    $PY heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
      --n_step 60 --n_walkers 500 --dtau_iMev $DT --count_skip 5 \
      --nf 0 --Nc 3 --N_coord 2 --masses 1 -1 --alpha $AL --spoila 1 \
      --wavefunction compact --potential full --color 1x1 \
      --Q_dir 1 0 0 --Q_mag $Q --seed $SEED --outdir "$O/" > "$O/Q$Q.log" 2>&1
  done
  extract "$O" meson $($PY -c "print(1.6875/$AL**2)")
  SEED=$((SEED+1))
done
