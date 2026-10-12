#!/bin/bash
# Menu of the options. Run individual blocks, not the whole file.
# 200 walkers here, so fast and noisy.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
O="$OUT/examples"; mkdir -p "$O"
CUR=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
MRS=heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py
B="--nf 0 --Nc 3 --spoila 1 --wavefunction compact --potential full --color 1x1 --Q_dir 1 0 0 --seed 0"
M="--mQ 1.0 --mu 2.0 --mufac 1.0 --Rstar 0.35"
Q="--Q_mag 0.1"
run () { echo; echo "$1"; shift; "$@"; }

# meson vs baryon
run "meson" \
  $PY $CUR --n_step 60 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
  --N_coord 2 --masses 1 -1 --alpha 0.3 $B $Q --outdir "$O/meson/"
run "baryon" \
  $PY $CUR --n_step 200 --n_walkers 200 --dtau_iMev 0.3 --count_skip 10 \
  --N_coord 3 --masses 1 1 1 --alpha 0.3 $B $Q --outdir "$O/baryon/"

# MRS takes --alpha_s, not --alpha
run "MRS LO" $PY $MRS --n_step 200 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
  --N_coord 2 --alpha_s 0.3 --OLO LO $M $B $Q --outdir "$O/mrs_lo/"
run "MRS NLO" $PY $MRS --n_step 200 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
  --N_coord 2 --alpha_s 0.3 --OLO NLO $M $B $Q --outdir "$O/mrs_nlo/"

# scale variation is --mufac (lambda). --mu is mu_cut, leave it fixed.
for MF in 0.5 1.0 2.0; do
  run "lambda=$MF" $PY $MRS --n_step 200 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
    --N_coord 2 --alpha_s 0.3 --OLO LO --mQ 1.0 --mu 2.0 --mufac $MF --Rstar 0.35 \
    $B $Q --outdir "$O/lambda$MF/"
done

# Q=0 is the charge check, F_N must equal N_coord
for QQ in 0.0 0.05 0.1 0.2 0.35; do
  run "Q=$QQ" $PY $CUR --n_step 60 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
    --N_coord 2 --masses 1 -1 --alpha 0.3 $B --Q_mag $QQ --outdir "$O/Q$QQ/"
done

# trial size. The answer should not depend on this.
for SP in 0.7 1.0 1.4; do
  run "spoila=$SP" $PY $CUR --n_step 60 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
    --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila $SP \
    --nf 0 --Nc 3 --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --seed 0 $Q --outdir "$O/spoila$SP/"
done
# variational tuner, no exact answer needed:
#   $PY optimize_spoila.py --alpha_s 0.3 --OLO LO --N_coord 3

# projection time, tau = dtau*n_step per leg
for NS in 60 200 400; do
  run "n_step=$NS" $PY $CUR --n_step $NS --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
    --N_coord 2 --masses 1 -1 --alpha 0.3 $B $Q --outdir "$O/nstep$NS/"
done

# discretization, keep dtau*|E0| around 0.014
for DT in 0.25 0.5 1.0; do
  run "dtau=$DT" $PY $CUR --n_step 60 --n_walkers 200 --dtau_iMev $DT --count_skip 5 \
    --N_coord 2 --masses 1 -1 --alpha 0.3 $B $Q --outdir "$O/dtau$DT/"
done

# changing alpha: dtau and Q have to be scaled with it or the runs are not
# comparable. a0 ~ 1/alpha, so dtau -> dtau*(a_ref/a)^2 and Q -> Q*(a/a_ref).
for AL in 0.3 0.1; do
  DT=$($PY -c "print(0.5*(0.3/$AL)**2)"); QQ=$($PY -c "print(0.1*$AL/0.3)")
  run "alpha=$AL dtau=$DT Q=$QQ" \
    $PY $CUR --n_step 60 --n_walkers 200 --dtau_iMev $DT --count_skip 5 \
    --N_coord 2 --masses 1 -1 --alpha $AL $B --Q_mag $QQ --outdir "$O/a$AL/"
done

# extraction, Ritz rank = max_tau_pts/2
#   $PY prod_extract_lanczos.py <dir> --max_tau_pts 40
#   $PY prod_extract_lanczos.py <dir> --max_tau_pts 10 --out_name prod_ME_rank5.csv
#   $PY prod_extract.py <dir> --out_name prod_ME_plateau.csv    plateau, cross-check
#   $PY "$ROOT/run/fit.py" <dir>/prod_ME.csv meson|baryon [expected]
