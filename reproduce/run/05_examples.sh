#!/bin/bash
# Menu of the options. Run individual blocks, not the whole file.
# 200 walkers, so fast and noisy. Every block uses the same seed, so
# differences between blocks are less noisy than either block alone.
# Never combine runs that share a seed as independent measurements.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
O="$OUT/examples"; mkdir -p "$O"
CUR=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
MRS=heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py
B="--nf 0 --Nc 3 --wavefunction compact --potential full --color 1x1 --Q_dir 1 0 0 --seed 1"
M="--mQ 1.0 --mu 2.0 --mufac 1.0 --Rstar 0.35"
NBINS=50

# run <name> <meson|baryon> <ref or ""> <script and args>
run () {
  local d="$O/$1" sys=$2 ref=$3; shift 3; mkdir -p "$d"; echo; echo "$d"
  for Q in $QS; do "$PY" "$@" $B --Q_mag $Q --outdir "$d/" > "$d/Q$Q.log" 2>&1; done
  extract "$d" $sys $ref
}

# meson vs baryon
run meson meson 18.75 $CUR --n_step 60 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
  --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila 1
run baryon baryon 42.813 $CUR --n_step 200 --n_walkers 200 --dtau_iMev 0.3 --count_skip 10 \
  --N_coord 3 --masses 1 1 1 --alpha 0.3 --spoila 1

# MRS takes --alpha_s, not --alpha
run mrs_lo meson 26.89 $MRS --n_step 200 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
  --N_coord 2 --alpha_s 0.3 --OLO LO --spoila 0.9241 $M
run mrs_nlo meson 28.22 $MRS --n_step 200 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
  --N_coord 2 --alpha_s 0.3 --OLO NLO --spoila 0.8914 $M

# scale variation is --mufac (lambda). --mu is mu_cut, leave it fixed.
for MF in 0.5 2.0; do
  run lambda$MF meson "" $MRS --n_step 200 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
    --N_coord 2 --alpha_s 0.3 --OLO LO --spoila 0.9241 --mQ 1.0 --mu 2.0 --mufac $MF --Rstar 0.35
done

# trial size. The answer should not depend on this.
for SP in 0.8 1.25; do
  run spoila$SP meson 18.75 $CUR --n_step 60 --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
    --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila $SP
done
# variational tuner for spoila, no reference needed:
#   $PY optimize_spoila.py --alpha_s 0.3 --OLO LO --N_coord 3

# projection time, tau = dtau*n_step per leg
for NS in 30 120; do
  run nstep$NS meson 18.75 $CUR --n_step $NS --n_walkers 200 --dtau_iMev 0.5 --count_skip 5 \
    --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila 1
done

# time step: half of the "meson" block, same tau range and spacing
run dtau0.25 meson 18.75 $CUR --n_step 120 --n_walkers 200 --dtau_iMev 0.25 --count_skip 10 \
  --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila 1

# Q=0 is the charge check: F(0) printed by extract must equal N_coord.
# The h5 files hold everything per walker; gfmc_reduce.py keeps the two-time
# objects that gfmc_korr.py needs (C2(t_i+t_j) and C_J(t_i,t_j)).
