#!/bin/bash
# Step 5: Nc=3 Coulomb baryon at alpha = 3/4 (alpha C_F = 1), one Q per invocation so the Q
# values can be run in parallel.
#
#   usage:  ./run_baryon_a075.sh <Q>
#   all Q:  printf '%s\n' 0.0 0.1 0.2 0.35 0.5 0.7 1.0 1.3 | xargs -P 3 -n 1 ./run_baryon_a075.sh
#
# Reference (correlated-Gaussian SVM, pair coupling kappa = alpha C_F/(Nc-1) = 2 alpha/3 = 0.5):
#   E0 = -0.267838,  E1 = -0.143540 (gap 0.1243),  <r^2>_ch = 6.808
#   F(q)/F(0) = 0.9887 (q=0.1), 0.9559 (0.2), 0.8729 (0.35), 0.7624 (0.5)
#
# Parameter choice differs deliberately from run/03_baryon.sh. That script uses n_step=300 so the
# excited states decay enough for a *plateau* average. The Ritz rotators remove excited states
# algebraically instead, so a shorter window is sufficient -- and since the three-point code runs
# a full leg-2 evolution for C2 plus each of 7 currents at every stored time slice, the cost goes
# as N_tau * 8 * n_step:
#     03_baryon.sh : n_step=300, count_skip=10 -> N_tau=31 -> 74400   (measured: ~5 h per Q)
#     here         : n_step=120, count_skip=8  -> N_tau=16 -> 15360   (~4.8x cheaper)
# dtau is kept at Ben's scaling: dtau*|E0| = 0.048*0.268 = 0.0129, inside his ~0.014 guidance.
# tau_max = 5.76 per leg, so exp(-gap*tau) ~ 0.49 and the first excited state is still present in
# the window -- which Ritz wants, rather than needing it gone.
set -eu
cd "$(dirname "$0")"

Q=${1:?usage: $0 <Q_mag>}
PY=${PY:-/Users/mwagman/opt/anaconda3/bin/python3}
O="$PWD/${OUT:-atomic_units}/coul_baryon_a0.75"
NW=${NW:-500}
mkdir -p "$O"

export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd mike_20260824/code
echo "=== baryon alpha=0.75  Q=$Q  n_walkers=$NW ==="
exec $PY heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
  --n_step 120 --n_walkers $NW --dtau_iMev 0.048 --count_skip 8 \
  --nf 0 --Nc 3 --N_coord 3 --masses 1 1 1 --alpha 0.75 --spoila 1 \
  --wavefunction compact --potential full --color 1x1 \
  --Q_dir 1 0 0 --Q_mag "$Q" --seed 0 --outdir "$O/"
