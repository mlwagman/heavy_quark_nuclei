#!/bin/bash
# dtau scan for the Coulomb meson at alpha=0.3, where <r^2> = 18.75 and E_0 = -0.04 exactly.
#
#   usage:  ./run_dtau_scan.sh <config> <Q>
#   all:    for C in A B C D; do printf '%s\n' 0.0 0.1 0.2 0.35 \
#             | xargs -P 3 -n 1 ./run_dtau_scan.sh $C; done
#
# The point of the scan is NOT only the Trotter extrapolation; it is to find out how the
# extraction cutoffs should scale with dtau. The Rayleigh-Ritz pencil at iteration m is built from
# C(s+t) and C(s+t+1) for s,t < m, so it reaches two-point index 2m-1, i.e. PHYSICAL time
#
#     t_reach(m) = (2m - 1) * Delta,     Delta = count_skip * dtau
#
# which gives two competing readings:
#
#   * what limits m from above is physical: N_eff degrades as exp(-sigma^2) with
#     sigma^2 ~ 2 lambda_tau t and lambda_tau ~ 0.05 |E_0| (weight_tails.py), and the spectral
#     information available is set by the window length and the gap -- both functions of
#     t_reach, not of the number of grid points;
#   * but if dtau shrinks at fixed count_skip then Delta shrinks too, so the grid refines,
#     N_tau grows, and the extra Hankel rows are increasingly linearly dependent. m_max would
#     then move with dtau for purely numerical reasons, and a dtau -> 0 extrapolation would be
#     partly measuring its own extraction.
#
# Prediction: holding Delta fixed (count_skip ~ 1/dtau) puts every dtau on an identical
# extraction footing -- same N_tau, same t_reach(m), same N_eff(k) -- so the only thing varying
# is the Trotter error, which is what we want to extrapolate. Configs A/B/C test that; D is the
# deliberate counter-example with count_skip held fixed instead.
#
# All four hold tau_max = n_step * dtau = 30 (one leg), matching run/01_check.sh.
#
#   config  dtau    count_skip  n_step   Delta   N_tau   dtau*|E_0|   rel cost
#   A       0.5     5           60       2.50    13      0.020        1x
#   B       0.25    10          120      2.50    13      0.010        2x
#   C       0.125   20          240      2.50    13      0.005        4x
#   D       0.25    5           120      1.25    25      0.010        2x   <- Delta NOT held
set -eu
cd "$(dirname "$0")"

CFG=${1:?usage: $0 <A|B|C|D> <Q>}
Q=${2:?usage: $0 <A|B|C|D> <Q>}
PY=${PY:-/Users/mwagman/opt/anaconda3/bin/python3}
NW=${NW:-500}

case "$CFG" in
  A) DTAU=0.5;   SKIP=5;  NSTEP=60  ;;
  B) DTAU=0.25;  SKIP=10; NSTEP=120 ;;
  C) DTAU=0.125; SKIP=20; NSTEP=240 ;;
  D) DTAU=0.25;  SKIP=5;  NSTEP=120 ;;
  *) echo "config must be A, B, C or D"; exit 1 ;;
esac

O="$PWD/${OUT:-dtau_scan}/cfg${CFG}_dtau${DTAU}_skip${SKIP}"
mkdir -p "$O"

export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd mike_20260824/code
echo "=== cfg $CFG: dtau=$DTAU count_skip=$SKIP n_step=$NSTEP Q=$Q (Delta=$($PY -c "print($DTAU*$SKIP)")) ==="
exec $PY heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
  --n_step $NSTEP --n_walkers $NW --dtau_iMev $DTAU --count_skip $SKIP \
  --nf 0 --Nc 3 --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila 1 \
  --wavefunction compact --potential full --color 1x1 \
  --Q_dir 1 0 0 --Q_mag "$Q" --seed 0 --outdir "$O/"
