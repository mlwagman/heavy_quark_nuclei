#!/bin/bash
# Steps 4 and 5: Coulomb meson and Nc=3 baryon in "atomic units", alpha = 3/4 so alpha*C_F = 1.
#
# Closed-form targets (meson):
#   a0 = 1/(mu alpha C_F) = 1.5/alpha = 2      (mu = 1/2 for masses 1,-1;  alpha C_F = 1)
#   F(0) = N_coord = 2
#   F(q)/F(0) = (1 + q^2 a0^2/16)^-2           exact dipole, p = 2
#   <r^2>_ch = 0.75 a0^2 = 1.6875/alpha^2 = 3.000   (one-body, about the CM)
# Baryon reference: svm_baryon.py --alpha 0.75 (correlated-Gaussian SVM, pair coupling
#   kappa = alpha C_F/(Nc-1) = 2 alpha/3 = 0.5), which prints <r^2>_ch and F(q)/F(0).
#
# Scaling rule from the bundle README: dtau ~ 1/alpha^2 and Q ~ alpha, so that a run at a
# different alpha is the SAME calculation in units of a0 and 1/|E|. Relative to the alpha=0.3
# reference (dtau=0.5): dtau = 0.5*(0.3/0.75)^2 = 0.08, and Q is 2.5x the reference grid.
# The Q grid is deliberately denser and longer than the stock five points: the goal is to
# *confirm* the dipole fall-off over a real dynamic range (Q a0 up to 3.5, F/F(0) down to ~0.32),
# not just to measure an initial slope.
set -eu
cd "$(dirname "$0")"

PY=${PY:-/Users/mwagman/opt/anaconda3/bin/python3}     # needs jax
CODE=mike_20260824/code
OUT=${OUT:-atomic_units}
NW=${NW:-500}
SYS=${1:-meson}

export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

QS="0.0 0.125 0.25 0.375 0.5 0.625 0.875 1.25 1.75"

case "$SYS" in
  meson)
    O="$PWD/$OUT/coul_meson_a0.75"; mkdir -p "$O"
    NCOORD=2; MASSES="1 -1"; NSTEP=60; DTAU=0.08; SKIP=5
    ;;
  baryon)
    # Follow 03_baryon.sh's shape: smaller dtau, more steps, coarser storage grid.
    # dtau 0.3 -> 0.3*(0.3/0.75)^2 = 0.048
    O="$PWD/$OUT/coul_baryon_a0.75"; mkdir -p "$O"
    NCOORD=3; MASSES="1 1 1"; NSTEP=300; DTAU=0.048; SKIP=10
    ;;
  *) echo "usage: $0 [meson|baryon]"; exit 1 ;;
esac

echo "system=$SYS  N_coord=$NCOORD  masses=$MASSES  n_step=$NSTEP  dtau=$DTAU  count_skip=$SKIP"
echo "n_walkers=$NW  outdir=$O"
echo "Q grid: $QS"

cd "$CODE"
for Q in $QS; do
  echo "=== Q=$Q ==="
  $PY heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
    --n_step $NSTEP --n_walkers $NW --dtau_iMev $DTAU --count_skip $SKIP \
    --nf 0 --Nc 3 --N_coord $NCOORD --masses $MASSES --alpha 0.75 --spoila 1 \
    --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --Q_mag $Q --seed 0 --outdir "$O/"
done
echo "GFMC done: $O"
