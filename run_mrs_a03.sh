#!/bin/bash
# MRS LO meson at alpha_s = 0.3, one Q per invocation so Q values can run in parallel.
#
#   usage:  ./run_mrs_a03.sh <Q>
#   all Q:  printf '%s\n' 0.0 0.05 0.1 0.15 0.2 0.3 0.35 0.5 | xargs -P 3 -n 1 ./run_mrs_a03.sh
#
# Same GFMC parameters as run/02_mrs_meson.sh, so this is a controlled comparison against its
# quoted numbers -- the Numerov reference 26.89 and Ben's full-statistics 25.69 +/- 0.37, which
# his README attributes to "a projection-time systematic, still being measured".  The only
# differences are (a) the generator now also stores the two-time C3_N_offdiag / Ws2_offdiag, and
# (b) three extra Q points, since a free-p dipole or a k_max=3 z-expansion needs more than the
# stock five.
#
# spoila=0.9241 matches the trial a0 to the exact ground state: a_eff = sqrt(4 <r^2>/3),
# spoila = a_eff / a0(spoila=1).
set -eu
cd "$(dirname "$0")"

Q=${1:?usage: $0 <Q_mag>}
PY=${PY:-/Users/mwagman/opt/anaconda3/bin/python3}
O="$PWD/${OUT:-mrs_korr}/mrs_lo_meson_a0.3"
NW=${NW:-500}
mkdir -p "$O"

export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd mike_20260824/code
echo "=== MRS LO meson alpha_s=0.3  Q=$Q  n_walkers=$NW ==="
exec $PY heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py \
  --n_step 200 --n_walkers $NW --dtau_iMev 0.5 --count_skip 5 \
  --nf 0 --Nc 3 --N_coord 2 --alpha_s 0.3 --OLO LO --spoila 0.9241 \
  --mQ 1.0 --mu 2.0 --mufac 1.0 --Rstar 0.35 \
  --wavefunction compact --potential full --color 1x1 \
  --Q_dir 1 0 0 --Q_mag "$Q" --seed 0 --outdir "$O/"
