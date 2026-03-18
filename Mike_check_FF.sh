#!/bin/bash
# Mike_check_FF: generate GFMC 3-point function data for Coulomb meson & baryon
# at weak coupling where the plateau method fails.
#
# Runs:
#   1) Meson  (N_coord=2): alpha=0.3, nf=0, Q=0.01, tau_max=500
#   2) Baryon (N_coord=3): alpha=0.3, nf=0, Q=0.01, tau_max=500
#
# The output h5 files contain per-walker C3 arrays suitable for
# Prony/spectral analysis.
#
# Expected physics (meson):
#   alpha_eff = CF * alpha = 4/3 * 0.3 = 0.4
#   E_0 = -alpha_eff^2 * mu/2 = -0.04  (mu=0.5 for equal-mass meson)
#   a0 = 1/(alpha_eff * mu) = 5.0
#   Delta_E (1s-2s) = 0.03
#   tau_plateau ~ 3/Delta_E = 100
#   We run to tau=500 (5x the plateau scale) and still see drift.
#
# Expected physics (baryon):
#   Coulomb with CF=4/3, 3 equal-mass quarks
#   Smaller energy gap, more excited state contamination
#
# Output h5 keys:
#   tau_values    : (N_tau,)           -- tau grid in MeV^-1
#   F_N, F_S, F_V0: (N_tau,)          -- form factor ratios vs tau
#   F_N_err, ...  : (N_tau,)          -- statistical errors
#   C3_N, C3_S, C3_V0: (N_walk,N_tau) -- per-walker C3 values (for bootstrap)
#   C2_2tau       : (N_tau,)          -- 2-point function normalization
#   Ws2_all       : (N_tau,N_walk)    -- per-walker leg-2 weights

set -e

OUTDIR="Mike_check_FF"
mkdir -p "$OUTDIR"

COMMON_ARGS="--n_step 1000 --n_walkers 1000 --dtau_iMev 0.5 --count_skip 10 \
--alpha 0.3 --nf 0 --OLO LO --spoila 1.0 --wavefunction compact --potential full \
--Nc 3 --Q_mag 0.01 --Q_dir 1 0 0 --outdir ${OUTDIR}/"

COMMON_ENV="JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

# ===================== MESON (N_coord=2) =====================
echo "============================================="
echo "Mike_check_FF: Run 1/2 — Meson (N_coord=2)"
echo "============================================="
echo "  alpha=0.3, nf=0, Q=0.01, dtau=0.5, tau_max=500"
echo "============================================="

env $COMMON_ENV python heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
    $COMMON_ARGS --N_coord 2

echo ""
echo "Meson done."
echo ""

# ===================== BARYON (N_coord=3) =====================
echo "============================================="
echo "Mike_check_FF: Run 2/2 — Baryon (N_coord=3)"
echo "============================================="
echo "  alpha=0.3, nf=0, Q=0.01, dtau=0.5, tau_max=500"
echo "============================================="

env $COMMON_ENV python heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
    $COMMON_ARGS --N_coord 3

echo ""
echo "============================================="
echo "Both runs done. Output in $OUTDIR/"
echo "============================================="
echo ""
echo "To inspect h5 files:"
echo "  python -c \"import h5py,glob; [print(f,sorted(h5py.File(f,'r').keys())) for f in glob.glob('${OUTDIR}/currents_3pt_*.h5')]\""
