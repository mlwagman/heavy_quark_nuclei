#!/bin/bash
# Decisive test of the 3pt Hankel indexing bug.
#
# Coulomb alpha=0.3 with a DELIBERATELY mismatched trial (spoila 0.7, so
# a0=3.5 against the true 5.0). The Hamiltonian is unchanged, so the exact
# answer is still <r^2> = 18.750 -- but the trial now has large excited-state
# overlap, so Z_i != 0 for i != 0 and the f != i interference terms exist.
#
# Those terms are what C3pt[s,t] = C3_N[s+t] gets wrong. With spoila=1 the
# Coulomb trial IS the ground state, Z_i = delta_i0, and the bug is inert --
# which is why the earlier test on that dataset showed no difference.
#
# Extract both ways and compare against 18.750.
set -u
cd /Users/user/bassi/heavy_quark_nuclei
ENV="JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
O=bugtest/spoila0.7; mkdir -p $O
for Q in 0.0 0.02 0.05 0.1 0.2 0.35 0.5; do
  ls "$O"/currents_3pt_*Qmag"$Q".h5 2>/dev/null | grep -qv korr && continue
  echo "### Q=$Q ($(date '+%H:%M'))"
  env $ENV taskpolicy -c utility nice -n 10 python3.13 \
    heavy_quark_nuclei_gfmc_boosted_FV_currents.py \
    --n_step 200 --n_walkers 2000 --dtau_iMev 0.5 --count_skip 5 \
    --nf 0 --Nc 3 --N_coord 2 --masses 1 -1 --alpha 0.3 --spoila 0.7 \
    --wavefunction compact --potential full --color 1x1 \
    --Q_dir 1 0 0 --Q_mag $Q --seed 0 --outdir "$O/" > "$O/run_Q$Q.log" 2>&1 \
    || echo "  !! FAILED Q=$Q"
done
echo "### extracting both ways ($(date '+%H:%M'))"
nice -n 10 python3.13 prod_extract_lanczos.py "$O" --max_tau_pts 40 \
     --estimator final --out_name prod_ME_BUGGY.csv > "$O/extract.log" 2>&1
nice -n 10 python3.13 prod_extract_fix.py     "$O" --max_tau_pts 40 \
     --estimator final --out_name prod_ME_FIXED.csv >> "$O/extract.log" 2>&1
rm -f "$O"/run_Q*.log "$O"/*.png
echo "BUGTEST_COMPLETE $(date '+%H:%M')"
