#!/bin/bash
# Fixed-order NLO baryon -- the one gap in the Nc=3, alpha_s(mQ)=0.1 table.  Partner of
# 18_fo_nlo_meson.sh (fo_meson_nlo) and of fo_baryon_lo.
#
# The fixed-order NLO pair potential is
#     V(r) = -c VB/r [ 1 + (alpha/4pi)(2 beta0 ln(r e^gamma mu) + a1) ],   VB = alpha CF/(Nc-1)
# with c = 1 for the 'OA' channel every pair takes inside the baryon singlet.  Note it does NOT
# scale with alpha the way Coulomb does -- the correction carries its own alpha -- so unlike the
# Nc scan this has to be solved and run at the coupling it is quoted at.
#
# REFERENCE, from ../../svm_nbody.py --potential fo_nlo (the kernel is closed form: the Gaussian
# average of the log is <ln sqrt(2 beta u)> = (1/2)ln(2 beta) - gamma/2, so no tabulation and no
# interpolation error):
#     E0 = -0.012228513   E1 = -0.0071711574   gap = 0.00505736   <r^2>_ch = 179.42049
#     F/F0 = 0.975254 0.905609 0.799237 0.679244 0.443745 0.261216
# Converged to < 2e-5 across ascale 0.6..0.8, nbasis 1200..2500 and two seeds.  The same solver
# at N=2 returns E0 = -0.010609125 and <r^2>_ch = 84.269834 against fo_two_body.py's
# -0.010609161 and 84.269031 -- two independent methods agreeing to 1e-5 on a potential with a
# log, which is the check that the NLO kernel is right.
#
# Run parameters matched to the LO baryon in the three dimensionless numbers that matter:
#
#              dtau   c_skip   n_step   dtau*|E0|   Delta*gap   tau*gap
#     LO        2.7     10       200      0.01286     0.05968    1.1936
#     NLO       1.05    11       225      0.01284     0.05841    1.1948
#
# spoila from the validated recipe, sqrt(target/c)/a0(spoila=1): c = 0.4181 +/- 0.0113 measured
# from the tau=0 Metropolis samples (consistent with the 0.4232 +/- 0.0051 of the Coulomb N=3
# trial), and a0(spoila=1) = 2/VB_NLO = 23.860644 -- note --log_mu_r is NOT inert here: it sets
# VB_NLO and hence the trial width, unlike in the MRS generator where it is dead.
#
# Q grid matched in q*sqrt(<r^2>) to the LO baryon's, so the two form factors overlay.  Q=0 is
# omitted: F(0)=N is an identity for the consistent-Gram GEVP, so extract with --f0_analytic.
set -eu
. "$(dirname "$0")/00_env.sh"

NW=${NW:-5000}
POOL=${POOL:-1}
NICE=${NICE:-10}
O="$OUT/refmass_a0.1"
LOGS="$O/logs"
FO="$CODE/heavy_quark_nuclei_gfmc_boosted_FV_currents.py"
NAME=fo_baryon_nlo
mkdir -p "$LOGS" "$O/$NAME"

COMMON="--nf 0 --Nc 3 --n_walkers $NW --no_color_vectors --wavefunction compact \
--potential full --color 1x1 --Q_dir 1 0 0 --seed 0 --mu 1.0 --log_mu_r 1"
ARGS="--N_coord 3 --masses 1 1 1 --alpha 0.1 --OLO NLO --spoila 0.868199 \
--n_step 225 --dtau_iMev 1.05 --count_skip 11"
QS=${QS:-"0.029 0.058 0.088 0.117 0.175 0.234"}

TASKS="$O/tasks_${NAME}.tsv"
WORKER="$O/worker_${NAME}.sh"
: > "$TASKS"
for q in $QS; do printf '%s\n' "$q" >> "$TASKS"; done
echo "$NAME: $(wc -l < "$TASKS" | tr -d ' ') jobs, $NW walkers, pool $POOL -> $O/$NAME"
echo "reference <r^2>_ch = 179.42049 (svm_nbody.py --potential fo_nlo)"

cat > "$WORKER" <<'WEOF'
#!/bin/bash
q="$1"
log="$LOGS/${NAME}_Q${q}.log"
if nice -n "${NICE:-10}" "$PY" "$FO" $COMMON $ARGS --Q_mag "$q" --outdir "$O/$NAME/" \
     > "$log" 2>&1; then
  echo "  ok    $NAME Q=$q"
else
  echo "  FAIL  $NAME Q=$q  (see $log)"
fi
WEOF
chmod +x "$WORKER"
export PY FO COMMON ARGS O NAME LOGS NICE
cd "$CODE"
xargs -P "$POOL" -n 1 -I{} "$WORKER" {} < "$TASKS"
echo "$NAME done"
