#!/bin/bash
# Fixed-order NLO quarkonium, the partner of fo_meson_lo, for the korr charge-radius tutorial.
#
# The LO run is the anchor: at spoila = 1 its trial state IS the exact 1s eigenstate
# (a0 = 1.5/alpha = 15 exactly), so every target is closed form -- E0 = -4/9 alpha^2,
# <r^2>_ch = 1.6875/alpha^2 = 168.75, and F(q)/F(0) = (1 + q^2 a0^2/16)^-2 an EXACT dipole.
# ../../fo_two_body.py reproduces all of it to six digits, which validates the solver before it
# is trusted at NLO.
#
# NLO has no closed form.  fo_two_body.py --OLO NLO --alpha 0.1 (mu = 1, log_mu_r = 1, the
# generator's defaults, which the LO run also used) gives
#
#     E0 = -0.010609161     E1 = -0.003333152     gap = 0.00727601
#     <r^2>_ch = 84.269031  (half the LO radius: the fixed-order log is a +66% correction to the
#                            potential at the Bohr radius, so the state is much more bound)
#     F/F0 at q = 0.025 .. 0.2: 0.991277 0.965759 0.925289 0.872629 0.744008 0.605647
#
# Run parameters are matched to LO in the two dimensionless numbers that matter, so the two
# datasets are methodologically comparable rather than merely both present:
#
#                 dtau   count_skip   n_step   Delta = c_s*dtau   dtau*|E0|   Delta*gap   tau*gap
#     LO           4.5        5          60          22.5           0.0200      0.0750     0.900
#     NLO          2.0        5          60          10.0           0.0212      0.0728     0.873
#
# dtau*|E0| controls the Trotter error, Delta*gap the Krylov grid's resolution of the gap, and
# tau*gap the projection.  Same Q grid as LO so the two form factors overlay directly.
#
# spoila from the validated recipe, spoila = sqrt(target/c)/a0(spoila=1) with c = 0.75 exact for
# a meson: sqrt(84.269031/0.75) = 10.6000 and a0(spoila=1) = 11.930322, so spoila = 0.888164,
# which puts the trial radius on the NLO answer.  (The VMC study showed radius matching IS the
# variational optimum.)
set -eu
. "$(dirname "$0")/00_env.sh"

NW=${NW:-5000}
POOL=${POOL:-5}
O="$OUT/refmass_a0.1"
LOGS="$O/logs"
FO="$CODE/heavy_quark_nuclei_gfmc_boosted_FV_currents.py"
NAME=fo_meson_nlo
mkdir -p "$LOGS" "$O/$NAME"

COMMON="--nf 0 --Nc 3 --n_walkers $NW --no_color_vectors --wavefunction compact \
--potential full --color 1x1 --Q_dir 1 0 0 --seed 0 --mu 1.0 --log_mu_r 1"
ARGS="--N_coord 2 --masses 1 -1 --alpha 0.1 --OLO NLO --spoila 0.888164 \
--n_step 60 --dtau_iMev 2.0 --count_skip 5"
QS=${QS:-"0.0 0.025 0.05 0.075 0.1 0.15 0.2"}

TASKS="$O/tasks_fo_nlo.tsv"
: > "$TASKS"
n=0
for q in $QS; do
  printf '%s\n' "$q" >> "$TASKS"
  n=$((n+1))
done

cat > "$O/worker_fo_nlo.sh" <<'WORKER'
#!/bin/bash
q="$1"
log="$LOGS/${NAME}_Q${q}.log"
if "$PY" "$FO" $COMMON $ARGS --Q_mag "$q" --outdir "$O/$NAME/" > "$log" 2>&1; then
  echo "  done  $NAME Q=$q"
else
  echo "  FAIL  $NAME Q=$q  (see $log)"
fi
WORKER
chmod +x "$O/worker_fo_nlo.sh"
export COMMON ARGS LOGS O PY FO NAME

echo "fixed-order NLO meson, alpha = 0.1, nf = 0, $NW walkers, pool $POOL"
echo "output -> $O/$NAME"
echo "$n runs queued"
xargs -P "$POOL" -n 1 -I{} "$O/worker_fo_nlo.sh" {} < "$TASKS"
echo "all done"
