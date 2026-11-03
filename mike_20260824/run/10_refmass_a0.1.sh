#!/bin/bash
# Form factors and charge radii at the Nf=0 reference mass, alpha_s(mQ) = 0.1.
#
# The reference mass. In Nf=0 the only dimensionless inputs are alpha_s(mQ),
# mu_cut/mQ, lambda=mufac and Rstar*mQ, so alpha_s(mQ)=0.1 fixes the RATIO
# mQ/Lambda^(0), not mQ itself:
#     LO  (1-loop running): mQ = 302.4714 Lambda^(0),  Lambda/mQ = 3.306098e-3
#     NLO (2-loop running): mQ =  93.9223 Lambda^(0),  Lambda/mQ = 1.064710e-2
# We therefore run at mQ=1 and quote lengths in units of 1/mQ; the ratios above
# convert to units of Lambda. This is the same construction Ben's campaign uses:
# his mu_cut = 3.4*Lambda_d(2-loop) = 0.0362001*mQ is 3.4/93.9223, i.e. the NLO
# reference mass. We adopt his cutoff so the numbers are directly comparable --
# the August bundle's "--mu 2.0" is a UV freeze and gives a different theory
# (MRS LO meson <r^2> = 133.04 at mu=2.0 versus 56.69 at mu_cut).
#
# spoila is the compact-trial width matched to the exact radius,
#     spoila = sqrt(4 <r^2>_ch / 3) / a0(spoila=1),
# the recipe in 02_mrs_meson.sh; it reproduces Ben's tuned 0.9241 to 0.9240.
# dtau follows Ben's 0.3*k^2 with k = a_eff/a_eff(alpha_s=0.3) for MRS, and the
# exact alpha^2 rule of 04_smallalpha.sh for the fixed-order Coulomb runs.
#
# References to hit (all <r^2>_ch, one-body about the CM):
#   FO  meson  LO  : 168.75      exact, 27/(16 alpha^2)
#   FO  baryon LO  : 343.62      svm_baryon.py --alpha 0.1
#   MRS meson  LO  :  56.69      mrs_numerov.py, mu_cut, lambda=1
#   MRS meson  NLO :  52.07      mrs_numerov.py, mu_cut, lambda=1
#   MRS baryon LO  : 114.251     SVM; Ben's GFMC 116.744 +/- 2.16
#   MRS baryon NLO : 103.142     SVM; Ben's GFMC 105.539 +/- 2.06
set -eu
. "$(dirname "$0")/00_env.sh"
cd "$CODE"

POOL=${POOL:-5}
NW=${NW:-5000}
MU=0.0362001                     # 3.4 * Lambda_d(2-loop) at alpha_s = 0.1
O="$OUT/refmass_a0.1"
LOGS="$O/logs"
mkdir -p "$LOGS"

FO=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
MRS=heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py

COMMON="--nf 0 --Nc 3 --n_walkers $NW --no_color_vectors --wavefunction compact \
--potential full --color 1x1 --Q_dir 1 0 0 --seed 0"

# name|script|Q list|config-specific args
JOBS=$(cat <<'SPEC'
fo_meson_lo|FO|0.0 0.025 0.05 0.075 0.1 0.15 0.2|--N_coord 2 --masses 1 -1 --alpha 0.1 --spoila 1 --n_step 60 --dtau_iMev 4.5 --count_skip 5
fo_baryon_lo|FO|0.0 0.02 0.04 0.06 0.08 0.12 0.16|--N_coord 3 --masses 1 1 1 --alpha 0.1 --spoila 0.713489 --n_step 200 --dtau_iMev 2.7 --count_skip 10
mrs_meson_lo|MRS|0.0 0.05 0.1 0.15 0.2 0.3 0.4|--N_coord 2 --alpha_s 0.1 --OLO LO --spoila 0.611170 --n_step 200 --dtau_iMev 0.632466 --count_skip 10
mrs_meson_nlo|MRS|0.0 0.05 0.1 0.15 0.2 0.3 0.4|--N_coord 2 --alpha_s 0.1 --OLO NLO --spoila 0.582644 --n_step 200 --dtau_iMev 0.580922 --count_skip 10
mrs_baryon_lo|MRS|0.0 0.04 0.08 0.12 0.16 0.22 0.28|--N_coord 3 --alpha_s 0.1 --OLO LO --spoila 0.611170 --n_step 200 --dtau_iMev 0.632466 --count_skip 10
mrs_baryon_nlo|MRS|0.0 0.04 0.08 0.12 0.16 0.22 0.28|--N_coord 3 --alpha_s 0.1 --OLO NLO --spoila 0.582644 --n_step 200 --dtau_iMev 0.580922 --count_skip 10
SPEC
)

# Build one tab-separated task per line, then run the pool with xargs -P.
# (bash 3.2 on macOS has no `wait -n`, so no hand-rolled pool.)
TASKS="$O/tasks.tsv"
: > "$TASKS"
n=0
while IFS='|' read -r name which qs args; do
  [ -z "$name" ] && continue
  case "$which" in FO) script=$FO ;; MRS) script=$MRS ;; esac
  extra=""
  [ "$which" = "MRS" ] && extra="--mQ 1.0 --mu $MU --mufac 1.0 --Rstar 0.35"
  for q in $qs; do
    mkdir -p "$O/$name"
    n=$((n+1))
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$script" "$q" "$args" "$extra" >> "$TASKS"
  done
done <<< "$JOBS"

cat > "$O/worker.sh" <<'WORKER'
#!/bin/bash
IFS=$'\t' read -r name script q args extra <<< "$1"
log="$LOGS/${name}_Q${q}.log"
if "$PY" "$script" $COMMON $args $extra --Q_mag "$q" --outdir "$O/$name/" > "$log" 2>&1; then
  echo "  done  $name Q=$q"
else
  echo "  FAIL  $name Q=$q  (see $log)"
fi
WORKER
chmod +x "$O/worker.sh"
export COMMON LOGS O PY

echo "alpha_s(mQ) = 0.1, Nf=0, mu_cut = $MU mQ, lambda = 1, $NW walkers, pool $POOL"
echo "output -> $O"
echo "$n runs queued"
# -I{} passes each whole line as one argument; the worker splits on tabs.
tr '\n' '\0' < "$TASKS" | xargs -0 -P "$POOL" -I{} "$O/worker.sh" "{}"
echo "all $n runs finished"
