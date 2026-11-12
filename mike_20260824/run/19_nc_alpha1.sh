#!/bin/bash
# Nc scan: fixed-order Coulomb baryons at alpha = 1, N_coord = Nc.
#
# Coulomb scales exactly, so these runs rescale to any coupling: <r^2> -> <r^2>/alpha^2,
# E0 -> E0*alpha^2, F at q -> F at q*alpha.  See 17_nc_scan.sh for why alpha = 1 is the right
# place to work and where every number below comes from.
#
# Exact references from ../../svm_nbody.py (general-N correlated Gaussian), on these Q grids:
#   Nc=3  <r^2>_ch = 3.8260983   F/F0 = 0.974918 0.904503 0.801393 0.681613 0.447258 0.268560
#   Nc=4  <r^2>_ch = 2.7191553   F/F0 = 0.974332 0.906044 0.801557 0.679270 0.443208 0.258455
# The grids are matched in q*sqrt(<r^2>), which is why the two F/F0 rows nearly coincide --
# every Nc is probed at the same place on its own form factor.
#
# Nc=3 is a NULL TEST: rescaled by alpha=0.1 it must reproduce fo_baryon_lo, 389.2 +/- 5.9
# against 384, since Coulomb scaling is exact and that run used the same generator.
#
# Nc=5 is deliberately absent: svm_nbody.py does not yet converge at nj = 4 (virial 1.006..1.096
# across seeds), so there would be nothing to compare against.  Add it once the solver does
# one-at-a-time SVM selection instead of random widths.
#
# NW and QS are the two cost dials.  A 200-walker, 270-step Nc=4 job takes 14m27s here, so size
# the tranche before raising them: N_coord enters the gradient and Laplacian through triply
# nested loops, on top of Nc(Nc-1)/2 pairs.  --f0_analytic in the extractor means the Q=0 run is
# optional (F(0) = N is an identity for the consistent-Gram GEVP), so Q=0 is NOT in the default
# grid.
set -eu
. "$(dirname "$0")/00_env.sh"

FO=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
NW=${NW:-5000}
POOL=${POOL:-2}
NICE=${NICE:-5}      # this box is shared with other sessions and other projects
O="$OUT/nc_alpha1"
LOGS="$O/logs"
mkdir -p "$LOGS"

COMMON="--nf 0 --n_walkers $NW --no_color_vectors --wavefunction compact --potential full \
--color 1x1 --Q_dir 1 0 0 --seed 0 --alpha 1 --n_skip 50"

# name|Nc|Q list|config-specific args
JOBS=$(cat <<'SPEC'
nc3_baryon|3|0.2 0.4 0.6 0.8 1.2 1.6|--spoila 1.002226 --n_step 200 --dtau_iMev 0.027 --count_skip 10
nc4_baryon|4|0.24 0.47 0.71 0.95 1.42 1.90|--spoila 0.996484 --n_step 270 --dtau_iMev 0.011824 --count_skip 14
SPEC
)

TASKS="$O/tasks.tsv"
: > "$TASKS"
while IFS='|' read -r name nc qs extra; do
  [ -z "$name" ] && continue
  ms=$(for i in $(seq "$nc"); do printf '1 '; done)
  for q in ${QS:-$qs}; do
    printf '%s\t%s\t%s\n' "$name" "$q" \
      "$COMMON --Nc $nc --N_coord $nc --masses $ms $extra --Q_mag $q --outdir $O/$name/" \
      >> "$TASKS"
  done
done <<< "$JOBS"

n=$(wc -l < "$TASKS" | tr -d ' ')
echo "Nc scan at alpha = 1: $n jobs, $NW walkers, pool $POOL -> $O"
echo "References: Nc=3 <r^2>_ch = 3.8260983, Nc=4 <r^2>_ch = 2.7191553 (svm_nbody.py)"

# A worker script file rather than an inline bash -c: BSD xargs caps the -I replacement at
# -S bytes (255 by default), and "command line cannot be assembled, too long" is what it says
# when a task line plus an inline script exceeds that.  Same pattern as 10/12/13/15.
cat > "$O/worker.sh" <<'WEOF'
#!/bin/bash
name=$(printf '%s' "$1" | cut -f1); q=$(printf '%s' "$1" | cut -f2)
cmd=$(printf '%s' "$1" | cut -f3)
mkdir -p "$OUT/nc_alpha1/$name"
cd "$CODE"
if nice -n "${NICE:-5}" "$PY" "$CODE/$FO" $cmd > "$OUT/nc_alpha1/logs/${name}_Q${q}.log" 2>&1
then echo "ok   $name Q=$q"
else echo "FAIL $name Q=$q  -- see logs/${name}_Q${q}.log"
fi
WEOF
chmod +x "$O/worker.sh"
export PY CODE FO OUT NICE

# xargs rather than wait -n: bash 3.2 here.  -S raises the -I replacement buffer.
tr '\n' '\0' < "$TASKS" | xargs -0 -S 65536 -P "$POOL" -I{} "$O/worker.sh" "{}"
echo "done; resume anything missing with 14_resume.sh $TASKS"
