#!/bin/bash
# Resume an interrupted tranche: re-run only the tasks whose output h5 is missing.
#
# The box has rebooted twice while tranche 3 was running (killing 16 of 23 runs the
# first time), so the driver must be restartable without redoing finished work.
# A run is "done" iff its currents_3pt_*Qmag<q>.h5 exists in the config's outdir.
# Grep the listing rather than glob: currents_3pt_, Hammys_ and Rs_ all end Qmag<q>.h5,
# so a bare glob matches three files and overflows the argument.
#
# usage: ./14_resume.sh <tasks.tsv> [worker.sh]
set -eu
. "$(dirname "$0")/00_env.sh"
cd "$CODE"

TASKS="$1"
O="$OUT/refmass_a0.1"
WORKER="${2:-$O/worker3.sh}"
LOGS="$O/logs"
POOL=${POOL:-4}
mkdir -p "$LOGS"

FO=heavy_quark_nuclei_gfmc_boosted_FV_currents.py
MRS=heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py
BASE="--nf 0 --Nc 3 --no_color_vectors --wavefunction compact --potential full \
--color 1x1 --Q_dir 1 0 0"
export BASE LOGS O PY

TODO="$O/$(basename "$TASKS" .tsv)_todo.tsv"
: > "$TODO"
done_n=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  name=$(printf '%s' "$line" | cut -f1)
  q=$(printf '%s' "$line" | cut -f3)
  if ls "$O/$name" 2>/dev/null | grep -q "^currents_3pt_.*Qmag${q}\.h5$"; then
    done_n=$((done_n+1))
  else
    printf '%s\n' "$line" >> "$TODO"
  fi
done < "$TASKS"

n=$(wc -l < "$TODO" | tr -d ' ')
echo "resume $(basename "$TASKS"): $done_n already done, $n to run, pool $POOL"
[ "$n" -eq 0 ] && { echo "nothing to do"; exit 0; }
tr '\n' '\0' < "$TODO" | xargs -0 -P "$POOL" -I{} "$WORKER" "{}"
echo "resume complete: all $n runs finished"
