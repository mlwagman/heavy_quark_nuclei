#!/bin/bash
#SBATCH -J nd_node
#SBATCH -p preq,defq
#SBATCH -N 1
#SBATCH -c 80
#SBATCH --mem=0
#SBATCH -t 23:59:00
#SBATCH -o /home/bassi/hqn/cfree/logs/node_%A_%a_%j.out
# NPER parallel workers on one node (or a sharedq slice). Each worker walks $QUEUE in priority order and claims a
# line with an atomic mkdir in claims/; lines with results or live claims are skipped. One line = one
# (dataset, Q, split). Workers touch claims/NAME/alive every 2 min; cleanup_claims.py releases claims whose
# heartbeat stopped (preemption, node loss), and workers keep passing over the queue until a pass claims nothing.
# A line that fails is released for retry at most twice.
cd ~/hqn/cfree
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=$HOME/hqn/env/bin/python
NPER=${NPER:-44}; QUEUE=${QUEUE:-code/queue_rerun.txt}
COMMON="--nf 0 --mQ 1.0 --Rstar 0.35 --potential full --Q_dir 1 0 0 --wavefunction compact --no_color_vectors --only_N --sym_only --no_Rs_file --fast_lap --fast_metropolis 2000"
mkdir -p claims results runs logs/runs logs/retry
echo "node $(hostname) $NPER workers on $QUEUE, restart ${SLURM_RESTART_COUNT:-0}, start $(date +%m-%d_%H:%M)"
$PY code/cleanup_claims.py 10 120
worker () {
  while :; do
    made=0
    while IFS='|' read -r NAME ARGS; do
      [ -s "results/$NAME.npz" ] && continue
      mkdir "claims/$NAME" 2>/dev/null || continue
      made=1
      echo "$SLURM_JOB_ID ${SLURM_RESTART_COUNT:-0} $(hostname) $1" > "claims/$NAME/who"; touch "claims/$NAME/alive"
      ( while sleep 120; do touch "claims/$NAME/alive" 2>/dev/null || exit 0; done ) & HB=$!
      O="$HOME/hqn/cfree/runs/$NAME"; rm -rf "${O:?}"; mkdir -p "$O"
      if (cd code && $PY heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py $ARGS $COMMON --outdir "$O/") > "$O/log" 2>&1 \
         && $PY code/reduce_sym.py "$O" "results/$NAME.npz" 50 >> "$O/log" 2>&1; then
        grep -v "computed step" "$O/log" | tail -60 > "logs/runs/$NAME.log"; rm -rf "${O:?}"
        echo "$NAME done $(date +%m-%d_%H:%M)"
      else
        echo "$NAME FAILED $(date +%m-%d_%H:%M)"; tail -3 "$O/log"
        grep -v "computed step" "$O/log" | tail -60 > "logs/runs/$NAME.FAILED.log"; rm -rf "${O:?}"
        k=$(cat "logs/retry/$NAME" 2>/dev/null || echo 0); echo $((k+1)) > "logs/retry/$NAME"
        [ "$k" -lt 2 ] && rm -rf "claims/${NAME:?}"
      fi
      kill $HB 2>/dev/null
    done < "$QUEUE"
    [ $made = 0 ] && break
    [ "$1" = 1 ] && $PY code/cleanup_claims.py 10 120
  done
}
for k in $(seq 1 $NPER); do worker $k & sleep 1; done
wait
echo "NODE_DONE $(date +%m-%d_%H:%M)"
