#!/bin/bash
#SBATCH -J hqn_svm
#SBATCH -p preq
#SBATCH -N 1
#SBATCH -t 23:59:00
#SBATCH --requeue
#SBATCH -o /home/bassi/hqn/logs/svm_node_%j.out
# MRS baryons with SVM trials (svm_baryon.py --save_trial), mu_cut = 3.4 Lambda_d, lambda = mufac in {1/2,1,2}; 4000 walkers, 1000 bins.
# forward-walking radius per dataset. Queue file: $QFILE.
# Per dataset: tune spoila (baryon) -> 7 Q -> reduce -> verify -> korr -> delete h5.
# Resumes after a requeue: finished Q are skipped, finished datasets too.
cd ~/hqn
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=$HOME/hqn/env/bin/python
QBASE="0.0 0.02 0.05 0.1 0.2 0.35 0.5"
COMMON="--nf 0 --Nc 3 --potential full --color 1x1 --Q_dir 1 0 0 --mQ 1.0 --Rstar 0.35"

run_one () {
  IFS='|' read -r NAME SYS REF K ARGS <<< "$1"
  [ -s "results/$NAME.csv" ] && return
  O=runs/$NAME; mkdir -p "$O"; L=logs/$NAME.log
  echo "$NAME start $(date '+%m-%d %H:%M')"
  if [[ "$ARGS" == *"--spoila OPT"* ]]; then
    C=$(echo "$ARGS" | sed 's/.*--spoila OPT\([0-9.]*\).*/\1/')
    if [ ! -s "$O/spoila.txt" ]; then
      AL=$(echo "$ARGS" | sed 's/.*--alpha_s \([0-9.]*\).*/\1/'); OLO=$(echo "$ARGS" | sed 's/.*--OLO \([A-Z]*\).*/\1/')
      MU=$(echo "$ARGS" | sed 's/.*--mu \([0-9.eE-]*\).*/\1/'); MUF=$(echo "$ARGS" | sed 's/.*--mufac \([0-9.]*\).*/\1/')
      GRID=$($PY -c "print(','.join(f'{$C*f:.4f}' for f in (0.85,0.9,0.95,1.0,1.05,1.1,1.15)))")
      (cd code && $PY optimize_spoila.py --OLO $OLO --alpha_s $AL --N_coord 3 --mu $MU --mufac $MUF --grid $GRID) > "$O/spoila_scan.txt" 2>>"$L"
      grep "optimal spoila" "$O/spoila_scan.txt" | sed 's/optimal spoila = \([0-9.]*\).*/\1/' > "$O/spoila.txt"
    fi
    SP=$(cat "$O/spoila.txt"); [ -z "$SP" ] && SP=$C
    ARGS=$(echo "$ARGS" | sed "s/--spoila OPT[0-9.]*/--spoila $SP/")
  fi
  [ "$SYS" = baryon ] && ARGS="$ARGS --masses 1 1 1"
  for QB in $QBASE; do
    Q=$($PY -c "print(f'{$QB/$K:.6g}')")
    ls "$O"/currents_3pt_*Qmag"$Q".h5 2>/dev/null | grep -qv korr && continue
    $PY code/heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py $ARGS $COMMON --Q_mag "$Q" --outdir "$O/" >> "$L" 2>&1 \
      || echo "$NAME !! Q=$Q nonzero exit"
  done
  R=""; [ "$REF" != "-" ] && R=$REF
  if $PY code/gfmc_reduce.py "$O" "results/$NAME.npz" 1000 >> "$L" 2>&1 \
     && $PY -c "import numpy as n;d=n.load('results/$NAME.npz');assert len(d['Q'])==7" 2>>"$L" \
     && $PY code/gfmc_korr.py "results/$NAME.npz" "$SYS" $R --out "results/$NAME.csv" > "results/$NAME.txt" 2>>"$L"; then
    cat "results/$NAME.txt"
    [ -s "$O/spoila_scan.txt" ] && cp "$O/spoila_scan.txt" "results/$NAME.spoila.txt"
    $PY code/fw.py "$O" > "results/$NAME.fw.txt" 2>>"$L" && tail -1 "results/$NAME.fw.txt"
    rm -rf "$O"; echo "$NAME done $(date '+%m-%d %H:%M'), h5 deleted"
  else
    echo "$NAME !! reduce/extract failed, h5 kept in $O"
  fi
}

while IFS= read -r line; do [ -n "$line" ] && run_one "$line" & sleep 5; done < $QFILE
wait
echo "NODE_DONE $(date '+%m-%d %H:%M')"
