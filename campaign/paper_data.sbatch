#!/bin/bash
#SBATCH -J hqn_paper
#SBATCH -p debugq
#SBATCH -N 1
#SBATCH -t 00:59:00
#SBATCH -o /home/bassi/hqn/logs/paper_data_%j.out
cd ~/hqn/code
export JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NPROC=40 $HOME/hqn/env/bin/python paper_solver.py svm_baryon_lam_old.csv ../results/svm_baryon_lam.csv ../results/svm_ff.csv svm_queue_1.txt svm_queue_2.txt
cd ~/hqn
$HOME/hqn/env/bin/python code/svm_cache.py
echo PAPER_DATA_DONE
