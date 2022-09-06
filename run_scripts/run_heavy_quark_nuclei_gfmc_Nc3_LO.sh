#!/bin/bash

N_coord=3

n_walkers=1000
n_step=800
dtau=0.2
alpha=0.1

mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --alpha $alpha --outdir "data/" > logs/gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_alpha${alpha}_a0${a0} &

n_walkers=1000
n_step=200
dtau=0.2
alpha=0.4

mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --alpha $alpha --outdir "data/" > logs/gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_alpha${alpha}_a0${a0} &

wait

exit
