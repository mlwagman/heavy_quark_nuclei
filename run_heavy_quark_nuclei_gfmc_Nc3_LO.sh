#!/bin/bash

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.002 --n_step 10 --n_walkers 5000
#gsed -i 's/nCoord = .*/nCoord = 3/' config.py
#gsed -i 's/VB = .*/VB = 1.18/' config.py

N_coord=3

n_walkers=1000
n_step=800
dtau=0.2
alpha=0.1
VB=`bc -l <<< "2*${alpha}/3"`
echo "alpha = ${alpha}, VB = ${VB}"

mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_alpha${alpha}_a0${a0} &

n_walkers=1000
n_step=200
dtau=0.2
alpha=0.4
VB=`bc -l <<< "2*${alpha}/3"`
echo "alpha = ${alpha}, VB = ${VB}"

mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_alpha${alpha}_a0${a0} &

wait
exit

