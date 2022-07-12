#!/bin/bash

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.002 --n_step 10 --n_walkers 5000
#gsed -i 's/nCoord = .*/nCoord = 3/' config.py
#gsed -i 's/VB = .*/VB = 1.18/' config.py

N_coord=3

n_walkers=1000
n_step=1000
dtau=0.04
VB=0.211
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} &

dtau=0.2
VB=0.0874
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} &

dtau=0.2
VB=0.0437
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} &

dtau=0.2
VB=0.0202
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} &

dtau=0.2
VB=0.0132
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} &

dtau=0.2
VB=0.00878
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} &

wait

exit


#gsed -i 's/VB = .*/VB = .00878/' config.py
#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.0002 --n_step 100 --n_walkers 5000
