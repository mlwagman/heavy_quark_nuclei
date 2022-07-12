#!/bin/bash

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.002 --n_step 10 --n_walkers 5000
#gsed -i 's/nCoord = .*/nCoord = 3/' config.py
#gsed -i 's/VB = .*/VB = 1.18/' config.py

N_coord=3

n_walkers=1000
n_step=100
dtau=0.2
VB=1.18
a0=1.694
mpirun -np 1 python3 heavy_quark_nuclei_gfmc_poor.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB --a0 $a0 > gfmc_poor_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=1.18
a0=2.542
mpirun -np 1 python3 heavy_quark_nuclei_gfmc_poor.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB --a0 $a0 > gfmc_poor_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=1.18
a0=3.3898
mpirun -np 1 python3 heavy_quark_nuclei_gfmc_poor.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB --a0 $a0 > gfmc_poor_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=1.18
a0=4.2372
mpirun -np 1 python3 heavy_quark_nuclei_gfmc_poor.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB --a0 $a0 > gfmc_poor_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

wait

exit


#gsed -i 's/VB = .*/VB = .00878/' config.py
#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.0002 --n_step 100 --n_walkers 5000
