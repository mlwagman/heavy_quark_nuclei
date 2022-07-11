#!/bin/bash

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.002 --n_step 10 --n_walkers 5000
#gsed -i 's/nCoord = .*/nCoord = 3/' config.py
#gsed -i 's/VB = .*/VB = 1.18/' config.py

N_coord=3

n_walkers=1000
n_step=20
dtau=0.2
VB=1.18
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} & 

n_step=100
dtau=0.04
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB} & 

wait

exit

gsed -i 's/VB = .*/VB = .211/' config.py
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 200 --n_walkers 1000
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.04 --n_step 1000 --n_walkers 1000
gsed -i 's/VB = .*/VB = .0874/' config.py
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000
gsed -i 's/VB = .*/VB = .0437/' config.py
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000
gsed -i 's/VB = .*/VB = .0202/' config.py
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000
gsed -i 's/VB = .*/VB = .0132/' config.py
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000
gsed -i 's/VB = .*/VB = .00878/' config.py
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.0002 --n_step 100 --n_walkers 5000
