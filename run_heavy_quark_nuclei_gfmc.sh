#!/bin/bash

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.002 --n_step 10 --n_walkers 5000
#gsed -i 's/nCoord = .*/nCoord = 3/' config.py
#gsed -i 's/VB = .*/VB = 1.18/' config.py

N_coord=4

n_walkers=1000
n_step=200
dtau=0.2
VB=0.83078
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.061478
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.018506
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.0092534
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.0061689
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

wait

N_coord=5

n_walkers=1000
n_step=50
dtau=0.2
VB=0.63804
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.047215
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.014213
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.0071066
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.0047377
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

wait

N_coord=6

n_walkers=250
n_step=50
dtau=0.2
VB=0.51693
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.038253
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.0115154
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.0057576
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

dtau=0.2
VB=0.0038384
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=$dtau --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --VB $VB > gfmc_log_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_VB${VB}_a0${a0} &

exit


#gsed -i 's/VB = .*/VB = .00878/' config.py
#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 1000 --n_walkers 1000

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.0002 --n_step 100 --n_walkers 5000
