#!/bin/bash

n_walkers=1000
dtau=0.2
outdir="/work1/bbind/heavy_quark_nuclei/data/"
logdir="/work1/bbind/heavy_quark_nuclei/logs/"

Nc=3
N_coord=2

OLO="LO"

nf=4
nf4steplist=(567 425 330 264 216 180 152 130)
nf4alphalist=(0.187861 0.216975 0.24609 0.275204 0.304319 0.333433 0.362547 0.391662)

for ii in ${!nf4steplist[@]}; do
n_step=${nf4steplist[$ii]}
alpha=${nf4alphalist[$ii]}
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py  --dtau_iMev=$dtau --OLO=$OLO --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --Nc $Nc --nf $nf --alpha $alpha --outdir $outdir > ${logdir}/gfmc_log_OLO${OLO}_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_Nc${Nc}_nf${nf}_alpha${alpha} &
done

nf=5
nf5steplist=(1700 1095 763 562 431 341 277 229)
nf5alphalist=(0.10845 0.135171 0.161892 0.188614 0.215335 0.242056 0.268777 0.295499)

for ii in ${!nf5steplist[@]}; do
n_step=${nf5steplist[$ii]}
alpha=${nf5alphalist[$ii]}
mpirun -np 1 python3 heavy_quark_nuclei_gfmc.py  --dtau_iMev=$dtau --OLO=$OLO --n_step $n_step --n_walkers $n_walkers --N_coord $N_coord --Nc $Nc --nf $nf --alpha $alpha --outdir $outdir > ${logdir}/gfmc_log_OLO${OLO}_dtau${dtau}_nstep${n_step}_Ncoord${N_coord}_Nc${Nc}_nf${nf}_alpha${alpha} &
done

wait

exit
