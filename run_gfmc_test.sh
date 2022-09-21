#!/bin/bash

set -x

python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --OLO=NNLO --n_step 20 --n_walkers 20 --N_coord 3 --Nc 3 --nf 4 --alpha 0.324475 --mu 1.2979 --outdir data
