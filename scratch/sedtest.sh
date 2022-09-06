#!/bin/bash

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.002 --n_step 10 --n_walkers 5000
gsed -i 's/nCoord = .*/nCoord = 3/' config.py

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.0002 --n_step 100 --n_walkers 5000
