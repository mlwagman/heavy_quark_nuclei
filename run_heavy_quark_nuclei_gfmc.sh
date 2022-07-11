#!/bin/bash

#python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.002 --n_step 10 --n_walkers 5000
gsed -i 's/nCoord = .*/nCoord = 3/' config.py
gsed -i 's/VB = .*/VB = 1.18/' config.py
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.2 --n_step 20 --n_walkers 1000
python3 heavy_quark_nuclei_gfmc.py --dtau_iMev=0.04 --n_step 100 --n_walkers 1000
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
