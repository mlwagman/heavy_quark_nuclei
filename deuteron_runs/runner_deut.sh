#!/bin/bash

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.282678 --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 1.130712 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.282678 --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 1.130712 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.214850 --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 0.8594 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.214850 --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 0.8594 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.313613 --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 1.254452 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.313613 --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 1.254452 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.227318 --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 0.909272 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --alpha 0.227318 --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --mu 0.909272 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.05 --mu 0.2 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.05 --mu 0.2 --wavefunction "two_baryon_product" --potential "antisymetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.1--mu 0.4 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.1--mu 0.4 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.15--mu 0.6 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.15--mu 0.6 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.2--mu 0.8 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.2--mu 0.8 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.25--mu 1 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.25--mu 1 --wavefunction "two_baryon_product" --potential "antisymetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.3--mu 1.2 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.3--mu 1.2 --wavefunction "two_baryon_product" --potential "antisymetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.35--mu 1.4 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "LO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.35--mu 1.4 --wavefunction "two_baryon_product" --potential "antisymetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.05 --mu 0.2 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.05 --mu 0.2 --wavefunction "two_baryon_product" --potential "antisymetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.1--mu 0.4 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.1--mu 0.4 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.15--mu 0.6 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 5 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.15--mu 0.6 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.2--mu 0.8 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.2--mu 0.8 --wavefunction "two_baryon_product" --potential "antisymmetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.25--mu 1 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.25--mu 1 --wavefunction "two_baryon_product" --potential "antisymetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.3--mu 1.2 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.3--mu 1.2 --wavefunction "two_baryon_product" --potential "antisymetric" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.35--mu 1.4 --wavefunction "two_baryon_product" --potential "full" &

python3 heavy_quark_nuclei_gfmc_col.py --log_mu_r 0.5 --OLO "NLO" --n_step 200 --n_walkers 100 --dtau 0.2 --Nc 3 --nf 4 --N_coord 6 --outdir "~/bassi/heavy_quark_nuclei/deuteron_runs/" --verbose --alpha 0.35--mu 1.4 --wavefunction "two_baryon_product" --potential "antisymetric" &
