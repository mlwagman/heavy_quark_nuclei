#!/bin/bash

python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.236083 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.50278543307  -0.493255690035 1.50278543307 -0.493255690035 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.236083 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.50278543307 0.493255690035 -1.50278543307 0.493255690035 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.236083 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.50278543307 -1.50278543307 0.493255690035 -0.493255690035 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.236083 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.50278543307 1.50278543307 -0.493255690035 0.493255690035 &

wait

python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253597 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 2.0256143712574853 0.6617448144189821 -0.6617448144189821 0.6617448144189821  &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253597 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -2.0256143712574853 0.6617448144189821 -0.6617448144189821 0.6617448144189821  &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253597 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.6617448144189821 -0.6617448144189821 2.0256143712574853 -0.6617448144189821  &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253597 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.6617448144189821 0.6617448144189821 -2.0256143712574853 0.6617448144189821  &

wait

python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.223949 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.19832 1.20461 -1.20461 0.39412 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.223949 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.19832 1.20461 -1.20461 0.39412 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.223949 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.19832 -0.39412 -1.20461 1.20461 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.223949 --log_mu_r 0.0 --OLO "LO" --n_step 400 --n_walkers 10000 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.19832 0.39412 1.20461 -1.20461 &

wait


python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253567 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.30712 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.48874847891  1.49027759882 -0.506642457868 0.506171577775  &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253567 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.30712 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.48874847891  -1.49027759882 0.506642457868 -0.506171577775  &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253567 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.30712 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.48874847891 -0.506642457868 1.48874847891 -0.506642457868  &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.253567 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.30712 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.48874847891 0.506642457868 -1.48874847891 0.506642457868  &

wait


python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.275793 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 2.70991 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.982441368815592  0.6726221786447639 -0.6726221786447639 0.6726221786447639 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.275793 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 2.70991 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.982441368815592  -0.6726221786447639 0.6726221786447639 -0.6726221786447639 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.275793 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 2.70991 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.6726221786447639 -0.6726221786447639 0.6726221786447639 -1.982441368815592 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.275793 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 2.70991 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.6726221786447639 0.6726221786447639 -0.6726221786447639 1.982441368815592 &

wait

python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.238471 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.87728 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.982441368815592  1.982441368815592 -1.982441368815592 0.6726221786447639 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.238471 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.87728 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.982441368815592  -1.982441368815592 1.982441368815592 -0.6726221786447639 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.238471 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.87728 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.6726221786447639  1.982441368815592  -1.982441368815592 1.982441368815592 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.238471 --log_mu_r 0.0 --OLO "NLO" --n_step 400 --n_walkers 10000 --mu 3.87728 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.6726221786447639  -1.982441368815592  1.982441368815592 -1.982441368815592 &

wait


python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.24612 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.31807 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.527951489 0.524816896 -1.475585212 1.472450619 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.24612 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.31807 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.527951489 -0.524816896 1.475585212 -1.472450619 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.24612 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.31807 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.527951489 1.472450619 -0.527951489 1.472450619 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.24612 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.31807 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.527951489 -1.475585212 0.527951489 -1.475585212 &

wait

python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.265493 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 2.72996 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.689365  -1.937736 0.689365 -0.689365 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.265493 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 2.72996 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.689365 1.937736 -0.689365 0.689365 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.265493 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 2.72996 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.689365  -0.689365 0.689365 -1.937736 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.265493 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 2.72996 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.689365  0.689365 -0.689365 1.937736 &


wait

python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.232626 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.88029 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 0.425440137686  -1.19182258709 1.19075815773 -1.19182258709 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.232626 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.88029 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -0.425440137686  1.19075815773 -1.19182258709 1.19075815773 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.232626 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.88029 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses 1.19075815773  -1.19182258709 0.425440137686 -1.19182258709 &
python3 heavy_quark_nuclei_gfmc_FV.py --alpha 0.232626 --log_mu_r 0.0 --OLO "NNLO" --n_step 400 --n_walkers 10000 --mu 3.88029 --dtau 0.4 --Nc 3 --nf 4 --N_coord 4 --outdir "" --wavefunction "product" --potential "full" --Lcut 5 --L 0 --afac 5 --masses -1.19182258709  1.19075815773 -0.425440137686 1.19075815773 &
