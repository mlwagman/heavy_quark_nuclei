# run GFMC and fit wavefunction based on Hyleraas trial state
nw=200
nstep=20
nskip=100

alpha=0.75

dt=0.04
#dt=0.02

# molecular spatial wavefunction 
wvfn="product"

# check uncorrelated product is exact
sa=1
af=10000000000
g=0.0
color="1x1"
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 -1 1 -1 --g $g --color $color --verbose 

# Hylleraas wavefunction including symmetrization
sa=1.00033
af=5.82843
g=1.0
color="1x1"
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 -1 1 -1 --g $g --color $color --verbose 

# Hylleraas without spatial symmetrization
sa=1.00033
af=5.82843
g=0.0
color="1x1"
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 -1 1 -1 --g $g --color $color --verbose 

# attractive diquark cluster
sa=2.00066
af=5.82843
g=1.0
color="3x3bar"

python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 1 -1 -1 --g $g --color $color --verbose 

# repulsive diquark anti-cluster
sa=6.0
af=0.5
g=1.0
color="6x6bar"

python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 1 -1 -1 --g $g --color $color --verbose 
