# run GFMC and fit wavefunction based on Hyleraas trial state
nw=2
nstep=2
nskip=100

alpha=0.75

dt=0.04
#dt=0.02

# molecular spatial wavefunction 
wvfn="product"

# check uncorrelated product is exact
sa=1
af=1.0
g=0.0
color="1x1"
echo python3 heavy_quark_nuclei_deut.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 -1 1 -1 --g $g --color $color --ferm_symm "s" --verbose

# Hylleraas wavefunction including symmetrization
sa=1.0
af=1.0
g=0.0
color="1x1"
#color="3x3bar"
ferm_symm="mas"

masses="1.0 -1.0 1.0 -1.0"
formatted_masses=$(echo $masses | sed 's/ /, /g')
file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.h5"
fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.csv"

echo "looking for $file"

#if [ ! -f "$file" ]; then
#python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 -1 1 -1 --g $g --color $color --verbose
python3 heavy_quark_nuclei_deut.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 -1 1 -1 --g $g --color $color --ferm_symm $ferm_symm --verbose
#fi
#if [ ! -f "$fit" ]; then
#python3 average_constant_fit.py "--database=${file}" --dtau $dt
#fi

exit

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
