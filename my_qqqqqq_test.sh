# run GFMC and fit wavefunction based on Hyleraas trial state
nw=1000
#nstep=1000
nstep=400
#nstep=0
nskip=100

alpha=1.5

#dt=0.04
#dt=0.02
#
dt=0.02
#dt=0.01
#dt=0.002



# Hylleraas without spatial symmetrization
sa=1.00033
#af=5.82843
#af=6.82843
af=7.1
g=0.0
color="1x1"
wvfn="product"

ferm_symm="mas"

# dineutron

masses="1.0 2.0 2.0 4.0 2.0 2.0"
formatted_masses=$(echo $masses | sed 's/ /, /g')
file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.h5"
fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.csv"
echo "looking for $file"

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_deut.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $masses --g $g --color $color --ferm_symm $ferm_symm # --verbose 
fi
if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt
fi

exit

# deuteron

masses="1.0 2.0 2.0 1.0 1.0 2.0"
formatted_masses=$(echo $masses | sed 's/ /, /g')
file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.h5"
fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.csv"
echo "looking for $file"

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_deut.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $masses --g $g --color $color --ferm_symm $ferm_symm # --verbose 
fi
if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt
fi

# H-dibaryon

masses="1.0 2.0 3.0 1.0 2.0 4.0"
formatted_masses=$(echo $masses | sed 's/ /, /g')
file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.h5"
fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.csv"
echo "looking for $file"

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_deut.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $masses --g $g --color $color --ferm_symm $ferm_symm #  --verbose 
fi
if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt
fi

# Nilmanion

masses="1.0 1.0 1.0 1.0 1.0 1.0"
formatted_masses=$(echo $masses | sed 's/ /, /g')
file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.h5"
fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.csv"
echo "looking for $file"

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_deut.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $masses --g $g --color $color --ferm_symm $ferm_symm # --verbose 
fi
if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt
fi
