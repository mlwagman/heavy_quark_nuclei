# run GFMC and fit wavefunction based on Hyleraas trial state
#nw=10000
#nw=20
nw=$2

out=/wclustre/bbind/data/deuteron/

nskip=100

alpha=1.5

dtArray=(0.4 0.2 0.1 0.05 0.02)
nstepArray=(25 50 100 200 500)

dt=${dtArray[$1]}
nstep=${nstepArray[$1]}

wvfn=$3

if [ $wvfn -eq 0 ]; then

# check uncorrelated product is exact
wvfn="compact"
sa=1.0
af=1.0
g=0.0
color="1"

file="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord3_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0]_color_${color}_g${g}.h5"

fit="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord3_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0]_color_${color}_g${g}.csv"

if [ ! -f "$file" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 3 --outdir $out --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1.0 1.0 1.0 --g $g --color $color > out_logs/nuc_${color}_afac${af}_nstep${nstep}_nwalkers${nw} #--verbose 
fi

if [ ! -f "$fit" ]; 
then
echo "fitting $fit"
python average_constant_fit.py "--database=${file}" --dtau $dt >> out_logs/nuc_${color}_afac${af}_nstep${nstep}_nwalkers${nw}
fi

fi
if [ $wvfn -eq 1 ]; then


# molecular spatial wavefunction 

# Hylleraas without spatial symmetrization
wvfn="product"
sa=1.00033
af=5.82843
g=0.0
color="1x1"

file="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.h5"

fit="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.csv"

if [ ! -f "$file" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir $out --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1.0 1.0 1.0 1.0 1.0 1.0 --g $g --color $color > out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw} #--verbose 
fi

if [ ! -f "$fit" ]; 
then
echo "fitting $fit"
python average_constant_fit.py "--database=${file}" --dtau $dt >> out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw}
fi

fi
if [ $wvfn -eq 2 ]; then

wvfn="diquark"
sa=1.0
af=4.0
g=0.0
color="AAA"

file="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.h5"

fit="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.csv"

if [ ! -f "$file" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir $out --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1.0 1.0 1.0 1.0 1.0 1.0 --g $g --color $color > out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw} #--verbose 
fi

if [ ! -f "$fit" ]; 
then
echo "fitting $fit"
python average_constant_fit.py "--database=${file}" --dtau $dt >> out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw}
fi

fi
if [ $wvfn -eq 3 ]; then

wvfn="product"
sa=1.00033
af=5.82843
g=0.0
color="AAS"

file="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.h5"

fit="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.csv"

if [ ! -f "$file" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir $out --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1.0 1.0 1.0 1.0 1.0 1.0 --g $g --color $color > out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw} #--verbose 
fi

if [ ! -f "$fit" ]; 
then
echo "fitting $fit"
python average_constant_fit.py "--database=${file}" --dtau $dt >> out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw}
fi

fi
if [ $wvfn -eq 4 ]; then

wvfn="diquark"
sa=1000.0
af=0.0016
g=0.0
color="SSS"

file="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.h5"

fit="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.csv"

if [ ! -f "$file" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir $out --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1.0 1.0 1.0 1.0 1.0 1.0 --g $g --color $color > out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw} #--verbose 
fi

if [ ! -f "$fit" ]; 
then
echo "fitting $fit"
python average_constant_fit.py "--database=${file}" --dtau $dt >> out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw}
fi

fi
if [ $wvfn -eq 5 ]; then

# product

wvfn="product"
sa=1.0
af=10000000000.0
g=0.0
color="1x1"

file="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.h5"

fit="${out}/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.csv"

if [ ! -f "$file" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir $out --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1.0 1.0 1.0 1.0 1.0 1.0 --g $g --color $color  > out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw} #--verbose 
fi

if [ ! -f "$fit" ]; 
then
echo "fitting $fit"
python average_constant_fit.py "--database=${file}" --dtau $dt >> out_logs/deut_${color}_afac${af}_nstep${nstep}_nwalkers${nw}
fi

fi
