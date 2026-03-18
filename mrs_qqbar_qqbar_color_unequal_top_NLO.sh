#python=/opt/homebrew/Cellar/jupyterlab/4.2.1/libexec/bin/python
python="python3"

gfmc="simon/heavy_quark_nuclei_gfmc_MRS_mu_prime.py"
Rstar=0.35
mu=1.0

#mt=172.56
mt=172.52
mc=1.55227
mb=4.80312

#mc_under_mt=111.166
mc_under_mt=111.140
#mb_under_mt=35.9266
mb_under_mt=35.9183


# run GFMC and fit wavefunction based on Hyleraas trial state
nw=10000
nstep=1000
nskip=100
dt=0.001

g=0.0

mufac_list="1.0 2.0 0.5"
for mufac in $mufac_list; do

#### tc ###

m1=1.0
m2=$mc_under_mt

color="1"
wvfn="compact"
sa=1.0
af=1.0

file="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mc}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mc}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
echo $gfmc --mQ $mc --Rstar $Rstar --mu $mu --mufac $mufac  --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
$python $gfmc --mQ $mc --Rstar $Rstar --mu $mu --mufac $mufac  --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
$python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

#### tb ###

m1=1.0
m2=$mb_under_mt

color="1"
wvfn="compact"
sa=1.0
af=1.0

file="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mb}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mb}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
echo $python $gfmc --mQ $mb --Rstar $Rstar --mu $mu --mufac $mufac  --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
$python $gfmc --mQ $mb --Rstar $Rstar --mu $mu --mufac $mufac  --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
$python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

#### ttcc ###

sa=1.0
color="1"

m1=1.0
m2=$mc_under_mt

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

gfac=1.0

sa=2.0
af=2.0
color="3x3bar"

file="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mc}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mc}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
echo $python $gfmc --mQ $mc --Rstar $Rstar --mu $mu --mufac $mufac --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
$python $gfmc --mQ $mc --Rstar $Rstar --mu $mu --mufac $mufac --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
$python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

#### ttbb ###

sa=1.0
color="1"

m1=1.0
m2=$mb_under_mt

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

gfac=1.0

sa=2.0
af=2.0
color="3x3bar"

file="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mb}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="mrs_data/Hammys_muPrime_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_radial_n1_mQ${mb}_mu${mu}_mufac${mufac}_Rstar${Rstar}_spoila${sa}_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
echo $python $gfmc --mQ $mb --Rstar $Rstar --mu $mu --mufac $mufac  --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
$python $gfmc --mQ $mb --Rstar $Rstar --mu $mu --mufac $mufac  --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "mrs_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
$python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

done

