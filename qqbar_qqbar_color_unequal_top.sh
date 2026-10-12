# run GFMC and fit wavefunction based on Hyleraas trial state
nw=10000
nstep=1000
nskip=100

alpha=0.75
mu=1.0
dt=0.001

#### ttcc ###

sa=1.0
g=0.0
color="1"

m1=1.0
m2=110.598

wvfn="compact"
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color --verbose
file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100


# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

#### ttbb ###

sa=1.0
g=0.0
color="1"

m1=1.0
m2=36.2149

wvfn="compact"
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color --verbose
file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

