# run GFMC and fit wavefunction based on Hyleraas trial state
nw=2
nstep=25
nskip=10

alpha=0.75
mu=1.0
dt=0.01

#### ttcc ###

sa=1.0
g=0.0
color="1"

m1=1.0
m2=1.0

wvfn="compact"
sa=1.0
af=1
python3 heavy_quark_nuclei_gfmc_boosted_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color  --verbose --mtm_x 1 1

#file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
#python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

g=1.0
gfac=1.0

sa=1.0
af=1000000000000000.0
lmur=0.0
color="1x1"

python3 heavy_quark_nuclei_gfmc_boosted_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1  -$m2  $m1 -$m2 --g $g --color $color --gfac $gfac --verbose --mtm_x 1 1 -1 -1

exit

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac --verbose

#file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
#python average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100

