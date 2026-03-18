# run GFMC and fit wavefunction based on Hyleraas trial state
nw=10000
nstep=1000
nskip=100


##################################### ttcc ############################

alpha=0.253206
mu=2.00644
dt=0.05

sa=1.0
g=1.0
color="1"

m1=1.0
m2=104.442


wvfn="compact"
sa=1.0
af=1.0

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

# molecular spatial wavefunction 
wvfn="product"
dt=0.01

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

alpha=0.20027
mu=4.01287
dt=0.05

sa=1.0
g=1.0
color="1"

m1=1.0
m2=104.442

wvfn="compact"
sa=1.0
af=1.0

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

# molecular spatial wavefunction 
wvfn="product"
dt=0.01

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

alpha=0.355891
mu=1.00322
dt=0.02

sa=1.0
g=1.0
color="1"

m1=1.0
m2=104.442

wvfn="compact"
sa=1.0
af=1.0

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

# molecular spatial wavefunction 
wvfn="product"
dt=0.002

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

##################################### ttbb ############################

alpha=0.19474
mu=1.51522
dt=0.1

sa=1.0
g=1.0
color="1"

m1=1.0
m2=35.4866

wvfn="compact"
sa=1.0
af=1.0

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

alpha=0.165129
mu=3.03045
dt=0.1

sa=1.0
g=1.0
color="1"

m1=1.0
m2=35.4866

wvfn="compact"
sa=1.0
af=1.0

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

alpha=0.242995
mu=0.757611
dt=0.1

sa=1.0
g=1.0
color="1"

m1=1.0
m2=35.4866

wvfn="compact"
sa=1.0
af=1.0

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord2_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 --g $g --color $color
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

g=1.0
gfac=1.0

sa=2.0
af=2.0
lmur=0.0
color="3x3bar"

file="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.h5"
fit="data/Hammys_NLO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m1, -$m2, -$m2]_color_${color}_g${g}.csv"
echo looking for $fit

if [ ! -f "$file" ]; then
python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "NLO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m1  -$m2 -$m2 --g $g --color $color --gfac $gfac
fi

if [ ! -f "$fit" ]; then
python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
fi
