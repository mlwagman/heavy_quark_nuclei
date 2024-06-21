# run GFMC and fit wavefunction based on Hyleraas trial state
nw=500
nstep=0
#nstep=0
nskip=100

alpha=1.5

dt=0.05
#dt=0.04
#dt=0.02



# Hylleraas without spatial symmetrization
sa=1.00033
#af=5.82843
af=6.82843
g=0.0
color="1x1"
python3 heavy_quark_nuclei_deut.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses 1 2 3 1 2 3 --g $g --color $color --verbose 

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]_color_${color}_g${g}.h5"
python average_constant_fit.py "--database=${file}" --dtau $dt
