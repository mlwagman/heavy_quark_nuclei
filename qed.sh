# run GFMC and fit wavefunction based on Hyleraas trial state
nw=10000
nstep=500
nskip=100

#alpha=0.007297352569
alpha=1.0
#alpha=0.1

dt=0.02
#dt=1.0


#bsq = 0.5
sa=1.00033
af=5.82843


wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

exit

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $af #--verbose  #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

exit

#bsq = 0
sa=1.65789
af=1.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

#bsq = 0.25
sa=1.08047
af=3.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

#bsq = 0.75
sa=0.998975
af=13.9282

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

#bsq = 1.0
sa=1.0
af=1000000000000

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"
