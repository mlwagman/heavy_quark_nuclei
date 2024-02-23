# run GFMC and fit wavefunction based on Hyleraas trial state
nw=10000
nstep=500
nskip=100

#nw=2000
#nstep=100
#nskip=100

#nw=1000
#nstep=1000
#nstep=500
#nstep=20
#nskip=10
#L=100.0
L=0
Lcut=5

alpha=1.0

dt=0.02

g=1.0


m1=1.0

for m2 in 1.0 1.111111111111111 1.25 1.4285714285714286 1.6666666666666667 2.0 2.5 3.333333333333333 5.0 6.666666666666667 10.0 15.0 20.0 30.0 40.0 60.0 80.0 100.0 206.7683 1000.0 1836.15267 10000.0; do 

echo -e "\n\n\n\nstarting m2=$m2 \n\n\n\n"

#wvfn="hylleraas"
wvfn="product"
sa=1.00033
af=5.82843
sf=$af

fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].csv"

if [ ! -f "$fit" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses ${m1} -${m2} ${m1} -${m2} --samefac $sf # --g $g  --eps_fac 20.0  

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt
fi

wvfn="hylleraas"
sa=1.0
sf=1.0
af=12.0
#BO=1.5
BO=1.4

fit="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_BO${BO}_masses[${m1}, -${m2}, ${m1}, -${m2}].csv"

if [ ! -f "$fit" ]; 
then
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses ${m1} -${m2} ${m1} -${m2} --BO_fac $BO # --g $g  --eps_fac 20.0   --samefac $sf

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_BO${BO}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt
fi

done

exit

#bsq = 0.5
#sa=1.00033
#sa=1.00033
#af=5.82843
## Coolidge 1933
m1=1.0
m2=$1
sa=1.866667
af=1.0
sf=`echo "1.4 * $m2" | bc -l`
#sf=10000000000.0


#sf=100.0
#sf=$af

sa=1.0
af=10000000000.0
g=0.0

#wvfn="hylleraas"
#python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses 1, -1, 1, -1

#file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
##python average_constant_fit.py "--database=${file}" --dtau $dt


wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses -${m1} ${m2} -${m1} ${m2} #--verbose  

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[-${m1}, ${m2}, -${m1}, ${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

exit

m1=1.0
m2=2.0

wvfn="product"
#python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
#python average_constant_fit.py "--database=${file}" --dtau $dt


m1=1.0
m2=4.0

wvfn="product"
#python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
#python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=6.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=8.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=10.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=12.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=14.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=16.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=18.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=20.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=24.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=28.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=32.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=36.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt

m1=1.0
m2=40.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $sf --masses ${m1} -${m2} ${m1} -${m2} #--verbose  #

file="data/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[${m1}, -${m2}, ${m1}, -${m2}].h5"
python average_constant_fit.py "--database=${file}" --dtau $dt
