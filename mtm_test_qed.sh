# run GFMC and fit wavefunction based on Hyleraas trial state
nw=100
nstep=5
nskip=100

# Function to calculate pi using bc
calculate_pi() {
    echo "scale=10; 4*a(1)" | bc -l
}

# Store the value of pi in a variable
pi=$(calculate_pi)

# Print the value of pi
echo "The value of pi is: $pi"

alpha=1.0
mu=1.0
dt=0.2


sa=1.0
g=0.0

m1=1.0
m2=1.0

wvfn="compact"
sa=1.0
af=1
#L=20

# Define the set of tuples for n1
n1_values="(0,0,0) (0,0,1) (0,1,1) (1,1,1) (0,0,2) (0,1,2) (0,2,2)"

# Loop over the Lattice size
for L in 10000000; do
    # Loop over the n1 tuples
    for n1 in $n1_values; do
        # Parse the n1 tuple into its components
        n_x=$(echo $n1 | cut -d',' -f1 | tr -d '()')
        n_y=$(echo $n1 | cut -d',' -f2 | tr -d '()')
        n_z=$(echo $n1 | cut -d',' -f3 | tr -d '()')

        mtm_x_1=$n_x
        mtm_x_2=-$n_x
        mtm_y_1=$n_y
        mtm_y_2=-$n_y
        mtm_z_1=$n_z
        mtm_z_2=-$n_z

        # Run the Python script with the current parameters
        #python heavy_quark_nuclei_boosted_FV_hulthen.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 2 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L $L --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --mtm_x $mtm_x_1 $mtm_x_2 --mtm_y $mtm_y_1 $mtm_y_2 --mtm_z $mtm_z_1 $mtm_z_2 --verbose
        #python heavy_quark_nuclei_gfmc_boosted_FV_qed.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 2 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L $L --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --mtm_x $mtm_x_1 $mtm_x_2 --mtm_y $mtm_y_1 $mtm_y_2 --mtm_z $mtm_z_1 $mtm_z_2 --verbose
        #python heavy_quark_nuclei_gfmc_boosted_FV_qed.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 2 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --mtm_x $mtm_x_1 $mtm_x_2 --mtm_y $mtm_y_1 $mtm_y_2 --mtm_z $mtm_z_1 $mtm_z_2 --verbose
        python heavy_quark_nuclei_boosted_FV_hulthen.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 2 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --mtm_x $mtm_x_1 $mtm_x_2 --mtm_y $mtm_y_1 $mtm_y_2 --mtm_z $mtm_z_1 $mtm_z_2 --verbose
        python heavy_quark_nuclei_boosted_FV_hulthen.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 2 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L $L --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --mtm_x $mtm_x_1 $mtm_x_2 --mtm_y $mtm_y_1 $mtm_y_2 --mtm_z $mtm_z_1 $mtm_z_2 --verbose

    done
done

# molecular spatial wavefunction 
wvfn="product"

# attractive diquark cluster

g=1.0
gfac=1.0

sa=1.0
af=1e8
lmur=0.0
color="1x1"

n1_values="(0,0,0) (0,0,1/2) (0,1/2,1/2)"
n2_values="(0,0,0) (0,0,1/2) (0,1/2,1/2)"

# Loop over the Lattice size
for L in 50 100 200 400 800; do

    for n1 in $n1_values; do
        n_x=$(echo $n1 | cut -d',' -f1 | tr -d '()')
        n_y=$(echo $n1 | cut -d',' -f2 | tr -d '()')
        n_z=$(echo $n1 | cut -d',' -f3 | tr -d '()')
        mtm_x_1=$n_x
        mtm_x_2=$n_x
        mtm_x_3=-$n_x
        mtm_x_4=-$n_x
        mtm_y_1=$n_y
        mtm_y_2=$n_y
        mtm_y_3=-$n_y
        mtm_y_4=-$n_y
        mtm_z_1=$n_z
        mtm_z_2=$n_z
        mtm_z_3=-$n_z
        mtm_z_4=-$n_z

        #python3 heavy_quark_nuclei_gfmc_boosted_FV_qed.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L $L --afac $af --spoila $sa  --n_skip $nskip --masses $m1  -$m2  $m1 -$m2 --g $g --gfac $gfac --mtm_x $mtm_x_1 $mtm_x_2 $mtm_x_3 $mtm_x_4 --mtm_y $mtm_y_1 $mtm_y_2 $mtm_y_3 $mtm_y_4 --mtm_z $mtm_z_1 $mtm_z_2 $mtm_z_3 $mtm_z_4 #--verbose
    done
done

#don't fit if pdf already exists
for file in mtm_data_qed/Hammys*mtm*.h5
    do
        python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
    done
done

throw(17)

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
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"



wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --samefac $af #--verbose  #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"



#bsq = 0
sa=1.65789
af=1.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

#bsq = 0.25
sa=1.08047
af=3.0

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

#bsq = 0.75
sa=0.998975
af=13.9282

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

#bsq = 1.0
sa=1.0
af=1000000000000

wvfn="product"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"

wvfn="hylleraas"
python3 heavy_quark_nuclei_gfmc_FV_qed.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 1 --nf 0 --N_coord 4 --outdir "mtm_data_qed/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip #--verbose #--masses [1, -1, 1, -1]

file="mtm_data_qed/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc1_nskip${nskip}_Nf0_alpha${alpha}_spoila${sa}_spoilaket1_spoilfhwf_spoilS1_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[1, -1, 1, -1].h5"
python average_constant_fit.py "--database=${file}"
