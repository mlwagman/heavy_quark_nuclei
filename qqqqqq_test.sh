# Define parameters
nw=200 
nstep=100 
nskip=100
alpha=1.5
dt=0.02
wvfn="product"

# Define the mass configurations and fermion symmetries
mass_values=("1.0 1.0 2.0 1.0 2.0 2.0" "1.0 2.0 2.0 1.0 2.0 2.0" "1.0 2.0 3.0 1.0 2.0 3.0" "1.0 1.0 1.0 1.0 1.0 1.0")
ferm_symmetries=("a" "s")

# Loop over mass configurations
for masses in "${mass_values[@]}"; do
    echo "masses = $masses"
    # Loop over fermion symmetries
    for ferm_symm in "${ferm_symmetries[@]}"; do
        
        # Define other variables and their combinations
        sa_values=(1.00033 2 1.0 6.0)
        af_values=(6.82843 2 5.82843 0.5)
        colors=("1x1" "AAA" "SAA" "SSS")

        for i in {0..3}; do
            sa=${sa_values[i]}
            af=${af_values[i]}
            color=${colors[i]}
            
            # Run the first Python script
            python3 heavy_quark_nuclei_deut.py --alpha $alpha --ferm_symm $ferm_symm --log_mu_r 0.5 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 6 --outdir "data_deut/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $masses --g 0.0 --color $color #--verbose 

            formatted_masses=$(echo $masses | sed 's/ /, /g')
            file="data_deut/Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord6_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.5_wavefunction_${wvfn}_potential_full_afac${af}_masses[${formatted_masses}]_color_${color}_g0.0_ferm_symm${ferm_symm}.h5"
            # Run the second Python script
            python3 average_constant_fit.py "--database=${file}" --dtau $dt
        done
    done
done
