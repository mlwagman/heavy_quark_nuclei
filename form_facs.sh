# run GFMC and fit wavefunction based on Hyleraas trial state
python=/opt/homebrew/Cellar/jupyterlab/4.2.1/libexec/bin/python

nw=5000
nstep=1000
nskip=10

alpha=0.75
mu=1.0
dt=0.1


sa=1.0
g=0.0
color="1"

m1=1.0
m2=1.0
m3=1.0

wvfn="compact"
sa=1.0
af=1
#L=20
n_x=0
n_y=0
n_z=1

#n1_values="(1,0,0) (0 1 0) (0,0,1) (0,1,1) (1 1 0) (1,1,1)"
n1_values="(1,0,0)"
#change the L/change alpha small
# Loop over the Lattice size
for Q in 0.0 0.002 0.004 0.006 0.008 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1; do
#for Q in 0.6; do
    for n1 in $n1_values; do
        # Parse the n1 tuple into its components
        n_x=$(echo $n1 | cut -d',' -f1 | tr -d '()')
        n_y=$(echo $n1 | cut -d',' -f2 | tr -d '()')
        n_z=$(echo $n1 | cut -d',' -f3 | tr -d '()')

        mtm_x_1=$n_x
        #mtm_x_2=-$n_x
        mtm_y_1=$n_y
        #mtm_y_2=-$n_y
        mtm_z_1=$n_z
        #mtm_z_2=-$n_z

        #python heavy_quark_nuclei_gfmc_boosted_FV.py --Q $Q --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 3 --outdir "mtm_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa --n_skip $nskip --masses $m1 $m2 $m3 --g $g --color $color --mtm_x $mtm_x_1 --mtm_y $mtm_y_1 --mtm_z $mtm_z_1 --verbose
        
        file="mtm_data/Hammys_LO_dtau0.1_Nstep1000_Nwalkers5000_Ncoord2_Nc3_Nf4_alpha0.75_spoila1.0_wavefunction_compact_potential_full_afac1_masses[${m1}, -${m2}]_mtm[${mtm_x_1}.0][${mtm_y_1}.0][${mtm_z_1}.0]_color_1_Q${Q}.h5"

        $python heavy_quark_nuclei_gfmc_boosted_FV.py --Q $Q --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "mtm_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --color $color --mtm_x $mtm_x_1 --mtm_y $mtm_y_1 --mtm_z $mtm_z_1 #--verbose

        $python Lanczos_gfmc.py --database="${file}" --dtau=0.1 --n_skip=50
        
    done
done

#!/bin/bash
# Loop over all files whose names begin with "currents" (adjust the path if needed)
#!/bin/bash
# Loop over all files whose names begin with "currents" (adjust the path if needed)
for file in mtm_data/currents*.h5; do
    echo "Processing file: $file"
    # Use h5ls with -d to list only datasets.
    # Then, use grep to only keep lines that start with a letter (dataset names),
    # and awk to extract the first token.
    datasets=$(h5ls -d "$file" | grep -E '^[A-Za-z]' | awk '{print $1}')
    for ds in $datasets; do
        echo "   Running average_constant_fit.py for dataset: $ds"
        python average_constant_fit.py --database "$file" --dataset "$ds"
    done
done





