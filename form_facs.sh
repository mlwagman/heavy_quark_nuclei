# run GFMC and fit wavefunction based on Hyleraas trial state
nw=1000
nstep=3
nskip=10

# Function to calculate pi using bc
calculate_pi() {
    echo "scale=10; 4*a(1)" | bc -l
}

# Store the value of pi in a variable
pi=$(calculate_pi)

# Print the value of pi
echo "The value of pi is: $pi"

alpha=0.75
mu=1.0
dt=0.3


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
#for Q in 0.0 0.01 0.1 1.0 10.0; do
for Q in 0.0 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01; do

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
        #
        python3 heavy_quark_nuclei_gfmc_boosted_FV.py --Q $Q --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "mtm_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --color $color --mtm_x $mtm_x_1 --mtm_y $mtm_y_1 --mtm_z $mtm_z_1 --verbose
    done
done



