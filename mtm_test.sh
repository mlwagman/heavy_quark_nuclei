# run GFMC and fit wavefunction based on Hyleraas trial state
nw=1000
nstep=300 
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

wvfn="compact"
sa=1.0
af=1
#L=20
n_x=1.0
n_y=0.0
n_z=0.0

# Loop over the Lattice size
for L in 10 20 30 40 50 60 70 80 90 100; do
    mtm_x_1=$(echo "scale=10; 2 * $pi * $n_x / $L" | bc -l)
    mtm_x_2=$(echo "scale=10; 2 * $pi * $n_x / $L" | bc -l)
    mtm_y_1=$(echo "scale=10; 2 * $pi * $n_y / $L" | bc -l)
    mtm_y_2=$(echo "scale=10; 2 * $pi * $n_y / $L" | bc -l)
    mtm_z_1=$(echo "scale=10; 2 * $pi * $n_z / $L" | bc -l)
    mtm_z_2=$(echo "scale=10; 2 * $pi * $n_z / $L" | bc -l)

    #python3 heavy_quark_nuclei_gfmc_boosted_FV.py --alpha $alpha --log_mu_r 0.0 --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 2 --outdir "mtm_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L $L --spoila $sa --n_skip $nskip --masses $m1 -$m2 --g $g --color $color --mtm_x $mtm_x_1 $mtm_x_2 --mtm_y $mtm_y_1 $mtm_y_2 --mtm_z $mtm_z_1 $mtm_z_2 #--verbose
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

# Loop over the Lattice size
for L in 10 20 30 40 50 60 70 80 90 100; do
    mtm_x_1=$(echo "scale=10; 2 * $pi * $n_x / $L" | bc -l)
    mtm_x_2=$(echo "scale=10; 2 * $pi * $n_x / $L" | bc -l)
    mtm_x_3=$(echo "scale=10; 2 * $pi * $n_x / $L" | bc -l)
    mtm_x_4=$(echo "scale=10; 2 * $pi * $n_x / $L" | bc -l)
    mtm_y_1=$(echo "scale=10; 2 * $pi * $n_y / $L" | bc -l)
    mtm_y_2=$(echo "scale=10; 2 * $pi * $n_y / $L" | bc -l)
    mtm_y_3=$(echo "scale=10; 2 * $pi * $n_y / $L" | bc -l)
    mtm_y_4=$(echo "scale=10; 2 * $pi * $n_y / $L" | bc -l)
    mtm_z_1=$(echo "scale=10; 2 * $pi * $n_z / $L" | bc -l)
    mtm_z_2=$(echo "scale=10; 2 * $pi * $n_z / $L" | bc -l)
    mtm_z_3=$(echo "scale=10; 2 * $pi * $n_z / $L" | bc -l)
    mtm_z_4=$(echo "scale=10; 2 * $pi * $n_z / $L" | bc -l)

    python3 heavy_quark_nuclei_gfmc_boosted_FV.py --alpha $alpha --log_mu_r $lmur --mu $mu --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir "mtm_data/" --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1  -$m2  $m1 -$m2 --g $g --color $color --gfac $gfac --mtm_x $mtm_x_1 $mtm_x_2 $mtm_x_3 $mtm_x_4 --mtm_y $mtm_y_1 $mtm_y_2 $mtm_y_3 $mtm_y_4 --mtm_z $mtm_z_1 $mtm_z_2 $mtm_z_3 $mtm_z_4 #--verbose
done

#don't fit if pdf already exists
for file in mtm_data/Hammys*mtm*.h5
    do
        python3 average_constant_fit.py "--database=${file}" --dtau $dt # --n_tau_tol 100
done