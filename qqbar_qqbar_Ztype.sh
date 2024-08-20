# run GFMC and fit wavefunction based on Hyleraas trial state
nw=20000
nstep=1200
nskip=100

alpha=0.75

#write a for loop over mbig values

for mbig in 1.0 40.0; do
    echo "mbig = $mbig"
    m1=1.0
    m2=$mbig

    dt=0.1

    dt=$(echo "scale=10; $dt / $mbig" | bc)
    # Trim trailing zeros using sed
    dt=$(printf "%.10g" $dt)
    echo "dt = $dt"

    # T-type t t bbar bbar and we want to check Z-type t b tbar bbar
    datadir="dataZtype/"

    # molecular spatial wavefunction
    wvfn="product"

    #python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir $datadir --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 $m1 -$m2 --g $g --color $color ##--verbose

    # check uncorrelated product is exact
    sa=1.0
    af=10000000.0
    g=0.0
    color="1x1"
    #python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir $datadir --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m1 $m2 -$m2 --g $g --color $color #--verbose


    file="${datadir}Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m1, $m2, -$m2]_color_${color}_g${g}.h5"
    #python3 average_constant_fit.py "--database=${file}" --dtau $dt --output "dataZtype"

    # Hylleraas wavefunction including symmetrization
    sa=1.00033
    af=5.82843
    g=1.0
    color="1x1"
    #python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir $datadir --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 $m1 -$m2 --g $g --color $color ##--verbose

    file="${datadir}Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2, $m1, -$m2]_color_${color}_g${g}.h5"
    #python3 average_constant_fit.py "--database=${file}" --dtau $dt --output "dataZtype"


    # Hylleraas without spatial symmetrization
    sa=1.00033
    af=5.82843
    g=0.0
    color="1x1"
    #python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir $datadir --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 -$m2 $m1 -$m2 --g $g --color $color ##--verbose

    file="${datadir}Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, -$m2, $m1, -$m2]_color_${color}_g${g}.h5"
    #python3 average_constant_fit.py "--database=${file}" --dtau $dt --output "dataZtype"

    # attractive diquark cluster
    sa=2.00066
    af=2.02843
    g=0.0
    color="3x3bar"
    dt=$(echo "scale=10; $dt / 3.5" | bc)
    dt=$(printf "%.10g" $dt)
    echo "dt = $dt"
    python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir $datadir --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m2 -$m1 -$m2 --g $g --color $color --verbose

    file="${datadir}Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m2, -$m1, -$m2]_color_${color}_g${g}.h5"
    python3 average_constant_fit.py "--database=${file}" --dtau $dt --output "dataZtype"

    # repulsive diquark anti-cluster
    sa=6.0
    af=0.5
    g=0.0
    color="6x6bar"
    #python3 heavy_quark_nuclei_gfmc_FV.py --alpha $alpha --log_mu_r 0.0 --OLO "LO" --n_step $nstep --n_walkers $nw --dtau $dt --Nc 3 --nf 4 --N_coord 4 --outdir $datadir --wavefunction $wvfn --potential "full" --Lcut 5 --L 0 --afac $af --spoila $sa  --n_skip $nskip --masses $m1 $m2 -$m1 -$m2 --g $g --color $color ##--verbose

    file="${datadir}Hammys_LO_dtau${dt}_Nstep${nstep}_Nwalkers${nw}_Ncoord4_Nc3_nskip${nskip}_Nf4_alpha${alpha}_spoila${sa}_log_mu_r0.0_wavefunction_${wvfn}_potential_full_afac${af}_masses[$m1, $m2, -$m1, -$m2]_color_${color}_g${g}.h5"
    #python3 average_constant_fit.py "--database=${file}" --dtau $dt --output "dataZtype"



done