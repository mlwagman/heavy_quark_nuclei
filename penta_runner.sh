#!/usr/bin/env bash
# ------------------------------------------------------------------
# Sweep over meson / baryon references and three pentaquark states
# ------------------------------------------------------------------
#set -euo pipefail          # ←-uncomment when everything is stable

# ─── shared GFMC parameters ────────────────────────────────────────
nw=1000                 # walkers
nstep=600
nskip=100
alpha=0.75
dt=0.02                # dtau in MeV⁻¹
OLO="LO"
outdir="penta_data/"   # keep the trailing slash – no double “//”

code="heavy_quark_nuclei_gfmc_penta.py"
fit="average_constant_fit.py"
mkdir -p "$outdir"

# ─── helper: one GFMC run + automatic plateau fit ──────────────────
run_and_fit () {
    local Ncoord=$1   color=$2   wavefn=$3
    local afac=$4     spoila=$5  g=$6   masses="$7"

    echo "▶  N=$Ncoord  colour=$color  wf=$wavefn  afac=$afac"

    # ---------- RUN GFMC ------------------------------------------------
    python "$code" \
        --alpha "$alpha"          --log_mu_r 0.0    --OLO "$OLO" \
        --n_step "$nstep"         --n_walkers "$nw" --dtau "$dt" \
        --Nc 3                    --nf 4            --N_coord "$Ncoord" \
        --outdir "$outdir"        --wavefunction "$wavefn" \
        --potential full          --Lcut 5          --L 0 \
        --afac "$afac"            --spoila "$spoila" --n_skip "$nskip" \
        --masses $masses          --g "$g"          --color "$color" #--verbose

    # ---------- IDENTIFY THE OUTPUT FILE -------------------------------
    # The Python code prints str(masses) and str(afac), which may contain
    # spaces or switch from “1e10” → “10000000000.0”.  We therefore just
    # look for the ONLY file that matches the rigid prefix / suffix.
    pattern="${outdir}Hammys_${OLO}_dtau${dt}_Nstep${nstep}"\
"_Nwalkers${nw}_Ncoord${Ncoord}_Nc3_nskip${nskip}*"\
"_color_${color}_g${g}.h5"

    file=$(ls $pattern 2>/dev/null | head -n 1)

    if [[ -z "$file" ]]; then
        echo "  ✗  ERROR: no Hammys_*.h5 matching:"
        echo "     $pattern"
    fi

    # ---------- FIT THE PLATEAU ----------------------------------------
    python "$fit" --database="$file" --dtau "$dt"
    echo
}

# ───────────────────────── reference systems ────────────────────────
run_and_fit 2  "1x1"  "compact"   1.0       1.0      0.0   "1.0 -1.0" 
run_and_fit 3  "1x1"  "compact"   1.0       1.0      0.0   "1.0 1.0 1.0" 

# ───────────────────────── pentaquarks ───────────────────────────────
run_and_fit 5  "1x1"  "product"   1e10      1.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 
run_and_fit 5  "1x1"  "product"   5.82843   1.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 
run_and_fit 5  "8x8"  "product"   2.0       2.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 




OLO="NLO"
alpha=0.227325
nstep=1000
dt=0.2   

# ─── helper: one GFMC run + automatic plateau fit ──────────────────
run_and_fit_NLO () {
    local Ncoord=$1   color=$2   wavefn=$3
    local afac=$4     spoila=$5  g=$6   masses="$7"

    echo "▶  N=$Ncoord  colour=$color  wf=$wavefn  afac=$afac"

    # ---------- RUN GFMC ------------------------------------------------
    python "$code" \
        --alpha "$alpha"          --log_mu_r 0.5    --OLO "$OLO" \
        --n_step "$nstep"         --n_walkers "$nw" --dtau "$dt" \
        --Nc 3                    --nf 4            --N_coord "$Ncoord" \
        --outdir "$outdir"        --wavefunction "$wavefn" \
        --potential full          --Lcut 5          --L 0 \
        --afac "$afac"            --spoila "$spoila" --n_skip "$nskip" \
        --masses $masses          --g "$g"          --color "$color" --mu 0.9093

    # ---------- IDENTIFY THE OUTPUT FILE -------------------------------
    # The Python code prints str(masses) and str(afac), which may contain
    # spaces or switch from “1e10” → “10000000000.0”.  We therefore just
    # look for the ONLY file that matches the rigid prefix / suffix.
    pattern="${outdir}Hammys_${OLO}_dtau${dt}_Nstep${nstep}"\
"_Nwalkers${nw}_Ncoord${Ncoord}_Nc3_nskip${nskip}*"\
"_color_${color}_g${g}.h5"

    file=$(ls $pattern 2>/dev/null | head -n 1)

    if [[ -z "$file" ]]; then
        echo "  ✗  ERROR: no Hammys_*.h5 matching:"
        echo "     $pattern"
    fi

    # ---------- FIT THE PLATEAU ----------------------------------------
    python "$fit" --database="$file" --dtau "$dt"
    echo
}


# ───────────────────────── reference systems ────────────────────────
run_and_fit_NLO 2  "1x1"  "compact"   1.0       1.0      0.0   "1.0 -1.0" 
run_and_fit_NLO 3  "1x1"  "compact"   1.0       1.0      0.0   "1.0 1.0 1.0" 

# ───────────────────────── pentaquarks ───────────────────────────────
run_and_fit_NLO 5  "1x1"  "product"   1e10      1.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 
run_and_fit_NLO 5  "1x1"  "product"   5.828   1.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 
run_and_fit_NLO 5  "8x8"  "product"   2.0       2.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 


echo "────────────────────────────────────────────────────────────────"
echo "All GFMC runs finished – plateau energies printed above."


OLO="NLO"
alpha=0.227325
nstep=200
dt=0.2   

# ─── helper: one GFMC run + automatic plateau fit ──────────────────
run_and_fit_NLO () {
    local Ncoord=$1   color=$2   wavefn=$3
    local afac=$4     spoila=$5  g=$6   masses="$7"

    echo "▶  N=$Ncoord  colour=$color  wf=$wavefn  afac=$afac"

    # ---------- RUN GFMC ------------------------------------------------
    python "$code" \
        --alpha "$alpha"          --log_mu_r 0.0    --OLO "$OLO" \
        --n_step "$nstep"         --n_walkers "$nw" --dtau "$dt" \
        --Nc 3                    --nf 4            --N_coord "$Ncoord" \
        --outdir "$outdir"        --wavefunction "$wavefn" \
        --potential full          --Lcut 5          --L 0 \
        --afac "$afac"            --spoila "$spoila" --n_skip "$nskip" \
        --masses $masses          --g "$g"          --color "$color" --mu 0.9093

    # ---------- IDENTIFY THE OUTPUT FILE -------------------------------
    # The Python code prints str(masses) and str(afac), which may contain
    # spaces or switch from “1e10” → “10000000000.0”.  We therefore just
    # look for the ONLY file that matches the rigid prefix / suffix.
    pattern="${outdir}Hammys_${OLO}_dtau${dt}_Nstep${nstep}"\
"_Nwalkers${nw}_Ncoord${Ncoord}_Nc3_nskip${nskip}*"\
"_color_${color}_g${g}.h5"

    file=$(ls $pattern 2>/dev/null | head -n 1)

    if [[ -z "$file" ]]; then
        echo "  ✗  ERROR: no Hammys_*.h5 matching:"
        echo "     $pattern"
    fi

    # ---------- FIT THE PLATEAU ----------------------------------------
    python "$fit" --database="$file" --dtau "$dt"
    echo
}


# ───────────────────────── reference systems ────────────────────────
run_and_fit_NLO 2  "1x1"  "compact"   1.0       1.0      0.0   "1.0 -1.0" 
run_and_fit_NLO 3  "1x1"  "compact"   1.0       1.0      0.0   "1.0 1.0 1.0" 

# ───────────────────────── pentaquarks ───────────────────────────────
run_and_fit_NLO 5  "1x1"  "product"   1e10      1.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 
run_and_fit_NLO 5  "1x1"  "product"   5.828   1.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 
run_and_fit_NLO 5  "8x8"  "product"   2.0       2.0      0.0   "-1.0 1.0 1.0 1.0 1.0" 


echo "────────────────────────────────────────────────────────────────"
echo "All GFMC runs finished – plateau energies printed above."
