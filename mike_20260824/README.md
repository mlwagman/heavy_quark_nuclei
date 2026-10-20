Dark baryon form factors: GFMC + Lanczos.

Requires python3.13 or python3 with numpy scipy jax h5py matplotlib tqdm.
Set PY=/path/to/python if the interpreter is not found.

The pipeline:

    heavy_quark_nuclei_gfmc_boosted_FV_currents.py       Coulomb potential
    heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py   MRS potential
    prod_extract_lanczos.py -> Lanczos_twoleg.py -> korr_dev/korr/
    run/fit.py                                           dipole fit to <r^2>

Run it:

    . run/00_env.sh
    bash run/01_check.sh        10 min    Coulomb meson, reference 18.75
    bash run/04_smallalpha.sh   10 min    alpha 0.3 and 0.001, both give 1.6875
    bash run/02_mrs_meson.sh    30 min    MRS LO meson,  reference 26.89
    bash run/03_baryon.sh       1 h       MRS LO baryon, we get 59.74

Each prints <r^2>, the reference value and the pull. Output in out/.
500 walkers, so errors are about twice the production ones.

run/05_examples.sh is a menu of the options. Run individual blocks, not the
whole file.

Main options:
    --alpha (Coulomb) / --alpha_s (MRS)     --N_coord 2 meson, 3 baryon
    --OLO LO|NLO                            --n_walkers
    --n_step, --dtau_iMev                   --Q_mag
    --spoila       trial size               --mufac    MRS scale, lambda
    --mu           mu_cut, leave at 2.0

prod_extract_lanczos.py <dir> --max_tau_pts 40 sets the Ritz rank to 20.
The built-in default is 10, which leaves contamination for diffuse states.

When changing alpha, scale dtau as 1/alpha^2 and Q as alpha. Otherwise the
runs are not the same calculation and cannot be compared.

Cross-checks. Not part of the pipeline, included so the reference values
above are not circular:

    run/06_crosschecks.sh   mrs_numerov.py, 2-body MRS by Numerov
                            svm_baryon.py, 3-body Coulomb by SVM
                            No 3-body solver exists for the MRS potential,
                            which is why the baryon needs GFMC.
    prod_extract.py         plateau extraction, cross-check of the Lanczos step
