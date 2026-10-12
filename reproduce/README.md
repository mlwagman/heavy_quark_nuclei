Dark baryon form factors: GFMC + Lanczos.

Runnable entry points for the pipeline at the repo root. These replace the
zip that was circulated earlier -- `git pull` now gets you the current code.

Requires python3.13 or python3 with numpy scipy jax h5py matplotlib tqdm.
Set PY=/path/to/python if the interpreter is not found.

Pipeline (all at the repo root):
    heavy_quark_nuclei_gfmc_boosted_FV_currents.py       Coulomb
    heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py   MRS
    prod_extract_lanczos.py -> Lanczos_twoleg.py -> korr_dev/korr/
    reproduce/run/fit.py                                 dipole fit to <r^2>

Run it, from anywhere in the repo:
    . reproduce/run/00_env.sh
    bash reproduce/run/01_check.sh        10 min    Coulomb meson, reference 18.75
    bash reproduce/reproduce/run/04_smallalpha.sh   10 min    alpha 0.3 and 0.001, both give 1.6875
    bash reproduce/run/02_mrs_meson.sh    30 min    MRS LO meson,  reference 26.89
    bash reproduce/run/03_baryon.sh       1 h       MRS LO baryon

Each prints <r^2>, the reference value and the pull. Output in out/.
500 walkers, so errors are about twice the production ones.

reproduce/run/05_examples.sh is a menu of the options.
reproduce/run/06_crosschecks.sh runs the independent solvers (Numerov, SVM).


## Ritz-rank systematic  (new, read this)

The extraction gives F(q^2) as a function of Ritz rank m, and F drifts with m.
The highest rank is m_max = N_tau/2, set by how many tau points are stored, so
taking the last rank makes the answer depend on n_step. This was the cause of
several apparent discrepancies.

    n_step 60   -> 12 tau pts -> m_max 6
    n_step 200  -> 40 tau pts -> m_max 18

Every run now also writes the full curve:

    *_twoleg_Lanczos_vsm.csv    current, m, F, F_err

and the main CSV gains F_stab, F_stab_err, m_stab, stab_window, m_max.
To report the radius with the rank systematic:

    $PY reproduce/run/radius_with_rank_systematic.py <dir>
    -> <r^2> = value (stat) (rank)

Drift is about 1% for baryons and 5% for mesons.

Validation on the Coulomb family, where the answer is exact at every coupling
(alpha^2<r^2> = 27/16 for the meson, 3.8532 for the baryon):

    state    alpha    alpha^2<r^2>   (stat)   (rank)    dev
    meson    0.5         1.7468      0.0280   0.0260   +3.5%
    meson    0.03        1.6500      0.0262   0.0530   -2.2%
    meson    0.001       1.6419      0.0232   0.0406   -2.7%
    baryon   0.5         3.8807      0.0548   0.0255   +0.7%
    baryon   0.03        3.7641      0.0551   0.0154   -2.3%
    baryon   0.001       3.9443      0.0703   0.0188   +2.4%

All within 1.6 sigma over a factor 500 in alpha.


## Earlier fixes, still in

nf never reached the NLO coupling: Alpha_s2Loop had no dark-sector branch and
ran nf=4 instead of nf=0. All MRS NLO before 2026-08-22 is invalid.

Lanczos died below alpha ~ 0.05: E_shift used a fixed absolute 0.015 while the
physical rate scales as alpha^2. Now 0.1875*|c|. reproduce/run/04_smallalpha.sh tests it.

optimize_spoila was unusable below alpha_s ~ 0.1: <H> printed to 5 decimals
while the energies scale as alpha^2, so the variational minimum was rounding
noise. Now 12 decimals.

Main options:
    --alpha (Coulomb) / --alpha_s (MRS)     --N_coord 2 meson, 3 baryon
    --OLO LO|NLO                            --n_walkers
    --n_step, --dtau_iMev                   --Q_mag
    --spoila       trial size               --mufac    MRS scale, lambda
    --mu           mu_cut, leave at 2.0
    prod_extract_lanczos.py --estimator final|stab

When changing alpha, scale dtau as 1/alpha^2 and Q as alpha.
