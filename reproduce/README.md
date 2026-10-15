Dark baryon form factors: GFMC + Lanczos. What changed recently: NOTES.md.

Requires python3 with numpy scipy jax h5py matplotlib tqdm mpmath
timeout-decorator, and korr
(pip install -e korrfitter_v0). Set PY=/path/to/python if needed.

Pipeline, at the repo root:
    heavy_quark_nuclei_gfmc_boosted_FV_currents.py       GFMC, Coulomb, one h5 per Q
    heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py   GFMC, MRS, one h5 per Q
    gfmc_reduce.py    h5 of one dataset -> npz with the two-time objects
    gfmc_korr.py      npz -> J_00(Q) with korr, dipole fit -> <r^2>

Run from the repo root:
    bash reproduce/run/01_check.sh        15 min   Coulomb meson alpha=0.3, ref 18.75
    bash reproduce/run/02_mrs_meson.sh    30 min   MRS LO meson alpha_s=0.3, ref 26.89
    bash reproduce/run/03_baryon.sh       2 h      MRS LO baryon alpha_s=0.3
    bash reproduce/run/04_smallalpha.sh   30 min   Coulomb meson alpha=0.3 and 0.001, ref alpha^2 <r^2> = 27/16
    reproduce/run/05_examples.sh          menu of options, run blocks one at a time
    bash reproduce/run/06_crosschecks.sh  references without GFMC (Numerov, SVM)

Each prints F(0), <r^2>, the reference and the pull. Output in reproduce/out/.
500 walkers here; production used 1000 (baryon) and 2000 (meson).
Times are for one core. The h5 files are large (~150 MB per dataset); the npz
keeps what the extraction needs, so the h5 can be deleted after gfmc_reduce.py.

By hand:
    for Q in 0.0 0.02 0.05 0.1 0.2 0.35 0.5; do
      python heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py <args> --Q_mag $Q --seed 7 --outdir run/
    done
    python gfmc_reduce.py run/ run.npz
    python gfmc_korr.py run.npz meson [ref]

Seeds: one seed for all Q of a dataset, a different seed for every dataset.
Q only enters the current, so a shared seed gives every Q the same walk, which
makes the slope less noisy. Two datasets with a shared seed and similar
physics replay nearly the same walk and are not independent.

Main options:
    --alpha (Coulomb) / --alpha_s (MRS)     --N_coord 2 meson, 3 baryon
    --OLO LO|NLO                            --n_walkers, --seed
    --n_step, --dtau_iMev, --count_skip     --Q_mag
    --spoila       trial size               --mufac    MRS scale, lambda
    --mu           mu_cut, leave at 2.0

When changing alpha, scale dtau as 1/alpha^2 and Q as alpha.
