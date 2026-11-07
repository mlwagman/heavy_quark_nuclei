# N_d = 5 and 7 dark baryons (colour-free GFMC)

Scripts as run on Symmetry in one directory (`~/hqn/cfree/code`) holding these files plus the repo-root
`heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py`, `mrs_numerov.py` and `afdmc_lib_col/`. Outputs went to
`~/hqn/cfree/{results,claims,runs,logs}` next to it. Paper text: App. "Other numbers of dark colors".

Physics. For one colour-singlet hadron the colour state is an eigenstate of every pair colour operator, so the
generator runs with `--no_color_vectors` (pair factor C_F/(N_d-1) for the baryon, `make_pairwise_singlet_potential`
and `singlet_S` in `afdmc_lib_col/lib_col.py`, same idea as Mike's ae6d179). N_coord = N_d > 3 is allowed in that mode.
Lambda_d/m_psi depends only on N_d alpha_d (one and two loops, n_f = 0), so datasets are matched in N_d alpha_d
(0.003 ... 0.21, i.e. alpha_d <= 0.07 at N_d = 3), with mu_cut = 3.4 Lambda_d(2-loop) as for N_d = 3.

New generator flags (defaults unchanged): `--no_color_vectors`, `--only_N` (leg 2 only for C2 and J_N),
`--sym_only` (leg 2 only up to t: symmetric points C_J(t,t)), `--no_Rs_file`, `--fast_lap` (vectorized gradient
and Laplacian of the compact trial, equal to the loop version, `test_fast_lap.py N`), `--fast_metropolis S`
(independent chains per walker for the compact trial). `mrs_numerov.py --Nc` gives the meson for any N_d.

1. `nd_plan.py`: for each dataset, mu_cut, the VMC-optimal compact trial `prod exp(-r_ij/a0)` (`vmc_trial.py`,
   a0 at the minimum of <H> with the generator's own potential), dtau = 0.0125/|E_T|, Numerov meson radius.
   Writes `datasets.csv` (in the repo as `paper_final/nd_datasets.csv`). Includes four N_d = 3 checks against the
   variational radii and, at N_d alpha_d = 0.15 LO lambda = 1, starts from trials scaled by 0.7 (`_cmp`)
   and 1.4 (`_wide`, added by `rerun_queue.py`).
2. `make_queue.py datasets.csv queue.txt`: one line per (dataset, q, walker split), 8 splits x 2000 walkers,
   800 steps, six q with q sqrt(r2_T) = 0.151 ... 3.78.
3. `nd_node.sbatch` (`NPER`, `QUEUE`): whole-node job, NPER workers claim lines with `mkdir claims/NAME`,
   run the generator with `--wavefunction compact --no_color_vectors --only_N --sym_only --no_Rs_file
   --fast_lap --fast_metropolis 2000`, reduce each run to a small npz (`reduce_sym.py`) and delete the h5.
   Workers keep a heartbeat; `cleanup_claims.py` releases claims of preempted or killed workers.
   Memory: a 7-body worker peaks near 2.5 GB, so NPER = 44 per 80-core node (76 ran out of memory);
   sharedq: `-c 8 --mem=16000 NPER=6`. Each live run holds a ~100 MB h5 in `runs/`.
4. `nd_collect.py results datasets.csv coll`: late-time average (tau >= 0.7 tau_max) of
   C_J(t,t)/(N_d C2(2t)), fit (1 + r2 q^2/6p)^-p to the five lowest q, jackknife over walker blocks with refits.
   Output `coll_results.csv`, `coll_ff.npz`, `coll_curves.npz` (in the repo as `paper_final/nd_*`).
   `nd_report.py` makes quick-look plots.
5. `paper_nd.py paper_final` in the repo root: the trial correction from the three starts, `table_nd.tex`,
   `nd_radius.pdf`, `nd_convergence.pdf`, and every number quoted in the paper text (printed).

Checks: N_d = 3 GFMC vs variational radius -0.36, -0.25, -1.6, -0.17% (0.3, 0.2, 1.6, 0.1 sigma), <H> vs E0
within 0.02%. Three starts on one line: chi2 0.01 (N_d = 5), 0.45 (N_d = 7), remaining trial fraction
rho = 0.165(21) and 0.274(18).
