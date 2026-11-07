# Campaign scripts for the MRS baryon results

Scripts as run on Symmetry, where `~/hqn/code` holds these plus the repo-root Python files, and
`~/hqn/results`, `~/hqn/logs`, `~/hqn/trials` hold the outputs. Paths inside are that layout.

Couplings used in the paper: alpha_d <= 0.07. mu_cut = 3.4 Lambda_d (two-loop), lambda = mufac in {1/2, 1, 2}.

1. `build_lam_queue.py`: mu_cut and the exact meson radii, writes `lam_queue_{1,2}.txt`.
2. `build_trials.sh`: one variational trial per baryon dataset (100 symmetrized draws = 600 Gaussians).
3. `svm_node.sbatch` with `QFILE=svm_queue_{1,2}.txt`: GFMC with those trials, `gfmc_reduce.py`, `gfmc_korr.py`, `fw.py`.
   `lam_node.sbatch` is the earlier run with compact trials, kept because it defines the queue format.
4. `paper_data.sbatch`: `paper_solver.py` (exact radii, F(q), <K>, with s = lambda at NLO), then `svm_cache.py`.
5. `coulval_cache.py`: re-extracts the scaled Coulomb family (npz from the mac-mini, `reptest/coulval_npz`).
6. `paper_korr.py paper_final` in the repo root: all tables and figures, from the committed `paper_final` inputs.

Checks: `blockcheck.py` (error against walker block size), `flat_inr.py` (drift of C3/C2 in tau),
`echeck.py` (GFMC <H> against the variational E0).

N_d = 5 and 7 baryons (colour-free GFMC, matched N_d alpha_d): see `nd/README.md`.
