# Form factors: what changed (Oct 2026)

Extraction is now korr: gfmc_reduce.py (h5 -> npz with C2(t_i+t_j) and
C_J(t_i,t_j)) then gfmc_korr.py (spectroscopy + matrix_elements, dipole fit).
Lanczos_twoleg.py / prod_extract_lanczos.py are no longer used. They took the
highest Ritz rank, which made the answer depend on n_step, and built the 3pt
Hankel from the symmetric point C_J(t,t) instead of the two-time object.

The MRS generator now stores the two-time object (C3_N_offdiag, Ws2_offdiag)
and runs leg 2 the full n_step. MRS h5 made before commit bfde265 cannot be
re-extracted.

Radius errors: all Q of a dataset share walkers, so F(Q) are strongly
correlated. gfmc_korr.py refits every bootstrap draw (korr uses the same
draws at every Q). The error of a single fit with diagonal errors was 2.2x too
small for mesons and 1.8x for baryons.

Seeds: one seed per dataset, shared by its Q values. Earlier runs used
--seed 0 everywhere, so datasets with similar physics replayed nearly the same
walk and their deviations were correlated.

With both, the ~3-4% low MRS mesons are within errors. Against the references
(closed form, SVM, Numerov), 24 datasets, Coulomb alpha 0.001-0.5:

    Coulomb meson    chi2/dof 0.49
    Coulomb baryon   chi2/dof 0.38
    MRS LO meson     chi2/dof 0.58
    MRS NLO meson    chi2/dof 0.98
    worst pull       -1.4 sigma

Rerun with new seeds, deviation from Numerov (seed 0 in brackets):

    MRS LO meson  alpha 0.3   +1.1%, +3.5%   (-2.6%)
    MRS NLO meson alpha 0.2   +1.7%          (-5.0%)
    MRS NLO meson alpha 0.3   -3.4%          (-3.4%)
