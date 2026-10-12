Extracted form factors for the 3pt Hankel test (see ../../BUGNOTE_3pt_hankel.md).
Coulomb alpha=0.3, n_step 200, 2000 walkers. Exact <r^2> = 18.750.

  coul_a0.3_spoila1.0_*   exact trial     -> fix changes nothing (17.707 vs 17.706)
  bugtest_spoila0.7_*     30% mismatch    -> fix moves 1.35% (12.837 vs 12.664)

Fit any of them with:
    python3.13 ../run/fit.py <csv> meson 18.75
