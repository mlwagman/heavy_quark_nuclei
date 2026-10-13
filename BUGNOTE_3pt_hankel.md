# 3pt Hankel indexing: confirmed, measured, and not the main systematic

## The defect (found by Mike's agent, verified independently)

The 3pt correlator is a two-leg object,

    C_J(s,t) = <psi| e^{-Hs} J e^{-Ht} |psi>
             = sum_{f,i} Z*_f Z_i <f|J|i> e^{-E_f s} e^{-E_i t}

`Lanczos_twoleg.py` built its Hankel as `C3pt[s,t] = C3_N[s+t]`. But the stored
`C3_N[k]` is the SYMMETRIC point `C_J(t_k,t_k)` -- the generator puts both legs
at the same time -- while the Ritz basis (step 2*dtau, since each stored index
advances both legs) needs `C_J(2s dt, 2t dt)`:

    needed:    sum Z*_f Z_i J_fi e^{-2E_f s dt - 2E_i t dt}
    supplied:  sum Z*_f Z_i J_fi e^{-(E_f+E_i)(s+t) dt}

These agree only for f=i. The diagonal terms are right; the f!=i interference
terms -- exactly what the Ritz projector exists to cancel -- are wrong.

At Q=0 the number operator is N_coord * identity, which commutes with H, so
J_fi ~ delta_fi and the error is exactly inert. That is why `F(0) = N_coord`
has always been machine-precision exact while the Q!=0 shape was off.

The pattern was inherited from `Lanczos_gfmc.py`, where it is correct: there the
current is H itself, which commutes with e^{-H tau}, so C_H genuinely depends
only on s+t. Carrying it to a non-commuting current is the error.

## Measured size: ~1%, and it does not explain the 4-6% deficit

Testing on Coulomb alpha=0.3, n_step 200, where the exact answer is 18.750:

| trial | extraction | <r^2> | dev |
|---|---|---|---|
| spoila 1.0 (exact trial) | diagonal (buggy) | 17.707(300) | -5.56% |
| spoila 1.0 (exact trial) | offdiag (fixed)  | 17.706(288) | -5.57% |
| spoila 0.7 (30% mismatch)| diagonal (buggy) | 12.837(704) | -31.5% |
| spoila 0.7 (30% mismatch)| offdiag (fixed)  | 12.664(929) | -32.5% |

With an exact trial the fix changes nothing, as it must: Z_i = delta_i0, so there
are no interference terms to corrupt. With a deliberately mismatched trial the
fix moves the answer by 1.35% -- and AWAY from exact.

So the defect is real and worth fixing, but it is a ~1% effect, not the
explanation for the 4-6% MRS meson deficit.

## Correction to what was previously circulated

`run/02_mrs_meson.sh` in the August pack described the ~4.5% deficit as a
"projection-time systematic, still being measured". That reading was wrong and
is retracted. The evidence against it:

- Coulomb at tau=30, 100 and 200 is -0.80%, -5.56%, -0.32%: not monotonic in
  tau, so not a projection-time effect.
- What looked like tau dependence was the extraction taking the HIGHEST Ritz
  rank, whose ceiling m_max = N_tau/2 is set by n_step. Longer runs stored more
  tau points, reached higher m, and drifted further. Fixed separately; the
  radius is now quoted as value(stat)(rank).
- The residual after that is ~4%, and this note's test shows the Hankel defect
  accounts for ~1% of it.

## What the test did turn up

The same table shows a 30% trial mismatch moving the extracted radius by 31%
(17.707 -> 12.837), on an unchanged Hamiltonian with an unchanged exact answer.
Extraction sensitivity to trial quality dominates everything else we have
measured. That is now the leading candidate for the MRS deficit, and it is
measurable with a spoila scan rather than new production.

## Status of the data

- Coulomb generator stored `C3_N_offdiag` all along (unused). Those runs can be
  corrected by re-extraction alone.
- The MRS generator never stored it, and could not have: it ran leg 2 only to
  `N_leg2 = max(t,1)`, i.e. the symmetric point. Fixed here by running leg 2 the
  full `n_step` (~2x the leg-2 work). Existing MRS h5 cannot be corrected by
  re-extraction; they need re-simulating.
- `Lanczos_twoleg.py` now prefers the two-time object and falls back to the old
  path when `C3_N_offdiag` is absent, so old files still extract.

## Reproducing

    bash bugtest_mismatched.sh     # ~1.5 h, 7 Q points
    # then compare prod_ME_BUGGY.csv and prod_ME_FIXED.csv against 18.750
