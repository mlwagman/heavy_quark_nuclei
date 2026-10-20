# GFMC matrix elements: audit findings and the korr-based pipeline

Handoff for Ben, 2026-10-20. Covers the `mike_20260824` bundle: what we reproduced, three bugs we
found, the replacement extraction pipeline, and what to expect to change on your end.

Short version: **three bugs, all in the extraction rather than the GFMC.** The GFMC itself came out
clean on every test we threw at it — including a check that its Q = 0 output is exact to all printed
digits.

1. The three-point Hankel is built from the symmetric-point diagonal instead of the two-time
   correlator (§2). Demonstrable with zero statistics. You have independently fixed this on
   `fix/3pt-hankel-offdiag`; see §9 for how the two fixes compare.
2. The Ritz rank `m_max` is bounded by the time extent rather than by the data, with no criterion
   for choosing it (§3). Still open in both pipelines.
3. A single spiking walker out of 500 silently destroys the extraction, and nothing screens for it
   (§3b). This one accounted for the whole of what we had attributed to (2) on the baryon.

With all three handled, every case we can check agrees with its exact reference: MRS LO meson
26.9 ± 2.3 against Numerov 26.89, Nc = 3 baryon 6.61 ± 0.33 against SVM 6.808, and the two Coulomb
mesons against closed form.

Where we corrected ourselves mid-investigation, the note says so — those are the places most worth
your scepticism.

---

## 1. Reproduction of the bundle as shipped

| script | outcome |
|---|---|
| `run/01_check.sh` | OK. `<r^2>` = 17.577 ± 0.649 vs exact 18.75, −6.26% |
| `run/04_smallalpha.sh` | OK. α=0.3 and α=0.001 (dτ=45000) both scale as advertised |
| `run/02_mrs_meson.sh` | **FAILS**, exit 1, 0 of 5 files extracted |
| `run/03_baryon.sh` | OK. `<r^2>` = 60.219 ± 2.117 vs your 59.740, +0.80% |

`02_mrs_meson.sh` dies inside `korr_dev`: `adaptive eps_ZCW = nan`, the ZCW filter then keeps no
states at all (`last iteration with no spurious states = 0`, `Z_final = []`, `E_final = []`,
`N_states = 0`), so `korrelator.final_E[0]` raises. The retry at `final_m=1` fails the same way.
We could not get the quoted 26.89 comparison out of the bundle as delivered.

A practical note on re-running it: `Lanczos_twoleg.py` spent **4.4 hours on a single file** at
`--max_tau_pts 40`. A 200×200 nested bootstrap over a 20×20 Hankel at 20 iterations is just very
expensive; we abandoned the re-run rather than spend ~20 h on five files.

---

## 2. Bug 1 — the three-point Hankel is built from the wrong correlator

`Lanczos_twoleg.py:213`:

```python
C3pt[:, si, ti] = C3_binned[:, si + ti]
```

`C3_N[:, k]` is the **symmetric-point** two-leg correlator `C_J(kΔ, kΔ)` — generator line 906
evolves leg 1 and leg 2 both to `t` — not a one-leg correlator. Its two-point partner
`Ws2_all[k] = C2(2kΔ)` *is* a valid Hankel in step `2Δ` (the `/2.0` at line 173 concedes this), so
the rotators live on that grid and the entry required is `C_J(2sΔ, 2tΔ)`. Spectrally:

```
needed     Σ_fi Z'_f Z_i J_fi exp(−2 E_f sΔ − 2 E_i tΔ)
supplied   Σ_fi Z'_f Z_i J_fi exp(−(E_f + E_i)(s+t)Δ)
```

These agree **only for f = i**. The diagonal spectral terms survive; the f ≠ i interference terms —
exactly what the Ritz projector exists to cancel — do not.

**Why it stayed hidden.** At Q = 0 the number operator is `J_N = Σ_k exp(i q·x_k) = N_coord · 1`,
which commutes with H, so `J_fi ∝ δ_fi`, every off-diagonal term vanishes, and the defect is
*identically* zero. That is why `F(0) = N_coord` always passed. It switches on only at Q ≠ 0 —
precisely where the form-factor shape, and hence `<r^2>`, lives.

**A trap worth knowing:** the legacy path reports `F(0) = 2.000000 ± 0.000000` *exactly*, at every
`m_max`. A perfect charge check is not evidence the extraction is right; Q = 0 is the one place
this bug is structurally invisible.

**Provenance.** The pattern is inherited from `Lanczos_gfmc.py:98` in the project root, where it is
*correct*: there the data is one-leg and `J = H` commutes with `exp(−Hτ)`, so `C_H` genuinely
depends on σ+τ alone. Carrying it over to a non-commuting current is the error.

### Evidence

- **Noiseless.** `gfmc_me_korr.py --self_test` builds a synthetic two-state system with a known
  `J_fi`, zero statistics. With `J_01 = 0.4` the legacy construction returns `2.057915974` against
  an exact `2.0` — **2.90% off with no noise at all**. With `J_01 = 0` (a current that commutes
  with H, i.e. the Q = 0 case) both constructions are exact to 1e-15. That isolates the mechanism
  completely.
- **On your data.** `audit_c3_two_time.py` compares the effective form factor built both ways on
  the same walkers. At Q = 0 the difference is exactly `0.000e+00` at every time lag. At Q ≠ 0 it
  grows monotonically with `|s−t|` — the f ≠ i interference signature — reaching 2.72σ on the MRS
  meson at Q = 0.15 and 1.35σ on the Coulomb meson at Q = 0.2.

### Size, and a warning about its sign

| case | legacy − corrected, in `<r^2>` |
|---|---|
| MRS LO meson, α_s = 0.3 | **+4.1%** (legacy too large) |
| Coulomb meson, α = 0.3, 500 walkers | −3.7% (legacy too small) |
| Coulomb meson, α = 0.3, 1000 walkers | −0.85% |

**The sign is potential-dependent** — it follows the signs of the off-diagonal `J_fi` and the
energy gaps. Do not assume a direction; measure it per case.

### It does not explain the "4.5% low" offset

`run/02_mrs_meson.sh` attributes a 4.5% deficit (25.69 ± 0.37 vs 26.89) to "a projection-time
systematic, still being measured". It is not this bug. Holding the Hankel at your construction and
using modern korr gives **+3.5%**, not −4.5%. See §6 — we could not explain your number, and we
stopped guessing at mechanisms for it.

---

## 3. Bug 2 — the Ritz iteration count is not bounded by the statistics

`Lanczos_twoleg.py`'s own `--max_tau_pts` help states the rule:

> keep `2*N_ritz_max << sqrt(N_samps)`

**Nothing enforces it.** korr picks `m_max` from the number of time slices, `final_m`/`m_window`
then average the *last* few iterations, and those can sit entirely inside the region where the
analysis has already broken down. Neither pipeline enforces it numerically: your `n_bins = 100`
gives `sqrt(N_samps) = 10` against `2*m_max = 24`, and our `n_block = 5` on 1000 walkers gives
`sqrt = 14.1`.

**An attribution we got wrong, and then corrected.** We first caught this on the Nc = 3 baryon,
where `J^(m)` was clean to m = 9, the bootstrap errors went `nan` from m = 10, and the automatic
window averaged m = 12, 13, 14 to return F(0) = 2.9279 against an exact 3. We reported that as a
statistics-driven conditioning limit, and pointed at the weight-growth rate `c` (0.04 for the
Coulomb meson vs 0.27 for the baryon) as the explanation.

**That was wrong.** It was a single spiking walker (§3b). With that one walker removed, `J^(m)` is
flat from m = 1 to the maximum available and *no* breakdown is visible on the baryon at all. The
rank issue below is real and still open, but it is **not** the larger of the three effects, and the
weight-growth-rate story does not hold up: a weight-SNR criterion built on it fails outright (§7,
item 0).

**How much it matters — and a figure we previously overstated.** Scanning `m_max` on the MRS meson
(our pipeline, `--n_block 1`, Numerov 26.89):

| m_max | Q=0 charge check | `<r^2>` (z-exp, correlated) | vs 26.89 |
|---|---|---|---|
| 6 | 0.141% | 27.239 ± 2.624 | +1.30% |
| 10 | 0.145% | 26.851 ± 2.308 | −0.14% |
| 12 | 0.006% | 26.749 ± 2.291 | −0.53% |
| 16 | **0.463%** | 28.261 ± 2.267 | +5.10% |
| 20 | **0.551%** | 29.093 ± 2.267 | +8.19% |

There is a clean **plateau**: `m_max` = 6–12 all agree within 1.3% of the exact value, well inside
the statistical error, and `m_max` ≥ 16 breaks away monotonically. An earlier draft of this note
quoted "~6%" as the rank systematic; that was wrong, because it averaged across the breakdown.
**Inside the plateau the rank spread is ~1%.**

### The fix: a veto, not an objective

1. **`--n_block 1`** — use all walkers as independent samples. Justified because the walker
   autocorrelation is negligible (§5): `N_samps` goes from 100 to 500.
2. **Use the exact Q = 0 charge check as a VETO on `m_max`, and quote the spread over the
   survivors as the rank systematic.**

The distinction matters and we got it wrong twice before arriving at it. At Q = 0,
`J_N = N_coord · 1` has **no off-diagonal elements**, so `J^(m) = N_coord` tests only that
`P^L C2 P^R = 1` — the rotator *normalization*. It is therefore **structurally blind to
excited-state contamination**, which is what sets the error at Q ≠ 0. (Same blindness as bug 1,
for the same reason.) Minimizing it is meaningless: one rule we tried parked `m_max` at the
tolerance edge, and another — reacting to any rise in the deviation — collapsed to `m_max = 3`
while discarding `m_max = 12` at 0.011%, because two entries printing "0.000%" differed in the
16th decimal.

What it *does* detect is numerical breakdown of the pencil, which shows up in the normalization —
empirically with a clean 3–4× step at exactly the `m_max` where `<r^2>` departs (table above). So:
reject what it rejects (`--charge_tol`, default 0.0025), take the largest survivor, and scan the
surviving range to get the rank systematic. `gfmc_me_korr.py --dir` does the veto automatically and
prints the surviving range.

Caveat: the 0.0025 threshold is calibrated on two systems (MRS meson, and the Nc=3 baryon after
spike screening, where every `m_max` ≤ 15 passes at ≤ 0.149%). Re-check it on a new system.

---

## 3b. Bug 3 — a single spiking walker destroys the extraction, and nothing screens for it

Found 2026-10-20, while testing whether a weight-SNR criterion could set `m_max` adaptively (it
cannot — see §7, item 0). The Nc = 3 baryon's noise profile was not a smooth exponential; it was
dominated by **one walker out of 500**:

* the pathology is confined to leg-1 row `i = 0` (mean CV 2.161, against 0.22–0.38 for every other
  row);
* walker 223 carries `w = +760` in cell `(0, 15)` — **24% of the total |weight| over all 500
  walkers**. The same walker is `+1.739` in a clean cell;
* a leverage score `max_ij |w| / median_w |w|` puts it at **170x** with the next-worst at **6x**.
  The meson ensembles top out at 16–37x with smooth tails and no outlier, so this is isolated to
  the baryon, not systemic.

Dropping that one walker:

| baryon, Q = 0, exact F(0) = 3 | all 500 | 499 |
|---|---|---|
| NaN iterations (of 15) | 4 | **0** |
| F(0) | 2.941677 ± 0.107972 | **2.995531 ± 0.005376** |
| E_0 vs SVM −0.267838 | −0.265077 ± 0.004588 | **−0.268157 ± 0.002035 (0.16σ)** |
| `<r^2>` vs SVM 6.808 | 6.465 (−5.04%) | **6.605 ± 0.334 (−2.98%)** |

So **the baryon had no `m_max` problem at all** — `J^(m)` runs clean and flat from m = 1 to the
maximum available once the spike is gone. What looked like a conditioning limit was one walker, and
it also inflated the error bar 20×.

Two things worth knowing:

* **Binning cannot fix this.** Binning preserves the grand mean *exactly*, and the Hankel entries
  **are** sample means. Measured across a 25× range of `n_block` the failure count barely moved
  (4 → 5 NaN). Mitigation has to act on the data.
* **You already had machinery for this and we had dropped it.** `Lanczos_twoleg.py`'s
  `--spike_thresh` interpolates time slices whose CV exceeds a threshold. Our rewrite lost it;
  `gfmc_me_korr.py --spike_leverage 50` is the per-walker replacement (it leaves unaffected walkers
  at a contaminated time alone, where slice interpolation replaces them all). Your
  `fix/3pt-hankel-offdiag` branch does not screen for spikes either.

Because row `i = 0` feeds the Hankel's first row and column, a spike there contaminates **every**
iteration `m`, not only large ones.

## 4. What we changed in your code

Exactly one file: `heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py`. It is a separate commit so
the diff is reviewable on its own.

The Coulomb generator already wrote the correct two-time objects —
`C3_N_offdiag[i,j] = C_J(t_i Δ, t_j Δ)` and `Ws2_offdiag[i,j] = C2((t_i + t_j)Δ)`, lines 848, 908,
1148–1149 — and **nothing in the bundle ever read them**. They were there for "two-sided GEVP" and
never consumed. The MRS generator had no equivalent, because of line 1270:

```python
N_leg2 = max(t, 1)  # leg-2 only needs t steps (not full n_step)
```

Leg 2 was only evolved as far as the diagonal needed, so the off-diagonal entries did not exist to
store. Rather than lengthen every leg-2 run (a 2× cost), we run full length **only for C2 and
J_N** — the two the matrix-element analysis consumes — and leave the other six currents on the
short path:

```
before:  8 × (n_step/2)            = 4 · n_step
after:   6 × (n_step/2) + 2 × n_step = 5 · n_step      ≈ 25% more
```

The diagonal is taken from those same full-length sequences, so `Ws2_all[k] == Ws2_od[k,k]` holds
*exactly* (verified: max difference `0.000e+00` walker-by-walker). Nothing else in the file changed
and the other currents' results are unaffected.

**Caveat:** only `C3_N` has two-time data, in either generator. Every other channel — `C3_S`,
`C3_V0`, `C3_Vi`, `C3_Ai`, `C3_T0i`, `C3_Tij` — is still restricted to the buggy Hankel. If you
need form factors in those channels, the generator needs extending further and a re-run.

---

## 5. The new pipeline

Replaces `prod_extract_lanczos.py → Lanczos_twoleg.py → korr_dev` with modern `korr`
(`~/Dropbox/korr_development/korr`), which is where the method is now maintained.

```
# one file
python3.12 gfmc_me_korr.py --currents_file <currents_3pt_*.h5> --n_block 1

# a whole output directory -> prod_ME.csv, drop-in for prod_extract_lanczos.py
python3.12 gfmc_me_korr.py --dir out/mrs_lo_meson_a0.3 --n_block 1

# then your own fitter, unchanged:
python3 run/fit.py out/mrs_lo_meson_a0.3/prod_ME.csv meson 26.89

# or ours, which adds a free-p dipole and a z-expansion:
python3.12 fit_ff.py --korr_dir out/mrs_lo_meson_a0.3 --form all --t_cut 0.4485 --k_max 3
```

`--dir` writes the same `Q, channel, F, F_err` schema, so `run/fit.py` and everything downstream of
it work untouched. Verified both ways: your unmodified `fit.py` on our output gives 18.572 ± 0.653
on the Coulomb meson and 27.462 ± 0.899 on MRS.

What it does differently: the two-time Hankel; modern korr; `m_max` calibrated on the Q = 0 charge
check; resampling alignment taken from the rotators' fingerprint rather than a shared
`np.random.seed(17)`; and the aligned bootstrap draws saved per Q (`*_korr_ME_boot.npy`) so fits can
use the covariance across Q. It also runs in minutes per file rather than hours, and handles the
MRS correlator that made `korr_dev` return `N_states = 0`.

### Scripts

| file | what it is |
|---|---|
| `gfmc_me_korr.py` | the extraction. `--self_test` proves bug 1 noiselessly; `--hankel legacy` reproduces your construction through the same backend for controlled comparison; `--dir` replaces `prod_extract_lanczos.py` |
| `audit_c3_two_time.py` | the bug-1 audit: diagonal/antidiagonal consistency, σ↔τ orientation, the defect vs Q and vs lag, and the CM-recoil check |
| `fit_ff.py` | charge radius. `--self_test` checks the machinery against the exact Coulomb form factor; fixed-p dipole, free-p dipole, z-expansion; correlated refit over the aligned bootstrap |
| `walker_autocorr.py` | τ_int along the walker index plus a binning plateau |
| `run_mrs_a03.sh`, `run_baryon_a075.sh`, `run_atomic_units.sh` | one Q per invocation, so `xargs -P` parallelises |

### Environment

No single interpreter runs both halves on this machine. GFMC needs `jax`
(`/Users/mwagman/opt/anaconda3/bin/python3`, 3.8.10); korr needs ≥ 3.12
(`/usr/local/bin/python3.12`). The HDF5 file is the interface. Set
`PY=/Users/mwagman/opt/anaconda3/bin/python3` before `run/00_env.sh` so it cannot drift.

---

## 6. Results

All errors below are **correlated** across Q (see §7). References: closed form where it exists,
otherwise `mrs_numerov.py` (2-body MRS, same potential code `exec`'d from the production source) or
`svm_baryon.py` (3-body Coulomb, correlated-Gaussian SVM).

| system | `<r^2>` | reference | dev |
|---|---|---|---|
| Coulomb meson, α = 0.3, 1000 walkers | 18.692 ± 0.975 | 18.75 exact | −0.31% |
| Coulomb meson, α = 3/4 | 3.0966 ± 0.2442 | 3.000 exact | +3.2% (+0.4σ) |
| Nc = 3 baryon, α = 3/4 (spike-screened) | 6.605 ± 0.334 | 6.808 SVM | −3.0% (−0.6σ) |
| MRS LO meson, α_s = 0.3 | 26.9 ± 2.3 (stat) ± 0.3 (rank) | 26.89 Numerov | −0.0% |

Ground-state energies, as an independent check on the spectroscopy that produces the rotators
(these weights carry no trial-energy subtraction, so `E_0 − E_T` is absolute — confirmed on the
Coulomb case where E_0 = −μ(αC_F)²/2 is closed form):

| system | korr | reference |
|---|---|---|
| Coulomb meson, α = 0.3 | −0.040261 ± 0.000390 | −0.0400 exact |
| MRS LO meson, α_s = 0.3 | −0.028154 ± 0.000163 | −0.028033 Numerov |
| Nc = 3 baryon, α = 3/4 (spike-screened) | −0.268157 ± 0.002035 | −0.267838 SVM |

Two shape results worth flagging:

- **The Coulomb meson form factor is an exact dipole, and we confirmed it three ways** at α = 3/4:
  p floated to 2.1831 ± 0.8046; the z-expansion coefficients came out c₁/c₀ = −8.13, c₂/c₀ = +32.48
  against the exact −8, +32 of the fourth-order pole at z = −1; and the p = 2 form describes all
  8 Q points out to Q a₀ = 3.5 (F/F(0) down to 0.31).
- **The baryon is not a dipole.** Free-p gives **p = 2.99 ± 0.93** — a tripole. Forcing p = 2
  biases `<r^2>` *up* by ~4% (6.72 vs 6.49), and χ²/dof is 0.408 for p = 2 against 0.012 for p
  free. Your `run/fit.py` already floats p for baryons; keep it that way. Note also that p = 2 is
  exact only for *pure Coulomb* — for MRS the free-p fit gives p = 1.72 ± 0.67, so the `meson`
  branch's hard-coded `p = 2.0` is an assumption there, not a theorem.

---

## 7. Things worth cross-checking on your end

Roughly in order of how much we think they matter.

0. **A weight-SNR criterion for `m_max` does not work — we tried it and it failed.** The idea is
   attractive: the pencil at iteration `m` reaches two-point index `2m-1` (korr says so itself:
   *"the Hankel matrices of iteration m need C(t) through t=2m-1"*), and GFMC weight variance grows
   exponentially, so `SNR(k) = sqrt(N_samps)/CV(k)` ought to set the cap. Measured over 5 systems
   spanning `lambda` = 0.022-0.080 and `N_samps` = 500/1000, **no threshold separates usable from
   broken iterations**: the broken ones had *higher* median SNR (130.5) than the usable ones (84.4),
   and purity never exceeded 0.83. The reason is §3b — the noise is not smooth exponential growth
   but localized spikes, so there is nothing for an SNR cut to bite on. `mmax_snr_study.py` has the
   harness if you want to re-run it on your own data; the negative result is worth knowing before
   anyone else tries it.

1. **Re-run `02_mrs_meson.sh` through the new pipeline** and see whether you also land near 26.75.
   This is the single most informative check, since your 25.69 is the number we could not explain.
2. **Your production (full-statistics) numbers will move.** Every published `<r^2>` extracted
   through `Lanczos_twoleg.py` at Q ≠ 0 carries both bugs. Expect shifts of order a few percent
   from the Hankel and up to ~6% from `m_max`, in a direction that depends on the potential.
3. **Check `m_max` against the Q = 0 charge check on each of your production runs.** It is free, it
   is exact, and it is the fastest way to find out whether a given run was extracted inside or
   outside the usable Krylov region. We would be surprised if the rank-20 default
   (`--max_tau_pts 40`) is safe for the baryon or for any weakly bound case.
4. **Your quoted errors are ~2–2.6× too small.** Not a bug, a correlation: all Q values share the
   GFMC seed, so the per-Q points are strongly correlated and `run/fit.py`'s
   `absolute_sigma=True` on independent per-Q errors is not valid. The tell is χ²/dof ≈ 0.002–0.04
   in the existing fits. Our `fit_ff.py --korr_dir` refits over aligned bootstrap replicas to get
   this right; ours moved e.g. ±0.09 → ±0.24 on the α = 3/4 meson.
5. **Channels other than `C3_N` are still on the buggy Hankel** — see the caveat in §4.
6. **The baryon production value (59.74)** — worth confirming which extraction produced it. If it
   came through the Lanczos path at rank 20, bug 2 applies; if through the plateau method, it does
   not, but then it carries the plateau systematic instead.
7. **`mrs_numerov.py` box convergence.** Its own comment notes a fixed `rmax = 300` previously
   truncated the wavefunction for a₀ ≳ 10, now scaled with the trial Bohr radius. At α_s = 0.3,
   a₀ = 6.48 so we are inside the safe region, but we did not independently verify convergence by
   varying the box — worth doing before leaning on it at weaker coupling.
8. **The dropped autocorrelation inflation is fine, but only here.** `Lanczos_gfmc.py:126` inflates
   errors by `2·τ_int`; `Lanczos_twoleg.py` dropped that. We measured τ_int ≤ 0.59 along the walker
   index (inflation ≤ 1.09×) and confirmed with a binning plateau flat from bin size 1 to 25, so
   `n_block` is irrelevant at these parameters. It is worth re-measuring if `--n_skip` is ever
   reduced from 100 or resampling is switched on, since either would introduce real correlation.

### Things we checked that came out clean

So you don't re-audit them:

- **The CM recoil cancellation works.** The `conj(r01)` sink-side reweighting (line 855, with the
  comment about the unconjugated version doubling the phase) cancels the recoil completely: 0–0.9%
  of `Q²/2M` left over, against a rate that would have been ~3% per unit τ at Q = 0.35. Good call.
- **The two-time data is self-consistent**: diagonal reduction exact to `0.000e+00`,
  antidiagonal spread ≈ 1–1.6σ (it is a genuine function of the total time only).
- **The GFMC data itself is exact at Q = 0**: `C3_N/C2 per-walker: min = max = 3.0000000000`.
- **The `E_shift` choice is a genuine gauge choice** — it cancels in `P^L C3 P^R` to 3e-11 on real
  data and 1.8e-14 synthetically. Your scale-covariant `extra_decay_rel` fix is right, and matters:
  `c ∝ α²`, so a fixed absolute value really does underflow at weak coupling.
- **The in-place mutation at `Lanczos_twoleg.py:138–140`** (`Ws2_wt = Ws2_all.T` then `*=`) does
  modify `Ws2_all`, but harmlessly, because `shift_vec[0] == 1` leaves the `F_m1` normalization at
  line 205 intact. Worth tidying; not a bug today.

---

## 8. Open, and limits of what we showed

- **Your 25.69 (−4.5%) is unexplained.** We offered the Hankel bug, then the korr_dev backend, and
  retracted both: holding the Hankel at your construction gives +3.5%, and the `m_max` systematic
  alone spans 25.9–27.6, so your value sits inside that band — but the trend is *increasing* with
  `m_max` and your rank-20 setting is higher than our 16, which would push you above, not below. We
  cannot reproduce your pipeline (it fails outright here), so we have no mechanism, only the
  observation that it lies within the extraction-settings spread. Treat "projection-time
  systematic" as unverified rather than established.
- **Reproducing the quoted numbers needs non-default flags.** The table above is at
  `--n_block 1`, and the baryon additionally at `--spike_leverage 50`. The committed defaults
  (`n_block = 5`, no spike screening) give different answers — notably ≈25.8 for MRS. Either pass
  the flags or change the defaults before comparing.
- **The veto threshold rests on two systems.** `--charge_tol 0.0025` cleanly separates plateau from
  breakdown on the MRS meson and the screened baryon, but that is n = 2.
- **Only the baryon has been re-run with spike screening.** The meson and MRS numbers above predate
  it. Their leverage scores are modest (16–37x, smooth tails) so we expect little change, but it
  has not been checked.
- **Everything here is at 500–1000 walkers**, i.e. roughly 2× your production errors. The α = 0.3
  Coulomb case taught us to be careful: at 500 walkers it looked like a −5% systematic, and at 1000
  it was −0.31%. Several of the smaller numbers above may be statistics.
- **The baryon's −5.0% vs SVM** is −1.0σ, so consistent, but it is the largest central deviation in
  the table and worth more statistics.
- We did not extend the audit to NLO, to `--mufac` variation, or to the `qqq_qqq` / tetraquark
  channels.

---

## 9. Reproducing the headline numbers

```sh
# bug 1, noiseless, no data needed (~1 min)
python3.12 gfmc_me_korr.py --self_test

# fit machinery against the exact Coulomb form factor (~10 s)
python3.12 fit_ff.py --self_test

# bug 1 on real data: zero at Q=0, growing with lag at Q>0
python3 audit_c3_two_time.py out/mrs_lo_meson_a0.3/currents_3pt_*Qmag0.0.h5 \
                            out/mrs_lo_meson_a0.3/currents_3pt_*Qmag0.15.h5

# MRS end to end
printf '%s\n' 0.0 0.05 0.1 0.15 0.2 0.3 0.35 0.5 | xargs -P 3 -n 1 ./run_mrs_a03.sh
python3.12 gfmc_me_korr.py --dir mrs_korr/mrs_lo_meson_a0.3 --n_block 1
python3.12 fit_ff.py --korr_dir mrs_korr/mrs_lo_meson_a0.3 --form all \
                     --t_cut 0.448532 --k_max 3 --expect 26.89

# bug 2, if you want to see it fail: drop --n_block 1 and force the cap past the limit
python3.12 gfmc_me_korr.py --currents_file <baryon Q=0 file> --m_max 14   # returns F(0)=2.93
```

For the z-expansion, `t_cut` is the nearest singularity in Q², fixed by the exponential falloff of
the one-body density: if `x_1 − R_cm = s·y` for the relevant Jacobi coordinate and the density falls
as `exp(−2 κ_y y)` with `κ_y = sqrt(2 μ_sep B_sep)`, then `t_cut = (2 κ_y / s)²`. This reproduces
the meson's known `16/a_0²` exactly, which is how we validated it; for the MRS meson it gives
`16|E_0| = 0.4485`, and for the Nc = 3 Coulomb baryon at α = 3/4, `2.4641`. Note the z-coefficients
are *not* O(1) here — the singularity is a pole, not a bounded branch cut, so the exact meson series
is 1, −8, +32, −88, +192. Stability in `k_max` and `t_cut` is the diagnostic, not coefficient size;
`k_max ≥ 3` is needed (validated against the SVM baryon form factor, which it reproduces to 0.0σ).

---

## 9. Comparison with your `fix/3pt-hankel-offdiag` branch

Audited 2026-10-20 against commit `e5145b7` (merge-base `0051df0`), read-only.

**Bug 1: you fixed it, correctly, in two places.** `Lanczos_twoleg.py` subsamples the two-time array
at even indices `[2*si, 2*ti]`, which keeps you self-consistently on the `2Δ` grid — so your `/2` on
the Ritz rate stays correct — and `gfmc_korr.py` works on the step-`Δ` grid. `E_shift` cancels
identically in both. At matched `m_max = 12` our two pipelines agree on the MRS meson to 0.1
(25.78 ± 1.79 vs 25.89 ± 0.88).

We also note the diagnosis predates this branch: `twoleg_gevp.py` on `main` already states that the
old code "fed the DIAGONAL C3(tau_i+tau_j) into the off-diagonal Hankel slots". We rediscovered a
bug you had already found. Two things there may still be worth having: the zero-statistics
isolation, and the observation that Q = 0 makes it *identically* inert (which is why `F(0)` could
never catch it).

One claim worth retracting: `twoleg_gevp.py` says the defect makes the meson radius overshoot to
`a = 8.75` vs analytic `5.0` — a ~200% effect — while `Lanczos_twoleg.py:132` on this branch says
1.35%, and we measure ~4% on MRS. The `8.75` figure looks wrong.

**The one substantive difference: which two-point vector you pair with the 3pt.** You use row 0 of
`Ws2_offdiag` (length `N_tau`); we average its antidiagonals (length `2 N_tau − 1`). Both are
unbiased, but the longer vector doubles the attainable Krylov dimension, and yours caps `m` at
`N_tau/2`. On the Coulomb meson, where the answer is *exactly* 18.75:

| | F(0), exact = 2 | `<r^2>` |
|---|---|---|
| your `gfmc_korr.py`, `m` capped at 6 | 2.00717 | 17.824 ± 0.953 (−4.9%) |
| ours, `m_max = 12` | 1.99529 | 18.522 ± 0.664 (−1.2%) |

You cannot reach the plateau with a 13-entry 2pt vector. The antidiagonal spread on that file is
0.3–1.0%, so averaging it is a legitimate variance reduction rather than a trick.

**Bug 2: not addressed on the branch.** `gfmc_korr.py:23` hardcodes `m_max = 12` with no CLI flag,
no coupling to `N_samps`, and no rank systematic in the output; passing no `config` means korr's
defaults then median over m = 9, 10, 11. On the MRS meson that setting violates the exact Q = 0
identity by 0.79% and sits outside the plateau we measure. Note also that
`radius_with_rank_systematic.py` — which states the problem correctly, "the drift (4-7%) exceeds the
statistical error (~1.5%). Quoting a single m ... hides it" — reads a file only the now-disused
`Lanczos_twoleg.py` produces, so it is orphaned from the live pipeline. **That tool had the right
idea and we would suggest reviving it;** our §3 veto is the complementary half (which ranks to
exclude), not a replacement for quoting the spread.

**Bug 3: not screened for** on your branch either, and `Lanczos_twoleg.py`'s spike interpolation is
applied to `Ws2_all` but not to `C3_N_od_raw`, so if it ever fires the 2pt and the two-time 3pt no
longer describe the same walker ensemble at that slice.

**MRS generator.** You set `N_leg2 = max(n_step, 1)` unconditionally, which your own comment notes
costs ~2x the leg-2 work. Because `_safe_leg2` closes over the single `N_leg2`, all 16 leg-2 runs
lengthen — but only `C3_N` is stored two-time, so the extra cost buys nothing. Ours threads the
length through and runs full-length for C2 and `J_N` only (~25%). Both preserve exact diagonal
consistency and both preserve `conj(r01)`; we verified the other six currents are bitwise unchanged
by the lengthening (identical `dtau`, same `rand_draws` prefix).

**Minor issues found, all cheap to fix:**

* `gfmc_reduce.py:19` — the `'korr' not in x` filter tests the whole path, not the basename, so any
  run directory whose *path* contains "korr" silently finds zero files. Confirmed: our `mrs_korr/`
  directory reduces to "no currents_3pt h5 found". It exits loudly, so nothing is corrupted.
* `gfmc_korr.py:21` — blanket `warnings.filterwarnings('ignore')` suppresses korr's
  `MatrixElementWindowWarning` (which exists to catch a 2pt/3pt window mismatch) along with every
  singular-matrix warning. It would not fire today, but it will hide it later.
* `gfmc_korr.py:36` — `assume_aligned=True` is unnecessary (both rotators come from the same
  `SpectrumResult`, so the fingerprints match by construction) and disables the only guard against
  feeding mismatched ensembles.
* `Lanczos_twoleg.py:273` — silent fallback to the known-wrong construction if `C3_N_offdiag` is
  absent, with no warning; and the two-time branch is gated on `cur_name == 'C3_N'`, so S, V0, Vi,
  Ai, T0i, Tij all still use the wrong Hankel there.
* `reproduce/run/00_env.sh:6` prefers `python3.13` then `python3`; on this machine only
  `python3.12` has `korr`, and the guard warns rather than failing.

**Where we genuinely disagree and neither of us can settle it from the code:** the binning
convention. You fix the bin *count* (`n_bins = 100`), we fix the bin *size* (`n_block`, and the
numbers in this note use 1). Fixing the count holds `N_samps` constant across datasets, which is the
right variable for the `2m << sqrt(N_samps)` rule; fixing the size holds the autocorrelation
argument constant. It interacts with both bug 2 and the error treatment, so it is worth agreeing on
explicitly rather than by default.

Your bootstrap error treatment (`gfmc_korr.py:58-67`, refitting every aligned draw) is the right one
and matches what we converged on independently; our diagonal χ²/dof of 0.011 on the same data is the
symptom you describe in `NOTES.md`.
