"""
Lanczos/Prony analysis for two-leg GFMC 3pt data.

Two bugs in the original version:
  1. E_shift = sum(abs(masses)) = 2 (rest mass). WRONG.
     The correct E_shift must make the correlator *decay*; it should be
     c/2 + epsilon where c = d(log <Ws2>)/dtau ~0.08. Using E_shift=2
     applied exp(-2*2*tau); at tau=5 that is exp(-20)~2e-9, killing the
     signal in one step.
  2. All 101 tau points were fed to Korrelator, including the weight-collapse
     region (tau>30) where CV>2 and N_eff<135. Korrelator needs the clean
     multi-exponential region only.

Fix:
  - Measure c = d(log <Ws2>)/dtau from the first few stored taus.
  - Set E_shift_correct = c/2 + extra, where extra sets the decay rate of
    the shifted correlator (~0.015 per tau-unit is good).
  - Truncate to tau_max where CV = std(Ws2)/mean(Ws2) < CV_thresh (default 1.5).
  - Run Korrelator on the (now decaying) truncated shifted correlator.
  - Extract ground-state form factor via Ritz projectors.

Output CSV: current, F_m1, F_final, F_err  (same format as before)

Usage:
    python Lanczos_twoleg.py --currents_file <currents_3pt_*.h5> \
        [--n_block 5] [--n_boot 200] [--thresh 0.05] [--cv_thresh 1.5]
"""
import numpy as np
import h5py
import sys
import argparse
import csv

sys.path.insert(0, './korr_dev')
from korr.drivers.korrelator import korrelator
from korr.bootstrap import make_bs_tensor, make_nbs_tensor
from korr.outlier import batched_median, inner_median, make_boot_ci

parser = argparse.ArgumentParser()
parser.add_argument('--currents_file', type=str, required=True)
parser.add_argument('--n_block',   type=int,   default=5)
parser.add_argument('--n_boot',    type=int,   default=200)
parser.add_argument('--thresh',    type=float, default=0.05)
parser.add_argument('--final_m',   type=int,   default=None)
parser.add_argument('--cv_thresh', type=float, default=1.5,
                    help='max CV=std/mean of Ws2 for usable tau points')
parser.add_argument('--extra_decay', type=float, default=0.015,
                    help='extra decay rate added on top of measured c/2')
parser.add_argument('--max_tau_pts', type=int, default=10,
                    help='max usable tau points fed to Korrelator; keep 2*N_ritz_max << sqrt(N_samps)')
parser.add_argument('--cv_consec', type=int, default=3,
                    help='require CV > cv_thresh for this many consecutive pts before cutting')
parser.add_argument('--spike_thresh', type=float, default=1.5,
                    help='CV above this is a single-step spike: interpolate from neighbors')
# legacy arg, ignored
parser.add_argument('--E_shift', type=float, default=None)
args = parser.parse_args()

# ── Load ────────────────────────────────────────────────────────────────────
f = h5py.File(args.currents_file, 'r')
Ws2_all   = np.real(np.array(f['Ws2_all']))   # (N_tau, N_walkers)
tau_values = np.array(f['tau_values'])
dtau_stored = float(tau_values[1] - tau_values[0])
N_tau_full, N_walkers = Ws2_all.shape
print(f"Loaded: N_tau={N_tau_full}, N_walkers={N_walkers}, dtau_stored={dtau_stored:.2f}")

# ── Measure growth rate c from early tau ────────────────────────────────────
# Use first 10 stored points (or up to tau=50) for a stable fit
n_fit = min(11, N_tau_full)
W_mean_full = Ws2_all.mean(axis=1)            # (N_tau,)
log_W = np.log(np.maximum(W_mean_full[:n_fit], 1e-30))
c = np.polyfit(tau_values[:n_fit], log_W, 1)[0]  # growth rate per tau unit
E_shift = c / 2.0 + args.extra_decay
print(f"Ws2 growth rate c = {c:.5f}  =>  E_shift = c/2 + {args.extra_decay} = {E_shift:.5f}")

# ── Find usable tau window (CV < cv_thresh) ──────────────────────────────────
W_std_full = Ws2_all.std(axis=1)
CV_full = W_std_full / np.maximum(W_mean_full, 1e-30)
tau_cut_idx = N_tau_full
consec = 0
for i in range(1, N_tau_full):
    if CV_full[i] > args.cv_thresh:
        consec += 1
        if consec >= args.cv_consec:
            tau_cut_idx = i - consec + 1   # first bad point in the run
            break
    else:
        consec = 0
# ensure at least 5 points (need Hankel size >= 2)
tau_cut_idx = max(5, tau_cut_idx)
# optional cap (default 200 = effectively uncapped; set lower via --max_tau_pts)
if args.max_tau_pts > 0:
    tau_cut_idx = min(tau_cut_idx, args.max_tau_pts)

tau_values  = tau_values[:tau_cut_idx]
Ws2_all     = Ws2_all[:tau_cut_idx, :]
N_tau = tau_cut_idx
print(f"CV threshold={args.cv_thresh}: using tau=0..{tau_values[-1]:.1f} "
      f"({N_tau} points, CV_last={CV_full[N_tau-1]:.3f})")

# ── Interpolate isolated CV spikes to preserve equal tau spacing ──────────────
# A spike is a single point where CV > spike_thresh but neighbors are fine.
# Replace with linear interpolation so the Hankel structure stays valid.
CV_window = CV_full[:tau_cut_idx]
n_interp = 0
for i in range(1, N_tau - 1):
    if CV_window[i] > args.spike_thresh:
        Ws2_all[i, :] = 0.5 * (Ws2_all[i-1, :] + Ws2_all[i+1, :])
        n_interp += 1
        print(f"  interpolated spike at tau={tau_values[i]:.2f} (CV={CV_window[i]:.3f})")
if n_interp:
    print(f"  {n_interp} spike(s) interpolated; Hankel structure preserved")

# ── Build shifted, binned 2pt correlator ─────────────────────────────────────
def bin_data(data, N_bin):
    n = data.shape[0] // N_bin
    return data[:n*N_bin].reshape(n, N_bin, *data.shape[1:]).mean(axis=1)

# Transpose to (N_walkers, N_tau) then bin over walkers
Ws2_wt = Ws2_all.T                              # (N_walkers, N_tau)
shift_vec = np.exp(-2 * E_shift * tau_values)   # (N_tau,)
Ws2_wt *= shift_vec[np.newaxis, :]              # apply shift

# truncate to even length for Hankel
N_tau_even = (N_tau // 2) * 2
if N_tau_even < 4:
    print("ERROR: fewer than 4 usable tau points — cannot run Korrelator")
    sys.exit(1)
Ws2_wt     = Ws2_wt[:, :N_tau_even]
tau_values = tau_values[:N_tau_even]

Ct = bin_data(Ws2_wt, args.n_block)  # (N_samps, N_tau_even)
N_samps, Nt = Ct.shape
print(f"Binned Ct shape: {Ct.shape} (Hankel size {Nt//2}x{Nt//2})")

# ── Run Korrelator on 2pt ────────────────────────────────────────────────────
basename = args.currents_file[:-3]
print("\n=== Korrelator on shifted Ws2 ===")
for _fm in [args.final_m, 1]:
    try:
        korr = korrelator(np.real(Ct), N_outer=args.n_boot, N_inner=args.n_boot,
                          store_P=True, thresh=args.thresh,
                          database=basename + '_twoleg_korr', final_m=_fm)
        break
    except (ValueError, IndexError) as _e:
        print(f"  Korrelator failed (final_m={_fm}): {_e!s:.80}; retrying with final_m=1")
        if _fm == 1:
            raise

# Ritz rates are in units of (1/dtau_stored); physical energy:
# shifted rate = 2*(E_n - E_T + E_shift)  [two-leg, factor 2]
# => E_n_phys = (ritz_rate/dtau / 2) - E_shift + E_T
# We don't know E_T exactly, but for the form factor it cancels in the ratio.
final_ritz = korr.final_E[0] / dtau_stored
final_E_rel = (final_ritz / 2.0) - E_shift   # E_0 - E_T
print(f"Final E (relative to trial) = {final_E_rel:.6f}")

# ── Extract form factors for each current ───────────────────────────────────
currents = ['C3_N', 'C3_S', 'C3_V0', 'C3_Vi', 'C3_Ai', 'C3_T0i', 'C3_Tij']

np.random.seed(17)
bs_tensor  = make_bs_tensor(N_samps=N_samps, N_boots=args.n_boot)
nbs_tensor = make_nbs_tensor(N_samps=N_samps, N_outer=args.n_boot, N_inner=args.n_boot)

results = {}
for cur_name in currents:
    if cur_name not in f:
        continue
    C3_raw = np.real(np.array(f[cur_name]))   # (N_walkers, N_tau_full) or (..., 3)
    is_vector = (C3_raw.ndim == 3)
    n_comp = C3_raw.shape[2] if is_vector else 1

    for comp in range(n_comp):
        if is_vector:
            C3_slice = C3_raw[:, :N_tau_even, comp]
            label = f"{cur_name}_{['x','y','z'][comp]}"
        else:
            C3_slice = C3_raw[:, :N_tau_even]
            label = cur_name

        print(f"\n=== {label} ===")
        C3_shifted = C3_slice * shift_vec[:N_tau_even][np.newaxis, :]
        C3_binned  = bin_data(C3_shifted, args.n_block)   # (N_samps, N_tau_even)

        # F_m1: VMC (tau=0) form factor
        F_m1_raw = C3_raw[:, 0, comp] if is_vector else C3_raw[:, 0]
        W_m1_raw = Ws2_all[0, :]          # tau=0 weights (all 1.0 since normalized)
        F_m1 = float(np.sum(F_m1_raw * W_m1_raw) / np.sum(W_m1_raw)) if W_m1_raw.sum() > 0 \
               else float(np.mean(F_m1_raw))

        # Build Hankel 3pt matrix
        C3pt = np.zeros((N_samps, Nt//2, Nt//2))
        for ti in range(Nt//2):
            for si in range(Nt//2):
                C3pt[:, si, ti] = C3_binned[:, si + ti]

        bs_C3pt  = np.einsum('bn,n...->b...', bs_tensor,  C3pt)
        nbs_C3pt = np.einsum('oin,n...->oi...', nbs_tensor, C3pt)

        med_F     = np.zeros(Nt // 2)
        med_F_err = np.zeros(Nt // 2)
        for m in range(1, Nt // 2 + 1):
            if m not in korr.bs_Pl:
                med_F[m-1] = np.nan
                med_F_err[m-1] = np.nan
                continue
            bs_P  = korr.bs_Pl[m][:, 0, :, 0]
            nbs_P = korr.nbs_Pl[m][:, :, 0, :, 0]
            bs_F  = np.real(np.einsum('bst,bs,bt->b',
                                      bs_C3pt[:, :m, :m], bs_P, np.conj(bs_P)))
            med_F[m-1]     = batched_median(bs_F, thresh=args.thresh)
            nbs_F = np.real(np.einsum('bist,bis,bit->bi',
                                      nbs_C3pt[:, :, :m, :m], nbs_P, np.conj(nbs_P)))
            med_F_err[m-1] = make_boot_ci(inner_median(nbs_F, thresh=args.thresh))

        print(f"  F vs m: {np.round(med_F, 5)}")
        print(f"  F_m1={F_m1:.6f}  F_final={med_F[-1]:.6f} ± {med_F_err[-1]:.6f}")
        results[label] = {'F_m1': F_m1, 'F_vs_m': med_F, 'F_vs_m_err': med_F_err}

f.close()

print(f"\nwriting {basename}_twoleg_Lanczos.csv")
with open(basename + '_twoleg_Lanczos.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['current', 'F_m1', 'F_final', 'F_err'])
    writer.writeheader()
    for label, res in results.items():
        writer.writerow({'current': label,
                         'F_m1':    res['F_m1'],
                         'F_final': res['F_vs_m'][-1],
                         'F_err':   res['F_vs_m_err'][-1]})
print("Done.")
