#!/bin/bash
# Extract form factors and charge radii from the alpha_s=0.1 reference-mass runs.
#
# Runs Ben's extraction (gfmc_reduce.py + gfmc_korr.py from his
# fix/3pt-hankel-offdiag branch) so our numbers are directly comparable to his
# paper_final/gfmc_radii.csv, and our gfmc_me_korr.py as an independent check.
#
# korr needs python >= 3.12, which has no jax, so this is a different
# interpreter from the generators (see split-interpreters note in the handoff).
set -eu
. "$(dirname "$0")/00_env.sh"

PY312=${PY312:-/usr/local/bin/python3.12}
BEN=${BEN:?set BEN to the directory holding gfmc_reduce.py and gfmc_korr.py}
O="$OUT/refmass_a0.1"
A="$O/analysis"
mkdir -p "$A"

# config | system | reference <r^2>_ch | what the reference is
SPEC=$(cat <<'EOF'
fo_meson_lo|meson|168.75|exact, 27/(16 alpha^2)
fo_baryon_lo|baryon|343.62|svm_baryon.py --alpha 0.1
mrs_meson_lo|meson|56.69|mrs_numerov.py, mu_cut, lambda=1
mrs_meson_nlo|meson|52.07|mrs_numerov.py, mu_cut, lambda=1
mrs_baryon_lo|baryon|114.251|SVM (Ben's GFMC: 116.744 +/- 2.16)
mrs_baryon_nlo|baryon|103.142|SVM (Ben's GFMC: 105.539 +/- 2.06)
EOF
)

echo "=== reduce + extract, alpha_s(mQ)=0.1, Nf=0, mu_cut=3.4 Lambda_2loop ==="
while IFS='|' read -r name system ref what; do
  [ -z "$name" ] && continue
  d="$O/$name"
  n=$(ls "$d"/currents_3pt_*.h5 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -eq 0 ]; then echo "-- $name: no h5 yet, skipping"; continue; fi
  echo "-- $name  ($n Q points)   ref $ref  [$what]"
  "$PY312" -I "$BEN/gfmc_reduce.py" "$d" "$A/$name.npz" 100
  "$PY312" -I "$BEN/gfmc_korr.py" "$A/$name.npz" "$system" "$ref" --out "$A/${name}_ff.csv"
done <<< "$SPEC"

echo
echo "=== summary ==="
for f in "$A"/*_ff.csv; do
  [ -e "$f" ] || continue
  printf '%-16s %s\n' "$(basename "$f" _ff.csv)" \
    "$(awk -F, '$1=="r2"{r=$2} $1=="r2_err"{e=$2} $1=="p"{p=$2} $1=="F0"{f0=$2} $1=="ref"{ref=$2} \
        END{printf "F(0)=%.4f  <r^2>=%.4g +/- %.3g  p=%.2f  ref=%.4g  (%+.2f%%)", f0, r, e, p, ref, 100*(r/ref-1)}' "$f")"
done
