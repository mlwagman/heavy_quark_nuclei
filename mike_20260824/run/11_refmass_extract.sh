#!/bin/bash
# Form factors and charge radii from the alpha_s(mQ)=0.1 reference-mass runs, with the
# consistent-Gram GEVP extraction (gfmc_me_gevp.py).
#
# Validated recipe, from the two configs that have exact answers:
#   --spike_leverage 100   drop walkers whose max_ij |w|/median_w |w| exceeds 100.  On the
#                          fixed-order meson this is 2 of 5000 walkers and moves <r^2> from
#                          -13% to -2.7% while shrinking errors 3x.  On the MRS meson it finds
#                          nothing to drop (worst leverage 8x) because that run's dtau is 7x
#                          finer -- the tails are a dtau effect.
#   --gevp_cut 1e-2        the MEASURED Gram is not positive definite (negative eigenvalues to
#                          -6e-2 of the largest plus a ~1e-4 noise floor); 1e-10 makes the GEVP
#                          explode.  Insensitive across 1e-2..1e-1.
#   m from the <r^2> plateau, NOT from the Q=0 charge check, which is now an identity
#   (F(0) = N exactly at every rank) and was structurally blind anyway.
#
# References, all <r^2>_ch (one-body about the CM):
#   FO  meson  LO   168.75     exact, 27/(16 alpha^2)
#   FO  baryon LO   343.62     svm_baryon.py --alpha 0.1
#   MRS meson  LO    56.69     mrs_numerov.py at mu_cut, lambda=1   (= Ben's r2_meson)
#   MRS meson  NLO   52.07     mrs_numerov.py at mu_cut, lambda=1   (= Ben's r2_meson)
#   MRS baryon LO   114.251    SVM;  Ben's GFMC 116.744 +/- 2.16, fw 115.639 +/- 1.51
#   MRS baryon NLO  103.142    SVM;  Ben's GFMC 105.539 +/- 2.06, fw 102.199 +/- 1.27
set -eu
. "$(dirname "$0")/00_env.sh"

PY312=${PY312:-/usr/local/bin/python3.12}
GEVP="$ROOT/../gfmc_me_gevp.py"
O="$OUT/refmass_a0.1"
A="$O/analysis"
mkdir -p "$A"
MS=${MS:-"2 3 4 5 6 8 10 12 14"}

SPEC=$(cat <<'EOF'
fo_meson_lo|meson|168.75|exact 27/(16 a^2)|-
fo_baryon_lo|baryon|343.62|svm_baryon.py|-
mrs_meson_lo|meson|56.69|Numerov|Ben has no GFMC meson
mrs_meson_nlo|meson|52.07|Numerov|Ben has no GFMC meson
mrs_baryon_lo|baryon|114.251|SVM|Ben 116.744 +/- 2.16
mrs_baryon_nlo|baryon|103.142|SVM|Ben 105.539 +/- 2.06
EOF
)

while IFS='|' read -r name system ref what bens; do
  [ -z "$name" ] && continue
  d="$O/$name"
  n=$(ls "$d"/currents_3pt_*.h5 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -eq 0 ]; then echo "-- $name: no h5 yet, skipping"; continue; fi
  echo
  echo "================================================================================"
  echo "$name   ($n Q points)   reference $ref [$what]${bens:+   $bens}"
  echo "================================================================================"
  "$PY312" -I "$GEVP" --dir "$d" --system "$system" --ref "$ref" \
      --spike_leverage 100 --gevp_cut 1e-2 --n_boot 200 --m $MS \
      2>&1 | tee "$A/${name}_gevp.txt" | grep -vE "^\s*$"
done <<< "$SPEC"

echo
echo "Reference mass: alpha_s^{Nf=0}(mQ) = 0.1  =>  mQ = 302.4714 Lambda (1-loop),"
echo "                                             mQ =  93.9223 Lambda (2-loop)."
echo "Runs are at mQ = 1 with mu_cut = 3.4 Lambda_2loop = 0.0362001; divide lengths by"
echo "the ratio above to express them in units of Lambda."
