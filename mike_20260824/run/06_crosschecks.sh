#!/bin/bash
# Cross-checks. Independent solvers, no GFMC, for the meson numbers only.
# There is no 3-body solver for the MRS potential, so the baryon has none.
set -eu; . "$(dirname "$0")/00_env.sh"; cd "$CODE"
echo "2-body MRS, Numerov:"
for A in 0.5 0.3 0.2 0.1 0.05 0.02; do
  for O in LO NLO; do
    printf "  alpha_s=%-6s %-4s " $A $O
    $PY mrs_numerov.py --alpha_s $A --OLO $O --mu 2.0 --mufac 1.0 2>/dev/null \
      | grep -E "charge radius|Lambda_dark =" | tr '\n' ' '; echo
  done
done
echo
echo "3-body Coulomb, SVM:"
for A in 0.2 0.3 0.5; do $PY svm_baryon.py --alpha $A 2>/dev/null | grep -E "E0|r\^2"; done
