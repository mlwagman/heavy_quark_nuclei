#!/bin/bash
# one SVM trial per lambda-campaign baryon dataset (100 symmetrized draws = 600 Gaussians)
cd /Users/user/bassi/heavy_quark_nuclei
P=~/.venvs/hqn_korr/bin/python
out=/Users/user/hqn_local/trials.csv; echo "name,alpha,OLO,lam,r2_trial,r2_svm,dev_pct" > $out
cat /Users/user/hqn_local/lam_queue_1.txt /Users/user/hqn_local/lam_queue_2.txt | while IFS='|' read -r NAME SYS REF K ARGS; do
  [ -z "$NAME" ] && continue
  a=$(echo "$ARGS" | sed 's/.*--alpha_s \([0-9.]*\).*/\1/'); O=$(echo "$ARGS" | sed 's/.*--OLO \([A-Z]*\).*/\1/')
  mu=$(echo "$ARGS" | sed 's/.*--mu \([0-9.eE-]*\).*/\1/'); lam=$(echo "$ARGS" | sed 's/.*--mufac \([0-9.]*\).*/\1/')
  row=$(grep "^$a,$O,$lam," paper_final/svm_baryon_lam.csv); r2s=$(echo $row | cut -d, -f6); r2m=$(echo $row | cut -d, -f10)
  asc=$(python3 -c "print(($r2m*16*$a*$a/27)**0.5)")
  r2t=$(nice $P svm_baryon.py --alpha $a --potential mrs --OLO $O --mu $mu --mufac $lam --ascale $asc --nbasis 100 \
        --symmetrize --scut 1e-10 --save_trial /Users/user/hqn_local/trials/$NAME.npz 2>&1 | sed -nE 's/.*<r\^2>_ch = ([0-9.eE+-]+).*/\1/p')
  echo "$NAME,$a,$O,$lam,$r2t,$r2s,$(python3 -c "print(f'{($r2t/$r2s-1)*100:+.3f}')")" >> $out
done
echo BUILD_DONE
