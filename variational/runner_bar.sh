#!/bin/bash

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=LO --alpha=0.2 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=LO --alpha=1 &

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=LO --alpha=0.2 &

wait

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=LO --alpha=0.6 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=LO --alpha=1 &

wait

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NLO --alpha=0.2 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NLO --alpha=1 &

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NLO --alpha=0.2 &

wait

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NLO --alpha=0.6 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NLO --alpha=1 &

wait

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=mNLO --alpha=0.2 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=mNLO --alpha=1 &

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=mNLO --alpha=0.2 &

wait

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=mNLO --alpha=0.6 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=mNLO --alpha=1 &

wait

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NNLO --alpha=0.2 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NNLO --alpha=1 &

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NNLO --alpha=0.2 &

wait

python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NNLO --alpha=0.6 &

#python3 heavy_quark_nuclei_variational_Bar.py --log10_learn_rate=2 --c_loss=1  --v_loss=1 --N_exp=3 --OLO=NNLO --alpha=1 &

wait
