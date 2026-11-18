#!/bin/bash
# fresh winter 2024/25 refit with usable inputs; regime-memory layer with the shrinkage of the unit-test bar (LAM1=3) next to LAM1=10
cd "$(dirname "$0")"
rm -rf ../out/struct_h24; ./srun.sh h24 win24 DRIFT=s BTOP=45 D1FILE=d1_known_win24
../venv/bin/python dyn2.py win24 h24 known x+xh 2>&1 | grep -v -i warn | grep -v "first day"
echo "--- LAM1=10"; ../venv/bin/python dyn6.py win24 h24 known 2>&1 | grep -v -i warn; cp ../out/structfc_h24_known_dyn6.npz ../out/structfc_h24_known_dyn6L10.npz
for pr in "winter hw2 known,ts" "autumn ha known,ts" "summer hs2 known,ts" "aut25 h25 known"; do set -- $pr; for a in ${3//,/ }; do cp ../out/structfc_$2_${a}_dyn6.npz ../out/structfc_$2_${a}_dyn6L10.npz; done; done
echo "--- LAM1=3"; LAM1=3 ../venv/bin/python dyn6.py winter hw2 known unit 2>&1 | grep -v -i warn | head -3
for pr in "win24 h24 known" "winter hw2 known,ts" "autumn ha known,ts" "summer hs2 known,ts" "aut25 h25 known"; do set -- $pr; LAM1=3 ../venv/bin/python dyn6.py $1 $2 $3 2>&1 | grep -v -i warn; for a in ${3//,/ }; do cp ../out/structfc_$2_${a}_dyn6.npz ../out/structfc_$2_${a}_dyn6L3.npz; done; done
echo QUEUE720 DONE
