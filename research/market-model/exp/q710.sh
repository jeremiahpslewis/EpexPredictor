#!/bin/bash
# regime-memory layer on the existing fits; fresh winter 2024/25 (published residual load known incl. D+1), frozen spec
cd "$(dirname "$0")"
for pr in "winter hw2 known,ts" "autumn ha known,ts" "summer hs2 known,ts" "aut25 h25 known"; do set -- $pr; ../venv/bin/python dyn6.py $1 $2 $3 2>&1 | grep -v -i warn; done
[ -f ../out/inputs_win24.npz ] || ../venv/bin/python prep_inputs.py win24 2024-12-02 2025-02-27 3 none 2>&1 | tail -1
../venv/bin/python - <<'PY'
import numpy as np, live_common as lc, dense_common as dc
I = np.load(f"{lc.SP}/out/inputs_win24.npz"); F = lc.load(); rl = dc.tso(F["DE"].index)["rl"].to_numpy(float)
d1 = np.stack([rl[int(c) - 96:int(c)] for c in I["cuts"]]); np.savez_compressed(f"{lc.SP}/out/d1_known_win24.npz", cutoffs=I["cutoffs"], rl_d1=d1.astype(np.float32), rl_d1_true=d1.astype(np.float32)); print("d1 file", d1.shape, int(np.isnan(d1).sum()))
PY
./srun.sh h24 win24 DRIFT=s BTOP=45 D1FILE=d1_known_win24
../venv/bin/python dyn2.py win24 h24 known x+xh 2>&1 | grep -v -i warn | grep -v "first day"
../venv/bin/python dyn6.py win24 h24 known 2>&1 | grep -v -i warn
echo QUEUE710 DONE
