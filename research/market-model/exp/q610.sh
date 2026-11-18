#!/bin/bash
# fresh test period: autumn 2025, published residual load known (also for D+1), frozen spec vs segment trend
cd "$(dirname "$0")"
[ -f ../out/inputs_aut25.npz ] || ../venv/bin/python prep_inputs.py aut25 2025-09-02 2025-11-28 2 none 2>&1 | tail -1
../venv/bin/python - <<'PY'
import numpy as np, live_common as lc, dense_common as dc
I = np.load(f"{lc.SP}/out/inputs_aut25.npz"); F = lc.load(); rl = dc.tso(F["DE"].index)["rl"].to_numpy(float)
d1 = np.stack([rl[int(c) - 96:int(c)] for c in I["cuts"]]); np.savez_compressed(f"{lc.SP}/out/d1_known_aut25.npz", cutoffs=I["cutoffs"], rl_d1=d1.astype(np.float32), rl_d1_true=d1.astype(np.float32)); print("d1 file", d1.shape, np.isnan(d1).sum())
PY
./srun.sh h25 aut25 DRIFT=s BTOP=45 D1FILE=d1_known_aut25
./srun.sh p25 aut25 DRIFT=sc BTOP=45 D1FILE=d1_known_aut25
for r in h25 p25; do ../venv/bin/python dyn2.py aut25 $r known x+xh 2>&1 | grep -v -i warn; ../venv/bin/python dyn4.py aut25 $r known 2>&1 | grep -v -i warn; done
echo QUEUE610 DONE
