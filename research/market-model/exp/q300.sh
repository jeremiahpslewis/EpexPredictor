#!/bin/bash
../venv/bin/python prep_inputs.py march 2026-03-02 2026-03-31 1 none 2>&1 | tail -1
../venv/bin/python mkgen.py march
./srun.sh m1 march DRIFT=s
SETS=BC ../venv/bin/python ladder.py march 2>&1 | tail -1
GENFILE=genarm_march_known ../venv/bin/python live_test3_season.py two_march_known 2026-03-02 2026-03-31 1 D:pes:T 2>&1 | tail -1
cp ../out/two_march_known.npz ../out/two_march_own.npz
echo finished
