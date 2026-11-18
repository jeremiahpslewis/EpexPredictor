#!/bin/bash
cd "$(dirname "$0")"; L=../out/q_fresh.log; export HF_HUB_DISABLE_XET=1
../venv/bin/python prep_inputs.py winterf 2025-12-03 2026-02-28 3 none 2>&1 | tail -1 >> $L
../venv/bin/python prep_inputs.py summerf 2026-04-02 2026-08-27 3 none 2>&1 | tail -1 >> $L
for P in winterf summerf; do TORCH_THREADS=4 ../tsvenv/bin/python ts_gen0.py mid_$P $P joint 2>&1 | grep "finished\|Error\|Traceback" -A5 >> $L; done
../venv/bin/python fresh_gen.py winterf summerf 2>&1 | grep -v -i warn >> $L
for P in winterf summerf; do SFX=2 ../tsvenv/bin/python ts_nbm.py $P 2>&1 | grep "finished\|Error\|Traceback" -A5 >> $L; done
for P in winterf summerf; do ../tsvenv/bin/python ts_shift.py $P 2>&1 | grep "finished\|Error\|Traceback" -A5 >> $L; LAGS=1 ../tsvenv/bin/python ts_shift.py $P 2>&1 | grep "finished\|Error\|Traceback" -A5 >> $L; done
for P in winterf summerf; do FIXCTX=1 ../tsvenv/bin/python ts_live3m.py $P rl 2>&1 | grep "cutoffs in\|Error\|Traceback" -A5 >> $L; ../tsvenv/bin/python ts_live3m.py $P rl 2>&1 | grep "cutoffs in\|Error\|Traceback" -A5 >> $L; done
echo done >> $L
