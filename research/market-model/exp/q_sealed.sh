#!/bin/bash
cd "$(dirname "$0")"; L=../out/q_sealed.log; export HF_HUB_DISABLE_XET=1
../venv/bin/python prep_inputs.py winters 2025-12-04 2026-03-01 3 none 2>&1 | tail -1 >> $L
../venv/bin/python prep_inputs.py summers 2026-04-03 2026-08-28 3 none 2>&1 | tail -1 >> $L
for P in winters summers; do TORCH_THREADS=4 ../tsvenv/bin/python ts_gen0.py mid_$P $P joint 2>&1 | grep "finished\|Error\|Traceback" -A5 >> $L; done
../venv/bin/python fresh_gen.py winters summers 2>&1 | grep -v -i warn >> $L
for P in winters summers; do SFX=2 ../tsvenv/bin/python ts_nbm.py $P 2>&1 | grep "finished\|Error\|Traceback" -A5 >> $L; done
for P in winters summers; do ../tsvenv/bin/python ts_shift.py $P 2>&1 | grep "finished\|Error\|Traceback" -A5 >> $L; done
for P in winters summers; do FIXCTX=1 ../tsvenv/bin/python ts_live3m.py $P rl 2>&1 | grep "cutoffs in\|Error\|Traceback" -A5 >> $L; FIXCTX=1 ../tsvenv/bin/python ts_live3m.py $P rl8 2>&1 | grep "cutoffs in\|Error\|Traceback" -A5 >> $L; done
echo done >> $L
