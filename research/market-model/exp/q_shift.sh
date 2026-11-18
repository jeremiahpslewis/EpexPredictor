#!/bin/bash
cd "$(dirname "$0")"
until grep -q "^done" ../out/q_p8_autumn.log 2>/dev/null; do sleep 20; done
for P in winter summer autumn; do HF_HUB_DISABLE_XET=1 ../tsvenv/bin/python ts_shift.py $P 2>&1 | grep "finished\|Error\|Traceback" -A6 >> ../out/q_shift.log; done
for P in winter summer autumn; do HF_HUB_DISABLE_XET=1 ../tsvenv/bin/python ts_live3m.py $P rl8 2>&1 | grep "cutoffs in\|Error\|Traceback" -A6 >> ../out/q_shift.log; done; echo done >> ../out/q_shift.log
