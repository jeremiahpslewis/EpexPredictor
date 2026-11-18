#!/bin/bash
cd "$(dirname "$0")"
for P in winter summer autumn; do NONB=1 HF_HUB_DISABLE_XET=1 ../tsvenv/bin/python ts_shift.py $P 2>&1 | grep "finished\|Error\|Traceback" -A6 >> ../out/q_nonb.log; done; echo done >> ../out/q_nonb.log
