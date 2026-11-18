#!/bin/bash
cd "$(dirname "$0")"; export HF_HUB_DISABLE_XET=1
for P in summer autumn winter; do TORCH_THREADS=3 ../tsvenv/bin/python ts_live3m.py $P rl 2>&1 | grep "cutoffs in\|Error\|Traceback" -A3 >> ../out/q_tsm.log; done; echo done >> ../out/q_tsm.log
