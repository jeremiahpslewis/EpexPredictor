#!/bin/bash
# Timestral load forecast from a midnight issue (context to the start of the target day), notes 21.29
cd "$(dirname "$0")"; export HF_HUB_DISABLE_XET=1
for P in summer autumn winter; do TORCH_THREADS=4 ../tsvenv/bin/python ts_gen0.py mid_$P $P joint 2>&1 | grep "finished\|Error\|Traceback" -A4 >> ../out/q_mid.log; done; echo "done $(date +%H:%M:%S)" >> ../out/q_mid.log
