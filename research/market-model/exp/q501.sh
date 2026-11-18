#!/bin/bash
export HF_HUB_DISABLE_XET=1
while pgrep -f q500.sh > /dev/null; do sleep 10; done
for P in autumn winter summer; do TORCH_THREADS=3 ../tsvenv/bin/python ts_nb.py $P 2>&1 | grep "finished\|Error\|rror"; done; echo all finished
