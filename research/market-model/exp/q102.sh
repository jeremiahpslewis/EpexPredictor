#!/bin/bash
export HF_HUB_DISABLE_XET=1
while pgrep -f "q100.sh|q101.sh" > /dev/null; do sleep 20; done
for d in 0 -1 1; do ../tsvenv/bin/python ts_live3.py tsens_$d rl $d 3 2>&1 | grep "cutoffs in"; done; echo finished
