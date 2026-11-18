#!/bin/bash
export HF_HUB_DISABLE_XET=1
for m in rl cal rlc base; do [ -f ../out/tso_$m.npz ] || ../tsvenv/bin/python ts_live3.py tso_$m $m 2>&1 | grep "cutoffs in\|Error\|error" ; done; echo finished
