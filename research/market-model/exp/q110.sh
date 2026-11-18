#!/bin/bash
export HF_HUB_DISABLE_XET=1
S=2026-04-01; E=2026-08-27
true
../venv/bin/python gen_fc_season.py gen_ssn $S $E 3 2>&1 | grep -v "Warn\|for c in\|X\[" | tail -6
GENFILE=gen_ssn ../venv/bin/python live_test3_season.py ssn_rl $S $E 3 D:pes:T 2>&1 | tail -1
TS_START=$S TS_END=$E ../tsvenv/bin/python ts_live3.py ssn_ts base 0 3 2>&1 | grep "cutoffs in"
echo finished
