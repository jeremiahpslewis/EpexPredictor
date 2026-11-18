#!/bin/bash
run() { name=$1; shift; [ -f ../out/$name.npz ] || env "$@" ../venv/bin/python live_test3_season.py $name 2026-09-03 2026-11-07 1 D:pes:T 2>&1 | tail -1; }
run two_autumn_ts_d1 GENFILE=gen_ts_autumn D1=d1_ts_autumn
run two_autumn_avg_d1 GENFILE=gen_avg_autumn D1=d1_ts_autumn
run two_autumn_known_d1 GENFILE=genarm_autumn_known D1=d1_ts_autumn
run two_autumn_own_d1 GENFILE=genarm_autumn_own D1=d1_ts_autumn
run two_autumn_known_drop GENFILE=genarm_autumn_known D1=drop
echo finished
