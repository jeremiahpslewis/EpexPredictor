#!/bin/bash
# the candidate's curve without the Europe-wide weather term (EU=0); otherwise exactly the configuration of hs2 / ha / hw2 (q402.sh). Notes 21.21.
cd "$(dirname "$0")"
./srun.sh es0 summer DRIFT=s BTOP=45 EU=0 D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer FALLBACK=ts=gen_ts_autumn
./srun.sh ea0 autumn DRIFT=s BTOP=45 EU=0 D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer
./srun.sh ew0 winter DRIFT=s BTOP=45 EU=0 D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter FALLBACK=ts=gen_ts_autumn
for P in summer:es0 autumn:ea0 winter:ew0; do ../venv/bin/python dyn2.py ${P%%:*} ${P##*:} known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day" >> ../out/q_eu0.log; done; echo "done $(date +%H:%M:%S)" >> ../out/q_eu0.log
