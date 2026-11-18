#!/bin/bash
C="DRIFT=s BTOP=45"
./srun.sh ha autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer
./srun.sh hw2 winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter FALLBACK=ts=gen_ts_autumn
./srun.sh hs2 summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer FALLBACK=ts=gen_ts_autumn
for P in autumn:ha winter:hw2 summer:hs2; do ../venv/bin/python dynfilter.py ${P%%:*} ${P##*:} known,ts > /dev/null 2>&1; ../venv/bin/python dyn2.py ${P%%:*} ${P##*:} known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day"; done
echo finished
