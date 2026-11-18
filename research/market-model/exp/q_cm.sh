#!/bin/bash
# the candidate's curve with the midnight-issue load forecast (notes 21.29); otherwise the configuration of hs2 / ha / hw2
cd "$(dirname "$0")"
./srun.sh cms summer DRIFT=s BTOP=45 D1FILE=d1_mid_summer RLFILES=ts=gen_mid_summer FALLBACK=ts=gen_ts_autumn
./srun.sh cma autumn DRIFT=s BTOP=45 D1FILE=d1_mid_autumn RLFILES=ts=gen_mid_autumn FALLBACK=ts=gen_ts_summer
./srun.sh cmw winter DRIFT=s BTOP=45 D1FILE=d1_mid_winter RLFILES=ts=gen_mid_winter FALLBACK=ts=gen_ts_autumn
for P in summer:cms autumn:cma winter:cmw; do ../venv/bin/python dyn2.py ${P%%:*} ${P##*:} known,ts x+xh > /dev/null 2>&1; done; echo done > ../out/q_cm.log
