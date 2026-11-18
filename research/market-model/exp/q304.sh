#!/bin/bash
C="DRIFT=s BTOP=45"
./srun.sh fa autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn,avg=gen_avg_autumn FALLBACK=own=gen_ssn,ts=gen_ts_summer,avg=gen_avg_summer
./srun.sh fw winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter,avg=gen_avg_winter FALLBACK=own=gen_fc_app,ts=gen_ts_autumn,avg=gen_avg_autumn
./srun.sh fs summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer,avg=gen_avg_summer FALLBACK=own=gen_fc_app,ts=gen_ts_autumn,avg=gen_avg_autumn
for P in autumn:fa winter:fw summer:fs; do RKEY=R1 ARMS=known,own,ts,avg ../venv/bin/python level_filter.py ${P%%:*} ${P##*:} > lf_${P##*:}.txt 2>&1; done
echo finished
