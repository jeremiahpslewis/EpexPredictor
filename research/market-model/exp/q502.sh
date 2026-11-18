#!/bin/bash
while pgrep -f q501.sh > /dev/null; do sleep 15; done
C="DRIFT=s BTOP=45 NB=sum"
rm -rf ../out/struct_na
./srun.sh na autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn
./srun.sh nw winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter
./srun.sh ns summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer
for P in autumn:na winter:nw summer:ns; do ../venv/bin/python dyn2.py ${P%%:*} ${P##*:} known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day"; done
echo finished
