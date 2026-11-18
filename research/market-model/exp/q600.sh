#!/bin/bash
# curve-segment trend with Laplace changepoints on top of the frozen spec
cd "$(dirname "$0")"
C="DRIFT=sc BTOP=45"
./srun.sh pw winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter FALLBACK=ts=gen_ts_autumn
./srun.sh pa autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer
./srun.sh ps summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer FALLBACK=ts=gen_ts_autumn
for pr in "winter pw" "autumn pa" "summer ps"; do set -- $pr; ../venv/bin/python dyn2.py $1 $2 known,ts x+xh 2>&1 | grep -v -i warn; done
echo QUEUE600 DONE
