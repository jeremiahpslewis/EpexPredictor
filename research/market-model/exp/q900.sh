#!/bin/bash
# candidate's structural curve with the supply-side option (AV=1); otherwise exactly the configuration of ha / hw2 / hs2 (q402.sh)
cd "$(dirname "$0")"
C="DRIFT=s BTOP=45 AV=1"
./srun.sh va autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer; echo "autumn fits done $(date +%H:%M:%S)" >> ../out/q900.log
./srun.sh vw winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter FALLBACK=ts=gen_ts_autumn; echo "winter fits done $(date +%H:%M:%S)" >> ../out/q900.log
./srun.sh vs summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer FALLBACK=ts=gen_ts_autumn; echo "summer fits done $(date +%H:%M:%S)" >> ../out/q900.log
for P in autumn:va winter:vw summer:vs; do ../venv/bin/python dynfilter.py ${P%%:*} ${P##*:} known,ts > /dev/null 2>&1; ../venv/bin/python dyn2.py ${P%%:*} ${P##*:} known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day" >> ../out/q900_dyn2.log; done
echo "finished $(date +%H:%M:%S)" >> ../out/q900.log
