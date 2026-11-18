#!/bin/bash
# candidate's curve with the gas-share prior (GS=1), refitted at every origin; otherwise exactly ha / hw2 / hs2
cd "$(dirname "$0")"
C="DRIFT=s BTOP=45 GS=1 SAVEPAR=1"
./srun.sh ga autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer; echo "autumn fits done $(date +%H:%M:%S)" >> ../out/q920.log
./srun.sh gw winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter FALLBACK=ts=gen_ts_autumn; echo "winter fits done $(date +%H:%M:%S)" >> ../out/q920.log
./srun.sh gs summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer FALLBACK=ts=gen_ts_autumn; echo "summer fits done $(date +%H:%M:%S)" >> ../out/q920.log
for P in autumn:ga winter:gw summer:gs; do ../venv/bin/python dyn2.py ${P%%:*} ${P##*:} known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day" >> ../out/q920_dyn2.log; done
echo "gs finished $(date +%H:%M:%S)" >> ../out/q920.log
# convergence probe: the candidate as it is, 12000 steps, autumn (the period where the in-sample error moved most)
./srun.sh ca autumn DRIFT=s BTOP=45 STEPS=12000 D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer; ../venv/bin/python dyn2.py autumn ca known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day" >> ../out/q920_dyn2.log
echo "ca finished $(date +%H:%M:%S)" >> ../out/q920.log
