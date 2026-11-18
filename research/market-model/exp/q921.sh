#!/bin/bash
# candidate's curve with the gas-share prior (GS=1), refitted at every origin; names ka / kw / ks (ga / gw / gs were taken by an earlier session)
cd "$(dirname "$0")"
until grep -q "ca finished" ../out/q920.log; do sleep 10; done
C="DRIFT=s BTOP=45 GS=1 SAVEPAR=1"
./srun.sh ka autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer; echo "autumn fits done $(date +%H:%M:%S)" >> ../out/q921.log
./srun.sh kw winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter FALLBACK=ts=gen_ts_autumn; echo "winter fits done $(date +%H:%M:%S)" >> ../out/q921.log
./srun.sh ks summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer FALLBACK=ts=gen_ts_autumn; echo "summer fits done $(date +%H:%M:%S)" >> ../out/q921.log
for P in autumn:ka winter:kw summer:ks; do ../venv/bin/python dyn2.py ${P%%:*} ${P##*:} known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day" >> ../out/q921_dyn2.log; done
echo "finished $(date +%H:%M:%S)" >> ../out/q921.log
