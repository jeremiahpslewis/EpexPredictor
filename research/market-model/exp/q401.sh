#!/bin/bash
C="DRIFT=s BTOP=45"
./srun.sh ga autumn $C D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn
./srun.sh gw winter $C D1FILE=d1_ts_winter RLFILES=ts=gen_ts_winter
./srun.sh gs summer $C D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer
echo finished
