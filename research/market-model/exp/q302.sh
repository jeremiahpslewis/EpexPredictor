#!/bin/bash
A="D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn,avg=gen_avg_autumn"
./srun.sh c1 autumn DRIFT=s BTOP=45 $A; ./srun.sh mc1 march DRIFT=s BTOP=45
./srun.sh c3 autumn DRIFT=s SCALE_RW=0.05 $A; ./srun.sh mc3 march DRIFT=s SCALE_RW=0.05
./srun.sh c2 autumn DRIFT=s BTOP=45 SCALE_RW=0.05 $A; ./srun.sh mc2 march DRIFT=s BTOP=45 SCALE_RW=0.05
echo finished
