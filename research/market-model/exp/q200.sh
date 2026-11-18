#!/bin/bash
./srun.sh w01 autumn STRIDE=3 TAUW_FIX=0.1
./srun.sh w10 autumn STRIDE=3 TAUW_FIX=1.0
./srun.sh lw1 autumn STRIDE=3 TAUW_FIX=0.3 LW=1
./srun.sh lw03 autumn STRIDE=3 TAUW_FIX=0.3 LW=0.03
./srun.sh add autumn STRIDE=3 TAUW_FIX=0.3 ADD=1
./srun.sh d365 autumn STRIDE=3 TAUW_FIX=0.3 TRAIN_DAYS=365
./srun.sh d700 autumn STRIDE=3 TAUW_FIX=0.3 TRAIN_DAYS=700
./srun.sh noeu autumn STRIDE=3 TAUW_FIX=0.3 EU=0
./srun.sh dp autumn STRIDE=3 TAUW_FIX=0.3 DRIFT=p
echo finished
