#!/bin/bash
./srun.sh s1 autumn TAUW_FIX=0.3 DRIFT=s
./srun.sh s2 autumn TAUW_FIX=0.3 DRIFT=xs
./srun.sh hl autumn TAUW_FIX=0.3 HL=120
./srun.sh bt autumn TAUW_FIX=0.3 BTOP=45
echo finished
