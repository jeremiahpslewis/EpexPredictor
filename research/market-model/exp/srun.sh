#!/bin/bash
# usage: srun.sh NAME PERIOD [env assignments...]   runs 4 shards in parallel
NAME=$1; PERIOD=$2; shift 2
for i in 0 1 2 3; do env "$@" ../venv/bin/python smodel.py $NAME $PERIOD $i 4 2>&1 | grep -v "Warn\|warn" > slog_${NAME}_$i.log & done; wait
grep -L "^finished" slog_${NAME}_*.log | sed 's/^/NOT FINISHED: /'
