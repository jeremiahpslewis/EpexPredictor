#!/bin/bash
# Screen 3 (SCREEN3_PREREG.md): ONE wave, 4 procs x 1 thread x 3 days, after 10:30 UTC 2026-11-27, empty machine, via queue.
# Usage: screen3_wave.sh [nogas]   (nogas: gas-day-D precondition not met -> only rlmH (free) and nbpast)
cd /home/user/EpexPredictor/research/coupled-auction/io_min/team/C
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad; PY=$SP/tsvenv/bin/python
ALSO=nbpast,gasD,nbpast_gasD; [ "$1" = nogas ] && ALSO=nbpast
now=$(date -u +%H%M); [ $(date -u +%Y%m%d) = 20261127 ] && [ $now -lt 1030 ] && { echo "not before 10:30 UTC"; exit 1; }
[ $(ps aux | grep "[p]ython" | awk '$3>30' | wc -l) -gt 0 ] && { echo "machine not empty"; exit 1; }
echo "- [running $(date -u +%H:%M)] C | screen 3 Timestral (SCREEN3_PREREG.md), also=$ALSO | 4 x 1 thread, EMPTY machine | <=20 |" >> $SP/team/queue.md
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FAITHFUL_FRESH=1
for days in 2026-09-08,2026-09-16,2026-09-24 2026-10-01,2026-10-07,2026-10-14 2026-10-22,2026-10-31,2026-11-07 2026-11-11,2026-11-17,2026-11-24; do
  timeout 1200 $PY dump_ts.py --rl-variant matched --de-only --context 8192 --tag screen3 --also $ALSO --days $days > $SP/C/s3_${days%%,*}.log 2>&1 &
done
wait; echo "- [done $(date -u +%H:%M)] C | screen 3 Timestral" >> $SP/team/queue.md; echo done
