#!/bin/bash
# One confirmation wave (S-approved, option A): 4 procs x 1 thread, days pre-registered in CONFIRM_DAYS.md. Not extended.
cd /home/user/EpexPredictor/research/coupled-auction/io_min/team/C
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad; PY=$SP/tsvenv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout 1140 $PY dump_ts.py --de-only --context 8192 --tag confirm --also nbpast --days 2026-09-09,2026-09-14,2026-09-18,2026-09-23,2026-09-27 > $SP/C/conf1.log 2>&1 &
timeout 1140 $PY dump_ts.py --de-only --context 8192 --tag confirm --also nbpast --days 2026-10-02,2026-10-06,2026-10-11,2026-10-15,2026-10-19 > $SP/C/conf2.log 2>&1 &
FAITHFUL_FRESH=1 timeout 1140 $PY dump_ts.py --de-only --context 8192 --tag confirmf --also nbpast --days 2026-10-24,2026-10-28,2026-11-02,2026-11-06,2026-11-09 > $SP/C/conf3.log 2>&1 &
FAITHFUL_FRESH=1 timeout 1140 $PY dump_ts.py --de-only --context 8192 --tag confirmf --also nbpast --days 2026-11-12,2026-11-15,2026-11-18,2026-11-20,2026-11-23 > $SP/C/conf4.log 2>&1 &
wait; echo done
