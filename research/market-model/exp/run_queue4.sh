#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
while pgrep -f run_queue3.sh > /dev/null; do sleep 5; done
$PY batch.py batch_T_ho.txt --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE --start 2026-09-01 --end 2026-11-09
$PY batch.py batch_T_ty.txt --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE --stride 2
