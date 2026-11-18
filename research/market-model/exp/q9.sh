#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
RW='{"gas_ratio":true,"ratio_weight":true}'
H="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_xl --learn-days 540"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
for lag in 7 14 30; do
  run ho_s1_DE_L540_ratio_w_lag$lag $H --start 2026-09-01 --end 2026-11-09 --gas-lag $lag --s1 "$RW"
done
for lag in 7 30; do
  run s1_DE_L540_ratio_w_lag$lag $H --stride 2 --gas-lag $lag --s1 "$RW"
done
