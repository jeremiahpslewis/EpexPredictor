#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
ALL="DE,AT,BE,FR,NL,SE4,DK1,DK2"
HO="--start 2026-09-01 --end 2026-11-09"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
PR="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval $ALL --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_pr --learn-days 540 --gas-lag 0 --s1-zero eu_"
R3='{"gas_ratio":true,"ratio_weight":true,"ratio_regions":["DE","AT","BE"]}'
R1='{"gas_ratio":true,"ratio_weight":true,"ratio_regions":["DE"]}'
BASE="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval $ALL --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_long"
run ho_pr_r3   $PR $HO --s1 "$R3"
run ho_pr_base $BASE $HO
run yr_pr_r3   $PR --stride 4 --s1 "$R3"
run yr_pr_base $BASE --stride 4
run ho_pr_r1   $PR $HO --s1 "$R1"
run yr_pr_r1   $PR --stride 4 --s1 "$R1"
