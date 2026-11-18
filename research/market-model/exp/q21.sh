#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
ALL="DE,AT,BE,FR,NL,SE4,DK1,DK2"
HO="--start 2026-09-01 --end 2026-11-09"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
while pgrep -f q20.sh > /dev/null; do sleep 5; done
PR="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval $ALL --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_real --learn-days 540 --gas-lag 0 --s1-zero eu_"
R2='{"gas_ratio":true,"ratio_weight":true,"ratio_regions":["DE","BE"]}'
run ho_pr_r2 $PR $HO --s1 "$R2"
run yr_pr_r2 $PR --stride 4 --s1 "$R2"
