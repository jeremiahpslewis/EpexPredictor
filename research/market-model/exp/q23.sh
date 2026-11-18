#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
ALL="DE,AT,BE,FR,NL,SE4,DK1,DK2"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
while pgrep -f q22.sh > /dev/null; do sleep 5; done
PR="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval $ALL --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_real --learn-days 540 --gas-lag 0 --s1-zero eu_"
run ho_pr_blend $PR --start 2026-09-01 --end 2026-11-09 --s1 '{"gas_blend":true}'
run yr_pr_blend $PR --stride 4 --s1 '{"gas_blend":true}'
