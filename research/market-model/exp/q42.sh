#!/bin/bash
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
cd $SP/exp
while pgrep -f q41.sh > /dev/null; do sleep 5; done
$SP/venv/bin/python harness_dw.py --out $SP/out --frames $SP/frames_now --learn-days 540 --gas-lag 0 --eval DE --name now_dw --start 2026-10-12 --end 2026-11-17 --s1 '{"gas_blend":true}' | tail -1
