#!/bin/bash
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
cd $SP/exp
ALL="DE,AT,BE,FR,NL,SE4,DK1,DK2"
PR="$SP/venv/bin/python harness_dw.py --out $SP/out --frames $SP/frames_off --learn-days 540 --gas-lag 0 --eval $ALL"
$PR --name ho_off --start 2026-09-01 --end 2026-11-09 --s1 '{"gas_blend":true}' | tail -1
$PR --name yr_off --stride 4 --s1 '{"gas_blend":true}' | tail -1
