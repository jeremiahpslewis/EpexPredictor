#!/bin/bash
# final design (two stages, both with the two-model average), keeping both models' forecasts of each stage
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
cd $SP/exp
ALL="DE,AT,BE,FR,NL,SE4,DK1,DK2"
PR="$SP/venv/bin/python harness_dw.py --out $SP/out --frames $SP/frames_real --learn-days 540 --gas-lag 0 --eval $ALL"
$PR --name ho_dw --start 2026-09-01 --end 2026-11-09 --s1 '{"gas_blend":true}' | tail -1
$PR --name yr_dw --stride 4 --s1 '{"gas_blend":true}' | tail -1
