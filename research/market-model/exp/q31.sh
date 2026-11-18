#!/bin/bash
# final design, but the weight of the two stage-1 models is learned at every training from a 28-day holdout tail
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
cd $SP/exp
PR="$SP/venv/bin/python harness_dw.py --out $SP/out --frames $SP/frames_real --learn-days 540 --gas-lag 0 --eval DE,AT,BE"
$PR --name ho_dw_t28 --start 2026-09-01 --end 2026-11-09 --s1 '{"gas_blend":true,"blend_tail":28}' --s2 '{"gas_blend":true}' | tail -1
$PR --name yr_dw_t28 --stride 4 --s1 '{"gas_blend":true,"blend_tail":28}' --s2 '{"gas_blend":true}' | tail -1
