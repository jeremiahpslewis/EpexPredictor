#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
HO="--start 2026-09-01 --end 2026-11-09"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
S1="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE,AT,BE --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_real --learn-days 540 --gas-lag 0 --s1-zero eu_"
run ho_m_raw   $S1 $HO
run ho_m_ratio $S1 $HO --s1 '{"gas_ratio":true,"ratio_weight":true}'
run ho_m_vc    $S1 $HO --s1 '{"gas_ratio":true,"vc":true}'
run yr_m_raw   $S1 --stride 4
run yr_m_ratio $S1 --stride 4 --s1 '{"gas_ratio":true,"ratio_weight":true}'
run yr_m_vc    $S1 --stride 4 --s1 '{"gas_ratio":true,"vc":true}'
