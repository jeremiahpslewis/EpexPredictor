#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
H="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE"
HO="--start 2026-09-01 --end 2026-11-09"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
# 1) does the gas-normalised target carry through stage 2 on the recent period?
run ho_2s_DE_base            $H $HO --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_long
run ho_2s_DE_L540_ratio_all  $H $HO --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_gall --gas-lag 1 --learn-days 540 --s1 '{"gas_ratio":true,"ratio_weight":true}'
# 2) window length with the gas-normalised target (stage 1)
run ho_s1_DE_L540_ratio_w    $H $HO --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_xl --gas-lag 1 --learn-days 540 --s1 '{"gas_ratio":true,"ratio_weight":true}'
run ho_s1_DE_L900_ratio_w    $H $HO --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_xl --gas-lag 1 --learn-days 900 --s1 '{"gas_ratio":true,"ratio_weight":true}'
run s1_DE_L900_ratio_w       $H --stride 2 --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_xl --gas-lag 1 --learn-days 900 --s1 '{"gas_ratio":true,"ratio_weight":true}'
run s1_DE_L730_ratio_w       $H --stride 2 --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_xl --gas-lag 1 --learn-days 730 --s1 '{"gas_ratio":true,"ratio_weight":true}'
# 3) two-stage on the test year
run 2s_DE_L540_ratio_all     $H --stride 4 --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_gall --gas-lag 1 --learn-days 540 --s1 '{"gas_ratio":true,"ratio_weight":true}'
# 4) remaining holdout two-stage variants
run ho_2s_DE_L540_ratio      $H $HO --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_long --gas-lag 1 --learn-days 540 --s1 '{"gas_ratio":true,"ratio_weight":true}'
run ho_2s_DE_L540            $H $HO --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_long --learn-days 540
# 5) small extras on top (stage 1)
run s1_DE_L540_ratio_w_r300  $H --stride 2 --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_xl --gas-lag 1 --learn-days 540 --s1 '{"gas_ratio":true,"ratio_weight":true,"rounds":300,"params":{"learning_rate":0.05}}'
run ho_s1_DE_L540_ratio_w_r300 $H $HO --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_xl --gas-lag 1 --learn-days 540 --s1 '{"gas_ratio":true,"ratio_weight":true,"rounds":300,"params":{"learning_rate":0.05}}'
