#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
RW='{"gas_ratio":true,"ratio_weight":true}'
HO="--start 2026-09-01 --end 2026-11-09"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
S1="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_eu1 --learn-days 540 --gas-lag 1"
run ho_s1_DE_eu1_zones $S1 $HO --features "drop:eu_" --s1 "$RW"
run ho_s1_DE_eu1_tot   $S1 $HO --features "drop:xw_" --s1 "$RW"
run ho_s1_DE_eu1_both  $S1 $HO --s1 "$RW"
run s1_DE_eu1_zones    $S1 --stride 2 --features "drop:eu_" --s1 "$RW"
run s1_DE_eu1_tot      $S1 --stride 2 --features "drop:xw_" --s1 "$RW"
run s1_DE_eu1_both     $S1 --stride 2 --s1 "$RW"
