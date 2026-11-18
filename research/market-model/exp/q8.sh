#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
H="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_self --learn-days 540"
HO="--start 2026-09-01 --end 2026-11-09"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
for p in pn28 pn14 pa14 pn07; do
  others=$(for q in pn28 pn14 pa14 pn07; do if [ $q != $p ]; then printf "+drop:gas_$q"; fi; done)
  run ho_s1_DE_L540_self_$p $H $HO --features "drop:gasprice$others" --s1 "{\"gas_ratio\":\"gas_$p\",\"ratio_weight\":true}"
  run s1_DE_L540_self_$p    $H --stride 2 --features "drop:gasprice$others" --s1 "{\"gas_ratio\":\"gas_$p\",\"ratio_weight\":true}"
done
