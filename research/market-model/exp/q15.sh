#!/bin/bash
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
PY=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/venv/bin/python
RW='{"gas_ratio":true,"ratio_weight":true}'
HO="--start 2026-09-01 --end 2026-11-09"
run() { name=$1; shift; if [ ! -f /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/$name.npz ]; then "$@" --name $name | tail -1; fi; }
S1="$PY harness.py --out /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out --eval DE --one-stage --frames /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/frames_eu2 --learn-days 540 --gas-lag 1"
declare -A V
V[ns]="drop:xw_+drop:eu_+drop:na_+drop:ld_"
V[na]="drop:xw_+drop:eu_+drop:ns_+drop:ld_"
V[ld]="drop:xw_+drop:eu_+drop:ns_+drop:na_"
V[totns]="drop:xw_+drop:na_+drop:ld_"
V[totnsna]="drop:xw_+drop:ld_"
V[all]="drop:xw_"
for v in ns na ld totns totnsna all; do run ho_s1_DE_eu2_$v $S1 $HO --features "${V[$v]}" --s1 "$RW"; done
for v in ns na ld totns totnsna all; do run s1_DE_eu2_$v $S1 --stride 2 --features "${V[$v]}" --s1 "$RW"; done
