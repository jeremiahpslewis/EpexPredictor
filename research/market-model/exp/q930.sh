#!/bin/bash
# full sampling of the candidate's curve at one origin (2026-09-18): tempered posterior with and without the gas-share prior, then the priors alone (likelihood weight 0)
cd "$(dirname "$0")"
B="DRIFT=s BTOP=45 D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer ONLY=2026-09-18 FIT=nuts WARM=300 DRAWS=150 CHAINS=4"
env $B ../venv/bin/python smodel.py sph autumn 0 1 2>&1 | grep -v "Warn\|warn" > ../out/q930_sph.log; echo "sph done $(date +%H:%M:%S)" >> ../out/q930.log
env $B GS=1 ../venv/bin/python smodel.py spg autumn 0 1 2>&1 | grep -v "Warn\|warn" > ../out/q930_spg.log; echo "spg done $(date +%H:%M:%S)" >> ../out/q930.log
env $B GS=1 LW=0 ../venv/bin/python smodel.py sp0g autumn 0 1 2>&1 | grep -v "Warn\|warn" > ../out/q930_sp0g.log; echo "sp0g done $(date +%H:%M:%S)" >> ../out/q930.log
env $B LW=0 ../venv/bin/python smodel.py sp0h autumn 0 1 2>&1 | grep -v "Warn\|warn" > ../out/q930_sp0h.log; echo "sp0h done $(date +%H:%M:%S)" >> ../out/q930.log
echo "finished $(date +%H:%M:%S)" >> ../out/q930.log
