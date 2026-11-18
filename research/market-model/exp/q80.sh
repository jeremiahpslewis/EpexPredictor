#!/bin/bash
while [ ! -f ../dense/hist.parquet ]; do pgrep -f dl_dense2.py >/dev/null || (../venv/bin/python dl_dense2.py >> dl_dense2.log 2>&1); sleep 15; done
[ -f ../out/gen_fc.npz ] || ../venv/bin/python gen_fc.py 2>&1 | grep -v Warn
for m in pes:O opt:O pes:T opt:T; do n=rl_D_${m/:/_}; [ -f ../out/$n.npz ] || ../venv/bin/python live_test3.py $n 2026-09-03 2026-11-07 1 D:$m 2>&1 | tail -1; done; echo finished
