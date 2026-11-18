#!/bin/bash
# walk-forward refits, one at a time (notes 21.19): market model M0e and the plain curve, data ending at the first of each month
cd "$(dirname "$0")"
for E in 2025-11-01 2025-12-01 2026-01-01 2026-02-01 2026-03-01 2026-04-01 2026-05-01 2026-06-01 2026-07-01 2026-08-01 2026-09-01 2026-10-01 2026-11-01; do T=_r${E//-/}
  END=$E TAG=$T WARM=350 DRAWS=250 ../venv/bin/python market0e.py fit 2>&1 | grep "^refit" >> ../out/q_refit.log
  ../venv/bin/python l1_refit.py $E $T 2>&1 | grep "^plain" >> ../out/q_refit.log
done; echo "all refits done $(date +%H:%M:%S)" >> ../out/q_refit.log
