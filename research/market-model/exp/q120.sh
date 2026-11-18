#!/bin/bash
export WARM=600 DRAWS=400
for m in ar nocarry; do
  for attempt in 1 2 3; do
    ../venv/bin/python bayes.py aut_$m autumn $m 2>&1 | grep -v "Warn\|warn\|forecast done" >> q120_$m.log
    grep -q "^finished" q120_$m.log && break
  done
  echo "aut_$m done"
done
