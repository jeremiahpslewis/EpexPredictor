#!/bin/bash
while pgrep -f q120.sh > /dev/null; do sleep 20; done
export WARM=600 DRAWS=400 TRAIN_DAYS=120
for attempt in 1 2 3; do
  ../venv/bin/python bayes.py aut_ar120 autumn ar 2>&1 | grep -v "Warn\|warn\|forecast done" >> q121.log
  grep -q "^finished" q121.log && break
done
echo done >> q121.log
