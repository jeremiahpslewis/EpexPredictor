#!/bin/bash
export WARM=600 DRAWS=400 TRAIN_DAYS=120
for attempt in 1 2 3; do
  ../venv/bin/python bayes.py ssn_ar120 summer ar 2>&1 | grep -v "Warn\|warn\|forecast done" >> q122.log
  grep -q "^finished" q122.log && break
done
echo done >> q122.log
