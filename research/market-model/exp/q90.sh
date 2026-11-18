#!/bin/bash
../venv/bin/python gen_fc_app.py 2>&1 | grep -v "Warn\|for c in"
for m in opt:T pes:T; do ../venv/bin/python live_test3_app.py rl_E_${m/:/_} 2026-09-03 2026-11-07 1 D:$m 2>&1 | tail -1; done; echo finished
