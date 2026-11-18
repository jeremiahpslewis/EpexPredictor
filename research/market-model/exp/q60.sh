#!/bin/bash
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad; cd $SP/exp; export HF_HUB_DISABLE_XET=1
$SP/venv/bin/python live_test.py live_ho 2026-09-03 2026-11-07 1 2>&1 | tail -1
$SP/tsvenv/bin/python ts_live.py tslive_ho 2026-09-03 2026-11-07 1 2>&1 | grep -v Warn | tail -1
while pgrep -f download_prev_all > /dev/null; do sleep 10; done
$SP/venv/bin/python live_test.py live_yr 2025-09-03 2026-08-27 4 2>&1 | tail -1
$SP/tsvenv/bin/python ts_live.py tslive_yr 2025-09-03 2026-08-27 4 2>&1 | grep -v Warn | tail -1
echo finished
