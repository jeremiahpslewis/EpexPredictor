#!/bin/bash
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad; cd $SP/exp; export HF_HUB_DISABLE_XET=1
T="$SP/tsvenv/bin/python ts_test.py"
$T ts28_ho mistralai/Timestral-28M 8192 2026-09-01 2026-11-09 1 2>&1 | grep -v Warn | tail -1
$T ts28_yr mistralai/Timestral-28M 8192 2025-09-01 2026-09-01 4 2>&1 | grep -v Warn | tail -1
$T ts28nc_ho mistralai/Timestral-28M 8192 2026-09-01 2026-11-09 1 nocov 2>&1 | grep -v Warn | tail -1
$T ts209_ho mistralai/Timestral-209M 4096 2026-09-01 2026-11-09 1 2>&1 | grep -v Warn | tail -1
$T ts209_yr mistralai/Timestral-209M 4096 2025-09-01 2026-09-01 4 2>&1 | grep -v Warn | tail -1
echo finished
