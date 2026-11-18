#!/bin/bash
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
cd $SP/exp
D="$SP/venv/bin/python download_off.py $SP/data_off_raw 2024-03-01 2026-11-10"
$D NL  51.70,52.60,54.04 3.00,4.00,5.96      # Borssele, Hollandse Kust, Gemini
$D DK1 55.70,56.35,56.60 7.67,7.75,11.21     # Horns Rev, Thor, Anholt
$D DK2 55.02,54.56 12.94,11.55               # Kriegers Flak, Roedsand
echo finished
