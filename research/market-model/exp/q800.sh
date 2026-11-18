#!/bin/bash
# version 11 (heat-led must-run, weather-dependent border capacity): fits for the two dates, checks, forward check
cd "$(dirname "$0")"
../venv/bin/python star.py v11 autumn 2026-09-18 2>&1 | grep -v -i warn | cut -c1-110 &
../venv/bin/python star.py v11 winter 2026-02-02 2>&1 | grep -v -i warn | cut -c1-110 &
wait
for pt in "autumn 2026-09-18" "winter 2026-02-02"; do set -- $pt; ../venv/bin/python star_check.py v11 $1 $2 2>&1 | grep -v -i "warn\|H\[" | grep "^fit\|  de:\|net exports\|lignite  \|hard coal\|gas   \|heat-led\|other heat\|border cap\|^   [a-z][a-z]: wind\|inertia\|German price\|residual load GW\|hour block\|month block\|gas tercile\|figure"; done
echo "--- February fit applied unchanged to Mar-Sep 2026:"
FWD=autumn:2026-09-18 ../venv/bin/python star_check.py v11 winter 2026-02-02 2>&1 | grep -v -i "warn\|H\[" | grep "lignite  \|hard coal\|gas   \|German price\|residual load GW\|month block\|gas tercile\|figure"
echo "--- September fit applied unchanged to the following seven weeks:"
FWD=autumn:2026-11-06 ../venv/bin/python star_check.py v11 autumn 2026-09-18 2>&1 | grep -v -i "warn\|H\[" | grep "lignite  \|hard coal\|gas   \|German price\|residual load GW\|month block\|gas tercile\|figure"
echo QUEUE800 DONE
