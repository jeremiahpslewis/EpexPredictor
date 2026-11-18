#!/bin/bash
# version 13: exchange curves learned from the scheduled exchange alone, heating threshold per zone
cd "$(dirname "$0")"; F='warn\|H\[\|  t = \|  _w = '; export XCUT=1 TSTARZ=1
until grep -q "QUEUE830 DONE" q830.log; do sleep 20; done
../venv/bin/python star.py v13 autumn 2026-09-18 2>&1 | grep -v -i "$F" | cut -c1-200 &
../venv/bin/python star.py v13 winter 2026-02-02 2>&1 | grep -v -i "$F" | cut -c1-200 &
wait
for pt in "autumn 2026-09-18" "winter 2026-02-02"; do set -- $pt; ../venv/bin/python star_check.py v13 $1 $2 2>&1 | grep -v -i "$F" | grep "^fit\|premium\|market-expected\|exchange shift\|^  [a-z][a-z]: export\|net exports\|lignite  \|hard coal\|gas   \|heat-led\|German price\|residual load GW\|hour block\|month block\|gas tercile\|figure"; ../venv/bin/python star_parts.py v13 $1 $2 2>&1 | grep "figure"; done
echo "--- February fit applied unchanged to Mar-Sep 2026:"
FWD=autumn:2026-09-18 ../venv/bin/python star_check.py v13 winter 2026-02-02 2>&1 | grep -v -i "$F" | grep "German price\|month block\|figure"
echo "--- September fit applied unchanged to the following seven weeks:"
FWD=autumn:2026-11-06 ../venv/bin/python star_check.py v13 autumn 2026-09-18 2>&1 | grep -v -i "$F" | grep "German price\|month block\|figure"
echo QUEUE840 DONE
