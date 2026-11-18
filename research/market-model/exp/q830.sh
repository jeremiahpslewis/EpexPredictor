#!/bin/bash
# version 12: fits for the two dates, checks, forward checks; then each of the four changes switched off in turn
cd "$(dirname "$0")"; F='warn\|H\[\|  t = \|  _w = '
../venv/bin/python star.py v12 autumn 2026-09-18 2>&1 | grep -v -i "$F" | cut -c1-200 &
../venv/bin/python star.py v12 winter 2026-02-02 2>&1 | grep -v -i "$F" | cut -c1-200 &
wait
for pt in "autumn 2026-09-18" "winter 2026-02-02"; do set -- $pt; ../venv/bin/python star_check.py v12 $1 $2 2>&1 | grep -v -i "$F" | grep "^fit\|premium\|market-expected\|exchange shift\|  de:\|net exports\|lignite  \|hard coal\|gas   \|heat-led\|inertia\|German price\|residual load GW\|hour block\|month block\|gas tercile\|figure"; done
echo "--- February fit applied unchanged to Mar-Sep 2026:"
FWD=autumn:2026-09-18 ../venv/bin/python star_check.py v12 winter 2026-02-02 2>&1 | grep -v -i "$F" | grep "lignite  \|hard coal\|gas   \|German price\|residual load GW\|month block\|gas tercile\|figure"
echo "--- September fit applied unchanged to the following seven weeks:"
FWD=autumn:2026-11-06 ../venv/bin/python star_check.py v12 autumn 2026-09-18 2>&1 | grep -v -i "$F" | grep "lignite  \|hard coal\|gas   \|German price\|residual load GW\|month block\|gas tercile\|figure"
echo "MAIN DONE"
run1() { env $2 ../venv/bin/python star.py $1 winter 2026-02-02 2>&1 | grep -v -i "$F" | cut -c1-170 | sed "s/^/$1: /"; env $2 FWD=autumn:2026-09-18 ../venv/bin/python star_check.py $1 winter 2026-02-02 2>&1 | grep "German price\|month block" | sed "s/^/$1 forward: /"; }
run1 v12np PREM=0 & run1 v12nt TSTAR=0 & wait
run1 v12nb BLEND=0 & run1 v12nx WXA=0 & wait
echo QUEUE830 DONE
