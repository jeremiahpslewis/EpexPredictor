#!/bin/bash
# attribution of version 11: heat-led must-run alone (v11h), weather-dependent border capacity alone (v11w)
cd "$(dirname "$0")"
for tagp in "autumn 2026-09-18" "winter 2026-02-02"; do set -- $tagp
  HEAT=1 WXC=0 ../venv/bin/python star.py v11h $1 $2 2>&1 | grep "seed 0" | cut -c1-110 &
  HEAT=0 WXC=1 ../venv/bin/python star.py v11w $1 $2 2>&1 | grep "seed 0" | cut -c1-110 &
  wait
done
for r in v11h v11w; do
  [ $r = v11h ] && export HEAT=1 WXC=0 || export HEAT=0 WXC=1
  echo "--- $r: February fit applied unchanged to Mar-Sep 2026:"; FWD=autumn:2026-09-18 ../venv/bin/python star_check.py $r winter 2026-02-02 2>&1 | grep "German price\|month block\|lignite  \|gas   "
  echo "--- $r: September fit applied unchanged to the following seven weeks:"; FWD=autumn:2026-11-06 ../venv/bin/python star_check.py $r autumn 2026-09-18 2>&1 | grep "German price\|month block\|lignite  \|gas   "
  ../venv/bin/python star_check.py $r autumn 2026-09-18 2>&1 | grep "^fit"; ../venv/bin/python star_check.py $r winter 2026-02-02 2>&1 | grep "^fit"
done
echo QUEUE820 DONE
