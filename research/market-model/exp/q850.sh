#!/bin/bash
# version 14 (homogeneous costs, step one): fits, part checks, pass-through, forward checks
cd "$(dirname "$0")"; F='warn\|H\[\|  t = \|  _w = \|H = {'; export HOMO=1; R=${1:-v14}; [ "$R" = v15 ] && export SHARE=1; [ "$R" = v16 ] && export STACK=1; [ "$R" = v17 ] && export STACK=1 AGG=1; [ "$R" = v18 ] && export STACK=1 AGG=1 REL=1
../venv/bin/python star.py $R autumn 2026-09-18 2>&1 | grep -v -i "$F" | cut -c1-200 &
../venv/bin/python star.py $R winter 2026-02-02 2>&1 | grep -v -i "$F" | cut -c1-200 &
wait
for pt in "autumn 2026-09-18" "winter 2026-02-02"; do set -- $pt; ../venv/bin/python star_check.py $R $1 $2 2>&1 | grep -v -i "$F" | grep "^fit\|premium\|share by\|dispatch layer\|market-expected\|  de:\|  fr:\|  pl:\|net exports\|lignite  \|hard coal\|gas   \|heat-led\|German price\|residual load GW\|hour block\|month block\|gas tercile\|figure"; done
../venv/bin/python - <<PY 2>&1 | grep -v -i "$F"
import numpy as np, star as S
for tag in ("2026-09-18", "2026-02-02"):
    z = np.load(f"{S.lc.SP}/out/star_$R/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")})
    print(f"{tag}: fixed terms DE {np.round(q['alpha'][0], 1)}; heat rates combined cycle {np.round(q['b_c'], 2)}, peakers {np.round(q['b_p'], 2)}; coal coefficient {float(q['bcoal']):.2f}; water value share fr/at/ch {np.round(q['omA'][[1, 4, 7]], 2)}; relative widths DE {np.round(q['wdrel'][0, 1:5], 3)}, median neighbours {np.round(np.median(q['wdrel'][1:, 1:5], 0), 3)}; gas links of DE lignite / hard coal {q['b_a'][0]:+.2f} / {q['b_b'][0]:+.2f}")
PY
echo "--- pass-through, September fit:"; ../venv/bin/python passthru.py $R autumn 2026-09-18 2>&1 | grep -v -i "$F" | cut -c1-330
echo "--- pass-through, February fit (2026 is out of sample):"; ../venv/bin/python passthru.py $R winter 2026-02-02 2>&1 | grep -v -i "$F" | cut -c1-330
echo "--- February fit applied unchanged to Mar-Sep 2026:"
FWD=autumn:2026-09-18 ../venv/bin/python star_check.py $R winter 2026-02-02 2>&1 | grep -v -i "$F" | grep "lignite  \|hard coal\|gas   \|German price\|residual load GW\|month block\|gas tercile\|figure"
echo "--- September fit applied unchanged to the following seven weeks:"
FWD=autumn:2026-11-06 ../venv/bin/python star_check.py $R autumn 2026-09-18 2>&1 | grep -v -i "$F" | grep "lignite  \|hard coal\|gas   \|German price\|residual load GW\|month block\|gas tercile\|figure"
echo QUEUE850 DONE
