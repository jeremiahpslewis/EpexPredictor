#!/bin/bash
# energy-charts public power for the neighbours not yet downloaded (one directory, raw JSON only)
cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/ext/ecnb
for c in be ch at nl dk; do for r in "2024-03-01 2024-08-31 a" "2024-09-01 2025-02-28 b" "2025-03-01 2025-08-31 c" "2025-09-01 2026-02-28 d" "2026-03-01 2026-08-31 e" "2026-09-01 2026-11-16 f"; do set -- $r
  [ -s pp_${c}_$3.json ] || curl -sS -m 240 -o pp_${c}_$3.json -w "$c $3 %{http_code} %{size_download}\n" "https://api.energy-charts.info/public_power?country=$c&start=$1&end=$2"; sleep 2; done; done
echo ECNB DONE
