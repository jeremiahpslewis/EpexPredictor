#!/bin/bash
# weekly fits from 2025-11-24 for the four arms, one arm at a time on four cores
cd "$(dirname "$0")"
for v in c p1 p3 p2; do
  for s in 0 1 2 3; do OCXV=$v ../venv/bin/python ocx_fits.py $s 4 2025-11-24 2026-11-09 > ../out/ocx_fits_${v}_$s.log 2>&1 & done; wait
  echo "$v done $(date +%H:%M:%S)" >> ../out/q_priors.log
done
