#!/bin/bash
cd "$(dirname "$0")"
for P in winter summer autumn; do HF_HUB_DISABLE_XET=1 TORCH_THREADS=4 ../tsvenv/bin/python ts_gen0.py mid_p8_$P $P jointnb 2>&1 | grep -v -i warn > ../out/q_p8_$P.log; done; echo "done" >> ../out/q_p8_autumn.log
