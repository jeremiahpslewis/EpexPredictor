#!/bin/bash
cd "$(dirname "$0")"
for p in winter summer autumn; do HF_HUB_DISABLE_XET=1 ../tsvenv/bin/python ts_nbm.py $p 2>&1 | grep -v -i warn > ../out/tsnbm2_$p.log; done; echo done >> ../out/tsnbm2_autumn.log
