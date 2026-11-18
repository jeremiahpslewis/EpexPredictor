#!/bin/bash
# per-zone Timestral forecasts of the neighbours' residual load: autumn, summer, winter; two processes with two threads each
cd "$(dirname "$0")"; export HF_HUB_DISABLE_XET=1
for P in autumn summer winter; do
  for s in 0 1; do TORCH_THREADS=2 ../tsvenv/bin/python ts_nbz.py $P $s 2 > tsnbz_${P}_$s.log 2>&1 & done; wait
  echo "$P done $(ls ../out/tsnbz_$P | wc -l)"
done
echo QUEUE700 DONE
