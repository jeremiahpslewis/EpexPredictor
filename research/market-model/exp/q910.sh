#!/bin/bash
# frozen-fit runs of the candidate's curve: parameters of one fit (fh4 / fh12 / fg4 / fg12 at 2026-01-06 or 2026-03-31) applied unchanged to summer and autumn
cd "$(dirname "$0")"
for src in fh4 fg4 fh12 fg12; do for dm in 2026-01-06:j 2026-03-31:m; do tag=${dm%%:*}; n=z${src#f}${dm##*:}
  ./srun.sh $n summer DRIFT=s BTOP=45 FREEZE=$src/$tag D1FILE=d1_ts_summer RLFILES=ts=gen_ts_summer FALLBACK=ts=gen_ts_autumn
  ./srun.sh $n autumn DRIFT=s BTOP=45 FREEZE=$src/$tag D1FILE=d1_ts_autumn RLFILES=ts=gen_ts_autumn FALLBACK=ts=gen_ts_summer
  echo "$n fits done $(date +%H:%M:%S)" >> ../out/q910.log
done; done
for src in h4 g4 h12 g12; do for d in j m; do n=z$src$d
  (../venv/bin/python dyn2.py summer $n known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day" > ../out/q910_dyn2_${n}_summer.log) &
  (../venv/bin/python dyn2.py autumn $n known,ts x+xh 2>&1 | grep -v Warn | grep -v "first day" > ../out/q910_dyn2_${n}_autumn.log) &
done; wait; done
echo "finished $(date +%H:%M:%S)" >> ../out/q910.log
