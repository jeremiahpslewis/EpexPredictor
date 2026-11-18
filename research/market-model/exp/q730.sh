#!/bin/bash
# when the per-zone neighbour forecasts of a period are complete: score them and rerun the structural model (neighbours as a shift) with them
cd "$(dirname "$0")"
for PR in autumn:ya:ma:ha:7:tsl_123_8k:vint summer:ys:ms:hs2:3:ssn_ts:median winter:yw:mw:hw2:3:win_ts:median; do IFS=: read P R M H B TSF TSK <<< "$PR"
  until grep -q "$P done" q700.log; do sleep 30; done
  echo "=== $P"; ../venv/bin/python nbz_eval.py $P 2>&1 | grep -v -i warn
  ./srun.sh $R $P DRIFT=s BTOP=45 NB=sum NBMAP=tsNK:true,knNT:fc NBDIR=tsnbz D1FILE=d1_ts_$P RLFILES=ts=gen_ts_$P,tsNK=gen_ts_$P,knNT=genarm_${P}_known
  ../venv/bin/python dyn2.py $P $R known,ts,tsNK,knNT x+xh 2>&1 | grep -v -i warn | grep -v "first day"
  ../venv/bin/python - <<PY 2>&1 | grep -v -i warn
import numpy as np
SP="/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; ld=lambda f,k="vint": np.load(f"{SP}/out/{f}.npz")[k].astype(float)
np.savez_compressed(f"{SP}/out/candy_$P.npz", vint=(ld("structfc_${R}_ts_dyn2xxh")+ld("$TSF","$TSK"))/2)
PY
  ../venv/bin/python cmp_eval.py $P $B "structural, neighbours from per-zone forecasts (live)=structfc_${R}_ts_dyn2xxh" "neighbours from the one-sum forecast (before)=structfc_${M}_ts_dyn2xxh" "no neighbours=structfc_${H}_ts_dyn2xxh" "neighbours published (ceiling)=structfc_${M}_tsNK_dyn2xxh" 2>&1 | grep -v -i warn
  ../venv/bin/python cmp_eval.py $P $B "candidate now: (no neighbours + Timestral price)/2=cand_$P" "(per-zone neighbours + Timestral price)/2=candy_$P" 2>&1 | grep -v -i warn | tail -2
done
echo QUEUE730 DONE
