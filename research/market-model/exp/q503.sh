#!/bin/bash
C="DRIFT=s BTOP=45 NB=sum NBMAP=tsNK:true,knNT:fc"
for PR in autumn:ma winter:mw summer:ms; do P=${PR%%:*}; R=${PR##*:}
  ./srun.sh $R $P $C D1FILE=d1_ts_$P RLFILES=ts=gen_ts_$P,tsNK=gen_ts_$P,knNT=genarm_${P}_known
  ../venv/bin/python dyn2.py $P $R known,ts,tsNK,knNT x+xh 2>&1 | grep -v Warn | grep -v "first day"
done; echo finished
