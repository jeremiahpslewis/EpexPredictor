#!/bin/bash
# two-zone equilibrium with transfer capacity, started from the linear-shift fits (ma/mw/ms); same arms as those runs
cd "$(dirname "$0")"
C="DRIFT=s BTOP=45 NB=sum ZONE=1 NBMAP=tsNK:true,knNT:fc"
for PR in winter:zw:mw autumn:za:ma summer:zs:ms; do IFS=: read P R REF <<< "$PR"
  ./srun.sh $R $P $C INITREF=$REF D1FILE=d1_ts_$P RLFILES=ts=gen_ts_$P,tsNK=gen_ts_$P,knNT=genarm_${P}_known
  ../venv/bin/python dyn2.py $P $R known,ts,tsNK,knNT x+xh 2>&1 | grep -v -i warn | grep -v "first day"
done
echo QUEUE620 DONE
