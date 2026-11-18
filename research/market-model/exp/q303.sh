#!/bin/bash
export HF_HUB_DISABLE_XET=1
while pgrep -f q302.sh > /dev/null; do sleep 15; done
W1=2025-12-02; W2=2026-02-27; S1=2026-04-01; S2=2026-08-27
[ -f ../out/gen_win.npz ] || ../venv/bin/python gen_fc_season.py gen_win $W1 $W2 3 2>&1 | grep -v "Warn\|for c in\|X\[" | tail -5
[ -f ../out/inputs_winter.npz ] || ../venv/bin/python prep_inputs.py winter $W1 $W2 3 gen_win 2>&1 | tail -1
[ -f ../out/inputs_summer.npz ] || ../venv/bin/python prep_inputs.py summer $S1 $S2 3 gen_ssn 2>&1 | tail -1
for P in winter summer; do
  ../venv/bin/python mkgen.py $P
  TORCH_THREADS=4 ../tsvenv/bin/python ts_gen.py joint_$P $P joint 2>&1 | grep "finished\|Error"
done
echo "inputs and Timestral generation done"
for P in winter summer; do
  if [ $P = winter ]; then A=$W1; B=$W2; else A=$S1; B=$S2; fi
  [ -f ../out/genarm_${P}_own.npz ] && [ -d ../out/tsgen_joint_$P ] && ../venv/bin/python ts_gen_eval.py joint_$P $P gen_$( [ $P = winter ] && echo win || echo ssn ) > tsgen_eval_$P.txt 2>&1
  run() { name=$1; shift; [ -f ../out/$name.npz ] || env "$@" ../venv/bin/python live_test3_season.py $name $A $B 3 D:pes:T 2>&1 | tail -1; }
  run two_${P}_known GENFILE=genarm_${P}_known D1=d1_ts_$P
  run two_${P}_own GENFILE=genarm_${P}_own D1=d1_ts_$P
  run two_${P}_avg GENFILE=gen_avg_$P D1=d1_ts_$P
  run two_${P}_ts GENFILE=gen_ts_$P D1=d1_ts_$P
done
[ -f ../out/win_ts.npz ] || TS_START=$W1 TS_END=$W2 ../tsvenv/bin/python ts_live3.py win_ts base 0 3 2>&1 | grep "cutoffs in"
[ -f ../out/win_trees.npz ] || ../venv/bin/python live_test.py win_trees $W1 $W2 3 2>&1 | tail -1
echo finished
