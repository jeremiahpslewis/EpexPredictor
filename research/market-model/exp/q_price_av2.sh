cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
export HF_HUB_DISABLE_XET=1
for p in winter summer; do
MOD=lite /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_shift.py $p avgA > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_av2_shift_$p.log 2>&1
GENF=avgA /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_live3m.py $p rl > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_av2_p_$p.log 2>&1
done
echo done > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_av2_done.txt
