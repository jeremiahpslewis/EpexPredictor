cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
export HF_HUB_DISABLE_XET=1
for p in "winter mid" "summer mid" "autumn v2" "winterf mid" "summerf mid"; do
JOIN=k1a /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_core.py $p > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core_k1a_${p%% *}.log 2>&1
done
echo done > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core3_done.txt
