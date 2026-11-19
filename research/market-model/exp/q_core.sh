cd /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/exp
export HF_HUB_DISABLE_XET=1
/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_core.py winter mid > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core_winter.log 2>&1
/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_core.py summer mid > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core_summer.log 2>&1
/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_core.py autumn v2 > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core_autumn.log 2>&1
/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_core.py winterf mid > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core_winterf.log 2>&1
/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/tsvenv/bin/python ts_core.py summerf mid > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core_summerf.log 2>&1
echo done > /tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/q_core_done.txt
