#!/bin/bash
export HF_HUB_DISABLE_XET=1
P=../tsvenv/bin/python
for spec in "tsl_123_8k mistralai/Timestral-123M 8192" "tsl_209_8k mistralai/Timestral-209M 8192" "tsl_28_2k mistralai/Timestral-28M 2048" "tsl_209_2k mistralai/Timestral-209M 2048" "tsl_5_8k mistralai/Timestral-5M 8192"; do
  set -- $spec; $P ts_live2.py $1 2026-09-03 2026-11-07 1 $2 $3 2>&1 | grep cutoffs
done
echo finished
