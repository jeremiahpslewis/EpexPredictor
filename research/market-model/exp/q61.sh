#!/bin/bash
export HF_HUB_DISABLE_XET=1
P=../tsvenv/bin/python
$P ts_live2.py tsl_28_2k 2026-09-01 2026-11-05 1 mistralai/Timestral-28M 2048
$P ts_live2.py tsl_123_2k 2026-09-01 2026-11-05 1 mistralai/Timestral-123M 2048
$P ts_live2.py tsl_209_2k 2026-09-01 2026-11-05 1 mistralai/Timestral-209M 2048
$P ts_live2.py tsl_123_8k 2026-09-01 2026-11-05 1 mistralai/Timestral-123M 8192
