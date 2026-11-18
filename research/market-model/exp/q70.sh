#!/bin/bash
for m in A B C C0; do ../venv/bin/python live_test2.py rl_$m 2026-09-03 2026-11-07 1 $m 2>&1 | tail -2; done; echo finished
