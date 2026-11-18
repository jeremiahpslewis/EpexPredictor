#!/bin/bash
while pgrep -f q100.sh > /dev/null; do sleep 20; done
../venv/bin/python sens.py sens_E 2026-09-03 2026-11-07 1 D:pes:T 2>&1 | tail -2; echo finished
