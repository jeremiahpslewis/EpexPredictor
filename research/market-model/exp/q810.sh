#!/bin/bash
cd "$(dirname "$0")"
until grep -q "QUEUE800 DONE" q800.log; do sleep 20; done
../venv/bin/python star_sim.py v11 autumn 2026-09-18 2>&1 | grep -v -i "warn\|H\["
echo QUEUE810 DONE
