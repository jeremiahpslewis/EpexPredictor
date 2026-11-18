#!/bin/bash
cd "$(dirname "$0")"
../venv/bin/python market1b.py prior 2>&1 | grep -v -i warn | sed 's/; /\n   /g' | cut -c1-200 > ../out/market1b_checks.log
../venv/bin/python market1b.py recover 2>&1 | grep -v -i warn | cut -c1-260 >> ../out/market1b_checks.log
../venv/bin/python market1b.py fit 2>&1 | grep -v -i warn | cut -c1-330 >> ../out/market1b_checks.log; echo done >> ../out/market1b_checks.log
