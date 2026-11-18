#!/bin/bash
cd "$(dirname "$0")"
../venv/bin/python market1n.py prior 2>&1 | grep -v -i warn | cut -c1-400 > ../out/market1n_checks.log
../venv/bin/python market1n.py recover 2>&1 | grep -v -i warn | grep "recover\|kn true\|inside" | cut -c1-260 >> ../out/market1n_checks.log
../venv/bin/python market1n.py fit 2>&1 | grep -v -i warn | grep "fitted\|kn,\|x1 \|sd_X\|conflict\|period" | cut -c1-330 >> ../out/market1n_checks.log; echo done >> ../out/market1n_checks.log
