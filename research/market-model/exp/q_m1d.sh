#!/bin/bash
cd "$(dirname "$0")"
../venv/bin/python market1d.py prior 2>&1 | grep -v -i warn > ../out/market1d_checks.log
../venv/bin/python market1d.py recover 2>&1 | grep -v -i warn >> ../out/market1d_checks.log
../venv/bin/python market1d.py fit 2>&1 | grep -v -i warn >> ../out/market1d_checks.log; echo done >> ../out/market1d_checks.log
