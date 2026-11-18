#!/bin/bash
for P in winter summer autumn; do SETS=ABCZYX ../venv/bin/python ladder.py $P 2>&1 | tail -1; mv ../out/ladder_$P.npz ../out/ladder2_$P.npz; done; echo finished
