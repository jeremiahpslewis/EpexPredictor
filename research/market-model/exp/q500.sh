#!/bin/bash
for P in autumn winter summer; do SETS=DE ../venv/bin/python ladder.py $P 2>&1 | tail -1; mv ../out/ladder_$P.npz ../out/ladder_nb_$P.npz; done; echo finished
