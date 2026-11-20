# Coupled day-ahead auction: small clearing solver

A toy clearing in the style of EUPHEMIA, used to check the specification of a generative model of the coupled
day-ahead auction (see the specification document). Not a replica of the production algorithm.

- `clearing.py` - zones, periods, step orders, block orders with a minimum acceptance ratio, ATC lines, price bounds;
  surplus maximisation over block decisions (mixed-integer, HiGHS through SciPy), price determination with no
  paradoxically accepted blocks, no-good cuts; the interval of valid prices per zone and period.
- `example.py` - the worked two-zone, three-period example, cleared with a 400 MW and a 1500 MW limit.
- `verify.py` - hand-solvable cases, a forced paradoxically-accepted block, and 300 random markets checked against brute
  force over all block selections, a market-consistency audit, the surplus identity and monotonicity in line capacity.

Run with Python 3.11+ and SciPy 1.9+: `python verify.py`, `python example.py`.
