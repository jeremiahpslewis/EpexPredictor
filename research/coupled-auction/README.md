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

After prices are fixed, `clear(volume_rule=...)` settles the accepted volumes of at-the-money orders: `least_flow`
(default), `max_ab` / `max_ba` (the two extremes), or `None` (whatever vertex the surplus problem returned). With one
line between two zones the flow is the exporting zone's net position, so this is volume indeterminacy (EUPHEMIA 7.9.1-
7.9.4), not flow indeterminacy, which only arises in meshed networks.

Vertical slice and inference (notes 21.152-21.159 in `research/market-model/notes/coupled_market_notes_2.txt`):

- `vslice.py` - two zones, 96 quarters, bids from parameters and day shocks with capacity accounting and storage.
- `demo.py`, `demo_fig.py` - one draw, the transmission and gas counterfactuals, two books that fit the same day.
- `vol_rule.py` - the volume-indeterminacy case and the admissible flow range per quarter.
- `sbi_sim.py`, `sbi_fit.py`, `sbi_score.py` - BayesFlow recovery of markup, netting and flexible demand's willingness to
  pay; the trained estimator is saved and scored on fresh seeds and under another volume rule.
- `sbi_sim2.py`, `sbi_fit2.py`, `sbi_cf.py` - joint recovery of markup, offered share and forced-outage share, with and
  without published availability, and how their uncertainty moves the counterfactuals.
- `hist_window.py`, `hist_fig.py` - a prior-predictive check of zone A against public DE-LU day-ahead data (inputs from
  the ENTSO-E and gas stores, which stay outside the repository).
