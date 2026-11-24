# Coupled-auction model: state and handoff

A short entry point. Detail and the full record are in
`research/market-model/notes/coupled_market_notes_2.txt` (sections 21.151–21.175 cover this line of work) and in
`research/coupled-auction/notes/*.md` (one file per delegated task). The specification document is the Claude Doc
"coupled-auction spec" (https://claude.ai/artifact/RazjAfN1RMhbS7tzgXitfF).

## What exists

- `clearing.py` – a toy clearing in the style of EUPHEMIA: period and block orders, ATC lines, price determination
  without paradoxically accepted blocks, and an explicit volume rule after prices (`volume_rule`). `verify.py` checks it,
  including under each volume rule.
- `vslice.py` – two zones, 96 quarters. Zone A is Germany-like; zone B is a synthetic neighbour. Bids come from
  parameters and day shocks, with capacity accounting and storage. All data-driven behaviour is opt-in through `Info`
  fields. The default path reproduces the old simulations exactly; check with `sbi_sim.py` old seeds against
  `$SP/out/sims.npz` and with `verify.py`.
- `hist_window.py` – a prior-predictive check of zone A against real DE-LU days, with these switches:
  - `COND` fixes Germany's observed net position;
  - `FLEET` and `FLEETMODE=output` set the fleet from data before the window;
  - `COSTS` uses the real carbon price, a coal-price prior and scaled flexible demand;
  - `GASCURVE` uses the learned gas curve;
  - `SHAPE` is rejected; do not use it.
- Inference on synthetic data: `sbi_*.py` (BayesFlow; markup recovered, netting not identified, intervals slightly
  overconfident on fresh seeds) and the joint recovery of markup, offered share and outages (`sbi_sim2.py`,
  `sbi_fit2.py`, `sbi_cf.py`).
- Supply curves fitted on public data:
  - `gas_curve*.py`: gas responds to a multi-hour margin, because units commit in runs of about 12 h (`unit_commitment.py`);
  - `gas_floor.py`: the gas floor from heating degree days;
  - `lig_coal_curves.py`: lignite and coal; a moving top level helps, a drifting location does not.

## Best configuration so far (conditional on the observed net position)

`COND=1 COSTS=1 FLEETMODE=output GASCURVE=1`:

| Window | Inside 90% band | Median error |
|---|---|---|
| February 2026 | 83% | +1 EUR/MWh |
| August 2026 | 71% (band twice as wide) | +33 EUR/MWh |

In August, nights are priced too low. The working diagnoses:

- the summer gas floor is about 0.8 GW too low (a year-round small-CHP and process-heat floor of 1.2–1.5 GW);
- lignite is offered about 1.5 GW too large.

Both were tested (21.177, `notes/floor_lignite.md`): `LIGLEVEL` improves August (78% inside, median error +22) but
widens the band and brings back false scarcity in February; `FLOOR2` does not move the price. Neither is a default.
The slice prices too step-like: nights too low and evenings too high, with coal output too low in every variant.

## Open threads

- A neighbour zone built from public data. The data are assembled (21.176): `neighbour_data.py` and
  `notes/neighbour_data.md`, with data in `$SP/entsoe/neighbours/`. No day-ahead capacity is published on the
  flow-based Core borders, so run the aggregate limit at two or three values.
- A solver cut over fill-or-kill blocks only. For now, `GASCURVE` makes the thermal peak blocks divisible.
- Germany's load forecast exceeds scheduled generation net of exports by about 4.6 GW in winter. That is unexplained.
- The commitment horizon may be longer than ±3 h, since λ sits at 1.
- The posterior fit to real days, as opposed to prior-predictive checks, is not started. Forecast MAE is not computed.

## Working rules

- Write expectations, with thresholds, before every test; report each as met or not met. A third window held out of
  all choices is the test.
- Decide from pictures; look at every figure before using it. Titles state the finding.
- This session coordinates. Modellers get self-contained briefs and work only in their own new files or in opt-in
  switches. Their results are checked against their files, figures and the reproducibility checks before they are
  committed.
- Long runs: any run (a fit, a study, a rebuild, a fetch) expected to take more than 20 minutes needs the user's
  explicit approval, every time; approval of one run does not carry over to the next or to a rerun. Before asking,
  consult the advisor to identify blockers and design errors, and report its findings with the request, together with
  the expected run time and what a stop would lose.
- Branch `claude/eager-fermat-vyfgf0` only.
- Secrets and licensed data:
  - the ENTSO-E key stays in `$SP/entsoe/key`; it is never printed or committed;
  - no EPEX licensed data or rules documents in the repo;
  - no data files in the repo.

`$SP` = `/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad`. It is
session-local and holds the data, the virtualenv (`$SP/venv`) and the run outputs.
