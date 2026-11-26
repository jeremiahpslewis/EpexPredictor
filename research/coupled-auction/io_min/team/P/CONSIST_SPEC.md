# consist: frozen specification (as run on the 6 screened days, P011; written 2026-11-26 ~23:45 UTC, before the confirmation)

**Market route, unchanged:** marketmodel curves, carbon 75, other plant 8 GW, gas lagged 192 qh, Timestral-123M at
the pinned revision forecasting the supply shift with demand as the known covariate and weekday as categorical,
cleared on D with D's forecast RL.

**The one change.** Over the 8192-qh context, the RL of every earlier day d < D, used in both the shift history and
the covariate history, is the Timestral-mixed RL the route read at d's own gate (`rl_mixed` of d's faithful day
file). The default is the published TSO figures.
- **Where no stored day exists, the published RL stays.** That covers days before 2026-09-08 and the missing day
  2026-11-08.
- **Log length** is the number of logged context days: 0 for D = 09-08, rising by 1 per day to 60 for D = 11-07, and
  about 61-79 for the fresh days, out of about 85 context days.
- **D's own RL** is the usual forecast, rl_mixed at D's gate.

**Leak assertion, per day.** Every logged value used for D lies on a day d < D, so it was issued at d's gate, which is
at or before D-1's gate. No published RL at or after D's first slot enters the history (it ends at the cutoff), and gas
is cut at the cutoff.

**Known data defect.** Stored `rl_mixed` on the 61-window days (`out_strict`) differs from what the route read on the
last 3 quarter hours of each day: Timestral alone there (README erratum), up to 0.8 GW. It affects D's demand
identically in both arms, and the log days in the consist arm.

**Confirmation construction** (`market_ts2.py`).
- Frames are rebuilt from the stores. They match C's dumped frames to 4e-6 on history price, RL and gas on 4 checked
  days.
- The horizon is D only (96 qh) in both arms, against D plus D+1 in the app and in the screen run. The control is
  therefore `repro96`, built the same way, and repro96 vs the stored route is reported as the construction effect.

**Live version.**
- The app's `loadforecastlog` logs the trees' RL and Timestral's RL separately ("trees", "timestral"), each issue
  after the last published time. It does not log the mix the route reads.
- The mix is `TimestralLoad.weights(len) * timestral + (1 - weights) * trees`, so it can be rebuilt from the log if
  the issue at each day's gate is selected (the last issue before the gate).
- Simpler: log `rl_mixed` as the route reads it at each gate, 96 qh per day.
