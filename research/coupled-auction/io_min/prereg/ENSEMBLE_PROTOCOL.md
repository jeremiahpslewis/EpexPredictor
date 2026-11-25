# Ensemble backtest of the deployed techs with the eleven-zone model (fixed 2026-11-25, before any ensemble score)

**Window.** Local delivery days 2026-09-08 to 2026-11-07 (61 days), quarter-hour resolution, ct/kWh. Forecasts are
issued with prices known to the end of D−1, the information of the noon D−1 gate. Slots are aligned by UTC time from
the stored cutoffs (local midnight of D), 96 per day. On 2026-10-25 (100 local quarter-hours) the first 96 are
scored.

**Members.** The best stored data for each deployed tech (research implementations of the app's methods, from
`research/market-model`, already run for these days):
- **T, trees.** Two-stage gas-ratio LightGBM, retrained daily on 540 days, 8 zones, with the residual-load input.
  Primary: weather vintage `opt` (1-day-old runs for the first 14 h, 2-day-old after), the freshest stored
  (`rl_E_opt_T.npz`). Reported for reference: `pes` (2–3-day-old).
- **P, Timestral on the price.** Deployed lead-weighted residual-load form (`tsm_rllwA_autumn.npz`, median).
- **A, market route.** Deployed slim form with the lead-weighted load (`tsshift_lwA_lite_autumn.npz`), cleared with
  `market1.clear` at the posterior median, gas frozen at issue, carbon price 75.
- **E, eleven-zone model.** Variant (b), previous-day shocks (`oos11.pkl`), median over draws, EUR/MWh / 10.
  - Information advantage, disclosed: ENTSO-E wind and solar forecasts published at 18:00 on D−1, realised heating
    degree days, and the published Core domain.
  - A strictly pre-gate E (the deployed residual-load forecast for Germany) is a separate run.

**Ensembles.** Equal weights; no weights are fitted, because no clean validation set exists for all four members.
- M3 = (T + P + A) / 3, the deployed mix.
- M4 = (T + P + A + E) / 4.
- M3E = (T + P + E) / 3, with E replacing the slim market route.

**Metrics.**
- MAE over the whole day, and over night 0–6 local and evening peak 16–21 local.
- Paired differences in absolute error with a block bootstrap by day (90%).

**Expectations.**
- X1: M3 reproduces research note 21.121's autumn whole-day MAE (1.740 ct/kWh) within 0.05 on these 61 days. This
  checks the reconstruction.
- X2: E alone is worse than M3. The structural model's own forecast is not expected to beat a three-way mix.
- X3, not predicted: whether M4 or M3E beats M3.

**Contamination, disclosed.**
- The window was already used: by the eleven-zone holdout scoring, and for the deployed techs by the research notes
  (the "autumn" set).
- Results are therefore descriptive. A clean comparison needs days after 2026-11-08, which is the frozen forward test
  of `research/market-model`, plus E.

## Addendum 1 (2026-11-25, before the strict run): strictly pre-gate eleven-zone member E_s

Variant (b) with inputs known at the gate:
- **Germany.** Yesterday's published residual load at the same MTU, plus the change in the deployed lead-weighted
  residual-load forecast (`gen_lwA_autumn.npz`, `rl_pes`) between its forecast for D (cutoff of D) and its forecast
  for D−1 (cutoff of D−1), at the same slot. Both come from the same series, because the two sources differ in level
  and scope.
- **Neighbours.** Today's ENTSO-E load forecast (published before the gate), minus yesterday's published wind and
  solar forecast for the same hour.
- **Heating degree days** from D−1.

Everything else is as for E: the same posterior, draws, domain and fallback. E_s is scored in the same table. Two
further joins are post hoc and labelled as such:
- a renormalised mix that drops a member's missed quarter-hours, as the deployed blend does;
- time-of-day weights fitted leaving out weekly blocks.
