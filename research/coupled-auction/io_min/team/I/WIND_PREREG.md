# Wind booster screen: pre-registration (written 2026-11-26 23:05 UTC, before any fit)

## Question

Can the DE wind booster (`tso_wind`, onshore + offshore TSO day-ahead figure) beat the `matched` booster by at least
10% in wind MAE, using only weather history already on disk?

## Setup (`wind_lab.py`, offline, 1 process, 1 thread)

- **Weather.** Grid weather on the app's 58 points:
  - history and forecast both `_previous_day2` (`dense/hist.parquet` `*_d2`, plus `fresh_data/grid_prev.parquet`
    `*_d2`), as `load_d2_history()` in `faithful_bt.py`;
  - hourly, interpolated to 15 min as the app does;
  - training rows from 2025-10-01, as `matched`.
- **Target.** `tso_wind` from the stored generation figures (fresh stores, to 11-26).
- **Aux features.** `AuxDataStore._compute_data`.
- **Evaluation days:**
  - "61": 2026-09-08..11-07;
  - "fresh18": 11-09..11-26.
- **Evaluation slots.** The slots of D with grid vintage k ≥ 2 (`rlc_k` from the stored `out_strict_rlmatched` /
  `out_fresh_strict_rlmatched` npz). Each is predicted with d2 weather; where the harness used d3, d2 is used here,
  which is slightly optimistic for every variant alike.
- **Training cut.** Rows up to D-1 23:45 local (the cutoff).
- **Refit cadence:**
  - the main screen refits weekly: each eval day uses the model trained to the cutoff of the most recent Monday on
    or before D;
  - a cadence check refits daily on 8 fixed days (09-10, 09-24, 10-08, 10-22, 11-05, 11-12, 11-19, 11-26) for V0
    and the best two variants;
  - if daily refits change the ranking, the weekly screen is not used to rank.
- **Score.** Wind MAE in GW over the eval slots, separately for 61 and fresh18:
  - paired per-day difference against V0;
  - 90% circular 7-day block bootstrap.
- **Sanity check.** V0 on the daily-refit days should be close to the stored `rlc_matched_tso_wind` MAE on the same
  slots (61 days: 1.71 GW on d2+ slots).
- **Promotion.** At least 10% lower wind MAE than V0 on both windows. That goes to S for a price screen on the 9
  fixed oracle days (one wave), with S's approval first.

## Variants (fixed list; every one is logged)

| id | change against V0 |
|---|---|
| V0 | app features and LightGBM settings (400 rounds, lr 0.05, 31 leaves, L2), trained on d2 history (= `matched` wind) |
| V1 | target = capacity factor: wind / installed wind capacity (energy-charts yearly values, end-of-year anchors linearly interpolated, extrapolated with the last slope; public); prediction × capacity |
| V2 | L1 objective |
| V3 | recency weighting, sample weight 0.5^(age / 120 d) |
| V4 | physical power-curve features: per point a smooth power curve on the 120 m speed (cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s, in m/s), plus the same on w80, mean onshore/offshore speed, cross-point speed std; replaces the cubed/capped speed |
| V5 | V0 features plus aggregates: mean of speed³ capped at rated, onshore 90th-percentile speed, 3 h speed change (mean onshore) |
| V6 | average of the V0 prediction and an app-weather-trained model (store `gridweather_v1` history, the app's booster) applied to the same d2 weather (the d1/d2 mix S listed; the live lead lies between) |
| V7 | the best combination of V1–V5 that individually beat V0 on both windows (chosen after V1–V5; marked post hoc) |

## Expectation (before fitting)

- **V1:** −2 to −5% (the trend matters across 13 months).
- **V2:** about 0.
- **V3:** −0 to −3%.
- **V4 and V5:** −0 to −4%.
- **V6:** worse on d2 weather (the earlier RL study found staler-trained models worse on fresh weather and the
  reverse).
- **Overall:** −10% on both windows is unlikely from features alone. Most remaining wind error is weather error at
  1.5–2 days, not booster form.
