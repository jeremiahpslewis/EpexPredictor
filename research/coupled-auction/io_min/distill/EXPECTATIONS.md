# Distillation study: expectations (written 2026-11-25, before any experiment below was run)

Window: local delivery days 2026-09-08..2026-11-07 (61 days, 96 QH), MAE ct/kWh, paired 7-day block bootstrap as in
`ensemble_bt.py`. E = eleven-zone (b) with post-gate TSO wind/solar; E_s = strictly pre-gate eleven-zone (b).
Anything chosen by looking at the 61 days is labelled post hoc in the results.

X0 Reproduction gate: T 1.818, P 1.747, A 1.891, E 2.245, E_s 2.867, M3 1.642 reproduced to 0.002.

## 1. Error anatomy
- A1: the daily-level component (daily mean error squared) is a larger share of MSE for E_s than for T and P
  (E_s > 45%, T and P < 45%).
- A2: after removing each day's mean error, E_s is still worse than T (shape also loses), but the gap shrinks by >40%.
- A3: E_s's excess over T is largest at night and in the morning ramp, and in the lowest forecast-RL quintile.
- A4: error correlation between E_s and T is 0.4–0.7, lower than T–P (> 0.6).
- A5 (diagnostic): persistence-plus-slope `p_lag + 2.15 * dRL_forecast` (yesterday's same-QH price plus the change in
  the deployed RL forecast) is within 0.25 ct/kWh of E_s's MAE, i.e. E_s is close to persistence plus a slope.

## 2. What the trees learn
- S1: the cf.py/slope11 net-slope estimator applied to out-of-fold tree-fitted prices on training hours gives
  1.8–2.8 EUR/MWh per GW, i.e. near the empirical 2.15–2.38 and below the structural 3.15.
- S2: the finite-difference partial dp/dRL of the full-feature tree (rl shifted, weather held) is smaller than S1
  (weather features share the RL signal); a model with rl + calendar + gas only gives a larger, closer-to-empirical
  slope. Both are steeper in the high-price band than in the middle band.
- L1: on the 61 days, removing `ts` (time trend) and using the raw target instead of the gas-ratio blend worsens a
  single-stage tree by > 0.15 ct/kWh each; a "merit-order only" tree (rl + calendar + gas, raw target, no ts, no
  weather) is worse than the full single-stage tree by > 0.3 ct/kWh.

## 3. Distillation / grey-box (all pre-gate on the holdout, trained on data before 2026-09-01 unless stated)
- R1 (residual learner, frozen pre-09): a LightGBM correction of the structural (b) error, trained on 2026-01..08
  structural forecasts, improves E_s by >= 0.3 ct/kWh but the corrected E_s stays worse than T (MAE 2.0–2.6).
- R2 (residual learner, expanding through D-1 on strict holdout errors): better than R1 by >= 0.1.
- R3: neither corrected E_s improves M3 by more than 0.05 when added as a 4th member.
- K1 (scalar slope distillation, kappa = tree slope / structural slope fixed from training): p_lag + kappa (E_s - p_lag)
  improves E_s by 0.05–0.25 ct/kWh; still worse than T.
- K2 (re-clear with the German RL change scaled by kappa): within 0.1 of K1.
- C1 (tree forecast as shock pseudo-observation) collapses to a price-space blend of E_s and T; not run separately.

## 4. Stability around regime changes (added on request)
Daily-retrained simplified single-stage tree (540-day window), stored market route / Timestral (summer + autumn sets),
structural (b) forecasts (training-period run + holdout E).
- B1: no clear jump (> 30% rise in 14-day mean of MAE / mean |price| relative to the 28 days before) in tree error at
  advanced hybrid coupling go-live (mid-June 2026).
- B2: around the fuel-cost rise above the earlier range (Aug–Sep 2026) the raw-target tree's relative error rises more
  than the gas-ratio tree's.
- B3: tree error is not more sensitive to French availability (pregate GW, low vs high tercile) than the structural
  (b) forecast's error; both rise when French availability is low.
