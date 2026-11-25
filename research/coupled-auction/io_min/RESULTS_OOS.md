# Out-of-sample accuracy test: results (holdout 2026-09-08 to 2026-11-08)

Protocol: `prereg/OOS_PROTOCOL.md` with addenda 1–3. Everything was fixed before the holdout was scored, and this is
the single scoring of the frozen specification.

The models:
- **Eleven-zone.** Current curves, training-window posterior `post_train_old.npz`.
- **Two-zone benchmark.** Mode 0, training-window refit `post2_train.npz`: 2 × (300 + 300) draws, max r-hat 1.008,
  no divergences, 29 minutes.
- **Naive.** The same hour of the previous day.

Shock variants:
- **(a) unconditional:** shocks drawn from the fitted distribution;
- **(b) previous-day shocks:** the shock recovered from the same MTU of the previous day.

All forecasts use only inputs known before the D−1 gate, except the disclosed deviations: ENTSO-E wind and solar
forecasts as a stand-in for participants' own, and outage notices not filtered by publication time.

## Verdicts on the pre-registered expectations

- **O1 met.** Variant (b) beats (a) on price MAE for both models. Eleven-zone: (b) better by 9.9 [7.3, 12.2];
  two-zone: (b) better by 5.5 [0.8, 9.8].
- **O2 met, more strongly than stated.** In variant (b) the eleven-zone model beats the two-zone benchmark on price
  MAE by 6.9 EUR/MWh [4.9, 9.2].
- **O3 met.** In variant (b) the eleven-zone model beats the two-zone benchmark on net-import MAE by 0.45 GW
  [0.26, 0.66].
- **O4 (not predicted).** All four structural forecasts beat the naive forecast on price. The eleven-zone (b)
  forecast is better by 24.1 EUR/MWh [19.5, 28.7]. The naive forecast is weak in this window because day-to-day
  swings were large.
- **O5 (caution).** Net-import errors are large in absolute terms for every forecast: 3.0–4.1 GW MAE, against
  4.9 for naive. The window is entirely under hybrid coupling, and the forecasts cannot use inferred constraints.

## Where the errors sit

- **Prices at or below zero** (35 hours). Naive is best (MAE 23.6). The eleven-zone forecast misses by 37–50 and the
  two-zone (b) by 91. This is the low-end curve misfit found by the slope check and the curve-end work.
- **Prices above €200** (582 hours). The eleven-zone (b) forecast is best (MAE 21.2), against 29.3 for the two-zone
  (b) and 41.7 for naive.
- **Time of day** (bias / MAE, EUR/MWh):
  - The eleven-zone (b) forecast has small biases in every window: −5.0 at the evening peak, −1.0 at night.
  - The unconditional eleven-zone forecast (a) underestimates strongly at the evening peak (−43) and morning ramp
    (−31), much less at night (−8). Its shock distribution is centred on the training period, and the holdout's
    price level is higher. This is the pattern reported for the deployed predictor: peaks underestimated, nights
    fine.
  - Carrying over yesterday's shock removes that level error.

| forecast   |   price_MAE |   price_RMSE |   import_MAE |   missed_solves |
|:-----------|------------:|-------------:|-------------:|----------------:|
| E11a       |      30.373 |       40.716 |        3.107 |           0.023 |
| E11b       |      20.46  |       31.035 |        2.963 |           0.009 |
| Z2a        |      32.833 |       46.218 |        4.124 |           0.06  |
| Z2b        |      27.368 |       40.168 |        3.414 |           0.004 |
| naive      |      44.579 |       63.801 |        4.945 |           0     |

## Full table: Price MAE by observed price band

| p       |   E11a |   E11b |   Z2a |   Z2b |   naive |   hours |
|:--------|-------:|-------:|------:|------:|--------:|--------:|
| <=0     |  36.94 |  49.61 | 44.64 | 91.39 |   23.56 |      35 |
| 0-40    |  26.72 |  28.49 | 22.46 | 32.95 |   62.89 |     110 |
| 40-120  |  19.28 |  20.76 | 20.23 | 22.13 |   59.08 |     199 |
| 120-200 |  26.63 |  16.2  | 23.22 | 22.15 |   40.19 |     561 |
| >200    |  38.07 |  21.19 | 47.66 | 29.29 |   41.65 |     582 |

## Full table: Price by time of day (local): bias (forecast − observed) / MAE, EUR/MWh

|                    | E11a         | E11b        | Z2a          | Z2b         | naive       |   hours |
|:-------------------|:-------------|:------------|:-------------|:------------|:------------|--------:|
| night 0-6          | -8.3 / 19.8  | -1.0 / 16.0 | +10.9 / 27.8 | +0.8 / 22.7 | -0.9 / 32.9 |     371 |
| ramp 6-10          | -30.7 / 36.7 | -0.4 / 17.6 | +10.5 / 37.5 | +3.4 / 28.9 | -0.2 / 50.3 |     248 |
| midday 10-16       | -8.5 / 25.5  | -2.7 / 25.2 | +3.0 / 28.4  | -2.0 / 28.3 | -2.3 / 47.0 |     372 |
| evening peak 16-21 | -43.0 / 47.1 | -5.0 / 26.5 | +0.1 / 41.7  | +0.2 / 32.8 | -1.9 / 57.0 |     310 |
| late 21-24         | -20.1 / 24.8 | -2.6 / 13.7 | +18.2 / 30.7 | +4.5 / 23.7 | -1.1 / 34.7 |     186 |

## Full table: Paired differences in absolute error (first minus second; negative = first better), 90% block-bootstrap band

| pair         | price                      | net_import              |
|:-------------|:---------------------------|:------------------------|
| E11a − naive | -14.206 [-19.988, -8.497]  | -1.838 [-2.484, -1.208] |
| E11b − naive | -24.119 [-28.651, -19.483] | -1.982 [-2.511, -1.465] |
| Z2a − naive  | -11.745 [-18.869, -4.289]  | -0.821 [-1.482, -0.160] |
| Z2b − naive  | -17.211 [-22.324, -11.433] | -1.531 [-2.052, -1.036] |
| E11a − E11b  | +9.913 [+7.269, +12.164]   | +0.144 [-0.239, +0.521] |
| E11a − Z2a   | -2.460 [-6.650, +1.307]    | -1.017 [-1.379, -0.643] |
| E11a − Z2b   | +3.005 [-0.851, +6.463]    | -0.307 [-0.756, +0.139] |
| E11b − Z2a   | -12.373 [-16.402, -8.533]  | -1.161 [-1.580, -0.739] |
| E11b − Z2b   | -6.908 [-9.154, -4.902]    | -0.451 [-0.655, -0.259] |
| Z2a − Z2b    | +5.465 [+0.779, +9.846]    | +0.711 [+0.216, +1.197] |
