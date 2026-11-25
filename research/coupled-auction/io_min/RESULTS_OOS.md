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

## Headline

- **Best forecast.** The best forecast is yesterday's recovered shock plus the structural model's response to
  today's inputs. Its small bias comes from the carried shock.
- **The model's own level** (the unconditional variant) is off by 8–43 EUR/MWh depending on time of day.
- **Comparisons.** The naive baseline is weak in this window, and the deployed predictor was not scored, so nothing
  here shows an improvement on the deployed model. The earlier sealed backtest (`research/market-model/exp/sealed_score.py`)
  scores pre-registered winter and summer days, not this window, so it cannot stand in for that comparison.
- **Targets and curve ends.** The slope target (ratio 0.8–1.25) was not met (0.69 on training), and both curve-end
  variants were dropped.

## Where the errors sit

- **Fuel costs outside the training range.** The fuel-cost index L_t = 1.85 P_g + 0.37 τ ran 166–221 in the holdout
  (5–95%), entirely above the training range of 86–152. The neighbour curves are extrapolated.
- **Recovered shocks shift.** Mean shocks (GW) at draw 0, training against holdout (lagged): France +2.8 → +15.5,
  NO2 +0.6 → −4.9, AT +0.3 → +2.8, PL +0.6 → +2.1, SE4 +0.2 → +2.1; Germany −2.1 → −1.7. Summed over zones, the
  evening shock rises from +15 to +29 GW. A positive shock means the model's supply exceeds what is needed at the
  observed price, so it predicts too low.
- **Why only the eleven-zone unconditional forecast underestimates: cause not established.** It underestimates at
  every time of day, most at the evening peak (−43) and morning ramp (−31). The unconditional two-zone forecast does
  not (+0.1 to +18), so the cause is specific to the eleven-zone model. The carried shock (variant b) absorbs the
  error. Fuel-cost extrapolation was the first candidate, but a training-only check (2026-11-25) points elsewhere:
  - France's modelled supply at the holdout's typical price and cost index (174, 182) is 62.7 GW, against 63.0 at
    training's (100, 108). The curve itself barely moves between the windows.
  - France's recovered shock is strongly seasonal within training: +8.2 GW in October 2025 at a low cost index
    (90), −3.6 to −9.1 in winter, then rising through 2026 to +18 in August and +16 in September. The holdout's
    +15.5 continues that late-summer level.
  - The likely cause is French need or availability (nuclear output, heating load), which the neighbour curves do
    not include. The cost index rose over 2026 alongside the season, so the two are confounded.
- **Deployed predictor (hypothesis only).** The deployed predictor's reported pattern (peaks underestimated, nights
  fine) resembles the unconditional variant's. That rests on one day of retail prices and is not tested here.
- **Price bands.** These condition on the observed price, so every smooth forecast looks worst at the extremes. At
  prices ≤ 0 (35 hours) naive has the lowest error (23.6); above €200 (582 hours) the eleven-zone (b) forecast does
  (21.2). No cause is attributed.
- **Post hoc, not pre-registered.** Regressing the realised day-over-day price change on the eleven-zone (b)
  predicted change gives a slope of 0.91 (intercept +2.5, 1,480 hours). The predicted changes are close to the right
  scale on average. This does not echo the failed slope check (0.69 on training), which weights large demand
  surprises.
- **Additional deviation.** Heating degree days come from the realised daily temperature, not a forecast.

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
