# Faithful rerun of the deployed DE ensemble: results (post hoc, window already used)

Days scored: 61 of 61 (local 2026-09-08..11-07), 96 quarter-hours each; MAE in ct/kWh. Runtime per day: median 99 s, max 169 s. Largest gap between the harness's actual price and the stored one: 0.0000 ct/kWh.

|                       |   MAE |   bias |   00-06 |   06-10 |   10-16 |   16-21 |   21-24 |
|:----------------------|------:|-------:|--------:|--------:|--------:|--------:|--------:|
| faithful trees        | 1.787 | -0.336 |    1.16 |    1.86 |    1.95 |    2.51 |    1.41 |
| faithful Timestral    | 1.732 | -0.294 |    1.02 |    1.77 |    1.95 |    2.52 |    1.36 |
| faithful market route | 1.872 | -0.231 |    1.08 |    1.85 |    2.15 |    2.76 |    1.46 |
| faithful issued blend | 1.648 | -0.287 |    0.98 |    1.67 |    1.87 |    2.4  |    1.26 |
| stored T trees (opt)  | 1.818 | -0.282 |    1.23 |    1.7  |    1.89 |    2.71 |    1.55 |
| stored P Timestral    | 1.747 | -0.124 |    1.14 |    1.67 |    1.94 |    2.49 |    1.43 |
| stored A market route | 1.891 |  0.087 |    1.15 |    1.75 |    2.31 |    2.67 |    1.43 |
| stored M3             | 1.642 | -0.106 |    1.05 |    1.53 |    1.81 |    2.43 |    1.34 |

## Paired differences (first minus second; negative = first better), 90% block bootstrap

- faithful issued blend − stored M3: +0.006 [-0.049, +0.055]
- faithful trees − stored T trees (opt): -0.032 [-0.099, +0.032]
- faithful Timestral − stored P Timestral: -0.015 [-0.089, +0.054]
- faithful market route − stored A market route: -0.019 [-0.099, +0.057]
- faithful issued blend − faithful trees: -0.139 [-0.199, -0.082]
- faithful issued blend − faithful Timestral: -0.084 [-0.162, -0.009]
- faithful issued blend − faithful market route: -0.224 [-0.292, -0.161]

## Residual load (GW, mean over days of daily MAE on D)

- faithful tree series: 2.56; Timestral-mixed series: 2.28; stored pessimistic (rl_pes): 2.11
- F6: days where the faithful tree series exceeds rl_pes by more than 1 GW: 19 of 61
- largest excesses (day, faithful, rl_pes): 2026-09-09 7.32 vs 4.16; 2026-10-14 4.77 vs 1.68; 2026-10-08 5.06 vs 2.72; 2026-10-17 4.47 vs 2.58; 2026-09-14 3.67 vs 1.87; 2026-10-10 3.04 vs 1.39; 2026-10-28 6.77 vs 5.20; 2026-10-18 5.91 vs 4.39
## Expectations (EXPECTATIONS.md) against outcomes

- **F1 met.** The issued blend scores 1.648, inside 1.60–1.85. It equals the stored M3 (1.642): the difference is
  +0.006 [−0.049, +0.055].
- **F2 not met, on the good side.** The trees score 1.787, just below the expected 1.80–2.10, and level with the stored
  opt-weather trees (−0.032 [−0.099, +0.032]).
- **F3 met.** Timestral scores 1.732 and the market route 1.872.
- **F4 met.** The blend beats every member clearly:
  - the trees by −0.139 [−0.199, −0.082];
  - Timestral by −0.084 [−0.162, −0.009];
  - the market route by −0.224 [−0.292, −0.161].
- **F5 met.** Tree residual-load MAE is 2.56 GW.
- **F6 not met.** 19 of 61 days are more than 1 GW worse than the stored rl_pes, against at most 10 allowed. The
  Timestral-mixed series has 7 such days.
- **F7 met.** No day failed, and every day passed the leak assertions.

## F6: where the residual-load excess comes from (post hoc)

Split by the weather vintage the harness used for each quarter-hour of D: GW, mean over days; bias is forecast minus
the TSO figure.

| series | weather | bias | MAE | stored rl_pes bias | rl_pes MAE |
|---|---|---|---|---|---|
| harness trees | d1 (≈ first 6 h of D) | −0.11 | 1.41 | +0.24 | 1.49 |
| harness trees | d2 (rest of D) | −1.32 | 2.97 | +0.09 | 2.32 |
| harness Timestral-mixed | d1 | +0.07 | 1.18 | +0.24 | 1.49 |
| harness Timestral-mixed | d2 | −0.93 | 2.64 | +0.09 | 2.32 |

- **Where the excess sits.** On 1-day-old weather the harness series is as good as or better than the stored one. The
  whole excess is on 2-day-old weather, as a negative bias: the forecast residual load is too low, meaning wind and
  solar are over-forecast.
- **This matches the residual-load study** (`../rl/RESULTS.md`). There, trees trained on near-current weather and fed
  2-day-old weather over-forecast wind by about 0.8 GW.
- **Not yet explained:** why the stored research series does not show the bias at the same weather age. It may be
  trained on different weather history; this is not established.
- **Prices are not hurt by it here.** The faithful trees match the stored trees on price.
- **Bearing on the app.** The deployed app, run as of the gate, carries a −1.3 GW residual-load bias for D hours on
  older weather. This supports the residual-load study's matched-age recommendation.

## Summary

- **Faithful versus stored.** The faithful rerun of the deployed ensemble reproduces the stored-array backtest to within
  ±0.03 ct/kWh for every member and for the blend, with every interval containing zero. The earlier ensemble
  conclusions therefore stand on a like-for-like rerun of the app code.
- **Members.** The equal-thirds blend beats each member clearly. The market route is the weakest member on its own,
  but it improves the blend.
