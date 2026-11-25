# What the trees learn: implied slopes and level ablations

## S1: cf.py net slope (EUR/MWh per GW of German demand; day and hour-month-weekend residualised; neighbour demand control; training hours 2025-10-01..2026-08-31)

|                                 | all               | low               | middle            | high              |
|:--------------------------------|:------------------|:------------------|:------------------|:------------------|
| observed price (empirical)      | 2.14 [1.92, 2.43] | 2.52 [2.22, 2.92] | 1.75 [1.42, 2.14] | 2.03 [1.46, 2.50] |
| OOF tree-fitted price, full     | 2.82 [2.60, 3.01] | 3.34 [3.10, 3.60] | 2.21 [1.95, 2.47] | 2.70 [2.28, 3.15] |
| OOF tree MAE ct/kWh, full       | 1.48              | nan               | nan               | nan               |
| OOF tree-fitted price, rlcalgas | 2.77 [2.57, 3.01] | 3.30 [2.99, 3.65] | 2.49 [2.16, 2.77] | 2.30 [1.75, 2.71] |
| OOF tree MAE ct/kWh, rlcalgas   | 1.70              | nan               | nan               | nan               |

## S2: dp/dRL by observed price band (EUR/MWh per GW; mean, and Dr^2-weighted as in slope11 in brackets)

Tree: central difference +-2 GW in the rl input of the model trained to 2026-09-01, on the last 365 training days, other inputs held.

|                                    | -1e+09..0     | 0..40         | 40..120       | 120..200      | 200..1e+09      | all           |
|:-----------------------------------|:--------------|:--------------|:--------------|:--------------|:----------------|:--------------|
| tree finite diff, full             | 2.27 (w 2.26) | 3.68 (w 3.35) | 2.19 (w 2.38) | 2.88 (w 2.58) | 11.28 (w 9.46)  | 2.77 (w 2.64) |
| tree finite diff, rlcalgas         | 2.77 (w 2.75) | 3.84 (w 3.54) | 2.22 (w 2.39) | 3.01 (w 2.71) | 12.06 (w 10.12) | 2.89 (w 2.75) |
| structural dp/dD (slope_train_old) | 5.69 (w 6.52) | 4.66 (w 5.01) | 2.08 (w 2.20) | 2.53 (w 2.42) | 30.35 (w 59.20) | 3.34 (w 3.47) |
| n tree QH                          | 2552.0        | 2566.0        | 18644.0       | 10273.0       | 1005.0          | nan           |
| n structural MTU                   | 124.0         | 127.0         | 996.0         | 541.0         | 46.0            | nan           |

Tree finite difference by the tree's own fitted-price band:

|          | -1e+09..0     | 0..40         | 40..120        | 120..200       | 200..1e+09    |
|:---------|:--------------|:--------------|:---------------|:---------------|:--------------|
| full     | 2.30 (n 1720) | 3.23 (n 3310) | 2.20 (n 18955) | 2.89 (n 10125) | 12.42 (n 930) |
| rlcalgas | 3.10 (n 1543) | 3.28 (n 3401) | 2.23 (n 18999) | 2.99 (n 10155) | 13.37 (n 942) |

## L: ablations, single-stage tree retrained per holdout day (61 days), deployed pre-gate rl, archive weather

variants: full = all frame features + rl; no_ts = without the time index; rlcalgas = calendar + gas + rl + ts; merit = calendar + gas + rl;
|raw = price target only, |ratio = price/gas target only, default = mean of both (as deployed); |frozen = trained once to 2026-09-01.

|                                |   MAE |   bias | vs full                 |
|:-------------------------------|------:|-------:|:------------------------|
| full                           | 1.965 | -0.011 | +0.000 [+0.000, +0.000] |
| full|raw                       | 2.136 | -0.258 | +0.171 [+0.062, +0.294] |
| full|ratio                     | 2.017 |  0.235 | +0.052 [-0.047, +0.155] |
| no_ts                          | 1.972 | -0.136 | +0.008 [-0.040, +0.056] |
| no_ts|raw                      | 2.174 | -0.443 | +0.210 [+0.052, +0.378] |
| rlcalgas                       | 2.260 | -0.012 | +0.295 [+0.094, +0.520] |
| merit|raw                      | 2.488 | -0.337 | +0.523 [+0.291, +0.781] |
| merit                          | 2.368 | -0.215 | +0.404 [+0.217, +0.591] |
| full|frozen                    | 2.474 | -0.363 | +0.509 [+0.210, +0.829] |
| T (deployed two-stage, stored) | 1.818 | -0.282 | -0.146 [-0.262, -0.027] |
