# Agent P (price members: trees and market route): results, 2026-11-26

All numbers are MAE in ct/kWh over the first 96 slots of each day. Comparisons are paired; intervals are 90% circular
7-day block bootstrap. All windows were looked at before: "61" is 09-08..11-07 and "fresh18" is 11-09..11-26. Trials
are logged in `../TRIALS.md` as P001-P011.

## Verdict

- **No change to either member is promotable.**
- **Trees:** shiftl1 and fclog both failed their pre-registered tests. The trees target (≤ 1.61) is not reached.
- **Market route:** one lead, "consist": -0.370 on 6 days, blend -0.104. It needs E's walk-forward sample before it
  counts. The market target (≤ 1.69) is not established.

## Trees

| Trial | Variant | Window / cadence | MAE | vs control | Verdict |
|---|---|---|---|---|---|
| P003 | shiftl1 (supply-shift target, L1, cleared via marketmodel) | fresh18, weekly | 2.165 | -0.255 [-0.494, -0.028] vs weekly base 2.420 | selection set |
| P007 | shiftl1, frozen spec | 61, refit every 8 d | 1.988 | +0.074 [-0.084, +0.219] vs 1.914 | **dropped** |
| P007 | shiftl1 | fixed days, daily, 61 half / fresh half | 2.191 / 2.196 | +0.332 (wins 2/6) / +0.179 (wins 1/6) | **dropped** |
| P009 | fclog1 (log of forecast-time rows, w = 1) | fresh18, daily stage-2 proxy | 2.381 | +0.162 [+0.010, +0.314] vs base@d2 2.219 | **dropped** |
| P009 | fclog5 (w = 5) | fresh18 | 2.416 | +0.197 [+0.104, +0.291] | **dropped** |
| P009 | fclog1 | 61: all / 1st half / 2nd half | 1.809 | +0.099 [+0.005, +0.204] / +0.202 / -0.002 | **dropped** |
| P009 | fclog5 | 61 | - | cut by the 15-min timeout, not run | - |

- **Blend, fclog1 replacing the trees (fresh18, members with matched RL):** +0.030 [-0.039, +0.117] vs the champion
  blend. Error correlation with Timestral is 0.82, against 0.85 for base@d2.
- **Blend, weekly shiftl1 (fresh18):** -0.058. This is superseded by P007.
- **Method (P008).** Weekly refit is not a valid ranker of tree variants:
  - the weekly base is 0.13-0.29 worse than daily, and the sign of shiftl1 flipped between the two;
  - so the weekly drops in P002 (L1, recency τ 90 d, no ts, gas-only, price lags, a same-day-type anchor for Monday)
    are inconclusive, not negative;
  - at daily cadence, `dump.py` reproduces the champion trees exactly.
- **Daily stage-2 proxy `@d2`.** The control: stage 1 weekly, stage 2 daily with the forecast-time frames since the
  origin.
  - Base RL: 1.799 vs the champion's 1.787.
  - Matched RL: 1.710 on 61 days, -0.077 [-0.161, +0.003] vs the champion trees (base RL), in line with `matched`
    (T000).
- **Anatomy (P001, `anatomy.md`).**
  - The tree level miss tracks the RL bias (correlation 0.63); no gate-known price variable explains it.
  - The 18-21 h ramp and forecast RL ≥ 50 GW are the worst regimes.
  - Monday misses are not fixed by price lags or anchors.
- **Live caveat.** In live operation the forecast-time rows come from live weather at 24-44 h lead, not the strict
  vintages. Not relevant now that fclog is dropped.

## Market route

| Trial | Variant | Days | MAE (variant vs stored) | Diff | Wins | Blend diff |
|---|---|---|---|---|---|---|
| P010 | repro (`market_ts.py` re-runs Timestral on the shift) | 09-20, 11-10 | 3.264 vs 3.264 | 0.000 (max abs 0.000) | - | 0.000 |
| P010 | gas1: gas lag 1 d (gas day D-1) in history and clearing | 11 (5 from 61, 6 fresh) | 2.119 vs 2.089 | +0.030 | 5/11 | +0.019 |
| P011 | consist: past RL in shift history and covariate = forecast-time RL read at each earlier gate | 6 fresh (11-10 … 11-25) | 2.087 vs 2.457 | **-0.370** | 5/6 | **-0.104** (wins 4/6) |

- **consist per day:** 11-10 -0.64, 11-13 -0.08, 11-16 -1.69, 11-19 -0.08, 11-22 +0.59, 11-25 -0.31. One day carries
  a large share, and there are 6 days, so there is no interval.
- **Mechanism.** The shift the route forecasts was built from published RL, while D's demand is forecast RL. That
  mismatch puts the RL forecast error into the shift twice. P006 found the same thing offline: with the shift held,
  true RL gives -0.095 and half-true RL -0.166.
- **Deployability.** It needs the forecast-time RL of the ~85 context days before D. The app writes a load-forecast
  log already (priceapi `loadforecastlog`), so live it would accumulate from now. In the backtest, the stored faithful
  days start 09-08; older context days keep the published RL.
- **Dropped heuristics (P005).** 10 stored-array shift or demand changes, all worse.

## Next

1. **consist on E's walk-forward days.** It needs forecast-time RL for each sample day's 85-day context.
   - That is a cache-wide RL log, either from E's champion cache or a cheap RL-only replay per context day.
   - For the 12 pre-September sample days the vintages are missing (E), so it can only be tested where a stored RL
     history exists: the 4 Sep-Nov sample days plus more fresh days.
2. **Tree changes at daily cadence only.** This means the exact daily dump (about 190 s/day at 1 thread, matched RL)
   or the @d2 proxy, never weekly.
