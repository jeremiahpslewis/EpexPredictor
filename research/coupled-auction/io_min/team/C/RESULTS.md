# Agent C results (2026-11-26): combination and Timestral

## Combination (closed as a lever)
- Bounds on 61 days, in sample (TRIALS C001):
  - fixed L1 weights: 1.637;
  - ToD-24 blocks with bias: 1.598;
  - per-day oracle weights: 1.412;
  - per-slot convex floor: 0.953.
- Champion: 1.648.
- 89% of the blend's absolute error sits in slots where all three members are on the same side of the actual (C003).
- Walk-forward weights, bias, stacking and disagreement features: 81+ variants (C004-C008). None beats equal thirds on the 61 days.
- Fourth member "SD" (D-1/D-7 curve plus RL delta): error corr 0.4-0.57, but MAE 2.9-4.2, so it hurts the blend (C009).

## Timestral: neighbour prices as past covariates (nbpast)
- Infrastructure:
  - `dump_ts.py` dumps Timestral's exact inputs from the faithful harness, imported unchanged, with leak assertions.
  - `ts_variants.py` replays variants offline.
  - Base replay equals the stored champion Timestral exactly on all 28 days used.
- 12-day screen (C011-C013), Timestral Δ:
  - nbpast: -0.055 (61 half), -0.107 (fresh half). Selected.
  - nbpast + matched RL: -0.072 / -0.155. The gains roughly add.
  - Timestral-209M: -0.027 on 4 days.
  - No RL covariate: +0.279.
- Confirmation on 16 other pre-registered days, one wave (C014):

| | champion | nbpast | diff | 90% 7-day block bootstrap | days better | Sep | Oct | Nov |
|---|---|---|---|---|---|---|---|---|
| Timestral | 2.145 | 2.078 | -0.067 | [-0.148, +0.013] | 9/16 | -0.093 | +0.021 | -0.146 |
| Blend (nbpast replaces Timestral) | 2.163 | 2.143 | -0.020 | [-0.046, +0.005] | 8/16 | -0.033 | +0.009 | -0.041 |

- Verdict: promising, not confirmed. The interval includes 0, and October is slightly worse.
- Next step: the option-B walk-forward sample, as a Timestral-only replay through `dump_ts.py --also nbpast`.
- Size: about 3-4% on Timestral, short of the 10% target. The blend gain is about 1%.

## Not done / next candidates
- Gas-day-D price as Timestral's horizon gas value. Today the gas covariate is forward-filled flat over D.
- Combine nbpast with matched RL once the matched RL is promoted.

Pooled over the 12 screen days and the 16 confirmation days (28 days). This includes the 12 days used to select the spec, so it is biased toward nbpast and is a record only:
- Timestral -0.073, 90% [-0.131, -0.011], better on 18 of 28 days.
- Blend -0.024, 90% [-0.048, -0.001].
- By month: September -0.122, October +0.035, November -0.125. October is negative in every cut.

Next screen, pre-registered: SCREEN3_PREREG.md.

## Screen 3 (2026-11-27): a negative result
Setup:
- Third pre-registered list of 12 days, all valid. These are calm days: the champion's Timestral error averages 1.52 here, against 2.1 on the confirmation days.
- Dumps used the matched RL. The gas variants were not run.

Results, Timestral against the champion:
- rlmH (matched RL over the whole horizon): +0.045, 90% [+0.014, +0.077], better on 4 of 12 days.
- nbpast (on top of matched RL): +0.090, 90% [+0.033, +0.146], better on 2 of 12 days. On the fresh days alone it is +0.245.
- In the blend: rlmH +0.001, nbpast +0.019.

What this means:
- The nbpast gain seen earlier came from days with large level misses (09-14, 11-09, 11-10). On calm days nbpast adds noise.
- Over all 44 days scored, the evidence for nbpast is mixed. I recommend it is not carried into the walk-forward as a candidate, or only as a diagnostic.
- Matched RL as Timestral's covariate does not help on these days either.
