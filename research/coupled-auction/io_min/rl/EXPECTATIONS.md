# RL forecast experiments - expectations written BEFORE running

Set: research autumn targets, 66 days 2026-09-03 .. 2026-11-07 (cutoff = local midnight of D, 153 qh). This set has been
used heavily before (notes 21.1xx); every number here is descriptive on a used set. Gate = 12:00 local on D-1.
Gate weather: d1 for the first 32 qh of D, d2 after (bracketed by all-d1 and all-d2). Anchor reproduced before writing
this: stored archive-trained app trees, opt vintage = 1.59 / 2.48 / 3.43 (night / day 1 / h24-38), pes 2.75 / 3.52 / 3.76,
archive weather 1.68 / 2.15 / 2.26 (note: night with archive weather is WORSE than with 1-day-old weather).

Protocols: ROLLING = weekly refit on everything up to noon of D-2 of the refit day (as the app retrains each train());
STATIC = one fit on targets up to 2026-08-31. Paired block bootstrap by day (blocks of 7), 5-95 %.

## E1 anatomy (written before running anatomy.py)
- Wind carries most of the RL error: wind error variance >= 60 % of RL error variance on day 1; load 15-30 %; solar <= 10 %
  (autumn). Wind-load covariance small (|share| < 10 %).
- Swapping wind to truth cuts RL day-1 MAE by >= 40 %; swapping load by 10-25 %; solar by < 10 %.
- Weather vs model: with the archive-trained model, archive (d0) weather still leaves >= 75 % of the gate-vintage day-1 MAE
  (i.e. most of the error is "model error" against the TSO figure, not weather-forecast error). Weather age matters for wind
  only; load error nearly vintage-independent (within 5 %).
- By local hour: solar error peaks 10-15 h; wind error grows with lead (d2 part of the day > d1 part).
- Hypothesis for the night anomaly: the TSO figure is itself made from ~1-day-old weather, so d1 matches the target's
  information better than d0. Expect d1 <= d0 on night wind error.

## E2 candidates (written before running run.py), RL MAE vs base, gate weather, ROLLING unless said
- base_static vs base_rolling: static worse by 3-10 % on day 1.
- (a) trd2 (train on 2-day-old weather instead of archive): day 1 -5 to -12 % (clear), night 0 to -10 %; on all-d2 test weather
  the gain is larger (-10 to -20 %); on d0 test weather it is worse. Wind is where the gain sits.
- (a') stack (archive + d2 rows stacked, no flag): between base and trd2 (-3 to -8 %).
- (a'') trd1 (rolling only: d1 weather where it exists, 2026-08-25 on, d2 before): level with trd2 (within 3 %).
- (b) w80 (adds 80 m wind power per point, means, shear): within +-2 %, not clear.
- (d) split (onshore and offshore wind as separate targets): within +-3 %, not clear.
- (d') recency (sample weight, half-life 90 days): rolling 0 to -5 %; static -3 to -8 % (drift matters more when stale).
- (c) correction with the out-of-sample residual of D-1 (published figures of D-1 are out at the gate; residual from d1
  weather for D-1): "decay" form (last-6 h residual of D-1 per component x exp(-lead/24 h)) night -5 to -15 % (clear),
  day 1 -2 to -6 %; "mean" form (whole-D-1 mean residual, all leads) day 1 -0 to -4 %, not clear. Most of the gain in load.
- Earlier period (spring/summer 2026-04-01 .. 08-28, rolling 14-day refit, all-d2 test weather since d1 does not exist there):
  trd2 better than base by 10-20 % on day 1 (clear) - same direction as autumn.

## E2b, added after seeing base_rolling / base_static / trd2_static / recency_static and the anatomy (before running these)
Finding that prompted it: the static base has a day-1 LOAD bias of -3.3 GW (rolling: -0.5 GW), while the published load
level in autumn 2026 is not stepped (it is 1-1.5 GW BELOW autumn 2025). Diagnosis: the app's feature "ts" (time stamp)
cannot extrapolate; test rows beyond the training range fall into the latest-time leaves, i.e. the model forecasts the
load of the last training weeks (August, low) for autumn. In the rolling (app-like) form the lag is smaller but still there.
- nots (drop "ts"): rolling: load day-1 MAE -10 to -30 %, load bias within +-0.3 GW; RL day 1 -3 to -8 % (clear);
  static: RL day 1 -20 to -35 % (load bias mostly gone; remaining drift from the -1.5 GW year-on-year level).
- doy (drop "ts", add sin/cos of day of year): similar to nots in rolling (within 3 %), may carry a +1 GW load bias
  (autumn 2025 level) in static.
- the D-1 residual correction ("mean"/"decay" in score.py) on base: load error falls most (-20 to -40 % of load MAE)
  because it removes the level lag; on nots the correction gain is smaller (half or less).
- Combination to be run after singles: trd2 + nots (+ correction at scoring). Expect gains roughly additive (wind from
  trd2, load from nots/correction): RL day 1 -10 to -20 % vs base rolling on gate weather.

## E2c, post-hoc combinations, written after scoring the rolling singles above (base, trd2, stack, nots, trd2_nots) and before computing these
What the singles showed (rolling, gate): trd2 hurts the night (weather there is d1: train/test vintage mismatch) and helps the
d2 part (h24-38 -0.36, clear); dropping ts hurts load in rolling (ts acts as a recent-level tracker) but helps wind (1.92 -> 1.79).
Combinations from stored outputs (no new fits), labelled POST-HOC:
- VM (vintage-matched): base model for the leads that get d1 weather, trd2 model for the leads that get d2 weather.
  Expect: night = base (1.70), h24-38 = trd2 (3.08), day 1 between, about -0.1 to -0.2 GW vs base (-3 to -7 %), clear or nearly.
- COMP: load from base (with ts), wind and solar from nots. Expect day 1 -0.1 GW (-3 to -5 %), night level.
- VM + load-only decay correction (solar not corrected: a flat solar correction is meaningless at night, seen in "+mean").
  Expect the best of these: day 1 -6 to -12 % vs base, night -3 to -8 %.
- (added after trd1 single: gate 1.87 / 2.56 / 2.93) VM1 = base for d1 leads, trd1 for d2 leads, +decay(all): expect day 1
  about 2.40-2.50 (-13 to -17 %), h24-38 ~2.8; not clearly better than VM+decay(all) (difference within 0.1 GW).
