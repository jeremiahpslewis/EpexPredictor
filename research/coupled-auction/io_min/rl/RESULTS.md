# Residual-load forecast (DE_LU): error anatomy and candidate improvements

Scripts here: `common.py` (data, app-feature port, gate vintage map, bootstrap), `run.py` (one candidate/protocol/period),
`anatomy.py`, `score.py`, `combo.py` (post-hoc combinations), `mix.py` (descriptive Timestral mix). Expectations, written
before each run: `EXPECTATIONS.md`. Outputs: `$SP/out/rl/*.npz`, tables `$SP/out/rl/score_*.txt`, `combo_*.txt`,
`anatomy_base_rolling.txt`, `mix_rolling.txt` (SP = the session scratchpad).

## Setup
- Target: TSO day-ahead figures from energy-charts (wind on+off, solar, load), RL = load - wind - solar.
- Days: research autumn set, 66 days 2026-09-03 .. 2026-11-07 (cutoff = local midnight of D, 153 qh; night = first 24 qh,
  day 1 = first 96, h24-38 = 96..152). **Heavily used set** (notes 21.1xx): descriptive, not a sealed test.
- Gate: 12:00 local on D-1. Weather the gate has (Open-Meteo previous-runs semantics, ~3 h run latency): the 1-day-old run
  (d1) for about the first 8 h of D (32 qh), the 2-day-old run (d2) after. This "gate" map is slightly pessimistic for the
  later hours (live, the 06Z run of D-1 is available, lead 18-42 h, between d1 and d2); "all-d1" is the optimistic bracket.
  The old research "opt" (d1 to 14 h) sits between: it gave 1.59 / 2.48 / 3.43 (anchor, reproduced).
- Model: exact port of `residualload.generation_features` + LightGBM (lr 0.05, 31 leaves, 400 rounds), 58-point grid.
- Protocols: ROLLING = weekly refit on everything up to noon of D-2 of the refit day (app-like; the app retrains at every
  train()). STATIC = one fit on targets up to 2026-08-31. Second period: SPRING = 150 days 2026-04-01 .. 08-28, rolling with
  14-day refit, all-d2 test weather (d1 does not exist there).
- Intervals: paired block bootstrap by day (blocks of 7, 2000 reps), candidate minus base, 5-95 %. "clear" = excludes 0.
- Not testable today: new variables (gusts, 100 m wind, cloud cover, direct/diffuse radiation), denser grid, ensemble
  members, d1 training history before 2026-08-25 - Open-Meteo answers "Daily API request limit exceeded". These are
  untested, not negative.

## E1 anatomy (base, rolling, autumn)
RL MAE night / day 1 / h24-38 by test weather: gate 1.70 / 2.88 / 3.43; all-d1 1.70 / 2.16 / 2.25; all-d2 2.86 / 3.27 / 3.43;
archive d0 1.79 / 2.18 / 2.32.
- Components, day 1, gate: wind 1.92 GW (bias +0.55), solar 0.87 (+0.22), load 1.05 (-0.51). Share of RL mean squared error:
  wind 49 %, solar 21 %, load 13 %, covariances +17 % (all positive: errors compound). Night: wind 82 %, load 23 %.
  RL day-1 MAE with one part replaced by truth: wind 1.65 (-43 %), solar 2.43 (-16 %), load 2.36 (-18 %).
- Weather vs model: with archive weather the model still has 76 % of its gate day-1 MAE (2.18 of 2.88), 68 % in h24-38,
  105 % at night. So ~3/4 of the day-1 error is "model error against the TSO figure", ~1/4 is the age of the weather - but
  the whole difference between gate and all-d1 (2.88 vs 2.16) is the d2 part of the day: weather age matters a lot once the
  run is two days old. Weather-age shift is almost all wind (shift MAE wind 2.1, solar 0.7, load 0.2 GW).
- Night anomaly confirmed: d1 weather beats archive weather at night (1.70 vs 1.79) and is level over day 1 (2.16 vs 2.18).
  Reading: the TSO figure is itself made from ~1-day-old weather; archive is not "truth" for this target.
- Systematic biases (the actionable part): (i) with 2-day-old weather the archive-trained wind model over-forecasts by
  +0.85 GW (autumn) / +0.80 GW (spring); with archive weather it is -0.26. (ii) Load: -0.51 GW in autumn, +0.40 in spring
  (rolling); -3.3 GW in the static fit. Cause: the app feature `ts` cannot extrapolate, so the load model forecasts the
  level of its last training weeks (late-summer low for autumn). Rolling refits shrink but do not remove the lag; the sign
  follows the season (load rising in autumn, falling in spring).
- By local hour (gate, day 1): RL error 1.7 GW at 0-6 h, peaks 4.0-4.1 GW at 10-15 h (solar 2.5-2.6 + wind on d2), 3.4 at 18 h
  (load 1.47 peak).

## E2 RL MAE (GW), same 66 held-out days, ROLLING (app-like), gate weather
| candidate | night | day 1 | h24-38 | day-1 comps w/s/l |
|---|---|---|---|---|
| base (deployed-style trees) | 1.70 | 2.88 | 3.43 | 1.92/0.87/1.05 |
| (a) trd2: train on d2 weather | 2.41 (+0.72 [+0.28,+1.04]) | 2.83 (-0.04 [-0.41,+0.25]) | 3.08 (-0.36 [-0.68,-0.02]) | 2.11/0.81/1.04 |
| (a) stack: archive + d2 rows | 2.04 (+0.35 [+0.06,+0.71]) | 2.83 (-0.05 [-0.26,+0.22]) | 3.31 (-0.12 [-0.34,+0.21]) | 2.04/0.80/1.01 |
| (a) trd1: d1 rows where they exist (from 25 Aug), d2 before | 1.87 (+0.17 [-0.07,+0.35]) | 2.56 (-0.32 [-0.53,-0.12]) | 2.93 (-0.50 [-0.74,-0.25]) | 1.75/0.80/1.04 |
| (b) w80: + 80 m wind | 1.68 (-0.02 [-0.09,+0.06]) | 2.88 (+0.00 [-0.06,+0.07]) | 3.46 (+0.03) | 1.92/0.87/1.05 |
| (d) split on/offshore targets | 1.62 (-0.08 [-0.14,+0.03]) | 2.84 (-0.03 [-0.09,+0.02]) | 3.43 (-0.01) | 1.88/0.87/1.05 |
| (d) recency weights (half-life 90 d) | 1.67 (-0.03) | 2.88 (-0.00 [-0.12,+0.08]) | 3.43 (+0.00) | 1.95/0.93/1.05 |
| nots: drop `ts` | 1.92 (+0.23 [-0.01,+0.49]) | 2.83 (-0.05 [-0.49,+0.37]) | 3.32 (-0.11) | 1.79/0.88/1.43 |
| doy: drop `ts`, add sin/cos day of year | 1.62 (-0.08 [-0.28,+0.16]) | 2.60 (-0.28 [-0.67,+0.14]) | 3.14 (-0.29 [-0.75,+0.20]) | 1.82/0.80/1.20 |
| (c) base + D-1 load residual, decay 24 h | 1.72 (+0.02 [-0.14,+0.17]) | 2.71 (-0.17 [-0.27,-0.07]) | 3.33 (-0.10 [-0.17,-0.03]) | 1.92/0.87/0.93 |
| (c) base + D-1 residual all comps, decay | 1.63 (-0.07 [-0.26,+0.16]) | 2.74 (-0.14 [-0.33,+0.07]) | 3.32 (-0.11) | 1.99/0.87/0.93 |
| (c) base + D-1 mean residual, all leads | 1.80 (+0.10) | 2.81 (-0.07 [-0.31,+0.19]) | 3.21 (-0.23 [-0.48,+0.01]) | 2.06/1.11/0.94 |
| POST-HOC VM: base for d1 leads, trd2 for d2 leads | 1.70 (0) | 2.62 (-0.26 [-0.55,-0.02]) | 3.08 (-0.36 [-0.68,-0.02]) | 1.87/0.81/1.05 |
| POST-HOC VM1: base for d1 leads, trd1 for d2 leads | 1.70 (0) | 2.52 (-0.36 [-0.55,-0.16]) | 2.93 (-0.50 [-0.74,-0.25]) | 1.72/0.79/1.03 |
| POST-HOC VM1 + load decay | 1.72 (+0.02 [-0.14,+0.17]) | **2.42 (-0.46 [-0.72,-0.20])** | **2.85 (-0.58 [-0.86,-0.31])** | 1.72/0.79/0.91 |
| POST-HOC VM + load decay | 1.72 (+0.02) | 2.51 (-0.37 [-0.69,-0.09]) | 2.98 (-0.45 [-0.79,-0.10]) | 1.87/0.81/0.92 |

All-d1 bracket (optimistic weather, `score_rolling_d1.txt`): base 1.70 / 2.16 / 2.25; base + decay 2.07 day 1; models trained
on staler weather are clearly worse there (trd2 +0.52, stack +0.28) - the gains above come only from the d2 part of the day.

STATIC (one fit to 31 Aug, same days, gate): base 3.15 / 4.51 / 4.73 (load bias -3.3 GW); nots 1.89 / 2.94 / 3.43;
doy 1.58 / 2.75 / 3.27; trd2 2.24 / 3.24 / 3.29 (gain mostly wind bias cancelling load bias); base + load decay 2.01 / 3.24 / 3.96;
VM + load decay 2.01 / 2.68 / 2.92. Shows how fast the `ts` lag grows when the model is not refitted.

SPRING (150 days, rolling 14-day refit, all-d2 test weather): base 2.48 / 2.90 / 2.90; trd2 2.38 / 2.86 / 2.89 (RL not clear,
but wind MAE 1.99 -> 1.81 and wind bias +0.80 -> +0.06, same as autumn d2: 2.34 -> 2.16, +0.85 -> -0.06); base + load decay
2.31 / 2.77 (-0.13 [-0.21,-0.04]) / 2.83; trd2 + load decay 2.08 (-0.40 [-0.59,-0.19]) / 2.67 (-0.23 [-0.39,-0.11]) / 2.80;
base + decay all comps 2.04 (-0.44 [-0.62,-0.27]) / 2.72 (-0.18 [-0.27,-0.09]).

DESCRIPTIVE (no decision, notes 21.121): inside the deployed lead-weighted Timestral mix the tree improvements mostly wash
out (VM1 vs base trees, Timestral on 2-day-old weather: day 1 2.18 vs 2.24, -0.06 [-0.11,0.00]; h24-38 2.72 vs 2.82). Timestral's
own weather age matters more: Timestral on 1-day-old instead of 2-day-old covariates, day 1 1.90 vs 2.24 (-0.34 [-0.46,-0.20]).

## Expectations: met / not met
- E1: wind >= 60 % of RL MSE on day 1: NOT MET (49 %; solar 21 %, covariances +17 %); night yes (82 %). Swap wind -> -40 %: MET
  (-43 %); load 10-25 %: MET (-18 %); solar < 10 %: NOT MET (-16 %). Archive leaves >= 75 % of the error: MET (76 %); load
  vintage-independent: MET. d1 <= d0 at night: MET.
- base static worse than rolling by 3-10 %: NOT MET - far worse (+57 %), the `ts` lag.
- trd2 -5..-12 % on day 1 clear: NOT MET (-1 %, not clear; night clearly worse, h24-38 clearly better). Larger on all-d2:
  autumn MET (h24-38 -10 %), spring NOT MET for RL (-1 %) though wind -9 % in both. Worse on d0: MET.
- stack between: roughly MET on day 1 (-2 %), but worse at night. trd1 level with trd2: NOT MET (clearly better than trd2).
- w80 within +-2 %: MET. split within +-3 %: MET. recency rolling 0..-5 %: MET (0); static -3..-8 %: NOT MET (+1 %).
- correction decay: night -5..-15 %: NOT MET in autumn (-4 %, not clear; load-only +1 %); MET in spring (-18 %, clear, all comps);
  day 1 -2..-6 %: MET (-5 to -6 %, load-only clear); mean form not clear: MET; most gain in load: MET.
- nots rolling: NOT MET (load worse 1.05 -> 1.43; in the rolling app form `ts` acts as a recent-level tracker). nots static
  -20..-35 %: MET (-35 %). doy similar to nots in rolling: MET-ish (better, not clear).
- trd2 + nots additive: NOT MET (worse than either). Post-hoc VM -3..-7 %: MET (-9 %, clear). VM + load decay -6..-12 %:
  MET (-13 %, clear); night -3..-8 %: NOT MET (+1 %). VM1 + decay within 0.1 of VM + decay: MET (0.05).

## Runtime data and availability at the gate
- Vintage-matched training (VM/VM1): needs the training history of the grid weather *as forecast 1 and 2 days ahead*
  (Open-Meteo previous-runs API, `_previous_day1/2` variables, same grid and variables), plus the live forecast at the gate.
  All available before the gate. The app today keeps the near-current forecast as history (archive-like), so it needs a
  second store (or the previous-runs endpoint) and two model sets (or one per weather age), selected by lead.
- D-1 load residual correction: needs the published TSO load figure for D-1 (published on D-2, available) and the model's
  forecast of D-1 from weather that existed at the gate (d1). No new source.
- doy instead of `ts`: no new data.

## Recommendation
1. Carry into a price-level backtest first: **vintage-matched trees (VM1 form: archive/d1-trained model for leads with
   1-day-old weather, d2-trained (or d1-where-available) model for later leads) + load-only D-1 residual correction (decay
   24 h)**. Day 1 2.88 -> 2.42 GW (-16 %, [-0.72,-0.20]), h24-38 3.43 -> 2.85, night unchanged; the load correction is clear in
   both periods and the d2-trained wind model removes a +0.8 GW wind over-forecast in both periods. Caveats: VM1 is post-hoc
   on a used set; trd1's d1 history covers only 25 Aug onward; the gate map's later hours are slightly too stale.
2. The cheapest single change for the app: the load-only D-1 residual correction (-0.17 autumn, -0.13 spring on day 1, both
   clear; no new data). It treats the `ts` level lag; replacing `ts` by day-of-year is the structural fix but was not clear in
   rolling (doy -0.28, interval [-0.67,+0.14]) and needs more than one year of history to trust.
3. Do not carry: w80, on/offshore split, recency weights, stacking, dropping `ts` without a replacement.
4. For the Timestral routes the bigger lever is Timestral's covariate freshness, not the trees (descriptive).
5. Once Open-Meteo is reachable: fetch `_previous_day1` for the whole year (makes trd1 a full-year d1 model) and retest VM1
   on a fresh period (late Nov onward) rather than on these 66 days.
