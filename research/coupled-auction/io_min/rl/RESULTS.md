# Residual-load forecast (DE_LU): error anatomy and candidate improvements

Scripts here: `common.py` (data, app-feature port, weather-age maps, bootstrap), `run.py` (one candidate/protocol/period),
`anatomy.py`, `score.py`, `combo.py` (post-hoc combinations), `bylead.py`, `daily_load.py` (refit-cadence check), `mix.py`
(descriptive Timestral mix). Expectations, written before each run: `EXPECTATIONS.md` (E1-E4). Outputs and tables in
/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/out/rl/ (`*.npz`, `score_*.txt`,
`combo_*.txt`, `anatomy_base_rolling.txt`, `anatomy_base_spring.txt`, `bylead.txt`, `mix_rolling.txt`, `logs/daily.log`).

## Setup
- Target: TSO day-ahead figures from energy-charts (wind on+off, solar, load); RL = load - wind - solar.
- Days: research autumn set, 66 days 2026-09-03 .. 2026-11-07 (cutoff = local midnight of D, 153 qh; night = first 24 qh,
  day 1 = first 96, h24-38 = 96..152). **Heavily used set** (notes 21.1xx): descriptive, not a sealed test.
- Gate: 12:00 local on D-1. Open-Meteo docs: `_previous_day1` = "the value predicted 24 hours before valid time", `_day2` =
  48 h before. So at the gate the 1-day-old forecast exists for the first 12 h of D; hours 12-24 of D have a forecast 24-36 h
  old; h24-38 have 36-50 h. Two maps bracket this: **gate12** = d1 for 12 h then d2 (pessimistic for hours 12-38) and
  **interp** = per-lead mix of the d1- and d2-weather predictions by age (optimistic-ish approximation). Earlier in this
  work an 8-h map ("gate8") was used; its numbers are kept in `score_rolling_gate.txt` / `combo_rolling_gate.txt` but superseded.
- Model: exact port of `residualload.generation_features` + LightGBM (lr 0.05, 31 leaves, 400 rounds), 58-point grid.
- Protocols: ROLLING = weekly refit on everything up to noon of D-2 of the refit day (app-like; checked against a daily refit
  for load, E3). STATIC = one fit on targets up to 2026-08-31. SPRING = 150 days 2026-04-01 .. 08-28, rolling with 14-day refit,
  all-d2 test weather (d1 does not exist there).
- Intervals: paired block bootstrap by day (blocks of 7, 2000 reps), candidate minus base, 5-95 %. "clear" = excludes 0.
- Not testable today: new variables (gusts, 100 m wind, cloud cover, direct/diffuse radiation), denser grid, ensemble members,
  d1 history before 2026-08-25 - Open-Meteo's API answers "Daily API request limit exceeded". Untested, not negative.

## E1 anatomy (base = deployed-style trees, rolling)
Autumn RL MAE night / day 1 / h24-38 by weather: all-d1 1.70 / 2.16 / 2.25; gate12 1.70 / 2.69 / 3.43; interp 1.70 / 2.18 / 3.09;
all-d2 2.86 / 3.27 / 3.43; archive d0 1.79 / 2.18 / 2.32.
- Components (gate8, day 1): wind 1.92 GW (bias +0.55), solar 0.87 (+0.22), load 1.05 (-0.51). Share of RL mean squared error:
  wind 49 %, solar 21 %, load 13 %, covariances +17 %. Night: wind 82 %, load 23 %. RL day-1 MAE with one part replaced by
  truth: wind -43 %, solar -16 %, load -18 %. Spring (all-d2): wind 56 %, solar 25 %, load 21 % of day-1 MSE; load bias +0.40.
- Weather vs model (MSE identity, RL): gate8 day 1 MSE 14.2 = archive-weather error 7.7 (54 %) + weather shift 12.6 + cross
  -6.1; h24-38 20.3 = 8.7 (43 %) + 16.7 - 5.1. Spring all-d2 day 1: 15.0 = 11.4 (76 %) + 11.5 - 7.9. With fresh (d1) weather the
  error is all "model error against the TSO figure" (d1 7.6 vs archive 7.7 on day 1). The weather shift is almost all wind
  (shift MAE wind 2.1, solar 0.7, load 0.2 GW). So: about half of the day-1 squared error at the gate is weather age, rising
  for later leads; the other half persists with near-perfect weather.
- Archive weather is not truth for this target: at night d1 beats archive (1.70 vs 1.79). The TSO figure is itself made from
  ~1-day-old weather.
- Systematic biases (the actionable part): (i) with 2-day-old weather the archive-trained wind model over-forecasts by
  +0.85 GW (autumn) / +0.80 GW (spring); on archive weather it is -0.26 / +0.08. (ii) Load: -0.51 GW in autumn, +0.40 in spring
  (rolling), -3.3 GW in the static fit (the `ts` feature cannot extrapolate, the static model forecasts August load for autumn).
  A daily refit leaves the rolling bias almost unchanged (-0.46, E3), so it is not the weekly cadence.
- By local hour (gate8, day 1): RL error 1.7 GW 0-6 h, 4.0-4.1 GW 10-15 h (solar 2.5-2.6 + wind on old weather), 3.4 at 18 h.
  By lead (gate12): 1.70 / 2.43 / 3.87 / 2.77 / 2.90 / 3.81 for 0-6 / 6-12 / 12-18 / 18-24 / 24-30 / 30-38 h.

## E2 RL MAE (GW), same 66 held-out days, ROLLING (app-like), diff vs base [5,95]
gate12 map (pessimistic). Columns: night / day 1 / h24-38; day-1 component MAE wind/solar/load.
| candidate | night | day 1 | h24-38 | w/s/l |
|---|---|---|---|---|
| base (deployed-style trees) | 1.70 | 2.69 | 3.43 | 1.77/0.80/1.05 |
| (a) trd2: train on d2 weather | 2.41 (+0.72 [+0.28,+1.04]) | 2.75 (+0.06 [-0.25,+0.30]) | 3.08 (-0.36 [-0.68,-0.02]) | 2.06/0.75/1.03 |
| (a) stack: archive + d2 rows (gate8) | 2.04 (+0.35 [+0.06,+0.71]) | 2.83 vs 2.88 (-0.05) | 3.31 (-0.12) | - |
| (a) trd1: d1 rows from 25 Aug, d2 before | 1.87 (+0.17 [-0.07,+0.35]) | 2.46 (-0.23 [-0.36,-0.07]) | 2.93 (-0.50 [-0.74,-0.25]) | 1.69/0.74/1.03 |
| (b) w80: + 80 m wind (gate8) | -0.02 [-0.09,+0.06] | +0.00 [-0.06,+0.07] | +0.03 | - |
| (d) split on/offshore targets (gate8) | -0.08 [-0.14,+0.03] | -0.03 [-0.09,+0.02] | -0.01 | - |
| (d) recency weights, half-life 90 d (gate8) | -0.03 | -0.00 [-0.12,+0.08] | +0.00 | - |
| nots: drop `ts` (gate8) | +0.23 [-0.01,+0.49] | -0.05 [-0.49,+0.37] | -0.11 | load 1.43 |
| doy: `ts` -> sin/cos day of year | 1.62 (-0.08 [-0.28,+0.16]) | 2.56 (-0.13 [-0.44,+0.23]) | 3.14 (-0.29 [-0.75,+0.20]) | 1.69/0.74/1.21 |
| (c) base + D-1 load residual, decay 24 h | 1.72 (+0.02 [-0.14,+0.17]) | 2.54 (-0.15 [-0.24,-0.05]) | 3.33 (-0.10 [-0.17,-0.03]) | 1.77/0.80/0.93 |
| (c) base + D-1 residual all comps, decay | 1.63 (-0.07 [-0.26,+0.16]) | 2.55 (-0.14 [-0.31,+0.04]) | 3.32 (-0.11 [-0.24,+0.03]) | 1.81/0.80/0.93 |
| POST-HOC VM: base for d1 leads, trd2 for d2 leads | 1.70 | 2.47 (-0.22 [-0.42,-0.04]) | 3.08 (-0.36 [-0.68,-0.02]) | 1.74/0.74/1.05 |
| **POST-HOC VM + load decay** | 1.72 (+0.02 [-0.14,+0.17]) | **2.36 (-0.33 [-0.55,-0.12])** | **2.98 (-0.45 [-0.79,-0.10])** | 1.74/0.74/0.92 |
| POST-HOC VM1 (trd1 for d2 leads) + load decay | 1.72 | 2.30 (-0.39 [-0.56,-0.19]) | 2.85 (-0.58 [-0.86,-0.31]) | 1.63/0.74/0.92 |

Same with the interp map (optimistic-ish): base 1.70 / 2.18 / 3.09; base + load decay 2.07 (-0.11 [-0.19,-0.02]) / 2.99 (-0.10
[-0.17,-0.04]); VM 2.13 (-0.05 [-0.08,-0.01]) / 2.81 (-0.28 [-0.56,0.00]); **VM + load decay 2.03 (-0.15 [-0.23,-0.05]) / 2.71
(-0.38 [-0.67,-0.07])**; VM1 + load decay 2.02 (-0.16) / 2.61 (-0.48). All-d1 (fully optimistic): VM reduces to base; base + load
decay -0.09 on day 1; models trained on staler weather are clearly worse on d1 weather (trd2 +0.52, stack +0.28): the
vintage gain comes only from leads that really have old weather. So the honest day-1 gain of VM + load decay is
**-0.15 to -0.33 GW (-7 % to -12 %)**, h24-38 -0.38 to -0.45 (-12 % to -13 %), night nil.

By lead (gate12 / interp), RL MAE base -> VM + load decay: 0-6 h 1.70 -> 1.72 / same; 6-12 h 2.43 -> 2.29 / same;
12-18 h 3.87 -> 3.21 / 2.54 -> 2.35; 18-24 h 2.77 -> 2.28 / 2.05 -> 1.77; 24-30 h 2.90 -> 2.39 / 2.29 -> 1.91;
30-38 h 3.81 -> 3.43 / 3.65 -> 3.30.

E3 refit cadence (load model refit daily): load bias day 1 -0.46 (weekly -0.51), load MAE 1.01 (1.05). Load decay on top of the
daily model: load 1.01 -> 0.90, RL day 1 -0.15 [-0.25,-0.05], h24-38 -0.08 [-0.14,-0.02]. The correction is not an artefact of
the weekly cadence.

STATIC (one fit to 31 Aug, gate8): base 3.15 / 4.51 / 4.73 (load bias -3.3 GW); nots 1.89 / 2.94 / 3.43; doy 1.58 / 2.75 / 3.27;
trd2 2.24 / 3.24 / 3.29 (gain mostly wind bias cancelling load bias); base + load decay 2.01 / 3.24 / 3.96; VM + load decay
2.01 / 2.68 / 2.92. Shows how fast the `ts` lag grows without refits; the app refits, so this is not the live state.

SPRING (150 days, rolling 14-day refit, all-d2 test weather): base 2.48 / 2.90 / 2.90; trd2 2.38 / 2.86 / 2.89 (RL not clear,
but wind MAE 1.99 -> 1.81 and wind bias +0.80 -> +0.06, as in autumn on d2: 2.34 -> 2.16, +0.85 -> -0.06); base + decay all comps
(pre-registered) 2.04 (-0.44 [-0.62,-0.27]) / 2.72 (-0.18 [-0.27,-0.09]) / 2.85. POST-HOC (no expectation written for these
rows): base + load decay 2.31 (-0.17) / 2.77 (-0.13 [-0.21,-0.04]) / 2.83; trd2 + load decay 2.08 (-0.40 [-0.59,-0.19]) / 2.67
(-0.23 [-0.39,-0.11]) / 2.80. In spring the whole day is d2, so "trd2 + load decay" is the VM + load decay analogue.

DESCRIPTIVE (no decision, notes 21.121): inside the deployed lead-weighted Timestral mix the tree improvements mostly wash out
(VM1 vs base trees, Timestral on 2-day-old weather, gate8: day 1 2.18 vs 2.24, -0.06 [-0.11,0.00]; h24-38 2.72 vs 2.82). Timestral's
own weather age matters more: 1-day-old instead of 2-day-old covariates, day 1 1.90 vs 2.24 (-0.34 [-0.46,-0.20]).

## Expectations: met / not met
- E1: wind >= 60 % of RL MSE on day 1: NOT MET (49 %; solar 21 %, covariances +17 %); night yes (82 %). Swap wind -40 %: MET
  (-43 %); load 10-25 %: MET (-18 %); solar < 10 %: NOT MET (-16 %). ">= 75 % of the error remains with archive weather": MET by
  MAE (76 %) but by MSE only 54 % - the MSE figure is the one to use. Load vintage-independent: MET. d1 <= d0 at night: MET.
- base static worse than rolling by 3-10 %: NOT MET - far worse (+57 %), the `ts` lag.
- trd2 -5..-12 % on day 1 clear: NOT MET (night clearly worse, h24-38 clearly better). Larger on all-d2: autumn MET
  (-10 % h24-38), spring NOT MET for RL (-1 %), though wind -9 % in both. Worse on d0: MET.
- stack between base and trd2: MET on day 1, worse at night. trd1 level with trd2: NOT MET (better than trd2, unexplained -
  trd1 also beats trd2 on d2 weather, which the vintage mechanism does not predict; `ts` may act as a vintage flag since the
  recent rows are d1. Treat VM1 as an unexplained variant).
- w80 within +-2 %: MET. split within +-3 %: MET. recency rolling 0..-5 %: MET (0); static -3..-8 %: NOT MET (+1 %).
- D-1 correction "decay": night -5..-15 %: NOT MET in autumn (-4 %, not clear), MET in spring (-18 %, clear); day 1 -2..-6 %: MET;
  "mean" not clear: MET; most gain in load: MET.
- nots rolling: NOT MET (load worse, 1.05 -> 1.43). nots static -20..-35 %: MET (-35 %). doy similar to nots in rolling: better,
  not clear.
- trd2 + nots additive: NOT MET (worse than either). E2c post-hoc VM -3..-7 %: MET (gate8 -9 %, clear); VM + load decay
  -6..-12 %: MET (gate8 -13 %); night -3..-8 %: NOT MET (+1 %). VM1 + decay within 0.1 of VM + decay: MET.
- E3 daily refit: bias shrinks to about -0.3: NOT MET (-0.46); correction still -5..-10 % of load MAE, clear: MET (-11 %).
- E4 corrected map: base day 1 about 2.6: MET (2.69); VM + load decay -6..-10 % clear: MET (-12 %, at the upper end); interp
  -3..-7 %: MET (-7 %, clear).

## Runtime data and availability at the gate
- Vintage-matched training (VM): training history of the grid weather *as forecast 2 days ahead* (Open-Meteo previous-runs
  API, `_previous_day2` of the same variables and points) for the model used on leads that only have 1.5-2-day-old weather; the
  current model (near-current weather history) for leads with the 1-day-old forecast. All available before the gate. The
  app keeps the near-current forecast as history today, so it needs a second history (one extra request set per refresh, or a
  backfill from the previous-runs endpoint) and a second model set, picked by the age of the weather for each lead.
- D-1 load residual correction: published TSO load of D-1 (out since D-2's publication) and the model's forecast of D-1 from
  weather that existed by then. No new source; a few lines in `ResidualLoad.get_data`.

## Recommendation
1. Carry into the app and a price-level backtest: **D-1 load residual correction** (add the mean load residual of the last 6 h of
   D-1, decaying exp(-lead/24 h)). Clear in every view: autumn day 1 -0.11 to -0.15 GW (both maps), daily-refit check -0.15, spring
   -0.13; load MAE -11 %. No new data. Solar must not get a flat correction (it hurt in the "mean" form).
2. Carry into a price-level backtest, then the app if it holds: **vintage-matched trees (VM)**: a second model set trained on
   2-day-old weather, used for the leads whose weather is older than ~1 day (the later hours of D and D+1). Removes a +0.8 GW
   wind over-forecast seen in both periods. With the load correction: day 1 -7 to -12 % depending on the true weather age, h24-38
   -12 to -13 %, night unchanged. Post-hoc combination on a used set; retest on fresh days (late Nov onward) and, once Open-Meteo
   is reachable, with a full-year d1 history (makes VM1 testable properly).
3. Do not carry: w80, on/offshore split, recency weights, stacking, dropping `ts` without replacement. `ts` -> day of year is a
   plausible structural fix of the load lag but was not clear in rolling and has only one year of history behind it.
4. For the two Timestral routes the trees matter little (lead-weighted mix); Timestral's covariate freshness is the bigger lever
   (descriptive).
