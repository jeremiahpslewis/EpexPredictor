# S: error anatomy and lever ranking (2026-11-26, v1)

Source: stored champion arrays, `out_strict` (61 days 09-08..11-07) and `out_fresh_strict` (18 days 11-09..11-26).
Scripts: `S/anatomy.py`, `S/sizing.py` (seconds, offline). Everything here is post hoc on scored days: sizing, not gains.
`sizing.py` scores 60 of the 61 days (10-25, the 100-slot day, is dropped) after a 14-day warm-up.

## Where the blend error sits (share of absolute error)

| | 61 days (MAE 1.648) | fresh 18 (MAE 2.173) |
|---|---|---|
| 00-06 local | 15% (MAE 0.98) | 13% (1.08) |
| 06-10 | 17% (1.67) | 19% (2.53) |
| 10-16 | 28% (1.87) | 33% (2.86) |
| 16-21 | 30% (2.40); hour 19 alone 9% (3.49) | 26% (2.67) |
| 21-24 | 10% (1.25) | 10% (1.67) |
| slots on 2-day-old weather (74% of slots) | 84% (1.88 vs 0.98 on 1-day-old; confounded with night) | 87% |
| actual price <= 3 ct (9% of slots) | 8% | 0% |
| actual 3-13 ct (18%) | 20%, bias +0.9 below 8 ct | 6% |
| actual > 13 ct (74%) | 72%, bias -0.5 | 94%, bias -1.2 |
| tree-RL error < -2 GW (RL too low; 33% / 43% of slots) | 41%, price bias -1.5 | 51%, bias -2.0 |
| worst 10 days | 29% | (worst 3 of 18: 36%) |
| Mon / Fri (61) ; Mon (fresh) | 1.98 / 1.88 vs 1.37-1.45 Sat-Sun-Tue | Mon 3.80, bias -3.7 |

- Member errors are highly correlated: 0.77-0.85 (61 days), 0.84-0.94 (fresh).
- The error is a **day-by-window level miss shared by all members**: removing the per-day mean error (oracle) gives
  1.40; removing it per day and 5 windows gives 0.90 (61 days) and 0.82 (fresh). Shape inside a window is the minority.
- Fresh days: a persistent under-forecast in daytime (bias -1.5 to -2.3 from 08 to 19 h) at a price level 5.6 ct higher.

## Lever sizes

| lever | evidence | size on blend | cost |
|---|---|---|---|
| Re-weighting the 3 members | in-sample best fixed weights 1.638; per-hour weights+intercept in sample 1.605; per-day oracle member 1.451. Causal: LS 14/28 d +0.03..+0.04 worse, inverse-MAE flat | **about 0** (≤ 2.6% even in sample) | seconds |
| Rolling bias correction of the blend | causal, hourly, 1-14 d, shrink 0.5/1.0: all flat or worse (+0.02..+0.5), both windows | **0 or negative** | seconds |
| Residual load | (a) `matched` variant: RL 2.56→2.29, blend -0.061 [-0.102,-0.018] (61 d), -0.031 [-0.089,+0.017] (fresh), pooled 79 d -0.054 [-0.091,-0.015]; (b) linear proxy with the true RL error (not a bound): -0.18 (61 d), -0.09 (fresh); I's true-RL feed is the real bound | **3-4% in hand, about 10% ceiling** | minutes per day |
| Fresher weather at the gate | live uses the latest run; the backtest uses 2-day-old weather on 74% of slots | unknown; part of the RL ceiling. Measured live from 11-27 | none (log) |
| New gate-known information for the day-by-window level (neighbour RL/prices, FR nuclear, gas/CO2 anchor, outages, D-1 curve shape) | oracle day-window shift 1.65→0.90 | **the only place 20% can come from**; realised fraction unknown | features: minutes; member refit: per-day harness time |
| Negative/low price regime | 8% of error (61 d), 0% fresh/winter | ≤ 2% | low priority until spring |
| Evening peak 17-20 h | 20% of error in 3 hours (61 d) | a few % if a scarcity signal exists | with the item above |

## Ranking by expected gain per compute-minute

1. **Day-by-window level diagnosis (P + I, offline first).** Regress the stored member errors by day-window on
   gate-known candidates; only signals that explain error there go into a member refit.
2. **Residual load (I).** `matched` is in hand; published TSO load on D and the true-RL bound next.
3. **A less correlated member / Timestral covariates (C).** The blend can only gain from diversity it does not have.
4. Evening-peak and Monday targeted features (P), inside 1.
5. Negative-price regime: parked.
6. Blend weights and bias correction: **closed** (logged in TRIALS.md).

## Update 16:30 UTC (results from I, C, E)

- **True-RL bound (I, TRIALS I001; 9 days of the 61, chosen before scoring):** blend 1.808 → 1.583, -0.225
  [-0.468, +0.068] (-12.5%); trees -0.408 (-20%), Timestral -0.238 (-13%), market route -0.047 (about 0; worse on 3 of
  9 days). "True" here means the TSO day-ahead figures. `matched` realises about -0.06 of it. The RL lever stays open;
  the market route does not profit from better RL, which is a model question for P.
- **Cross-border and fuel drivers as linear corrections of the day-window miss (I, I002; 79 candidates):** none helps
  causally (-0.001..+0.12). Closed in that form.
- **89% of blend error** is on slots where all three members are on the same side of the actual (C003). A similar-day
  fourth member is decorrelated (0.4-0.57) but too weak (MAE 2.9-4.2) and hurts the blend (C009). Dropped.
- **Mondays:** 11 Mondays carry 19% of blend error; all members miss low together (C). Passed to P.
- **Forecast RL ≥ 50 GW:** 4% of slots, MAE above 5 for every member, about 13% of error (P anatomy). Gate-known.
- **Weather age:** live weather is 24-44 h lead on D; the strict backtest uses 48 h+ on 74% of slots. The backtest
  champion is therefore pessimistic about live RL quality; the gap is measured live from 11-27.

Revised ranking: (1) residual load: published load, weather age, wind/solar models (ceiling about 12% of the blend,
20% of trees); (2) tree/market-route structure for Mondays, the ≥ 50 GW regime and hours 18-21, and the market
route's use of RL; (3) Timestral variants on replayed inputs; (4) everything on weights, bias correction, stacking,
similar-day member, linear cross-border corrections: closed.

**Blend target check.** Stacking the levers at their bounds: RL oracle -12.5%, plus member structure fixes of unknown
size. The 20% blend target is not reachable from RL alone even with perfect RL; it needs (2) to deliver on top.

## Update 17:10 UTC (I's decomposition, P's and C's first screens)

- **Residual load, decomposed (I003, 4 high-miss days):** true load alone (= `published`) moves the blend by +0.005;
  true wind and solar alone by -0.288 (4 of 4 days). Published load is parked as a price lever.
- **Room beyond `matched` (same 9 days):** oracle minus `matched` is -0.069 on the blend (trees -0.122, Timestral
  -0.179, market route -0.018). `matched` already holds most of the RL gain; what is left is about 0.07-0.15 ct and
  sits in wind, mostly through Timestral's own RL path.
- **TSO day-ahead wind/solar for D is not gate-known** (ENTSO-E probe 16:26 UTC on 11-26: DE wind/solar for 11-27
  absent, load present).
- **Trees, structural "shift" target with L1 (P, screen on fresh18, weekly refits, best of about 8 variants):** 2.420 →
  2.165, -0.255 [-0.494, -0.028], largest in 16-21 h. Unconfirmed; fresh18 is now its selection set.
- **Timestral with neighbours' price history as past covariates (C011, 4 days of the 61):** 1.935 → 1.802, blend
  -0.039. Unconfirmed; 12-day screen running, 20-day confirmation designed.
- **Market route:** gains nothing from true RL and gets worse when Timestral's RL is also true; it double-corrects (P).

Ranking now: (1) tree shift target (P) and Timestral neighbour covariates (C): confirm; (2) `matched` RL as the base
for both; (3) wind boosters offline, then more grid points (I, open); (4) market-route RL consistency (P).

**Honest total.** `matched` -0.06, remaining RL ≤ -0.07..-0.15, plus the two unconfirmed member changes at blend level
(a third of each member gain at best, less if correlation rises): the visible sum is about -0.15 to -0.25 ct,
9-15% of 1.648. The 20% blend target (1.32) is not in sight with the levers found so far.
