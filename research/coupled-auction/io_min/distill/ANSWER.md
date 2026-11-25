# Why are the flexible forecasters more accurate than the eleven-zone model, and can we distill them into it?

**Short answer.**
- **The trees do learn market structure, but only in reduced form.** Their implied price response to residual load
  matches the eleven-zone curves in the mid band (40–200 EUR/MWh). At the curve ends it is much flatter: 2.5× at
  non-positive prices, about 2.7× above 200 EUR/MWh, 1.3× at 0–40.
- **Most of their accuracy advantage is not structure.** It comes from:
  - daily retraining, which absorbs level drift (a frozen copy is worse by 0.51 ct/kWh);
  - a fuel-normalised target (0.17);
  - weather and Europe-wide inputs that stand in for missing state (up to 0.30).
- **Importing the trees' slopes into the structural model does not help**, whether as a scalar or by band.
- **What helps is a learned correction of the structural forecast on a fuel-normalised scale.**
  - It improves the strictly pre-gate model by 0.48 ct/kWh when trained once before September, and by 0.58 when
    refit daily.
  - It still trails the trees by 0.47–0.57 ct/kWh and adds nothing to the deployed mix.
- **Recommendation.** Keep the eleven-zone model as a counterfactual and analysis tool with a grey-box correction.
  Use the empirical net slope (and the trees' band slopes, qualitatively) as validation targets. Do not develop it as
  a forecast member.

All numbers are MAE in ct/kWh over the 61 local days 2026-09-08..2026-11-07 (96 quarter-hours each). Brackets are
90% intervals from a paired circular 7-day block bootstrap over days. Expectations: `EXPECTATIONS.md`, plus
addendum 1 for the post hoc runs, which were written after seeing the first scoring and before running them.
Detailed tables:
- `anatomy.md`
- `trees_slope.md`
- `distill_results.md`
- `stability.md`

Scripts are in this folder; large intermediate files are in `SP/distill/`. The members reproduce the known values
exactly: T 1.818, P 1.747, A 1.891, E 2.245, E_s 2.867, M3 1.642.

## 1. Where the structural model loses (`anatomy.md`)

| | MAE | bias | daily-level MSE | within-day shape MSE | MAE after removing each day's mean error |
|---|---|---|---|---|---|
| T trees | 1.82 | −0.28 | 2.19 | 5.28 | 1.55 |
| P Timestral | 1.75 | −0.12 | 2.64 | 4.39 | 1.37 |
| A market route | 1.89 | +0.09 | 2.69 | 5.52 | 1.56 |
| E (post-gate inputs) | 2.25 | −0.26 | 2.78 | 11.18 | 1.94 |
| E_s (strict) | 2.87 | −0.16 | 5.44 | 15.07 | 2.31 |

- **E_s loses on both level and shape, but more on shape.** Its shape MSE is 15.1 against 5.3 for the trees, and
  removing each day's mean error shrinks its gap to the trees by only 27% (from 1.05 to 0.76). Expectations A1 and
  A2 are **not met**.
- **Where the gap is largest:**
  - by local hour, midday 10–16: E_s 3.76 against 1.89 for the trees (solar and negative-price hours);
  - by observed price band, non-positive prices: 7.41 against 0.62 (177 quarter-hours);
  - by forecast residual-load quintile (a pre-gate variable), the lowest quintile (−19 to 14 GW): 4.08 against 2.01;
  - the night is not where it loses. Expectation A3 is **partly met**: the low-residual-load part holds, the
    night/ramp part does not.
- **Missed solves explain little.** 85 E_s quarter-hours have no solution and score the fallback; 65 of them are at
  midday and 40 at non-positive prices.
  - Excluding them, E_s scores 6.62 at non-positive prices, against 0.64 for the trees on the same slots. The
    fallback explains about 12% of that excess (addendum expectation met).
  - Overall, solved slots only: 2.76 against 1.83 for the trees.
- **Error correlations** (expectation A4 **met**):
  - E_s with the trees: 0.37 at quarter-hour level, 0.46 for daily means.
  - Among the flexible models: 0.71–0.82.
  - Day-to-day error autocorrelation is negative for E_s and E (−0.11, −0.20) and positive for the trees (+0.14).
    This is post hoc. It fits the carried shock importing yesterday's transient into today, but it is a hypothesis:
    smoother shock forecasts (RESULTS_OOS variants c and d) did not improve price.
- **E_s is effectively persistence plus a steep slope** (expectation A5 **met**).
  - "Yesterday's same quarter-hour price + 3.15 × the change in the deployed residual-load forecast" scores 2.73.
    That is within 0.14 [−0.12, +0.38] of E_s.
  - With slope 2.15 it scores 3.09: for day-over-day changes the steeper response is better.

## 2. What the flexible models learn (`trees_slope.md`)

Single-stage LightGBM in the style of `live_test3_app.py`: frame_DE features plus the TSO residual-load forecast,
540-day window, mean of a raw-target and a gas-ratio-target model, trained before 2026-09-01.

**Implied response to residual load** (EUR/MWh per GW, by observed price band). Trees: central ±2 GW finite
difference on the last 365 training days, other inputs held. Structure: per-MTU dp/dD from `slope_train_old`.

| price band (EUR/MWh) | ≤0 | 0–40 | 40–120 | 120–200 | >200 | all, Dr²-weighted |
|---|---|---|---|---|---|---|
| trees, all features | 2.27 | 3.68 | 2.19 | 2.88 | 11.3 | 2.64 |
| trees, rl + calendar + gas | 2.77 | 3.84 | 2.22 | 3.01 | 12.1 | 2.75 |
| eleven-zone model | 5.69 | 4.66 | 2.08 | 2.53 | 30.4 | 3.47 |

- **Mid band.** In 40–200 EUR/MWh, where most hours lie, the trees and the structural model agree to within about
  15%. They disagree at the ends, consistent with the earlier finding that the structural slope gap sits at the curve
  ends. Expectation S2 is **mostly met**.
- **Like-for-like net slope.** The cf.py estimator applied to out-of-fold tree-fitted prices gives 2.82 [2.60, 3.01];
  on observed prices it gives 2.14 [1.92, 2.43]. Expectation S1 (1.8–2.8) is **marginally not met**.
  - The trees' reduced form is steeper than the data on this estimator.
  - Their partial dependences are regression slopes of a predictor on correlated inputs (residual load moves with
    weather, season and neighbour conditions), not supply-curve derivatives.
  - The empirical net slope should stay the validation target, with the tree band slopes as a qualitative check.

**Where the trees' accuracy comes from.** Single-stage tree retrained daily, deployed pre-gate residual load, archive
weather. The deployed two-stage T scores 1.82.

| variant | MAE | vs full |
|---|---|---|
| full (blend target, `ts`, weather, eu features) | 1.97 | — |
| raw target only | 2.14 | +0.17 [+0.06, +0.29] |
| ratio target only | 2.02 | +0.05 [−0.05, +0.16] |
| no time index `ts` | 1.97 | +0.01 [−0.04, +0.06] |
| calendar + gas + rl + ts (no weather) | 2.26 | +0.30 [+0.09, +0.52] |
| "merit order only": calendar + gas + rl | 2.37 | +0.40 [+0.22, +0.59] |
| same, raw target | 2.49 | +0.52 [+0.29, +0.78] |
| **full, trained once to 2026-09-01 (not retrained)** | **2.47** | **+0.51 [+0.21, +0.83]** |

- **The largest single lever is recency.** Daily retraining on a window that ends at the forecast is worth 0.51. The
  trees have no lagged-price input; they follow the drifting level (French availability, fuel costs above their
  earlier range, coupling changes) by refitting.
- **The gas-ratio target is worth 0.17** in an autumn whose fuel-cost index (166–221) lies wholly above the training
  range.
- **Weather and Europe-wide features beyond residual load are worth up to 0.30.** That is an upper bound: archive
  weather is used here, which overstates what the forecast weather would give.
- **A flexibly learned merit-order mapping** (calendar + gas + rl, pre-gate residual load, no weather) scores
  2.37–2.49. At equal information it beats the strictly pre-gate structural model (2.87) by about 0.4–0.5. Even the
  merit-order part alone is fitted better by the trees, mainly because they are flat at the ends and refit daily.
- Expectation L1 is **mostly met**. Dropping `ts` costs nothing, because the rolling window already supplies recency.

## 3. Distillation experiments (`distill_results.md`)

All are applied to E_s (strict, pre-gate) unless marked. The training-period structural forecasts come from this
study: 5,705 hourly MTUs, 2026-01..08, variant (b), with published TSO inputs. Their errors are in-sample for the
posterior.

| variant | MAE | vs E_s | vs T | M3 + it (4 equal) vs M3 |
|---|---|---|---|---|
| E_s | 2.87 | — | +1.05 [+0.77, +1.32] | +0.12 (known) |
| K1 scalar slope: p_lag + κ(E_s − p_lag), κ = 0.76 (trees' weighted finite difference / structural) | 2.91 | +0.04 [−0.10, +0.19] | +1.09 | +0.13 |
| K1, κ = training OLS 0.60 | 3.09 | +0.23 [−0.00, +0.44] | +1.28 | +0.15 |
| K1, κ = training-MAE optimum 0.85 | 2.86 | −0.01 [−0.10, +0.08] | +1.04 | +0.12 |
| K2 re-clear with the German residual-load change × 0.76 | 2.99 | +0.13 [−0.01, +0.25] | +1.17 | +0.13 |
| K3 band κ from the tree/structural band slopes (addendum, post hoc) | 2.87 | +0.00 [−0.17, +0.18] | +1.05 | +0.12 |
| R1 residual LightGBM, raw price scale, trained before 09 | 2.99 | +0.12 [−0.17, +0.48] | +1.17 | +0.12 |
| R2 same, refit daily (training + earlier holdout days) | 2.55 | −0.32 [−0.48, −0.17] | +0.73 [+0.54, +0.92] | +0.04 [−0.01, +0.10] |
| **R1s residual LightGBM on a fuel-normalised scale, trained before 09 (addendum)** | **2.39** | **−0.48 [−0.62, −0.35]** | +0.57 [+0.35, +0.80] | +0.01 [−0.03, +0.06] |
| **R2s same, refit daily (addendum)** | **2.29** | **−0.58 [−0.74, −0.42]** | +0.47 [+0.29, +0.66] | −0.01 [−0.05, +0.04] |

- **Slope distillation does not help** (K1, K2, K3). Expectation K1 is **not met**; K2 is within 0.1 of K1.
  - The choice of the 0.76 ratio was made after the holdout anatomy was visible. 2.82/3.15 ≈ 0.90 would also have
    been defensible. Both lie in the flat region around zero gain.
  - The band version (K3) does what distillation is supposed to do. It fixes the non-positive band
    (7.41 → 5.08 MAE) and loses elsewhere (morning, evening and late hours worse), so the net effect is zero.
  - The reason: the price change from yesterday that the structural model predicts is roughly right in scale (a
    steep slope beats a flat one for these changes). Its errors come from the state that changes overnight, not from
    the derivative.
- **The residual learner works once it is put on a fuel-normalised scale** (addendum, post hoc).
  - On the raw price scale (R1), the frozen correction extrapolates from the top of its training range and adds a
    −1.4 bias. Expectation R1 is **not met**.
  - Dividing the price-level features and the target by the fuel index removes that confound. R1s improves E_s by
    0.48 with almost no bias (+0.01), mostly at midday (3.76 → 2.88) and at non-positive prices (7.41 → 4.26).
  - Refitting daily adds a further 0.10 (R2s). Expectations R2 and R3 are **met**. The addendum expectation for R1s
    is **met**; R2s is within 0.1 of R2 is **not met**, because R2s is better by 0.25.
  - The fuel-normalised target is the same device that makes the trees robust.
- **None of the corrected structural forecasts improves the deployed mix** (−0.01 to +0.04). They remain 0.47–0.57
  behind the trees.
- **Using the tree forecast as a pseudo-observation for the shock (C1) was not run.** To first order it is a
  price-space blend of E_s and T, and the time-of-day weights already gave E_s weight 0–0.1.

## 4. Stability around regime changes (`stability.md`)

Single-stage trees retrained daily over 2026-04..11, using the TSO residual-load forecast and archive weather, against
the stored market route and Timestral (summer set every third day, autumn daily) and the structural (b) forecasts
(published inputs; in-sample before September).

- **Advanced hybrid coupling (first MTU 2026-06-11).**
  - The trees' error relative to the price level rose 37% in the 14 days after go-live (MAE 1.34 → 1.97, bias
    −0.41 → −0.87).
  - **This jump is exceptional for the trees.** Computing the same 14-day-after / 28-day-before ratio at every date
    from April to October puts it at the 99th percentile; the 90th percentile of that null is +15%.
  - For the market route (82nd percentile) and Timestral (93rd) the jump is less exceptional. The structural (b)
    forecast did not move (51st), because the carried shock absorbs a level shift at once.
  - Expectations B1 and the addendum null are **not met**: the trees do show a break effect here.
- **Retraining absorbs the break.** A tree frozen on 2026-06-10, compared with the daily-retrained one:

  | days after go-live | 1–14 | 15–28 | 29–42 |
  |---|---|---|---|
  | frozen − daily, MAE | +0.28 | +0.33 | +0.40 |
  | daily-retrained bias | −0.87 | −0.46 | +0.27 |
  | frozen bias | −0.89 | −1.66 | −1.09 |

  The daily-retrained tree's bias decays within about four weeks; the frozen tree's does not. Addendum expectation
  **met**. Part of the frozen gap is ordinary staleness: the September-frozen tree was 0.51 behind over the autumn.
- **Fuel-cost index leaving its training range** (daily mean above 152 from 2026-08-23; the first pass used 127 and
  2026-07-16).
  - No jump for the trees: blend −2%, raw target +5% (addendum B2' marginally met).
  - The market route (+83%) and Timestral (+52%, 5 days) jumped more; structural +19%.
  - The raw-target tree falls behind the blend by 0.16–0.19 in September and October, as the index climbs to
    170–190.
- **French nuclear unavailability.** Expectation B3 is **met**: daily MAE barely correlates with it for any model
  (trees 0.08 and 0.13, structural 0.04, market route −0.09).

**Reading.**
- The trees' non-causality does show at regime changes. After hybrid coupling went live they underpredicted by about
  0.9 ct/kWh for two weeks, an unusually large jump for them.
- Daily retraining removes that bias within about four weeks. The fuel-normalised target protects them against
  fuel-level breaks.
- For a day-ahead forecaster this is second order: the cost is about +0.6 ct/kWh MAE for a fortnight after a rare
  event, and the other flexible models show similar or larger transients.
- It is first order for counterfactuals (what if a link is added or a plant closes). There the trees have nothing to
  say, and the structural model, whose carried shock absorbed the break at once, does.

## 5. Answer

**(i) Do the flexible models capture structure? Yes, in reduced form, and that is enough to forecast better.**
- The trees reproduce the merit-order slope where data are dense (40–200 EUR/MWh) and are much flatter at the sparse
  curve ends, where the structural curves are too steep.
- What they fit is the conditional distribution of price given the information set. That includes:
  - the drifting level, through daily refitting (0.51 in this window);
  - fuel-cost scaling (0.17);
  - neighbour and availability conditions, implicitly through correlated weather and Europe-wide inputs (up to 0.30).
- The structural model must represent each of these explicitly. Where it lacks one (French availability, flexibility
  at negative prices, curve-end behaviour), the error lands in the shock, and the shock is then carried from
  yesterday.
- Its losses are concentrated at midday and negative prices, and in within-day shape, not in the daily level.
- Even restricted to the merit-order inputs, the flexible fit beats the strict structural model by about 0.4–0.5. The
  structural model's deficit is misspecified ends and missing state, not a lack of structure in the data.

**(ii) Why distillation is limited.**
- **Accuracy and structure are identified differently.** Forecast accuracy is a property of the reduced form given
  the information set; structure is identified from exogenous variation and constraints. A reduced form can only
  improve a structural model where it reveals a misspecification.
- **Here that is the curve ends,** and importing the trees' end slopes (K3) fixes the negative-price band but costs
  as much elsewhere. Price changes the structural model predicts from yesterday are not too steep on average; its
  errors come from state changes the curves do not see.
- **The trees' slopes are not causal derivatives.** On this study's own estimator they give 2.82 against 2.14
  observed.
- **Most of the trees' edge cannot be expressed as curve parameters.** It is recency, normalisation and implicit
  state. It can be copied only as a learned correction, which is what worked:
  - the fuel-normalised residual learner R1s/R2s: −0.48/−0.58;
  - but it remains a correction learned from data, not structure learned from the trees.

**(iii) Recommended path, with numbers.**
- **The best grey-box is the structural core plus a fuel-normalised LightGBM correction.**
  - Trained once before September: 2.39. Refit daily: 2.29. Strict E_s: 2.87.
  - It closes about half of E_s's gap to the trees (1.05 → 0.47) but adds nothing to the mix (1.64 stays).
- **Do not invest further in the eleven-zone model as a forecast member.**
- **Use it for counterfactuals and flows**, with the R2s-type correction if a forecast is needed for context. Validate
  its slopes against the empirical net slope (2.14 [1.92, 2.43] on training), using the trees' band slopes only
  qualitatively (mid band agrees; ends much flatter).
- **If a structural forecast member is pursued anyway,** the levers in order of evidence are:
  1. residual-load forecast quality (0.62 between E and E_s);
  2. a fuel-normalised learned correction (0.48–0.58);
  3. an explicit French availability state.

## Expectations: met / not met

| expectation | result |
|---|---|
| X0 | met |
| A1 | not met |
| A2 | not met |
| A3 | partly met |
| A4 | met |
| A5 | met |
| S1 | marginally not met (2.82) |
| S2 | mostly met |
| L1 | mostly met (`ts` not) |
| K1 | not met |
| K2 | met |
| R1 | not met |
| R2 | met |
| R3 | met |
| B1 | not met (+37%, 99th percentile of the trees' own null) |
| B2 | not met at the 2026-07-16 date |
| B3 | met |

Addendum 1:

| expectation | result |
|---|---|
| R1s | met |
| R2s within 0.1 of R2 | not met (better by 0.25) |
| K3 | not met (no gain) |
| fallback share < 30% | met (about 12%) |
| B2' | marginally met |
| B1 null | not met |
| frozen-tree gap > 0.1 | met |

## Caveats

- **Information sets.**
  - The ablation and stability trees use archive weather. The stability trees use the TSO residual-load forecast.
  - The structural training run uses published TSO inputs and is in-sample for the posterior, so the structural line
    in the stability tables is optimistic before September.
- **Post hoc.** Addendum-1 runs were designed after seeing the first scoring, so R1s/R2s and K3 are post hoc. They
  used no holdout tuning: κ bands came from training tables, and the fuel scaling is a fixed choice.
- **Intervals.** The bootstrap is circular over all 61 days. `ensemble_bt.py` resamples only the first 56 days, so
  its intervals differ slightly.
- **Runtime.** The stability run took about 24 minutes, above the 20-minute guideline, because other jobs were using
  the shared CPU. Its computation is about 10 minutes of fits.
- **Storage.** Large intermediate files are outside the repository, in `SP/distill/`: training records and solves,
  K2 solves and forecast arrays.
