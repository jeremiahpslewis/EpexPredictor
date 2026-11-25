# Why are the flexible forecasters more accurate than the eleven-zone model, and can we distill them into it?

**Short answer.**
- **The trees do learn market structure, but only in reduced form.** Their implied price response to residual load
  matches the eleven-zone curves in the mid band (40–200 EUR/MWh). It is 2–3 times flatter at the curve ends.
- **Most of their advantage is not structure.** It comes from three things:
  - daily retraining, which absorbs level drift (a frozen copy is worse by 0.51 ct/kWh);
  - a fuel-normalised target;
  - not anchoring on yesterday's shock.
- **Distillation did not close the gap.** Every grey-box variant trained only on pre-September data was no better
  than the strict structural model, or worse. Slope recalibration did not help either. Only a correction that is
  refit every day on the structural model's own recent errors helped: −0.32 ct/kWh. It is still 0.73 ct/kWh behind
  the trees and adds nothing to the deployed mix.
- **Recommendation.** Keep the eleven-zone model as an analysis and counterfactual tool, and the trees as a slope
  check on it. Do not develop it as a forecast member.

All numbers are MAE in ct/kWh over the 61 local days 2026-09-08..2026-11-07 (96 quarter-hours each). Brackets are
90% intervals from a circular 7-day block bootstrap over days, paired. Expectations were written to `EXPECTATIONS.md`
before the runs. Detailed tables:
- `anatomy.md`
- `trees_slope.md`
- `distill_results.md`
- `stability.md`

Scripts are in this folder; large intermediate files are in `SP/distill/`. The members reproduce the known values
exactly: T 1.818, P 1.747, A 1.891, E 2.245, E_s 2.867, M3 1.642.

## 1. Where the structural model loses (`anatomy.md`)

| | MAE | bias | RMSE | daily-level share of MSE | MAE after removing each day's mean error |
|---|---|---|---|---|---|
| T trees | 1.82 | −0.28 | 2.73 | 0.29 | 1.55 |
| P Timestral | 1.75 | −0.12 | 2.65 | 0.38 | 1.37 |
| A market route | 1.89 | +0.09 | 2.87 | 0.33 | 1.56 |
| E (post-gate inputs) | 2.25 | −0.26 | 3.74 | 0.20 | 1.94 |
| E_s (strict) | 2.87 | −0.16 | 4.53 | 0.27 | 2.31 |

- **It loses mostly on within-day shape and at the extremes, not on the daily level.**
  - E_s's shape MSE is 15.1 against 5.3 for the trees. Its level MSE is 5.4 against 2.2.
  - Its RMSE/MAE ratio (1.58) shows heavy tails.
  - Expectation A1 (that E_s loses mainly on level) is **not met**.
- **By observed price band, the biggest gap is at non-positive prices.** E_s MAE 7.41 against 0.62 for the trees
  (177 quarter-hours). At 0–5 ct/kWh it is 4.26 against 2.10.
- **By forecast residual-load quintile** (a pre-gate variable), the biggest gap is in the lowest quintile (−19 to
  14 GW): 4.08 against 2.01.
- **By local hour,** the largest excess is midday 10–16 (3.76 against 1.89), the solar and negative-price hours. It
  is not at night. Expectation A3 is **partly met**: the low-residual-load part holds, the night/ramp part does not.
- **The gap shrinks after removing the daily mean, but only by 27%** (from 1.05 to 0.76). Expectation A2 is
  **not met**.
- **Error correlations** (expectation A4 **met**):
  - E_s with the trees: 0.37 at quarter-hour level, 0.46 for daily means.
  - Among the flexible models: 0.71–0.82.
  - E_s errors have negative day-to-day autocorrelation (−0.11; E −0.20), while the trees' are positive (+0.14).
    This is the signature of carrying yesterday's transient shock into today.
- **E_s is effectively persistence plus a slope.**
  - "Yesterday's same quarter-hour price + 3.15 × the change in the deployed residual-load forecast" scores 2.73.
    That is within 0.14 [−0.12, +0.38] of E_s. With slope 2.15 it scores 3.09. Expectation A5 is **met**.
  - So the eleven-zone machinery adds little beyond a steep local slope around yesterday's outcome.
  - Even here, the steeper 3.15 slope beats the empirical 2.15 for day-over-day changes. This anticipates why slope
    flattening fails in section 3.

## 2. What the flexible models learn (`trees_slope.md`)

Single-stage LightGBM in the style of `live_test3_app.py`: frame_DE features plus the TSO residual-load forecast,
540-day window, mean of a raw-target and a gas-ratio-target model.

**Implied response to residual load** (EUR/MWh per GW). Trees: central ±2 GW finite difference on the last 365
training days, other inputs held. Structure: per-MTU `slope_train_old` dp/dD.

| observed price band (EUR/MWh) | ≤0 | 0–40 | 40–120 | 120–200 | >200 | all, Dr²-weighted |
|---|---|---|---|---|---|---|
| trees, all features | 2.27 | 3.68 | 2.19 | 2.88 | 11.3 | 2.64 |
| trees, rl + calendar + gas | 2.77 | 3.84 | 2.22 | 3.01 | 12.1 | 2.75 |
| eleven-zone model | 5.69 | 4.66 | 2.08 | 2.53 | 30.4 | 3.47 |

- **The net-slope estimator applied to out-of-fold tree-fitted prices** (cf.py recipe) gives 2.82 [2.60, 3.01]. On
  observed prices it gives 2.14 [1.92, 2.43].
  - Expectation S1 (1.8–2.8) is **marginally not met**.
  - The trees' reduced form sits between the empirical value and the structural one.
- **Mid band.** In 40–200 EUR/MWh, where most hours lie, the trees and the structural model agree to within about
  15%. The trees have learned the merit-order slope there.
- **Curve ends.** They disagree at the ends: the structural curve is 2.5× steeper at non-positive prices, 1.3× at
  0–40, and about 2.7× above 200. This independently confirms the earlier finding that the structural slope error
  sits at the curve ends. Expectation S2 is **mostly met**: steeper at high prices; the slopes with and without
  weather are close.
- **These partial dependences are not causal effects.** The `rl` input is correlated with the weather inputs, so
  "residual load +2 GW, weather held" is off the data manifold. The rl + calendar + gas model avoids most of that
  and gives a similar answer. The slope readings are therefore reasonably robust, but they are still regression
  slopes of a predictor, not supply-curve derivatives.

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

- **The largest single lever is recency.** Daily retraining on a window that ends at the forecast is worth 0.51.
  - The trees have no lagged-price input. They track the drifting level (French availability, fuel costs above
    their earlier range, coupling changes) by refitting.
  - This is the reduced-form counterpart of the structural model's carried shock, and it works better. It updates
    a whole mapping smoothly, instead of copying yesterday's transient into today.
- **The gas-ratio target is worth 0.17** in an autumn whose fuel-cost index (166–221) is wholly above the training
  range.
- **Weather and Europe-wide features beyond residual load are worth 0.30.** They carry neighbour and availability
  effects that the structural model needs explicit state variables for.
- **The pure merit-order mapping, even learned flexibly, scores 2.37–2.49.** That is no better than the structural
  model with post-gate inputs (2.25).
- Expectation L1 is **mostly met**: raw target +0.17 and merit-only +0.40/+0.52 meet it. Dropping `ts` does not,
  because the rolling window already supplies recency.

## 3. Distillation experiments (`distill_results.md`)

All are applied to E_s (strict, pre-gate). The training-period structural forecasts come from this study: 5,705
hourly MTUs, 2026-01..08, variant (b). The training-run values below are in-sample for the posterior, and the training
records use published TSO inputs, not the deployed forecast.

| variant | MAE | vs E_s | vs T | M3 + it (4 equal) vs M3 |
|---|---|---|---|---|
| E_s | 2.87 | — | +1.05 | +0.12 (known) |
| K1 slope distillation p_lag + κ(E_s − p_lag), κ = tree/structural slope 0.76 | 2.91 | +0.04 [−0.10, +0.19] | +1.09 | +0.13 [+0.07, +0.18] |
| K1, κ = training OLS 0.60 | 3.09 | +0.23 [−0.00, +0.44] | +1.28 | +0.15 |
| K1, κ = training-MAE optimum 0.85 | 2.86 | −0.01 [−0.10, +0.08] | +1.04 | +0.12 |
| K2 re-clear with the German residual-load change × 0.76 | 2.99 | +0.13 [−0.01, +0.26] | +1.17 | +0.13 |
| R1 residual LightGBM trained before 09 | 2.99 | +0.12 [−0.17, +0.48] | +1.17 | +0.12 [+0.05, +0.20] |
| R1 applied to E (post-gate) | 2.47 | +0.22 vs E [−0.13, +0.65] | +0.65 | −0.01 |
| R2 residual LightGBM refit daily (training + earlier holdout days) | **2.55** | **−0.32 [−0.47, −0.17]** | +0.73 [+0.54, +0.92] | +0.04 [−0.01, +0.10] |

- **Slope distillation (K1, K2) does not help.** Expectation K1 is **not met**. K2 is within 0.1 of K1, as expected.
  - The trees' slope is only flatter at the curve ends, and a scalar κ also flattens the mid band, where the
    structural slope is right.
  - For day-over-day changes, the full steep response is better than the flat one (section 1).
  - A band-dependent fix is the curve-end re-specification, which already failed the null recovery screens.
- **A frozen residual learner (R1) fails.** Expectation R1 is **not met**: it adds a −1.4 ct/kWh bias.
  - Training-period structural errors are in-sample and come from a different fuel and availability regime.
  - The correction learned there extrapolates badly into an autumn with a fuel-cost index above the training range
    and +15 GW French shocks.
- **The refit residual learner (R2) is the only variant that helps.** It is the grey-box analogue of the trees'
  daily retraining.
  - It improves E_s by 0.32, mainly at midday (3.76 → 3.09) and at non-positive prices (7.41 → 5.04).
  - It is still 0.73 behind the trees and does not improve the mix (+0.04). Expectations R2 and R3 are **met**.
- **Using the tree forecast as a pseudo-observation for the shock (C1) was not run.** To first order it is a
  price-space blend of E_s and T. The time-of-day weights already showed such a blend gives E_s weight 0–0.1.

## 4. Stability around regime changes (`stability.md`)

Single-stage trees retrained daily over 2026-04..11, using the TSO residual-load forecast, against the stored market
route and Timestral (summer set every third day, autumn daily) and the structural (b) forecasts.

- **Advanced hybrid coupling (first MTU 2026-06-11).** Expectation B1 is **not met as stated**, but the jump is
  common to all flexible models.
  - Tree error relative to the price level rose 37% in the following 14 days (MAE 1.34 → 1.97, bias −0.41 → −0.87).
    It fell back to within 8% of the pre-break level in days 15–42.
  - The market route (+30%) and Timestral (+61%, only 5 days) jumped as much or more. Neither uses a
    coupling-dependent feature, so the jump is not specific to the trees' non-causality.
  - It may be the coupling change or simply a hard fortnight; these data cannot separate the two.
  - The structural (b) forecast did not jump (−0%). It is in-sample on these months, and the carried shock absorbs
    level shifts at once.
- **Fuel-cost index leaving its earlier range (daily mean above 127 from 2026-07-16).** Expectation B2 is
  **not met**:
  - no jump for any model (trees −2%, raw-target trees −10%, structural −12%);
  - the raw-target tree falls behind the blend only later, in September and October (by 0.19 and 0.16), as the
    index climbs to 170–190.
- **French nuclear unavailability.** Expectation B3 is **met**: daily MAE barely correlates with it for any model
  (trees 0.08 and 0.13, structural 0.04, market route −0.09). By unavailability tercile, no model's relative error
  rises.

**Reading.** The trees' non-causality appears as a short-lived bias after a break: they underpredict for about two
weeks, then retraining absorbs it. For a day-ahead forecaster this is second order. The cost is around +0.5 ct/kWh
for a fortnight, and the market-based Timestral and market route show the same. It matters for counterfactuals
(what if a link is added or a plant closes), not for next-day accuracy.

## 5. Answer

**(i) Do the flexible models capture structure? Yes, in reduced form, and that is enough to forecast better.**
- The trees reproduce the merit-order slope where the data are dense (40–200 EUR/MWh) and flatten it at the sparse
  curve ends, where the structural curves are too steep.
- What they learn is the conditional distribution of price given the information set. This includes:
  - persistence and level, through daily refitting on a window ending at the forecast (worth 0.51 in this window);
  - fuel-cost scaling (the ratio target, 0.17);
  - neighbour and availability conditions, implicitly through weather and Europe-wide features correlated with them
    (0.30).
- The structural model has to represent each of these as an explicit mechanism or state variable. Where it lacks
  one (French availability, curve-end flexibility, storage and demand response at negative prices), the error lands
  in the shock.
- The structural model then carries that shock forward from yesterday. That imports yesterday's transient: its
  errors have negative day-to-day autocorrelation and the largest shape errors at midday and at negative prices.
- So the trees are more accurate for two reasons. They fit the reduced form that forecasting needs. The structural
  model's errors come from misspecified curve ends, missing state variables, and a noisy one-day anchor, not from a
  lack of structure in the data.

**(ii) Why distillation is limited.**
- **Forecast accuracy is a property of the reduced form given the information set.** Structure is identified from
  exogenous variation and constraints. Distilling a reduced form into a structural model can only help where the
  structural model is misspecified in a way the reduced form reveals.
- **Here that is the curve ends.** The trees confirm it: 2–3× flatter at ≤0–40 and >200 EUR/MWh. A scalar slope
  change cannot fix it without breaking the mid band (K1/K2: +0.04 to +0.23). Curve-end re-specifications already
  failed identification: the lignite floor and dispersion ridge absorb the flexibility.
- **The trees' partial dependences are not causal derivatives.** Residual load is correlated with weather, season and
  neighbour conditions. A slope "learned" by the trees mixes the supply response with those correlations. This
  study's own estimator gives 2.82 on tree-fitted prices against 2.14 on observed prices.
- **Most of the trees' edge is recency and normalisation, not a better curve.** That part cannot be distilled into
  curve parameters. It can only be copied as a learned, frequently refit correction, and then the structural core
  contributes little.
- **A correction learned once on the structural model's training errors does not transfer** (R1: +0.12, bias −1.4).
  Those errors are in-sample and regime-specific.

**(iii) What the experiments show, and the recommended path.**
- **Best grey-box.** The structural core plus a LightGBM correction refit daily on its own recent errors (R2):
  2.55 against 2.87 for E_s. That is −0.32 [−0.47, −0.17], but still +0.73 behind the trees, and +0.04 [−0.01, +0.10]
  in the mix.
- **None of the distilled variants earns a place in the deployed mix.** The mix stays at 1.64.
- **Recommended path.**
  1. Do not invest further in making the eleven-zone model a forecast member.
  2. Use it for what only it can do: counterfactuals (outages, links, fuel shocks) and flows and net positions. Use
     the trees' implied slopes by band as an external validation target (mid band ≈ 2.2–2.9, ends much flatter).
  3. If a structural forecast member is still wanted, the levers are:
     - residual-load forecast quality (0.62 between E and E_s, the largest single gap);
     - an explicit French availability state;
     - a daily-refit learned correction (R2 form), instead of the raw carried shock.
     Expect at best about 2.2–2.5 strictly pre-gate, which is not competitive with 1.64.
- **Stability.** Daily refitting is what keeps the trees stable across regime changes. Breaks cost a transient bias
  of about two weeks, shared with the other flexible models.

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
| B1 | not met as stated (+37%, but common to all flexible models and transient) |
| B2 | not met |
| B3 | met |

## Caveats

- **Information sets.**
  - The single-stage ablation trees and the stability trees use archive weather, an advantage shared by all their
    variants. The stability trees also use the TSO residual-load forecast.
  - The structural training run uses published TSO inputs, and its errors are in-sample for the posterior. So the
    structural line in the stability table is optimistic before September.
- **Intervals.** The bootstrap is circular over all 61 days. `ensemble_bt.py` instead resamples the first 56 days,
  so its intervals differ slightly.
- **Runtime.** The stability run took about 24 minutes, above the 20-minute guideline, because other jobs were using
  the shared CPU. Its computation is about 10 minutes of fits.
- **Post hoc.** κ = 0.85 (training-MAE optimum) was chosen on training data, not on the holdout. Nothing was tuned on
  the 61 days.
