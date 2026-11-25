# Out-of-sample accuracy test: protocol (fixed 2026-11-24, before any holdout record is built)

Holdout window and training window: `prereg/HOLDOUT.md`. Nothing below has been run on the holdout.

## Targets and resolution

- **German day-ahead price**, hourly mean of the four MTUs (EUR/MWh).
- **German scheduled net import from the ten SDAC neighbours**, hourly mean (GW). This is `m` in the two-zone panel
  and −n_DE in the eleven-zone model, the same series.
- **Resolution.** Every holdout hour is scored at hourly resolution. The eleven-zone prediction for an hour is the
  mean of its four MTU predictions.

## Forecasts compared

1. **Eleven-zone model, refitted on the training window.** Two refits: the current curves, and the curve-end
   specification once it is frozen. Each is scored in two shock variants:
   - (a) *unconditional*: u is drawn from the fitted multivariate t. The point forecast is the median over
     2 posterior draws × 5 shock draws per MTU.
   - (b) *previous-day shocks*: u is the shock recovered from the same MTU on the previous day (an outcome known
     before the gate). The point forecast is the median over the same 2 posterior draws.
2. **Two-zone benchmark, refitted on the training window**, with the same two variants: (a) its fitted shock
   distribution; (b) the previous day's recovered shock, at hourly resolution.
3. **Naive baseline:** the same hour of the previous day, for both price and net import.

Variant (b) gives the models the same persistence the naive forecast uses. Without it, a loss to the naive forecast
would measure the absence of persistence in the shock model, not the supply curves.

## What a forecast may use (known before the D−1 noon gate)

- Published load, wind and solar forecasts for day D, i.e. the residual loads D_z.
- THE gas and EUA prices of D−1; availability (unavailability notices published before the gate).
- The Core flow-based domain for day D, from the JAO final computation, which is published before the gate. The
  publication time is verified on a sample of days before the run; if any part is published after the gate, it is
  replaced by the previous day's domain for that hour.
- From day D−1, whose auction ran on D−2: held exchanges with the rest of Europe and among the Nordic zones, the
  omitted Core hubs' net positions, link outage flags and the recovered shocks.
- Link limits: the largest flow in each direction over the seven days before D.

**Not available before the gate, so never read for day D:**
- the eleven prices and all net positions;
- the link schedules and flows;
- the shadow prices, binding set and inferred constraints;
- the calibration of violated rows;
- realised fuel outputs.

Holdout prediction records are built by a separate script that does not open those fields for day D.

## Scoring

- **Metrics.** MAE and RMSE of the hourly German price, and MAE of the hourly German net import. Reported for the
  whole window and by price band (≤ 0, 0–40, 40–120, 120–200, > 200, using the observed price).
- **Non-converged predictions are misses, not dropped.** The hour is scored with the training-window mean price and
  mean net import, and the share of such hours is reported.
- **Paired comparison.** Paired differences in absolute error between forecasts are summarised with a block
  bootstrap by day (1,000 resamples, 90% intervals).
- **Single scoring.** Each frozen specification is scored once. A specification changed after seeing holdout
  results is reported as such, with both scores.

## Expectations (before the run)

- O1: on price MAE, every structural forecast in variant (b) beats its own variant (a), since shocks are persistent.
- O2: on price MAE in variant (b), the eleven-zone model is no worse than the two-zone benchmark, within the
  bootstrap interval.
- O3: on net import MAE in variant (b), the eleven-zone model beats the two-zone benchmark. It uses the flow-based
  domain and ten neighbour curves rather than a single exchange function.
- O4: the naive forecast is hard to beat on price in this window. Whether either model beats it is not predicted.
- O5: the window is entirely under advanced hybrid coupling, where 95–97% of MTUs needed inferred constraints in
  training. The eleven-zone model cannot use inferred constraints out of sample, so its net-import accuracy may
  suffer most here.

## Addendum 1 (2026-11-24, before any holdout record; after a dry run on two training days)

- **Infeasible domain.** If the published domain for an MTU is infeasible for the solver, the forecast retries once
  with every RAM raised by 10% of |RAM| + 50 MW, and the share of such solves is reported. If that also fails, the
  solve is a miss.
- **Misses at draw level.** Every missed solve is replaced by the training-window mean (price, net import) before the
  median over draws is taken. A median over converged draws only would drop the steep cases.
- **Wind and solar forecasts.** ENTSO-E publishes the day-ahead wind and solar forecast by 18:00 on D−1, after the
  noon gate (Transparency Regulation, Art. 14.1(d)). The load forecast is published before the gate. All three
  structural forecasts use the published wind/solar forecast as a stand-in for participants' own pre-gate forecasts.
  This is a known deviation from a strict pre-gate information set, and it applies equally to both models. The naive
  forecast does not use it. The JAO final-domain publication time is taken as before the gate; it has not been
  verified independently here.
- **Two-zone inputs.** German residual demand is rl_h − (other + psgen + pump − ch) at h − 24 h, with the realised
  components lagged, as for the eleven-zone model.
- **Starting values.** The training-window refits start their chains at the medians of the earlier eleven-zone and
  two-zone posteriors, which were fitted on data including the holdout. Only the starting point is shared; the
  posteriors are from training data. This is disclosed rather than avoided.
- **Dry run** (two training days, earlier posteriors, plumbing only): forecasters and scorer ran end to end. A swap
  test on those days attributed most of the price error to the shock carry-over:
  - own shocks and exchanges on the pre-gate domain: price MAE 4.3, import MAE 0.43 GW;
  - with lagged exchanges: 5.2 and 0.88;
  - with lagged shocks too: 15.3 and 1.73.
- **Resolution.** The quarter-hour −n_DE differs from the hourly `m` by up to 3 GW; at hourly means the two agree
  within 0.05 GW. Scoring is hourly.

## Addendum 2 (2026-11-25, before any holdout forecast): frozen specification and run settings

- **Eleven-zone model.** Current curves. Both curve-end variants failed their null screens and are dropped
  (EXPECTATIONS_11ZONE.md). Posterior: the training-window refit `post_train_old.npz` (sha256 prefix 6e18322b59ed56d5), 800 draws.
  The forecast uses 2 posterior draws (indices 0 and 799, one per chain), 5 shock draws per draw in variant (a), and
  seed 1.
- **Two-zone benchmark.** Basin mode 0, which led by about 167 nats on the full sample (BENCHMARK_RESULTS.md). Fitted
  on training hours only: 2 chains started at the mode-0 medians, 300 warmup and 300 draws, fixed seed 10, saved every
  100 draws. Convergence rule: max r-hat ≤ 1.05 over the German fuel block and the shock parameters. If it is not
  met, the fit is reported as not converged and still used, with the r-hat stated beside every two-zone score. The
  forecast uses 2 draws (first and last) and 5 shock draws in variant (a).
- **Link limits** (changed from the dry-run builder). Training data show that outage notices mostly coincide with
  flow. On DE–DK1 and DE–DK2, 99% of covered MTUs have flow; on NordLink, 91%. A limit is therefore zero only on the
  three links with genuine full outages (Baltic Cable, NorNed, SwePol), and only when a notice covers the MTU and the
  D−1 schedule was zero. There the rule is right about 85% of the time. Otherwise the limit is the trailing 7-day
  maximum. Notices are not filtered by publication time; that is a disclosed deviation.
- **Realised generation timing.**
  - Variant (b): realised German other / storage output enters through t − 24 h on both sides and cancels. The load
    reduces to yesterday's supply minus net position plus the change in published forecasts.
  - Variant (a): uses the realised output of t − 48 h, which is known at the gate, in both models.
- **Targets.** 1,488 holdout hours. One hour (2026-11-08 23:00 UTC) has no observed target and is dropped and
  reported.
- **Holdout prediction records:** `pred_holdout.pkl` (sha256 prefix a30ebc4650ffcd2d), 5,952 MTUs, none skipped.
