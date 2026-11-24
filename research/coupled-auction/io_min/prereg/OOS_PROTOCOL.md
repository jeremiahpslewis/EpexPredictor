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
