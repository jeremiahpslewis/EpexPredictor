# Faithful rerun of the deployed DE ensemble: expectations (written 2026-11-26, before the full run)

**Status.** Post hoc. The window 2026-09-08..11-08 has already been used for the ensemble backtest
(`../RESULTS_ENSEMBLE.md`) and many research runs, so this rerun describes those days; it is not a sealed test.

## Run

- Command: `faithful_bt.py --start 2026-09-08 --end 2026-11-08 --vintage strict` (latency 6 h), run in two chunks
  of 31 days.
- Issue time: the noon gate on D−1.
  - Prices run to D−1 23:45. Gas for D−1 counts as known, which is slightly optimistic.
  - Weather comes from the newest run published 6 h or more before the gate: d1 for about the first 6 h of D, d2 for
    the rest of D, d3 for D+1.
- d1 for every zone was fetched on 2026-11-26. Checked against d2 at the same points: correlation 0.82–0.95 for wind
  and 0.96–0.98 for temperature. Irradiance is about 20 W/m² lower in d1 in the northern zones, which removes the
  d2 irradiance bias the smoke test found.
- The two smoke-test days are deleted before the run and redone in it.
- Idle-machine timing: 89–113 s per day. Rerunning 2026-09-10 on its own, rather than after 2026-09-09, gives
  identical output to floating-point precision, so no state carries over between days.

## Scoring

- The same 61 local days as `ensemble_bt.py` (2026-09-08..11-07), 96 quarter-hours each, MAE in ct/kWh.
- 2026-11-08 is run but not in the headline, because it has a missing target hour.
- Paired 7-day block bootstrap, 90% intervals, against the stored-array members.

## Expectations

Reference values from the stored arrays (cutoff at local midnight of D; weather opt = d1 then d2; X1 reproduced the
issued blend at 1.740 earlier): trees 1.818 (opt weather), Timestral 1.747, market route 1.891, M3 1.642.

- **F1.** The issued blend scores 1.60–1.85. This is close to M3 or slightly worse, because strict weather is
  older for D afternoon than the opt weather in the stored arrays.
- **F2.** Trees score 1.80–2.10, between the stored opt and pes tree scores.
- **F3.** Timestral scores 1.70–1.95, and the market route 1.80–2.10.
- **F4.** The blend beats each of its three members on the 61-day mean.
- **F5.** Residual-load MAE of the tree series over D is 2.2–2.9 GW. This range runs from the residual-load study's
  optimistic weather-age bracket (interp, day 1 2.18) to its pessimistic one (gate12, 2.69), widened for the
  harness's mixed vintages.
- **F6. Per-day check against the stored pessimistic residual-load forecast** (`gen_lwA_autumn.npz` rl_pes). The
  harness tree residual load should exceed it by more than 1 GW on at most 10 of the 61 days. On 2026-09-09 the
  timing run gave 7.32 GW against 4.16 for rl_pes; if that is a pattern rather than one day, it is a defect to
  explain before any number from this run is used.
- **F7.** Every day passes the leak assertions. Failed days are logged and listed; if more than 3 fail, the run is
  reported as incomplete.

---

# Addendum 2026-11-26: residual-load variants (`--rl-variant`), smoke test, written before running

**What is tested.** The two recommendations of `../rl/RESULTS.md`, wired into the harness without touching `predictor/`:
- `loadcorr`: tso_load forecast + mean(published − held-out model forecast) over the last 24 quarter hours of D-1, times
  exp(−(j+1)/96) for the j-th slot after the cutoff. The residual comes from a load booster trained on rows up to
  noon D-2 (the study's `d0 − 144`), so D-1 is out of sample, as in the study.
- `matched`: slots whose grid weather is d2/d3 use wind/solar/load boosters trained on the `_previous_day2` grid
  history (`SP/dense/hist.parquet`, from 2025-10-01); d1 slots keep the app's boosters.
- `both`: each model set with its own D-1 load residual (as `../rl/combo.py` "VM+decay(load)").

**Smoke days:** 2026-09-09 and 2026-10-14, picked because base tree RL was bad there (7.32 and 4.77 GW). Biased choice
on purpose: they show the mechanism, not the average effect. Base figures (out_strict): 09-09 tree RL bias +7.5 GW on d1
slots and −6.8 on d2 slots; 10-14 −5.5 / −4.5.

**Expectations (no tuning on these days; numbers only describe two days):**
- S1. `base` rerun into a scratch dir reproduces `out_strict` for these days (all arrays to ~1e-6; prices to 1e-3 ct/kWh).
- S2. `matched`: d1 slots bit-identical to base in RL. d2 slots: RL bias moves upward (less negative) on both days,
  because the study found the near-current-weather model over-forecasts wind on 2-day-old weather by ~0.8 GW. Expected
  d2-slot RL MAE change: −0.3 to −2 GW on 10-14; on 09-09 direction unclear (the d1/d2 sign flip suggests something
  other than the wind bias, e.g. a TSO-figure jump).
- S3. `loadcorr`: wind and solar identical to base; the load change is a constant times the decay. |correction| at the
  first slot of D ≤ 2 GW. RL MAE change on D within ±0.5 GW on each day.
- S4. `both`: d2-slot RL MAE below base on 10-14; tree price MAE change within ±0.5 ct/kWh; blend change within
  ±0.2 ct/kWh (Timestral's lead-weighted mix dilutes tree RL changes, study: −0.06 GW).
- S5. The variant RL is what every consumer reads: `residual_load` in DE base and stage-2 `prepare_dataframe` equals
  the saved variant series; Timestral/market inputs equal w·Timestral + (1−w)·variant; `rl_timestral` equals base.
- S6. Every leak assertion passes; each run < 20 min wall clock.
