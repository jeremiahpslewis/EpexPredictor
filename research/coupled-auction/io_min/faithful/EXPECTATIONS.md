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

---

# Addendum 2 (2026-11-26): revised definitions, re-smoke of 09-09 and 10-14, written before running

**Changes after review.**
- **Correction window.** The residual window is now the 24 quarter hours of valid time ending at gate − 2 h. That is
  D-1 04:00–10:00 local. The decay runs from the window's end: exp(−lead/24 h), lead = window end → end of slot, about 0.55 at D 00:00 local.
  2 h is an assumed, conservative latency: the TSO day-ahead figures for D-1 were published on D-2 (README), so the
  old window (D-1 18:00–24:00) was not a publication leak either. Its residual is still stored as a diagnostic
  (`studywin_*`, decay from the cutoff).
- **Window weather.** The window's forecast is made on the 1-day-old grid weather (the study's `prev_d1`), issued
  ≤ gate − 6 h. Asserted.
- **Primary loadcorr form (the study's).** The held-out load booster (app weather, rows up to noon D-2) forecasts D's
  load, plus its own window residual × decay. Wind and solar stay with the app's boosters, trained to the cutoff.
- **Primary `both`.** d1 slots as loadcorr. d2+ slots: matched wind/solar plus the held-out matched load booster
  (d2 weather, rows to noon D-2) with its own residual. Stored diagnostic only: the held-out residual added to the app
  model's load (`rlc_base_tso_load + rlc_corr_app`).
- **Control set**, in matched and `both`, stored only: three boosters on app weather, trained on exactly the matched
  set's rows (from 2025-10-01).

**Expectations (the same two hand-picked days; post-hoc on used days, describing the mechanism only).**
- R1. `matched` reproduces the earlier `smoke_matched` price arrays and `rl_trees` to float noise (the control is not
  fed).
- R2. **Control vs matched on d2+ slots.**
  - The control's RL bias lies between the app model's and the matched model's.
  - On both days, at least half of the bias reduction from app → matched remains between control and matched. This
    would mean weather age, not history length, is the main driver.
  - Solar: the control removes most of the solar bias on 09-09 (+2.64 app). That would make solar a history-length
    effect. This is a hunch, stated so it can fail.
- R3. **Window residual of the held-out app booster.**
  - Between −1.5 and +1.5 GW on each day.
  - Sign the same as the study-window residual (+1.01 / +0.94) on at least one day.
- R4. **loadcorr (study form) vs base on D:** RL MAE change within ±0.6 GW, and tree price change within ±0.3
  ct/kWh, on each day. The correction is weaker than the old one, since the decay starts at about 0.55.
- R5. **`both`:** d2+ RL MAE below base on both days. Blend MAE below base on both days. Gains are smaller than the old
  `both` on d1 slots.
- R6. Every leak and wiring assertion passes, including the study-form identity, and each run takes < 20 min.

---

# Addendum 3 (2026-11-26): 61-day used-window runs, `matched` and `both`, written before launch (not yet approved)

**Amendment (2026-11-26, after review, before anything ran).**
- **`both` is deferred.** Only `matched` is requested for the 61 days. M4 is void; M5 is replaced by L1–L5 below.
- **Load forms are scored at RL level only.** The `matched` run stores every load form as RL-level arrays: held-out
  boosters of both weather sets, both residual windows, and the diagnostics. `score_variants.py` scores them at RL
  level on the 61 days.
- **A price pass for a load form is requested only if a form wins at RL level** under L5's rule. Nothing is chosen
  from the two smoke days.
- **The forms.** Each one keeps the matched run's wind and solar (app boosters on d1 slots, matched on d2+). Load is
  the set's own on each slot (app set on d1, matched set on d2+):
  - **F0**, as fed in `matched`: the cutoff-trained load, uncorrected.
  - **F1**, primary study form: held-out load (to noon D-2) + its own residual over the gate window (D-1 04:00–10:00
    local), decay from 10:00.
  - **F2**: held-out load + its own residual over the study window (D-1 18:00–24:00), decay from the cutoff.
  - **F3**: the cutoff-trained load + the held-out residual over the gate window (the old diagnostic).
  - **F4**: the cutoff-trained load + the held-out residual over the study window (the first-smoke `loadcorr`).
  - **REF**: D's published TSO load as at the gate (load error 0). Energy-charts does not revise these figures, so
    this equals the stored figure. It is an upper bound, not a candidate.
- **Expectations** (RL MAE over all slots of D, form − F0; paired 7-day block bootstrap, 90 %):
  - **L1.** F1 within ±0.10 GW of F0. The held-out booster is weaker on load; its own residual offsets that only in
    part.
  - **L2.** F3 and F4 below F0 by 0.03–0.20 GW. At least one of them has an interval excluding 0. The study: load
    decay −0.11 to −0.15 GW on day 1.
  - **L3.** F2 between F1 and F4 on the 61-day mean.
  - **L4.** REF below F0 by 0.3–0.9 GW. Of the RL error at gate weather, load contributes about 13–18 % (study E1).
    By MAE the drop may be larger, because the load bias is systematic.
  - **L5, decision rule** (fixed now). A price pass is requested for at most one of F1–F4: the one with the largest
    RL improvement over F0, provided it is ≥ 0.10 GW and its 90 % interval excludes 0. The rule applies to all
    slots of D. If no form meets it, there is no price pass and loadcorr is reported as not shown at RL level.

**Run.**
- Commands: `faithful_bt.py --start 2026-09-08 --end 2026-11-08 --rl-variant matched`, then the same with `both`.
  Strict vintages, sequential, one process at a time.
- Outputs: `SP/faithful/out_strict_rlmatched` and `out_strict_rlboth`.
- Scoring: `score_variants.py <dir> SP/faithful/out_strict <explicit .md>` on the same 61 local days as RESULTS.md
  (2026-09-08..11-07).
- **Status.** Post hoc: the window has been used many times, including the residual-load study, and 2 of its days
  were the smoke days. This describes the mechanism; it is not a sealed test.

**Expectations.** Base references: trees 1.787, Timestral 1.732, market 1.872, blend 1.648; tree RL bias on d2
slots −1.32 GW, MAE 2.97.

- **M1. Residual load on d2+ slots (`matched`).**
  - The bias rises from −1.32 to between −0.6 and +0.3 GW.
  - MAE falls by 0.2–0.6 GW (study: VM day 1 −0.05 to −0.22, h24-38 −0.28 to −0.36), with the interval excluding 0.
  - d1 slots are unchanged, to float noise.
- **M2. Control vs matched.**
  - The control's d2+ bias is within ±0.3 GW of the app model's.
  - Matched beats the control on d2+ MAE, with the interval excluding 0.
- **M3. Prices under `matched`.**
  - Trees: −0.02 to −0.10, the interval may include 0.
  - Blend: within −0.06 to +0.02.
  - Timestral and the market route: within ±0.05 each. The lead-weighted mix dilutes tree RL changes (study:
    −0.06 GW on the mixed series).
- **M4 (VOID: `both` deferred, see the amendment). `both` vs `matched`.**
  - Day-level RL difference within ±0.15 GW, with the interval including 0. The held-out boosters are weaker on
    load, which offsets the correction (smoke: load MAE 1.52 uncorrected, 1.27 corrected, against 1.02 for the app's
    booster).
  - Blend difference within ±0.03.
- **M5 (REPLACED by L1–L5). Primary loadcorr form against the stored alternatives** (scorer "load forms", app-weather side).
  - The study-window residual and the old diagnostic (held-out residual added to the app's load) both have lower
    load MAE than the primary form.
  - The primary form's load MAE is within ±0.15 GW of the uncorrected app load.
- **M6.** Every day passes the leak and wiring assertions. Failed days are listed; more than 3 makes the run
  incomplete.

---

# Addendum 4 (2026-11-26): fresh days 2026-11-09..11-26 (18 days), written before any fresh variant run

**Status.** Sealed so far:
- No accuracy figure for these days has been looked at.
- A stage-only check confirmed coverage.
- The base output of 11-26 exists, from a feasibility run whose accuracy was not read.

**Run.**
- `FAITHFUL_FRESH=1 faithful_bt.py --start 2026-11-09 --end 2026-11-26` for `base` (to `SP/faithful/out_fresh_strict`),
  then `--rl-variant matched` and `--rl-variant both`. `loadcorr` is optional.
- Scoring: `score_variants.py` against `out_fresh_strict`, with an explicit output path.

**Caveats.**
- With 18 days and 7-day blocks, the bootstrap resamples only about 3 blocks. Intervals will be wide and unstable, so
  a non-zero interval on one metric is weak evidence.
- Treat the 61-day post-hoc result and this result as two looks, not one test.
- November weather: more wind, little solar. Solar-bias effects will be small.

**Expectations, given before any fresh figure.**
- **N1.** `matched` lowers tree RL MAE on d2+ slots compared with base, with a point estimate below 0, on at least
  11 of the 18 days.
- **N2.** The matched d2+ RL bias is closer to zero than base's on the 18-day mean.
- **N3.** Blend MAE change: `matched` and `both` each within −0.10 to +0.03 ct/kWh. The sign may go either way; the
  interval is expected to include 0.
- **N4.** `both` vs `matched` on blend: within ±0.04.
- **N5.** No leak or wiring failure. A failed day is listed and not replaced.

**Amendment to addendum 4 (2026-11-26, after the 61-day matched result, before any fresh figure is seen).**
- `both` is dropped (deferred on the used window; L5 found no load form worth a price pass). N4 is void; N3 applies to
  `matched` only.
- Runs: `base` for 11-09..11-25 (11-26 exists, unread) to `--out SP/faithful/out_fresh_strict`, and `matched` for
  11-09..11-26 to `--out SP/faithful/out_fresh_strict_rlmatched`, several processes in parallel with `--torch-threads 1`.
  `--out` is given explicitly because the default path ignores `FAITHFUL_FRESH`.
- Added expectations, using the 61-day result as the prior:
  - **N6.** Matched d2+ tree RL bias within ±0.6 GW on the 18-day mean; base's is negative.
  - **N7.** Matched minus control d2+ RL MAE has a point estimate below 0.
  - **N8.** Price changes (matched − base), point estimates: trees −0.20..+0.05, blend −0.12..+0.03, Timestral and
    market route each within ±0.08. Intervals may include 0 (about 3 blocks).
  - **N9.** REF (published load) below F0 on RL MAE; no F1–F4 form below F0 by more than 0.10 GW.

---

# Addendum 5 (2026-11-26): published-load variants `published` and `matchedpub`, written before any run of them

**What changes.**
- **`published`:** the app's boosters for wind and solar. On every slot of the local delivery day D, `tso_load` is
  replaced by D's day-ahead TSO load as published by energy-charts at the gate.
- **`matchedpub`:** the `matched` wind and solar (app boosters on d1 slots, d2-trained boosters on d2+ slots), with the
  published load on D.
- **After D** (D+1 and D+2, inside Timestral's horizon), both variants keep the model's load: the app booster under
  `published` and the matched set under `matchedpub`.
- The variant RL goes to every consumer: the DE base and stage-2 features, and the Timestral-mixed series that
  Timestral and the market route read.
- **Assumption.** The published value is taken from the stored generation figures (`live_data`, or the fresh refetch),
  because historical publication times cannot be checked. Two pieces of evidence support this:
  - D's load was out before the gate in a probe on 11-26;
  - the stored figures are never revised (4,892 slots).

  Both probes were on a CET day, at 10:22 and 14:49 UTC. The CET gate is 11:00 UTC; in CEST it is 10:00 UTC, so the
  evidence does not directly cover the CEST days, which are most of the 61. Under ENTSO-E's transparency rules the
  day-ahead load is due 2 h before the day-ahead gate; energy-charts' own delay is not known. A module flag in
  `faithful_bt.py` disables both variants if tomorrow's timing recheck fails.
- **Status of the windows.**
  - Both windows have been looked at once at RL level: the REF row in RESULTS_MATCHED.md and RESULTS_FRESH.md. The
    RL figures below are therefore identities computed from stored arrays, not predictions.
  - The fresh window stays sealed for the price effects of these two variants. The one fresh smoke day is reported as
    pass/fail only.

**Smoke test (2026-09-09 and 2026-10-14, strict; post hoc, 2 hand-picked days).** The exact RL comes from
`smoke3_matched`'s stored arrays: `pub_tso_load_gate`, `rlc_base_*` and `rlc_sel_*`.
- **P1. Tree RL MAE on D (GW), all slots and then d1/d2+.** Each must match to 0.01 GW; anything else is a wiring
  error.

  | D | base | `published` | `matched` | `matchedpub` |
  |---|---|---|---|---|
  | 09-09 | 7.32 (7.54/7.24) | 6.99 (8.22/6.55) | 4.73 | 4.66 (8.22/3.41) |
  | 10-14 | 4.77 (5.49/4.52) | 3.79 (3.57/3.86) | 3.35 | 2.83 (3.57/2.56) |
- **P2. Wiring identities, asserted on every day.**
  - `rl_trees` and the DE base and stage-2 features equal the variant series.
  - On D, the selected load equals the published load. A published value exists for each of D's 92, 96 or 100 slots,
    and none has a valid time after D.
  - After D, the selected load is the model's, and so is `get_data`'s RL.
  - `rl_mixed` = w·`rl_timestral` + (1−w)·variant. The spy on Timestral's inputs sees `rl_mixed`, including the D+1
    slots.
  - Wind and solar: under `published`, identical to the app boosters; under `matchedpub`, identical to `matched`'s
    selection.
- **P3. Cross-run checks, offline, to 1e-6.**
  - `rl_timestral` equals `out_strict`'s on both days.
  - `published`'s wind and solar equal `smoke3_matched`'s `rlc_base_*`; `matchedpub`'s equal its `rlc_sel_*`.
- **P4. Prices.**
  - 10-14: the tree price under `published` is below base, and under `matchedpub` below `matched`. RL falls by 0.98
    and 0.52 GW.
  - 09-09: the direction is unclear, with the tree change within ±0.3 ct/kWh. RL rises on d1 slots and falls on d2+.
  - Blend changes are within ±0.25 ct/kWh on each day.
- **P5.** Each 2-day run takes under 20 min with `--torch-threads 1`. Runtime per day is at most `matched`'s, because
  these variants train no held-out or control boosters. The fresh day (`matchedpub`, `FAITHFUL_FRESH=1`, explicit
  `--out`) passes every assertion.

**Later runs (expectations; a launch is not requested here).**
- **Q1. RL, identities.** Means over the days, with the paired 7-day block bootstrap at 90 %:
  - **61-day `published` − base:** tree RL MAE −0.333 [−0.42, −0.23] GW. That is −0.22 on d1 slots and −0.37 on d2+.
    It is better on 50 of 61 days.
  - **61-day `matchedpub` − `matched`:** −0.262 [−0.39, −0.15], which is the REF row.
  - **Fresh `matchedpub` − `matched`:** −0.435 [−0.59, −0.28], the REF row.
  - **Fresh `published` − base:** not computed, to avoid a new look. The expected range is −0.3 to −0.7 GW, larger
    than on the used window because the fresh load bias is about −0.8 GW against −0.1.
  - A deviation of more than 0.005 GW from any of the computed identities is a bug.
- **Q2. Timestral-mixed RL.** With `matched` the mixed series moved about 45 % as much as the tree RL; the published
  load acts like that, with a little extra on D+1 from Timestral's covariate. Expected ranges:

  | comparison | 61 days | fresh |
  |---|---|---|
  | `published` − base | −0.08 to −0.20 GW | −0.10 to −0.35 |
  | `matchedpub` − `matched` | −0.05 to −0.15 | −0.10 to −0.30 |
- **Q3. Prices, 61 days.** The prior is `matched`: −0.27 GW of tree RL gave trees −0.093, blend −0.061, Timestral
  −0.025 and the market route −0.048. The load error is smoother and more bias-like than the wind error, so the price
  effect per GW may be smaller.

  | comparison | trees | blend | Timestral, market route |
  |---|---|---|---|
  | `published` − base | −0.02 to −0.12 | −0.01 to −0.07 | each within −0.06 to +0.02 |
  | `matchedpub` − `matched` | −0.01 to −0.10 | 0.00 to −0.06 | |

  The intervals may include 0.
- **Q4. Prices, fresh (sealed until the fresh variant runs; about 3 blocks).**
  - Base and `matched` carry a load bias of about −0.8 GW on fresh days, so gains are expected to be larger than on
    the 61 days.
  - `matchedpub` − `matched`: trees −0.03 to −0.20, blend −0.01 to −0.10.
  - `published` − base: blend −0.02 to −0.12.
  - The intervals are expected to include 0 on at least one of the two.
- **Q5. Every day passes all leak and wiring assertions.** A failed day is listed and not replaced; more than 3 make a
  run incomplete. If the timing recheck finds D's load published after the gate on CEST-like timing, the price
  results of both variants are void (the flag is turned off), and only the RL REF stays as an upper bound.

*Note (2026-11-26, after the smoke run):* the ENTSO-E rule cited above (day-ahead load due 2 h before the day-ahead
gate) is recalled, not verified in this session.
