# Faithful as-of-gate backtest of the deployed DE ensemble

`faithful_bt.py` replays the **deployed** EpexPredictor stack for DE_LU, one target delivery day D at a time, using the
app code in `predictor/` unchanged. Network access and persistence are switched off by monkeypatching. The stack is:

- the 8 stage-1 base models: DE, AT, BE, FR, NL, SE4, DK1 and DK2, as `BaseModelCoordinator` builds them;
- the DE stage-2 model, as `RegionPriceManager` builds it;
- the residual-load trees;
- Timestral-123M on the price, at the app's pinned revision;
- the slim market route;
- `TimestralBlend.blend(base, [prices, eval], {"market": market_route})`, called as `RegionPriceManager.update_data_if_needed` calls it.

`fetch_d1.py` fetches the Open-Meteo previous-runs `_previous_day1` weather that is not yet local (see below).

## Information set per target day D

| item | value at issue |
|---|---|
| gate (time of issue) | D-1 12:00 Europe/Berlin (10:00 UTC in CEST, 11:00 UTC in CET), computed with zoneinfo |
| cutoff | D-1 23:45 local, the last known quarter hour. D-1 prices come from the D-2 auction, so they are known at the gate. |
| prices, all 8 zones | rows after the cutoff are **physically removed** from the working copy and also hidden with `set_horizon_cutoff`. Asserted: `get_last_known() == cutoff`, and `price` is all NaN on D in every model's `prepare_dataframe`. |
| TSO generation forecasts (wind, solar, load) | cut after the cutoff. Asserted. |
| gas (THE) | cut after the cutoff. Gas day D is indexed at D 00:00 UTC, which is after the cutoff, so gas day **D-1 is known**. This is the app's behaviour; see deviations. |
| weather, 8 zones (`weather_v2`, 15 min) and DE grid (`gridweather_v1`, hourly) | archive up to just before the gate. From the gate to the end of D+2, rows are **deleted** and replaced with previous-runs vintages. Nothing after D+2 is kept, so the archive cannot leak through a hole: a hole becomes NaN. Asserted: the wind columns the model reads on D equal the vintage arrays. |
| training window | stage-1, stage-2 and residual-load trees are retrained on [cutoff − 540 d, cutoff] (`TRAINING_DAYS`) |

### Weather vintages: read this first

Open-Meteo documents `<var>_previous_dayK` as **the value predicted 24·K hours before valid time**. It is a fixed lead
time, not "the run of the previous calendar day". This was checked locally: the mean |d1 − d0| by UTC hour is flat,
with no reset at 00 UTC. So d1 for the whole of D is **not** honest gate weather. For most of D it comes from runs made
up to about 12 h after the gate.

There are two policies (`--vintage`):

- **`strict`** (default): for each timestamp t, use the smallest K ≥ 1 with `t − 24h·K ≤ gate − latency`. `--latency`
  defaults to 6 h and stands for model run plus publication delay; a non-default latency gets its own output
  directory, `out_<vintage>_lat<h>`. In CEST this gives:
  - d1 for D-1 afternoon and D up to about 06:00 local;
  - d2 for the rest of D and the start of D+1;
  - d3 for the rest of D+1;
  - d4 for D+2 after about 06:00 local. d4 is not held locally, so d3 is used there and flagged as `wanted_k_gt_max`.
    It only touches the faded tail of Timestral's residual-load horizon.
- **`d1`** (optimistic, the calendar mapping): local day D-1 and D use d1, D+1 uses d2, D+2 uses d3. Some of this
  weather is newer than the gate.

The true gate forecast lies between the two. Lead times on D are about 18–42 h, against 24–48 h for `strict` and
about 12–36 h for `d1`. Run both and report the spread.

Hourly vintages are interpolated to 15 minutes in time. Radiation is shifted by −22.5 min (an hourly mean of the
preceding hour, against a mean of the preceding quarter hour), which was checked against the archive (d0 versus
`weather_v2`).

The grid vintages map `w120`, `sw` and `t2` to the store's `wind`, `radiation` and `temp`. The order of the points is
asserted to match `PriceRegion.generation_*`.

Where a vintage is missing for a column, the next older one is used (d1 → d2 → d3). Each day's file records the counts
in `meta["weather"][zone]`: `fallback_cells`, `missing_cells` and `k_hist`.

### Staging sweep (all 62 days, both policies, no training)

No weather cell is missing, every day has 96 or 100 slots, and no row is kept after D+2. The vintage minus the
archive on D:

| variable | bias | MAE |
|---|---|---|
| wind | −1.6 to +0.6 km/h | 3–4.6 |
| temperature | about 0 | 0.6–1.1 °C |
| pressure | about 0.1 | 0.7 hPa |
| irradiance | **+15 to +18 W/m² in BE, NL, SE4, DK1, DK2 and at DE's North Sea points**; small onshore | |

The irradiance bias is present in the raw hourly vintages, so it is not caused by the interpolation. It is a property
of forecasts versus the archive, which a live run faces as well: the trees are trained on archive-like history. The
sweep script is `SP/faithful/stage_sweep.py` and its output `SP/faithful/stage_sweep.log`.

### Local vintage sources (scratchpad SP)

| data | source |
|---|---|
| DE points 0–5 (onshore), d0..d3 | `prevruns_DE.parquet` |
| all zones' weather points, d2 and d3 | `prev_all/prev_<zone>.parquet` |
| 58-point DE grid, d1..d3 | `dense/prev.parquet` |
| **d1 for the 27 points of AT, BE, FR, NL, SE4, DK1 and DK2, and DE's 5 North Sea points** | `prev_d1/prev_d1_<zone>.parquet`, written by `fetch_d1.py`. **Not fetched yet:** Open-Meteo's daily limit was already used up (HTTP 429). Until it is fetched, these columns fall back to d2. Under `strict` that touches only D-1 afternoon and the first ~6 h of D; under `d1` it touches all of D. |

## Outputs

One file per day: `SP/faithful/out_<vintage>/<D>.npz`. A day whose file exists is skipped, so the run is resumable.
The arrays cover the local day D, so they have 96 slots, or 100 on 2026-10-25:

| array | content |
|---|---|
| `idx` | UTC ns |
| `trees` | stage-2 trees, `fill_known=False` |
| `timestral` | Timestral on the price |
| `market` | the market route |
| `core` | the market curves at a constant shift (logged only in the app) |
| `issued` | the blend the app would issue: `newprices` after `blend` |
| `issued_eval` | |
| `actual` | ct/kWh |
| `rl_trees` | residual load as the price trees read it |
| `rl_mixed` | Timestral-mixed residual load, as Timestral and the market route read it |
| `rl_timestral` | |
| `rl_true` | published TSO figure |
| `weather_k_DE` | vintage per slot |
| `meta` | JSON: gate, cutoff, weather stats, training rows and range, and runtime by phase |

Checks run every day:

- no price, gas or generation data after the cutoff;
- the model reads the overwritten weather;
- the tree forecast has no gaps on D;
- `issued == (trees + timestral + market) / 3` on D, which is inside the full-weight window of 152 quarter hours;
- 96, 100 or 92 slots.

## How to run

```
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
cd research/coupled-auction/io_min/faithful
$SP/venv/bin/python fetch_d1.py                     # once Open-Meteo's daily quota allows (about 8 small requests)
$SP/tsvenv/bin/python faithful_bt.py --start 2026-09-08 --end 2026-11-08 --vintage strict
$SP/tsvenv/bin/python faithful_bt.py --start 2026-09-08 --end 2026-11-08 --vintage d1     # optional bracket
```

The first run copies the store files from `SP/live_data`, which stays read-only, to `SP/faithful/stores`. Nothing is
ever written back to the stores.

Run `fetch_d1.py` **before** the full run. Otherwise days computed with the d2 fallback for those 32 points are
cached, and their `.npz` files must be deleted to recompute.

## Deviations from a true live run

1. **Weather vintages approximate the gate run.** They are fixed-lead previous-runs values at 24·K h, not the run that
   was actually in the store at the gate. The app also refreshes weather only every 3 h, which is not modelled. The
   Single Runs API (`&run=`) could give the exact gate run, but would cost far more quota (future work).
2. **Hourly vintages interpolated to 15 min** for `weather_v2`, which the app fetches natively at 15 min. The 15-min
   detail of the archive is lost from the gate onward, including radiation shape.
3. **History before the gate is the archive** (historical-forecast API), as in a live store that refreshes past days.
   That includes the training rows and D-1 before 12:00.
4. **Gas for gas day D-1 counts as known.** This is app behaviour. At 12:00 on D-1 that gas day is still trading, so
   this is a slight optimism. The research harnesses froze gas at issue.
5. **Store coverage is that of `live_data`.** Grid weather starts on 2025-04-06 and TSO generation on 2025-05-24.
   Training rows need the residual-load feature, which needs grid weather, so for early target days **the price trees
   too** start at 2025-04-06 instead of cutoff − 540 d. For D = 2026-09-08 that is 49,912 stage-1 rows; D = 2026-10-26
   has the full 51,840. A live app would have fetched further back where the source has it.
6. **The cross-predictor set is the 8 energy-charts zones without ENTSO-E load forecasts.** There is no ENTSO-E key
   and the store set holds no `entsoe_v1` data. This matches the deployed store set; it may differ from a deployment
   with a key.
7. **Training window** [cutoff − 540 d, cutoff] instead of [now − 540 d, now + 7 d]. Rows after the cutoff have no
   price or published figure, so they are dropped either way, and the start differs by about 12 h.
8. **Prediction starts 14 days before the cutoff, not at `now − 540 d`.** Per-row tree predictions do not depend on
   the start; the earlier start is kept so that gas can be filled forward. The blend only touches slots after the last
   known price.
9. **Not replicated:** the forecast logs, `cleanup()`, the wall-clock source-horizon gate (`apply_horizon` is patched
   to apply only the cutoff) and `_do_update`'s change detection, because everything is retrained every day.
10. **The tree residual-load model is trained twice per day**, once in the DE base and once in stage-2's
    `with_shared_stores()` copy, exactly as in the app. The saved `rl_*` series come from the base copy.

## Smoke test (2026-11-25, `--vintage strict`, latency 6 h, d1 for 32 points falling back to d2)

| D | trees | Timestral | market | issued blend | RL MAE (GW): trees / mixed / Timestral |
|---|---|---|---|---|---|
| 2026-09-08 | 1.452 | 1.236 | 1.367 | **1.239** | 2.62\* / 2.44 / 2.55 |
| 2026-10-26 (cutoff on the 100-slot DST day) | 1.851 | 1.692 | 1.554 | **1.515** | 4.36\* / 3.83 / 3.61 |

All figures are MAE against the actual price in ct/kWh, except the residual-load column. The `core` curves, which are
not issued, scored 2.27 and 1.86.

Every check passed: no price, gas or generation data after the cutoff in any of the 8 zones; the model read the
overwritten weather; the blend equals the mean of the three forecasts. On D, the DE weather is d1 for the first 25
slots and d2 for the other 71.

\* In the smoke files, `rl_trees` misses the last 3 slots of D. The harness asked for too narrow a range, so the
hourly weather was not interpolated there. This is fixed in the code (it now asks a few hours wider and asserts that
there are no gaps); the 2 smoke files keep the gap.

### Runtime

The smoke test ran on a machine where other jobs held about 3 of the 4 cores (load average about 11):

| phase | 2026-09-08 | 2026-10-26 |
|---|---|---|
| stage 1 (8 bases + residual-load trees) | 88 s | 84 s |
| stage 2 | 62 s | 51 s |
| predict + Timestral, market and blend | 920 s | 1306 s |
| **total** | **18 min** | **24 min** |

Almost all of the time is Timestral-123M on the CPU: three autoregressive forecasts per day (price, supply shift and
residual load), and its 4 threads were starved.

For comparison, the live app on this machine took about 70–210 s from "retraining stage-2" to "blended", including
loading the model (two samples). On an idle 4-core machine, expect about **3–5 min per day**, which is **about 3–5 h
for the 62 days** (inferred, not measured).
At the contention seen here it is about **20 h**. `--torch-threads 1` may help when the machine is shared (untested).
