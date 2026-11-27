# Daily live forecast (team E)

## Command (morning of D-1, machine otherwise idle, start ~09:30 UTC, CET gate 11:00 UTC)

```
SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
cd /home/user/EpexPredictor/research/coupled-auction/io_min/team/E
SP=$SP $SP/tsvenv/bin/python live_forecast.py --day 2026-11-28
```

- Output directory `$SP/live/<D>/`; it refuses to run if that directory already holds an issued forecast (`SHA256SUMS`)
  or another run holds `$SP/live/<D>/.lock` (the lock is released automatically on any exit, abort included).
- Expected runtime on an idle machine ~12-17 min (dry run under load: fetch 6 min + forecasts 6.5 min); hard stop for
  the matched shadow at gate - 55 min (10:05 UTC); app and champion may run to gate - 10 min, so a late start still
  issues (`--deadline YYYY-MM-DDTHH:MM` overrides all). The input log records whether THE gas day D was already served
  (`gas_day_D_present`); the app path keeps it, the champion cuts it.
- If the fetch fails outright: rerun with `--skip-fetch` on a snapshot made earlier (`--out` pointing to a directory whose
  `stores/` and `fetch_log.json` come from an earlier `live_fetch.py`, e.g. the 06:05 prefetch
  `$SP/live/prefetch/2026-11-28_0605`, copied to a new directory first).
- Dry run for a day whose auction is past: `--dry-run` (writes to `$SP/live/dry/<D>/`, labelled, never scored).

## What it does

1. `live_fetch.py`: the app's own store classes update the persistent store directory `$SP/live/stores` (prices for
   8 zones, THE gas, TSO wind/solar/load, weather for 8 zones and the DE grid from `api.open-meteo.com` over the app's
   refresh window now - 1 d .. now + 8 d), then snapshot it to `$SP/live/<D>/stores`.
   - Weather contingency: on a failure (HTTP 429 daily quota) the identical request goes to
     `previous-runs-api.open-meteo.com` (unsuffixed variables = latest forecast, `minutely_15` supported); if that fails
     too, the store keeps the last good fetch. Each store's source and fetch time are logged and printed in every CSV
     header (`primary` / `fallback` / `stored`).
   - Also fetched: Open-Meteo previous-runs d1..d3 for the 58 grid points (matched shadow), energy-charts' day-ahead
     load / wind / solar for D as served at that moment (publication-timing evidence), Open-Meteo model run metadata.
2. If every Open-Meteo host failed and no earlier fetch covers D, the snapshot (never the persistent store) is filled
   from the newest local previous-runs files (d1 > d2 > d3; this run's grid file first), labelled `EMERGENCY fill` in
   every header: a forecast on stale weather is issued, not withheld. Checks that abort: DE prices end exactly at D-1 23:45 local with all of D-1; TSO figures complete for D-1; weather
   covers D. The matched shadow runs only if its d2 grid history is genuine to the cutoff (valid time - 48 h before the
   file's fetch time).
3. Three forecasts in parallel, one thread each, on the same snapshot (hard stops above):
   - `app`: the deployed app path (`app_run.py`: `BaseModelCoordinator._do_update` + `RegionPriceManager`), network off;
   - `champion`: `faithful_bt.py --rl-variant base` with `FAITHFUL_LIVE=1` (the frozen champion config on live inputs);
   - `matched`: `faithful_bt.py --rl-variant matched`, **shadow only** (d2-trained RL boosters on live weather has no
     backtest; S, 2026-11-26).
4. Files in `$SP/live/<D>/`: `team_forecast_<D>.csv` (= champion, else app), `forecast_<D>_{app,champion,matched}.csv`,
   `forecast_<D>_all.csv` (all members + residual load per run and model set, GW), `input_log_<D>.{md,json}`,
   `SHA256SUMS`; the hashes, absolute paths and input log are copied as text to `team/E/live/<D>.md` (no data in the
   repo). CSVs are ct/kWh, one row per local quarter hour of D, '#' header lines (read with `comment="#"`).

## Differences between `app` and `champion`

Both run the app's code on the same stores. `app` trains on [now - 540 d, now + 7 d] exactly as the running app;
`champion` on [cutoff - 540 d, cutoff] (harness deviation 7), and its stage-1/stage-2 are the harness's explicit calls.
Rows after the cutoff have no price, so both train on the same rows except the first ~12 h of the window.
Known app quirk, not reproduced: the running app computes `currts` once per cycle and refreshes the 8 regions with
5-20 s pauses, so after about a minute `needs_history_query` switches its later regions to the historical-forecast host.
Here `now` is recomputed per store, so every store asks the forecast host.

## Why `app` and `champion` differ (dry run 11-27: max |diff| trees 1.79, Timestral 0.31, market 0.32, blend 0.69 ct/kWh)

Same code, same snapshot, one thread each, but not the same training set. Explanation from the code, not tested by an
ablation run:
- **Training window (harness deviation 7):** `app` trains on [now - 540 d, now + 7 d], `champion` on
  [cutoff - 540 d, cutoff]. Rows after the cutoff have no price and are dropped either way. But the window starts about
  6.5 h apart in the dry run (run at 16:20 UTC vs cutoff 22:45 UTC), and about 13 h apart in a 09:30 live run. So the
  8 stage-1 models, the stage-2 model and the residual-load trees all see slightly different rows. LightGBM trees
  are not stable under a change of training rows. This is the main source of the tree difference (1.79 at most,
  smaller on average).
- **Knock-on to Timestral and the market route:** both read the residual load from the trees, mixed with Timestral's
  own. Different trees give a slightly different mixed residual load, hence the 0.3 differences. The price context is
  identical.
- **Gas:** not a factor in the 11-27 dry run, because the app path still cut gas then. From 11-28 the app path keeps
  THE's gas day D if it is served before the gate, and the champion cuts it. That can add a difference on live days.
  `gas_day_D_present` in the input log says whether it applied.
- **Not a cause:**
  - Prediction start: the harness predicts from cutoff - 14 d, the app from now - 540 d. Per-row tree predictions do not
    depend on it.
  - Seeds: LightGBM's default seed is fixed, and Timestral is deterministic given its inputs.

Expect the two forecasts to differ by a few tenths of a ct/kWh on the blend every day.

## Daily rhythm (S, 2026-11-26)

| when (UTC) | what |
|---|---|
| D-1 ~09:30 | `live_forecast.py --day D` (machine reserved 09:20-10:30; no Open-Meteo use by anyone before it ends) |
| D-1 by 10:15 | files hashed; `team/E/live/<D>.md` holds hashes, absolute paths, input log; coordinator sends the team file |
| D-1 ~11:45 | D's prices published: first price score of app vs champion vs matched shadow (posted D-1 afternoon) |
| D+1 | D's TSO wind/solar/load complete: RL score per run and model set (base vs matched boosters on live weather) |

Live and dry runs are never scored inside `live_forecast.py` (actual prices are blanked); scoring is a separate step.

## Incident 2026-11-27 (D = 11-28): the grid weather was not refreshed

- **What happened**: the DE grid store (`gridweather_v1`), which feeds the residual load of every member, kept the
  stored 2026-11-26 15:56 UTC fetch. The 8 zone stores were refreshed between 09:17 and 09:22: DE, AT, BE, SE4 and DK1
  from the forecast host, FR, NL and DK2 from the fallback host.
- **Cause**: Open-Meteo returned 429 ("Daily API request limit exceeded") on and off from 09:17. The grid store came
  last in the loop, after AT had used the retries. Its two attempts at 09:22:22 failed on both hosts, and the shared
  4-minute retry budget was spent, so it was not retried.
- **Quota state**: at 10:37 the previous-runs host reported X-RateLimit-Remaining 0 of 10,000 for the day, with reset
  at 00:00:54 UTC; the forecast host answered. The daily counters appear to be shared beyond our own use: this egress
  IP had exhausted the previous-runs quota by 10:37.
- **Fix in `live_fetch.py`, from 11-28 on**:
  - the grid store is fetched first;
  - each store gets its own retries: primary, fallback, then 40 s pause, then both again, then 80 s pause, then both
    again;
  - the overall cap is 9 min, after which a store keeps its stored forecast, labelled;
  - 15 s spacing between stores, as in the app.
  - In the worst case the fetch takes about 11 min, so a 09:15 start still finishes before 09:45.
