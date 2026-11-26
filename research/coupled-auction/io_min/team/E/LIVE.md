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

## Daily rhythm (S, 2026-11-26)

| when (UTC) | what |
|---|---|
| D-1 ~09:30 | `live_forecast.py --day D` (machine reserved 09:20-10:30; no Open-Meteo use by anyone before it ends) |
| D-1 by 10:15 | files hashed; `team/E/live/<D>.md` holds hashes, absolute paths, input log; coordinator sends the team file |
| D-1 ~11:45 | D's prices published: first price score of app vs champion vs matched shadow (posted D-1 afternoon) |
| D+1 | D's TSO wind/solar/load complete: RL score per run and model set (base vs matched boosters on live weather) |

Live and dry runs are never scored inside `live_forecast.py` (actual prices are blanked); scoring is a separate step.
