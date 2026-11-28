# Open-Meteo weather from the AWS open-data bucket (team E, 2026-11-28)

Why: the Open-Meteo HTTP API's daily quota on this shared egress IP is used up by others within minutes of the 00:00
UTC reset. The public bucket `s3://openmeteo` (https://openmeteo.s3.amazonaws.com/, no quota, anonymous ranged reads)
holds the same model data. Scripts: `s3_weather.py` (reader), `s3_validate.py` (comparison with API data on disk);
`live_fetch.py --weather-source` (wiring). They run in `$SP/omvenv` (uv venv: omfiles 1.2.0, s3fs, pyproj, pandas).

## What the app requests (predictor/, unchanged)

| store | resolution | variables | points |
|---|---|---|---|
| `weather_v2_<zone>` (8 zones) | minutely_15 | wind_speed_80m, temperature_2m, global_tilted_irradiance (tilt 0, azimuth 0), pressure_msl, relative_humidity_2m | zone `latitudes` + `extra_weather_*` (DE 11, AT 5, BE 4, FR 5, NL 4, SE4/DK1/DK2 3) |
| `gridweather_v1_DE_LU` | hourly | wind_speed_120m, shortwave_radiation, temperature_2m | 52 onshore + 6 offshore DE grid points |

No `models=` parameter: the API default ("best match") is used. Backtest vintages: the same variables with
`_previous_day1..3` (hourly) from previous-runs-api.

## What the bucket has

- `data/<model>/<var>/chunk_<n>.om`: the rolling series the API reads (newest run overwrites; past = stitched short
  leads; hourly, already time-interpolated). Years of history but **no `_previous_dayN` variables**.
- `data_run/<model>/YYYY/MM/DD/HHMMZ/<var>.om` + `meta.json` (`created_at`): single runs, native steps (ICON-EU/ICON
  3-hourly after 78 h, IFS 3-hourly; radiation starts one step after init; each file carries its own `time`).
  Retention ~3 months: **first runs 2026-08-25/26** for dwd_icon_d2, dwd_icon_eu, dwd_icon, ecmwf_ifs025; the
  `*_native` ICON grids and `dwd_icon_d2_ruc_native` from 2026-10. At most one run per 3 h is kept (RUC runs hourly, so
  2 of 3 RUC runs are missing). ICON-EU 03/09/15/21Z runs end after 30 h; ICON 06/18Z after 120 h.
- Stored: u/v wind (10/80/120/180 m for ICON-D2/EU, ICON data_run without 80/120 m, IFS 10/100 m), temperature_2m,
  relative_humidity_2m, pressure_msl, direct + diffuse radiation (IFS: shortwave), cloud cover. Speed = hypot × 3.6.
- Latency (data_run `created_at` − init): ICON-D2 ~2 h, ICON-EU ~3.5 h, ICON ~3.4 h, IFS ~7 h; `data/` is
  updated earlier (API metadata availability: D2 ~1.4 h, EU ~2.9-3.5 h, ICON ~3.3 h, IFS ~8 h).

## Match against the API (data on disk; S3 only, no API calls)

Reference: `$SP/live/prefetch/2026-11-29_0011/stores/gridweather_v1_DE_LU.json.gz` (api.open-meteo.com, 00:12 UTC
11-28), reproduced with `latest --asof 00:12` (D2 > EU > ICON > IFS, nearest cell). |diff| mean / exact share:

| variable | all leads | lead 11-29 03-17Z (plain ICON-D2) | 11-27..11-29 02Z | 11-30..12-02 (ICON-EU) | 12-03.. |
|---|---|---|---|---|---|
| wind_speed_120m km/h | 1.63 / 16 % (max 42) | 84 % exact | 0.9-1.9 / 3 % | 1.3-2.1 / ~12 % | 1.4-2.1 / 4-25 % |
| shortwave W/m2 | 9.7 / 64 % (incl. night) | 79-92 % of daylight values | 4-5 | ~11 | 11-21 |
| temperature_2m K | 0.35 / 18 % (max 5.8) | 0.22 / 30 % | 0.3 | 0.3-0.4 | 0.3-0.6 |

Vintages vs `grid_prev_live.parquet` (previous-runs API, 11-21..11-29, genuine part only), rule "newest run with
init <= valid − 24k h that reaches it, lead < 24(k+1) h":
d1 wind 0.74 km/h mean, 82 % exact; d1 radiation 94 % exact; d2/d3 wind 1.4-2.0 km/h, 11-15 % exact; temperature
0.2-0.37 K, 16-34 % exact.

**So S3 values of one model do not reproduce the API's default exactly.** What the API blends (found by matching):
1. **ICON-D2-RUC** first, for roughly the first 27 h: nothing in the regular-grid ICON-D2/EU runs matches the first
   day, and exactness jumps from 3 % to 84 % at 11-29 03Z (= the 23Z RUC run + 28 h). RUC is on S3 only on the native
   grid and only every 3rd run, so the API's exact RUC run cannot be rebuilt.
2. **KNMI Harmonie-Arome Netherlands** on the 9 western grid points (lon <= 7.5 E: points 10, 15, 20, 21, 27, 28, 34,
   41, 55): point 20 temperature matches it (mean 0.04 K), not ICON-D2.
3. **ICON-D2** to 48 h on the other 49 points (wind and radiation exact on the nearest regular cell), then
   **ICON-EU** (18Z run, native steps match within ~0.3 km/h), then ICON.
4. **temperature_2m** is never exact from ICON-D2 at any cell (probably an elevation correction): expect ~0.2-0.4 K.

Zone store BE (15 min, `rolling` vs the 09:46 fallback-host fetch, after D2's 09Z run had already replaced the data):
wind 4.6 km/h, temperature 1.3 K, GTI 14 W/m2, pressure 2.1 hPa mean — a weak reference (unknown run, KNMI domain),
but clearly not a drop-in match.

## Wiring (`live_fetch.py`, default unchanged)

`--weather-source api|api-then-s3|s3`, or env `LIVE_WEATHER_SOURCE` (`live_forecast.py` passes the environment through):
- `api` (default): as before.
- `api-then-s3`: API primary + fallback host once each, then S3 `rolling` into the app store (complete rows only,
  `_update_data` + `serialize`), else keep the stored forecast. Grid previous-runs and model metadata also fall back
  to S3. Logged per store as source `s3`, with the S3 file under `$SP/live/<D>/s3_weather_<store>.parquet(.json)`.
- `s3`: no API call at all.
Smoke test on copied stores (never `$SP/live/stores`): GRID and DE written, columns as the app expects, 205 / 814
complete rows (11-27 00Z .. 12-05 12Z; beyond that IFS has no 80/120 m wind - power-law from 100 m is used, still
NaN-free only to the ICON end), ~1.5 min per store.

## Walk-forward backfill (fetch_wf16.py)

Cannot be served from S3: `data_run` starts 2026-08-25 and `data/` has no previous-day vintages (its past is the
latest-run stitch). The 12 days 2025-12-02 .. 2026-08-15 still need the previous-runs API. Days from 2026-08-26 on
(09-07, 10-08, 10-16, 11-02) can be rebuilt with `s3_weather.py vintage`, with the d1/d2 accuracy above (not exact).

## Remaining

- Add ICON-D2-RUC (native grid, `data/dwd_icon_d2_ruc_native`; 120 m wind is in `*_model_level`) and KNMI NL to the
  blend for `rolling`, with the native-grid cell lookup (`static/coordinates.om`), and find the temperature correction.
- Re-validate `rolling` against a same-minute API fetch (the 09:38 reference was overwritten before it could be read).
- Zone stores: check AROME/KNMI coverage per zone and Open-Meteo's 15-min radiation interpolation.
