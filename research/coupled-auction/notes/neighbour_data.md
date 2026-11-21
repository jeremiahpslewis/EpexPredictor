# Day-ahead data for an aggregate neighbour zone of DE_LU

## What this is

This note covers the public ENTSO-E day-ahead data assembled for one aggregate "neighbour" zone of Germany (DE_LU). The zones are the ten SDAC bidding zones coupled to DE_LU in the day-ahead auction: FR, NL, BE, AT, CZ, PL, DK_1, DK_2, SE_4 and NO_2. Switzerland (CH) is fetched as well but kept apart, because it is not part of the EU day-ahead coupling. The period is 2025-08-01 00:00 to 2026-11-09 23:45 (Europe/Berlin), queried with the end date 2026-11-10 exclusive. This note only assembles and documents the data; no modelling is done.

The script is `research/coupled-auction/neighbour_data.py` (entsoe-py 0.8.1). It follows the set-up of `research/market-model/exp/entsoe_da.py`: quarterly chunks (2025-08, 2025-11, 2026-02, 2026-05, 2026-08, plus 2026-11 for the last nine days), resumable, and up to four attempts with growing back-off. Installed capacity is queried per calendar year (2025 and 2026). Every chunk ends in one of three states:
- a parquet file, when data came back;
- a `.nodata` marker, when the API said there was no matching data (never retried);
- nothing, when it failed after retries, so a rerun fetches it again.

The run finished with no failed chunks. A few transient HTTP 599 errors and one read timeout were resolved by retrying. The fetch ran as three workers on disjoint zone lists, with one request per worker at a time and a 1 s pause, because each request took about 30 s on the server side. The API key is read from the scratchpad and masked out of every logged error; a search of the logs and outputs for it came back empty.

Data lives only in the scratchpad under `entsoe/neighbours/`:
- `raw/<zone>/<series>_<chunk>.parquet` and `.nodata`: the raw chunks;
- `derived/<zone>_<series>_15min.parquet`: the quarter-hour series from 2025-10-01;
- `derived/DE_LU_sched_net_export_15min.parquet`: per-border net export of DE_LU;
- `summary/*.csv`: availability, NTC, reconciliation, mean net export and capacity 2026;
- `fetch.log`: the fetch log.

Run `neighbour_data.py process` to regenerate everything in `derived/` and `summary/` from the raw chunks.

## Series and queries

| Name | Content | entsoe-py call |
|---|---|---|
| load | day-ahead total load forecast | `query_load_forecast(Z)` |
| ws | day-ahead solar, wind onshore and wind offshore forecasts | `query_wind_and_solar_forecast(Z)` |
| price | day-ahead price | `query_day_ahead_prices(Z)` |
| sched_out / sched_in | day-ahead scheduled commercial exchange, DE_LU→Z and Z→DE_LU | `query_scheduled_exchanges(..., dayahead=True)` |
| ntc_out / ntc_in | day-ahead NTC, DE_LU→Z and Z→DE_LU | `query_net_transfer_capacity_dayahead` |
| cap | installed capacity per production type, per year | `query_installed_generation_capacity(Z)` |

All eleven zones border DE_LU directly, so each one has exchange and NTC queries in both directions. The connections are:
- AC borders: FR, NL, AT, CZ, PL, DK_1, CH;
- DC links: BE (ALEGrO), DK_2 (Kontek), SE_4 (Baltic Cable), NO_2 (NordLink).

## Native resolution, gaps and the 15-minute series

Native resolution is taken per column and per local calendar day, as the most common spacing between that column's timestamps on that day. It has to be per column because one response can mix resolutions: since 2026-10-26, BE solar is 15-minute while BE wind is still hourly. The table below shows it separately for August–September 2025 and from 1 October 2025, the SDAC quarter-hour go-live. Where a zone switched mid-period, the table gives the date of the switch.

Each native value is spread over its own interval on a 15-minute grid. An hourly value is repeated over its four quarter hours, unchanged. It is not interpolated, and it is not divided by four, since these are MW averages and EUR/MWh prices. The interval is the distance to the next timestamp, capped at that day's native resolution (and at 60 minutes). A missing timestamp therefore stays empty rather than being filled from the previous value. The grid is built tz-aware in Europe/Berlin, so the DST days have 92 and 100 quarter hours.

The quarter-hour files in `derived/` cover 2025-10-01 to 2026-11-09 23:45. They are the input to use for the quarter-hourly market. Before October the raw native data is kept as it is.

Each column is expanded on its own, after dropping the NaN padding that a mixed frame carries. The gap share is the fraction of quarter hours on that grid with no value, counting both missing timestamps and NaN. It is reported for August–September 2025, for the period from October, and for the whole period. Chunk boundaries are inclusive in some entsoe-py calls, so duplicate timestamps are dropped when chunks are joined. Every remaining gap was checked by date and by minute of the hour. All are whole missing days, or the extra hour of an autumn DST day. None is the :15/:30/:45 pattern that would indicate hourly data counted against a 15-minute grid.

Some resolutions come from entsoe-py itself rather than from the data. `query_day_ahead_prices` and `query_net_position` force SDAC values to 60 minutes before 2025-10-01 and to 15 minutes after. The resolutions shown for SDAC prices are therefore set by the library, not observed from the data.

| Zone | Series | Available | Native resolution, Aug–Sep 2025 | Native resolution, from 1 Oct 2025 | Gap share Aug–Sep / from Oct / whole period | First – last stamp |
|---|---|---|---|---|---|---|
| FR | load forecast | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| FR | solar | yes | 60 min | 60 min, then 15 min from 2026-01-09 | 0.0% / 0.3% / 0.2% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| FR | wind onshore | yes | 60 min | 60 min, then 15 min from 2026-01-09 | 0.0% / 0.3% / 0.2% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| FR | wind offshore | yes | 60 min | 60 min, then 15 min from 2026-01-09 | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| FR | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| FR | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| FR | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| FR | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| FR | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| FR | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 15 production types with capacity > 0 in 2026 |
| NL | load forecast | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NL | solar | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NL | wind onshore | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NL | wind offshore | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NL | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NL | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NL | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NL | NTC DE_LU→Z | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| NL | NTC Z→DE_LU | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| NL | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 10 production types with capacity > 0 in 2026 |
| BE | load forecast | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| BE | solar | yes | 60 min | 60 min, then 15 min from 2026-10-26 | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| BE | wind onshore | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| BE | wind offshore | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| BE | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| BE | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| BE | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| BE | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| BE | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| BE | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 12 production types with capacity > 0 in 2026 |
| AT | load forecast | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| AT | solar | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| AT | wind onshore | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| AT | wind offshore | no (column not returned) | – | – | – | – |
| AT | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| AT | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| AT | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| AT | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| AT | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| AT | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 11 production types with capacity > 0 in 2026 |
| CZ | load forecast | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CZ | solar | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CZ | wind onshore | no (column not returned) | – | – | – | – |
| CZ | wind offshore | no (column not returned) | – | – | – | – |
| CZ | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CZ | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CZ | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CZ | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| CZ | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| CZ | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 13 production types with capacity > 0 in 2026 |
| PL | load forecast | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| PL | solar | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| PL | wind onshore | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| PL | wind offshore | yes | - | 15 min | 100.0% / 64.2% / 68.9% | 2026-06-18 00:00 – 2026-11-09 23:45 |
| PL | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| PL | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| PL | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| PL | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| PL | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| PL | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 12 production types with capacity > 0 in 2026 |
| DK_1 | load forecast | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| DK_1 | solar | yes | 15 min | 15 min | 1.6% / 0.2% / 0.4% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | wind onshore | yes | 15 min | 15 min | 4.9% / 0.2% / 0.9% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | wind offshore | yes | 15 min | 15 min | 4.9% / 0.7% / 1.3% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | NTC DE_LU→Z | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | NTC Z→DE_LU | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_1 | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 8 production types with capacity > 0 in 2026 |
| DK_2 | load forecast | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| DK_2 | solar | yes | 15 min | 15 min | 1.6% / 1.2% / 1.3% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_2 | wind onshore | yes | 15 min | 15 min | 4.9% / 0.2% / 0.9% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_2 | wind offshore | yes | 15 min | 15 min | 4.9% / 0.7% / 1.3% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_2 | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_2 | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_2 | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| DK_2 | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| DK_2 | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| DK_2 | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 7 production types with capacity > 0 in 2026 |
| SE_4 | load forecast | yes | 60 min | 60 min, then 15 min from 2025-12-02 | 0.0% / 0.2% / 0.2% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| SE_4 | solar | yes | 60 min | 60 min, then 15 min from 2025-11-25 | 0.0% / 1.2% / 1.1% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| SE_4 | wind onshore | yes | 60 min | 60 min, then 15 min from 2025-11-25 | 0.0% / 1.2% / 1.1% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| SE_4 | wind offshore | no (column not returned) | – | – | – | – |
| SE_4 | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| SE_4 | sched. DE_LU→Z | yes | 60 min | 15 min | 1.6% / 0.0% / 0.2% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| SE_4 | sched. Z→DE_LU | yes | 60 min | 15 min | 1.6% / 0.0% / 0.2% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| SE_4 | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| SE_4 | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| SE_4 | installed capacity | yes: 2026 (2025: no matching data) | yearly | yearly | – | 3 production types with capacity > 0 in 2026 |
| NO_2 | load forecast | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NO_2 | solar | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NO_2 | wind onshore | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NO_2 | wind offshore | yes | 15 min | 15 min | 65.6% / 0.0% / 8.6% | 2025-09-10 00:00 – 2026-11-09 23:45 |
| NO_2 | DA price | yes | 60 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NO_2 | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NO_2 | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| NO_2 | NTC DE_LU→Z | no (API: no matching data) | – | – | – | – |
| NO_2 | NTC Z→DE_LU | no (API: no matching data) | – | – | – | – |
| NO_2 | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 7 production types with capacity > 0 in 2026 |
| CH | load forecast | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| CH | solar | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| CH | wind onshore | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| CH | wind offshore | no (column not returned) | – | – | – | – |
| CH | DA price | yes | 60 min | 60 min, then 15 min from 2026-11-04 | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CH | sched. DE_LU→Z | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CH | sched. Z→DE_LU | yes | 15 min | 15 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:45 |
| CH | NTC DE_LU→Z | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| CH | NTC Z→DE_LU | yes | 60 min | 60 min | 0.0% / 0.0% / 0.0% | 2025-08-01 00:00 – 2026-11-09 23:00 |
| CH | installed capacity | yes: 2025, 2026 | yearly | yearly | – | 4 production types with capacity > 0 in 2026 |

"No" means that every chunk came back with no matching data from the API. Those chunks are marked  and are not missing downloads.

**Notable points in the table:**
- **CZ:** the wind/solar forecast has only a Solar column. ENTSO-E publishes no day-ahead wind forecast for CZ in this period.
- **PL wind offshore:** the column only starts on 2026-06-18, presumably when the first PL offshore farm began delivering. Its mean is 121 MW and its maximum 356 MW. The 64% "gap" from October is the period before the farm existed, not missing data. In a model, read it as zero before that date.
- **NO_2 wind offshore:** it starts on 2025-09-10 and is tiny, with a mean of 2 MW and a maximum of 6 MW. It matches the 6 MW of offshore capacity reported for NO_2, and is likewise zero before that date.
- **Switches to 15-minute resolution:** several wind/solar forecasts moved from hourly to 15-minute well after 1 October 2025:
  - SE_4 on 2025-11-25, with the SE_4 load forecast following on 2025-12-02;
  - FR on 2026-01-09;
  - BE solar on 2026-10-26, while BE wind stays hourly.
- **Still hourly:** the load forecasts of DK_1, DK_2 and CH, and all CH forecasts.
- **Prices:**
  - All SDAC prices are hourly until 2025-09-30 and 15-minute from 2025-10-01.
  - The CH price is hourly until 2026-11-03. From 2026-11-04 it comes back at 15-minute resolution. I did not look into why.
- **Real gaps:** all are whole missing days. The largest are:
  - DK_1 and DK_2 wind and solar: 1.6–4.9% in August–September 2025 (2025-08-30, 2025-09-23 and 2025-09-24). After that, DK_1 and DK_2 miss one to five further days each, such as 2026-04-14 and 2026-05-07/08 for offshore wind;
  - SE_4 wind and solar: 1.2%, which is five days (2025-11-01/02, 2026-03-11, 2026-04-29, 2026-06-26);
  - SE_4 load: 2025-12-04;
  - FR wind onshore and solar: 2026-01-02;
  - SE_4 exchanges: 2025-09-30 is missing in both directions.
  - Everything else is complete, or nearly so.

## NTC

The day-ahead NTC exists on only three borders, and only CH and DK_1 publish a full one. FR, CZ and PL return no data, as expected for flow-based Core borders. The same holds for AT and BE, which are also Core borders, and for DK_2, SE_4 and NO_2. For those three DC links the capacity is not published as a day-ahead NTC on this endpoint.

| Border | Resolution | p10 | Median | p90 | Min | Max | Share of zero values |
|---|---|---|---|---|---|---|---|
| DE_LU→DK_1 | 60 min to 2025-09-30, then 15 min | 2500 | 2500 | 3500 | 665 | 3500 | 0% |
| DK_1→DE_LU | same | 2500 | 2500 | 3500 | 600 | 3500 | 0% |
| DE_LU→CH | 60 min | 800 | 900 | 1619 | 600 | 2200 | 0% |
| CH→DE_LU | 60 min | 2380 | 3591 | 4000 | 0 | 4000 | 0% |
| DE_LU→NL | 60 min | 0 | 1080 | 1975 | 0 | 4818 | 23% |
| NL→DE_LU | 60 min | 0 | 1081 | 2971 | 0 | 7477 | 15% |

All values are in MW, over the whole period.

**DK_1:** the border runs at two levels, 2500 and 3500 MW, in both directions, with occasional reductions.

**CH:** the border is strongly asymmetric. Import capacity into Germany is typically 3.6 GW, while export capacity to Switzerland is typically 0.9 GW.

**NL:** this is the "partly" case. Until January 2026 both directions are a flat placeholder of about 1080 MW, constant at every hour. From February 2026 the values vary: in some hours they reach 4–7 GW, but many hours are zero. The zero share rises over 2026. For DE_LU→NL it reaches 61% in September 2026, 78% in October and 90% in early November; NL→DE_LU runs at 12–46% zeros. DE–NL is part of Core flow-based, so these figures look like a derived or residual NTC rather than a binding limit. This is borne out by the schedules, which exceed the published NTC in about a quarter of quarter hours in both directions (25% DE_LU→NL, 22% NL→DE_LU). It is not a binding limit and should not be used as a constraint. On DK_1 the scheduled exchange never exceeds the NTC, and on CH it does so in only 1.1% (DE_LU→CH) and 0.1% (CH→DE_LU) of quarter hours. This also confirms that the direction labels are the right way round.

## Do the scheduled exchanges reconcile with the DE_LU net position?

The net position used here is `$SP/entsoe/da/netpos_*.parquet`, from `query_net_position(DE_LU, dayahead=True)`. In entsoe-py, positive means export. The comparison takes:
- the DE_LU net export per border, defined as (DE_LU→Z) minus (Z→DE_LU);
- summed over the borders;
- compared stamp by stamp with the net position;
- on the 15-minute grid from 2025-10-01, and on hourly means before that.

Residuals are the summed exchanges minus the net position, in MW.

| Case | Stamps | Bias | MAE | p99 \|res\| | Max \|res\| | Correlation |
|---|---|---|---|---|---|---|
| 15 min from 2025-10-01, 10 SDAC borders | 38 884 | −10.3 | 12.5 | 50.6 | 1 827 | 0.99997 |
| 15 min from 2025-10-01, 10 SDAC borders + CH | 38 884 | +197.9 | 1 061 | 3 997 | 4 041 | 0.984 |
| hourly 2025-08-01 to 09-30, 10 SDAC borders | 1 440 | −10.3 | 10.5 | 37.7 | 53 | 1.00000 |
| hourly 2025-08-01 to 09-30, 10 SDAC borders + CH | 1 440 | −473.4 | 1 293 | 4 014 | 4 016 | 0.975 |

**Result:** over the ten SDAC borders, the scheduled exchanges reconcile with the net position to within a few tens of MW. The mean net position is about −0.5 GW, and the median residual is −5 MW. Adding CH breaks the reconciliation, with an MAE above 1 GW. The SDAC net position therefore does not include the Swiss exchange, as expected, since the CH border is allocated outside SDAC.

**Remaining residual:** there is a small negative bias of about −5 to −17 MW per month. It is larger when Germany exports than when it imports, on most borders: NO_2 −32/−3 MW, DK_1 −24/−5 MW, SE_4 −15/−10 MW, DK_2 −14/−8 MW, but also BE −15/−7 MW and NL −14/−7 MW. Losses on the HVDC links (NordLink, Baltic Cable, Kontek, ALEGrO) could explain part of it. Since the AC borders DK_1 and NL show the same asymmetry, I could not identify the cause from this data.

**Outliers:** 84 of the 38 884 quarter hours (0.2%) differ by more than 200 MW.
- Forty-five of them fall on the spring DST day, 2026-03-29. That day the BE (ALEGrO) schedule, as returned by entsoe-py, is one hour out of step; I cannot tell whether that comes from the publication or from the parsing. Moving the BE series one hour earlier brings the whole-day MAE from 551 MW to 113 MW, and the number of quarter hours above 200 MW that day from 45 to 9. The raw data is left uncorrected.
- The rest are isolated quarter hours of 200–500 MW, mainly in September to November 2026, with no single border responsible.

On average from October 2025, DE_LU exported over the following borders:

| Border | Mean net export (MW) |
|---|---|
| AT | +1 455 |
| CZ | +427 |
| PL | +273 |
| CH | +208 |

It imported over these:

| Border | Mean net export (MW) |
|---|---|
| DK_1 | −927 |
| FR | −886 |
| NO_2 | −451 |
| NL | −184 |
| BE | −104 |
| DK_2 | −81 |
| SE_4 | −65 |

## Installed capacity of the aggregate neighbour, 2026

These figures use the ENTSO-E installed capacity per production type for 2026, as published at the start of the year. All zones have a 2026 value. SE_4 has nothing for 2025. The aggregate is the sum of the ten SDAC zones; CH is shown separately. All values are in MW.

| Production type | Aggregate (10 SDAC zones) | Largest contributors | CH |
|---|---|---|---|
| Solar | 108 170 | NL 29 282, FR 28 254, PL 20 734, BE 11 683, AT 9 766 | 0 (not reported) |
| Nuclear | 69 572 | FR 62 990, CZ 4 040, BE 2 056 | 2 983 |
| Wind onshore | 56 243 | FR 23 672, PL 10 700, NL 7 178, AT 4 140 | 0 (not reported) |
| Fossil gas | 52 428 | NL 18 412, FR 12 801, BE 8 402, PL 5 431, AT 4 276 | 0 |
| Fossil hard coal | 26 562 | PL 18 543, NL 4 012, FR 1 811 | 0 |
| Hydro water reservoir | 22 780 | NO_2 9 854, FR 8 791, AT 2 547 | 5 799 |
| Hydro run-of-river and pondage | 20 086 | FR 11 748, AT 5 937, NO_2 1 569 | 632 |
| Fossil brown coal/lignite | 13 991 | CZ 7 040, PL 6 951 | 0 |
| Hydro pumped storage | 13 622 | FR 5 061, AT 4 022, PL 1 525, BE 1 310 | 6 702 |
| Wind offshore | 11 174 | NL 4 739, BE 2 262, DK_1 1 615, FR 1 528, DK_2 1 024 | 0 |
| Other | 6 935 | SE_4 2 160, PL 1 980, FR 1 249 | 0 |
| Biomass | 6 600 | FR 1 435, DK_2 1 238, DK_1 977 | 0 |
| Fossil oil | 4 658 | FR 2 986, DK_2 504, BE 477 | 0 |
| Waste | 2 457 | FR 1 040, NL 777 | 0 |
| Energy storage | 1 962 | AT 1 100, BE 862 | 0 |
| Fossil coal-derived gas | 879 | PL 499, CZ 380 | 0 |
| Other renewable | 460 | CZ 460 | 0 |
| Marine | 243 | FR 243 | 0 |
| Geothermal | 2 | FR 2 | 0 |

**Totals:** the 2026 total is about 418 GW for the ten zones and 16.1 GW for CH. The full per-zone table is in `summary/capacity_2026.csv`.

**Points to keep in mind when using these figures:**
- **NL:** the 2026 figures are identical to 2025 in every type, so NL apparently did not update its publication.
- **CH:** the capacity reports no solar and no wind at all, although CH publishes solar and wind forecasts. It covers hydro and nuclear only and understates the Swiss fleet.
- **SE_4:** the 2026 entry has wind onshore 2 134, "Other" 2 160 and hydro 347, with no solar. There is no 2025 entry.
- **PL:** offshore capacity is 0, although the PL offshore forecast starts in June 2026. The yearly figure predates the farm.
- **DK_1 and DK_2:** solar (2 386 and 734) looks low.
- **BE:** nuclear falls from 3 929 MW in 2025 to 2 056 MW in 2026, following the reactor closures.
