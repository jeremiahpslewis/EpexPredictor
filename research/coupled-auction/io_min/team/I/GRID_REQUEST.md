# More grid points / a second NWP for the wind booster: request list (document only, nothing fetched)

**Rules.**
- Fetching starts only after the 11-27 live run ends (after 10:30 UTC).
- E's walk-forward backfill (~1,080 weighted calls) has priority.
- Everything fetched is public Open-Meteo data. The previous-runs vintages are what was available at the gate.

## Cost model (assumption, to check on the first request's answer)

- **Weighting.** Open-Meteo counts one call per location, multiplied by max(1, variables / 10) and by
  max(1, days / 14).
- **Models.** Asking for several models in one request is assumed to multiply by the number of models.
- **Daily limit.** 10,000 weighted calls.
- **Evidence for the formula.** E's `fetch_wf16.py` uses the same formula, and the earlier `dl_dense.py` fetch of 58
  points × 14 months went through within quota over several requests.
- **Unit cost.** One location × 14 days × ≤ 10 variables = 1 weighted call. A training history from 2025-10-01 to
  2026-11-27 (423 days) costs about 30 weighted calls per location and series.

## Stage 0: sizing pilot (do this first; ~2,700 weighted calls)

**Question.** With near-perfect weather, does a denser grid cut the wind booster's error against the TSO figure? If
not, no vintage history is worth fetching.

- **Data.** Historical-forecast API (archive: latest run per hour), `wind_speed_120m` only.
- **Points.** About 90 new points, densifying the existing 1.2° × 0.8° grid to 0.6° × 0.4° where most wind capacity
  is:
  - latitudes 51.4–54.6 (Lower Saxony, Schleswig-Holstein, Brandenburg, Mecklenburg-Western Pomerania,
    Saxony-Anhalt, northern NRW);
  - land and coastal points only;
  - the existing points are kept.
- **Period and requests.** 2025-10-01..2026-11-27 (423 days ≈ 30 units per point), in requests of ≤ 15 points × 45
  days.
- **Total.** About 90 × 30 ≈ 2,700 weighted calls in about 60 requests.
- **Test.** `wind_lab.py` with app weather (`F_app`, archive) for the existing 58 points against 58 + 90:
  - walk-forward weekly on the 61 + fresh18 days, wind MAE on all slots of D;
  - **go** if the dense grid cuts the archive-weather wind MAE by ≥ 5% on both windows.

## Stage 1, if stage 0 is a go: vintage history for the new points (~2,700)

- `wind_speed_120m_previous_day1` and `_previous_day2` (2 variables) from the previous-runs API, same points and
  period.
- About 90 × 30 ≈ 2,700 weighted calls.
- This feeds both `matched` (d2) and the app-weather booster on d1 slots.
- The live pipeline then needs the 90 extra points in its daily grid request: about 90 extra weighted calls per day.

## Alternative or addition: second NWP on the existing 58 points (~1,750 per series)

- `&models=ecmwf_ifs025` (ECMWF IFS 0.25°) next to the default (best-match: ICON for DE), `wind_speed_100m` (IFS has
  no 120 m).
- **Pilot (archive):** 58 × 30 ≈ 1,750 weighted calls.
- **Vintage history (d1, d2):** another ~1,750, if the previous-runs API serves IFS `_previous_dayK` that far back.
  This must be checked with one 1-point request before committing.
- **Test.** Average or stack the two models' boosters; the go rule is the same as stage 0.
- **Ensemble spread** (ensemble API, 51 members) is far more expensive per call and has no previous-runs archive.
  Not proposed.

## Order and total

| item | weighted calls |
|---|---|
| E's backfill (priority) | ~1,080 |
| stage 0 | ~2,700 |
| IFS archive pilot | ~1,750 |
| total if both pilots run the same day | ~5,500, inside one day's quota with margin for the live run |

Stage 1 and the IFS vintages (~4,450) only follow a go, on a later day.

## Why this is worth only a pilot

- **The remaining RL room beyond `matched` is small:** about 0.07–0.08 ct on the blend (I001/I003). The wind/solar
  oracle is −0.288 against `matched`'s −0.212 on 4 high-miss days.
- **Most remaining wind error at 1.5–2 days is weather error.** The ../../rl study found about half the RL MSE stays
  with archive weather.
- **So more points can only reach the model-error half.** Stage 0 measures exactly that half before any vintage
  fetch.
