# Long walk-forward window: sample, coverage, cost (team E, 2026-11-26)

## Sample

`sample_core36.txt` (written by `sample_days.py` before any scoring, seed 20261126): 3 days per month for
2025-12..2026-11, one per third of the month, weekdays assigned so each appears 5-6 times. It is never changed after
scoring starts; an extension (2 per ISO week) would be drawn separately.

## Data coverage for 2025-12-01..2026-11-26 (checked)

| input | from | to | consequence |
|---|---|---|---|
| prices, 8 zones; gas | 2024-02/03 | 11-26 (live store: on) | full |
| zone weather archive (`weather_v2`) | 2024-03-01 | 11-26 | full |
| DE grid weather archive (`gridweather_v1`) | 2025-04-06 | 11-26 | price trees start there (harness deviation 5), as on the 61 days |
| TSO generation figures | 2025-05-24 | 11-26 | RL trees for a December 2025 day have ~6 months of history (min 60 days met) |
| zone vintages d2/d3 (`prev_all`) | 2025-08-28 | 11-10 (+ fresh 11-01..11-28) | full |
| DE onshore d1 (`prevruns_DE`) | 2025-08-28 | 11-09 | full |
| d1 for 27 zone points + 5 DE North Sea points (`prev_d1`) | 2026-09-01 | 11-09 | before 2026-09: falls back to d2 (first ~6 h of D only), flagged `fallback_cells` |
| **58-point grid d1/d2/d3 (`dense/prev`)** | **2026-08-25** | 11-10 | **missing before 2026-08-25** |
| grid d2 history (`dense/hist`, matched training) | 2025-10-01 | 11-10 (+ fresh) | matched trains on >= 61 days for the first sample day |

**Gap to close before the champion run:** the grid vintages before 2026-08-25. `hist.parquet` holds d2 from 2025-10-01,
so D's d2 slots could be filled from it, but d1 (first ~6 h of D, falls back to d2: conservative, allowed) and d3
(D+1 afternoon / D+2, Timestral's RL horizon: no older vintage to fall back to, becomes NaN) are absent. Proposal:
fetch grid d1/d3 for 2025-11-25..2026-08-25 from the previous-runs API (58 points, 4 requests per ~month chunk;
roughly 1-2.5k weighted Open-Meteo calls), **after** the live run on 11-27 (quota embargo until then), and add a
`--grid-hist` source to `faithful_bt.py` that reads it plus `hist` d2. Without it, the 24 sampled days before
2026-08-25 cannot run strict.

## Cost (measured, not estimated)

Per day, one thread per process, 3-4 processes on 4 cores (stored meta, `out_fresh_strict`, `out_strict_rlmatched`):

| run | stage 1 | stage 2 | predict + Timestral + market + blend | total |
|---|---|---|---|---|
| champion (base), 18 fresh days | 74 s | 61 s | 102 s | **237 s** (max 308) |
| matched, 61 days | 160 s | 55 s | 101 s (median) | **320 s median** (mean 382) |
| base on an idle machine, default threads (`out_strict`) | 36 s | 33 s | 33 s | 99 s |

- Live dry run 2026-11-26 (1 thread, 3-4 processes on the machine): champion 255 s, matched 393 s - consistent.
- With 9 days per process, 4 processes x 1 thread give about 9 x 240 s = 36 min (the 99 s/day idle figure is one
  process with all 4 cores). **One wave fits with margin at 16 days (4 per process, ~16-17 min)**; 20 days (5 per
  process, ~20-21 min) is too tight. A cut rule must be fixed before any scoring; proposal: the first-third day of every
  month (12) plus the second-third day of 4 months chosen by the same seed (16). Alternative: ask the user for two
  waves of 18. Decision for S / the user.
- **36 days do not fit one <= 20 min wave.** 4 processes x 1200 s = 4800 process-seconds; 36 x 237 s = 8530. One wave
  holds about **16-20 base days** (4-5 per process) or about 14 matched days. Per S's rule this goes to the user rather than
  being split into chained waves.
- **Weekly refit does not make the sparse sample cheaper.** The sampled days lie in different weeks, so every day
  trains anyway. With models reused (consecutive days), a day costs only the predict/Timestral/blend part, about
  100 s at one thread (33 s idle): useful for dense windows (the 61 / 18 days), not for this sample.
- Options for the user: (a) 16-day core (12 months, 1-2 per month) as one wave; (b) 36 days as two waves on two
  separate occasions, explicitly approved; (c) champion cache built once with more wall time (one-off), challengers
  screened on the cache and only finalists rerun.

## Training history for early sample days

The 540-day window reaches back to mid-2024 for a December 2025 day, and prices / zone weather / gas exist there, but
rows need the residual-load feature: grid weather starts 2025-04-06 and TSO figures 2025-05-24. So for D = 2025-12-02
the price trees train on ~8 months (not 18) and the RL trees on ~6 months; matched RL on 2 months (d2 history from
2025-10-01, just over the 60-day minimum). This handicaps champion and challengers equally, but the champion's
early-month MAE is pessimistic relative to a mature deployment.

## Scorer

`score_cache.py --champion DIR --challenger DIR [--days sample_core36.txt]`: per member and blend, window MAE, rMAE vs
naive, paired diff with the circular 7-day block bootstrap (90 %), months won, promotion flag (>= 9/12 months and
interval excluding 0). Checked: it reproduces `RESULTS_MATCHED.md`'s fresh-day figures exactly.
