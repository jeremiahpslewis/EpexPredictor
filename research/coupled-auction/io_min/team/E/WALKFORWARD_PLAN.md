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

## Option B (coordinator decision, 2026-11-26 evening): one wave of 16 days

**Sample** (`sample_wf16.txt`, written by `sample_days.py` before any scoring): the first-third day of every month of
`sample_core36.txt` (12) plus the second-third day of one month per quarter, drawn with seed 20261126 + 16.
Quarters are meteorological: Q1 = Dec–Feb, Q2 = Mar–May, Q3 = Jun–Aug, Q4 = Sep–Nov. Each quarter has 4 days.

| quarter | days |
|---|---|
| Q1 (DJF) | 2025-12-02, 2025-12-17, 2026-01-02, 2026-02-03 |
| Q2 (MAM) | 2026-03-05, 2026-03-18, 2026-04-05, 2026-05-05 |
| Q3 (JJA) | 2026-06-05, 2026-07-07, 2026-08-07, 2026-08-15 |
| Q4 (SON) | 2026-09-07, 2026-10-08, 2026-10-16, 2026-11-02 |

Caveats, stated before scoring:
- **Weekends are under-represented**: 2 of 16 days, against about 4.6 expected. This follows from the rule
  (first-third days); it is not changed now.
- **Overlap with the used window**: 10-08, 10-16 and 11-02 are also among the 61 used days. A pooled set
  "61 + fresh18 + 16" therefore counts each distinct day once: **92 days**.
- **No fresh day**: every day is on or before 11-08, so no process needs `FAITHFUL_FRESH=1`.

**Backfill** (`fetch_wf16.py`; run only after the live run ends, after 10:30 UTC; `--dry` prints the plan):
- **Which days**: the 12 days from 2025-12-02 to 2026-08-15. Their window D-1..D+2 lies before the local grid
  vintages (2026-08-25) and before the local zone d1 files (2026-09-01).
- **Requests per day**:
  - grid: 58 points × w120/sw/t2 × previous_day1..3, in 4 requests of ≤ 15 points;
  - zone d1: 8 requests, one per zone (27 points plus DE's 5 North Sea points, 5 variables, previous_day1).
- **Size**: each request covers 4 dates and fewer than 10 variables, so each location counts as one weighted call.
  That is about 90 weighted calls and 12 requests per day, and **~1,080 weighted calls and 144 requests** in total.
  The previous-runs host's daily limit is 10,000, and the 5-s pacing keeps it under the per-minute limit.
- **Batching**: one batch is one day; `SP/wf16_hist/days/<D>_{grid,zones}.parquet` is the resume marker. The script
  stops at the first HTTP error and is rerun later. `--max-days N` spreads the batches out.
- **Output**: merged into `SP/wf16_hist/grid_prev.parquet` and `prev_<zone>.parquet`.

**`faithful_bt.py` changes** (env-gated; the champion path is unchanged when they are unset):
- `FAITHFUL_VINTAGE_EXTRA=DIR` or `--vintage-extra DIR` adds those files to the zone and grid vintages. Local values
  win, so days that already have data stage identically. `FAITHFUL_VINTAGE_EXTRA_ONLY=1` (tests only) makes the extra
  files replace the local ones.
- With extra vintages, `--out` is required.
- `FAITHFUL_RL_LEAN=1`: `matched` skips the boosters it only stores (held-out load, control set, held-out matched load).
  What is fed is unchanged. This saves about 90 s per day, so matched fits 4 days per process.
- Tests: `stage_check.py`, on 2026-09-15 (a day with local data). Default staging is compared with staging where the
  extra files are built from the local data and replace it; the two must be identical. Matched lean is compared with
  `out_strict_rlmatched` on one day; the prices must be identical.

**Waves** (each its own single wave of ≤ 20 min on an empty machine, 4 processes × 4 days, 1 thread, same staging):

| wave | command per process (4 days each) | est. per day |
|---|---|---|
| champion cache | `FAITHFUL_VINTAGE_EXTRA=$SP/wf16_hist faithful_bt.py --days <4> --rl-variant base --out $SP/E/wf16/champion --torch-threads 1` | 240 s → ~16 min |
| matched RL | same + `--rl-variant matched`, `FAITHFUL_RL_LEAN=1`, `--out $SP/E/wf16/matched` | ~270 s → ~18 min (to be confirmed) |
| C nbpast Timestral | `FAITHFUL_VINTAGE_EXTRA=… team/C/dump_ts.py --days <4> --de-only --context 8192 --tag wf16 --also nbpast`; blend = (champion trees + nbpast + champion market)/3. Check: base Timestral = champion Timestral | ~205 s |
| P fclog trees (if it survives) | `team/P/fclog_member.py --days <4> --log-days 21 --rl-variant base --out $SP/E/wf16/fclog`; blend from the champion cache | ~265 s. Its log days before D have **no** strict vintages for the 12 pre-Sep days (backfill covers D-1..D+2 only); see note to P |

Scoring: `score_cache.py --champion $SP/E/wf16/champion --challenger <dir> --days team/E/sample_wf16.txt` (it reads the quarter column and adds a by-quarter table with the sign count), which
reports by month. Quarters are 3-month groups of those months.

**Promotion rule text for option B** (proposed for `DECISIONS.md`):
> A change is promoted if (1) the paired 90 % circular 7-day block-bootstrap interval of its blend MAE difference
> against the champion, over the pooled distinct days (61 used + 18 fresh + the 16 walk-forward days, 92 distinct),
> excludes 0 on the negative side; and (2) its mean difference is negative in at least 3 of the 4 quarters of the
> 16-day sample.

**Statistical note on (2).** It is a sign check, not a test. Daily blend differences of matched versus base over the
78 stored days have a standard deviation of 0.218 ct/kWh, so a 4-day quarter mean has a standard error of about 0.11.

| true effect (ct/kWh) | P(quarter mean < 0) | P(at least 3 of 4 quarters) |
|---|---|---|
| 0 (no effect) | 0.50 | **0.31** |
| -0.03 | 0.61 | 0.49 |
| -0.054 (matched's pooled estimate) | 0.69 | 0.63 |
| -0.10 | 0.82 | 0.85 |
| -0.165 (10 % of the blend) | 0.94 | 0.98 |

- Under no effect, the check passes 31 % of the time.
- A real effect the size of matched's fails it about a third of the time.
- It filters only effects of about -0.1 or larger.
- The evidence comes from the pooled interval: roughly ±0.04 at 90 % on 92 days if days were independent, wider with
  the block bootstrap.
- (2) is best read as a guard against a gain carried by one season, not as significance.
- Pooling the 61 used days, on which every candidate was developed, makes (1) optimistic. The 16 walk-forward days and
  the forward test are the only untouched evidence.

**Test results (2026-11-26 23:11-23:17 UTC, 1 process, other jobs running):**
- **`--vintage-extra`, stage-only, D = 2026-09-15:** in the second configuration the extra files *replace* the local
  ones (`FAITHFUL_VINTAGE_EXTRA_ONLY=1`). The staged weather for the 8 zones and the grid, the price tails and the
  generation tail are identical to the default staging (max |diff| 0.0, same NaN pattern, no fallback cells).
- **`FAITHFUL_RL_LEAN=1` matched, D = 2026-09-15, against `out_strict_rlmatched`:**
  - trees 1.0e-13, Timestral 0, market 1.1e-14, blend 3.9e-14, rl_trees 9.2e-14, rl_mixed 1.4e-14;
  - this is LightGBM thread noise, the same order as in README R1, so the variant is the same;
  - it took 244 s for the day (stage 1 105 s, stage 2 48 s, predict + blend 90 s), so 4 days per process fit in about 17 min.
