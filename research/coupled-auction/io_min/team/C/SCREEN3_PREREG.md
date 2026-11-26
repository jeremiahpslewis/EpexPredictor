# Pre-registered screen 3: next Timestral increments (written 2026-11-26 ~23:20 UTC, before any run; design only)

Run tomorrow 2026-11-27 after 10:30 UTC, ONE wave, 4 procs x 1 thread x 3 days, 20-min cap per process; an unfinished day is
dropped and never topped up. Selection rule fixed here; no comparison on the 12 screen days or the 16 (+5 listed) confirmation days.

## Day list (third list, disjoint from every earlier list; deterministic)
Candidates: 09-08..11-26 minus the 12 screen days, the 21 days ever listed for confirmation, 10-25 (clock change) and 11-08.
61-window: 40 candidates, 9 taken evenly (positions round(i*(n-1)/8), i=0..8). Fresh: 5 candidates, 3 taken evenly.
- proc 1: 2026-09-08,2026-09-16,2026-09-24
- proc 2: 2026-10-01,2026-10-07,2026-10-14
- proc 3: 2026-10-22,2026-10-31,2026-11-07
- proc 4: 2026-11-11,2026-11-17,2026-11-24
All procs with FAITHFUL_FRESH=1 (fresh mode reproduced 61-window days exactly: 10-15, 10-19, 10-24 in C014).

Swap (2026-11-26, before any run): 2026-10-08 is in E's frozen walk-forward list (team/E/sample_wf16.txt) and is
replaced by the nearest 61-window day not in any screen list, confirmation list, this list or sample_wf16 (tie broken
toward the earlier day): 2026-10-07 (10-09 equally near, later). sample_wf16 also holds 10-16 and 11-02: neither is in
this list. gasD runs only if E's 11-27 input log again shows THE gas-day-D present before the gate ("observed twice",
not proven for summer).

## What runs per day
`dump_ts.py --rl-variant matched --de-only --context 8192 --also nbpast,gasD,nbpast_gasD`, tag `screen3`.
- The dump runs under the harness's `matched` RL variant, so Timestral's RL covariate is matched over the WHOLE horizon
  (history = published TSO figures either way). Validity: the dump's own forecast (fc_D) must equal the stored
  `out_*strict_rlmatched` Timestral exactly; a day that fails is invalid (reported, not replaced).
- Variants replayed on the loaded model (cost ~150 s dump + ~70+50+70 s replays ≈ 340 s/day -> 3 days ≈ 17 min + setup):
  - `rlmH`   = the dump's own forecast (matched RL over the horizon, base otherwise) -- free
  - `nbpast` (on matched RL) = rlmH + 7 neighbours' price history as past covariates (the frozen spec)
  - `gasD`   = rlmH with the gas covariate over the horizon set to THE's value for gas day D (store value stamped D 01:00
    local) instead of the D-1 value forward-filled; history unchanged
  - `nbpast_gasD`
- Reference: the champion's Timestral (`out_*strict`, base RL), stored -- no compute.

## Precondition (gate legality of gasD)
gasD runs only if E/I confirm, from the source, that THE's gas-day-D price is published before the 11:00 UTC (10:00 UTC
summer) gate on D-1 for every day in the list. Otherwise gasD and nbpast_gasD are not run (3 replays -> 1; days per proc
stay 3) and that is logged.

## Selection and reporting (fixed now)
- Report each variant vs champion Timestral and in the blend (variant replacing Timestral; trees and market from the
  champion), 61 / fresh parts, per-month signs, 90% 7-day block bootstrap (12 days: indicative only).
- The next spec = the variant with the lowest mean Timestral MAE over the 12 days, provided it beats `nbpast`-on-matched
  on both the 61 and fresh parts; otherwise the spec stays nbpast (+ matched RL when matched RL is promoted).
- Confirmation of any new spec is NOT on these days; it goes to E's option-B walk-forward sample.
- Every variant is a TRIALS row (screen 3), kept or dropped.

## Code needed before the run (done 11-26, no run)
- `dump_ts.py --rl-variant {base,matched}` (sets the harness RLV variant; default base).
- `ts_variants.py`: variants `gasD`, `nbpast_gasD` (gas row over the horizon from the THE store value of gas day D).
