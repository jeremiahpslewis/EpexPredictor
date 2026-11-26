# Agent team plan: DE day-ahead accuracy (2026-11-26)

## Goal and benchmarks

**Champion (frozen).** The faithful rerun of the deployed ensemble at commit 37978d1, `--rl-variant base`, strict
vintage: trees, Timestral, market route, equal-thirds blend. Every claim is a paired comparison with this champion on
the same days, as MAE in ct/kWh over 96 quarter hours. MAE relative to a naive forecast (same quarter hour one week
earlier) is reported as well, so months with different price levels compare fairly.

**Targets.**
- Each price member at least 10% better than its champion version.
- Residual load (the input all members share) at least 10% better.
- The blend at least 20% better.

| | Champion, 61 days (09-08..11-07) | Target | Champion, 18 fresh days (11-09..11-26) |
|---|---|---|---|
| Trees | 1.787 | ≤ 1.61 | 2.205 |
| Timestral | 1.732 | ≤ 1.56 | 2.316 |
| Market route | 1.873 | ≤ 1.69 | 2.362 |
| Blend | 1.648 | ≤ 1.32 | 2.173 |
| Residual load (GW) | 2.56 | ≤ 2.30 | 2.45 |

The long walk-forward window (below) replaces the 61-day figures as the reference once its champion run exists. The
targets then become 10% and 20% relative to those numbers.

## Evaluation without a sealed window

1. **Long walk-forward window.**
   - Every day from 2025-12-01 to the present (about 360 origins) is forecast with information up to its gate only.
   - That makes every forecast out of sample by construction.
   - Members train weekly for screening and daily for the final check.
2. **Promotion rule.**
   - A change is promoted only if it beats the champion in at least 9 of the 12 months.
   - Its paired 7-day block-bootstrap interval over the whole window must exclude 0.
   - Every trial is logged (`TRIALS.md`: what was tried, score, kept or dropped), so the number of comparisons is
     visible.
3. **Fast loop.**
   - Members write per-day npz files in the existing format to a shared cache.
   - Combination and post-processing experiments run on the cached arrays in seconds.
   - Only finalists go through the full faithful harness.
4. **Size the prize first.**
   - For each lever, compute a hindsight bound before building it:
     - the ensemble with best fixed weights, weights by time of day, and a per-day oracle;
     - price models fed the true residual load;
     - price models fed archive weather.
   - Staff the levers with the most room.
5. **Forward test.**
   - Every day from 2026-11-27, the team issues two forecasts for D+1 before the 11:00 UTC gate: the deployed app's and
     the current champion's.
   - The forecast file goes to the user before the gate, and its SHA-256 hash is committed. No data files go in the repo.
   - Every input is logged with its arrival time (weather run used, published TSO load, prices, gas). This makes the
     weather-age and publication-timing questions measured facts.
   - One day is noisy (daily blend MAE ranged from about 0.4 to 3.2 ct/kWh), so a rolling live scoreboard is kept.

## Team (5 agents)

| Role | Model | Owns | First deliverable |
|---|---|---|---|
| **S. Supervisor** | Fable | Plan, scoreboard, `TRIALS.md`, compute queue, promotions, leak review; replaces the per-run advisor consult | Breakdown of error by time of day and regime on the 79 stored days; lever ranking |
| **E. Evaluation and live operations** | Opus | Fast walk-forward harness, shared cache, naive baseline, daily live forecast and input log. Builds no models (neutral scorer) | The 11-28 live forecast before the 11-27 gate; then the champion over the long window |
| **I. Inputs** | Opus | Residual load (matched-age, published TSO load, more grid points), cross-border drivers (neighbours' residual load, French nuclear, gas, CO2) | Price models fed the true residual load (bound); published load at the gate in the live pipeline |
| **P. Price members** | Opus | Trees and market route: targets (spread to a fuel-scaled anchor), features, regimes (negative prices, evening ramp), refit cadence | Error breakdown of trees and market route; first targeted change |
| **C. Timestral and combination** | Opus | Timestral context and covariates, post-processing; the blend: learned weights by time of day and regime, stacking, bias correction | Hindsight bounds on stored arrays; walk-forward learned weights |

**Working rules.**
- Each agent writes only in its own directory (`team/<role>/`) and emits members in the shared npz contract. Only E
  edits `faithful_bt.py`.
- `predictor/` is not modified. Promotion to the deployed app is a separate decision for the user.
- The ENTSO-E key stays in `$SP/entsoe/key`. No EPEX licensed data, no data files in the repo. Branch
  `claude/eager-fermat-vyfgf0`.
- Compute: 4 cores. E schedules runs (at most 4 single-thread processes, each under 2 h). Compute runs within the user's
  standing budget; anything beyond it goes to the user.
- Check-ins: S posts a scoreboard to the user each day after scoring the live forecast.
