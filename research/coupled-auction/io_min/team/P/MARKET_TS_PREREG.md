# Market-route Timestral replay: written before the run (2026-11-26 ~23:00 UTC)

Script: `market_ts.py`. Inputs: C's dumps of the Timestral-mixed DE input frame
(`SP/C/tsdump/ctx8k`, `fresh8k`). One process, one thread, timeout 18 min. Days lost to the timeout are dropped.

**Questions, and nothing else in this run**

1. **Gas.** Does lagging gas by 1 day instead of 2 (`gas1`) help? That puts gas day D-1 on D, which is certainly
   gate-known, in both the shift history and the clearing.
   - Gas day D (which THE probably publishes before the gate) is not in the dumped frames. They are cut at the
     cutoff and ffilled, so it is not tested here.
2. **RL consistency.** In `consist`, the past days' RL in the shift history and in the covariate is the forecast-time
   RL the route read at each earlier gate (stored `rl_mixed`), instead of the published TSO figures.
   - This uses stored days from 09-08 on, and so covers about 75-80 of the ~85 context days for fresh days.

**Days**

| Variant | Days | Purpose |
|---|---|---|
| `repro` | 2026-09-20, 2026-11-10 | must match the stored faithful market route to within 0.01 ct/kWh MAE per day, else the run is void |
| `gas1` | 09-20, 09-30, 10-10, 10-20, 10-30, 11-10, 11-13, 11-16, 11-19, 11-22, 11-25 | |
| `consist` | 11-10, 11-13, 11-16, 11-19, 11-22, 11-25 | |

**Comparison**
- Each variant is paired per day against the stored champion market route, `out_strict` / `out_fresh_strict`, base
  RL.
- Reported: per-day differences, wins out of N, and the mean. With 11 and 6 days no interval is meaningful, so a
  change counts only as a lead for E's walk-forward sample, never as a promotion.
- Also reported: the blend with the variant replacing the market route.
