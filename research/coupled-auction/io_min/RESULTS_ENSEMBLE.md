# Ensemble backtest: results (protocol `prereg/ENSEMBLE_PROTOCOL.md`)

The window had already been used, so these results are descriptive. The deployed techs are research implementations
of the app's methods, from `research/market-model`. These are the best stored arrays, not the best possible data:
- **Trees.** Zone weather is the 2–3-day-old previous-runs vintage in every stored run. `opt` and `pes` only choose
  the residual-load input, built from 1-day-old or 2–3-day-old weather respectively.
- **Timestral and market route.** Deployed lead-weighted form. Their inputs include a component on 2-day-old weather,
  and the D−1 context uses archive weather.

A faithful version on the freshest data needs 1-day-old previous runs for all zones (not stored; Open-Meteo rate
limits were hit before) and the app code driven per day. That is a separate request.

**Expectations:**
- **X1 met.** Rebuilt as research note 21.121 did (trees `pes`, all 66 autumn days), the deployed mix scores
  1.740 ct/kWh, the same as the note. The reconstruction is faithful.
- **X2 met.** The eleven-zone model alone (2.245) is worse than the deployed mix (1.642) and worse than each deployed
  member.
- **X3.** Adding it as a fourth equal member (M4) lowers the evening-peak MAE by 0.148 ct/kWh [0.033, 0.229], about
  6%. The whole-day gain (−0.046 [−0.105, +0.024]) is not established, and nights are unchanged.

**Caveats:**
- **Information advantage.** The eleven-zone forecasts use ENTSO-E wind and solar forecasts published after the gate,
  realised heating degree days, and the published Core domain. Part of the evening gain may come from that. A strictly
  pre-gate eleven-zone run is needed to settle it.
- **Missed solves.** 53 eleven-zone quarter-hours had no solution and score the training-mean fallback. A first
  scoring that dropped those days (51 days) overstated the gain (whole day −0.077), so it is not the result.
- **Residual-load input.** Trees with the residual-load input built from 1-day-old weather (`opt`) beat those built
  from 2–3-day-old weather (`pes`) by 0.25 ct/kWh over the whole day.
- **Alignment.** Stored actual prices equal ENTSO-E prices at slot offset 0 on every checked day.

Local delivery days 2026-09-08 to 2026-11-07; 61 days scored, 96 quarter-hours each; MAE in ct/kWh. Eleven-zone quarter-hours with a missed solve (53) score the training-mean fallback.

|                   |   whole day |   night 0-6 |   evening 16-21 |
|:------------------|------------:|------------:|----------------:|
| T trees (opt)     |       1.818 |       1.227 |           2.707 |
| T trees (pes)     |       2.08  |       1.355 |           3.088 |
| P Timestral       |       1.747 |       1.139 |           2.492 |
| A market route    |       1.891 |       1.155 |           2.666 |
| E eleven-zone (b) |       2.245 |       1.696 |           2.952 |
| M3 (T+P+A)/3      |       1.642 |       1.05  |           2.426 |
| M4 (T+P+A+E)/4    |       1.597 |       1.067 |           2.278 |
| M3E (T+P+E)/3     |       1.614 |       1.12  |           2.267 |

## Full table: Paired differences in MAE (first minus second; negative = first better), 90% block bootstrap (7-day blocks)

- M4 (T+P+A+E)/4 − M3 (T+P+A)/3, whole day: -0.046 [-0.105, +0.024]
- M4 (T+P+A+E)/4 − M3 (T+P+A)/3, night 0-6: +0.017 [-0.073, +0.111]
- M4 (T+P+A+E)/4 − M3 (T+P+A)/3, evening 16-21: -0.148 [-0.229, -0.033]
- M3E (T+P+E)/3 − M3 (T+P+A)/3, whole day: -0.029 [-0.101, +0.076]
- M3E (T+P+E)/3 − M3 (T+P+A)/3, night 0-6: +0.070 [-0.063, +0.212]
- M3E (T+P+E)/3 − M3 (T+P+A)/3, evening 16-21: -0.159 [-0.252, +0.005]
- E eleven-zone (b) − M3 (T+P+A)/3, whole day: +0.603 [+0.363, +0.938]
- E eleven-zone (b) − M3 (T+P+A)/3, night 0-6: +0.646 [+0.360, +0.929]
- E eleven-zone (b) − M3 (T+P+A)/3, evening 16-21: +0.526 [+0.323, +1.043]
- E eleven-zone (b) − A market route, whole day: +0.354 [+0.221, +0.651]
- E eleven-zone (b) − A market route, night 0-6: +0.541 [+0.254, +0.873]
- E eleven-zone (b) − A market route, evening 16-21: +0.286 [+0.167, +0.873]
- E eleven-zone (b) − T trees (opt), whole day: +0.427 [+0.141, +0.793]
- E eleven-zone (b) − T trees (opt), night 0-6: +0.469 [+0.159, +0.744]
- E eleven-zone (b) − T trees (opt), evening 16-21: +0.245 [+0.016, +0.808]
- E eleven-zone (b) − P Timestral, whole day: +0.498 [+0.274, +0.812]
- E eleven-zone (b) − P Timestral, night 0-6: +0.557 [+0.305, +0.839]
- E eleven-zone (b) − P Timestral, evening 16-21: +0.460 [+0.288, +0.854]
- T trees (opt) − T trees (pes), whole day: -0.262 [-0.376, -0.186]
- T trees (opt) − T trees (pes), night 0-6: -0.128 [-0.343, +0.110]
- T trees (opt) − T trees (pes), evening 16-21: -0.380 [-0.577, -0.242]