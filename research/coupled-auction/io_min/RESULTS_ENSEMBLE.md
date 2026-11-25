# Ensemble backtest: results (protocol `prereg/ENSEMBLE_PROTOCOL.md`)

The window had already been used, so these results are descriptive. The deployed techs are research implementations
of the app's methods, from `research/market-model`. These are the best stored arrays, not the best possible data:
- **Trees.** Zone weather is the 2–3-day-old previous-runs vintage in every stored run. `opt` and `pes` only choose
  the residual-load input, built from 1-day-old or 2–3-day-old weather respectively.
- **Timestral and market route.** Deployed lead-weighted form. Their inputs include a component on 2-day-old weather,
  and the D−1 context uses archive weather.

A faithful version on the freshest data needs 1-day-old previous runs for all zones (not stored; Open-Meteo rate
limits were hit before) and the app code driven per day. That is a separate request.

**Headline: with information sets equalised, the eleven-zone model does not improve the deployed mix.**
- **Strict member.** The strictly pre-gate eleven-zone member E_s (the deployed residual-load forecast for Germany,
  yesterday's wind and solar for the neighbours, heating degree days from D−1) has MAE 2.87 ct/kWh, against 2.25 for
  E with post-gate inputs.
- **In the mix.** Adding E_s to the deployed mix makes it worse: +0.116 [+0.058, +0.192] over the whole day, +0.146
  at night, and +0.035 [−0.033, +0.147] at the evening peak.
- **Time-of-day weights** (post hoc, leave-one-week-out) give E_s weight 0–0.1 and do not beat equal thirds (+0.033).
- **What drove the earlier gain.** The evening-peak gain of the first scoring came from the ENTSO-E wind and solar
  forecasts published after the gate, not from the market structure.
- **The lever that does show up is residual-load forecast quality.** It moves the eleven-zone member by 0.62 ct/kWh
  (E against E_s) and the trees by 0.25 (residual-load input from 1-day-old against 2–3-day-old weather).

**Expectations:**
- **X1 met.** Rebuilt as research note 21.121 did (trees `pes`, all 66 autumn days), the deployed mix scores
  1.740 ct/kWh, the same as the note. The reconstruction is faithful.
- **X2 met.** The eleven-zone model alone (2.245) is worse than the deployed mix (1.642) and worse than each deployed
  member.
- **X3 (first scoring, E with post-gate inputs).** Adding it as a fourth equal member (M4) lowers the evening-peak MAE by 0.148 ct/kWh [0.033, 0.229], about
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


Local delivery days 2026-09-08 to 2026-11-07; 61 days scored, 96 quarter-hours each; MAE in ct/kWh. Eleven-zone quarter-hours with a missed solve (53) score the training-mean fallback.

|                                        |   whole day |   night 0-6 |   evening 16-21 |
|:---------------------------------------|------------:|------------:|----------------:|
| T trees (opt)                          |       1.818 |       1.227 |           2.707 |
| T trees (pes)                          |       2.08  |       1.355 |           3.088 |
| P Timestral                            |       1.747 |       1.139 |           2.492 |
| A market route                         |       1.891 |       1.155 |           2.666 |
| E eleven-zone (b)                      |       2.245 |       1.696 |           2.952 |
| M3 (T+P+A)/3                           |       1.642 |       1.05  |           2.426 |
| M4 (T+P+A+E)/4                         |       1.597 |       1.067 |           2.278 |
| M3E (T+P+E)/3                          |       1.614 |       1.12  |           2.267 |
| M4 renormalised (post hoc)             |       1.586 |       1.058 |           2.252 |
| E_s strict eleven-zone                 |       2.867 |       2.083 |           3.57  |
| M4s (T+P+A+E_s)/4                      |       1.758 |       1.196 |           2.461 |
| M3Es (T+P+E_s)/3                       |       1.817 |       1.258 |           2.498 |
| M4s renormalised (post hoc)            |       1.739 |       1.187 |           2.447 |
| ToD weights, leave-week-out (post hoc) |       1.675 |       1.104 |           2.466 |

## Full table: Paired differences in MAE (first minus second; negative = first better), 90% block bootstrap (7-day blocks)

- E_s strict eleven-zone − E eleven-zone (b), whole day: +0.622 [+0.445, +0.888]
- E_s strict eleven-zone − E eleven-zone (b), night 0-6: +0.388 [+0.251, +0.514]
- E_s strict eleven-zone − E eleven-zone (b), evening 16-21: +0.618 [+0.366, +0.869]
- M4s (T+P+A+E_s)/4 − M3 (T+P+A)/3, whole day: +0.116 [+0.058, +0.192]
- M4s (T+P+A+E_s)/4 − M3 (T+P+A)/3, night 0-6: +0.146 [+0.053, +0.249]
- M4s (T+P+A+E_s)/4 − M3 (T+P+A)/3, evening 16-21: +0.035 [-0.033, +0.147]
- M3Es (T+P+E_s)/3 − M3 (T+P+A)/3, whole day: +0.175 [+0.110, +0.276]
- M3Es (T+P+E_s)/3 − M3 (T+P+A)/3, night 0-6: +0.208 [+0.069, +0.369]
- M3Es (T+P+E_s)/3 − M3 (T+P+A)/3, evening 16-21: +0.072 [-0.021, +0.238]
- M4s renormalised (post hoc) − M3 (T+P+A)/3, whole day: +0.096 [+0.048, +0.165]
- M4s renormalised (post hoc) − M3 (T+P+A)/3, night 0-6: +0.137 [+0.043, +0.239]
- M4s renormalised (post hoc) − M3 (T+P+A)/3, evening 16-21: +0.020 [-0.041, +0.129]
- M4 renormalised (post hoc) − M3 (T+P+A)/3, whole day: -0.056 [-0.116, +0.010]
- M4 renormalised (post hoc) − M3 (T+P+A)/3, night 0-6: +0.008 [-0.084, +0.104]
- M4 renormalised (post hoc) − M3 (T+P+A)/3, evening 16-21: -0.175 [-0.254, -0.068]
- M4 (T+P+A+E)/4 − M3 (T+P+A)/3, whole day: -0.046 [-0.107, +0.026]
- M4 (T+P+A+E)/4 − M3 (T+P+A)/3, night 0-6: +0.017 [-0.072, +0.109]
- M4 (T+P+A+E)/4 − M3 (T+P+A)/3, evening 16-21: -0.148 [-0.223, -0.030]
- M3E (T+P+E)/3 − M3 (T+P+A)/3, whole day: -0.029 [-0.101, +0.073]
- M3E (T+P+E)/3 − M3 (T+P+A)/3, night 0-6: +0.070 [-0.066, +0.207]
- M3E (T+P+E)/3 − M3 (T+P+A)/3, evening 16-21: -0.159 [-0.254, +0.008]
- E eleven-zone (b) − M3 (T+P+A)/3, whole day: +0.603 [+0.366, +0.933]
- E eleven-zone (b) − M3 (T+P+A)/3, night 0-6: +0.646 [+0.359, +0.923]
- E eleven-zone (b) − M3 (T+P+A)/3, evening 16-21: +0.526 [+0.337, +1.053]
- E eleven-zone (b) − A market route, whole day: +0.354 [+0.223, +0.653]
- E eleven-zone (b) − A market route, night 0-6: +0.541 [+0.255, +0.863]
- E eleven-zone (b) − A market route, evening 16-21: +0.286 [+0.162, +0.869]
- E eleven-zone (b) − T trees (opt), whole day: +0.427 [+0.149, +0.791]
- E eleven-zone (b) − T trees (opt), night 0-6: +0.469 [+0.156, +0.736]
- E eleven-zone (b) − T trees (opt), evening 16-21: +0.245 [+0.022, +0.813]
- E eleven-zone (b) − P Timestral, whole day: +0.498 [+0.267, +0.807]
- E eleven-zone (b) − P Timestral, night 0-6: +0.557 [+0.289, +0.835]
- E eleven-zone (b) − P Timestral, evening 16-21: +0.460 [+0.263, +0.854]
- T trees (opt) − T trees (pes), whole day: -0.262 [-0.376, -0.186]
- T trees (opt) − T trees (pes), night 0-6: -0.128 [-0.336, +0.104]
- T trees (opt) − T trees (pes), evening 16-21: -0.380 [-0.577, -0.235]

## Full table: Post hoc: time-of-day weights

Weights on (T, P, A, E_s) fitted on all days, for reference (each week's forecast used weights fitted without that week):

- 00-06 local: T 0.4, P 0.3, A 0.3, E_s 0.0
- 06-10 local: T 0.4, P 0.3, A 0.2, E_s 0.1
- 10-16 local: T 0.5, P 0.5, A 0.0, E_s 0.0
- 16-21 local: T 0.2, P 0.5, A 0.2, E_s 0.1
- 21-24 local: T 0.0, P 0.4, A 0.5, E_s 0.1
- ToD weights − M3, whole day: +0.033 [-0.001, +0.074]
- ToD weights − M3, night 0-6: +0.055 [+0.038, +0.079]
- ToD weights − M3, evening 16-21: +0.040 [-0.019, +0.139]