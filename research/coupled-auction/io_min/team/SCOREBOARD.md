# Scoreboard (S keeps it). Last update: 2026-11-26 23:50 UTC

MAE in ct/kWh over 96 quarter hours, mean of daily MAE. Champion = faithful rerun at 37978d1, `--rl-variant base`,
strict vintage. Differences are candidate minus champion on the same days; intervals are paired circular 7-day block
bootstrap, two-sided 90%. "61" = local 2026-09-08..11-07 (post hoc: this window was already used for design, so a win
here is evidence, not proof). "fresh18" = 2026-11-09..11-26. The walk-forward sample (WF) does not exist yet.
Naive (same quarter hour one week earlier, `E/NAIVE.md`): 7.528 on the 61 days, 8.276 on fresh18. MAE relative to naive:
blend 0.219 (61) and 0.263 (fresh18); trees 0.237, Timestral 0.230, market route 0.249 (61).

## Targets and current best

| | Champion 61 | Target 61 | Best candidate 61 | Champion fresh18 | Best candidate fresh18 | WF |
|---|---|---|---|---|---|---|
| Trees | 1.787 | ≤ 1.61 | 1.693 (`matched` RL) -0.093 [-0.176,-0.010] | 2.205 | 2.217 (+0.012 [-0.102,+0.105]) | – |
| Timestral | 1.732 | ≤ 1.56 | 1.707 (`matched`) -0.025 [-0.047,-0.002] | 2.316 | 2.273 (-0.043 [-0.078,-0.014]) | – |
| Market route | 1.873 | ≤ 1.69 | 1.825 (`matched`) -0.048 [-0.072,-0.022] | 2.362 | 2.316 (-0.046 [-0.091,-0.003]) | – |
| Blend | 1.648 | ≤ 1.32 | 1.587 (`matched`) -0.061 [-0.102,-0.018] | 2.173 | 2.142 (-0.031 [-0.089,+0.017]) | – |
| Residual load, tree series (GW) | 2.56 | ≤ 2.30 | 2.29 (`matched`) -0.273 [-0.471,-0.086] | 2.45 | 2.15 | – |

Progress to target (61 days): trees 53% of the way, Timestral 15%, market route 26%, blend 19%, residual load 100%
(to be confirmed on WF). Pooled 79 days, blend: -0.054 [-0.091,-0.015].

## Candidates in screening (not on the board until confirmed)

| candidate | owner | evidence so far | status |
|---|---|---|---|
| `matched` RL | I / E | 61 days: blend -0.061 [-0.102,-0.018]; pooled 79: -0.054 [-0.091,-0.015] | candidate; live shadow |
| Timestral + neighbours' price history (nbpast) | C | 12 screen days: Timestral -0.055 (61 half), -0.107 (fresh half). **16 confirmation days (C014): Timestral 2.145 → 2.078, -0.067 [-0.148,+0.013], 9 of 16 days; blend -0.020 [-0.046,+0.005]; Sep -0.093, Oct +0.021, Nov -0.146** | promising, not confirmed; goes to the 16-day walk-forward wave |
| Trees trained on forecast-time rows (fclog) | P | fresh18: +0.162 [+0.010,+0.314]; 61 days: +0.099 [+0.005,+0.204] | dropped |
| Market route with consistent RL history (consist) | P | 6 fresh screen days: market route 2.457 → 2.087 (-0.370, 5 of 6), blend -0.104 | confirmation run on the other 12 fresh days |
| Wind boosters on existing grid data | I | 7 variants, best -5% / -3% wind MAE at weekly refits; ranking not stable at daily refits | closed |
| Tree shift target (shiftl1) | P | fresh18 weekly screen -0.255; 61 days +0.074; daily fixed days +0.332 | dropped |
| Published TSO load on D | I | 4 days: blend +0.005 | parked |
| True RL (bound, not a forecast) | I | 9 days: blend -0.225 [-0.468,+0.068]; beyond `matched` -0.069 | bound |
| Weights, bias correction, stacking, similar-day member | S / C | about 150 variants, none better causally | closed |

## Champion status

- Champion: unchanged (frozen base). No promotion yet: `matched` has a 90% interval excluding 0 on the 61 days and
  pooled, is negative in all 3 available months (09: -0.027, 10: -0.085, 11: -0.063 on the 61; fresh 11: -0.031), but
  the rule needs the walk-forward months. Status: **candidate, live shadow**.

- **Caveat on `matched`.** Its gain, and the residual-load target it meets, are measured on strict-vintage weather
  (2-day-old on 74% of slots). Live weather is younger, and `matched` on live weather has no backtest. A
  strict-vintage walk-forward run would not settle that; the live residual-load log does. The Timestral candidate
  (neighbours' prices) does not depend on weather age.

## By window, blend (champion → `matched`)

| window (local) | 61: champion | 61: diff | fresh18: champion | fresh18: diff |
|---|---|---|---|---|
| 00-06 | 0.98 | +0.001 | 1.08 | -0.001 |
| 06-10 | 1.67 | -0.031 | 2.53 | -0.007 |
| 10-16 | 1.87 | -0.050 | 2.86 | -0.039 |
| 16-21 | 2.40 | -0.148 | 2.67 | +0.068 |
| 21-24 | 1.25 | -0.100 | 1.67 | -0.275 |

## Live forward test (from D = 2026-11-28)

| D | app | frozen champion | team forecast (variant) | shadows | issued UTC | hashes |
|---|---|---|---|---|---|---|
| 2026-11-28 | pending | pending | = frozen champion config on live inputs (see S/DECISIONS.md) | `matched` | | |
