# Scoreboard (S keeps it). Last update: 2026-11-26 16:00 UTC

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

## Champion status

- Champion: unchanged (frozen base). No promotion yet: `matched` has a 90% interval excluding 0 on the 61 days and
  pooled, is negative in all 3 available months (09: -0.027, 10: -0.085, 11: -0.063 on the 61; fresh 11: -0.031), but
  the rule needs the walk-forward months. Status: **candidate, live shadow**.

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
