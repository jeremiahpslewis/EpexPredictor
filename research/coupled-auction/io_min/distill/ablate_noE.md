# Ablation: the fuel-scaled learner without the structural forecast (post hoc)

Compare with R1s 2.39 and R2s 2.29 in distill_results.md (same learner, rows, split and features plus pE, dpE).

|                                    |   MAE |   nan filled | vs E_s                  | vs T                    |
|:-----------------------------------|------:|-------------:|:------------------------|:------------------------|
| E_s                                | 2.867 |              |                         |                         |
| T                                  | 1.818 |              |                         |                         |
| frozen, no E_s, target level       | 2.287 |            0 | -0.580 [-0.794, -0.378] | +0.468 [+0.252, +0.692] |
| expanding, no E_s, target level    | 2.065 |            0 | -0.802 [-1.038, -0.573] | +0.247 [+0.054, +0.443] |
| frozen, no E_s, target res-plag    | 2.275 |            0 | -0.593 [-0.785, -0.422] | +0.456 [+0.227, +0.695] |
| expanding, no E_s, target res-plag | 2.139 |            0 | -0.729 [-0.945, -0.518] | +0.320 [+0.113, +0.530] |
| persistence p_lag (slope 0)        | 4.554 |            0 | +1.687 [+1.079, +2.277] | +2.736 [+1.952, +3.504] |