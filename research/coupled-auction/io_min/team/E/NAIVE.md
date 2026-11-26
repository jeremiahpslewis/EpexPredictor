# Naive baseline on the stored faithful days (champion = out_strict / out_fresh_strict, base)

### 61 used days (61 days, 2026-09-08..2026-11-07)

| forecast | MAE | rMAE vs naive | forecast - naive, 90% interval |
|---|---|---|---|
| naive (same QH, D-7) | 7.528 | 1 | |
| trees | 1.787 | 0.237 | -5.742 [-7.267, -4.412] |
| Timestral | 1.732 | 0.230 | -5.796 [-7.267, -4.538] |
| market route | 1.873 | 0.249 | -5.655 [-7.223, -4.319] |
| blend | 1.648 | 0.219 | -5.880 [-7.370, -4.576] |

Per-month naive MAE: 2026-09: 7.099, 2026-10: 7.651, 2026-11: 8.395

### 18 fresh days (18 days, 2026-11-09..2026-11-26)

| forecast | MAE | rMAE vs naive | forecast - naive, 90% interval |
|---|---|---|---|
| naive (same QH, D-7) | 8.276 | 1 | |
| trees | 2.205 | 0.266 | -6.071 [-8.174, -4.113] |
| Timestral | 2.316 | 0.280 | -5.960 [-8.054, -4.099] |
| market route | 2.362 | 0.285 | -5.914 [-8.013, -3.999] |
| blend | 2.173 | 0.263 | -6.104 [-8.152, -4.184] |

Per-month naive MAE: 2026-11: 8.276

