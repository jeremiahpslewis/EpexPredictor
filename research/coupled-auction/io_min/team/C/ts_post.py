"""Timestral alone: level vs shape anatomy and walk-forward ToD bias correction (stored arrays)."""
import numpy as np
from panel import champion, day_mae
P = champion(); X, a, qh, wd = P["X"], P["actual"], P["qh"], P["wd"]; T = X[..., 1]; e = T - a
S61, S18 = np.arange(61), np.arange(62, 80)
de = np.nanmean(e, 1); print("Timestral MAE 61/18: %.3f %.3f; bias %.3f %.3f" % (day_mae(T, a)[S61].mean(), day_mae(T, a)[S18].mean(), de[S61].mean(), de[S18].mean()))
print("  after removing own daily median error (oracle): %.3f %.3f" % tuple(np.nanmean(np.abs(e - np.nanmedian(e, 1, keepdims=True)), 1)[s].mean() for s in (S61, S18)))
for nm, s in (("61", S61), ("18", S18)):
    print(f"  {nm} mean error by 4h block:", [round(float(np.nanmean(e[s][(qh[s] // 16) == b])), 2) for b in range(6)], " by weekday:", [round(float(np.nanmean(de[s][wd[s, 0] == k])), 2) for k in range(7)])
    print(f"  {nm} MAE by weekday:", [round(float(day_mae(T, a)[s][wd[s, 0] == k].mean()), 2) for k in range(7)])
def wf_bias(nb, k, win=None, burn=7):
    F = T.copy(); blk = np.floor(qh / (96 / nb))
    for d in range(burn, len(a)):
        tr = np.arange(0 if win is None else max(0, d - win), d); n = len(tr); s = n / (n + k)
        for b in range(nb):
            m = blk[tr] == b; F[d][blk[d] == b] -= s * np.nanmedian(e[tr][m])
    return F
for nb in (1, 6, 24):
    for k in (0, 14, 28):
        for win in (None, 21):
            F = wf_bias(nb, k, win); B = (X[..., 0] + F + X[..., 2]) / 3
            print(f"wf bias nb={nb:2d} k={k:2d} win={win}: Timestral {day_mae(F,a)[S61].mean():.3f}/{day_mae(F,a)[S18].mean():.3f}  blend {day_mae(B,a)[S61].mean():.3f}/{day_mae(B,a)[S18].mean():.3f}")
