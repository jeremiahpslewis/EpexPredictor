"""Does gate-known member disagreement predict the blend's day-window error? (diagnostic correlations, all 79 days)"""
import numpy as np
from panel import champion
P = champion(); X, a, qh = P["X"], P["actual"], P["qh"]; eq = X.mean(-1); e = a - eq   # e>0: blend too low
win = np.digitize(qh, [24, 40, 64, 80])   # 0-6,6-10,10-16,16-20,20-24 local
S = {"61": np.arange(61), "fresh18": np.arange(62, 80)}
for nm, idx in S.items():
    rows = []
    for w in range(5):
        m = (win == w)
        f = lambda Z: np.array([np.nanmean(Z[d][m[d]]) for d in idx])
        E = f(e); tr_ts = f(X[..., 0] - X[..., 1]); tr_mk = f(X[..., 0] - X[..., 2]); ts_mk = f(X[..., 1] - X[..., 2]); sp = f(X.max(-1) - X.min(-1)); lv = f(eq)
        rl = f(P["rl_mixed"]); rows.append([w] + [np.corrcoef(E, z)[0, 1] for z in (tr_ts, tr_mk, ts_mk, sp, lv, rl)] + [np.corrcoef(np.abs(E), sp)[0, 1]])
    print(nm, "corr(window error a-blend, feature): win | trees-ts | trees-mkt | ts-mkt | spread | level | RL | corr(|err|,spread)")
    for r in rows: print("  ", r[0], " ".join(f"{v:+.2f}" for v in r[1:]))
