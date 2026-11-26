"""Paired wind-MAE comparison of wind_lab variants against V0 (same cadence, same days). usage: wind_score.py SP CADENCE WHICH V1,V2,..."""
import sys
import numpy as np, pandas as pd

SP, CAD, WHICH, VS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
D = f"{SP}/team/I/windlab"
rng = np.random.default_rng(0)


def ci(v, blk=7, n=3000):
    v = np.asarray(v); n0 = len(v); st = []
    for _ in range(n):
        s = rng.integers(0, n0, int(np.ceil(n0 / blk)))
        st.append(v[np.concatenate([(np.arange(k, k + blk) % n0) for k in s])[:n0]].mean())
    return np.percentile(st, [5, 95])


def daily(v):
    P = pd.read_parquet(f"{D}/pred_{v}_{CAD}_{WHICH}.parquet")
    return P.assign(ae=(P.pred - P.true).abs()).groupby(["set", "day"]).ae.mean()


b = daily("V0")
for st in ("61", "fresh18"):
    if st not in b.index.get_level_values(0):
        continue
    print(f"[{st}] V0 wind MAE {b[st].mean():.3f} GW ({len(b[st])} days)")
    for v in VS:
        try:
            x = daily(v)[st]
        except FileNotFoundError:
            continue
        d = (x - b[st]).sort_index(); lo, hi = ci(d.to_numpy())
        print(f"   {v}: {x.mean():.3f}  diff {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  {100 * d.mean() / b[st].mean():+.1f}%  days better {(d < 0).sum()}/{len(d)}")
