"""Stability addendum 1 (EXPECTATIONS.md addendum 1). Uses SP/distill/stability_daily.parquet from stability.py.
(a) fuel break re-dated to the first day after June with a daily fuel index above the training-range top (152);
(b) null for the hybrid-coupling jump: the same 14-day-after / 28-day-before ratio of relative MAE at every date;
(c) a tree frozen at 2026-06-10 (trained once, 540-day window) against the daily-retrained tree over days 1-42 after
    go-live. Appends to stability.md."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import trees_slope as TS
D = pd.read_parquet(f"{SP}/distill/stability_daily.parquet")
Pn = pd.read_parquet(TS.PANEL); Lc = 1.85 * Pn.Pg + 0.37 * Pn.tau; Ld = Lc.groupby(Lc.index.tz_convert("Europe/Berlin").normalize().tz_localize(None)).mean()
brk = Ld[(Ld > 152) & (Ld.index >= "2026-06-01")].index.min()
def ratio(m, b, pre=28, post=14):
    x = D[D.model == m].set_index("day")
    a_ = x[(x.index >= b - pd.Timedelta(days=pre)) & (x.index < b)]; c_ = x[(x.index >= b) & (x.index < b + pd.Timedelta(days=post))]
    if len(a_) < 3 or len(c_) < 2: return np.nan
    return (c_.mae.sum() / c_.lvl.sum()) / (a_.mae.sum() / a_.lvl.sum()) - 1
def win(m, a, b):
    x = D[(D.model == m) & (D.day >= a) & (D.day < b)]; return f"{x.mae.mean():.2f} / {x.mae.sum() / x.lvl.sum():.2f} / {x.bias.mean():+.2f} / {len(x)}"
out = ["", "## Addendum 1 (post hoc; EXPECTATIONS.md addendum 1)", "",
       f"(a) First day after June with daily fuel index above 152 (the training-range top in RESULTS_OOS.md): {brk.date()}.", ""]
tab = {m: {"28d before": win(m, brk - pd.Timedelta(days=28), brk), "14d after": win(m, brk, brk + pd.Timedelta(days=14)),
           "days 15-42 after": win(m, brk + pd.Timedelta(days=14), brk + pd.Timedelta(days=42)), "rel. change 14d": f"{ratio(m, brk):+.0%}"} for m in D.model.unique()}
out += ["MAE / relative MAE / bias / days:", "", pd.DataFrame(tab).T.to_markdown(), ""]
# (b) null distribution of the ratio over all dates with enough data
AHC = pd.Timestamp("2026-06-11"); tab = {}
for m in D.model.unique():
    dates = pd.date_range("2026-04-29" if m != "E structural (b), published inputs" else "2026-01-29", "2026-10-24")
    r = np.array([ratio(m, d) for d in dates]); r = r[np.isfinite(r)]; v = ratio(m, AHC)
    tab[m] = {"ratio at 2026-06-11": f"{v:+.0%}", "percentile among all dates": f"{(r < v).mean():.0%}", "90th pct of null": f"{np.percentile(r, 90):+.0%}", "dates": len(r)}
out += ["(b) Null for the hybrid-coupling jump: the same ratio (14 days after vs 28 before, relative MAE) at every date:", "", pd.DataFrame(tab).T.to_markdown(), ""]
# (c) frozen at 2026-06-10 vs daily retrained
TS.init(); F = TS.G["F"]; c = TS.cols("full", F)
d_fz = F.index.get_loc(pd.Timestamp("2026-06-11", tz="Europe/Berlin").tz_convert("UTC")); i1 = d_fz - 34 * 4
pred = TS.fit_models(F, c, i1 - TS.L, i1, "blend"); rows = []
for k in range(42):
    day = pd.Timestamp("2026-06-11") + pd.Timedelta(days=k); d0 = F.index.get_loc(pd.Timestamp(day, tz="Europe/Berlin").tz_convert("UTC"))
    X = F[c].iloc[d0:d0 + 96].copy(); X["gasprice"] = F["gasprice"].iloc[d0 - 96]
    p = pred(X.to_numpy(float)); y = F.price.iloc[d0:d0 + 96].to_numpy()
    rows.append(dict(day=day, frozen=np.abs(p - y).mean(), bias_fz=(p - y).mean()))
Z = pd.DataFrame(rows).set_index("day"); x = D[D.model == "full"].set_index("day"); Z["daily"] = x.mae.reindex(Z.index); Z["bias_daily"] = x.bias.reindex(Z.index)
tab = {}
for nm, a_, b_ in (("days 1-14", 0, 14), ("days 15-28", 14, 28), ("days 29-42", 28, 42)):
    z = Z.iloc[a_:b_]; tab[nm] = {"frozen at 06-10 MAE (bias)": f"{z.frozen.mean():.2f} ({z.bias_fz.mean():+.2f})", "daily retrained MAE (bias)": f"{z.daily.mean():.2f} ({z.bias_daily.mean():+.2f})",
                                  "frozen - daily": f"{(z.frozen - z.daily).mean():+.2f}"}
out += ["(c) Tree frozen at 2026-06-10 against the daily-retrained tree after hybrid-coupling go-live (ct/kWh):", "", pd.DataFrame(tab).T.to_markdown(), ""]
open(os.path.dirname(os.path.abspath(__file__)) + "/stability.md", "a").write("\n".join(out)); print("\n".join(out))
