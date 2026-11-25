"""Advisor check on addendum-1 R1s/R2s: does the fuel-scaled learner need the structural forecast at all?
Same learner, data, features and training/holdout split as distill_exp.py R1s/R2s, but pE and dpE are dropped and the
target is the actual price itself (fuel-scaled), not the structural error. Variant 'res-plag' instead targets
(actual - p_lag) on the same scale. If these land near R1s/R2s (2.39 / 2.29), the gain comes from the learner,
not from correcting the structural model. Post hoc (window already used). Writes ablate_noE.md."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = sys.argv[:1]
import distill_exp as X
from common import *
M = load_members(); a, cut, Es, T = M["a"], M["cut"], M["Es"], M["T"]
tr = X.feats(f"{SP}/distill/pred_train.pkl", f"{SP}/distill/train11.pkl"); tr["y"] = X.fr.price.reindex(tr.index).to_numpy()
tr = tr[tr.index < pd.Timestamp("2026-09-01", tz="UTC")]
hs = X.feats(f"{E11}/pred_holdout_strict.pkl", f"{E11}/oos11s.pkl"); hs["y"] = X.fr.price.reindex(hs.index).to_numpy()
hs_day = hs.index.tz_convert("Europe/Berlin").normalize()
def dsn(df):
    Z = X.design(df).drop(columns=["pE", "dpE"]); Z["plag"] = Z["plag"] / df.Lc * 100; return Z
TGT = {"level": lambda df: (df.y / df.Lc * 100).to_numpy(), "res-plag": lambda df: ((df.y - df.plag) / df.Lc * 100).to_numpy()}
BACK = {"level": lambda df, z: z * df.Lc.to_numpy() / 100, "res-plag": lambda df, z: df.plag.to_numpy() + z * df.Lc.to_numpy() / 100}
def fit(Z, y):
    import lightgbm as lgb
    ok = np.isfinite(y); return lgb.train(X.PAR, lgb.Dataset(Z[ok], label=y[ok]), 300)
F = {}
for nm in TGT:
    m1 = fit(dsn(tr), TGT[nm](tr)); F[f"frozen, no E_s, target {nm}"] = series_to_grid(pd.Series(BACK[nm](hs, m1.predict(dsn(hs))), index=hs.index), cut)
    G = np.full_like(Es, np.nan)
    for j, c in enumerate(cut):
        dl = c.tz_convert("Europe/Berlin").normalize(); prev = hs[hs_day < dl]; today = hs[hs_day == dl]
        Z2 = pd.concat([dsn(tr), dsn(prev)]) if len(prev) else dsn(tr); y2 = np.r_[TGT[nm](tr), TGT[nm](prev)] if len(prev) else TGT[nm](tr)
        m2 = fit(Z2, y2)
        if len(today): G[j] = series_to_grid(pd.Series(BACK[nm](today, m2.predict(dsn(today))), index=today.index), [c])[0]
    F[f"expanding, no E_s, target {nm}"] = G
F = {k: np.where(np.isfinite(v), v, Es) for k, v in F.items()}
plag = series_to_grid(hs.plag, cut); F["persistence p_lag (slope 0)"] = np.where(np.isfinite(plag), plag, Es)
rows = {"E_s": {"MAE": f"{day_mae(Es, a).mean():.3f}"}, "T": {"MAE": f"{day_mae(T, a).mean():.3f}"}}
for k, v in F.items():
    rows[k] = {"MAE": f"{day_mae(v, a).mean():.3f}", "nan filled": int((~np.isfinite(v)).sum()), "vs E_s": paired(v, Es, a), "vs T": paired(v, T, a)}
out = ["# Ablation: the fuel-scaled learner without the structural forecast (post hoc)", "",
       "Compare with R1s 2.39 and R2s 2.29 in distill_results.md (same learner, rows, split and features plus pE, dpE).", "",
       pd.DataFrame(rows).T.fillna("").to_markdown()]
open(os.path.dirname(os.path.abspath(__file__)) + "/ablate_noE.md", "w").write("\n".join(out)); print("\n".join(out))
