"""Pre-registered out-of-window test A1/A2 for French availability (EXPECTATIONS_11ZONE.md). Usage: fr_test.py TRAIN_RECS.pkl POST.npz FR_AVAIL.parquet
Frozen regression from October-February: FR shock = -36.6 + 1.77 * unavailable GW + 0.33 * FR residual load GW."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import solve11 as V
RECS, POST, AV = sys.argv[1:4]
P = dict(np.load(POST, allow_pickle=True)); i = 0; keys = [str(x) for x in P["de_keys"]]; th = {k: float(P[k][i]) for k in keys}
nb = dict(sbar=P["sbar"][1:], al=P["al"][i], be=P["be"][i], aa=P["aa"][i], bb=P["bb"][i], sg=P["sg"][i])
d = [r for r in pickle.load(open(RECS, "rb"))["recs"]]
rows = []
for r in d:
    c, t = V.curves_logistic(th, nb, r); S, _ = V.supply_fn(c, t)(r["p"]); rows.append(dict(t=r["t"], fr=((S - r["D"] - r["X"] - r["n"]) / 1000)[1], load=r["D"][1] / 1000))
D = pd.DataFrame(rows); A = pd.read_parquet(AV).pregate
D["unav"] = A.reindex(pd.DatetimeIndex(D.t).normalize() + pd.Timedelta(hours=12)).to_numpy(); D["pred"] = -36.6 + 1.77 * D.unav + 0.33 * D.load
D["month"] = pd.DatetimeIndex(D.t).strftime("%Y-%m"); M = D.groupby("month")[["fr", "pred", "unav"]].mean(); M["pred_minus_rec"] = M.pred - M.fr
print(M.round(1).to_string())
summer = M.loc[[m for m in M.index if m in ("2026-06", "2026-07", "2026-08")]]; winter = M.loc[[m for m in M.index if m in ("2025-12", "2026-01", "2026-02")]]
a1 = bool((summer.pred_minus_rec.abs() <= 5).all()); a2 = float(summer.unav.mean() - winter.unav.mean())
print(f"A1 (|pred - recovered| <= 5 GW in Jun-Aug): {'met' if a1 else 'not met'}  {summer.pred_minus_rec.round(1).to_dict()}")
print(f"A2 (summer unavailable exceeds winter by >= 5 GW): {'met' if a2 >= 5 else 'not met'}  difference {a2:+.1f} GW (summer {summer.unav.mean():.1f}, winter {winter.unav.mean():.1f})")
