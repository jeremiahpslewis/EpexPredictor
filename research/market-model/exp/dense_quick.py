"""Dense (58 points) against sparse (11 points) weather for forecasting TSO wind/solar, both trained on 2-day-old forecasts
over the same weeks and tested on later weeks (expanding window, weekly refit)."""
import numpy as np, pandas as pd, lightgbm as lgb
import live_common as lc, dense_common as dc
F = lc.load(); D = F["DE"]; idx = D.index; fc = dc.tso(idx)
P15 = dc.to15(pd.read_parquet(f"{lc.SP}/dense/prev.parquet"), idx); S15 = dc.to15(pd.read_parquet(f"{lc.SP}/prev_all/prev_DE.parquet"), idx)
def dense(k):
    W = P15[[c for c in P15.columns if c.endswith(f"_d{k}")]].rename(columns=lambda c: c[:-3]); return dc.feats(W, D)
def sparse(k):
    W = S15[[c for c in S15.columns if c.endswith(f"_d{k}") and c.split("_")[0] in ("wind", "irradiance", "temp")]].rename(columns=lambda c: c[:-3]).copy()
    w = W[[c for c in W.columns if c.startswith("wind_")]]; W["agg"] = ((w.clip(upper=45) / 45) ** 3).mean(1)
    for c in ("ts", "slot", "weekday", "holiday", "sunelevation", "azimuth"): W[c] = D[c].to_numpy(float)
    return W
lo = idx.get_loc(pd.Timestamp("2026-08-27", tz="UTC")); hi = idx.get_loc(pd.Timestamp("2026-11-08", tz="UTC")); first = idx.get_loc(pd.Timestamp("2026-10-05", tz="UTC"))
X = {("dense", k): dense(k).to_numpy(float) for k in (1, 2, 3)} | {("sparse", k): sparse(k).to_numpy(float) for k in (2, 3)}
res = {}
for t in ("wind", "solar", "load"):
    y = fc[t].to_numpy()
    for name, ktr in (("dense", 2), ("sparse", 2)):
        for kte in ((1, 2, 3) if name == "dense" else (2, 3)):
            pred = np.full(len(idx), np.nan)
            for a in range(first, hi, 7 * 96):
                Xtr = X[(name, ktr)][lo:a]; ok = ~(np.isnan(Xtr).any(1) | np.isnan(y[lo:a]))
                m = lgb.train({"verbosity": -1, "num_threads": 4, "learning_rate": 0.05, "min_data_in_leaf": 40}, lgb.Dataset(Xtr[ok], label=y[lo:a][ok]), num_boost_round=300)
                pred[a:a + 7 * 96] = m.predict(X[(name, kte)][a:a + 7 * 96])
            res[(t, name, kte)] = pred
sl = slice(first, hi); print(f"test {idx[first].date()} to {idx[hi].date()}, MAE in GW against the TSO day-ahead figure; models trained on 5-10 weeks of 2-day-old forecasts")
print(f"{'':<8}sparse, 2 d old   sparse, 3 d old   dense, 1 d old   dense, 2 d old   dense, 3 d old")
E = lambda t, n, k: np.nanmean(np.abs(res[(t, n, k)][sl] - fc[t].to_numpy()[sl]))
for t in ("wind", "solar", "load"): print(f"{t:<8}{E(t,'sparse',2):.2f}              {E(t,'sparse',3):.2f}              {E(t,'dense',1):.2f}             {E(t,'dense',2):.2f}             {E(t,'dense',3):.2f}")
rl = lambda n, k: np.nanmean(np.abs((res[('load', n, k)] - res[('wind', n, k)] - res[('solar', n, k)])[sl] - fc['rl'].to_numpy()[sl]))
print(f"{'resid.':<8}{rl('sparse',2):.2f}              {rl('sparse',3):.2f}              {rl('dense',1):.2f}             {rl('dense',2):.2f}             {rl('dense',3):.2f}")
