"""Own forecast of TSO wind/solar/load from the dense grid, trained on 2-day-old weather forecasts since 2025-10.
Saves per-cutoff forecasts for the 66 live cutoffs (opt / pes vintages) and an out-of-fold series for training the price model."""
import glob, sys, numpy as np, pandas as pd, lightgbm as lgb
import live_common as lc, dense_common as dc
F = lc.load(); D = F["DE"]; idx = D.index; fc = dc.tso(idx)
H = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{lc.SP}/dense/hist_*.parquet"))]).sort_index(); H = H[~H.index.duplicated()]
H15 = dc.to15(H, idx); P15 = dc.to15(pd.read_parquet(f"{lc.SP}/dense/prev.parquet"), idx)
X2 = dc.feats(H15[[c for c in H15.columns if c.endswith("_d2")]].rename(columns=lambda c: c[:-3]), D); X0 = dc.feats(H15[[c for c in H15.columns if not c.endswith("_d2")]], D)
lo = idx.get_loc(pd.Timestamp("2025-10-02", tz="UTC")); T = ["wind", "solar", "load"]; P = {"verbosity": -1, "num_threads": 4, "learning_rate": 0.05, "num_leaves": 31}
def fit(X, y): ok = ~(np.isnan(X).any(1) | np.isnan(y)); return lgb.train(P, lgb.Dataset(X[ok], label=y[ok]), num_boost_round=400)
A = X0.to_numpy(float)
cuts = lc.cutoffs(idx, sys.argv[2], sys.argv[3], int(sys.argv[4])); out = {f"{t}_{k}": [] for t in T for k in ("arch", "opt", "pes", "true")}
for j, d0 in enumerate(cuts):
    if j % 2 == 0: M = {t: fit(A[lo:d0 - lc.ISSUE], fc[t].to_numpy()[lo:d0 - lc.ISSUE]) for t in T}       # weekly refit, data up to the issue time
    Xte = {"arch": X0.iloc[d0:d0 + lc.HZ].to_numpy(float)}
    W2 = H15[[c for c in H15.columns if c.endswith("_d2")]].rename(columns=lambda c: c[:-3]).iloc[d0:d0 + lc.HZ]   # 2-day-old forecast for all hours
    Xte["pes"] = dc.feats(W2, D.iloc[d0:d0 + lc.HZ])[X2.columns].to_numpy(float); Xte["opt"] = Xte["pes"]
    for t in T:
        for k in Xte: out[f"{t}_{k}"].append(M[t].predict(Xte[k]))
        out[f"{t}_true"].append(fc[t].to_numpy()[d0:d0 + lc.HZ])
R = {k: np.stack(v).astype(np.float32) for k, v in out.items()}
for k in ("arch", "opt", "pes", "true"): R[f"rl_{k}"] = R[f"load_{k}"] - R[f"wind_{k}"] - R[f"solar_{k}"]
np.savez_compressed(f"{lc.SP}/out/{sys.argv[1]}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **R)
print(f"{len(cuts)} forecasts; MAE in GW against the TSO day-ahead figure (11-point model before: wind 1.9 / 2.9, solar 0.8 / 1.4, load 1.0 / 1.0, residual load 3.0 / 4.2)")
print(f"{'':<8}latest weather   forecast 1-2 d earlier   forecast 2-3 d earlier")
for t in T + ["rl"]: print(f"{t:<8}" + "              ".join(f"{np.nanmean(np.abs(R[f'{t}_{k}'] - R[f'{t}_true'])):.2f}" for k in ("arch", "opt", "pes")))
