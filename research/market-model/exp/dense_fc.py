"""Own forecast of TSO wind/solar/load from the dense grid. Saves per-cutoff forecasts. usage: dense_fc.py STRIDE"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
import live_common as lc, dense_common as dc
F = lc.load(); D = F["DE"]; idx = D.index; fc = dc.tso(idx)
A15 = dc.to15(pd.read_parquet(f"{lc.SP}/dense/arch.parquet"), idx); P15 = dc.to15(pd.read_parquet(f"{lc.SP}/dense/prev.parquet"), idx)
print("missing share archive:", float(A15.isna().mean().mean().round(4)), " 120 m wind missing:", float(A15[[c for c in A15.columns if c.startswith('w120')]].isna().mean().mean().round(4)), flush=True)
XA = dc.feats(A15, D); cuts = lc.cutoffs(idx, "2026-09-03", "2026-11-07", int(sys.argv[1])); T = ["wind", "solar", "load"]
out = {f"{t}_{k}": [] for t in T for k in ("arch", "opt", "pes", "true")}
for d0 in cuts:
    Xtr = XA.to_numpy(float)[d0 - lc.L:d0]; Xte = {"arch": XA.iloc[d0:d0 + lc.HZ].to_numpy(float)}
    for k in ("opt", "pes"): Xte[k] = dc.feats(dc.vint_rows(P15, d0, k), D.iloc[d0:d0 + lc.HZ])[XA.columns].to_numpy(float)
    for t in T:
        y = fc[t].to_numpy()[d0 - lc.L:d0]; ok = ~(np.isnan(Xtr).any(1) | np.isnan(y))
        m = lgb.train({"verbosity": -1, "num_threads": 4, "learning_rate": 0.05, "num_leaves": 31}, lgb.Dataset(Xtr[ok], label=y[ok]), num_boost_round=300)
        for k in Xte: out[f"{t}_{k}"].append(m.predict(Xte[k]))
        out[f"{t}_true"].append(fc[t].to_numpy()[d0:d0 + lc.HZ])
    print(d0, flush=True)
R = {k: np.stack(v).astype(np.float32) for k, v in out.items()}; np.savez_compressed(f"{lc.SP}/out/dense_fc_s{sys.argv[1]}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **R)
print(f"{len(cuts)} forecasts; MAE in GW against the TSO day-ahead figure (earlier 11-point model: wind 1.9 / 2.9, solar 0.8 / 1.4, load 1.0 / 1.0, residual load 3.0 / 4.2)")
print(f"{'':<14}weather that occurred   forecast 1-2 days earlier   forecast 2-3 days earlier")
for t in T: print(f"{t:<14}" + "                    ".join(f"{np.nanmean(np.abs(R[f'{t}_{k}'] - R[f'{t}_true'])):.1f}" for k in ("arch", "opt", "pes")))
print(f"{'residual load':<14}" + "                    ".join(f"{np.nanmean(np.abs((R[f'load_{k}'] - R[f'wind_{k}'] - R[f'solar_{k}']) - (R['load_true'] - R['wind_true'] - R['solar_true']))):.1f}" for k in ("arch", "opt", "pes")))
