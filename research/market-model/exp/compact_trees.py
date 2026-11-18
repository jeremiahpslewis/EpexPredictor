"""Trees with the Bayesian model's inputs only (residual load, gas, calendar), single stage, live setup autumn."""
import numpy as np, pandas as pd, lightgbm as lgb
import live_common as lc, dense_common as dc
F = lc.load(); D = F["DE"]; idx = D.index; fc = dc.tso(idx); G = np.load(f"{lc.SP}/out/gen_fc_app.npz")
cuts = lc.cutoffs(idx, "2026-09-03", "2026-11-07", 1); cols = ["rl", "gasprice", "slot", "weekday", "holiday", "ts"]
X = D[["gasprice", "slot", "weekday", "holiday", "ts"]].copy(); X.insert(0, "rl", fc["rl"]); X = X[cols]; y = D["price"].to_numpy(float)
P = {"force_col_wise": True, "verbosity": -1, "num_threads": 4}; out = []
for j, d0 in enumerate(cuts):
    Xtr = X.to_numpy(float)[d0 - lc.L:d0]; ytr = y[d0 - lc.L:d0]; ok = ~(np.isnan(Xtr).any(1) | np.isnan(ytr))
    Xte = X.to_numpy(float)[d0:d0 + lc.HZ].copy(); Xte[:, 0] = G["rl_pes"][j]; Xte[:, 1] = D["gasprice"].to_numpy()[d0 - lc.ISSUE]
    g = 1; raw = lgb.train(P, lgb.Dataset(Xtr[ok], label=ytr[ok])).predict(Xte)
    rel = lgb.train(P, lgb.Dataset(Xtr[ok], label=ytr[ok] / Xtr[ok][:, g], weight=Xtr[ok][:, g] ** 2)).predict(Xte) * Xte[:, g]
    out.append((raw + rel) / 2)
np.savez_compressed(f"{lc.SP}/out/compact_trees.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), vint=np.stack(out).astype(np.float32))
print("compact trees done", flush=True)
