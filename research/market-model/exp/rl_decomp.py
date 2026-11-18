import numpy as np, pandas as pd, lightgbm as lgb
import live_common as lc
F = lc.load(); V = lc.vintage(F); D = F["DE"]; b = [c for c in D.columns if c != "price"]; idx = D.index
fc = pd.concat([pd.read_parquet(f"{lc.SP}/ecfc/{p}.parquet") for p in ("wind_onshore", "wind_offshore", "solar", "load")], axis=1).reindex(idx) / 1000
fc["wind"] = fc["wind_onshore"] + fc["wind_offshore"]; fc["rl"] = fc["load"] - fc["wind"] - fc["solar"]
cuts = lc.cutoffs(idx, "2026-09-03", "2026-11-07", 3); T = ["wind", "solar", "load", "rl"]; res = {(t, k): [] for t in T for k in ("arch", "vint")}; tru = {t: [] for t in T}
for d0 in cuts:
    Xtr = D[b].to_numpy(float)[d0 - lc.L:d0]; Xa = lc.test_matrix(F, V, "DE", d0, b, False); Wte = {z: lc.test_matrix(F, V, z, d0, [c for c in F[z].columns if c != "price"], True) for z in lc.ZONES}
    eu = lc.eu_feats(Wte); Xv = Wte["DE"].copy()
    for c in eu.columns: Xv[c] = eu[c].to_numpy()
    for t in T:
        y = fc[t].to_numpy()[d0 - lc.L:d0]; ok = ~(np.isnan(Xtr).any(1) | np.isnan(y))
        m = lgb.train({"verbosity": -1, "num_threads": 4}, lgb.Dataset(Xtr[ok], label=y[ok]))
        res[(t, "arch")].append(m.predict(Xa.to_numpy(float))); res[(t, "vint")].append(m.predict(Xv[b].to_numpy(float))); tru[t].append(fc[t].to_numpy()[d0:d0 + lc.HZ])
print(f"{len(cuts)} forecasts; MAE in GW of an own forecast of the TSO day-ahead figure")
print(f"{'':<14}with the weather that occurred   with weather as forecast 2-3 days earlier   average level")
for t in T:
    y = np.stack(tru[t]); print(f"{t:<14}{np.nanmean(np.abs(np.stack(res[(t,'arch')]) - y)):.1f}                             {np.nanmean(np.abs(np.stack(res[(t,'vint')]) - y)):.1f}                                         {np.nanmean(y):.1f}")
y = np.stack(tru["rl"]); print("residual load from the three parts (vint):", np.nanmean(np.abs(np.stack(res[("load","vint")]) - np.stack(res[("wind","vint")]) - np.stack(res[("solar","vint")]) - y)).round(1))
