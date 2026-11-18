"""If the realised plant mix were known: how much of the structural curve's error does it explain? (analysis only)"""
import json, glob, numpy as np, pandas as pd, lightgbm as lgb, live_common as lc, warnings; warnings.filterwarnings("ignore")
SP = lc.SP; F = lc.load(); idx = F["DE"].index; local = idx.tz_convert("Europe/Berlin")
fr = []
for f in sorted(glob.glob(f"{SP}/ext/ecpp/pp_*.json")):
    d = json.load(open(f)); fr.append(pd.DataFrame({p["name"]: pd.to_numeric(pd.Series(p["data"]), errors="coerce").to_numpy() for p in d["production_types"]}, index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)))
M = pd.concat(fr); M = M[~M.index.duplicated()].reindex(idx) / 1000
M = M.rename(columns={"Fossil brown coal / lignite": "lignite", "Fossil hard coal": "coal", "Fossil gas": "gas_gen", "Hydro pumped storage consumption": "pump", "Hydro pumped storage": "psgen", "Cross border electricity trading": "trade"})
M["thermal"] = M[["lignite", "coal", "gas_gen", "Fossil oil", "Fossil coal-derived gas"]].sum(axis=1, min_count=3); M.to_parquet(f"{SP}/out/de_mix.parquet")
tag = "2026-11-06"; z = np.load(f"{SP}/out/struct_dg/debug_{tag}.npz"); pos = z["pos"]
df = pd.DataFrame({"y": z["y"] / 10, "m": z["m"] / 10, "r": z["r"] / 10, "x": z["x"], "rl": z["rl"], "gas": z["gas"], "hour": z["hour"], "dtype": z["dtype"], "day": z["day"]})
for c in ("lignite", "coal", "gas_gen", "pump", "psgen", "trade", "thermal"): df[c] = np.nanmean(np.stack([M[c].to_numpy()[pos + k] for k in range(4)]), 0)
df = df.dropna(); lo, hi = np.percentile(df.r, [1, 99]); df["r"] = df.r.clip(lo, hi); df["week"] = (df.day - df.day.min()) // 7; night = (df.hour < 8).to_numpy()
print(f"fit of {tag}, {len(df)} hours. Correlation of net trade with residual load {np.corrcoef(df.trade, df.rl)[0, 1]:+.2f} (positive = imports when load is high); mean: lignite {df.lignite.mean():.1f}, coal {df.coal.mean():.1f}, gas {df.gas_gen.mean():.1f}, pumping {df.pump.mean():.1f}, trade {df.trade.mean():+.1f} GW")
print("\n1. the nights the curve missed: what ran (GW, 00-08 h means)")
for day in ("2026-01-10", "2026-02-03", "2026-09-19", "2026-09-05"):
    t0 = idx.get_loc(pd.Timestamp(day, tz="Europe/Berlin").tz_convert("UTC")); s = M.iloc[t0:t0 + 32]; print(f"   {day}: lignite {s.lignite.mean():.1f}  coal {s.coal.mean():.1f}  gas {s.gas_gen.mean():.1f}  pumping {s.pump.mean():+.1f}  net trade {s.trade.mean():+.1f}  price {F['DE']['price'].iloc[t0:t0 + 32].mean():.1f} ct")
def cv(cols, target="r"):
    pred = np.zeros(len(df)); folds = (df.week % 5).to_numpy()
    for k in range(5):
        tr = folds != k; mm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1", verbose=-1, n_jobs=2).fit(df.loc[tr, cols], df.loc[tr, target]); pred[~tr] = mm.predict(df.loc[~tr, cols])
    e = np.abs(df[target] - pred); return f"{e.mean():.2f}   night {e[night].mean():.2f}"
e0 = df.r.abs(); B = ["x", "hour", "dtype", "gas"]
print(f"\n2. curve's error: MAE {e0.mean():.2f}   night {e0[night].mean():.2f} ct. Left after trees on the error (weeks held out) knowing ...")
for lab, cols in {"x, hour, day type, gas (nothing new)": B, "+ realised lignite, coal, gas output": B + ["lignite", "coal", "gas_gen"], "+ realised net trade and pumping": B + ["trade", "pump", "psgen"], "+ all of the realised mix": B + ["lignite", "coal", "gas_gen", "trade", "pump", "psgen"], "+ realised thermal output only": B + ["thermal"]}.items(): print(f"   {lab:<40} {cv(cols)}")
print("\n3. price directly (ct): MAE of trees given ...")
for lab, cols in {"residual load, hour, day type, gas": ["rl", "hour", "dtype", "gas"], "realised thermal output, gas": ["thermal", "gas"], "realised lignite, coal, gas output, gas": ["lignite", "coal", "gas_gen", "gas"], "residual load + net trade + pumping, hour, gas": ["rl", "trade", "pump", "psgen", "hour", "dtype", "gas"]}.items(): print(f"   {lab:<46} {cv(cols, 'y')}")
print("\n4. how predictable is the net trade from what is known in advance? MAE of trees (GW); spread of net trade (mean abs deviation) %.1f GW" % np.abs(df.trade - df.trade.median()).mean())
nb = pd.read_parquet(f"{SP}/out/nb_rl.parquet"); df["nb"] = nb[[c for c in nb.columns if c.startswith("rl_")]].sum(axis=1, min_count=8).reindex(idx).to_numpy()[pos[:len(z['y'])]][df.index]
for lab, cols in {"German residual load, hour, day type, gas": ["rl", "hour", "dtype", "gas"], "+ neighbours' residual load": ["rl", "hour", "dtype", "gas", "nb"]}.items(): print(f"   {lab:<46} {cv(cols, 'trade')}")
