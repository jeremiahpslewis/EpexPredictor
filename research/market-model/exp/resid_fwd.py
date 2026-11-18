"""Forward check: does temperature explain the curve's error beyond what recent errors already tell?
Train on all days before T, test on the next 30 days, roll T on. Night hours reported separately."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for tag in sys.argv[1:]:
    z = np.load(f"{SP}/out/struct_dg/debug_{tag}.npz"); r = z["r"] / 10; lo, hi = np.percentile(r, [1, 99]); r = np.clip(r, lo, hi)
    df = pd.DataFrame({"r": r, "hour": z["hour"], "day": z["day"], "x": z["x"], "dtype": z["dtype"], "temp": z["eu"][:, 3]}); df["t"] = df.day - df.day.min()
    P = df.pivot_table(index="day", columns="hour", values="r", aggfunc="mean"); P = P.reindex(range(df.day.min(), df.day.max() + 1)); dm = P.mean(1)
    for k in (1, 2):
        df[f"dm{k}"] = df.day.map(dm.shift(k)); S = P.shift(k).stack(); df[f"sh{k}"] = pd.MultiIndex.from_arrays([df.day, df.hour]).map(S.to_dict())
    df["dm7"] = df.day.map(dm.shift(1).rolling(7, min_periods=3).mean()); df["dm28"] = df.day.map(dm.shift(1).rolling(28, min_periods=7).mean())
    hb = P.T.groupby(P.columns // 3).mean().T                                    # day x 3-h block
    for w in (7, 28): S = hb.shift(1).rolling(w, min_periods=3).mean().stack(); df[f"hb{w}"] = pd.MultiIndex.from_arrays([df.day, df.hour // 3]).map(S.to_dict())
    td = df.groupby("day").temp.mean(); df["temp_day"] = df.day.map(td); df["temp_anom"] = df.temp - df.day.map(td.shift(1).rolling(24, min_periods=7).mean()); df["temp_day_anom"] = df.temp_day - df.day.map(td.shift(1).rolling(24, min_periods=7).mean())
    night = (df.hour < 8).to_numpy(); Ts = list(range(240, int(df.t.max()) - 29, 30))
    def fwd(cols):
        pred = np.full(len(df), np.nan)
        for T in Ts:
            tr = (df.t < T).to_numpy(); te = ((df.t >= T) & (df.t < T + 30)).to_numpy()
            m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1", verbose=-1, n_jobs=4).fit(df.loc[tr, cols], df.r[tr]); pred[te] = m.predict(df.loc[te, cols])
        ok = np.isfinite(pred); e = np.abs(df.r - pred); return e[ok].mean(), e[ok & night].mean()
    ok = (df.t >= Ts[0]) & (df.t < Ts[-1] + 30); e0 = np.abs(df.r); print(f"\n{tag}: {len(Ts)} test months; curve's error there: all hours {e0[ok].mean():.2f}, night {e0[ok & night].mean():.2f} ct. Left after trees on the error, trained on the past only:")
    B = ["x", "hour", "dtype"]; L = B + ["dm1", "sh1", "dm2", "sh2"]; L2 = L + ["dm7", "dm28", "hb7", "hb28"]
    for lab, cols in {"x, hour, day type": B, "+ errors 1 and 2 days back": L, "+ errors 1, 2 days back + temperature level": L + ["temp", "temp_day"], "+ errors 1, 2 days back + temperature anomaly": L + ["temp_anom", "temp_day_anom"],
                      "+ errors 1, 2 d back and 7/28-day means": L2, "+ those + temperature level": L2 + ["temp", "temp_day"], "+ those + temperature anomaly": L2 + ["temp_anom", "temp_day_anom"], "+ those + date": L2 + ["t"]}.items():
        a, n = fwd(cols); print(f"  {lab:<48} all {a:.2f}   night {n:.2f}")
