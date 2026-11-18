"""Is the slow change of the curve's error seasonal physics (temperature, daylight) or drift over time? Night hours separately."""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for tag in sys.argv[1:]:
    z = np.load(f"{SP}/out/struct_dg/debug_{tag}.npz"); r = z["r"] / 10; lo, hi = np.percentile(r, [1, 99]); r = np.clip(r, lo, hi)
    df = pd.DataFrame({"r": r, "hour": z["hour"], "day": z["day"], "x": z["x"], "dtype": z["dtype"], "gas": z["gas"], "eu_wind": z["eu"][:, 0], "eu_ns": z["eu"][:, 1], "eu_irr": z["eu"][:, 2], "eu_temp": z["eu"][:, 3], "solar": z["solar"], "wind": z["wind"], "load": z["load"]})
    df["t"] = df.day - df.day.min(); df["week"] = df.t // 7; df["doy"] = df.day % 365.25
    g = df.groupby("day"); df["temp_day"] = g.eu_temp.transform("mean"); df["solar_day"] = g.solar.transform("max"); df["daylen"] = g.solar.transform(lambda s: (s > 0.5).sum())
    folds = df.week % 5; night = (df.hour < 8).to_numpy()
    def cv(cols):
        pred = np.zeros(len(df))
        for k in range(5):
            tr = folds != k; m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1", verbose=-1, n_jobs=4).fit(df.loc[tr, cols], df.r[tr])
            pred[~tr] = m.predict(df.loc[~tr, cols])
        e = np.abs(df.r - pred); return f"{e.mean():.2f}   night {e[night].mean():.2f}"
    e0 = np.abs(df.r - df.r.median()); print(f"\n{tag}: curve's error MAE {e0.mean():.2f}   night {e0[night].mean():.2f}; left after trees on the error (weeks held out)")
    B = ["x", "hour", "dtype"]
    for lab, cols in {"x, hour, day type": B, "+ date": B + ["t"], "+ day of year": B + ["doy"], "+ temperature (hour, day mean)": B + ["eu_temp", "temp_day"], "+ day length, solar peak of the day": B + ["daylen", "solar_day"],
                      "+ temperature + day length + solar peak": B + ["eu_temp", "temp_day", "daylen", "solar_day"], "+ temperature, day length, solar peak, gas": B + ["eu_temp", "temp_day", "daylen", "solar_day", "gas"],
                      "+ all of these + date": B + ["eu_temp", "temp_day", "daylen", "solar_day", "gas", "t"]}.items(): print(f"  {lab:<46} {cv(cols)}")
