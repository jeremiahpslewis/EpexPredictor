"""Trees as an analysis tool on the errors of the structural curve: which inputs explain them (held-out weeks)?"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for tag in sys.argv[1:]:
    z = np.load(f"{SP}/out/struct_dg/debug_{tag}.npz"); r = z["r"] / 10; lo, hi = np.percentile(r, [1, 99]); r = np.clip(r, lo, hi)
    df = pd.DataFrame({"r": r, "hour": z["hour"], "day": z["day"], "x": z["x"], "rl": z["rl"], "dtype": z["dtype"], "gas": z["gas"], "wind": z["wind"], "solar": z["solar"], "load": z["load"]})
    df["t"] = df.day - df.day.min(); df["week"] = df.t // 7; df["doy"] = (df.day % 365.25)
    g = df.groupby("day").rl; df["rl_min"] = g.transform("min"); df["rl_max"] = g.transform("max"); df["rl_mean"] = g.transform("mean"); df["rel"] = df.rl - df.rl_mean; df["range"] = df.rl_max - df.rl_min
    df["ramp"] = (df.rl.shift(-1) - df.rl.shift(1)).fillna(0); df["ramp3"] = (df.rl.shift(-3) - df.rl).fillna(0); df["back3"] = (df.rl - df.rl.shift(3)).fillna(0)
    df["rl_next_min"] = df.rl[::-1].rolling(8, min_periods=1).min()[::-1].to_numpy(); df["rl_next_max"] = df.rl[::-1].rolling(8, min_periods=1).max()[::-1].to_numpy(); df["rl_prev_max"] = df.rl.rolling(8, min_periods=1).max(); df["rl_prev_min"] = df.rl.rolling(8, min_periods=1).min()
    # lagged errors as known when forecasting: day mean and same hour, two days back
    P = df.pivot_table(index="day", columns="hour", values="r", aggfunc="mean"); dm = P.mean(1)
    df["dm_lag2"] = df.day.map(dm.shift(2)); df["same_lag2"] = [P.shift(2).get(h, pd.Series(dtype=float)).get(d, np.nan) for d, h in zip(df.day, df.hour)]
    df["dm_lag1"] = df.day.map(dm.shift(1)); df["same_lag1"] = [P.shift(1).get(h, pd.Series(dtype=float)).get(d, np.nan) for d, h in zip(df.day, df.hour)]
    folds = df.week % 5
    def cv(cols):
        pred = np.zeros(len(df))
        for k in range(5):
            tr = folds != k; m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1", verbose=-1, n_jobs=4).fit(df.loc[tr, cols], df.r[tr])
            pred[~tr] = m.predict(df.loc[~tr, cols])
        return np.abs(df.r - pred).mean()
    base = np.abs(df.r - df.r.median()).mean(); print(f"\n{tag}: MAE of the curve's error {base:.2f} ct; left after trees on the error with ... (weeks held out)")
    S = {"x, hour, day type (static interactions)": ["x", "hour", "dtype"],
         "+ wind, solar, load separately": ["x", "hour", "dtype", "wind", "solar", "load"],
         "+ position in the day's profile, ramps": ["x", "hour", "dtype", "rel", "range", "rl_min", "rl_max", "ramp", "ramp3", "back3", "rl_next_min", "rl_next_max", "rl_prev_max", "rl_prev_min"],
         "+ both": ["x", "hour", "dtype", "wind", "solar", "load", "rel", "range", "rl_min", "rl_max", "ramp", "ramp3", "back3", "rl_next_min", "rl_next_max", "rl_prev_max", "rl_prev_min"],
         "x, hour, day type + time (date)": ["x", "hour", "dtype", "t"],
         "x, hour, day type + day of year": ["x", "hour", "dtype", "doy"],
         "x, hour, day type + gas": ["x", "hour", "dtype", "gas"],
         "x, hour, day type + errors two days back": ["x", "hour", "dtype", "dm_lag2", "same_lag2"],
         "x, hour, day type + errors one day back": ["x", "hour", "dtype", "dm_lag1", "same_lag1"],
         "everything but lagged errors": ["x", "hour", "dtype", "wind", "solar", "load", "rel", "range", "rl_min", "rl_max", "ramp", "ramp3", "back3", "rl_next_min", "rl_next_max", "rl_prev_max", "rl_prev_min", "t", "gas"],
         "everything + errors two days back": ["x", "hour", "dtype", "wind", "solar", "load", "rel", "range", "rl_min", "rl_max", "ramp", "ramp3", "back3", "rl_next_min", "rl_next_max", "rl_prev_max", "rl_prev_min", "t", "gas", "dm_lag2", "same_lag2"]}
    for lab, cols in S.items(): print(f"  {lab:<46} {cv(cols):.2f}")
