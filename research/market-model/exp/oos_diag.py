"""LightGBM as analysis tool: what explains the structural model's out-of-sample errors (known residual load)?
Errors of all forecasts of the three periods, first forecast day; trees are fitted on them out of fold (blocks of
forecasts in time order) with one group of candidate inputs at a time. usage: oos_diag.py [neighbours]"""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
import live_common as lc
SP = lc.SP; F = lc.load(); idx = F["DE"].index; local = idx.tz_convert("Europe/Berlin")
rows = []
for period, run in (("winter", "fw"), ("summer", "fs"), ("autumn", "fa")):
    I = np.load(f"{SP}/out/inputs_{period}.npz"); S = np.load(f"{SP}/out/structfc_{run}_known_R1.npz")
    for j, d0 in enumerate(I["cuts"]):
        T = np.arange(int(d0), int(d0) + 96); d = pd.DataFrame({"pos": T, "origin": int(d0), "period": period, "y": I["actual"][j, :96], "curve": S["curve"][j, :96], "full": S["vint"][j, :96], "gas": float(I["gas"][j]),
                          "rl": I["rl_true"][j, :96], "wind": I["wind_true"][j, :96], "solar": I["solar_true"][j, :96], "load": I["load_true"][j, :96]})
        d[["eu_wind", "eu_ns", "eu_irr", "eu_temp"]] = I["eu"][j, :96]; rows.append(d)
d = pd.concat(rows, ignore_index=True); p = d.pos.to_numpy()
d["hour"] = local.hour.to_numpy()[p]; d["wd"] = F["DE"]["weekday"].to_numpy()[p]; d["hol"] = F["DE"]["holiday"].to_numpy()[p]; doy = local.dayofyear.to_numpy()[p]; d["sin"] = np.sin(2 * np.pi * doy / 365); d["cos"] = np.cos(2 * np.pi * doy / 365)
g = d.groupby("origin"); d["rl_max"] = g.rl.transform("max"); d["rl_min"] = g.rl.transform("min"); d["rl_mean"] = g.rl.transform("mean"); d["rl_dev"] = d.rl - d.rl_mean; d["rl_rank"] = g.rl.rank(pct=True)
d["ramp1"] = g.rl.diff(4).fillna(0); d["ramp3"] = g.rl.diff(12).fillna(0); d["fwd3"] = (g.rl.shift(-12) - d.rl).fillna(0)
d["solar_share"] = d.solar / d.load; d["wind_share"] = d.wind / d.load
for z in ("FR", "NL", "AT", "DK1", "BE"): d[f"p_{z}"] = F[z]["price"].to_numpy()[p]
NB = []
if os.path.isdir(f"{SP}/nb"):
    for f in sorted(os.listdir(f"{SP}/nb")):
        c = f[:-8]; n = pd.read_parquet(f"{SP}/nb/{f}").reindex(idx).interpolate(limit=3) / 1000
        if "load" not in n or n["load"].notna().mean() < 0.5: continue
        r = n["load"] - n.get("wind_onshore", 0).fillna(0) - (n["wind_offshore"].fillna(0) if "wind_offshore" in n else 0) - n.get("solar", 0).fillna(0)
        d[f"rl_{c}"] = r.to_numpy()[p]; d[f"rlz_{c}"] = ((r - r.rolling(96 * 365, min_periods=96 * 60).mean()) / r.rolling(96 * 365, min_periods=96 * 60).std()).to_numpy()[p]; NB.append(c)
d.to_parquet(f"{SP}/out/oos_errors.parquet")
orig = np.sort(d.origin.unique()); blocks = np.array_split(orig, 10); fold = d.origin.map({o: k for k, b in enumerate(blocks) for o in b}).to_numpy()
def cv(cols, target):
    e = (d.y - d[target]).to_numpy(); pred = np.zeros(len(d)); ok = d[cols].notna().all(1).to_numpy()
    for k in range(10):
        tr, te = (fold != k) & ok, (fold == k) & ok
        pred[te] = lgb.train({"verbosity": -1, "num_threads": 4, "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 100}, lgb.Dataset(d.loc[tr, cols], label=e[tr]), num_boost_round=200).predict(d.loc[te, cols])
    return 100 * (1 - np.abs(e - pred)[ok].mean() / np.abs(e)[ok].mean()), pred
G = {"hour, weekday, holiday": ["hour", "wd", "hol"], "residual load, gas": ["rl", "gas"], "wind / solar / load split": ["wind", "solar", "load", "solar_share", "wind_share"], "Europe-wide weather": ["eu_wind", "eu_ns", "eu_irr", "eu_temp"],
     "position within the day (max, min, mean, deviation, rank)": ["rl_max", "rl_min", "rl_mean", "rl_dev", "rl_rank"], "ramps": ["ramp1", "ramp3", "fwd3"], "season": ["sin", "cos"],
     "neighbours' prices (not known in advance; for scale)": [f"p_{z}" for z in ("FR", "NL", "AT", "DK1", "BE")]}
if NB: G["neighbours' published residual load: " + ", ".join(NB)] = [f"rl_{c}" for c in NB]
print(f"{len(d)} quarter hours from {len(orig)} forecasts; mean absolute error curve only {np.abs(d.y - d.curve).mean():.2f}, with level {np.abs(d.y - d.full).mean():.2f} ct/kWh")
print(f"{'out-of-fold reduction of the mean absolute error by trees on':<62}{'curve only':>12}{'with level':>12}")
for lab, cols in G.items(): print(f"  {lab + ' (+ hour, residual load)':<60}{cv(sorted(set(cols + ['hour', 'rl'])), 'curve')[0]:>11.1f}%{cv(sorted(set(cols + ['hour', 'rl'])), 'full')[0]:>11.1f}%")
allc = sorted(set(sum([v for k, v in G.items() if not k.startswith("neighbours' prices")], []))); print(f"  {'everything knowable together':<60}{cv(allc, 'curve')[0]:>11.1f}%{cv(allc, 'full')[0]:>11.1f}%")
