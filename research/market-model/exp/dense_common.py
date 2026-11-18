"""Dense-grid inputs and an own forecast of the TSO day-ahead wind, solar and load figures."""
import json, numpy as np, pandas as pd
import live_common as lc
NON = json.load(open(f"{lc.SP}/dense/points.json"))["onshore"]
def to15(df, idx):
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="15min")).interpolate(limit=3).reindex(idx)
def feats(W, base):
    """W: columns w80_i, w120_i, sw_i, t2_i (15 min); base: German frame rows with calendar/sun columns"""
    w = W[[c for c in W.columns if c.startswith("w120_")]].to_numpy(float); w8 = W[[c for c in W.columns if c.startswith("w80_")]].to_numpy(float)
    pc = np.clip((w - 10.8) / (43.2 - 10.8), 0, 1) ** 1.0; pc3 = np.clip(w / 43.2, 0, 1) ** 3          # km/h: cut-in 3 m/s, rated 12 m/s
    X = pd.DataFrame(index=W.index)
    for i in range(w.shape[1]): X[f"p_{i}"] = pc3[:, i]
    lat = np.array(json.load(open(f"{lc.SP}/dense/points.json"))["points"])[:, 0]; north = lat[:NON] >= 52.0
    X["on_all"] = pc3[:, :NON].mean(1); X["on_north"] = pc3[:, :NON][:, north].mean(1); X["on_south"] = pc3[:, :NON][:, ~north].mean(1); X["off_ns"] = pc3[:, NON:NON + 4].mean(1); X["off_baltic"] = pc3[:, NON + 4:].mean(1)
    X["on_lin"] = pc[:, :NON].mean(1); X["on80"] = (np.clip(w8[:, :NON] / 43.2, 0, 1) ** 3).mean(1); X["wmax"] = w[:, :NON].max(1)
    sw = W[[c for c in W.columns if c.startswith("sw_")]].to_numpy(float)[:, :NON]
    for i in range(sw.shape[1]): X[f"sw_{i}"] = sw[:, i]
    X["sw_all"] = sw.mean(1); X["sw_south"] = sw[:, ~north].mean(1); t = W[[c for c in W.columns if c.startswith("t2_")]].to_numpy(float)[:, :NON]; X["t_all"] = t.mean(1); X["t_south"] = t[:, ~north].mean(1)
    for c in ("ts", "slot", "weekday", "holiday", "sunelevation", "azimuth"): X[c] = base[c].to_numpy(float)
    return X
def tso(idx):
    fc = pd.concat([pd.read_parquet(f"{lc.SP}/ecfc/{p}.parquet") for p in ("wind_onshore", "wind_offshore", "solar", "load")], axis=1).reindex(idx) / 1000
    fc["wind"] = fc["wind_onshore"] + fc["wind_offshore"]; fc["rl"] = fc["load"] - fc["wind"] - fc["solar"]; return fc
def vint_rows(P15, d0, kind):
    """dense weather rows d0..d0+HZ as forecast earlier. kind 'opt': 1 day earlier for the first 14 h, then 2; 'pes': 2 then 3"""
    k1, k2 = (1, 2) if kind == "opt" else (2, 3); lead = np.arange(lc.HZ) / 4.0; out = {}
    for c in [c for c in P15.columns if c.endswith("_d1")]:
        b = c[:-3]; a1 = P15[f"{b}_d{k1}"].to_numpy()[d0:d0 + lc.HZ]; a2 = P15[f"{b}_d{k2}"].to_numpy()[d0:d0 + lc.HZ]; out[b] = np.where(lead <= 14, a1, a2)
    return pd.DataFrame(out, index=P15.index[d0:d0 + lc.HZ])
