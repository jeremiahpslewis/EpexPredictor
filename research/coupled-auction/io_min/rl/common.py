"""Shared pieces for the residual-load (RL) forecast experiments (DE_LU).

Target: the TSO day-ahead figures (energy-charts) for wind (onshore + offshore), solar and load; RL = load - wind - solar.
Issue: the day-ahead gate, 12:00 local on D-1. Forecast rows: d0 = local midnight of D, 153 quarter hours (as the research
autumn set: day 1 = first 96, night = first 24, "h24-38" = 96..152).

Weather vintages on the dense 58-point grid (Open-Meteo previous-runs API):
  d0 = latest run for that hour ("archive"; hist.parquet, no suffix), d1/d2/d3 = run made about 1/2/3 days earlier.
Gate vintage map (run latency about 3 h, issue 12:00 local): d1 exists for the first ~8 hours of D (32 qh), d2 after.
Brackets: all-d1 (optimistic) and all-d2 (strict).
Features = an exact port of predictor/model/residualload.generation_features (app feature set).
"""
import glob, json
import numpy as np, pandas as pd, lightgbm as lgb

SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
OUT = f"{SP}/out/rl"
HZ = 153
GATE_D1 = 32                      # quarter hours of D covered by the 1-day-old run at the gate
PTS = json.load(open(f"{SP}/dense/points.json")); NON = PTS["onshore"]; LAT = np.array(PTS["points"])[:, 0]; LON = np.array(PTS["points"])[:, 1]
NPT = len(LAT)
LGB = {"force_col_wise": True, "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31, "num_threads": 4}
ROUNDS = 400
SEG = {"night": slice(0, 24), "day1": slice(0, 96), "h24_38": slice(96, 153)}


def to15(df, idx):
    return df.reindex(df.index.union(idx)).interpolate(method="time", limit=3, limit_area="inside").reindex(idx)


def load_all():
    D = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet")
    idx = D.index
    aux = D[["ts", "slot", "weekday", "holiday", "sunelevation", "azimuth"]].astype(float)
    fc = pd.concat([pd.read_parquet(f"{SP}/ecfc/{p}.parquet") for p in ("wind_onshore", "wind_offshore", "solar", "load")], axis=1).reindex(idx) / 1000
    fc["wind"] = fc["wind_onshore"] + fc["wind_offshore"]; fc["rl"] = fc["load"] - fc["wind"] - fc["solar"]
    H = pd.read_parquet(f"{SP}/dense/hist.parquet"); P = pd.read_parquet(f"{SP}/dense/prev.parquet")
    H15 = to15(H, idx); P15 = to15(P, idx)
    W = {"d0": H15[[c for c in H15.columns if not c.endswith("_d2")]],
         "d2h": H15[[c for c in H15.columns if c.endswith("_d2")]].rename(columns=lambda c: c[:-3])}
    for k in (1, 2, 3):
        W[f"d{k}"] = P15[[c for c in P15.columns if c.endswith(f"_d{k}")]].rename(columns=lambda c: c[:-3])
    I = np.load(f"{SP}/out/inputs_autumn.npz", allow_pickle=True)
    return dict(idx=idx, aux=aux, fc=fc, W=W, cuts=I["cuts"].astype(int), cutoffs=I["cutoffs"], I=I)


def feats(W, aux, w80=False):
    """App features (residualload.generation_features). W: columns w120_i, sw_i, t2_i (and w80_i), rows like aux."""
    onshore = np.arange(NPT) < NON; north = onshore & (LAT >= 52.0); south = onshore & ~north
    w = W[[f"w120_{i}" for i in range(NPT)]].to_numpy(float); rad = W[[f"sw_{i}" for i in range(NPT)]].to_numpy(float)
    t = W[[f"t2_{i}" for i in range(NPT)]].to_numpy(float)
    pw = np.clip(w / 43.2, 0, 1) ** 3; pl = np.clip((w - 10.8) / (43.2 - 10.8), 0, 1)
    F = {f"wind_power_{i}": pw[:, i] for i in range(NPT)}
    F["wind_power_onshore"] = pw[:, onshore].mean(1); F["wind_power_north"] = pw[:, north].mean(1); F["wind_power_south"] = pw[:, south].mean(1)
    F["wind_power_offshore"] = pw[:, ~onshore].mean(1); F["wind_power_linear"] = pl[:, onshore].mean(1); F["wind_speed_max"] = w[:, onshore].max(1)
    F.update({f"radiation_{i}": rad[:, i] for i in np.flatnonzero(onshore)})
    F["radiation_all"] = rad[:, onshore].mean(1); F["radiation_south"] = rad[:, south].mean(1)
    F["temp_all"] = t[:, onshore].mean(1); F["temp_south"] = t[:, south].mean(1)
    if w80:
        w8 = W[[f"w80_{i}" for i in range(NPT)]].to_numpy(float); p8 = np.clip(w8 / 43.2, 0, 1) ** 3
        F.update({f"w80_power_{i}": p8[:, i] for i in range(NPT)})
        F["w80_onshore"] = p8[:, onshore].mean(1); F["w80_offshore"] = p8[:, ~onshore].mean(1)
        F["shear"] = (w[:, onshore] / np.maximum(w8[:, onshore], 1.0)).mean(1)
    for c in ("ts", "slot", "weekday", "holiday", "sunelevation", "azimuth"):
        F[c] = aux[c].to_numpy(float)
    return pd.DataFrame(F, index=W.index)


def gate_rows(Wv, d0, kind="gate"):
    """Weather rows d0..d0+HZ as known at the gate. kind: gate (d1 first GATE_D1 qh, then d2), d1, d2, d3, d0"""
    if kind == "gate":
        a = Wv["d1"].iloc[d0:d0 + HZ].to_numpy(); b = Wv["d2"].iloc[d0:d0 + HZ].to_numpy()
        m = (np.arange(HZ) < GATE_D1)[:, None]
        return pd.DataFrame(np.where(m, a, b), index=Wv["d1"].index[d0:d0 + HZ], columns=Wv["d1"].columns)
    return Wv[kind].iloc[d0:d0 + HZ]


def fit(X, y, w=None, params=None, rounds=ROUNDS):
    X = np.asarray(X, float); y = np.asarray(y, float)
    ok = ~(np.isnan(X).any(1) | np.isnan(y))
    return lgb.train(params or LGB, lgb.Dataset(X[ok], label=y[ok], weight=None if w is None else np.asarray(w)[ok]), num_boost_round=rounds)


def mae_seg(f, t):
    e = np.abs(f - t); return {k: float(np.nanmean(e[:, s])) for k, s in SEG.items()}


def boot(cand, base, true, seg, reps=2000, block=7, seed=0):
    """paired block bootstrap by day of MAE(cand) - MAE(base) on segment seg; returns (diff, lo5, hi95)"""
    s = SEG[seg]; dc = np.nanmean(np.abs(cand - true)[:, s], 1) - np.nanmean(np.abs(base - true)[:, s], 1)
    n = len(dc); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); out = []
    for _ in range(reps):
        st = rng.integers(0, n - block + 1, nb); ii = np.concatenate([np.arange(a, a + block) for a in st])[:n]
        out.append(dc[ii].mean())
    return float(dc.mean()), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def local_hour(idx, d0):
    return idx[d0:d0 + HZ].tz_convert("Europe/Berlin").hour.to_numpy()
