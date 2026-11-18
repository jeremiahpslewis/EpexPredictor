"""Ablation ladder for the German tree model, live setup autumn: which inputs carry the forecast.
Single-stage trees (price model + gas-relative model averaged), trained on TSO residual load; test rows with the own
residual-load forecast ("own") or the published one as if known ("known")."""
import sys, os
from multiprocessing import get_context
import numpy as np, pandas as pd
import live_common as lc, dense_common as dc
G = {}
CAL = ["holiday", "weekday", "sunelevation", "azimuth", "sr_influence", "ss_influence", "slot", "ts"]; EU = ["eu_wind", "eu_wind_northsea", "eu_irradiance", "eu_temperature"]
def init():
    import os; os.environ["OMP_NUM_THREADS"] = "1"
    F = lc.load(); G["V"] = lc.vintage(F); G["cols"] = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}
    F["DE"] = F["DE"].copy(); F["DE"]["rl"] = dc.tso(F["DE"].index)["rl"]
    nb = pd.read_parquet(f"{lc.SP}/out/nb_rl.parquet"); G["nb"] = [c for c in nb.columns if c.startswith("rl_")]
    for c in G["nb"]: F["DE"][c] = nb[c].to_numpy()
    F["DE"]["rl_region"] = F["DE"]["rl"] + nb[G["nb"]].sum(axis=1).to_numpy(); G["F"] = F
    g = np.load(f"{lc.SP}/out/inputs_{sys.argv[1]}.npz"); G["gen"] = {pd.Timestamp(c, tz="UTC"): ((g["rl_pes"][i] if "rl_pes" in g.files else g["rl_true"][i]), g["rl_true"][i]) for i, c in enumerate(g["cutoffs"])}
    W = [c for c in G["cols"]["DE"] if c.split("_")[0] in lc.VARS]
    G["sets"] = {"A_rl_gas_cal": ["rl", "gasprice"] + CAL, "B_plus_eu": ["rl", "gasprice"] + CAL + EU, "C_plus_de_weather": ["rl", "gasprice"] + CAL + EU + W,
                 "N_no_rl": ["gasprice"] + CAL + EU + W, "Z_rl_gas_clock": ["rl", "gasprice", "slot", "weekday", "holiday"], "Y_rl_gas_clock_ts": ["rl", "gasprice", "slot", "weekday", "holiday", "ts"],
                 "X_rl_gas_clock_sun": ["rl", "gasprice", "slot", "weekday", "holiday", "sunelevation", "azimuth", "sr_influence", "ss_influence"]}
    G["sets"]["D_plus_neighbours"] = ["rl", "gasprice"] + CAL + EU + G["nb"]; G["sets"]["E_plus_region_sum"] = ["rl", "gasprice"] + CAL + EU + ["rl_region"]
    G["cols"]["DE"] = [c for c in G["cols"]["DE"] if c not in G["nb"] + ["rl", "rl_region"]]
    if os.environ.get("SETS"): G["sets"] = {k: v for k, v in G["sets"].items() if k[0] in os.environ["SETS"]}
def fit(X, y, Xp, w=None):
    import lightgbm as lgb
    return lgb.train({"force_col_wise": True, "verbosity": -1, "num_threads": 1}, lgb.Dataset(X, label=y, weight=w)).predict(Xp)
def run(d0):
    F, V = G["F"], G["V"]; D = F["DE"]; out = {"d0": d0}
    Wte = {z: lc.test_matrix(F, V, z, d0, G["cols"][z], True) for z in lc.ZONES}; eu = lc.eu_feats(Wte); T = Wte["DE"].copy()
    for c in eu.columns: T[c] = eu[c].to_numpy()
    T["gasprice"] = D["gasprice"].to_numpy()[d0 - lc.ISSUE]; pes, tru = G["gen"][D.index[d0]]; ytr = D["price"].to_numpy(float)[d0 - lc.L:d0]
    for c in G["nb"] + ["rl_region"]: T[c] = D[c].to_numpy()[d0:d0 + lc.HZ]          # neighbours' published figures as if known
    for name, cols in G["sets"].items():
        Xtr = D[cols].to_numpy(float)[d0 - lc.L:d0]; ok = ~(np.isnan(Xtr).any(1) | np.isnan(ytr))
        for arm, rl in (("own", pes), ("known", tru)):
            if "rl" not in cols and arm == "known": continue
            Tt = T.copy(); Tt["rl"] = rl
            if "rl_region" in cols: Tt["rl_region"] = Tt["rl_region"] - D["rl"].to_numpy()[d0:d0 + lc.HZ] + rl
            Xte = Tt[cols].to_numpy(float); p = fit(Xtr[ok], ytr[ok], Xte)
            if "gasprice" in cols:
                g = cols.index("gasprice"); p = (p + fit(Xtr[ok], ytr[ok] / Xtr[ok][:, g], Xte, w=Xtr[ok][:, g] ** 2) * Xte[:, g]) / 2
            out[f"{name}__{arm}"] = p
    return out
if __name__ == "__main__":
    idx = lc.load()["DE"].index; cuts = [int(c) for c in np.load(f"{lc.SP}/out/inputs_{sys.argv[1]}.npz")["cuts"]]
    with get_context("spawn").Pool(4, initializer=init) as pool: res = pool.map(run, cuts, chunksize=1)
    res.sort(key=lambda r: r["d0"])
    np.savez_compressed(f"{lc.SP}/out/ladder_{sys.argv[1]}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **{k: np.stack([r[k] for r in res]).astype(np.float32) for k in res[0] if k != "d0"})
    print("ladder done", flush=True)
