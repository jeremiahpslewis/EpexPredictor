"""live_test.py with extra German inputs. usage: live_test2.py NAME START END STRIDE MODE
MODE: A  TSO day-ahead residual load + components, as if known for the forecast days (upper bound)
      B  A + within-day statistics of residual load (local day max/mean/min, deviation from day mean, rank in day)
      C  residual load predicted by an own model from the weather forecast (usable live) + within-day statistics
      C0 as C without the within-day statistics"""
import sys, os
from multiprocessing import get_context
import numpy as np, pandas as pd
import live_common as lc
G = {}
DAYCOLS = ["rl_daymax", "rl_daymean", "rl_daymin", "rl_dev", "rl_rank"]
def daystats(rl, index):
    d = index.tz_convert("Europe/Berlin").normalize(); g = pd.Series(np.asarray(rl, float), index=index).groupby(d)
    s = pd.DataFrame({"rl_daymax": g.transform("max"), "rl_daymean": g.transform("mean"), "rl_daymin": g.transform("min")})
    s["rl_dev"] = s.index.map(lambda _: 0.0); s["rl_dev"] = np.asarray(rl, float) - s["rl_daymean"].to_numpy(); s["rl_rank"] = g.rank(pct=True).to_numpy()
    return s[DAYCOLS]
def init(mode):
    import os; os.environ["OMP_NUM_THREADS"] = "1"
    F = lc.load(); G["V"] = lc.vintage(F); G["mode"] = mode
    G["base"] = {z: [c for c in F[z].columns if c != "price"] for z in lc.ZONES}
    fc = pd.concat([pd.read_parquet(f"{lc.SP}/ecfc/{p}.parquet") for p in ("wind_onshore", "wind_offshore", "solar", "load")], axis=1).reindex(F["DE"].index) / 1000
    D = F["DE"].copy(); D["rl"] = fc["load"] - fc["wind_onshore"] - fc["wind_offshore"] - fc["solar"]; extra = ["rl"]
    if mode in ("A", "B"):
        for c in fc.columns: D["fc_" + c] = fc[c]
        extra += ["fc_" + c for c in fc.columns]
    if mode in ("B", "C"):
        D = pd.concat([D, daystats(D["rl"], D.index)], axis=1); extra += DAYCOLS
    if mode.startswith("D"):
        _, kind, tr = mode.split(":"); G["kind"] = kind; g = np.load(f"{lc.SP}/out/" + os.environ.get("GENFILE", "gen_fc_app") + ".npz"); G["gen"] = {pd.Timestamp(c, tz="UTC"): g[f"rl_{kind}"][i] for i, c in enumerate(g["cutoffs"])}
        if tr == "O": D["rl"] = pd.read_parquet(f"{lc.SP}/dense/oof.parquet")["rl"].reindex(D.index)
    F["DE"] = D; G["F"] = F; G["cols"] = dict(G["base"]); G["cols"]["DE"] = G["base"]["DE"] + extra; G["extra"] = extra
def fit(X, y, Xp, w=None):
    import lightgbm as lgb
    m = lgb.train({"force_col_wise": True, "verbosity": -1, "num_threads": 1}, lgb.Dataset(X, label=y, weight=w))
    return [m.predict(p) for p in Xp]
def run(d0):
    F, V, mode = G["F"], G["V"], G["mode"]; out = {"d0": d0, "actual": F["DE"]["price"].to_numpy()[d0:d0 + lc.HZ]}
    Wte = {z: lc.test_matrix(F, V, z, d0, G["cols"][z], False) for z in lc.ZONES}
    eu = lc.eu_feats(Wte)
    for z in lc.ZONES:
        for c in eu.columns: Wte[z][c] = eu[c].to_numpy()
    out["rl_true"] = Wte["DE"]["rl"].to_numpy(float).copy()
    if mode in ("C", "C0"):
        b = G["base"]["DE"]; g = b.index("gasprice"); Xtr = F["DE"][b].to_numpy(float)[d0 - lc.L:d0]; ytr = F["DE"]["rl"].to_numpy(float)[d0 - lc.L:d0]; ok = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        Xte = Wte["DE"][b].to_numpy(float); Xte[:, g] = F["DE"]["gasprice"].to_numpy()[d0 - lc.ISSUE]
        (rl,) = fit(Xtr[ok], ytr[ok], [Xte]); Wte["DE"]["rl"] = rl
        if mode == "C":
            st = daystats(rl, Wte["DE"].index)
            for c in DAYCOLS: Wte["DE"][c] = st[c].to_numpy()
    if mode.startswith("D"): Wte["DE"]["rl"] = G["gen"][F["DE"].index[d0]].astype(float)
    out["rl_used"] = Wte["DE"]["rl"].to_numpy(float).copy()
    s1tr, s1te, own = {}, {}, {}
    D1 = os.environ.get("D1", ""); te_ = d0 - 96 if D1 == "drop" else d0
    for z in lc.ZONES:
        cols = G["cols"][z]; Xtr = F[z][cols].to_numpy(float)[max(te_ - lc.L, 0):te_]; ytr = F[z]["price"].to_numpy(float)[max(te_ - lc.L, 0):te_]; Xte = Wte[z][cols].to_numpy(float)
        if z == "DE" and D1 not in ("", "drop") and "rl" in cols:
            if "d1" not in G: g1 = np.load(f"{lc.SP}/out/{D1}.npz"); G["d1"] = {pd.Timestamp(c, tz="UTC"): g1["rl_d1"][i] for i, c in enumerate(g1["cutoffs"])}
            Xtr = Xtr.copy(); Xtr[-96:, cols.index("rl")] = G["d1"][F["DE"].index[d0]]
        ok = ~(np.isnan(Xtr[:, :len(G["base"][z])]).any(axis=1) | np.isnan(ytr)); g = cols.index("gasprice") if "gasprice" in cols else None
        if g is not None: Xte[:, g] = F[z]["gasprice"].to_numpy()[d0 - lc.ISSUE]
        ptr, pte = fit(Xtr[ok], ytr[ok], [Xtr, Xte])
        if g is not None:
            rtr, rte = fit(Xtr[ok], ytr[ok] / Xtr[ok][:, g], [Xtr, Xte], w=Xtr[ok][:, g] ** 2)
            ptr, pte = (ptr + rtr * Xtr[:, g]) / 2, (pte + rte * Xte[:, g]) / 2
        s1tr[z], s1te[z], own[z] = ptr, pte, (Xtr, ytr, ok, Xte, g)
    Xtr, ytr, ok, Xte, g = own["DE"]
    ctr = np.column_stack([s1tr[z] for z in lc.ZONES]); cte = np.column_stack([s1te[z] for z in lc.ZONES])
    A, B = np.column_stack([Xtr, ctr]), np.column_stack([Xte, cte]); n = Xtr.shape[1]
    (raw,) = fit(A[ok], ytr[ok], [B])
    A2, B2 = A.copy(), B.copy(); A2[:, n:] /= Xtr[:, g][:, None]; B2[:, n:] /= Xte[:, g][:, None]
    (rel,) = fit(A2[ok], ytr[ok] / Xtr[ok][:, g], [B2], w=Xtr[ok][:, g] ** 2)
    out["vint"] = (raw + rel * Xte[:, g]) / 2; out["ntrain"] = np.array([ok.sum()] * lc.HZ)
    return out
if __name__ == "__main__":
    name, start, end, stride, mode = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    idx = lc.load()["DE"].index; cuts = lc.cutoffs(idx, start, end, stride)
    with get_context("spawn").Pool(4, initializer=init, initargs=(mode,)) as pool: res = pool.map(run, cuts, chunksize=1)
    res.sort(key=lambda r: r["d0"])
    np.savez_compressed(f"{lc.SP}/out/{name}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **{k: np.stack([r[k] for r in res]).astype(np.float32) for k in res[0] if k != "d0"})
    print(name, len(cuts), "cutoffs", flush=True)
