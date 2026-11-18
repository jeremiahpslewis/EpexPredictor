"""Live-setup backtest of the PR's two-stage model for DE. usage: live_test.py NAME START END STRIDE (dates = local day D+2)"""
import sys
from multiprocessing import get_context
import numpy as np, pandas as pd
import live_common as lc
G = {}
def init():
    import os; os.environ["OMP_NUM_THREADS"] = "1"
    G["F"] = lc.load(); G["V"] = lc.vintage(G["F"])
    G["cols"] = {z: [c for c in G["F"][z].columns if c != "price"] for z in lc.ZONES}
def fit(X, y, Xp, w=None):
    import lightgbm as lgb
    m = lgb.train({"force_col_wise": True, "verbosity": -1, "num_threads": 1}, lgb.Dataset(X, label=y, weight=w))
    return [m.predict(p) for p in Xp]
def run(d0):
    F, V = G["F"], G["V"]; out = {"d0": d0, "actual": F["DE"]["price"].to_numpy()[d0:d0 + lc.HZ]}
    for tag, vint in (("arch", False), ("vint", True)):
        Wte = {z: lc.test_matrix(F, V, z, d0, G["cols"][z], vint) for z in lc.ZONES}
        if vint:
            eu = lc.eu_feats(Wte)
            for z in lc.ZONES:
                for c in eu.columns: Wte[z][c] = eu[c].to_numpy()
        s1tr, s1te, own = {}, {}, {}
        for z in lc.ZONES:
            cols = G["cols"][z]; Xtr = F[z][cols].to_numpy(float)[d0 - lc.L:d0]; ytr = F[z]["price"].to_numpy(float)[d0 - lc.L:d0]; Xte = Wte[z].to_numpy(float)
            ok = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr)); g = cols.index("gasprice") if "gasprice" in cols else None
            if g is not None: Xte[:, g] = F[z]["gasprice"].to_numpy()[d0 - lc.ISSUE]          # gas price as known at the issue time
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
        out[tag] = (raw + rel * Xte[:, g]) / 2
        if np.isnan(Xte).any(): out[tag][np.isnan(Xte).any(axis=1)] = np.nan
    return out
if __name__ == "__main__":
    name, start, end, stride = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    idx = lc.load()["DE"].index; cuts = lc.cutoffs(idx, start, end, stride)
    with get_context("spawn").Pool(4, initializer=init) as pool: res = pool.map(run, cuts, chunksize=1)
    res.sort(key=lambda r: r["d0"])
    np.savez_compressed(f"{lc.SP}/out/{name}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **{k: np.stack([r[k] for r in res]).astype(np.float32) for k in res[0] if k != "d0"})
    print(name, len(cuts), "cutoffs", flush=True)
