"""Stage-1 DE: long-term level (linear) + short-term deviation (trees), against the PR's two-model average.
Same rolling-origin setup as harness.py. usage: level_test.py NAME START END STRIDE"""
import sys, json
from multiprocessing import get_context
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
SLOTS, H, L = 96, 289, 540 * 96
G = {}
def init():
    import os; os.environ["OMP_NUM_THREADS"] = "1"
    df = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet")
    price = df["price"]
    cols = [c for c in df.columns if c != "price"]
    G["X"] = df[cols].to_numpy(float); G["y"] = price.to_numpy(float); G["gas"] = cols.index("gasprice")
    # long-term price level as known at each moment: mean of the last 28 / 90 days
    G["lvl28"] = price.rolling(28 * SLOTS, min_periods=7 * SLOTS).mean().shift(1).to_numpy(float)
    G["lvl90"] = price.rolling(90 * SLOTS, min_periods=28 * SLOTS).mean().shift(1).to_numpy(float)
    G["slot"] = df["slot"].to_numpy(int); G["index"] = df.index
def fit(X, y, Xp, w=None, **kw):
    import lightgbm as lgb
    p = {"force_col_wise": True, "verbosity": -1, "num_threads": 1}; p.update(kw)
    return lgb.train(p, lgb.Dataset(X, label=y, weight=w)).predict(Xp)
def run(d0):
    X, y = G["X"], G["y"]; tr = slice(d0 - L, d0); te = slice(d0, d0 + H)
    Xtr, ytr, Xte = X[tr], y[tr], X[te].copy()
    g = G["gas"]; Xte[:, g] = X[d0, g]
    ok = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
    gtr, gte = Xtr[:, g], Xte[:, g]
    out = {"d0": d0}
    raw = fit(Xtr[ok], ytr[ok], Xte)
    rel = fit(Xtr[ok], ytr[ok] / gtr[ok], Xte, w=gtr[ok] ** 2) * gte
    out["raw"], out["blend"] = raw, (raw + rel) / 2
    out["linear_tree"] = fit(Xtr[ok], ytr[ok], Xte, linear_tree=True, linear_lambda=1.0)
    # level = linear model of the price on slow inputs, trees learn the deviation from it
    def level_model(feats_tr, feats_te, name, mode):
        A = np.column_stack([np.ones(len(ytr))] + feats_tr); B = np.column_stack([np.ones(H)] + feats_te)
        m = ok & ~np.isnan(A).any(axis=1)
        coef, *_ = np.linalg.lstsq(A[m], ytr[m], rcond=None)
        ltr, lte = A @ coef, B @ coef
        if mode == "add":
            out[name] = lte + fit(Xtr[m], ytr[m] - ltr[m], Xte)
        else:
            ltr = np.clip(ltr, 1.0, None); lte = np.clip(lte, 1.0, None)
            out[name] = lte * fit(Xtr[m], ytr[m] / ltr[m], Xte, w=ltr[m] ** 2)
    l28tr, l28te = G["lvl28"][tr], np.full(H, G["lvl28"][d0])
    l90tr, l90te = G["lvl90"][tr], np.full(H, G["lvl90"][d0])
    level_model([gtr], [gte], "lvl_gas_add", "add")
    level_model([gtr], [gte], "lvl_gas_mul", "mul")
    level_model([l28tr], [l28te], "lvl_p28_add", "add")
    level_model([l28tr], [l28te], "lvl_p28_mul", "mul")
    level_model([gtr, l28tr], [gte, l28te], "lvl_gas_p28_add", "add")
    level_model([gtr, l28tr], [gte, l28te], "lvl_gas_p28_mul", "mul")
    level_model([gtr, l90tr], [gte, l90te], "lvl_gas_p90_mul", "mul")
    out["blend_raw_lvl_gas_p28_mul"] = (raw + out["lvl_gas_p28_mul"]) / 2
    out["actual"] = y[te]
    return out
if __name__ == "__main__":
    name, start, end, stride = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    init(); idx = G["index"]; cuts = []; d = pd.Timestamp(start, tz="UTC"); e = pd.Timestamp(end, tz="UTC")
    while d < e - pd.Timedelta(days=3): cuts.append(idx.get_loc(d)); d += pd.Timedelta(days=stride)
    with get_context("spawn").Pool(4, initializer=init) as pool: res = pool.map(run, cuts, chunksize=1)
    res.sort(key=lambda r: r["d0"])
    keys = [k for k in res[0] if k != "d0"]
    np.savez_compressed(f"{SP}/out/{name}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **{k: np.stack([r[k] for r in res]).astype(np.float32) for k in keys})
    print(name, len(cuts), "cutoffs", flush=True)
