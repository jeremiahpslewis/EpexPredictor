"""Stage-1 DE: do the trees gain from the most recent known prices? Training rows are stacked for 1, 2 and 3 days of
distance to the last known day, with that day's prices as inputs. usage: lag_test.py NAME START END STRIDE"""
import sys
from multiprocessing import get_context
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
S, H, L = 96, 289, 540 * 96
G = {}
def init():
    import os; os.environ["OMP_NUM_THREADS"] = "1"
    df = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet"); cols = [c for c in df.columns if c != "price"]
    G["X"] = df[cols].to_numpy(float); G["y"] = df["price"].to_numpy(float); G["gas"] = cols.index("gasprice"); G["index"] = df.index
def lagfeat(y, rows, k):
    """inputs from the last known UTC day, which ended k days (k = 1..3, scalar or per row) before the day of each row"""
    k = np.broadcast_to(np.asarray(k), rows.shape)
    end = (rows // S) * S - (k - 1) * S          # first unknown slot (the frames start at 00:00 UTC)
    v = np.nan_to_num(y); cs = np.concatenate([[0.0], np.cumsum(v)]); cn = np.concatenate([[0.0], np.cumsum(~np.isnan(y))])
    def m(n):
        lo = end - n; okk = lo >= 0; lo = np.clip(lo, 0, None); e = np.clip(end, 0, None)
        cnt = cn[e] - cn[lo]
        return np.where(okk & (cnt > 0), (cs[e] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    same = np.where(rows - k * S >= 0, y[np.clip(rows - k * S, 0, None)], np.nan)
    last = np.where(end >= 1, y[np.clip(end - 1, 0, None)], np.nan)
    return np.column_stack([k, last, m(4), m(24), m(96), m(7 * 96), same, (rows - end) / 4.0])
def fit(X, y, Xp, w=None):
    import lightgbm as lgb
    return lgb.train({"force_col_wise": True, "verbosity": -1, "num_threads": 1}, lgb.Dataset(X, label=y, weight=w)).predict(Xp)
def run(d0):
    X, y = G["X"], G["y"]; g = G["gas"]; rows = np.arange(d0 - L, d0); te = np.arange(d0, d0 + H)
    Xte = X[te].copy(); Xte[:, g] = X[d0, g]
    ok = ~(np.isnan(X[rows]).any(axis=1) | np.isnan(y[rows]))
    out = {"d0": d0, "actual": y[te]}
    raw = fit(X[rows][ok], y[rows][ok], Xte); rel = fit(X[rows][ok], y[rows][ok] / X[rows][ok][:, g], Xte, w=X[rows][ok][:, g] ** 2) * Xte[:, g]
    out["blend"] = (raw + rel) / 2
    # stacked training set with recent-price inputs
    A = np.vstack([np.column_stack([X[rows], lagfeat(y, rows, k)]) for k in (1, 2, 3)]); t = np.tile(y[rows], 3)
    m = ~(np.isnan(A).any(axis=1) | np.isnan(t))
    kte = np.minimum((te - d0) // S + 1, 3)
    B = np.column_stack([Xte, lagfeat(np.where(np.arange(len(y)) < d0, y, np.nan), te, kte)])
    raw2 = fit(A[m], t[m], B); rel2 = fit(A[m], t[m] / A[m][:, g], B, w=A[m][:, g] ** 2) * B[:, g]
    out["lag_blend"] = (raw2 + rel2) / 2
    return out
if __name__ == "__main__":
    name, start, end, stride = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    init(); idx = G["index"]; cuts = []; d = pd.Timestamp(start, tz="UTC"); e = pd.Timestamp(end, tz="UTC")
    while d < e - pd.Timedelta(days=3): cuts.append(idx.get_loc(d)); d += pd.Timedelta(days=stride)
    with get_context("spawn").Pool(4, initializer=init) as pool: res = pool.map(run, cuts, chunksize=1)
    res.sort(key=lambda r: r["d0"])
    np.savez_compressed(f"{SP}/out/{name}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **{k: np.stack([r[k] for r in res]).astype(np.float32) for k in res[0] if k != "d0"})
    print(name, len(cuts), "cutoffs", flush=True)
