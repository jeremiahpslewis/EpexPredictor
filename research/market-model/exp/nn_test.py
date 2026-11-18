"""Stage-1 DE: a small neural network (2 hidden layers) on the same inputs, same rolling-origin setup.
usage: nn_test.py NAME START END STRIDE"""
import sys, warnings
from multiprocessing import get_context
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
SLOTS, H, L = 96, 289, 540 * 96
G = {}
def init():
    import os; os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
    df = pd.read_parquet(f"{SP}/frames_real/frame_DE.parquet")
    cols = [c for c in df.columns if c != "price"]
    X = df[cols].copy()
    # cyclic encodings, which a network needs and trees do not
    X["slot_sin"] = np.sin(2 * np.pi * X["slot"] / 96); X["slot_cos"] = np.cos(2 * np.pi * X["slot"] / 96)
    for k in range(7): X[f"wd{k}"] = (X["weekday"] == k).astype(float)
    X = X.drop(columns=["ts"])
    G["X"] = X.to_numpy(float); G["y"] = df["price"].to_numpy(float); G["gas"] = list(X.columns).index("gasprice"); G["index"] = df.index
def mlp(X, y, Xp, seed=0):
    from sklearn.neural_network import MLPRegressor
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512, learning_rate_init=2e-3, max_iter=40, early_stopping=True,
                     validation_fraction=0.1, n_iter_no_change=5, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); m.fit((X - mu) / sd, y)
    return m.predict((Xp - mu) / sd)
def run(d0):
    X, y = G["X"], G["y"]; tr = slice(d0 - L, d0); te = slice(d0, d0 + H)
    Xtr, ytr, Xte = X[tr], y[tr], X[te].copy(); g = G["gas"]; Xte[:, g] = X[d0, g]
    ok = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr)); gtr, gte = Xtr[:, g], Xte[:, g]
    out = {"d0": d0}
    out["nn_raw"] = mlp(Xtr[ok], ytr[ok], Xte)
    out["nn_rel"] = mlp(Xtr[ok], ytr[ok] / gtr[ok], Xte) * gte
    out["nn_blend"] = (out["nn_raw"] + out["nn_rel"]) / 2
    out["actual"] = y[te]
    return out
if __name__ == "__main__":
    name, start, end, stride = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    init(); idx = G["index"]; cuts = []; d = pd.Timestamp(start, tz="UTC"); e = pd.Timestamp(end, tz="UTC")
    while d < e - pd.Timedelta(days=3): cuts.append(idx.get_loc(d)); d += pd.Timedelta(days=stride)
    with get_context("spawn").Pool(4, initializer=init) as pool: res = pool.map(run, cuts, chunksize=1)
    res.sort(key=lambda r: r["d0"]); keys = [k for k in res[0] if k != "d0"]
    np.savez_compressed(f"{SP}/out/{name}.npz", cutoffs=idx[cuts].tz_convert(None).to_numpy(), **{k: np.stack([r[k] for r in res]).astype(np.float32) for k in keys})
    print(name, len(cuts), "cutoffs", flush=True)
