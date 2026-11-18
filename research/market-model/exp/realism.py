"""
How much do real weather-forecast errors cost? The repo's backtest feeds the model archived (near-analysis)
weather for the forecast days. Here the model is trained as usual, but the forecast rows get weather as it was
predicted k days earlier (Open-Meteo previous-runs archive, hourly, interpolated to 15 min).

Variants written (evaluate.py-compatible, key s1_DE):
  A  archive 15-min weather (what the backtest does)
  B  latest-run hourly weather, interpolated (isolates the hourly->15min effect)
  C  lead-matched, fresh bound: target day h uses the run from h-1 days earlier
  D  lead-matched, stale bound: target day h uses the run from h days earlier
"""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

import harness

FRAMES, PREV, OUT, TAG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
LEARN_DAYS, STRIDE = int(sys.argv[5]), int(sys.argv[6])
SPEC = json.loads(sys.argv[7])
GAS_LAG = int(sys.argv[8])
START, END = sys.argv[9], sys.argv[10]
S = 96
W = {}


def init():
    harness.init_worker(FRAMES, "", ["DE"], GAS_LAG)
    cols = harness.G["cols"]["DE"]
    index = harness.G["index"]
    prev = pd.read_parquet(PREV)
    lead = {}
    for k in range(4):
        sub = prev[[c for c in prev.columns if c.endswith(f"_d{k}")]].copy()
        sub.columns = [c[:-3] for c in sub.columns]
        irr = [c for c in sub.columns if c.startswith("irradiance_")]
        other = [c for c in sub.columns if c not in irr]
        grid = pd.date_range(sub.index.min(), sub.index.max(), freq="15min")
        a = sub[other].reindex(grid).interpolate(method="time")
        # hourly radiation is the mean of the preceding hour -> centre it before interpolating
        b = sub[irr].copy()
        b.index = b.index - pd.Timedelta(minutes=30)
        b = b.reindex(b.index.union(grid)).interpolate(method="time").reindex(grid).clip(lower=0)
        full = pd.concat([a, b], axis=1).reindex(index)
        lead[k] = full
    W["lead"] = {k: v.to_numpy(dtype=np.float64) for k, v in lead.items()}
    W["wcols"] = [cols.index(c) for c in lead[0].columns]


def run(d0):
    G = harness.G
    X, y = G["X"]["DE"], G["y"]["DE"]
    L = LEARN_DAYS * S
    tr0 = max(d0 - L, 0)
    Xtr, ytr = X[tr0:d0], y[tr0:d0]
    valid = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
    age = (np.arange(d0 - tr0)[::-1] // S).astype(np.float64)
    te = slice(d0, d0 + 3 * S + 1)
    base = X[te].copy()
    for gi in G["gascols"]["DE"]:
        base[:, gi] = X[d0, gi]
    variants = {"A": base}
    wc = W["wcols"]

    def with_leads(leads):  # leads: lead index per target day (1..3); boundary slot follows day 3
        Z = base.copy()
        for h, k in enumerate(leads):
            a, b = h * S, (h + 1) * S + (1 if h == 2 else 0)
            Z[np.ix_(np.arange(a, b), wc)] = W["lead"][k][d0 + a:d0 + b]
        return Z

    variants["B"] = with_leads([0, 0, 0])
    variants["C"] = with_leads([0, 1, 2])
    variants["D"] = with_leads([1, 2, 3])
    g = harness.ratio_col("DE", SPEC)
    names = list(variants)
    preds = harness.fit_predict(Xtr[valid], ytr[valid], [variants[n] for n in names], SPEC, age[valid], None,
                                Xtr[valid][:, g] if g is not None else None,
                                [variants[n][:, g] for n in names] if g is not None else None)
    nan_share = float(np.isnan(variants["D"][:, wc]).mean())
    return d0, {n: p.astype(np.float32) for n, p in zip(names, preds)}, nan_share


if __name__ == "__main__":
    index = pd.read_parquet(f"{FRAMES}/frame_DE.parquet", columns=["price"]).index
    y = pd.read_parquet(f"{FRAMES}/frame_DE.parquet", columns=["price"])["price"].to_numpy()
    cutoffs = []
    d = pd.Timestamp(START, tz="UTC")
    while d < pd.Timestamp(END, tz="UTC") - pd.Timedelta(days=3):
        cutoffs.append(index.get_loc(d))
        d += pd.Timedelta(days=STRIDE)
    t0 = time.time()
    with get_context("spawn").Pool(4, initializer=init) as pool:
        res = pool.map(run, cutoffs, chunksize=1)
    res.sort(key=lambda r: r[0])
    print("max NaN share in substituted weather:", max(r[2] for r in res))
    for n in "ABCD":
        name = f"real_{TAG}_{n}"
        np.savez_compressed(f"{OUT}/{name}.npz", cutoffs=index[cutoffs].tz_convert(None).to_numpy(),
                            actual_DE=np.stack([y[c:c + 3 * S + 1] for c in cutoffs]).astype(np.float32),
                            s1_DE=np.stack([r[1][n] for r in res]))
        json.dump({"name": name, "cfg": {"eval_regions": ["DE"]}, "stride": STRIDE, "n_iter": len(cutoffs),
                   "seconds": round(time.time() - t0, 1)}, open(f"{OUT}/{name}.json", "w"))
    print(f"realism {TAG}: {len(cutoffs)} cutoffs in {time.time() - t0:.0f}s")
