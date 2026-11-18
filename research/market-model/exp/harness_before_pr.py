"""
Fast rolling-origin backtest that mirrors predictor/performance_testing.py:
  - per cutoff d0: train on [d0 - learn_days, d0 - 15min], predict [d0, d0 + 3d] (inclusive, 289 slots)
  - gas price for forecast rows frozen at its value at d0 (what gasstore.horizon_cutoff does)
  - stage 2 = own features + stage-1 forecasts of all cross regions
    (training rows use stage-1 *in-sample* predictions, exactly like the repo does today)
Everything else (hyper-parameters, weights, extra features, out-of-fold stacking, ...) is a variant knob.

Predictions for every cutoff are stored, so all metrics / comparisons are computed offline.
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

REGIONS = ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]
SLOTS = 96
HORIZON_ROWS = 3 * SLOTS + 1

G: dict = {}


# ----------------------------------------------------------------------------- features

def add_features(df: pd.DataFrame, featureset: str) -> pd.DataFrame:
    """Optional engineered columns, inserted before the price column. featureset = '+'-joined names."""
    if not featureset:
        return df
    price = df.pop("price")
    names = featureset.split("+")
    wind = df[[c for c in df.columns if c.startswith("wind_")]]
    irr = df[[c for c in df.columns if c.startswith("irradiance_")]]
    temp = df[[c for c in df.columns if c.startswith("temp_")]]
    if "agg" in names:
        df["wind_mean"] = wind.mean(axis=1)
        df["wind_cube_mean"] = (wind.clip(upper=45.0) ** 3).mean(axis=1)  # km/h; ~rated power above ~45 km/h
        df["irr_mean"] = irr.mean(axis=1)
        df["temp_mean"] = temp.mean(axis=1)
    if "agg2" in names:
        df["wind_min"] = wind.min(axis=1)
        df["wind_max"] = wind.max(axis=1)
        df["irr_min"] = irr.min(axis=1)
        df["irr_max"] = irr.max(axis=1)
    if "roll" in names:
        # smoothed / daily context of the weather drivers (all computable from the forecast alone)
        wm = wind.mean(axis=1)
        im = irr.mean(axis=1)
        tm = temp.mean(axis=1)
        df["wind_mean_d"] = wm.rolling(SLOTS, center=True, min_periods=SLOTS // 2).mean()
        df["irr_mean_d"] = im.rolling(SLOTS, center=True, min_periods=SLOTS // 2).mean()
        df["temp_mean_d"] = tm.rolling(SLOTS, center=True, min_periods=SLOTS // 2).mean()
        df["wind_mean_3h"] = wm.rolling(12, center=True, min_periods=6).mean()
        df["wind_ramp_1h"] = wm.diff(4)
        df["irr_ramp_1h"] = im.diff(4)
    if "daytype" in names:
        local = df.index.tz_convert("Europe/Berlin")
        hol = df["holiday"].to_numpy()
        wd = df["weekday"].to_numpy()
        # a (mostly) nation-wide holiday behaves like a Sunday
        df["daytype"] = np.where(hol >= 0.5, 6, wd)
        md = local.month * 100 + local.day
        df["xmas"] = ((md >= 1224) | (md <= 101)).astype(float)
        # bridge day: Friday after a Thursday holiday / Monday before a Tuesday holiday
        day_hol = pd.Series(hol, index=local.normalize()).groupby(level=0).max()
        prev_hol = day_hol.shift(1, freq="D").reindex(local.normalize()).to_numpy()
        next_hol = day_hol.shift(-1, freq="D").reindex(local.normalize()).to_numpy()
        prev_hol = np.nan_to_num(prev_hol)
        next_hol = np.nan_to_num(next_hol)
        df["bridge"] = (((wd == 4) & (prev_hol >= 0.5)) | ((wd == 0) & (next_hol >= 0.5))).astype(float)
    if "wdflags" in names:
        wd = df["weekday"].to_numpy()
        hol = df["holiday"].to_numpy()
        df["is_mon"] = (wd == 0).astype(float)
        df["is_fri"] = (wd == 4).astype(float)
        df["is_sat"] = (wd == 5).astype(float)
        df["is_sun"] = ((wd == 6) | (hol >= 0.5)).astype(float)
        df["is_workday"] = ((wd <= 4) & (hol < 0.5)).astype(float)
    if "wdslot" in names:
        # day type x time of day as one ordered feature: lets a single split isolate e.g. "weekend morning"
        wd = df["weekday"].to_numpy()
        hol = df["holiday"].to_numpy()
        dt = np.where((wd == 6) | (hol >= 0.5), 2, np.where(wd == 5, 1, 0))
        df["daytype_slot"] = dt * 96 + df["slot"].to_numpy()
    if "lag7" in names:
        df["price_lag7d"] = price.shift(7 * SLOTS)
    if "lag714" in names:
        df["price_lag7d"] = price.shift(7 * SLOTS)
        df["price_lag14d"] = price.shift(14 * SLOTS)
    for n in names:
        if n.startswith("drop:"):
            pref = n.split(":", 1)[1]
            df = df[[c for c in df.columns if not c.startswith(pref)]]
    df["price"] = price
    return df


def init_worker(frame_dir: str, featureset: str, regions: list[str], gas_lag: int = 0):
    os.environ["OMP_NUM_THREADS"] = "1"
    G.clear()
    G["X"], G["y"], G["gas"], G["cols"], G["gascols"] = {}, {}, {}, {}, {}
    for r in regions:
        df = pd.read_parquet(f"{frame_dir}/frame_{r}.parquet")
        df = add_features(df, featureset)
        cols = [c for c in df.columns if c != "price"]
        G["cols"][r] = cols
        gascols = [c for c in cols if c.startswith("gas")]
        if gas_lag:
            for c in gascols:
                df[c] = df[c].shift(gas_lag * SLOTS)
        G["gascols"][r] = [cols.index(c) for c in gascols]
        G["X"][r] = df[cols].to_numpy(dtype=np.float64)
        G["y"][r] = df["price"].to_numpy(dtype=np.float64)
        G["gas"][r] = cols.index("gasprice") if "gasprice" in cols else None
        G["index"] = df.index


# ----------------------------------------------------------------------------- model fitting

def fit_predict(Xtr, ytr, Xpred_list, spec: dict, age_days=None, cat=None, scale_tr=None, scale_pred=None):
    """Train one LightGBM model (or a small seed ensemble) and predict each matrix in Xpred_list.
    scale_tr / scale_pred: the model is fitted on y / scale and predictions are multiplied back."""
    import lightgbm as lgb

    params = {"force_col_wise": True, "verbosity": -1, "num_threads": 1}
    params.update(spec.get("params", {}))
    rounds = spec.get("rounds", 100)
    weight = None
    hl = spec.get("halflife")
    if hl and age_days is not None:
        weight = np.power(0.5, age_days / hl)
    transform = spec.get("transform")
    scale = spec.get("asinh_scale", 5.0)
    yfit = np.arcsinh(ytr / scale) if transform == "asinh" else ytr
    if scale_tr is not None:
        yfit = ytr / scale_tr
        if spec.get("ratio_weight"):
            # keep the objective a squared error in price units
            w2 = scale_tr ** 2
            weight = w2 if weight is None else weight * w2
    monotone = spec.get("monotone")
    if monotone is not None:
        params["monotone_constraints"] = monotone
    nseeds = spec.get("seeds", 1)
    preds = [np.zeros(len(X)) for X in Xpred_list]
    for s in range(nseeds):
        p = dict(params)
        if nseeds > 1:
            p["seed"] = 1000 + s
        booster = lgb.train(p, lgb.Dataset(Xtr, label=yfit, weight=weight, categorical_feature=cat or "auto"),
                            num_boost_round=rounds)
        for i, X in enumerate(Xpred_list):
            out = np.asarray(booster.predict(X)) if len(X) else np.zeros(0)
            if transform == "asinh":
                out = np.sinh(out) * scale
            if scale_pred is not None:
                out = out * scale_pred[i]
            preds[i] += out / nseeds
    return preds


def ratio_col(r, spec):
    """Column index of the gas series the target is divided by (None = model the raw price)."""
    gr = spec.get("gas_ratio")
    if not gr:
        return None
    name = "gasprice" if gr is True else gr
    return G["cols"][r].index(name) if name in G["cols"][r] else None


def run_iter(task):
    cfg, d0 = task
    L = cfg["learn_days"] * SLOTS
    tr0, tr1, te1 = d0 - L, d0, d0 + HORIZON_ROWS
    tr0 = max(tr0, 0)
    ntr = tr1 - tr0
    age_days = (np.arange(ntr)[::-1] // SLOTS).astype(np.float64)  # 0 for the most recent training day
    res = {"d0": d0}
    two_stage = cfg["two_stage"]
    oof = cfg.get("oof")

    s1_train, s1_test, own, cats = {}, {}, {}, {}
    for r in cfg["cross_regions"]:
        X, y = G["X"][r], G["y"][r]
        Xtr, ytr = X[tr0:tr1], y[tr0:tr1]
        valid = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        cat = [G["cols"][r].index(c) for c in cfg.get("categorical") or []]
        cats[r] = cat
        Xte = X[d0:te1].copy()
        for gi in G["gascols"][r]:
            Xte[:, gi] = X[d0, gi]  # forecast rows only know the gas price as of the cutoff
        g = ratio_col(r, cfg["s1"])
        ratio1 = g is not None
        sc_tr = Xtr[:, g] if ratio1 else None
        sc_te = Xte[:, g] if ratio1 else None
        ts_mode = cfg.get("ts_mode")
        if ts_mode:
            tsi = G["cols"][r].index("ts")
            Xtr = Xtr.copy()
            if ts_mode == "drop":
                Xtr[:, tsi] = 0.0
                Xte[:, tsi] = 0.0
            else:
                # "rel:<days>": time as whole buckets counted back from the cutoff, so the most recent
                # bucket always holds complete weeks; forecast rows sit in that most recent bucket
                bucket = int(ts_mode.split(":")[1])
                Xtr[:, tsi] = -(age_days // bucket)
                Xte[:, tsi] = 0.0
        own[r] = (Xtr, ytr, valid, Xte)
        if two_stage and not oof:
            ptr, pte = fit_predict(Xtr[valid], ytr[valid], [Xtr, Xte], cfg["s1"], age_days[valid], cat,
                                   sc_tr[valid] if ratio1 else None, [sc_tr, sc_te] if ratio1 else None)
        else:
            (pte,) = fit_predict(Xtr[valid], ytr[valid], [Xte], cfg["s1"], age_days[valid], cat,
                                 sc_tr[valid] if ratio1 else None, [sc_te] if ratio1 else None)
            ptr = None
            if two_stage and oof:
                # out-of-fold stage-1 predictions for the training rows: contiguous blocks, fold = block % k,
                # blocks counted back from the cutoff so the most recent block is always held out together
                block = (np.arange(ntr)[::-1] // (SLOTS * oof["block_days"]))
                fold = block % oof["k"]
                ptr = np.full(ntr, np.nan)
                for f in range(oof["k"]):
                    hold = fold == f
                    fit_mask = valid & ~hold
                    (ph,) = fit_predict(Xtr[fit_mask], ytr[fit_mask], [Xtr[hold]], cfg["s1"], age_days[fit_mask], cat,
                                        sc_tr[fit_mask] if ratio1 else None, [sc_tr[hold]] if ratio1 else None)
                    ptr[hold] = ph
        s1_train[r], s1_test[r] = ptr, pte
        if r in cfg["eval_regions"]:
            res[f"s1_{r}"] = pte.astype(np.float32)

    if two_stage:
        cross_tr = np.column_stack([s1_train[r] for r in cfg["cross_regions"]])
        cross_te = np.column_stack([s1_test[r] for r in cfg["cross_regions"]])
        for r in cfg["eval_regions"]:
            Xtr, ytr, valid, Xte = own[r]
            if cfg.get("s2_drop_own"):
                A, B = cross_tr, cross_te
            else:
                A, B = np.column_stack([Xtr, cross_tr]), np.column_stack([Xte, cross_te])
            v2 = valid & ~np.isnan(cross_tr).any(axis=1)
            g = ratio_col(r, cfg["s2"])
            if g is not None:
                gtr, gte = Xtr[:, g], Xte[:, g]
                nown = 0 if cfg.get("s2_drop_own") else Xtr.shape[1]
                A, B = A.copy(), B.copy()
                A[:, nown:] = A[:, nown:] / gtr[:, None]
                B[:, nown:] = B[:, nown:] / gte[:, None]
                (p2,) = fit_predict(A[v2], ytr[v2], [B], cfg["s2"], age_days[v2],
                                    None if cfg.get("s2_drop_own") else cats[r], gtr[v2], [gte])
            else:
                (p2,) = fit_predict(A[v2], ytr[v2], [B], cfg["s2"], age_days[v2], None if cfg.get("s2_drop_own") else cats[r])
            res[f"s2_{r}"] = p2.astype(np.float32)
    return res


# ----------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--learn-days", type=int, default=180)
    ap.add_argument("--cross", default=",".join(REGIONS))
    ap.add_argument("--eval", default=",".join(REGIONS))
    ap.add_argument("--one-stage", action="store_true")
    ap.add_argument("--features", default="")
    ap.add_argument("--s1", default="{}", help="json: params/rounds/halflife/transform/seeds/monotone")
    ap.add_argument("--s2", default=None, help="json; defaults to --s1")
    ap.add_argument("--oof", default=None, help="json: {k, block_days}")
    ap.add_argument("--s2-drop-own", action="store_true")
    ap.add_argument("--ts-mode", default=None, help="drop | rel:<bucket_days>")
    ap.add_argument("--categorical", default="", help="comma separated feature names treated as categorical")
    ap.add_argument("--gas-lag", type=int, default=0, help="days by which the gas price is known late")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()

    cross = a.cross.split(",")
    evalr = a.eval.split(",")
    s1 = json.loads(a.s1)
    s2 = json.loads(a.s2) if a.s2 is not None else dict(s1)
    cfg = {
        "learn_days": a.learn_days, "two_stage": not a.one_stage, "cross_regions": cross if not a.one_stage else evalr,
        "eval_regions": evalr, "s1": s1, "s2": s2, "oof": json.loads(a.oof) if a.oof else None,
        "s2_drop_own": a.s2_drop_own, "ts_mode": a.ts_mode,
        "categorical": [c for c in a.categorical.split(",") if c],
    }

    index = pd.read_parquet(f"{a.frames}/frame_{evalr[0]}.parquet", columns=["price"]).index
    start = pd.Timestamp(a.start, tz="UTC")
    end = pd.Timestamp(a.end, tz="UTC")
    cutoffs = []
    d = start
    while d < end - pd.Timedelta(days=3):  # same loop bound as performance_testing.py
        cutoffs.append(index.get_loc(d))
        d += pd.Timedelta(days=a.stride)
    assert cutoffs[-1] + HORIZON_ROWS <= len(index), "frames do not cover the requested range"

    t0 = time.time()
    regions_needed = sorted(set(cfg["cross_regions"]) | set(evalr), key=REGIONS.index)
    ctx = get_context("spawn")
    with ctx.Pool(a.procs, initializer=init_worker, initargs=(a.frames, a.features, regions_needed, a.gas_lag)) as pool:
        results = pool.map(run_iter, [(cfg, d0) for d0 in cutoffs], chunksize=1)
    results.sort(key=lambda r: r["d0"])

    out = {"cutoffs": index[cutoffs].tz_convert(None).to_numpy()}
    for r in evalr:
        y = pd.read_parquet(f"{a.frames}/frame_{r}.parquet", columns=["price"])["price"].to_numpy()
        out[f"actual_{r}"] = np.stack([y[d0:d0 + HORIZON_ROWS] for d0 in cutoffs]).astype(np.float32)
        out[f"s1_{r}"] = np.stack([x[f"s1_{r}"] for x in results])
        if cfg["two_stage"]:
            out[f"s2_{r}"] = np.stack([x[f"s2_{r}"] for x in results])
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(f"{a.out}/{a.name}.npz", **out)
    meta = {"name": a.name, "cfg": cfg, "features": a.features, "start": a.start, "end": a.end, "stride": a.stride,
            "n_iter": len(cutoffs), "seconds": round(time.time() - t0, 1)}
    with open(f"{a.out}/{a.name}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{a.name}: {len(cutoffs)} cutoffs in {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
