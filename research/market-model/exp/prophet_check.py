"""Spot check: Prophet on the same rolling backtest (DE, stage-1 inputs), on a subsample of cutoffs.
Output is evaluate.py-compatible (key s1_DE)."""
import json
import logging
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

FRAMES, OUT, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
LEARN_DAYS, STRIDE, VARIANT = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
PROCS = int(sys.argv[7]) if len(sys.argv) > 7 else 2
S = 96
G = {}


def init():
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    df = pd.read_parquet(f"{FRAMES}/frame_DE.parquet")
    wind = df[[c for c in df.columns if c.startswith("wind_")]]
    irr = df[[c for c in df.columns if c.startswith("irradiance_")]]
    temp = df[[c for c in df.columns if c.startswith("temp_")]]
    df["wind_mean"] = wind.mean(axis=1)
    df["wind_cube_mean"] = (wind.clip(upper=45.0) ** 3).mean(axis=1)
    df["irr_mean"] = irr.mean(axis=1)
    df["temp_mean"] = temp.mean(axis=1)
    local = df.index.tz_convert("Europe/Berlin")
    df["ds"] = local.tz_localize(None)
    off = (df["weekday"] >= 5) | (df["holiday"] >= 0.5)
    df["is_offday"] = off
    df["is_workday"] = ~off
    G["df"] = df


def run(d0):
    from prophet import Prophet

    df = G["df"]
    tr = df.iloc[d0 - LEARN_DAYS * S:d0]
    te = df.iloc[d0:d0 + 3 * S + 1].copy()
    te["gasprice"] = df["gasprice"].iloc[d0]
    weather = [c for c in df.columns if c.split("_")[0] in ("wind", "temp", "irradiance", "pressure", "humidity") and c[-1].isdigit()]
    regs = weather + ["sunelevation", "gasprice", "holiday"]
    if VARIANT == "agg":
        regs = regs + ["wind_mean", "wind_cube_mean", "irr_mean", "temp_mean"]
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
    # separate intraday shapes for working days and weekends/holidays
    m.add_seasonality(name="daily_workday", period=1, fourier_order=8, condition_name="is_workday")
    m.add_seasonality(name="daily_offday", period=1, fourier_order=8, condition_name="is_offday")
    for c in regs:
        m.add_regressor(c)
    cols = ["ds", "is_workday", "is_offday"] + regs
    m.fit(tr[cols].assign(y=tr["price"].to_numpy()))
    return d0, m.predict(te[cols])["yhat"].to_numpy().astype(np.float32)


if __name__ == "__main__":
    index = pd.read_parquet(f"{FRAMES}/frame_DE.parquet", columns=["price"]).index
    y = pd.read_parquet(f"{FRAMES}/frame_DE.parquet", columns=["price"])["price"].to_numpy()
    cutoffs = []
    d = pd.Timestamp("2025-09-01", tz="UTC")
    while d < pd.Timestamp("2026-09-01", tz="UTC") - pd.Timedelta(days=3):
        cutoffs.append(index.get_loc(d))
        d += pd.Timedelta(days=STRIDE)
    t0 = time.time()
    with get_context("spawn").Pool(PROCS, initializer=init) as pool:
        res = dict(pool.map(run, cutoffs, chunksize=1))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/{NAME}.npz", cutoffs=index[cutoffs].tz_convert(None).to_numpy(),
                        actual_DE=np.stack([y[c:c + 3 * S + 1] for c in cutoffs]).astype(np.float32),
                        s1_DE=np.stack([res[c] for c in cutoffs]))
    json.dump({"name": NAME, "cfg": {"eval_regions": ["DE"], "learn_days": LEARN_DAYS, "variant": VARIANT},
               "stride": STRIDE, "n_iter": len(cutoffs), "seconds": round(time.time() - t0, 1)},
              open(f"{OUT}/{NAME}.json", "w"))
    print(f"{NAME}: {len(cutoffs)} cutoffs in {time.time() - t0:.0f}s", flush=True)
