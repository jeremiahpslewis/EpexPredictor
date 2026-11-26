"""Naive baseline (same local quarter hour one week earlier) on the stored faithful days, and rMAE of the champion.

naive(t) = DE price at the same local wall-clock quarter hour 7 days before t (Europe/Berlin), from the app's price
store (SP/fresh_data/stores, which equals the stored `actual` on the used days). On the 100-slot DST day the repeated
hour takes the first occurrence a week earlier. Known at the gate: t - 7 d <= D-1 23:45.
MAE in ct/kWh over all quarter hours of a day, then the mean over days (the champion convention). rMAE = window MAE of
the forecast / window MAE of naive. Intervals: paired circular 7-day block bootstrap, 90 % (distill/common.boot).

usage: venv/bin/python naive.py   -> prints a markdown table; written to team/E/NAIVE.md by the caller
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../distill")))
from common import boot  # noqa: E402

SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
TZ = "Europe/Berlin"


def price_series():
    p = pd.read_json(f"{SP}/fresh_data/stores/prices_v3_DE_LU.json.gz", compression="gzip")
    if pd.api.types.is_integer_dtype(p.index.dtype):
        p.index = pd.to_datetime(p.index, unit="ms", utc=True)
    elif p.index.tz is None:
        p.index = p.index.tz_localize("UTC")
    return p["price"].dropna().sort_index()


def naive_for(idx_utc: pd.DatetimeIndex, price: pd.Series) -> np.ndarray:
    loc = idx_utc.tz_convert(TZ)
    by_local = price.copy()
    by_local.index = price.index.tz_convert(TZ).tz_localize(None)
    by_local = by_local[~by_local.index.duplicated(keep="first")]
    want = loc.tz_localize(None) - pd.Timedelta(days=7)
    return by_local.reindex(want).to_numpy(float)


def load_window(path, first="2000-01-01", last="2999-12-31"):
    rows = []
    price = price_series()
    for fn in sorted(glob.glob(f"{path}/*.npz")):
        if not first <= os.path.basename(fn)[:10] <= last:
            continue
        z = np.load(fn, allow_pickle=False)
        idx = pd.DatetimeIndex(pd.to_datetime(z["idx"], unit="us" if z["idx"][0] < 1e17 else "ns", utc=True))
        nv = naive_for(idx, price)
        a = z["actual"]
        # the store's price equals the stored actual on D
        assert np.allclose(price.reindex(idx).to_numpy(float), a, equal_nan=False), fn
        assert np.isfinite(nv).all(), fn
        r = {"day": os.path.basename(fn)[:10], "naive": np.mean(np.abs(nv - a))}
        for k in ("trees", "timestral", "market", "issued"):
            r[k] = np.mean(np.abs(z[k] - a))
        rows.append(r)
    return pd.DataFrame(rows).set_index("day")


def table(df, label):
    out = [f"### {label} ({len(df)} days, {df.index[0]}..{df.index[-1]})", "",
           "| forecast | MAE | rMAE vs naive | forecast - naive, 90% interval |", "|---|---|---|---|"]
    for k, name in (("naive", "naive (same QH, D-7)"), ("trees", "trees"), ("timestral", "Timestral"), ("market", "market route"), ("issued", "blend")):
        if k == "naive":
            out.append(f"| {name} | {df[k].mean():.3f} | 1 | |")
        else:
            d = (df[k] - df["naive"]).to_numpy()
            b = boot(d)
            out.append(f"| {name} | {df[k].mean():.3f} | {df[k].mean() / df['naive'].mean():.3f} | {d.mean():+.3f} [{b[0]:+.3f}, {b[1]:+.3f}] |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    md = ["# Naive baseline on the stored faithful days (champion = out_strict / out_fresh_strict, base)", ""]
    for path, label, lo, hi in ((f"{SP}/faithful/out_strict", "61 used days", "2026-09-08", "2026-11-07"),
                                (f"{SP}/faithful/out_fresh_strict", "18 fresh days", "2026-11-09", "2026-11-26")):
        df = load_window(path, lo, hi)
        md.append(table(df, label))
        md.append("Per-month naive MAE: " + ", ".join(f"{m}: {v:.3f}" for m, v in df["naive"].groupby(df.index.str[:7]).mean().items()) + "\n")
    print("\n".join(md))
