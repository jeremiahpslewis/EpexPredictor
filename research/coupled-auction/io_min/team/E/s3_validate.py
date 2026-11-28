"""Validate s3_weather.py against Open-Meteo API data already on disk (team E). Reads S3 only; no API calls.

  store:   s3_weather store frame (mode latest, asof = the API fetch time) vs an API-filled store file, over the
           API refresh window of that fetch (fetch day - 1 .. + 8);
  vintage: s3_weather vintage frame vs a previous-runs parquet (dense/prev.parquet, live grid_prev_live.parquet).
Prints, per variable: n, mean |diff|, max |diff|, share within the API's rounding (<= 0.05, GTI/humidity <= 0.5),
and per lead-day the same (to see where blends or interpolation differ).

usage (SP/omvenv/bin/python):
  s3_validate.py store --store GRID --api SP/live/prefetch/2026-11-29_0011/stores/gridweather_v1_DE_LU.json.gz --asof 2026-11-28T00:12:01Z
  s3_validate.py vintage --store GRID --api SP/live/prefetch/2026-11-29_0011/grid_prev_live.parquet --start 2026-11-21 --end 2026-11-29 --fetched 2026-11-28T00:21Z
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_weather as s3w  # noqa: E402


def load_store(fn):
    d = pd.read_json(fn, compression="gzip")
    if d.index.dtype.kind in "iu":
        d.index = pd.to_datetime(d.index, unit="ms", utc=True)
    elif d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def report(api, s3, ref_time=None, by_lead=True):
    cols = [c for c in s3.columns if c in api.columns]
    a, b = api.reindex(s3.index)[cols], s3[cols]
    out = {}
    groups = sorted({c.rsplit("_", 1)[0] if not c.split("_")[-1].startswith("d") else
                     f"{c.split('_')[0]}_{c.split('_')[-1]}" for c in cols})
    for g in groups:
        gc = [c for c in cols if (c.rsplit("_", 1)[0] == g) or (f"{c.split('_')[0]}_{c.split('_')[-1]}" == g)]
        d = (b[gc] - a[gc]).abs()
        tol = 0.5 if g.split("_")[0] in ("irradiance", "humidity") else 0.051
        v = d.to_numpy().ravel()
        v = v[~np.isnan(v)]
        row = {"n": int(v.size), "mean": round(float(v.mean()), 4) if v.size else None,
               "p99": round(float(np.quantile(v, 0.99)), 3) if v.size else None,
               "max": round(float(v.max()), 3) if v.size else None,
               "share_exact": round(float((v <= tol).mean()), 4) if v.size else None,
               "s3_missing": int((b[gc].isna() & a[gc].notna()).to_numpy().sum())}
        if by_lead and ref_time is not None:
            lead = ((d.index - ref_time) / pd.Timedelta(days=1)).astype(float)
            lead = np.floor(lead).astype(int)
            row["by_lead_day"] = {int(L): [round(float(np.nanmean(d[lead == L].to_numpy())), 3),
                                           round(float(np.nanmax(d[lead == L].to_numpy())), 2),
                                           round(float(np.nanmean(d[lead == L].to_numpy() <= tol)), 3)]
                                  for L in sorted(set(lead)) if np.isfinite(d[lead == L].to_numpy()).any()}
        out[g] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("store", "vintage"))
    ap.add_argument("--store", default="GRID")
    ap.add_argument("--api", required=True)
    ap.add_argument("--asof")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--fetched", help="vintage: the API file's fetch time (later values are not genuine k-day-old)")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--mode", default="latest", choices=("latest", "rolling"))
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--save")
    a = ap.parse_args()
    if a.cmd == "store":
        api = load_store(a.api)
        asof = pd.Timestamp(a.asof)
        s = (asof - pd.Timedelta(days=1)).date()
        e = (asof + pd.Timedelta(days=8)).date()
        s3 = s3w.store_frame(a.store, s, e, a.mode, asof, a.models)
        log = s3.attrs.get("s3_log")
        res = report(api, s3, asof.floor("D"))
    else:
        api = pd.read_parquet(a.api)
        s3 = s3w.vintage_frame(a.store, a.k, a.start, a.end, a.models)
        log = s3.attrs.get("s3_log")
        if a.fetched:
            f = pd.Timestamp(a.fetched)
            for c in api.columns:
                k = int(c.split("_d")[-1])
                api.loc[api.index > f + pd.Timedelta(hours=24 * k), c] = np.nan
        res = report(api, s3, None, by_lead=False)
    if a.save:
        s3.attrs = {}
        s3.to_parquet(a.save)
    print(json.dumps(res, indent=1))
    if log:
        print(json.dumps(log, default=str)[:3000])


if __name__ == "__main__":
    main()
