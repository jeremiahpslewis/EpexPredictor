"""Open-Meteo weather from the public open-data bucket on AWS S3 (s3://openmeteo, no quota), team E.

Replaces the Open-Meteo HTTP API calls of the app's weather stores (predictor/model/weatherstore.py WeatherStore,
predictor/model/residualload.py GridWeatherStore) for the live forecast and for previous-runs vintages, returning the
same DataFrame shapes:
  - zone stores (weather_v2): minutely_15, columns wind_<i> (wind_speed_80m km/h), temp_<i> (temperature_2m degC),
    irradiance_<i> (global_tilted_irradiance tilt 0 = shortwave W/m2), pressure_<i> (pressure_msl hPa),
    humidity_<i> (relative_humidity_2m %);
  - grid store (gridweather_v1): hourly, wind_<i> (wind_speed_120m km/h), radiation_<i> (shortwave_radiation W/m2),
    temp_<i> (temperature_2m degC);
  - vintages: hourly, <short>_<i>_d<k> as dense/prev.parquet and live_fetch.grid_prev_runs (w120, sw, t2 / zone names).
Index: UTC DatetimeIndex. Values rounded as the API rounds them (0.1; GTI and humidity to integers).

Bucket layout (README of github.com/open-meteo/open-data):
  data/<model>/<var>/chunk_<n>.om        rolling time series, overwritten by each run: what api.open-meteo.com reads
  data_run/<model>/YYYY/MM/DD/HHMMZ/<var>.om   one file per run and variable, native time steps, ~3 months retention
  data/<model>/static/meta.json, data_run/<model>/<run>/meta.json (created_at = when the run was published here)
Stored are u/v wind components (speed is derived: hypot * 3.6 km/h) and, for ICON, direct + diffuse radiation
(shortwave = sum). Backward-averaged radiation lacks the first step of each run in data_run.

Modes:
  rolling  - data/ as it is now (the API's own source: identical values for the latest forecast);
  latest   - "latest run published <= asof" per model, from data_run: for each valid time the newest run with
             init <= valid time among those published by asof (the stitch data/ performs);
  vintage  - "the run issued k days before each valid time": newest run with init <= valid - 24 h * k
             (previous-runs API semantics, previous_day<k>), from data_run.
Models are blended like the API's default for central Europe (see BLEND): the first model in the list that has a
value wins for each point and valid time.

Needs the `omfiles` package (and fsspec/s3fs/pyproj): run with SP/omvenv/bin/python, never in the app venvs.

usage:
  omvenv/bin/python s3_weather.py store --store GRID --start 2026-11-27 --end 2026-12-06 --out x.parquet [--mode rolling|latest] [--asof T]
  omvenv/bin/python s3_weather.py vintage --store GRID --k 1 2 3 --start 2026-11-20 --end 2026-11-30 --out y.parquet
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
BUCKET = "openmeteo"
CACHE = f"{SP}/s3cache"
THREADS = int(os.environ.get("S3W_THREADS", "4"))      # network-bound reads; the CPU work is small

_fs = None


def fs():
    global _fs
    if _fs is None:
        import fsspec
        _fs = fsspec.filesystem("s3", anon=True, default_block_size=65536)
    return _fs


def _reader(path):
    from omfiles import OmFileReader
    return OmFileReader.from_fsspec(fs(), f"{BUCKET}/{path}")


# ---------------------------------------------------------------- points and stores (the app's, read from predictor/)

def store_points(store: str) -> list[tuple[float, float]]:
    fn = f"{CACHE}/points.json"
    if not os.path.exists(fn):
        raise SystemExit(f"{fn} missing: run `tsvenv/bin/python s3_weather.py points` first (reads predictor/ regions)")
    return [tuple(p) for p in json.load(open(fn))[store]]


def dump_points():
    sys.path.insert(0, "/home/user/EpexPredictor")
    from predictor.model.priceregion import PriceRegionName
    out = {}
    for z in ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]:
        r = PriceRegionName(z).to_region()
        out[z] = list(zip(r.weather_latitudes, r.weather_longitudes))
    r = PriceRegionName.DE.to_region()
    out["GRID"] = list(zip(r.generation_latitudes + r.generation_offshore_latitudes,
                           r.generation_longitudes + r.generation_offshore_longitudes))
    os.makedirs(CACHE, exist_ok=True)
    json.dump(out, open(f"{CACHE}/points.json", "w"))
    return out


# store -> (resolution, {api variable: column name}); as in the app's store classes
ZONE_VARS = {"wind_speed_80m": "wind", "temperature_2m": "temp", "global_tilted_irradiance": "irradiance",
             "pressure_msl": "pressure", "relative_humidity_2m": "humidity"}
GRID_VARS = {"wind_speed_120m": "wind", "shortwave_radiation": "radiation", "temperature_2m": "temp"}
VINTAGE_SHORT = {"wind_speed_120m": "w120", "wind_speed_80m": "w80", "shortwave_radiation": "sw", "temperature_2m": "t2",
                 "global_tilted_irradiance": "irradiance", "pressure_msl": "pressure", "relative_humidity_2m": "humidity"}


def store_spec(store):
    if store == "GRID":
        return "hourly", GRID_VARS
    return "minutely_15", ZONE_VARS


ROUND = {"global_tilted_irradiance": 0, "relative_humidity_2m": 0}   # API output precision; everything else 0.1

# API default ("best match") for central Europe, highest priority first (validated: see S3_WEATHER.md)
BLEND = ["dwd_icon_d2", "dwd_icon_eu", "dwd_icon", "ecmwf_ifs025"]


# ---------------------------------------------------------------- grids

class Grid:
    """regular lat/lon grid of a model, from its meta.json crs_wkt BBOX and the file shape"""

    def __init__(self, model):
        self.model = model
        fn = f"{CACHE}/grid_{model}.json"
        if os.path.exists(fn):
            g = json.load(open(fn))
        else:
            meta = json.loads(fs().cat_file(f"{BUCKET}/data/{model}/static/meta.json"))
            with _reader(f"data/{model}/static/HSURF.om") as r:
                ny, nx = r.shape[:2]
            import re
            bb = [float(x) for x in re.search(r"BBOX\[([^\]]+)\]", meta["crs_wkt"]).group(1).split(",")]
            g = {"ny": ny, "nx": nx, "lat0": bb[0], "lon0": bb[1], "lat1": bb[2], "lon1": bb[3]}
            os.makedirs(CACHE, exist_ok=True)
            json.dump(g, open(fn, "w"))
        self.__dict__.update(g)
        self.dy = (self.lat1 - self.lat0) / (self.ny - 1)
        self.dx = (self.lon1 - self.lon0) / (self.nx - 1)

    def cell(self, lat, lon):
        y = int(round((lat - self.lat0) / self.dy))
        x = int(round((lon - self.lon0) / self.dx))
        if 0 <= y < self.ny and 0 <= x < self.nx:
            return y, x
        return None


_cells: dict = {}


def cells(model, points):
    """grid cell per point: the API's choice where validated (CACHE/cellmap.json, written by s3_validate.py), else nearest"""
    if model not in _cells:
        fn = f"{CACHE}/cellmap.json"
        _cells[model] = json.load(open(fn)).get(model, {}) if os.path.exists(fn) else {}
    g = Grid(model)
    out = []
    for la, lo in points:
        key = f"{la:.4f},{lo:.4f}"
        c = _cells[model].get(key)
        out.append(tuple(c) if c is not None else g.cell(la, lo))
    return out


# ---------------------------------------------------------------- raw readers

def _read_points(path, yx, tsl=slice(None)):
    """values [npoint, ntime] at the given cells from one .om file (time last); NaN where a cell is None"""
    with _reader(path) as r:
        nt = r.shape[2]
        t = range(nt)[tsl]
        out = np.full((len(yx), len(t)), np.nan, np.float32)
        for k, c in enumerate(yx):
            if c is not None:
                out[k] = r[c[0], c[1], tsl]
    return out


def rolling_raw(model, var, yx, start, end):
    """data/<model>/<var>/chunk_*.om between start and end (inclusive) -> (times, values[npoint, ntime])"""
    from omfiles.meta import OmChunksMeta
    meta = OmChunksMeta.from_dict(json.loads(fs().cat_file(f"{BUCKET}/data/{model}/static/meta.json")))
    t0, t1 = np.datetime64(pd.Timestamp(start).tz_localize(None), "s"), np.datetime64(pd.Timestamp(end).tz_localize(None), "s")
    ts, vs = [], []
    for c in meta.chunks_for_date_range(t0, t1):
        ct = meta.get_chunk_time_range(c)
        m = (ct >= t0) & (ct <= t1)
        if not m.any():
            continue
        ii = np.where(m)[0]
        try:
            v = _read_points(f"data/{model}/{var}/chunk_{c}.om", yx, slice(int(ii[0]), int(ii[-1]) + 1))
        except (FileNotFoundError, RuntimeError, OSError):
            continue
        ts.append(ct[m]); vs.append(v)
    if not ts:
        return pd.DatetimeIndex([], tz="UTC"), np.zeros((len(yx), 0), np.float32)
    return pd.DatetimeIndex(np.concatenate(ts)).tz_localize("UTC"), np.concatenate(vs, axis=1)


_runs: dict = {}


def list_runs(model, start, end):
    """runs of `model` in data_run with init in [start, end]: list of (init, created_at) as UTC Timestamps"""
    out = []
    day = pd.Timestamp(start).floor("D")
    while day <= pd.Timestamp(end):
        key = (model, day)
        if key not in _runs:
            try:
                ent = fs().ls(f"{BUCKET}/data_run/{model}/{day:%Y/%m/%d}/", detail=False)
            except FileNotFoundError:
                ent = []
            _runs[key] = sorted(pd.Timestamp(f"{day:%Y-%m-%d}T{e.rstrip('/').split('/')[-1][:4]}", tz="UTC") for e in ent
                                if e.rstrip("/").endswith("Z"))
        out += [r for r in _runs[key] if pd.Timestamp(start) <= r <= pd.Timestamp(end)]
        day += pd.Timedelta(days=1)
    return out


_runmeta: dict = {}


def run_meta(model, init):
    key = (model, init)
    if key not in _runmeta:
        try:
            _runmeta[key] = json.loads(fs().cat_file(f"{BUCKET}/data_run/{model}/{init:%Y/%m/%d/%H%M}Z/meta.json"))
        except FileNotFoundError:
            _runmeta[key] = None       # run still being written
    return _runmeta[key]


def run_raw(model, init, var, yx):
    """one run's file -> (valid times, values[npoint, ntime]); native steps (1 h, later 3 h / 6 h)"""
    meta = run_meta(model, init)
    if meta is None or var not in meta["variables"]:
        return None
    path = f"data_run/{model}/{init:%Y/%m/%d/%H%M}Z/{var}.om"
    with _reader(path) as r:
        # each file carries its own valid times (backward-averaged variables start one step after init)
        tm = None
        for i in range(r.num_children):
            c = r.get_child_by_index(i)
            if c.name == "time":
                tm = np.asarray(c[...])
        out = np.full((len(yx), r.shape[2]), np.nan, np.float32)
        for k, c in enumerate(yx):
            if c is not None:
                out[k] = r[c[0], c[1], :]
    vt = pd.DatetimeIndex(pd.to_datetime(tm, unit="s", utc=True)) if tm is not None else \
        pd.DatetimeIndex(pd.to_datetime(meta["valid_times"], utc=True))[-out.shape[1]:]
    return vt, out


# ---------------------------------------------------------------- derived variables and time interpolation

def stored_vars(model, var):
    """which stored variables give `var`, and how"""
    if var.startswith("wind_speed_"):
        h = var.split("_")[-1]
        return [f"wind_u_component_{h}", f"wind_v_component_{h}"], lambda u, v: np.hypot(u, v) * 3.6
    if var in ("shortwave_radiation", "global_tilted_irradiance"):
        if model.startswith("ecmwf"):
            return ["shortwave_radiation"], lambda s: s
        return ["direct_radiation", "diffuse_radiation"], lambda a, b: a + b
    return [var], lambda a: a


def hermite(t_src, v, t_dst):
    """Open-Meteo's hermite (Catmull-Rom) interpolation along time, for instantaneous variables; v [npoint, nsrc]"""
    xs = (t_src.asi8 // 10**9).astype(np.float64)
    xd = (t_dst.asi8 // 10**9).astype(np.float64)
    out = np.full((v.shape[0], len(xd)), np.nan, np.float32)
    j = np.searchsorted(xs, xd, side="right") - 1
    for n, (jj, x) in enumerate(zip(j, xd)):
        if jj < 0 or jj >= len(xs):
            continue
        if xs[jj] == x:
            out[:, n] = v[:, jj]
            continue
        if jj + 1 >= len(xs):
            continue
        B, C = v[:, jj], v[:, jj + 1]
        A = v[:, jj - 1] if jj >= 1 else B
        D = v[:, jj + 2] if jj + 2 < len(xs) else C
        f = (x - xs[jj]) / (xs[jj + 1] - xs[jj])
        a = -A / 2 + 3 * B / 2 - 3 * C / 2 + D / 2
        b = A - 5 * B / 2 + 2 * C - D / 2
        c = -A / 2 + C / 2
        out[:, n] = ((a * f + b) * f + c) * f + B
    return out


def backward_avg_linear(t_src, v, t_dst):
    """backward-averaged fluxes (radiation) onto a finer step: each source value is the mean over the preceding
    interval, so a step inside it gets the same mean (a flat fill; Open-Meteo uses a solar-geometry-aware fill)"""
    idx = np.searchsorted(t_src.asi8, t_dst.asi8, side="left")
    out = np.full((v.shape[0], len(t_dst)), np.nan, np.float32)
    ok = idx < len(t_src)
    out[:, ok] = v[:, idx[ok]]
    return out


def on_times(var, t_src, v, t_dst):
    if len(t_src) == 0:
        return np.full((v.shape[0], len(t_dst)), np.nan, np.float32)
    if var in RADIATION:
        return backward_avg_linear(t_src, v, t_dst)
    return hermite(t_src, v, t_dst)


# ---------------------------------------------------------------- one model, one variable

def model_var_rolling(model, var, yx, start, end, t_dst):
    names, f = stored_vars(model, var)
    parts = [rolling_raw(model, n, yx, start - pd.Timedelta(hours=6), end + pd.Timedelta(hours=6)) for n in names]
    t = parts[0][0]
    if len(t) == 0:
        return np.full((len(yx), len(t_dst)), np.nan, np.float32), {}
    vals = f(*[p[1] for p in parts])
    return on_times(var, t, vals, t_dst), {"source": f"data/{model} (rolling, now)"}


RADIATION = ("shortwave_radiation", "global_tilted_irradiance", "direct_radiation", "diffuse_radiation")


def model_var_runs(model, var, yx, t_dst, pick, strict_rad=False):
    """for each destination time, the run chosen by pick(valid time) -> run init (or None); runs read on demand.
    strict_rad: a backward-averaged variable at a run's init time comes from the run before (data_run lacks it)"""
    names, f = stored_vars(model, var)
    out = np.full((len(yx), len(t_dst)), np.nan, np.float32)
    eps = pd.Timedelta(seconds=1) if (strict_rad and var in RADIATION) else pd.Timedelta(0)
    chosen = pd.Series([pick(t - eps) for t in t_dst], index=t_dst)
    used = {}
    for init, ts in chosen.dropna().groupby(chosen.dropna()):
        raws = [run_raw(model, init, n, yx) for n in names]
        if any(r is None for r in raws):
            continue
        vt = raws[0][0]
        vals = f(*[r[1] for r in raws])
        mask = t_dst.isin(ts.index)
        out[:, mask] = on_times(var, vt, vals, t_dst[mask])
        used[str(init)] = int(mask.sum())
    return out, {"runs": used}


# ---------------------------------------------------------------- public API

def fetch(points, variables: dict, start, end, resolution="hourly", mode="rolling", asof=None, k=None,
          models=None, colfmt="{name}_{i}"):
    """DataFrame (UTC index from start 00:00 to end 23:xx, inclusive days like the API's start_date/end_date),
    columns colfmt.format(name=..., i=...), blended over `models` (first with a value wins)"""
    models = models or BLEND
    step = "15min" if resolution == "minutely_15" else "1h"
    s = pd.Timestamp(start, tz="UTC").floor("D")
    e = pd.Timestamp(end, tz="UTC").floor("D") + pd.Timedelta(days=1) - pd.Timedelta(step)
    t_dst = pd.date_range(s, e, freq=step)
    asof = pd.Timestamp(asof if asof is not None else datetime.now(timezone.utc))
    if asof.tz is None:
        asof = asof.tz_localize("UTC")
    log = {"mode": mode, "asof": str(asof), "models": {}}
    cols = {}

    def run_picker(model):
        runs = list_runs(model, s - pd.Timedelta(days=10), min(asof, e))
        if mode == "latest":
            ok = []
            for r in runs:
                m = run_meta(model, r)
                if m is not None and pd.Timestamp(m["created_at"]) <= asof:
                    ok.append(r)
            ok = pd.DatetimeIndex(ok)

            def pick(t):
                c = ok[ok <= t]
                return c.max() if len(c) else None
            return pick, [str(x) for x in ok[-3:]]
        lag = pd.Timedelta(hours=24 * k)
        ri = pd.DatetimeIndex(runs)

        def pick(t):
            c = ri[ri <= t - lag]
            return c.max() if len(c) else None
        return pick, None

    work = []
    for var, name in variables.items():
        for m in models:
            work.append((var, name, m))

    def one(job):
        var, name, m = job
        yx = cells(m, points)
        if mode == "rolling":
            return job, *model_var_rolling(m, var, yx, s, e, t_dst)
        pick, last = run_picker(m)
        v, info = model_var_runs(m, var, yx, t_dst, pick, strict_rad=(mode == "latest"))
        if last:
            info["latest_runs_published"] = last
        return job, v, info

    with ThreadPoolExecutor(THREADS) as ex:
        res = {job[:3]: (v, info) for job, v, info in ex.map(one, work)}
    for var, name in variables.items():
        acc = np.full((len(points), len(t_dst)), np.nan, np.float32)
        for m in models:
            v, info = res[(var, name, m)]
            fill = np.isnan(acc) & ~np.isnan(v)
            acc[fill] = v[fill]
            log["models"].setdefault(m, {})[var] = info | {"cells_filled": int(fill.sum())}
        nd = ROUND.get(var, 1)
        acc = np.round(acc.astype(np.float64), nd)
        for i in range(len(points)):
            cols[colfmt.format(name=name, i=i)] = acc[i]
    df = pd.DataFrame(cols, index=t_dst)
    df.index.name = "time"
    df.attrs["s3_log"] = log
    return df


def store_frame(store, start, end, mode="rolling", asof=None, models=None):
    res, vars_ = store_spec(store)
    return fetch(store_points(store), vars_, start, end, res, mode, asof, models=models)


def vintage_frame(store, ks, start, end, models=None, variables=None):
    pts = store_points(store)
    vars_ = variables or (GRID_VARS if store == "GRID" else ZONE_VARS)
    parts = []
    for k in ks:
        parts.append(fetch(pts, {v: VINTAGE_SHORT[v] for v in vars_}, start, end, "hourly", "vintage", k=k,
                           models=models, colfmt="{name}_{i}_d" + str(k)))
    df = pd.concat(parts, axis=1)
    return df[sorted(df.columns, key=lambda c: (int(c.split("_")[-2]), c))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("points", "store", "vintage"))
    ap.add_argument("--store", default="GRID")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--mode", default="rolling", choices=("rolling", "latest"))
    ap.add_argument("--asof")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "points":
        print({k: len(v) for k, v in dump_points().items()})
        return
    t0 = time.time()
    if a.cmd == "store":
        df = store_frame(a.store, a.start, a.end, a.mode, a.asof, a.models)
    else:
        df = vintage_frame(a.store, a.k, a.start, a.end, a.models)
    log = df.attrs.get("s3_log", {})
    log |= {"store": a.store, "start": a.start, "end": a.end, "secs": round(time.time() - t0, 1), "rows": len(df),
            "complete_rows": int(df.notna().all(axis=1).sum()), "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    df.attrs = {}
    if a.out:
        df.to_parquet(a.out)
        json.dump(log, open(a.out + ".json", "w"), indent=1, default=str)
    print(json.dumps({k: log[k] for k in ("store", "rows", "complete_rows", "secs")}))


if __name__ == "__main__":
    main()
