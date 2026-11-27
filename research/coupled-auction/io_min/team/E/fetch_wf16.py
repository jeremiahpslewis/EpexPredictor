"""Backfill of the previous-runs vintages that the 16 walk-forward days (sample_wf16.txt) lack locally (team E).

Only what those days need under --vintage strict: for each day D whose window D-1 .. D+2 is not covered locally,
  - the 58-point DE grid, wind_speed_120m / shortwave_radiation / temperature_2m x previous_day1..3
    (local dense/prev.parquet starts 2026-08-25): 4 requests of <= 15 points;
  - d1 of the zones' weather points that prev_d1/ holds only from 2026-09-01 (AT, BE, FR, NL, SE4, DK1, DK2 and DE's
    5 North Sea points; the 5 weather_v2 variables): 8 requests (one per zone).
Dates per request: D-1 .. D+2 (4 days, < 14 days and < 10 variables, so each location counts as one weighted call:
about 58 + 32 = 90 weighted calls and 12 requests per day; 12 days -> ~1,080 weighted calls, 144 requests).
Resumable: one batch = one day; its files SP/wf16_hist/days/<D>_{grid,zones}.parquet mark it done. Stops at the first
HTTP error (quota). Then merges everything into SP/wf16_hist/grid_prev.parquet and prev_<zone>.parquet, the formats
faithful_bt.py reads with --vintage-extra SP/wf16_hist.
NOT before the live run of the day has ended (coordinator, 2026-11-26: after 10:30 UTC, quota embargo).

usage: venv/bin/python fetch_wf16.py [--max-days N] [--dry]   (--dry: print the plan, fetch nothing)
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

import pandas as pd

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{SP}/wf16_hist"
GRID_LOCAL_FROM = date(2026, 8, 25)     # dense/prev.parquet
ZONE_D1_LOCAL_FROM = date(2026, 9, 1)   # prev_d1/
GV = {"wind_speed_120m": "w120", "shortwave_radiation": "sw", "temperature_2m": "t2"}
ZV = {"wind_speed_80m": "wind", "temperature_2m": "temp", "global_tilted_irradiance": "irradiance",
      "pressure_msl": "pressure", "relative_humidity_2m": "humidity"}
ZONES = {"DE": list(range(6, 11)), "AT": None, "BE": None, "FR": None, "NL": None, "SE4": None, "DK1": None, "DK2": None}


def get(url):
    for attempt in range(4):   # the 429s seen on 11-26/27 came and went within a minute: retry 3x after 60 s
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = json.loads(r.read())
            return [data] if isinstance(data, dict) and "hourly" in data else data
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code}: {e.read()[:200]!r}"
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            msg = repr(e)[:200]
        print(f"  {msg} (attempt {attempt + 1})", flush=True)
        if attempt < 3:
            time.sleep(60)
    raise SystemExit(f"{msg} - stopping (resume later)")


def prev_runs(lats, lons, variables, ks, s, e, extra=""):
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=" + ",".join(map(str, lats)) + "&longitude="
           + ",".join(map(str, lons)) + f"{extra}&start_date={s}&end_date={e}&hourly="
           + ",".join(f"{v}_previous_day{k}" for v in variables for k in ks) + "&timezone=UTC")
    data = get(url)
    assert isinstance(data, list) and len(data) == len(lats), str(data)[:200]
    time.sleep(5)
    return data


def needs(d):
    return d - timedelta(days=1) < GRID_LOCAL_FROM, d - timedelta(days=1) < ZONE_D1_LOCAL_FROM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=99)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    days = [date.fromisoformat(x.split()[0]) for x in open(f"{HERE}/sample_wf16.txt") if x.strip()]
    todo = [d for d in days if any(needs(d))]
    print(f"{len(todo)} of {len(days)} days need a backfill: {', '.join(map(str, todo))}")
    print(f"per day: grid 4 requests (58 locations), zones d1 8 requests (32 locations); total {12 * len(todo)} requests, "
          f"~{90 * len(todo)} weighted calls")
    if a.dry:
        return
    sys.path.insert(0, "/home/user/EpexPredictor")
    from predictor.model.priceregion import PriceRegionName
    os.makedirs(f"{OUT}/days", exist_ok=True)
    pts = json.load(open(f"{SP}/dense/points.json"))["points"]
    done = 0
    for d in todo:
        if done >= a.max_days:
            break
        s, e = d - timedelta(days=1), d + timedelta(days=2)
        g_fn, z_fn = f"{OUT}/days/{d}_grid.parquet", f"{OUT}/days/{d}_zones.parquet"
        if needs(d)[0] and not os.path.exists(g_fn):
            parts = []
            for i0 in range(0, len(pts), 15):
                ch = pts[i0:i0 + 15]
                for i, loc in enumerate(prev_runs([p[0] for p in ch], [p[1] for p in ch], GV, (1, 2, 3), s, e)):
                    parts.append(pd.DataFrame({f"{GV[v]}_{i0 + i}_d{k}": loc["hourly"][f"{v}_previous_day{k}"] for v in GV for k in (1, 2, 3)},
                                              index=pd.to_datetime(loc["hourly"]["time"], utc=True)))
            pd.concat(parts, axis=1).astype("float32").to_parquet(g_fn)
        if needs(d)[1] and not os.path.exists(z_fn):
            parts = []
            for z, idx in ZONES.items():
                r = PriceRegionName(z).to_region()
                idx = idx if idx is not None else list(range(len(r.weather_latitudes)))
                data = prev_runs([r.weather_latitudes[i] for i in idx], [r.weather_longitudes[i] for i in idx], ZV, (1,), s, e, "&azimuth=0&tilt=0")
                parts.append(pd.concat([pd.DataFrame({f"{z}|{ZV[v]}_{i}_d1": loc["hourly"][f"{v}_previous_day1"] for v in ZV},
                                                     index=pd.to_datetime(loc["hourly"]["time"], utc=True)) for i, loc in zip(idx, data)], axis=1))
            pd.concat(parts, axis=1).astype("float32").to_parquet(z_fn)
        done += 1
        print(f"{d}: done", flush=True)
    # merge into the formats faithful_bt reads (--vintage-extra OUT)
    grids = [pd.read_parquet(f"{OUT}/days/{f}") for f in sorted(os.listdir(f"{OUT}/days")) if f.endswith("_grid.parquet")]
    if grids:
        g = pd.concat(grids)
        g[~g.index.duplicated(keep="last")].sort_index().to_parquet(f"{OUT}/grid_prev.parquet")
    zs = [pd.read_parquet(f"{OUT}/days/{f}") for f in sorted(os.listdir(f"{OUT}/days")) if f.endswith("_zones.parquet")]
    if zs:
        z = pd.concat(zs)
        z = z[~z.index.duplicated(keep="last")].sort_index()
        for zone in ZONES:
            cols = [c for c in z.columns if c.startswith(f"{zone}|")]
            z[cols].rename(columns=lambda c: c.split("|", 1)[1]).to_parquet(f"{OUT}/prev_{zone}.parquet")
    print(f"merged into {OUT}: grid {'yes' if grids else 'no'}, zones {'yes' if zs else 'no'}; "
          f"{sum(1 for d in todo if os.path.exists(f'{OUT}/days/{d}_grid.parquet'))}/{len(todo)} days have grid")


if __name__ == "__main__":
    main()
