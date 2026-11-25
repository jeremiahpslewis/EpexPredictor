"""Fetch the missing `_previous_day1` weather (Open-Meteo previous-runs archive) for the faithful backtest.

What is already local (see README.md): d1 for DE's 6 onshore points (SP/prevruns_DE.parquet), d2/d3 for every
zone's weather points (SP/prev_all/prev_<zone>.parquet), d1..d3 for the 58-point grid (SP/dense/prev.parquet).
Missing, fetched here: d1 for the weather points of AT, BE, FR, NL, SE4, DK1, DK2 and for DE's 5 North Sea points.

One request per zone (all its points, the whole window, the 5 variables of weather_v2), saved per zone, resumable:
a zone whose file already covers the window is skipped. Stops at the first rate-limit/error answer.

usage: python fetch_d1.py [START] [END]      (default 2026-09-01 2026-11-09)
"""
import json, os, sys, time, urllib.error, urllib.request
from datetime import date

import pandas as pd

sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model.priceregion import PriceRegionName  # noqa: E402

SP = os.environ.get("SP", "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad")
OUT = f"{SP}/prev_d1"
START = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 9, 1)
END = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 11, 9)
VARS = {"wind_speed_80m": "wind", "temperature_2m": "temp", "global_tilted_irradiance": "irradiance",
        "pressure_msl": "pressure", "relative_humidity_2m": "humidity"}
# zone -> indices of its weather points to fetch (DE: only the North Sea points, 0..5 are in prevruns_DE.parquet)
ZONES = {"DE": list(range(6, 11)), "AT": None, "BE": None, "FR": None, "NL": None, "SE4": None, "DK1": None, "DK2": None}


def fetch(lats, lons):
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=" + ",".join(map(str, lats))
           + "&longitude=" + ",".join(map(str, lons)) + f"&azimuth=0&tilt=0&start_date={START}&end_date={END}"
           + "&hourly=" + ",".join(f"{v}_previous_day1" for v in VARS) + "&timezone=UTC")
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read()[:300]!r} - stopping (rate limit?)")
    if isinstance(data, dict) and "hourly" in data:
        data = [data]
    if not isinstance(data, list):
        raise SystemExit(f"unexpected answer, stopping: {str(data)[:300]}")
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    want = pd.date_range(pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC") + pd.Timedelta(hours=23), freq="1h")
    for zone, idx in ZONES.items():
        fn = f"{OUT}/prev_d1_{zone}.parquet"
        if os.path.exists(fn):
            have = pd.read_parquet(fn)
            if want.isin(have.index).all():
                print(zone, "complete, skipped", flush=True)
                continue
        r = PriceRegionName(zone).to_region()
        idx = idx if idx is not None else list(range(len(r.weather_latitudes)))
        t0 = time.time()
        data = fetch([r.weather_latitudes[i] for i in idx], [r.weather_longitudes[i] for i in idx])
        assert len(data) == len(idx)
        parts = [pd.DataFrame({f"{VARS[v]}_{i}_d1": loc["hourly"][f"{v}_previous_day1"] for v in VARS},
                              index=pd.to_datetime(loc["hourly"]["time"], utc=True)) for i, loc in zip(idx, data)]
        out = pd.concat(parts, axis=1).astype("float32")
        out.to_parquet(fn + ".tmp"); os.replace(fn + ".tmp", fn)
        print(zone, out.shape, out.index.min(), out.index.max(), "nulls", int(out.isna().sum().sum()),
              f"{time.time() - t0:.1f}s", flush=True)
        time.sleep(5)
    print("done")


if __name__ == "__main__":
    main()
