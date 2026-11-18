"""Hourly weather for DE's six sample locations as it was forecast 0/1/2/3 days earlier (Open-Meteo previous-runs archive)."""
import json
import sys
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

OUT = sys.argv[1]
START, END = date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
LATS = [48.4, 49.7, 51.3, 52.8, 53.8, 54.1]
LONS = [9.3, 11.3, 8.6, 12.0, 8.1, 11.6]
VARS = {"wind_speed_80m": "wind", "temperature_2m": "temp", "global_tilted_irradiance": "irradiance",
        "pressure_msl": "pressure", "relative_humidity_2m": "humidity"}
hourly = [v + (f"_previous_day{k}" if k else "") for v in VARS for k in range(4)]
import os

frames = [pd.read_parquet(OUT)] if os.path.exists(OUT) else []
have = set(frames[0].index.normalize().unique()) if frames else set()
# newest first, saved after every request so partial results are usable
e = END
while e >= START:
    d = max(e - timedelta(days=13), START)
    if pd.Timestamp(d, tz="UTC") in have and pd.Timestamp(e, tz="UTC") in have:
        e = d - timedelta(days=1)
        continue
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=" + ",".join(map(str, LATS)) + "&longitude=" + ",".join(map(str, LONS))
           + f"&azimuth=0&tilt=0&start_date={d}&end_date={e}&hourly=" + ",".join(hourly) + "&timezone=UTC")
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = json.loads(resp.read())
            assert isinstance(data, list), str(data)[:200]
            break
        except Exception as ex:
            print("retry", attempt, repr(ex)[:160], flush=True)
            time.sleep(20 + 15 * attempt)
    else:
        raise SystemExit("failed")
    parts = []
    for i, loc in enumerate(data):
        h = loc["hourly"]
        parts.append(pd.DataFrame({f"{VARS[v]}_{i}_d{k}": h[v + (f"_previous_day{k}" if k else "")] for v in VARS for k in range(4)},
                                  index=pd.to_datetime(h["time"], utc=True)))
    frames.append(pd.concat(parts, axis=1))
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated()]
    out.to_parquet(OUT + ".tmp")
    os.replace(OUT + ".tmp", OUT)
    print("fetched", d, e, "total", out.shape, out.index.min(), "nulls in chunk:", int(frames[-1].isna().sum().sum()), flush=True)
    e = d - timedelta(days=1)
    time.sleep(2)
print("done")
