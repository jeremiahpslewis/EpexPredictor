"""Hourly weather for a set of Europe-wide points outside the modelled zones (Open-Meteo historical forecast archive)."""
import json
import os
import sys
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

OUT = sys.argv[1]
START, END = date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
POINTS = {  # name: (lat, lon)
    # North Sea
    "ns_south": (53.0, 3.5), "ns_dogger": (54.7, 2.0), "ns_bight": (55.0, 6.5), "ns_central": (56.5, 3.0), "ns_north": (59.5, 2.0),
    # North Atlantic / Norwegian Sea
    "at_wire": (53.0, -15.0), "at_rockall": (57.5, -13.0), "at_mid": (50.0, -25.0), "at_biscay": (46.0, -6.0),
    "at_faroe": (62.0, -7.0), "at_norw": (65.0, 3.0), "at_iceland": (64.1, -21.9), "at_azores": (37.7, -25.7),
    # land areas outside the modelled zones
    "gb_scot": (57.0, -3.5), "gb_mid": (52.5, -1.5), "ie": (53.3, -8.0), "no_s": (59.0, 7.5),
    "pl_n": (54.2, 18.0), "pl_c": (52.0, 19.5), "cz": (49.8, 15.5), "ch": (46.9, 8.2), "it_n": (45.3, 9.8),
}
VARS = {"wind_speed_80m": "wind", "temperature_2m": "temp", "shortwave_radiation": "irr", "pressure_msl": "pres"}
names = list(POINTS)
frames = [pd.read_parquet(OUT)] if os.path.exists(OUT) else []
have = set(frames[0].index.normalize().unique()) if frames else set()
d = START
while d <= END:
    e = min(d + timedelta(days=44), END)
    if pd.Timestamp(d, tz="UTC") in have and pd.Timestamp(e, tz="UTC") in have:
        d = e + timedelta(days=1)
        continue
    url = ("https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=" + ",".join(str(POINTS[n][0]) for n in names)
           + "&longitude=" + ",".join(str(POINTS[n][1]) for n in names)
           + f"&start_date={d}&end_date={e}&hourly=" + ",".join(VARS) + "&timezone=UTC")
    for attempt in range(15):
        try:
            with urllib.request.urlopen(url, timeout=240) as resp:
                data = json.loads(resp.read())
            assert isinstance(data, list), str(data)[:200]
            break
        except Exception as ex:
            print("retry", attempt, repr(ex)[:160], flush=True)
            time.sleep(20 + 15 * attempt)
    else:
        raise SystemExit("failed")
    parts = []
    for n, loc in zip(names, data):
        h = loc["hourly"]
        parts.append(pd.DataFrame({f"{n}__{short}": h[v] for v, short in VARS.items()}, index=pd.to_datetime(h["time"], utc=True)))
    frames.append(pd.concat(parts, axis=1))
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated()]
    out.to_parquet(OUT + ".tmp")
    os.replace(OUT + ".tmp", OUT)
    print("fetched", d, e, "total", out.shape, "nulls in chunk:", int(frames[-1].isna().sum().sum()), flush=True)
    d = e + timedelta(days=1)
    time.sleep(3)
print("done")
