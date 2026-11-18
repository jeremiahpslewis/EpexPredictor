"""Hourly weather for every zone's sample locations as it was forecast 2 and 3 days earlier (Open-Meteo previous-runs archive).
usage: download_prev_all.py OUT_DIR START END"""
import json, os, sys, time, urllib.request
from datetime import date, timedelta
import pandas as pd
sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model.priceregion import PriceRegionName
OUT = sys.argv[1]; START, END = date.fromisoformat(sys.argv[2]), date.fromisoformat(sys.argv[3])
VARS = {"wind_speed_80m": "wind", "temperature_2m": "temp", "global_tilted_irradiance": "irradiance", "pressure_msl": "pressure", "relative_humidity_2m": "humidity"}
hourly = [f"{v}_previous_day{k}" for v in VARS for k in (2, 3)]
os.makedirs(OUT, exist_ok=True)
for zone in ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]:
    r = PriceRegionName(zone).to_region(); fn = f"{OUT}/prev_{zone}.parquet"
    frames = [pd.read_parquet(fn)] if os.path.exists(fn) else []
    have = set(frames[0].index.normalize().unique()) if frames else set()
    e = END
    while e >= START:
        d = max(e - timedelta(days=13), START)
        if pd.Timestamp(d, tz="UTC") in have and pd.Timestamp(e, tz="UTC") in have:
            e = d - timedelta(days=1); continue
        url = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude=" + ",".join(map(str, r.weather_latitudes)) + "&longitude=" + ",".join(map(str, r.weather_longitudes))
               + f"&azimuth=0&tilt=0&start_date={d}&end_date={e}&hourly=" + ",".join(hourly) + "&timezone=UTC")
        for attempt in range(12):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=120) as resp: data = json.loads(resp.read())
                if isinstance(data, dict) and "hourly" in data: data = [data]
                assert isinstance(data, list), str(data)[:200]; break
            except Exception as ex:
                print(zone, "retry", attempt, repr(ex)[:140], flush=True); time.sleep(20 + 15 * attempt)
        else: raise SystemExit("failed")
        parts = [pd.DataFrame({f"{VARS[v]}_{i}_d{k}": loc["hourly"][f"{v}_previous_day{k}"] for v in VARS for k in (2, 3)}, index=pd.to_datetime(loc["hourly"]["time"], utc=True)) for i, loc in enumerate(data)]
        frames.append(pd.concat(parts, axis=1)); out = pd.concat(frames).sort_index(); out = out[~out.index.duplicated()]
        out.to_parquet(fn + ".tmp"); os.replace(fn + ".tmp", fn); e = d - timedelta(days=1); time.sleep(1)
    print(zone, "done", pd.read_parquet(fn).shape, flush=True)
