"""Fetch hourly weather for every location the app uses from Bright Sky (DWD open data); one file per location."""
import sys, os, json, time, urllib.request
sys.path.insert(0, "/home/user/EpexPredictor")
from predictor.model.priceregion import PriceRegionName
OUT = sys.argv[1]; names = ["DE", "AT", "BE", "FR", "NL", "SE4", "DK1", "DK2"]; pts = {}
for n in names:
    r = PriceRegionName[n].to_region()
    for la, lo in zip(r.weather_latitudes if hasattr(r, "weather_latitudes") else r.latitudes, r.weather_longitudes if hasattr(r, "weather_longitudes") else r.longitudes): pts[(la, lo)] = 1
    for la, lo in zip(r.generation_latitudes + r.generation_offshore_latitudes, r.generation_longitudes + r.generation_offshore_longitudes): pts[(la, lo)] = 1
print(len(pts), "distinct locations", flush=True); bad = 0
for k, (la, lo) in enumerate(pts):
    fn = f"{OUT}/p_{la}_{lo}.json"
    if os.path.exists(fn): continue
    url = f"https://api.brightsky.dev/weather?lat={la}&lon={lo}&date=2026-11-01&last_date=2026-11-27&max_dist=200000&tz=UTC"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp: data = resp.read()
        json.loads(data); open(fn, "wb").write(data)
    except Exception as e: bad += 1; print("failed", la, lo, str(e)[:100], flush=True)
    time.sleep(0.7)
print("finished, failed:", bad, flush=True)
