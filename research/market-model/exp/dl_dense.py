"""Dense German weather grid (hourly): archive since 2024-03 and forecasts made 1, 2 and 3 days earlier for the backtest period."""
import json, os, sys, time, urllib.request
from datetime import date, timedelta
import numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
rows = {47.8: (8.0, 12.8), 48.6: (8.0, 13.2), 49.4: (6.8, 12.6), 50.2: (6.5, 12.2), 51.0: (6.3, 14.6), 51.8: (6.3, 14.5), 52.6: (7.0, 14.2), 53.4: (7.3, 14.0), 54.2: (8.9, 13.3)}
P = [(la, round(lo, 2)) for la, (a, b) in rows.items() for lo in np.arange(a, b + 0.01, 1.2)] + [(54.7, 9.0)]
OFF = [(54.0, 6.5), (54.4, 7.7), (54.3, 6.0), (54.9, 7.2), (54.6, 13.0), (54.8, 14.0)]
PTS = P + OFF; json.dump({"onshore": len(P), "points": PTS}, open(f"{SP}/dense/points.json", "w")); print(len(P), "onshore,", len(OFF), "offshore", flush=True)
VARS = {"wind_speed_80m": "w80", "wind_speed_120m": "w120", "shortwave_radiation": "sw", "temperature_2m": "t2"}
def fetch(host, hourly, d, e, pts):
    url = (f"https://{host}/v1/forecast?latitude=" + ",".join(str(p[0]) for p in pts) + "&longitude=" + ",".join(str(p[1]) for p in pts) + f"&start_date={d}&end_date={e}&hourly=" + ",".join(hourly) + "&timezone=UTC")
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=180) as r: data = json.loads(r.read())
            if isinstance(data, dict) and "hourly" in data: data = [data]
            assert isinstance(data, list), str(data)[:200]; return data
        except Exception as ex: print("retry", attempt, repr(ex)[:120], flush=True); time.sleep(30 + 20 * attempt)
    raise SystemExit("failed")
def run(name, host, suffixes, start, end, step):
    fn = f"{SP}/dense/{name}.parquet"
    if os.path.exists(fn): print(name, "exists"); return
    frames = []; d = start
    while d <= end:
        e = min(d + timedelta(days=step - 1), end); parts = []
        if os.path.exists(f"{SP}/dense/{name}_{d}.parquet"): frames.append(pd.read_parquet(f"{SP}/dense/{name}_{d}.parquet")); d = e + timedelta(days=1); continue
        for k in range(0, len(PTS), 14):
            data = fetch(host, [v + s for v in VARS for s in suffixes], d, e, PTS[k:k + 14])
            parts += [pd.DataFrame({f"{VARS[v]}_{k + i}{s[-5:].replace('_day', '_d') if s else ''}": loc["hourly"][v + s] for v in VARS for s in suffixes}, index=pd.to_datetime(loc["hourly"]["time"], utc=True)) for i, loc in enumerate(data)]
            time.sleep(20)
        pd.concat(parts, axis=1).astype("float32").to_parquet(f"{SP}/dense/{name}_{d}.parquet"); frames.append(pd.concat(parts, axis=1)); print(name, d, flush=True); d = e + timedelta(days=1)
    out = pd.concat(frames).sort_index(); out[~out.index.duplicated()].astype("float32").to_parquet(fn); print(name, "done", out.shape, flush=True)
run("prev", "previous-runs-api.open-meteo.com", ["_previous_day1", "_previous_day2", "_previous_day3"], date(2026, 8, 25), date(2026, 11, 10), 26)
run("arch", "historical-forecast-api.open-meteo.com", [""], date(2025, 10, 1), date(2026, 11, 10), 45)
