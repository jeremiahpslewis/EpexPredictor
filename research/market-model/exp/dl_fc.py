import json, time, urllib.request, pandas as pd, sys
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
edges = pd.date_range("2024-02-01", "2026-12-01", freq="MS")
for pt in ("wind_onshore", "wind_offshore", "solar", "load"):
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        url = f"https://api.energy-charts.info/public_power_forecast?country=de&production_type={pt}&forecast_type=day-ahead&start={a.date()}&end={(b - pd.Timedelta(days=1)).date()}"
        for k in range(5):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=90)); break
            except Exception as e: print(pt, a.date(), repr(e)[:80], flush=True); d = None; time.sleep(5 + 10 * k)
        if d and d.get("unix_seconds"): parts.append(pd.Series(d["forecast_values"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True), dtype=float))
        time.sleep(0.5)
    if parts:
        s = pd.concat(parts); s = s[~s.index.duplicated()].sort_index(); s.to_frame(pt).to_parquet(f"{SP}/ecfc/{pt}.parquet"); print(pt, len(s), s.index.min(), s.index.max(), s.isna().mean().round(3), s.mean().round(0), flush=True)
    else: print(pt, "nothing", flush=True)
