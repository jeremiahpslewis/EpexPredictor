"""TSO day-ahead forecasts (wind, solar, load) of Germany's neighbours from energy-charts, 15 min, since 2024-03."""
import json, os, time, urllib.request, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
edges = list(pd.date_range("2024-03-01", "2026-11-13", freq="180D")) + [pd.Timestamp("2026-11-13")]
for c in ("fr", "nl", "at", "be", "dk", "pl", "ch", "cz"):
    if os.path.exists(f"{SP}/nb/{c}.parquet"): continue
    cols = {}
    for pt in ("wind_onshore", "wind_offshore", "solar", "load"):
        parts = []
        for a, b in zip(edges[:-1], edges[1:]):
            url = f"https://api.energy-charts.info/public_power_forecast?country={c}&production_type={pt}&forecast_type=day-ahead&start={a.date()}T00:00Z&end={b.date()}T00:00Z"
            d = None
            for k in range(6):
                try: d = json.load(urllib.request.urlopen(url, timeout=120)); break
                except Exception as e:
                    if "404" in repr(e) or "Expecting value" in repr(e): break
                    time.sleep(8 + 12 * k)
            if d and d.get("unix_seconds"): parts.append(pd.Series(d["forecast_values"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True), dtype=float))
            time.sleep(1.5)
        if parts: s = pd.concat(parts); cols[pt] = s[~s.index.duplicated()].sort_index()
    if cols:
        df = pd.DataFrame(cols); df.to_parquet(f"{SP}/nb/{c}.parquet"); print(c, df.shape, df.index.min(), df.index.max(), df.notna().mean().round(2).to_dict(), (df.mean() / 1000).round(1).to_dict(), flush=True)
    else: print(c, "nothing", flush=True)
print("finished", flush=True)
