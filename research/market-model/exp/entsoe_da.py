"""ENTSO-E day-ahead basis for DE_LU (notes 21.59): total scheduled generation (14.1.C), day-ahead wind and solar forecast (14.1.D), day-ahead net
position from market coupling, and the day-ahead load forecast. Quarterly chunks, resumable. usage: entsoe_da.py"""
import time, warnings, os
import pandas as pd
from entsoe import EntsoePandasClient
warnings.simplefilter("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; c = EntsoePandasClient(api_key=open(f"{SP}/entsoe/key").read().strip()); out = f"{SP}/entsoe/da"; os.makedirs(out, exist_ok=True)
Q = {"gen": lambda s, e: c.query_generation_forecast("DE_LU", start=s, end=e), "ws": lambda s, e: c.query_wind_and_solar_forecast("DE_LU", start=s, end=e), "netpos": lambda s, e: c.query_net_position("DE_LU", start=s, end=e, dayahead=True), "load": lambda s, e: c.query_load_forecast("DE_LU", start=s, end=e)}
MS = pd.date_range("2024-03-01", "2026-12-01", freq="3MS", tz="Europe/Berlin")
for nm, f in Q.items():
    for s, e in zip(MS[:-1], MS[1:]):
        fn = f"{out}/{nm}_{s.strftime('%Y-%m')}.parquet"
        if os.path.exists(fn): continue
        for attempt in range(4):
            try:
                r = f(s, min(e, pd.Timestamp("2026-11-10", tz="Europe/Berlin"))); r = r.to_frame(nm) if isinstance(r, pd.Series) else r; r.columns = [str(x) for x in r.columns]; r.to_parquet(fn); print(nm, s.date(), r.shape, list(r.columns)[:4], flush=True); break
            except Exception as ex:
                print("retry", nm, s.date(), attempt, type(ex).__name__, str(ex)[:80].replace(c.api_key, "KEY"), flush=True); time.sleep(8 + 15 * attempt)
print("finished", flush=True)
