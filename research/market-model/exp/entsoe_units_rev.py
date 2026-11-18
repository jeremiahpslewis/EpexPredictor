"""ENTSO-E actual generation per generation unit, lignite, every second day from March 2024 to November 2026 (notes 21.38). One request per
area and day (the API's limit), about 10 s each; resumable: one parquet per area and month. usage: entsoe_units_all.py"""
import time, warnings, os
import pandas as pd
from entsoe import EntsoePandasClient
warnings.simplefilter("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; c = EntsoePandasClient(api_key=open(f"{SP}/entsoe/key").read().strip()); out = f"{SP}/entsoe/units_all"; os.makedirs(out, exist_ok=True)
def get(**kw):
    for attempt in range(4):
        try: return c.query_generation_per_plant(**kw)
        except Exception as e:
            if "NoMatchingData" in type(e).__name__: return None
            print("retry", attempt, type(e).__name__, flush=True); time.sleep(8 + 15 * attempt)
for m in pd.date_range("2024-03-01", "2026-11-01", freq="MS")[::-1]:
    for area in ("DE_50HZ", "DE_AMPRION"):
        fn = f"{out}/{area}_lignite_{m.strftime('%Y-%m')}.parquet"
        if os.path.exists(fn): continue
        parts = []
        for d in pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="2D"):
            if d > pd.Timestamp("2026-11-08"): break
            a = pd.Timestamp(d.date(), tz="Europe/Berlin"); x = get(country_code=area, start=a, end=a + pd.Timedelta(days=1), psr_type="B02")
            if x is not None and len(x): parts.append(x.resample("1h").mean())
            time.sleep(0.5)
        if parts:
            x = pd.concat(parts); x.columns = [" | ".join(map(str, col)) if isinstance(col, tuple) else str(col) for col in x.columns]; x.to_parquet(fn)
        print(area, m.strftime("%Y-%m"), len(parts), "days", flush=True)
print("UNITS DONE")
