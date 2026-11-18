"""ENTSO-E: unavailable generation capacity (DE-LU, FR) and day-ahead load forecast (DE-LU), month by month."""
import sys, time, warnings
import pandas as pd
from entsoe import EntsoePandasClient
warnings.simplefilter("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
c = EntsoePandasClient(api_key=open(f"{SP}/entsoe/key").read().strip())
months = pd.date_range("2024-03-01", "2026-12-01", freq="MS", tz="UTC")
def get(fn, label, **kw):
    for attempt in range(4):
        try: return fn(**kw)
        except Exception as e:
            if "NoMatchingData" in type(e).__name__: return None
            print(label, "retry", attempt, repr(e)[:120], flush=True); time.sleep(10 + 20 * attempt)
for zone in ("DE_LU", "FR"):
    for kind, fn in (("gen", c.query_unavailability_of_generation_units), ("prod", c.query_unavailability_of_production_units)):
        parts = []
        for a, b in zip(months[:-1], months[1:]):
            d = get(fn, f"{zone} {kind} {a.date()}", country_code=zone, start=a, end=b)
            if d is not None and len(d): parts.append(d)
        out = pd.concat(parts) if parts else pd.DataFrame()
        out = out[~out.index.duplicated()] if len(out) else out
        out.astype({k: str for k in out.columns if out[k].dtype == object}).to_parquet(f"{SP}/entsoe/unavail_{kind}_{zone}.parquet")
        print(zone, kind, len(out), list(out.columns)[:14], flush=True)
parts = [get(c.query_load_forecast, f"load {a.date()}", country_code="DE_LU", start=a, end=b) for a, b in zip(months[:-1], months[1:])]
pd.concat([p for p in parts if p is not None]).to_parquet(f"{SP}/entsoe/load_fc_DE_LU.parquet"); print("load forecast done", flush=True)
