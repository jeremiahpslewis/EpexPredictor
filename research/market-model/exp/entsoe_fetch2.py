"""ENTSO-E unavailability for the backtest period only. usage: entsoe_fetch2.py"""
import time, warnings
import pandas as pd
from entsoe import EntsoePandasClient
warnings.simplefilter("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
c = EntsoePandasClient(api_key=open(f"{SP}/entsoe/key").read().strip())
edges = pd.date_range("2026-08-01", "2026-11-14", freq="7D", tz="UTC")
def get(fn, label, **kw):
    for attempt in range(6):
        try: return fn(**kw)
        except Exception as e:
            if "NoMatchingData" in type(e).__name__: return None
            print(label, "retry", attempt, repr(e)[:100], flush=True); time.sleep(10 + 20 * attempt)
for zone, kind, fn in (("FR", "gen", c.query_unavailability_of_generation_units), ("DE_LU", "prod", c.query_unavailability_of_production_units), ("FR", "prod", c.query_unavailability_of_production_units)):
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        d = get(fn, f"{zone} {kind} {a.date()}", country_code=zone, start=a, end=b)
        if d is not None and len(d): parts.append(d)
    out = pd.concat(parts) if parts else pd.DataFrame()
    out.astype({k: str for k in out.columns if out[k].dtype == object}).to_parquet(f"{SP}/entsoe/bt_unavail_{kind}_{zone}.parquet")
    print(zone, kind, len(out), flush=True)
