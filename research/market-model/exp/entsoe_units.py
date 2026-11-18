"""ENTSO-E actual generation per generation unit (>= 100 MW), lignite, hard coal and gas, for two weeks in October 2025 and October 2026.
usage: entsoe_units.py"""
import time, warnings, os
import pandas as pd
from entsoe import EntsoePandasClient
warnings.simplefilter("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
c = EntsoePandasClient(api_key=open(f"{SP}/entsoe/key").read().strip()); out = f"{SP}/entsoe/units"; os.makedirs(out, exist_ok=True)
def get(label, **kw):
    for attempt in range(4):
        try: return c.query_generation_per_plant(**kw)
        except Exception as e:
            if "NoMatchingData" in type(e).__name__: return None
            print(label, "retry", attempt, repr(e)[:160], flush=True); time.sleep(8 + 15 * attempt)
for psr, nm in (("B02", "lignite"), ("B05", "coal"), ("B04", "gas")):
    for t0 in ("2026-10-05", "2025-10-06"):
        for area in (("DE_50HZ", "DE_AMPRION") if nm == "lignite" else ("DE_50HZ", "DE_AMPRION", "DE_TENNET", "DE_TRANSNET")):
            fn = f"{out}/{area}_{nm}_{t0}.parquet"
            if os.path.exists(fn): continue
            parts = []
            for k in range(14):
                a = pd.Timestamp(t0, tz="Europe/Berlin") + pd.Timedelta(days=k); d = get(f"{area} {nm} {a.date()}", country_code=area, start=a, end=a + pd.Timedelta(days=1), psr_type=psr)
                if d is not None and len(d): parts.append(d)
                elif k == 0: break                                  # nothing for this area and fuel: skip the rest
                time.sleep(0.6)
            if parts:
                x = pd.concat(parts); x.columns = [" | ".join(map(str, col)) if isinstance(col, tuple) else str(col) for col in x.columns]; x.to_parquet(fn)
            print(area, nm, t0, len(parts), "days", (parts[0].shape if parts else None), flush=True)
print("UNITS DONE")
