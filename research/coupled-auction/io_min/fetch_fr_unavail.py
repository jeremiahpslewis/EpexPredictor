"""French generation-unit unavailability notices for the whole sample, in weekly chunks saved as they arrive
(resumable). The API key is read from SP/entsoe/key and never printed. Usage: fetch_fr_unavail.py SP START END"""
import sys, os, time, warnings
import pandas as pd
from entsoe import EntsoePandasClient
warnings.simplefilter("ignore")
SP, START, END = sys.argv[1], sys.argv[2], sys.argv[3]
c = EntsoePandasClient(api_key=open(f"{SP}/entsoe/key").read().strip())
OUTD = f"{SP}/entsoe/fr_unavail_gen"; os.makedirs(OUTD, exist_ok=True)
edges = pd.date_range(START, END, freq="7D", tz="UTC")
for a, b in zip(edges[:-1], edges[1:]):
    f = f"{OUTD}/{a:%Y%m%d}.parquet"
    if os.path.exists(f): continue
    d = None
    for attempt in range(5):
        try: d = c.query_unavailability_of_generation_units(country_code="FR", start=a, end=b); break
        except Exception as e:
            if "NoMatchingData" in type(e).__name__: d = pd.DataFrame(); break
            print(a.date(), "retry", attempt, type(e).__name__, flush=True); time.sleep(10 + 20 * attempt)
    if d is None: print(a.date(), "failed", flush=True); continue
    d = d.reset_index() if len(d) else d
    d.astype({k: str for k in d.columns if d[k].dtype == object}).to_parquet(f + ".tmp"); os.replace(f + ".tmp", f)
    print(a.date(), len(d), flush=True)
