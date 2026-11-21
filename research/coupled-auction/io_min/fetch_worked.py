"""
ENTSO-E data for the eleven-zone worked MTUs (and later the full sample). Usage: fetch_worked.py SP START END [WORKER NWORKERS]
For each modelled zone: day-ahead net position, and day-ahead scheduled exchanges on EVERY border in both directions
(modelled and rest-of-Europe borders), so that zone net position = sum of border schedules can be checked (W3).
Writes SP/entsoe/eleven/<tag>.parquet or <tag>.nodata; resumable; the key is read from SP/entsoe/key and never printed.
"""
import sys, time
from pathlib import Path
import pandas as pd
from entsoe import EntsoePandasClient

SP = Path(sys.argv[1]); START = pd.Timestamp(sys.argv[2], tz="Europe/Berlin"); END = pd.Timestamp(sys.argv[3], tz="Europe/Berlin")
W, NW = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
OUT = SP / "entsoe/eleven"; OUT.mkdir(parents=True, exist_ok=True)
KEY = (SP / "entsoe/key").read_text().strip()
BORDERS = {   # every interconnected neighbour of each modelled zone
    "DE_LU": ["FR", "NL", "BE", "AT", "CZ", "PL", "DK_1", "DK_2", "SE_4", "NO_2", "CH"],
    "FR": ["DE_LU", "BE", "CH", "IT_NORD", "ES", "GB"], "NL": ["DE_LU", "BE", "GB", "NO_2", "DK_1"],
    "BE": ["FR", "NL", "DE_LU", "GB"], "AT": ["DE_LU", "CZ", "CH", "IT_NORD", "SI", "HU"],
    "CZ": ["DE_LU", "AT", "PL", "SK"], "PL": ["DE_LU", "CZ", "SK", "LT", "SE_4"],
    "DK_1": ["DE_LU", "NL", "NO_2", "DK_2", "SE_3", "GB"], "DK_2": ["DE_LU", "DK_1", "SE_4"],
    "SE_4": ["DE_LU", "DK_2", "PL", "LT", "SE_3"], "NO_2": ["DE_LU", "NL", "DK_1", "NO_1", "NO_5", "GB"],
}
jobs = [("np", z, None) for z in BORDERS] + [("sx", a, b) for a, nb in BORDERS.items() for b in nb]
jobs += [("sx", b, a) for a, nb in BORDERS.items() for b in nb if b not in BORDERS]     # imports from the rest of Europe
jobs = [j for i, j in enumerate(jobs) if i % NW == W]
cl = EntsoePandasClient(api_key=KEY)
tagd = f"{START:%Y%m%d}_{END:%Y%m%d}"
for kind, a, b in jobs:
    tag = f"{kind}_{a}" + (f"_{b}" if b else "") + f"_{tagd}"
    if (OUT / f"{tag}.parquet").exists() or (OUT / f"{tag}.nodata").exists(): continue
    for att in range(4):
        try:
            s = cl.query_net_position(a, start=START, end=END, dayahead=True) if kind == "np" else \
                cl.query_scheduled_exchanges(a, b, start=START, end=END, dayahead=True)
            s = s if isinstance(s, pd.Series) else s.squeeze()
            s.rename("v").to_frame().to_parquet(OUT / f"{tag}.parquet"); print("ok", tag, len(s), flush=True); break
        except Exception as e:
            msg = str(e).replace(KEY, "***")
            if "NoMatchingData" in type(e).__name__ or "No matching data" in msg:
                (OUT / f"{tag}.nodata").touch(); print("nodata", tag, flush=True); break
            print("err", tag, type(e).__name__, msg[:120], flush=True); time.sleep(5 * (att + 1))
    time.sleep(1)
