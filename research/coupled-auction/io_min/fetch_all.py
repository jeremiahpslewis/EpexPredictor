"""
Bulk data for the eleven-zone model, 2025-10-01 to 2026-11-09 (Europe/Berlin). Resumable; writes only under SP.
Usage: fetch_all.py SP jao|entsoe [WORKER NWORKERS]
- jao:    per two-day window, the presolved final domain (reduced columns), Core net positions, shadow prices
          -> SP/jao/bulk/<ep>_<day>.parquet
- entsoe: in quarterly chunks
          - border schedules in both directions for the eleven zones (incl. rest-of-Europe borders);
          - the German price, load and wind/solar forecasts;
          - the NordLink audit series: physical flows DE<->NO2, and transmission unavailability for DE_LU<->NO_2
          -> SP/entsoe/eleven_all/<tag>.parquet | .nodata
The ENTSO-E key is read from SP/entsoe/key and never printed.
"""
import sys, json, time, urllib.parse
from pathlib import Path
import pandas as pd, requests

SP = Path(sys.argv[1]); MODE = sys.argv[2]
W, NW = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
T0, T1 = pd.Timestamp("2025-10-01", tz="UTC") - pd.Timedelta(hours=2), pd.Timestamp("2026-11-10", tz="UTC")

if MODE == "jao":
    OUT = SP / "jao/bulk"; OUT.mkdir(parents=True, exist_ok=True); B = "https://publicationtool.jao.eu/core/api/data"
    FC_KEEP = ["dateTimeUtc", "cneName", "contName", "cneEic", "direction", "elementType", "tso", "ram", "fmax", "fref", "amr", "ltaMargin", "minRamFactor"]
    days = pd.date_range(T0.normalize(), T1, freq="2D")
    jobs = [(ep, d) for d in days for ep in ("finalComputation", "netPos", "shadowPrices")]
    for i, (ep, d) in enumerate(jobs):
        if i % NW != W: continue
        f = OUT / f"{ep}_{d:%Y%m%d}.parquet"
        if f.exists(): continue
        q = f"{B}/{ep}?FromUtc={d:%Y-%m-%dT%H:%M}:00.000Z&ToUtc={d + pd.Timedelta(days=2):%Y-%m-%dT%H:%M}:00.000Z&Take=50000"
        if ep == "finalComputation": q += "&Filter=" + urllib.parse.quote(json.dumps({"Presolved": True}))
        for att in range(5):
            try:
                r = requests.get(q, timeout=600); r.raise_for_status(); js = r.json(); df = pd.DataFrame(js["data"])
                if js.get("totalRowsWithFilter", len(df)) > len(df): print("TRUNCATED", ep, d, flush=True)
                if ep == "finalComputation" and len(df):
                    df = df[[c for c in df.columns if c in FC_KEEP or c.startswith("ptdf_")]]
                if ep == "shadowPrices" and len(df):
                    df = df.drop(columns=[c for c in df.columns if df[c].dtype == object and c not in ("dateTimeUtc", "cnecName", "contName", "direction", "tso", "hubFrom", "hubTo", "cnecEic")])
                for c in df.columns:
                    if c.startswith(("ptdf_", "hub_")): df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
                df.to_parquet(f); print("ok", ep, d.date(), len(df), flush=True); break
            except Exception as e:
                print("err", ep, d.date(), type(e).__name__, str(e)[:100], flush=True); time.sleep(10 * (att + 1))
        time.sleep(0.5)

if MODE == "entsoe":
    from entsoe import EntsoePandasClient
    OUT = SP / "entsoe/eleven_all"; OUT.mkdir(parents=True, exist_ok=True)
    KEY = (SP / "entsoe/key").read_text().strip(); cl = EntsoePandasClient(api_key=KEY)
    BORDERS = {"DE_LU": ["FR", "NL", "BE", "AT", "CZ", "PL", "DK_1", "DK_2", "SE_4", "NO_2", "CH"],
               "FR": ["DE_LU", "BE", "CH", "IT_NORD", "ES", "GB"], "NL": ["DE_LU", "BE", "GB", "NO_2", "DK_1"],
               "BE": ["FR", "NL", "DE_LU", "GB"], "AT": ["DE_LU", "CZ", "CH", "IT_NORD", "SI", "HU"],
               "CZ": ["DE_LU", "AT", "PL", "SK"], "PL": ["DE_LU", "CZ", "SK", "LT", "SE_4"],
               "DK_1": ["DE_LU", "NL", "NO_2", "DK_2", "SE_3", "GB"], "DK_2": ["DE_LU", "DK_1", "SE_4"],
               "SE_4": ["DE_LU", "DK_2", "PL", "LT", "SE_3"], "NO_2": ["DE_LU", "NL", "DK_1", "NO_1", "NO_5", "GB"]}
    pairs = sorted({(a, b) for a, nb in BORDERS.items() for b in nb} | {(b, a) for a, nb in BORDERS.items() for b in nb})
    jobs = [("sx", a, b) for a, b in pairs] + [("price", "DE_LU", None), ("load", "DE_LU", None), ("ws", "DE_LU", None),
            ("pf", "DE_LU", "NO_2"), ("pf", "NO_2", "DE_LU"), ("ut", "DE_LU", "NO_2"), ("ut", "NO_2", "DE_LU")]
    chunks = [(pd.Timestamp(a, tz="Europe/Berlin"), pd.Timestamp(b, tz="Europe/Berlin")) for a, b in
              [("2025-10-01", "2026-01-01"), ("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01"), ("2026-07-01", "2026-10-01"), ("2026-10-01", "2026-11-10")]]
    jobs = [(j, c) for j in jobs for c in chunks]
    for i, ((kind, a, b), (s, e)) in enumerate(jobs):
        if i % NW != W: continue
        tag = f"{kind}_{a}" + (f"_{b}" if b else "") + f"_{s:%Y%m%d}"
        if (OUT / f"{tag}.parquet").exists() or (OUT / f"{tag}.nodata").exists(): continue
        for att in range(4):
            try:
                if kind == "sx": x = cl.query_scheduled_exchanges(a, b, start=s, end=e, dayahead=True)
                elif kind == "price": x = cl.query_day_ahead_prices(a, start=s, end=e)
                elif kind == "load": x = cl.query_load_forecast(a, start=s, end=e)
                elif kind == "ws": x = cl.query_wind_and_solar_forecast(a, start=s, end=e)
                elif kind == "pf": x = cl.query_crossborder_flows(a, b, start=s, end=e)
                elif kind == "ut": x = cl.query_unavailability_transmission(a, b, start=s, end=e)
                x = x.to_frame("v") if isinstance(x, pd.Series) else x
                x.columns = [str(c) for c in x.columns]
                for c in x.columns:
                    if x[c].dtype == object: x[c] = x[c].astype(str)
                x.to_parquet(OUT / f"{tag}.parquet"); print("ok", tag, len(x), flush=True); break
            except Exception as ex:
                msg = str(ex).replace(KEY, "***")
                if "NoMatchingData" in type(ex).__name__ or "No matching data" in msg:
                    (OUT / f"{tag}.nodata").touch(); print("nodata", tag, flush=True); break
                print("err", tag, type(ex).__name__, msg[:120], flush=True); time.sleep(5 * (att + 1))
        time.sleep(0.5)
